"""ThetaData v3 REST 客户端与期权历史拉取（Mac 本地，需 Theta Terminal 运行）。

端点与参数来源：https://thetadata.net/docs/（openapiv3.yaml，servers: http://127.0.0.1:25503/v3，
2026-10-22 经 VPS 代理读取）：
- GET /option/list/expirations?symbol=
- GET /option/history/quote?symbol&expiration&strike=*&right=both&date&interval=1m
  （NBBO；interval 下取区间末次报价）
- GET /option/history/ohlc?symbol&expiration&strike=*&date&interval=1m（bar 时间戳为开盘时刻）
- GET /option/history/open_interest?symbol&expiration=*&max_dte&date（OPRA 约 06:30 ET 报前一日 OI）
- interval=tick 只允许单日；多日请求 ≤1 个月；并发按账户档位（Standard 4），本模块默认 2。
- 错误码：472 NO_DATA、473 INVALID_PARAMS、474 DISCONNECTED、570 LARGE_REQUEST
  （Making-Requests 页）。
- 时间戳为 YYYY-MM-DDTHH:mm:ss.SSS，美东无时区
  （时区按文档 at_time 说明 "assumed to be America/New_York"）。

两种模式：
  bulk_quotes_1m(underlying, dates)
    → ZR_DATA_ROOT/options_1m/underlying=XXX/date=YYYY-MM-DD/{quotes,ohlc,oi}.parquet
  ticks_window(contract, start, end)
    → ZR_DATA_ROOT/options_ticks/underlying=XXX/date=YYYY-MM-DD/<contract>_<HHMM>-<HHMM>.parquet
合约代码统一为长桥格式 SPY261021C774000.US，便于与录制器数据对齐。
测试用假 HTTP，不连终端。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from zr.config import get_settings

log = logging.getLogger("zr.thetadata")
ET = ZoneInfo("America/New_York")
BASE_URL = "http://127.0.0.1:25503/v3"
DEFAULT_CONCURRENCY = 2

# 行权价带宽（相对当日开盘价）与 DTE 上限，来自 DATA_SPEC / universe.yaml
STRIKE_BAND = {"SPY": 0.03, "QQQ": 0.03, "MU": 0.08, "AMD": 0.08}
MAX_DTE = 5

_CONTRACT_RE = re.compile(r"^(?P<und>[A-Z.\-]+?)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d+)\.US$")


class ThetaError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} {body[:200]} <- {url}")
        self.status = status
        self.body = body
        self.url = url


# ---------- 合约代码 ----------


def lb_symbol(und: str, expiry: dt.date, right: str, strike: float) -> str:
    r = "C" if right.lower().startswith("c") else "P"
    return f"{und}{expiry.strftime('%y%m%d')}{r}{int(round(strike * 1000))}.US"


def parse_contract(symbol: str) -> dict:
    m = _CONTRACT_RE.match(symbol)
    if not m:
        raise ValueError(f"不是长桥期权代码：{symbol}")
    return {
        "und": m["und"],
        "expiry": dt.datetime.strptime(m["exp"], "%y%m%d").date(),
        "right": "call" if m["right"] == "C" else "put",
        "strike": int(m["strike"]) / 1000,
    }


# ---------- HTTP ----------


@dataclass
class ThetaClient:
    base_url: str = BASE_URL
    timeout_s: float = 120.0
    concurrency: int = DEFAULT_CONCURRENCY
    retries: int = 3
    # 可注入的 GET，测试用假数据
    getter: Callable[[str], tuple[int, str]] | None = None

    def __post_init__(self) -> None:
        self._sem = threading.Semaphore(self.concurrency)

    def _http_get(self, url: str) -> tuple[int, str]:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - 本地终端
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def get_json(self, path: str, **params) -> list[dict]:
        clean = {k: v for k, v in params.items() if v is not None}
        clean["format"] = "json"
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(clean)}"
        getter = self.getter or self._http_get
        with self._sem:
            for attempt in range(self.retries):
                status, body = getter(url)
                if status == 200:
                    data = json.loads(body) if body.strip() else []
                    return data if isinstance(data, list) else data.get("data", [])
                if status == 472:  # NO_DATA
                    return []
                if status in (429, 474, 571) and attempt < self.retries - 1:
                    time.sleep(1.0 + attempt)
                    continue
                raise ThetaError(status, body, url)
        raise ThetaError(status, body, url)

    # ---- 端点 ----
    def list_expirations(self, symbol: str) -> list[dt.date]:
        rows = self.get_json("/option/list/expirations", symbol=symbol)
        return sorted(dt.date.fromisoformat(r["expiration"]) for r in rows)

    def history_quote(
        self,
        symbol: str,
        expiration: dt.date,
        date: dt.date,
        interval: str = "1m",
        strike: str = "*",
        right: str = "both",
        **kw,
    ) -> pd.DataFrame:
        rows = self.get_json(
            "/option/history/quote",
            symbol=symbol,
            expiration=expiration.strftime("%Y%m%d"),
            strike=strike,
            right=right,
            date=date.strftime("%Y%m%d"),
            interval=interval,
            **kw,
        )
        return _frame(rows, symbol, date)

    def history_ohlc(
        self,
        symbol: str,
        expiration: dt.date,
        date: dt.date,
        interval: str = "1m",
        strike: str = "*",
        right: str = "both",
    ) -> pd.DataFrame:
        rows = self.get_json(
            "/option/history/ohlc",
            symbol=symbol,
            expiration=expiration.strftime("%Y%m%d"),
            strike=strike,
            right=right,
            date=date.strftime("%Y%m%d"),
            interval=interval,
        )
        return _frame(rows, symbol, date)

    def history_open_interest(
        self, symbol: str, date: dt.date, max_dte: int = MAX_DTE
    ) -> pd.DataFrame:
        rows = self.get_json(
            "/option/history/open_interest",
            symbol=symbol,
            expiration="*",
            date=date.strftime("%Y%m%d"),
            max_dte=max_dte,
        )
        return _frame(rows, symbol, date)


def _frame(rows: list[dict], und: str, trade_date: dt.date) -> pd.DataFrame:
    """统一列：ts(UTC)、trade_date、symbol(长桥格式)、underlying、expiry、strike、right，其后为原字段。"""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiration"]).dt.date
    df["strike"] = df["strike"].astype(float)
    df["right"] = df["right"].str.lower()
    df["symbol"] = [
        lb_symbol(und, e, r, s)
        for e, r, s in zip(df["expiry"], df["right"], df["strike"], strict=True)
    ]
    df["underlying"] = und
    df["trade_date"] = trade_date
    if "timestamp" in df:
        ts = pd.to_datetime(df["timestamp"])
        df["ts"] = ts.dt.tz_localize(
            ET, ambiguous="NaT", nonexistent="shift_forward"
        ).dt.tz_convert("UTC")
    df = df.drop(columns=["expiration", "timestamp"], errors="ignore")
    front = [
        c
        for c in ["ts", "trade_date", "symbol", "underlying", "expiry", "strike", "right"]
        if c in df
    ]
    return df[front + [c for c in df.columns if c not in front]]


# ---------- 合约选择 ----------


def select_expirations(
    expirations: Iterable[dt.date], day: dt.date, max_dte: int = MAX_DTE
) -> list[dt.date]:
    return sorted(e for e in expirations if 0 <= (e - day).days <= max_dte)


def strike_filter(df: pd.DataFrame, open_px: float, band: float) -> pd.DataFrame:
    if df.empty:
        return df
    lo, hi = open_px * (1 - band), open_px * (1 + band)
    return df[(df["strike"] >= lo) & (df["strike"] <= hi)].reset_index(drop=True)


def open_price_from_bars(root: Path, und: str, day: dt.date) -> float | None:
    """当日 09:30 ET 第一根 1 分钟 K 的开盘价（长桥 bars_1m）。"""
    f = root / "bars_1m" / f"underlying={und}" / f"year={day.year}" / "bars.parquet"
    if not f.exists():
        return None
    bars = pd.read_parquet(f, columns=["ts", "open", "session"])
    t = bars["ts"].dt.tz_convert(ET)
    sel = bars[(t.dt.date == day) & (bars["session"] == "Intraday")]
    if sel.empty:
        return None
    return float(sel.sort_values("ts")["open"].iloc[0])


# ---------- 落盘 ----------


def day_dir(root: Path, kind: str, und: str, day: dt.date) -> Path:
    return root / kind / f"underlying={und}" / f"date={day.isoformat()}"


def day_done(root: Path, und: str, day: dt.date) -> bool:
    d = day_dir(root, "options_1m", und, day)
    return all((d / f"{n}.parquet").exists() for n in ("quotes", "ohlc", "oi"))


def write_day(
    root: Path, und: str, day: dt.date, quotes: pd.DataFrame, ohlc: pd.DataFrame, oi: pd.DataFrame
) -> None:
    d = day_dir(root, "options_1m", und, day)
    d.mkdir(parents=True, exist_ok=True)
    for name, df in (("quotes", quotes), ("ohlc", ohlc), ("oi", oi)):
        df.to_parquet(d / f"{name}.parquet", index=False)


def _append_manifest(root: Path, record: dict) -> None:
    p = root / "options_1m" / "manifest.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ---------- 模式 a：bulk_quotes_1m ----------


def fetch_day(
    client: ThetaClient,
    root: Path,
    und: str,
    day: dt.date,
    expirations: list[dt.date],
    open_px: float | None,
) -> dict:
    t0 = time.monotonic()
    exps = select_expirations(expirations, day)
    band = STRIKE_BAND.get(und, 0.05)
    quotes, ohlc = [], []
    # 每个到期日一次请求（strike=*），本地按带宽过滤；并发由 client 信号量限制
    with ThreadPoolExecutor(max_workers=client.concurrency) as ex:
        q_futs = {e: ex.submit(client.history_quote, und, e, day) for e in exps}
        o_futs = {e: ex.submit(client.history_ohlc, und, e, day) for e in exps}
        oi_fut = ex.submit(client.history_open_interest, und, day)
        for e in exps:
            quotes.append(q_futs[e].result())
            ohlc.append(o_futs[e].result())
        oi = oi_fut.result()
    q = (
        pd.concat([x for x in quotes if not x.empty], ignore_index=True)
        if any(not x.empty for x in quotes)
        else pd.DataFrame()
    )
    o = (
        pd.concat([x for x in ohlc if not x.empty], ignore_index=True)
        if any(not x.empty for x in ohlc)
        else pd.DataFrame()
    )
    if open_px is not None:
        q, o, oi = (
            strike_filter(q, open_px, band),
            strike_filter(o, open_px, band),
            strike_filter(oi, open_px, band),
        )
    write_day(root, und, day, q, o, oi)
    rec = {
        "underlying": und,
        "date": day,
        "expirations": [e.isoformat() for e in exps],
        "open_px": open_px,
        "rows_quotes": len(q),
        "rows_ohlc": len(o),
        "rows_oi": len(oi),
        "seconds": round(time.monotonic() - t0, 1),
    }
    _append_manifest(root, rec)
    return rec


def bulk_quotes_1m(
    client: ThetaClient,
    root: Path,
    und: str,
    days: Iterable[dt.date],
    *,
    open_px_fn: Callable[[dt.date], float | None] | None = None,
) -> list[dict]:
    pending = [d for d in days if not day_done(root, und, d)]
    if not pending:
        return []
    expirations = client.list_expirations(und)
    out = []
    for day in pending:
        open_px = open_px_fn(day) if open_px_fn else open_price_from_bars(root, und, day)
        try:
            rec = fetch_day(client, root, und, day, expirations, open_px)
        except ThetaError as e:
            rec = {"underlying": und, "date": day, "error": str(e)}
            _append_manifest(root, rec)
            log.error("%s %s 失败：%s", und, day, e)
        log.info("%s", rec)
        out.append(rec)
    return out


def trading_days_from_bars(root: Path, und: str, start: dt.date, end: dt.date) -> list[dt.date]:
    """用长桥 1 分钟 K 的日期集合作交易日历（不硬编码）。"""
    days: set[dt.date] = set()
    for f in sorted((root / "bars_1m" / f"underlying={und}").glob("year=*/bars.parquet")):
        ts = pd.read_parquet(f, columns=["ts", "session"])
        ts = ts[ts["session"] == "Intraday"]["ts"].dt.tz_convert(ET).dt.date
        days.update(d for d in ts.unique() if start <= d <= end)
    return sorted(days)


# ---------- 模式 b：ticks_window ----------


def ticks_window(
    client: ThetaClient, root: Path, contract: str, start: dt.datetime, end: dt.datetime
) -> Path:
    """某合约某时间窗（美东）的逐笔 NBBO；start/end 同一天。"""
    c = parse_contract(contract)
    s, e = start.astimezone(ET), end.astimezone(ET)
    if s.date() != e.date():
        raise ValueError("start/end 必须同一交易日")
    df = client.history_quote(
        c["und"],
        c["expiry"],
        s.date(),
        interval="tick",
        strike=f"{c['strike']:.3f}",
        right=c["right"],
        start_time=s.strftime("%H:%M:%S.000"),
        end_time=e.strftime("%H:%M:%S.000"),
    )
    d = day_dir(root, "options_ticks", c["und"], s.date())
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{contract}_{s.strftime('%H%M')}-{e.strftime('%H%M')}.parquet"
    df.to_parquet(p, index=False)
    return p


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ThetaData 期权历史拉取")
    parser.add_argument("--symbols", default=None, help="逗号分隔，默认 ZR_SYMBOLS")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    root = settings.zr_data_root
    unds = [
        s.split(".")[0] for s in (args.symbols.split(",") if args.symbols else settings.symbols)
    ]
    client = ThetaClient(concurrency=args.concurrency)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    for und in unds:
        days = trading_days_from_bars(root, und, start, end)
        log.info("%s 交易日 %d 个（%s ~ %s）", und, len(days), start, end)
        recs = bulk_quotes_1m(client, root, und, days)
        ok = [r for r in recs if "error" not in r]
        log.info(
            "%s 完成 %d 天，失败 %d，行 quotes=%d ohlc=%d oi=%d，耗时 %.0fs",
            und,
            len(ok),
            len(recs) - len(ok),
            sum(r["rows_quotes"] for r in ok),
            sum(r["rows_ohlc"] for r in ok),
            sum(r["rows_oi"] for r in ok),
            sum(r["seconds"] for r in ok),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
