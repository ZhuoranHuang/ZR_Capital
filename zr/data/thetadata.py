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

import numpy as np
import pandas as pd

from zr.config import get_settings

log = logging.getLogger("zr.thetadata")
ET = ZoneInfo("America/New_York")
BASE_URL = "http://127.0.0.1:25503/v3"
DEFAULT_CONCURRENCY = 4  # 账户上限 4（Standard），2026-10-22 起

# 行权价带宽（相对当日开盘价）与 DTE 上限，来自 DATA_SPEC / universe.yaml
STRIKE_BAND = {"SPY": 0.03, "QQQ": 0.03, "MU": 0.08, "AMD": 0.08}
MAX_DTE = 5

_CONTRACT_RE = re.compile(r"^(?P<und>[A-Z.\-]+?)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d+)\.US$")


CONNECT_ERROR = 599  # 伪状态码：连接失败
# 不可达 / DISCONNECTED / SERVER_STARTING / 503「MDDS upstream unavailable: io exception」
# （2026-10-22 实测：终端与上游断开时返回 503，恢复后自动可用）
OUTAGE_STATUSES = (CONNECT_ERROR, 474, 503, 571)


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
    retries: int = 6  # 429：指数退避 1,2,4,8,16 s
    # 终端不可达（重启 / 474 DISCONNECTED / 571 SERVER_STARTING）时的等待：总共最多 outage_wait_s 秒
    outage_wait_s: float = 600.0
    outage_poll_s: float = 10.0
    sleep: Callable[[float], None] = time.sleep
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
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            # 终端未运行 / 正在重启：用伪状态码 599 交给 get_json 等待重试
            return CONNECT_ERROR, repr(e)

    def get_json(self, path: str, **params) -> list[dict]:
        clean = {k: v for k, v in params.items() if v is not None}
        clean["format"] = "json"
        url = f"{self.base_url}{path}?{urllib.parse.urlencode(clean)}"
        getter = self.getter or self._http_get
        with self._sem:
            waited = 0.0
            attempt = 0
            while True:
                status, body = getter(url)
                if status in OUTAGE_STATUSES and waited < self.outage_wait_s:
                    if waited == 0:
                        log.warning(
                            "Theta 终端不可用（%s），最多等待 %.0fs：%s",
                            status,
                            self.outage_wait_s,
                            path,
                        )
                    self.sleep(self.outage_poll_s)
                    waited += self.outage_poll_s
                    continue
                if status == 200:
                    # 实测（2026-10-22）：无数据时可能返回 200 + 文本
                    # "No data found for your request"
                    try:
                        data = json.loads(body) if body.strip() else []
                    except json.JSONDecodeError:
                        return []
                    if isinstance(data, list):
                        return _flatten(data)
                    # 实测（2026-10-22）终端返回 {"response": [...]}；兼容 {"data": [...]}
                    return _flatten(data.get("response", data.get("data", [])))
                if status == 472:  # NO_DATA
                    return []
                if status == 429 and attempt < self.retries - 1:
                    self.sleep(float(2**attempt))  # 1,2,4,8,16 s
                    attempt += 1
                    continue
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

    def history_trade(
        self,
        symbol: str,
        expiration: dt.date,
        date: dt.date,
        strike: str,
        right: str,
        **kw,
    ) -> pd.DataFrame:
        """逐笔成交（OPRA）：price / size / exchange / condition / ts。单合约单日（可带 start_time /
        end_time，美东 HH:MM:SS.mmm）。2026-10-22 实测 Options Standard 可用。"""
        rows = self.get_json(
            "/option/history/trade",
            symbol=symbol,
            expiration=expiration.strftime("%Y%m%d"),
            strike=strike,
            right=right,
            date=date.strftime("%Y%m%d"),
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
        self, symbol: str, date: dt.date, expiration: dt.date | None = None, max_dte: int = MAX_DTE
    ) -> pd.DataFrame:
        # 实测（2026-10-22）：当日数据必须指定 expiration（400 "Cannot fetch current-day data
        # without specifying an expiration"），所以 bulk 模式按到期日逐个请求
        rows = self.get_json(
            "/option/history/open_interest",
            symbol=symbol,
            expiration=expiration.strftime("%Y%m%d") if expiration else "*",
            date=date.strftime("%Y%m%d"),
            max_dte=None if expiration else max_dte,
        )
        return _frame(rows, symbol, date)


def _flatten(rows: list) -> list[dict]:
    """终端真实形状（2026-10-22 实测）：
    [{contract:{symbol,expiration,right,strike}, data:[{...}]}]；
    展平为每条记录带 expiration/right/strike 的扁平 dict；已是扁平的原样返回。"""
    out: list[dict] = []
    for r in rows:
        if isinstance(r, dict) and "contract" in r and "data" in r:
            c = r["contract"]
            base = {
                "expiration": c.get("expiration"),
                "right": str(c.get("right", "")).lower(),
                "strike": c.get("strike"),
                "symbol": c.get("symbol"),
            }
            for d in r["data"]:
                out.append({**base, **d})
        else:
            out.append(r)
    return out


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
    expirations: Iterable[dt.date], day: dt.date, max_dte: int = MAX_DTE, n_nearest: int = 2
) -> list[dt.date]:
    """当日要拉的到期日 = 最近 n_nearest 个（≥ 当日，不论 DTE）∪ DTE ≤ max_dte 的全部。

    MU 在 2026-07 之前只有周五到期：周五 10:45 后（D-001）只能用下周五（DTE 7）的合约，
    只取 DTE ≤ 5 会导致回放找不到可买合约（负责人 2026-10-22 指令）。
    """
    fut = sorted(e for e in set(expirations) if e >= day)
    near = set(fut[:n_nearest])
    within = {e for e in fut if (e - day).days <= max_dte}
    return sorted(near | within)


def band_bounds(px_range: tuple[float, float, float], band: float) -> tuple[float, float]:
    """行权价保留区间 = [当日最低价×(1−band), 当日最高价×(1+band)]。

    用全天高低点而不是开盘价：盘中价格跑出开盘价±band 时，需要的行权价已经在原始响应里
    （请求本来就是 strike=*），只是本地过滤要放宽，不必补拉。
    """
    _, lo, hi = px_range
    return lo * (1 - band), hi * (1 + band)


def strike_filter(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["strike"] >= lo) & (df["strike"] <= hi)].reset_index(drop=True)


def day_price_range(root: Path, und: str, day: dt.date) -> tuple[float, float, float] | None:
    """当日 RTH (开盘价, 最低价, 最高价)，来自长桥 bars_1m；按交易日历过滤，
    wick_spike（疑似坏成交影线）收回到实体后再取高低点（zr/data/bar_quality.py）。"""
    from zr.data.bar_quality import clean_wicks
    from zr.data.calendar import rth_mask

    f = root / "bars_1m" / f"underlying={und}" / f"year={day.year}" / "bars.parquet"
    if not f.exists():
        return None
    bars = pd.read_parquet(f, columns=["ts", "open", "high", "low", "close", "volume", "session"])
    bars = bars[bars["session"] == "Intraday"].set_index("ts").sort_index()
    bars = bars[rth_mask(bars.index)]
    t = bars.index.tz_convert(ET)
    # 前一日的 bar 一起送进去，让 ATR 有历史
    sel_mask = t.date == day
    if not sel_mask.any():
        return None
    first = int(np.argmax(sel_mask))
    window = bars.iloc[max(0, first - 400) : first + int(sel_mask.sum())]
    clean = clean_wicks(window)
    sel = clean[window.index.tz_convert(ET).date == day]
    return float(sel["open"].iloc[0]), float(sel["low"].min()), float(sel["high"].max())


# ---------- 落盘 ----------


def day_dir(root: Path, kind: str, und: str, day: dt.date) -> Path:
    return root / kind / f"underlying={und}" / f"date={day.isoformat()}"


def requested_expirations(root: Path, und: str) -> dict[str, set[str]]:
    """manifest 中每个日期已请求过的到期日（所有成功记录的并集；含没有数据的到期日）。"""
    out: dict[str, set[str]] = {}
    mf = root / "options_1m" / "manifest.jsonl"
    if not mf.exists():
        return out
    for line in mf.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("underlying") != und or "error" in r:
            continue
        out.setdefault(str(r["date"]), set()).update(r.get("expirations", []))
    return out


def day_done(root: Path, und: str, day: dt.date) -> bool:
    d = day_dir(root, "options_1m", und, day)
    return all((d / f"{n}.parquet").exists() for n in ("quotes", "ohlc", "oi"))


def write_day(
    root: Path,
    und: str,
    day: dt.date,
    quotes: pd.DataFrame,
    ohlc: pd.DataFrame,
    oi: pd.DataFrame,
    merge: bool = False,
) -> None:
    """落盘；merge=True 时与已有文件合并去重（补拉缺失到期日用）。"""
    d = day_dir(root, "options_1m", und, day)
    d.mkdir(parents=True, exist_ok=True)
    keys = {"quotes": ["ts", "symbol"], "ohlc": ["ts", "symbol"], "oi": ["symbol"]}
    for name, df in (("quotes", quotes), ("ohlc", ohlc), ("oi", oi)):
        p = d / f"{name}.parquet"
        if merge and p.exists():
            old = pd.read_parquet(p)
            if len(old) and "symbol" in old.columns:
                df = pd.concat([old, df], ignore_index=True) if len(df) else old
                df = df.drop_duplicates(keys[name], keep="last").reset_index(drop=True)
        df.to_parquet(p, index=False)


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
    px_range: tuple[float, float, float] | None,
    only: list[dt.date] | None = None,
) -> dict:
    """拉一天；only 给定时只拉这些到期日并与已有文件合并（补拉）。"""
    t0 = time.monotonic()
    exps = sorted(only) if only is not None else select_expirations(expirations, day)
    band = STRIKE_BAND.get(und, 0.05)
    bounds = band_bounds(px_range, band) if px_range else None
    quotes, ohlc = [], []
    # 每个到期日一次请求（strike=*），本地按带宽过滤；并发由 client 信号量限制
    with ThreadPoolExecutor(max_workers=client.concurrency) as ex:
        q_futs = {e: ex.submit(client.history_quote, und, e, day) for e in exps}
        o_futs = {e: ex.submit(client.history_ohlc, und, e, day) for e in exps}
        oi_futs = {e: ex.submit(client.history_open_interest, und, day, e) for e in exps}
        for e in exps:
            quotes.append(q_futs[e].result())
            ohlc.append(o_futs[e].result())
        oi_parts = [oi_futs[e].result() for e in exps]
    oi = (
        pd.concat([x for x in oi_parts if not x.empty], ignore_index=True)
        if any(not x.empty for x in oi_parts)
        else pd.DataFrame()
    )
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
    chain_min = float(q["strike"].min()) if len(q) else None
    chain_max = float(q["strike"].max()) if len(q) else None
    if bounds is not None:
        q, o, oi = (
            strike_filter(q, *bounds),
            strike_filter(o, *bounds),
            strike_filter(oi, *bounds),
        )
    write_day(root, und, day, q, o, oi, merge=only is not None)
    rec = {
        "underlying": und,
        "date": day,
        "expirations": [e.isoformat() for e in exps],
        "open_px": px_range[0] if px_range else None,
        "day_low": px_range[1] if px_range else None,
        "day_high": px_range[2] if px_range else None,
        "band": band,
        "band_lo": bounds[0] if bounds else None,
        "band_hi": bounds[1] if bounds else None,
        "strike_min": float(q["strike"].min()) if len(q) else None,
        "strike_max": float(q["strike"].max()) if len(q) else None,
        "chain_min": chain_min,
        "chain_max": chain_max,
        "rows_quotes": len(q),
        "rows_ohlc": len(o),
        "rows_oi": len(oi),
        "seconds": round(time.monotonic() - t0, 1),
        "supplement": only is not None,
    }
    _append_manifest(root, rec)
    return rec


def bulk_quotes_1m(
    client: ThetaClient,
    root: Path,
    und: str,
    days: Iterable[dt.date],
    *,
    px_fn: Callable[[dt.date], tuple[float, float, float] | None] | None = None,
) -> list[dict]:
    days = list(days)
    if not days:
        return []
    expirations = client.list_expirations(und)
    requested = requested_expirations(root, und)
    out = []
    for day in days:
        need = select_expirations(expirations, day)
        if day_done(root, und, day):
            have = requested.get(day.isoformat(), set())
            missing = [e for e in need if e.isoformat() not in have]
            if not missing:
                continue
            only: list[dt.date] | None = missing
        else:
            only = None
        px_range = px_fn(day) if px_fn else day_price_range(root, und, day)
        try:
            rec = fetch_day(client, root, und, day, expirations, px_range, only=only)
        except ThetaError as e:
            rec = {"underlying": und, "date": day, "error": str(e)}
            _append_manifest(root, rec)
            log.error("%s %s 失败：%s", und, day, e)
        log.info("%s", rec)
        out.append(rec)
    return out


def trading_days_from_bars(root: Path, und: str, start: dt.date, end: dt.date) -> list[dt.date]:
    """交易日 = 交易日历（zr/data/calendar.py）∩ 长桥 1 分钟 K 有数据的日期。

    长桥 K 线在全天休市日也有零成交记录，不能单独作日历（2026-10-22 实测）。
    """
    from zr.data.calendar import is_trading_day

    days: set[dt.date] = set()
    for f in sorted((root / "bars_1m" / f"underlying={und}").glob("year=*/bars.parquet")):
        ts = pd.read_parquet(f, columns=["ts", "session"])
        ts = ts[ts["session"] == "Intraday"]["ts"].dt.tz_convert(ET).dt.date
        days.update(d for d in ts.unique() if start <= d <= end)
    return sorted(d for d in days if is_trading_day(d))


# ---------- 覆盖检查 ----------


def _chain_extents(root: Path, und: str) -> dict[str, tuple[float | None, float | None]]:
    """manifest 里每个日期原始链的 (chain_min, chain_max)，老记录没有则 (None, None)。"""
    out: dict[str, tuple[float | None, float | None]] = {}
    mf = root / "options_1m" / "manifest.jsonl"
    if not mf.exists():
        return out
    for line in mf.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("underlying") == und and "error" not in r:
            out[str(r["date"])] = (r.get("chain_min"), r.get("chain_max"))
    return out


def coverage_report(root: Path, und: str, days: Iterable[dt.date]) -> pd.DataFrame:
    """对照正股当日高低点，检查已落盘日期的行权价范围是否覆盖 [low×(1−band), high×(1+band)]。

    容差 = 该侧最外两档的间距（MU 近 ATM 有 0.5 档、远端 1 档，不能用全局最小间距）；
    若该侧已到原始链的尽头（manifest 的 chain_min/max），视为该侧覆盖——链上没有更远的行权价可拉。
    覆盖不足的日期用 `--repull-uncovered` 删掉后由续传重新拉取。
    """
    band = STRIKE_BAND.get(und, 0.05)
    chain = _chain_extents(root, und)
    rows = []
    for day in days:
        d = day_dir(root, "options_1m", und, day)
        qp = d / "quotes.parquet"
        if not qp.exists():
            continue
        px = day_price_range(root, und, day)
        q = pd.read_parquet(qp)
        has = len(q) > 0 and "strike" in q.columns  # 空日（无数据）写出的 parquet 没有列
        smin, smax = (float(q["strike"].min()), float(q["strike"].max())) if has else (None, None)
        if px is None or smin is None:
            rows.append({"date": day, "covered": None, "strike_min": smin, "strike_max": smax})
            continue
        lo, hi = band_bounds(px, band)
        strikes = np.sort(q["strike"].unique())
        lo_step = float(strikes[1] - strikes[0]) if len(strikes) > 1 else 0.0
        hi_step = float(strikes[-1] - strikes[-2]) if len(strikes) > 1 else 0.0
        cmin, cmax = chain.get(day.isoformat(), (None, None))
        low_ok = smin <= lo + lo_step or (cmin is not None and smin <= cmin)
        high_ok = smax >= hi - hi_step or (cmax is not None and smax >= cmax)
        rows.append(
            {
                "date": day,
                "covered": bool(low_ok and high_ok),
                "need_lo": round(lo, 2),
                "need_hi": round(hi, 2),
                "strike_min": smin,
                "strike_max": smax,
                "chain_min": cmin,
                "chain_max": cmax,
                "day_low": px[1],
                "day_high": px[2],
            }
        )
    return pd.DataFrame(rows)


def drop_days(root: Path, und: str, days: Iterable[dt.date]) -> int:
    import shutil

    n = 0
    for day in days:
        d = day_dir(root, "options_1m", und, day)
        if d.is_dir():
            shutil.rmtree(d)
            n += 1
    return n


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
    parser.add_argument(
        "--coverage-check", action="store_true", help="只检查已落盘日期的行权价覆盖"
    )
    parser.add_argument(
        "--repull-uncovered", action="store_true", help="覆盖不足的日期删掉以便续传重拉"
    )
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
        if args.coverage_check or args.repull_uncovered:
            cov = coverage_report(root, und, days)
            bad = cov[cov["covered"] == False]["date"].tolist() if len(cov) else []  # noqa: E712
            log.info(
                "%s 覆盖检查：已落盘 %d 天，覆盖不足 %d 天 %s",
                und,
                len(cov),
                len(bad),
                [d.isoformat() for d in bad][:20],
            )
            if args.repull_uncovered:
                from zr.data.calendar import is_trading_day

                base = root / "options_1m" / f"underlying={und}"
                stored = (
                    [dt.date.fromisoformat(p.name[5:]) for p in base.glob("date=*")]
                    if base.is_dir()
                    else []
                )
                non_trading = [d for d in stored if start <= d <= end and not is_trading_day(d)]
                if non_trading:
                    log.info(
                        "%s 删除休市日目录 %d 个：%s",
                        und,
                        drop_days(root, und, non_trading),
                        [d.isoformat() for d in non_trading],
                    )
                if bad:
                    log.info("%s 删除覆盖不足 %d 天以便重拉", und, drop_days(root, und, bad))
            if args.coverage_check:
                continue
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
