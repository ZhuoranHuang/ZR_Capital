# ruff: noqa: E501  （中文说明较长）
"""ThetaData 退订前的收尾拉取（Mac 本地，需 Theta Terminal；只拉数据，不做研究）。负责人 2026-11-18 指令。

P1  MU、QQQ、SPY 期权 1 分钟报价 / OHLC / 持仓，从本地最后日期补到最新交易日（沿用 zr.data.thetadata.bulk_quotes_1m 的
    合约范围：最近 2 个到期日 ∪ 到期日 5 天内；行权价在 [当日最低 ×(1−b), 当日最高 ×(1+b)]，SPY / QQQ b = 3%、MU 8%）。
    → options_1m/underlying=X/date=D/{quotes,ohlc,oi}.parquet
P2  QQQ、SPY：1 分钟 OHLC（含成交量与笔数）本地 options_1m 已覆盖「到期日 2 天内、开盘价 ±3%」，随 P1 补到最新；
    每日持仓量（全部到期日与行权价）2023-10-02 起 → options_eod/underlying=X/date=D/oi.parquet
P3  IWM、NVDA、TSLA、AAPL、AMD 期权 1 分钟报价，最近 12 个月，到期日 7 天内，行权价带同上（b = 3%）
    → options_1m/underlying=X/date=D/quotes.parquet
P4  日线级：自 2016-01-01，每个标的每天全部合约的收盘报价与成交量 + 隐含波动率与希腊值（/option/history/greeks/eod；
    该接口没有数据的日子退回 /option/history/eod，只有报价与成交量），以及持仓量
    → options_eod/underlying=X/date=D/{eod,oi}.parquet
价格带来源：ThetaData 正股日线（/stock/history/eod，开 / 高 / 低）。交易日：zr.data.calendar。
断点续传：文件已存在即跳过；没有数据的日子写 0 行文件。订阅失效（401 / 403）立即停止并写进 theta_final/PROGRESS.md。
运行：python -m zr.data.theta_final --phase time|p1|p2|p3|p4|all|coverage
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from zr.data import thetadata as td

log = logging.getLogger("zr.theta_final")
P1_SYMS = ("MU", "QQQ", "SPY")
P2_SYMS = ("QQQ", "SPY")
P3_SYMS = ("IWM", "NVDA", "TSLA", "AAPL", "AMD")
P4_SYMS = (
    "SPY",
    "QQQ",
    "IWM",
    "MU",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "AMD",
    "SMH",
    "TLT",
    "GLD",
)
P3_BAND, P3_MAX_DTE = 0.03, 7
EOD_KEEP = [
    "open", "high", "low", "close", "volume", "count", "bid", "ask", "bid_size", "ask_size",
    "implied_vol", "iv_error", "delta", "gamma", "theta", "vega", "rho", "underlying_price",
]  # fmt: skip


class Stop(Exception):
    """订阅失效或无权限。"""


def _guard(e: td.ThetaError) -> None:
    if e.status in (401, 403):
        raise Stop(f"HTTP {e.status}: {str(e)[:200]}") from None


# ---------------- 价格带与交易日 ----------------


class Prices:
    """ThetaData 正股日线 → {日: (开, 低, 高)}，按标的缓存；一次请求一年。"""

    def __init__(self, client: td.ThetaClient):
        self.c = client
        self.cache: dict[str, dict[dt.date, tuple[float, float, float]]] = {}
        self.lock = threading.Lock()

    def load(self, sym: str, start: dt.date, end: dt.date) -> dict:
        with self.lock:
            have = self.cache.setdefault(sym, {})
            y0 = start.year
            while y0 <= end.year:
                a, b = max(start, dt.date(y0, 1, 1)), min(end, dt.date(y0, 12, 31))
                if not any(a <= d <= b for d in have):
                    rows = self.c.get_json(
                        "/stock/history/eod",
                        symbol=sym,
                        start_date=a.strftime("%Y%m%d"),
                        end_date=b.strftime("%Y%m%d"),
                    )
                    for r in rows:
                        ts = r.get("last_trade") or r.get("created")
                        if ts and r.get("open") and r.get("high") and r.get("low"):
                            have[dt.date.fromisoformat(ts[:10])] = (
                                float(r["open"]),
                                float(r["low"]),
                                float(r["high"]),
                            )
                y0 += 1
            return have

    def fn(self, sym: str, start: dt.date, end: dt.date):
        m = self.load(sym, start, end)
        return lambda day: m.get(day)


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    from zr.data.calendar import trading_days as cal_days

    return cal_days(start, end)


def latest_complete_day(now: dt.datetime | None = None) -> dt.date:
    """最新已收盘的交易日（美东 17:30 之后才算当日完整）。"""
    from zoneinfo import ZoneInfo

    et = (now or dt.datetime.now(dt.UTC)).astimezone(ZoneInfo("America/New_York"))
    d = et.date() if (et.hour, et.minute) >= (17, 30) else et.date() - dt.timedelta(days=1)
    return trading_days(d - dt.timedelta(days=10), d)[-1]


# ---------------- 落盘 ----------------


def eod_dir(root: Path, und: str, day: dt.date) -> Path:
    return root / "options_eod" / f"underlying={und}" / f"date={day}"


def write_atomic(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)


class Progress:
    def __init__(self, root: Path, name: str, total: int):
        self.p = root / "theta_final" / "PROGRESS.md"
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.name, self.total, self.n = name, total, 0
        self.t0 = time.monotonic()
        self.lock = threading.Lock()

    def step(self, what: str) -> None:
        with self.lock:
            self.n += 1
            if self.n % 50 == 0 or self.n == self.total:
                el = time.monotonic() - self.t0
                rem = (self.total - self.n) * el / self.n / 3600
                self.p.write_text(
                    f"# ThetaData 收尾拉取进度\n\n- 阶段 {self.name}：{self.n:,} / {self.total:,}（最新 {what}）\n"
                    f"- 已用 {el / 3600:.2f} 小时，预计剩余 {rem:.2f} 小时\n- 更新 {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n",
                    encoding="utf-8",
                )
                log.info("%s %d/%d %s 剩余约 %.2f 小时", self.name, self.n, self.total, what, rem)


def note_stop(root: Path, phase: str, msg: str) -> None:
    p = root / "theta_final" / "STOPPED.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"# 停止\n\n- 阶段 {phase}\n- 原因：{msg}\n- 时间 {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


# ---------------- P1：MU / QQQ / SPY 1 分钟补到最新 ----------------


def p1(client: td.ThetaClient, root: Path, end: dt.date, syms=P1_SYMS) -> dict:
    px = Prices(client)
    out = {}
    for und in syms:
        dirs = sorted((root / "options_1m" / f"underlying={und}").glob("date=*"))
        done = [
            dt.date.fromisoformat(d.name[5:])
            for d in dirs
            if td.day_done(root, und, dt.date.fromisoformat(d.name[5:]))
        ]
        start = (max(done) + dt.timedelta(days=1)) if done else dt.date(2023, 10, 2)
        days = trading_days(start, end)
        if not days:
            out[und] = {"days": 0}
            continue
        fn = px.fn(und, days[0], days[-1])
        t0 = time.monotonic()
        try:
            recs = td.bulk_quotes_1m(client, root, und, days, px_fn=fn)
        except td.ThetaError as e:
            _guard(e)
            raise
        errs = [r for r in recs if "error" in r]
        for r in errs:
            if " 401" in r["error"] or " 403" in r["error"] or "subscription" in r["error"].lower():
                raise Stop(r["error"][:200])
        out[und] = {
            "days": len(days),
            "first": str(days[0]),
            "last": str(days[-1]),
            "errors": len(errs),
            "no_price": sum(1 for d in days if fn(d) is None),
            "seconds": round(time.monotonic() - t0, 1),
        }
        log.info("P1 %s %s", und, out[und])
    return out


# ---------------- 每日全链持仓与日线（P2 持仓、P4） ----------------


def fetch_oi(client: td.ThetaClient, und: str, day: dt.date) -> pd.DataFrame:
    rows = client.get_json(
        "/option/history/open_interest", symbol=und, expiration="*", date=day.strftime("%Y%m%d")
    )
    df = td._frame(rows, und, day)
    return (
        df[
            [
                c
                for c in (
                    "ts",
                    "trade_date",
                    "symbol",
                    "underlying",
                    "expiry",
                    "strike",
                    "right",
                    "open_interest",
                )
                if c in df
            ]
        ]
        if len(df)
        else df
    )


def fetch_eod(client: td.ThetaClient, und: str, day: dt.date) -> pd.DataFrame:
    d8 = day.strftime("%Y%m%d")
    rows = client.get_json(
        "/option/history/greeks/eod", symbol=und, expiration="*", start_date=d8, end_date=d8
    )
    src = "greeks_eod"
    if not rows:
        rows = client.get_json(
            "/option/history/eod", symbol=und, expiration="*", start_date=d8, end_date=d8
        )
        src = "eod"
    if not rows:
        return pd.DataFrame()
    for r in (
        rows
    ):  # 两个接口的时间字段名不同；统一成 last_trade，避免 _frame 按 timestamp 解析出不同含义
        if "timestamp" in r:
            r["last_trade"] = r.pop("timestamp")
    df = td._frame(rows, und, day)
    front = ["trade_date", "symbol", "underlying", "expiry", "strike", "right"]
    keep = [c for c in EOD_KEEP if c in df]
    df = df[front + keep + [c for c in ("last_trade",) if c in df]].copy()
    df["source"] = src
    return df


def daily_jobs(
    root: Path, syms, start: dt.date, end: dt.date, kinds: tuple[str, ...]
) -> list[tuple[str, dt.date, str]]:
    days = trading_days(start, end)
    return [
        (u, d, k)
        for u in syms
        for d in days
        for k in kinds
        if not (eod_dir(root, u, d) / f"{k}.parquet").exists()
    ]


def run_daily(client: td.ThetaClient, root: Path, jobs: list, name: str) -> dict:
    pr = Progress(root, name, len(jobs))
    stop: dict = {}
    errors: list = []

    def work(job):
        if stop:
            return
        u, d, k = job
        try:
            df = fetch_oi(client, u, d) if k == "oi" else fetch_eod(client, u, d)
        except td.ThetaError as e:
            if e.status in (401, 403):
                stop["reason"] = f"HTTP {e.status}: {str(e)[:160]}（{u} {d} {k}）"
            else:
                errors.append((u, str(d), k, str(e)[:120]))
            return
        write_atomic(df, eod_dir(root, u, d) / f"{k}.parquet")
        pr.step(f"{u} {d} {k}")

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=client.concurrency) as ex:
        list(ex.map(work, jobs))
    res = {
        "jobs": len(jobs),
        "done": pr.n,
        "errors": len(errors),
        "error_sample": errors[:5],
        "seconds": round(time.monotonic() - t0, 1),
    }
    if stop:
        note_stop(root, name, stop["reason"])
        raise Stop(stop["reason"])
    return res


def p2(client: td.ThetaClient, root: Path, end: dt.date) -> dict:
    return run_daily(
        client, root, daily_jobs(root, P2_SYMS, dt.date(2023, 10, 2), end, ("oi",)), "P2 持仓"
    )


def p4(
    client: td.ThetaClient,
    root: Path,
    end: dt.date,
    syms=P4_SYMS,
    start: dt.date = dt.date(2016, 1, 1),
) -> dict:
    out = {}
    for u in syms:  # 按标的顺序，逐个拉完
        out[u] = run_daily(
            client, root, daily_jobs(root, (u,), start, end, ("eod", "oi")), f"P4 {u}"
        )
        log.info("P4 %s %s", u, out[u])
    return out


# ---------------- P3：五只标的 1 分钟报价 ----------------


def p3_day(
    client: td.ThetaClient, root: Path, und: str, day: dt.date, exps: list[dt.date], px
) -> dict:
    sel = [e for e in exps if 0 <= (e - day).days <= P3_MAX_DTE]
    parts = [client.history_quote(und, e, day) for e in sel]
    q = (
        pd.concat([x for x in parts if not x.empty], ignore_index=True)
        if any(not x.empty for x in parts)
        else pd.DataFrame()
    )
    if len(q) and px is not None:
        lo, hi = td.band_bounds(px, P3_BAND)
        q = td.strike_filter(q, lo, hi)
    write_atomic(q, td.day_dir(root, "options_1m", und, day) / "quotes.parquet")
    return {
        "underlying": und,
        "date": str(day),
        "expirations": [str(e) for e in sel],
        "rows_quotes": len(q),
        "px": px,
    }


def p3(client: td.ThetaClient, root: Path, end: dt.date, syms=P3_SYMS, months: int = 12) -> dict:
    start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date() + dt.timedelta(days=1)
    days = trading_days(start, end)
    px = Prices(client)
    jobs = []
    for u in syms:
        exps = client.list_expirations(u)
        fn = px.fn(u, days[0], days[-1])
        jobs += [
            (u, d, exps, fn)
            for d in days
            if not (td.day_dir(root, "options_1m", u, d) / "quotes.parquet").exists()
        ]
    pr = Progress(root, "P3 报价", len(jobs))
    stop: dict = {}
    errors: list = []
    no_px = 0
    mf = root / "options_1m" / "manifest_p3.jsonl"

    def work(job):
        nonlocal no_px
        if stop:
            return
        u, d, exps, fn = job
        p = fn(d)
        try:
            rec = p3_day(client, root, u, d, exps, p)
        except td.ThetaError as e:
            if e.status in (401, 403):
                stop["reason"] = f"HTTP {e.status}: {str(e)[:160]}（{u} {d}）"
            else:
                errors.append((u, str(d), str(e)[:120]))
            return
        with pr.lock:
            no_px += p is None
            with mf.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pr.step(f"{u} {d}")

    t0 = time.monotonic()
    # 每个作业内部顺序请求各到期日；并发作业数 = 账户并发上限
    with ThreadPoolExecutor(max_workers=client.concurrency) as ex:
        list(ex.map(work, jobs))
    res = {
        "jobs": len(jobs),
        "done": pr.n,
        "errors": len(errors),
        "error_sample": errors[:5],
        "no_price_days": no_px,
        "first": str(days[0]),
        "last": str(days[-1]),
        "seconds": round(time.monotonic() - t0, 1),
    }
    if stop:
        note_stop(root, "P3", stop["reason"])
        raise Stop(stop["reason"])
    return res


# ---------------- 测速 ----------------


def timing(client: td.ThetaClient, end: dt.date) -> dict:
    """各接口单个标的一个月（最近约 21 个交易日）的耗时，不落盘。"""
    days = trading_days(end - dt.timedelta(days=30), end)
    out: dict = {"days": len(days), "first": str(days[0]), "last": str(days[-1])}

    def clock(name, fn, n_parallel=client.concurrency):
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=n_parallel) as ex:
            sizes = list(ex.map(fn, days))
        out[name] = {"seconds": round(time.monotonic() - t0, 1), "rows": int(sum(sizes))}

    clock("oi_full SPY", lambda d: len(fetch_oi(client, "SPY", d)))
    clock("eod_greeks SPY", lambda d: len(fetch_eod(client, "SPY", d)))
    clock("eod_greeks MU", lambda d: len(fetch_eod(client, "MU", d)))
    old = trading_days(dt.date(2017, 3, 1), dt.date(2017, 3, 31))
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=client.concurrency) as ex:
        n = sum(ex.map(lambda d: len(fetch_eod(client, "SPY", d)), old))
    out["eod_greeks SPY 2017-03"] = {
        "seconds": round(time.monotonic() - t0, 1),
        "rows": int(n),
        "days": len(old),
    }
    exps = client.list_expirations("NVDA")

    def q(d):
        sel = [e for e in exps if 0 <= (e - d).days <= P3_MAX_DTE]
        return sum(len(client.history_quote("NVDA", e, d)) for e in sel)

    clock("quotes_1m NVDA dte<=7", q)
    exps_s = client.list_expirations("SPY")

    def q1(d):
        sel = td.select_expirations(exps_s, d)
        return sum(
            len(client.history_quote("SPY", e, d)) + len(client.history_ohlc("SPY", e, d))
            for e in sel
        )

    clock("quotes+ohlc_1m SPY (P1 范围)", q1)
    return out


# ---------------- 覆盖检查 ----------------


def coverage(root: Path, end: dt.date) -> dict:
    out: dict = {}
    for u in sorted({*P1_SYMS, *P3_SYMS}):
        base = root / "options_1m" / f"underlying={u}"
        ds = sorted(
            dt.date.fromisoformat(p.parent.name[5:]) for p in base.glob("date=*/quotes.parquet")
        )
        if not ds:
            continue
        exp = trading_days(ds[0], end)
        out[f"options_1m {u}"] = {
            "first": str(ds[0]),
            "last": str(ds[-1]),
            "days": len(ds),
            "expected": len(exp),
            "missing": [str(d) for d in sorted(set(exp) - set(ds))][:10],
        }
    for u in P4_SYMS:
        base = root / "options_eod" / f"underlying={u}"
        for k in ("eod", "oi"):
            files = sorted(base.glob(f"date=*/{k}.parquet"))
            if not files:
                continue
            ds = [dt.date.fromisoformat(p.parent.name[5:]) for p in files]
            exp = trading_days(ds[0], end)
            empty = sum(1 for p in files if p.stat().st_size < 4000)
            out[f"options_eod {u} {k}"] = {
                "first": str(ds[0]),
                "last": str(ds[-1]),
                "days": len(ds),
                "expected": len(exp),
                "missing": len(set(exp) - set(ds)),
                "small_files(<4KB，多为无数据)": empty,
            }
    return out


def coverage_section(root: Path, end: dt.date) -> str:
    cov = coverage(root, end)
    stop = root / "theta_final" / "STOPPED.md"
    L = [
        f"## ThetaData 收尾拉取（`zr/data/theta_final.py --phase coverage`，{dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC 生成；应有交易日算到 {end}）",
        "",
        "`options_1m/`：期权 1 分钟（MU、QQQ、SPY 含报价 / OHLC / 持仓；IWM、NVDA、TSLA、AAPL、AMD 只有报价，到期日 7 天内）。"
        "`options_eod/`：日线级，每个标的每天全部合约——`eod.parquet`（收盘报价、成交量、隐含波动率与希腊值；source = eod 的日子只有报价与成交量）与 `oi.parquet`（持仓量）。",
        "",
        "| 数据集 | 最早 | 最晚 | 已有天数 | 应有天数 | 缺 | 小文件（< 4 KB，多为当日无数据） |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in cov.items():
        miss = v["missing"] if isinstance(v["missing"], int) else len(v["missing"])
        small = v.get("small_files(<4KB，多为无数据)", "—")
        L.append(
            f"| `{k}` | {v['first']} | {v['last']} | {v['days']:,} | {v['expected']:,} | {miss} | {small} |"
        )
    L += [
        "",
        ("**拉取中途停止**：" + stop.read_text(encoding="utf-8").replace("\n", " ").strip())
        if stop.exists()
        else "拉取没有因订阅失效而停止。",
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["time", "p1", "p2", "p3", "p4", "all", "coverage", "write-doc"],
        required=True,
    )
    ap.add_argument("--end", help="最新交易日，默认自动判断")
    ap.add_argument("--symbols", nargs="*")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(get_settings().zr_data_root)
    end = dt.date.fromisoformat(a.end) if a.end else latest_complete_day()
    if a.phase == "write-doc":
        from zr.data.inventory import replace_section

        doc = Path(__file__).resolve().parents[2] / "docs" / "DATA_INVENTORY.md"
        replace_section(doc, "## ThetaData 收尾拉取", coverage_section(root, end))
        print(doc)
        return 0
    if a.phase == "coverage":
        print(json.dumps(coverage(root, end), ensure_ascii=False, indent=1))
        return 0
    client = td.ThetaClient()
    res: dict = {"end": str(end)}
    try:
        if a.phase == "time":
            res["timing"] = timing(client, end)
        if a.phase in ("p1", "all"):
            res["p1"] = p1(client, root, end)
        if a.phase in ("p2", "all"):
            res["p2"] = p2(client, root, end)
        if a.phase in ("p3", "all"):
            res["p3"] = p3(client, root, end)
        if a.phase in ("p4", "all"):
            res["p4"] = p4(client, root, end, tuple(a.symbols) if a.symbols else P4_SYMS)
    except Stop as e:
        res["stopped"] = str(e)
        note_stop(root, a.phase, str(e))
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
