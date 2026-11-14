# ruff: noqa: E501  （中文日志与报告行较长）
"""R-057 数据拉取（只在 VPS 上运行，长桥只读行情接口；不碰录制器与现有定时任务）。落盘到 $ZR_DATA_ROOT/sip/：

- universe/：纳斯达克 Symbol Directory 原文件、过滤后的名单（zr.research.sip_orb.parse_symbol_directory）
- daily/part-*.parquet：不复权与前复权日线（2023-08-01 起），state/daily_done.txt
- derived/eligibility.parquet：每日合格集合（zr.research.sip_orb.daily_eligibility）
- bars5/<代码>.parquet：常规时段 5 分钟 K（不复权，2023-10-02 起），按平均成交额从高到低拉取，state/bars5_done.txt
- bars1/day=<日期>/part-*.parquet：常规时段 1 分钟 K（不复权），每日合格 ∩ RelVol ≥ 1 ∩ 非十字星的 RelVol 前 50，state/bars1_done.txt
- PROGRESS.md：阶段、计数、请求速率、预计剩余时间、错误（随现有同步任务传回 Mac）

限频：美股交易时段（交易日 09:00–16:30 ET，含前后 30 分钟余量）每次请求间隔 ≥ 0.5 秒，其余时段 ≥ 0.2 秒；
限频错误指数退避（2、4、8 … 300 秒）。断点续传：各阶段按 state/*.txt 跳过已完成的标的 / 标的日。
已核实：history_candlesticks_by_offset 的 time 参数按美东时间解释、返回 naive UTC（CLAUDE.md）。
其余用法（Period.Day / Min_5、AdjustType.ForwardAdjust、TradeSessions.Intraday、forward=True 取当日 1 分钟 K）在 --phase test
里实测并写入报告（未验证前不当作事实）。
运行（VPS）：cd ~/ZR_Capital && PYTHONPATH=<本代码副本> .venv/bin/python -m zr.data.sip_pull --phase test|all
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from zr.research import sip_orb as so

log = logging.getLogger("zr.sip_pull")
ET = "America/New_York"
DAILY_FROM = dt.date(2023, 8, 1)
INTRA_FROM = dt.date(2023, 11, 1)  # 长桥分钟 K 下限（301600 "out of minute candlestick limit begin date:2023-11-01"，2026-11-14 实测）
TOP_1M = 50
PRELIM_N = 500
MIN_FREE_GB = 5.0
MAX_HOURS = 72.0
RATE_WORDS = ("rate", "too many", "frequen", "301606", "429", "limit")
NOTFOUND_WORDS = ("not found", "invalid symbol", "301600", "no such", "not exist", "unknown symbol")


# ---------------- 限频与重试 ----------------


def us_session(now_utc: dt.datetime, is_trading_day=None) -> bool:
    """美股交易时段（含前后 30 分钟余量）：交易日 09:00–16:30 ET。"""
    t = (
        pd.Timestamp(now_utc).tz_convert(ET)
        if pd.Timestamp(now_utc).tzinfo
        else pd.Timestamp(now_utc, tz="UTC").tz_convert(ET)
    )
    if is_trading_day is None:
        from zr.data.calendar import is_trading_day as itd

        is_trading_day = itd
    m = t.hour * 60 + t.minute
    return bool(is_trading_day(t.date())) and 540 <= m < 990


def classify(msg: str) -> str:
    m = msg.lower()
    if any(w in m for w in RATE_WORDS):
        return "rate"
    if any(w in m for w in NOTFOUND_WORDS):
        return "notfound"
    return "other"


class NotFound(Exception):
    pass


class Client:
    """所有长桥请求经此发出：按时段控制间隔，限频错误指数退避，其他错误最多重试 3 次。"""

    def __init__(self, qctx: Any, clock=time.time, sleep=time.sleep, now=None, is_trading_day=None):
        self.q, self.clock, self.sleep = qctx, clock, sleep
        self.now = now or (lambda: dt.datetime.now(dt.UTC))
        self.itd = is_trading_day
        self.last = 0.0
        self.n = 0
        self.err = Counter()
        self.backoff_s = 0.0

    def interval(self) -> float:
        return 0.5 if us_session(self.now(), self.itd) else 0.2

    def call(self, name: str, *args):
        tries_other, k = 0, 0
        while True:
            wait = self.last + self.interval() - self.clock()
            if wait > 0:
                self.sleep(wait)
            self.last = self.clock()
            try:
                out = getattr(self.q, name)(*args)
                self.n += 1
                return out
            except Exception as e:  # noqa: BLE001 - 长桥 SDK 异常类型不统一，按消息分类
                kind = classify(str(e))
                self.err[kind] += 1
                if kind == "rate":
                    k += 1
                    w = min(300.0, 2.0**k)
                    self.backoff_s += w
                    log.warning("限频（%s），退避 %.0f 秒", str(e)[:120], w)
                    self.sleep(w)
                    continue
                if kind == "notfound":
                    raise NotFound(str(e)) from e
                tries_other += 1
                if tries_other >= 3:
                    raise
                self.sleep(2.0 * tries_other)


def et_naive(ts_utc: pd.Timestamp) -> dt.datetime:
    """UTC 时刻 → 美东 naive（history_candlesticks_by_offset 的 time 参数按美东解释）。"""
    return pd.Timestamp(ts_utc).tz_convert(ET).tz_localize(None).to_pydatetime()


def frame(symbol: str, bars) -> pd.DataFrame:
    rows = [
        {
            "ts": pd.Timestamp(b.timestamp).tz_localize("UTC")
            if pd.Timestamp(b.timestamp).tzinfo is None
            else pd.Timestamp(b.timestamp).tz_convert("UTC"),
            "symbol": symbol,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
            "turnover": float(b.turnover),
            "session": str(b.trade_session).split(".")[-1],
        }
        for b in bars
    ]
    return pd.DataFrame(
        rows,
        columns=["ts", "symbol", "open", "high", "low", "close", "volume", "turnover", "session"],
    )


# ---------------- 落盘与状态 ----------------


class Store:
    def __init__(self, root: Path, sub: str = "sip"):
        self.r = Path(root) / sub
        for d in ("universe", "daily", "derived", "bars5", "bars1", "state", "logs"):
            (self.r / d).mkdir(parents=True, exist_ok=True)

    def done(self, name: str) -> set[str]:
        p = self.r / "state" / f"{name}.txt"
        return set(p.read_text().split()) if p.exists() else set()

    def mark(self, name: str, keys) -> None:
        with open(self.r / "state" / f"{name}.txt", "a") as f:
            for k in keys:
                f.write(f"{k}\n")

    def fail(self, phase: str, key: str, msg: str) -> None:
        with open(self.r / "state" / "failed.tsv", "a") as f:
            f.write(f"{phase}\t{key}\t{msg[:200].replace(chr(9), ' ')}\n")

    def free_gb(self) -> float:
        return shutil.disk_usage(self.r).free / 1e9


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# ---------------- 各阶段 ----------------


def build_universe(st: Store) -> pd.DataFrame:
    u_dir = st.r / "universe"
    nas, oth = (u_dir / "nasdaqlisted.txt").read_text(), (u_dir / "otherlisted.txt").read_text()
    u, listed = so.parse_symbol_directory(nas, oth)
    u.to_parquet(u_dir / "universe.parquet", index=False)
    meta = {
        "listed_date": listed,
        "rows": len(u),
        "included": int(u["include"].sum()),
        "reasons": u["reason"].value_counts().to_dict(),
    }
    (u_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    return u[u["include"]]


def pull_daily(cl: Client, st: Store, symbols: list[str], batch: int = 100, progress=None) -> None:
    from longport.openapi import AdjustType, Period, TradeSessions

    done = st.done("daily_done")
    todo = [s for s in symbols if s not in done]
    buf, keys = [], []
    for i, s in enumerate(todo):
        try:
            for adj, at in (("none", AdjustType.NoAdjust), ("fwd", AdjustType.ForwardAdjust)):
                bars = cl.call(
                    "history_candlesticks_by_offset",
                    so.lb_symbol(s),
                    Period.Day,
                    at,
                    False,
                    1000,
                    None,
                    TradeSessions.Intraday,
                )
                f = frame(s, bars).assign(adj=adj)
                buf.append(
                    f[f["ts"].dt.tz_convert(ET).dt.date >= DAILY_FROM - dt.timedelta(days=3)]
                )
            keys.append(s)
        except NotFound as e:
            st.fail("daily", s, str(e))
            keys.append(s)
        except Exception as e:  # noqa: BLE001
            st.fail("daily", s, f"{type(e).__name__}: {e}")
        if len(keys) >= batch or i == len(todo) - 1:
            if buf:
                write_atomic(
                    pd.concat(buf, ignore_index=True),
                    st.r / "daily" / f"part-{int(time.time() * 1000)}.parquet",
                )
            st.mark("daily_done", keys)
            buf, keys = [], []
            if progress:
                progress(f"日线 {len(done) + i + 1}/{len(done) + len(todo)}")


def load_daily(st: Store) -> pd.DataFrame:
    fs = sorted((st.r / "daily").glob("part-*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["date"] = daily_date(d["ts"])
    return d.drop_duplicates(["symbol", "adj", "date"], keep="last")


def daily_date(ts: pd.Series) -> pd.Series:
    """日线时间戳 → 交易日。长桥日线时间戳的时刻在 --phase test 里实测（见 test_report.json）；
    这里取 UTC 时刻加 12 小时后的 UTC 日期，对「当日 00:00 UTC」「当日 04:00/05:00 UTC（美东 0 点）」两种写法都给出同一交易日。"""
    return (pd.to_datetime(ts, utc=True) + pd.Timedelta(hours=12)).dt.date


def eligibility(st: Store) -> pd.DataFrame:
    d = load_daily(st)
    raw = d[d["adj"] == "none"][
        ["symbol", "date", "open", "high", "low", "close", "volume", "turnover"]
    ]
    fwd = d[d["adj"] == "fwd"][["symbol", "date", "open"]].rename(columns={"open": "fwd_open"})
    m = raw.merge(fwd, on=["symbol", "date"], how="left")
    out = []
    for s, g in m.groupby("symbol", sort=False):
        e = so.daily_eligibility(g)
        e["symbol"] = s
        e["turnover"] = g.sort_values("date")["turnover"].to_numpy()
        out.append(e)
    E = pd.concat(out, ignore_index=True)
    write_atomic(E, st.r / "derived" / "eligibility.parquet")
    return E


def pull_bars5(cl: Client, st: Store, symbol: str, max_pages: int = 150) -> int:
    from longport.openapi import AdjustType, Period, TradeSessions

    cursor, parts, pages = None, [], 0
    while pages < max_pages:
        try:
            bars = cl.call(
                "history_candlesticks_by_offset",
                so.lb_symbol(symbol),
                Period.Min_5,
                AdjustType.NoAdjust,
                False,
                1000,
                cursor,
                TradeSessions.Intraday,
            )
        except NotFound:
            if pages == 0:
                raise
            break  # 翻到分钟 K 历史下限（301600）：已拉到的数据照常落盘
        pages += 1
        if not bars:
            break
        f = frame(symbol, bars)
        parts.append(f)
        earliest = f["ts"].min()
        nxt = et_naive(earliest)
        if cursor is not None and nxt >= cursor:
            break
        cursor = nxt
        if earliest.tz_convert(ET).date() < INTRA_FROM:
            break
    df = pd.concat(parts, ignore_index=True) if parts else frame(symbol, [])
    df = df.drop_duplicates("ts").sort_values("ts")
    df = df[df["ts"].dt.tz_convert(ET).dt.date >= INTRA_FROM]
    write_atomic(df, st.r / "bars5" / f"{symbol}.parquet")
    return pages


def opening_ranges(st: Store, symbols: list[str]) -> pd.DataFrame:
    """已拉取 5 分钟 K 的标的 → 每日开盘区间与 RelVol（缓存于 derived/or5/<代码>.parquet）。"""
    cache = st.r / "derived" / "or5"
    cache.mkdir(exist_ok=True)
    out = []
    for s in symbols:
        c = cache / f"{s}.parquet"
        p = st.r / "bars5" / f"{s}.parquet"
        if not p.exists():
            continue
        if not c.exists() or c.stat().st_mtime < p.stat().st_mtime:
            o = so.opening_range(pd.read_parquet(p))
            o["symbol"] = s
            write_atomic(o, c)
        out.append(pd.read_parquet(c))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def selection_1m(E: pd.DataFrame, ors: pd.DataFrame, last_day: dt.date) -> pd.DataFrame:
    c = ors.merge(E[["symbol", "date", "eligible"]], on=["symbol", "date"], how="left")
    c["eligible"] = c["eligible"].fillna(False)
    c = c[(c["date"] >= INTRA_FROM) & (c["date"] <= last_day)]
    return so.select(c, TOP_1M)


def pull_bars1(cl: Client, st: Store, pairs: pd.DataFrame, progress=None) -> None:
    from longport.openapi import AdjustType, Period, TradeSessions

    done = st.done("bars1_done")
    todo = pairs[[f"{r.symbol},{r.date}" not in done for r in pairs.itertuples()]]
    for k, (day, g) in enumerate(todo.groupby("date", sort=True)):
        buf, keys = [], []
        for s in g["symbol"]:
            t0 = dt.datetime.combine(day, dt.time(9, 30))  # 美东 naive
            try:
                bars = cl.call(
                    "history_candlesticks_by_offset",
                    so.lb_symbol(s),
                    Period.Min_1,
                    AdjustType.NoAdjust,
                    True,
                    400,
                    t0,
                    TradeSessions.Intraday,
                )
                f = frame(s, bars)
                buf.append(f[f["ts"].dt.tz_convert(ET).dt.date == day])
                keys.append(f"{s},{day}")
            except NotFound as e:
                st.fail("bars1", f"{s},{day}", str(e))
                keys.append(f"{s},{day}")
            except Exception as e:  # noqa: BLE001
                st.fail("bars1", f"{s},{day}", f"{type(e).__name__}: {e}")
        d = st.r / "bars1" / f"day={day}"
        d.mkdir(exist_ok=True)
        if buf:
            write_atomic(
                pd.concat(buf, ignore_index=True), d / f"part-{int(time.time() * 1000)}.parquet"
            )
        st.mark("bars1_done", keys)
        if progress and k % 10 == 0:
            progress(f"1 分钟 K：第 {k + 1}/{todo['date'].nunique()} 个交易日")


# ---------------- 编排 ----------------


class Progress:
    def __init__(self, st: Store, cl: Client):
        self.st, self.cl, self.t0, self.n0 = st, cl, time.time(), cl.n
        self.info: dict[str, Any] = {}

    def __call__(self, msg: str, **kw) -> None:
        self.info.update(kw)
        el = time.time() - self.t0
        rate = (self.cl.n - self.n0) / el if el > 0 else 0.0
        lines = [
            "# R-057 数据拉取进度（VPS，自动生成）",
            "",
            f"- 更新：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC",
            f"- 当前：{msg}",
            f"- 本次运行请求 {self.cl.n - self.n0:,} 次，用时 {el / 3600:.2f} 小时，平均 {rate:.2f} 次/秒；限频退避合计 {self.cl.backoff_s:.0f} 秒",
            f"- 错误计数：{dict(self.cl.err)}",
            f"- 磁盘剩余：{self.st.free_gb():.1f} GB",
        ]
        lines += [f"- {k}：{v}" for k, v in self.info.items()]
        (self.st.r / "PROGRESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info(msg)
        if self.st.free_gb() < MIN_FREE_GB:
            raise SystemExit(f"磁盘剩余 < {MIN_FREE_GB} GB，停止")


def last_complete_day() -> dt.date:
    from zr.data.calendar import is_trading_day, session_close

    now = pd.Timestamp.now(tz=ET)
    d = now.date()
    if not (is_trading_day(d) and now.time() >= session_close(d)):
        d -= dt.timedelta(days=1)
    while not is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


def run_all(cl: Client, st: Store) -> None:
    pr = Progress(st, cl)
    syms = build_universe(st)["symbol"].tolist()
    pr("名单", universe=f"{len(syms)} 只（纳斯达克目录日期见 universe/meta.json）")
    pull_daily(cl, st, syms + ["SPY"], progress=pr)  # SPY 只作 alpha / beta 基准，不进合格集合
    E = eligibility(st)
    E = E[E["symbol"].isin(set(syms))]
    last = last_complete_day()
    E = E[E["date"] <= last]
    elig = E[(E["date"] >= INTRA_FROM) & E["eligible"]]
    order = (
        elig.assign(tv=E["turnover"])
        .groupby("symbol")["tv"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    (st.r / "derived" / "order5.json").write_text(json.dumps(order))
    pr("合格集合", eligible_symbols=f"{len(order)} 只至少有一天合格（{INTRA_FROM}～{last}）")
    done5 = st.done("bars5_done")
    t5 = time.time()
    n5 = 0
    for i, s in enumerate(order):
        if s not in done5:
            try:
                pull_bars5(cl, st, s)
            except NotFound as e:
                st.fail("bars5", s, str(e))
            except Exception as e:  # noqa: BLE001
                st.fail("bars5", s, f"{type(e).__name__}: {e}")
                continue
            st.mark("bars5_done", [s])
            n5 += 1
        if (i + 1) % 25 == 0 or i + 1 == len(order):
            eta = (time.time() - t5) / max(n5, 1) * (len(order) - i - 1) / 3600
            pr(f"5 分钟 K {i + 1}/{len(order)}", eta5=f"5 分钟 K 剩余约 {eta:.1f} 小时")
        if i + 1 == min(PRELIM_N, len(order)):
            sel = selection_1m(E, opening_ranges(st, order[: i + 1]), last)
            pr("初步：前 500 只的 RelVol 前 50", prelim_pairs=f"{len(sel)} 个标的日")
            pull_bars1(cl, st, sel, progress=pr)
            (st.r / "derived" / "PRELIM_READY").write_text(dt.datetime.now(dt.UTC).isoformat())
            pr("初步数据就绪（前 500 只）")
    sel = selection_1m(E, opening_ranges(st, order), last)
    write_atomic(sel, st.r / "derived" / "selection_top50.parquet")
    pr("正式：全部合格标的的 RelVol 前 50", final_pairs=f"{len(sel)} 个标的日")
    pull_bars1(cl, st, sel, progress=pr)
    (st.r / "derived" / "FINAL_READY").write_text(dt.datetime.now(dt.UTC).isoformat())
    pr("全部完成")


def run_test(cl: Client, st: Store, n: int = 30) -> dict:
    """实测：成交额靠前与随机各半的 30 只，日线 2 次 + 5 分钟 K 全历史 + 每只 3 个交易日的 1 分钟 K；记录耗时与返回格式。"""
    from zr.data.calendar import trading_days

    u = build_universe(st)
    big = ["AAPL", "NVDA", "TSLA", "AMD", "MU", "BRK.B", "PLTR", "SMCI", "COIN", "MSTR"]
    rest = u[~u["symbol"].isin(big)]["symbol"].sample(n - len(big), random_state=0).tolist()
    syms = [s for s in big if s in set(u["symbol"])] + rest
    out: dict[str, Any] = {"symbols": syms}
    t = time.time()
    n0 = cl.n
    pull_daily(cl, st, syms, batch=len(syms))
    out["daily"] = {"sec": time.time() - t, "requests": cl.n - n0}
    d = load_daily(st)
    one = d[(d["symbol"] == "AAPL") & (d["adj"] == "none")].sort_values("ts").tail(3)
    out["daily_ts_example"] = [str(x) for x in one["ts"]]
    out["daily_rows"] = d.groupby(["symbol", "adj"]).size().describe().round(1).to_dict()
    t, n0 = time.time(), cl.n
    pages = {}
    for s in syms:
        try:
            pages[s] = pull_bars5(cl, st, s)
        except Exception as e:  # noqa: BLE001
            pages[s] = f"{type(e).__name__}: {str(e)[:100]}"
    out["bars5"] = {"sec": time.time() - t, "requests": cl.n - n0, "pages": pages}
    b = pd.read_parquet(st.r / "bars5" / "AAPL.parquet")
    tt = b["ts"].dt.tz_convert(ET)
    out["bars5_aapl"] = {
        "rows": len(b),
        "first": str(tt.min()),
        "last": str(tt.max()),
        "sessions": b["session"].value_counts().to_dict(),
        "minutes": sorted({x.strftime("%H:%M") for x in tt})[:3]
        + sorted({x.strftime("%H:%M") for x in tt})[-3:],
    }
    days = trading_days(dt.date(2024, 3, 4), dt.date(2024, 3, 6))
    pairs = pd.DataFrame([{"symbol": s, "date": dday} for s in syms[:10] for dday in days])
    t, n0 = time.time(), cl.n
    pull_bars1(cl, st, pairs)
    out["bars1"] = {"sec": time.time() - t, "requests": cl.n - n0, "pairs": len(pairs)}
    fs = sorted((st.r / "bars1").glob("day=2024-03-04/*.parquet"))
    if fs:
        x = pd.concat([pd.read_parquet(f) for f in fs])
        a = x[x["symbol"] == "AAPL"]["ts"].dt.tz_convert(ET)
        out["bars1_aapl_2024_03_04"] = {
            "rows": int(len(a)),
            "first": str(a.min()),
            "last": str(a.max()),
        }
    out["errors"] = dict(cl.err)
    out["failed"] = (
        (st.r / "state" / "failed.tsv").read_text()[-2000:]
        if (st.r / "state" / "failed.tsv").exists()
        else ""
    )
    out["universe"] = json.loads((st.r / "universe" / "meta.json").read_text())
    (st.r / "test_report.json").write_text(
        json.dumps(out, ensure_ascii=False, default=str, indent=1)
    )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("test", "all"), required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args(argv)
    from zr.broker.longport_client import make_quote_ctx
    from zr.config import get_settings

    st = Store(get_settings().zr_data_root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(st.r / "logs" / f"pull_{a.phase}.log"),
            logging.StreamHandler(),
        ],
    )
    cl = Client(make_quote_ctx())
    if a.phase == "test":  # 实测写到 sip_test/，不混入正式数据
        ts_ = Store(get_settings().zr_data_root, "sip_test")
        for f in ("nasdaqlisted.txt", "otherlisted.txt"):
            shutil.copy2(st.r / "universe" / f, ts_.r / "universe" / f)
        print(json.dumps(run_test(cl, ts_, a.n), ensure_ascii=False, default=str, indent=1))
    else:
        run_all(cl, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
