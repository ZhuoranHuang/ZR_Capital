"""逐分钟真实期权交易结果表（METHOD_V2 规格 2/7/8 共用）。

对每个有期权数据的交易日、每个决策分钟（09:31–15:25 ET）、每个方向（+1 买 call / −1 买 put）、
每个持仓上限（10 / 30 分钟），按负责人规则模拟一笔交易，结果落盘。规则回放、随机对照、
四条基线、模型决策模拟都只是在这张表上按入场分钟查表，保证同一套合约与出场规则。

合约（负责人 2026-10-22 补充 2）：最近的可用到期日（DTE ≤ 5，记录实际 DTE）；在该到期日上取
最近一档略虚值（call 行权价 > 现价、put 行权价 < 现价），且决策分钟必须有有效 ask，否则顺延到下一个到期日。
现价 = 决策分钟前一根（触发 bar）的收盘价。
入场：决策分钟 ask，2 张。出场：bid ≥ 入场 ask × 1.2 时卖 1 张（减半）；剩余在正股收盘跌破布林中轨
（多头；空头对称）时卖出；持仓上限到点卖出；15:55 ET 强制平仓；全部卖 bid。费用 D-013。
时间对齐：Theta 1m 时间戳 t 的报价 = t 时刻 NBBO；决策分钟 m 用时间戳 m 的 ask。
D-001 标记：DTE 0 且决策时刻晚于 10:15 ET 的记一个 d001_violation（不排除，分层报告）。

运行：uv run python -m zr.research.outcomes --symbol MU.US → ZR_DATA_ROOT/research/v2/outcomes_MU.parquet
（按日缓存，已算过的日期跳过；新拉到的期权日会被补算）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.features.bars import load_bars
from zr.features.features import compute_features
from zr.features.option_features import load_option_day, option_days

log = logging.getLogger("zr.research.outcomes")
ET = ZoneInfo("America/New_York")
MINUTES = 391  # 09:30 .. 16:00
FIRST_ENTRY = 1  # 09:31
LAST_ENTRY = 355  # 15:25
FORCE_EXIT = 385  # 15:55
QTY = 2
TAKE_PROFIT = 0.20
CAPS = (10, 30)
MAX_DTE = 5
D001_LAST_0DTE_ENTRY = 45  # 10:15


@lru_cache(maxsize=200_000)
def _fee(qty: int, side: str, prem_cents: int) -> float:
    return float(option_order_fees(qty, side, prem_cents / 100)["total"])


def fee(qty: int, side: str, prem: float) -> float:
    return _fee(qty, side, int(round(prem * 100)))


def minute_index(ts) -> np.ndarray:
    et = pd.DatetimeIndex(ts).tz_convert(ET)
    return (et.hour * 60 + et.minute - 570).to_numpy()


@dataclass
class Day:
    day: dt.date
    close: np.ndarray
    mid: np.ndarray
    expiries: list[dt.date]
    chains: dict[
        tuple[dt.date, str], pd.DataFrame
    ]  # (expiry, right) -> strike, symbol（按 strike 升序）
    bid: dict[str, np.ndarray]
    ask: dict[str, np.ndarray]


def build_day(stock: pd.DataFrame, mid: pd.Series, opt: dict, day: dt.date) -> Day | None:
    idx = minute_index(stock.index)
    ok = (idx >= 0) & (idx < MINUTES)
    close = np.full(MINUTES, np.nan)
    midv = np.full(MINUTES, np.nan)
    close[idx[ok]] = stock["close"].to_numpy()[ok]
    midv[idx[ok]] = mid.to_numpy()[ok]
    q = opt["quotes"]
    if q.empty:
        return None
    q = q[(q["bid"] > 0) & (q["ask"] > 0) & (q["ask"] >= q["bid"])]
    qi = minute_index(q["ts"])
    keep = (qi >= 0) & (qi < MINUTES)
    q = q.loc[keep]
    qi = qi[keep]
    bid: dict[str, np.ndarray] = {}
    ask: dict[str, np.ndarray] = {}
    sym = q["symbol"].to_numpy()
    b_all, a_all = q["bid"].to_numpy(float), q["ask"].to_numpy(float)
    order = np.argsort(sym, kind="stable")
    sym_s, qi_s, b_s, a_s = sym[order], qi[order], b_all[order], a_all[order]
    bounds = np.flatnonzero(np.r_[True, sym_s[1:] != sym_s[:-1], True])
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        bb = np.full(MINUTES, np.nan)
        aa = np.full(MINUTES, np.nan)
        bb[qi_s[lo:hi]] = b_s[lo:hi]
        aa[qi_s[lo:hi]] = a_s[lo:hi]
        bid[sym_s[lo]] = bb
        ask[sym_s[lo]] = aa
    con = q[["symbol", "expiry", "strike", "right"]].drop_duplicates("symbol")
    expiries = sorted(e for e in con["expiry"].unique() if 0 <= (e - day).days <= MAX_DTE)
    chains = {
        (e, r): g.sort_values("strike")[["strike", "symbol"]].reset_index(drop=True)
        for (e, r), g in con.groupby(["expiry", "right"])
        if e in expiries
    }
    return Day(day, close, midv, expiries, chains, bid, ask)


def pick(d: Day, direction: int, spot: float, minute: int) -> tuple[str, dt.date] | None:
    """最近的可用到期日上的最近一档略虚值，且决策分钟有 ask；否则顺延。"""
    right = "call" if direction == 1 else "put"
    for e in d.expiries:
        ch = d.chains.get((e, right))
        if ch is None or ch.empty:
            continue
        s = ch["strike"].to_numpy()
        if direction == 1:
            j = np.searchsorted(s, spot, side="right")  # 第一个 > spot
            if j >= len(s):
                continue
        else:
            j = np.searchsorted(s, spot, side="left") - 1  # 最后一个 < spot
            if j < 0:
                continue
        sym = ch["symbol"].iloc[j]
        a = d.ask.get(sym)
        if a is not None and np.isfinite(a[minute]):
            return sym, e
    return None


def simulate(d: Day, entry: int, direction: int, sym: str, cap: int) -> dict | None:
    bid, ask = d.bid[sym], d.ask[sym]
    ask0 = float(ask[entry])
    last = min(entry + cap, FORCE_EXIT)
    reason = "cap" if entry + cap <= FORCE_EXIT else "eod"
    exit_idx = last
    proceeds = 0.0
    fees = fee(QTY, "Buy", ask0)
    remaining = QTY
    half = False
    for j in range(entry + 1, last + 1):
        b = bid[j]
        if np.isfinite(b) and not half and b >= ask0 * (1 + TAKE_PROFIT):
            proceeds += b * 100
            fees += fee(1, "Sell", b)
            remaining -= 1
            half = True
        c, m = d.close[j], d.mid[j]
        if (
            np.isfinite(c)
            and np.isfinite(m)
            and ((direction == 1 and c < m) or (direction == -1 and c > m))
        ):
            exit_idx, reason = j, "struct_stop"
            break
    j = exit_idx
    while j > entry and not np.isfinite(bid[j]):
        j -= 1
    b = bid[j]
    if not np.isfinite(b):
        return None
    if remaining:
        proceeds += b * 100 * remaining
        fees += fee(remaining, "Sell", b)
    gross = proceeds - ask0 * 100 * QTY
    return {
        "exit": j,
        "hold": j - entry,
        "ask0": ask0,
        "gross": gross,
        "fees": fees,
        "net": gross - fees,
        "reason": reason,
        "half_tp": half,
    }


def day_outcomes(d: Day) -> pd.DataFrame:
    rows = []
    for m in range(FIRST_ENTRY, LAST_ENTRY + 1):
        spot = d.close[m - 1]
        if not np.isfinite(spot):
            continue
        for direction in (1, -1):
            pk = pick(d, direction, spot, m)
            if pk is None:
                continue
            sym, exp = pk
            dte = (exp - d.day).days
            for cap in CAPS:
                r = simulate(d, m, direction, sym, cap)
                if r is None:
                    continue
                r.update(
                    {
                        "day": d.day,
                        "minute": m,
                        "direction": direction,
                        "cap": cap,
                        "symbol": sym,
                        "expiry": exp,
                        "dte": dte,
                        "spot": spot,
                        "d001_violation": bool(dte == 0 and m > D001_LAST_0DTE_ENTRY),
                    }
                )
                rows.append(r)
    return pd.DataFrame(rows)


def build(root: Path, und: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    store = root / "research" / "v2" / "outcomes" / f"underlying={und}"
    store.mkdir(parents=True, exist_ok=True)
    days = [d for d in option_days(root, und) if start <= d <= end]
    todo = [d for d in days if not (store / f"{d.isoformat()}.parquet").exists()]
    if todo:
        stock = load_bars(root, und)
        mid = compute_features(stock)["bb_mid"]
        sday = pd.Series(stock.index.tz_convert(ET).date, index=stock.index)
        t0 = time.monotonic()
        for i, day in enumerate(todo):
            opt = load_option_day(root, und, day)
            m = (sday == day).to_numpy()
            if opt is None or not m.any():
                pd.DataFrame().to_parquet(store / f"{day.isoformat()}.parquet")
                continue
            dd = build_day(stock[m], mid[m], opt, day)
            out = day_outcomes(dd) if dd is not None else pd.DataFrame()
            out.to_parquet(store / f"{day.isoformat()}.parquet", index=False)
            if (i + 1) % 20 == 0:
                log.info(
                    "%s outcomes %d/%d 天，%.0fs", und, i + 1, len(todo), time.monotonic() - t0
                )
    parts = [pd.read_parquet(store / f"{d.isoformat()}.parquet") for d in days]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MU.US")
    p.add_argument("--start", default="2023-10-02")
    p.add_argument("--end", default="2026-10-21")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    out = build(
        get_settings().zr_data_root,
        a.symbol.split(".")[0],
        dt.date.fromisoformat(a.start),
        dt.date.fromisoformat(a.end),
    )
    log.info(
        "outcomes 行 %d，天 %d，%.0fs",
        len(out),
        out["day"].nunique() if len(out) else 0,
        time.monotonic() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
