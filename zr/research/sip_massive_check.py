# ruff: noqa: E501  （中文说明较长）
"""R-057 独立核对：随机抽若干交易日，直接从 Mac 归档的原始文件（massive_raw）用另写的逐行实现重算
合格条件（Wilder ATR 递推、14 日均量、开盘价）、开盘区间（09:30–09:34 五根 1 分钟 K）、RelVol、方向与前 20 名，
与流水线（sip_stream pass1 → sip_select）的 candidates.parquet 比对。不复用 sip_orb / sip_stream 的计算函数
（只复用读文件与交易日历）。拆股剔除与标的池直接取流水线结果（它们来自接口，不是计算）。
运行：python -m zr.research.sip_massive_check [--days 10]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from zr.data import massive as mv
from zr.data.massive_archive import raw_root
from zr.data.sip_massive import read_aggs
from zr.research.sip_massive_report import load_C
from zr.research.sip_select import mroot


def or_from_minute(path: Path, tickers: set, day: dt.date) -> dict:
    """09:30 ≤ 分钟 < 09:35 的 K 按时间顺序聚合。分钟数用当日美东相对 UTC 的偏移（zoneinfo，当日正午）做整数运算，
    不用流水线的 pandas 时区转换。"""
    from zoneinfo import ZoneInfo

    off = (
        dt.datetime(day.year, day.month, day.day, 12, tzinfo=ZoneInfo("America/New_York"))
        .utcoffset()
        .total_seconds()
    )
    m = read_aggs(path)
    m = m[m["ticker"].isin(tickers)]
    mins = ((m["window_start"].to_numpy() // 1_000_000_000 + int(off)) // 60) % 1440
    m = m[(mins >= 570) & (mins < 575)]
    acc: dict = {}
    for t, ws, o, h, lo, c, v in zip(
        m["ticker"],
        m["window_start"],
        m["open"],
        m["high"],
        m["low"],
        m["close"],
        m["volume"],
        strict=True,
    ):
        acc.setdefault(t, []).append((int(ws), o, h, lo, c, float(v)))
    out = {}
    for t, a in acc.items():
        a.sort()
        out[t] = {
            "open": a[0][1],
            "high": max(x[2] for x in a),
            "low": min(x[3] for x in a),
            "close": a[-1][4],
            "vol": sum(x[5] for x in a),
        }
    return out


def run(n_days: int, seed: int = 7) -> dict:
    from zr.config import get_settings

    root = get_settings().zr_data_root
    r = mroot(root)
    C = load_C(r)
    days_all = sorted(C["date"].unique())
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(np.array(days_all[30:], dtype=object), size=n_days, replace=False))
    last = pick[-1]
    # 日聚合：逐日递推（Wilder ATR、14 日量、开盘），全部代码
    trade_days = [
        d
        for d in sorted({dt.date.fromisoformat(p.stem) for p in (r / "daily").glob("*/*.parquet")})
        if d <= last
    ]
    atr: dict = {}
    n_tr: dict = defaultdict(int)
    prev_close: dict = {}
    vols: dict = defaultdict(lambda: deque(maxlen=14))
    state_at: dict = {}
    pick_set = set(pick)
    for d in trade_days:
        dd = read_aggs(raw_root(root) / mv.day_key(d, mv.DAY_PREFIX))
        if d in pick_set:  # 当日开盘前已知的状态
            state_at[d] = {
                t: (
                    atr.get(t) if n_tr[t] >= 14 else None,
                    (sum(vols[t]) / 14 if len(vols[t]) == 14 else None),
                )
                for t in set(dd["ticker"])
            }
            state_at[d]["__open__"] = dict(zip(dd["ticker"], dd["open"], strict=True))
        for t, h, lo, c, v in zip(
            dd["ticker"], dd["high"], dd["low"], dd["close"], dd["volume"], strict=True
        ):
            pc = prev_close.get(t)
            tr = h - lo if pc is None else max(h - lo, abs(h - pc), abs(lo - pc))
            atr[t] = tr if t not in atr else atr[t] + (tr - atr[t]) / 14
            n_tr[t] += 1
            prev_close[t] = c
            vols[t].append(float(v))
    # 开盘区间历史：只需要被抽查日之前的 ORVol（前 14 个有开盘区间的交易日）
    res = {}
    for d in pick:
        cd = C[C["date"] == d]
        pool = set(cd["symbol"])
        k = trade_days.index(d)
        hist_days = trade_days[max(0, k - 40) : k]
        orv: dict = defaultdict(list)
        for hd in hist_days:
            for t, x in or_from_minute(
                raw_root(root) / mv.day_key(hd, mv.MINUTE_PREFIX), pool, hd
            ).items():
                orv[t].append(x["vol"])
        today = or_from_minute(raw_root(root) / mv.day_key(d, mv.MINUTE_PREFIX), pool, d)
        st = state_at[d]
        rows = []
        for t in pool:
            if t not in today:
                continue
            a, v14 = st.get(t, (None, None))
            op = st["__open__"].get(t)
            blocked = bool(cd.loc[cd["symbol"] == t, "split_block"].iloc[0])
            elig = (
                op is not None
                and op > 5
                and v14 is not None
                and v14 >= 1e6
                and a is not None
                and a > 0.5
                and not blocked
            )
            hv = orv[t][-14:]
            rv = today[t]["vol"] / (sum(hv) / 14) if len(hv) == 14 and sum(hv) > 0 else math.nan
            dr = int(np.sign(round(today[t]["close"] - today[t]["open"], 6)))
            rows.append((t, elig, rv, dr))
        X = pd.DataFrame(rows, columns=["symbol", "eligible", "relvol", "dir"])
        sel = (
            X[X["eligible"] & (X["relvol"] >= 1) & (X["dir"] != 0)]
            .sort_values(["relvol", "symbol"], ascending=[False, True], kind="mergesort")
            .head(20)
        )
        pipe = cd[(cd["rank"] > 0) & (cd["rank"] <= 20)].sort_values("rank")
        m = X.merge(
            cd[["symbol", "eligible", "relvol", "dir"]], on="symbol", suffixes=("_ind", "_pipe")
        )
        res[str(d)] = {
            "pool": len(pool),
            "eligible_agree": float((m["eligible_ind"] == m["eligible_pipe"]).mean()),
            "relvol_max_abs_diff": float(np.nanmax(np.abs(m["relvol_ind"] - m["relvol_pipe"])))
            if m["relvol_ind"].notna().any()
            else None,
            "dir_agree": float((m["dir_ind"] == m["dir_pipe"]).mean()),
            "top20_same_order": list(sel["symbol"]) == list(pipe["symbol"]),
            "top20_overlap": len(set(sel["symbol"]) & set(pipe["symbol"])),
        }
    return res


def sim_independent(
    bars: pd.DataFrame, d: int, orh: float, orl: float, atr: float, slip: float, conservative: bool
):
    """逐根循环重写的执行规则（不调用 sip_orb.sim_core）：09:35 起首根触及区间边界即入场，成交 = 边界与开盘价中较不利者 ± 滑点；
    止损 = 入场 ∓ 0.10 × ATR；保守 = 入场那根的低（高）点触及止损即按止损价出场；之后触及按止损与开盘价中较不利者出场；否则按末根收盘。"""
    rows = list(bars.itertuples(index=False))
    for i, b in enumerate(rows):
        if b.m < 575:
            continue
        hit = b.high >= orh if d > 0 else b.low <= orl
        if not hit:
            continue
        entry = (max(orh, b.open) + slip) if d > 0 else (min(orl, b.open) - slip)
        stop = entry - 0.10 * atr if d > 0 else entry + 0.10 * atr
        if conservative and ((d > 0 and b.low <= stop) or (d < 0 and b.high >= stop)):
            return entry, (stop - slip) if d > 0 else (stop + slip)
        for b2 in rows[i + 1 :]:
            if d > 0 and b2.low <= stop:
                return entry, min(stop, b2.open) - slip
            if d < 0 and b2.high >= stop:
                return entry, max(stop, b2.open) + slip
        last = rows[-1].close
        return entry, (last - slip) if d > 0 else (last + slip)
    return None


def check_trades(n: int = 300, seed: int = 11) -> dict:
    from zr.research.sip_massive_report import bars_loader, run

    R = run()
    load = bars_loader(R["r"])
    out = {}
    for cfg in (("cons", 0.01), ("opt", 0.0)):
        T = R["T1"][cfg]
        T = T[T["flip"] == 0].sample(n=min(n, len(T)), random_state=seed).sort_values("date")
        bad = 0
        for r in T.itertuples():
            b = load(r.date)
            x = b[b["symbol"] == r.symbol].sort_values("ts")
            et = x["ts"].dt.tz_convert("America/New_York")
            x = x.assign(m=et.dt.hour * 60 + et.dt.minute)
            res = sim_independent(
                x, int(r.dir), r.or_high, r.or_low, r.atr_prev, cfg[1], cfg[0] == "cons"
            )
            if res is None or abs(res[0] - r.entry) > 1e-9 or abs(res[1] - r.exit) > 1e-9:
                bad += 1
        out[f"{cfg[0]} {cfg[1]}"] = {"checked": len(T), "mismatch": bad}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--trades", action="store_true", help="只做执行层核对（随机 300 笔）")
    a = ap.parse_args(argv)
    if a.trades:
        res = check_trades()
        out = mroot() / "derived" / "independent_trades_check.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False))
        return 0
    res = run(a.days)
    out = mroot() / "derived" / "independent_check.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
