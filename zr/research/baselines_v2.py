"""四条基线（METHOD_V2 规格 8，R-025）。

与规则回放、模型同一套合约与出场规则（zr/research/outcomes.py）。

1. 等权：全部检测器（events_v2）的全部响铃，按 (日, 决策分钟, 方向) 去重后入场
2. 随机入场：每天与等权同笔数，决策分钟与方向随机，1,000 次 → 分布
3. 日历入场：定时发布（config/calendar.yaml，RTH 内只有 FOMC 14:00 与个别 10:00 的 PCE）后 ≤1
  分钟决策，
   call、put 各一版
4. 跨式：同一分钟同时买 call 与 put（在等权的事件分钟、以及日历分钟），盈亏 = 两腿之和
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.features.calendar_features import release_times
from zr.research import outcomes as oc
from zr.research.replay_rule import Lookup, stats

ET = oc.ET


def equal_weight(ds: pd.DataFrame, lk: Lookup, cap: int, months: set | None = None) -> pd.DataFrame:
    e = ds[["day", "bar_minute", "direction"]].copy()
    e["minute"] = e["bar_minute"] + 1
    e = e[(e["minute"] >= oc.FIRST_ENTRY) & (e["minute"] <= oc.LAST_ENTRY)].drop_duplicates(
        ["day", "minute", "direction"]
    )
    if months is not None:
        e = e[pd.to_datetime(e["day"]).dt.to_period("M").astype(str).isin(months)]
    parts = []
    for d in (1, -1):
        x = e[e["direction"] == d]
        parts.append(x.assign(net=lk.get(x["day"], x["minute"], d, cap)))
    return pd.concat(parts, ignore_index=True).dropna(subset=["net"])


def random_entry(
    lk: Lookup, counts: pd.Series, cap: int, n: int = 1000, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    totals = np.zeros(n)
    for day, k in counts.items():
        i = lk.day_ix.get(day)
        if i is None:
            continue
        nets = np.stack([lk.net[(1, cap)][i], lk.net[(-1, cap)][i]])  # 2 × 391
        cand = np.flatnonzero(np.isfinite(nets).all(axis=0))
        cand = cand[(cand >= oc.FIRST_ENTRY) & (cand <= oc.LAST_ENTRY)]
        if len(cand) == 0:
            continue
        mins = cand[rng.integers(0, len(cand), size=(n, int(k)))]
        dirs = rng.integers(0, 2, size=(n, int(k)))
        totals += nets[dirs, mins].sum(axis=1)
    return totals


def calendar_minutes(lk: Lookup) -> pd.DataFrame:
    rel = release_times().tz_convert(ET)
    rows = []
    for t in rel:
        d = t.date()
        if d not in lk.day_ix:
            continue
        base = (t.hour * 60 + t.minute) - 570  # 发布时刻的分钟索引
        for m in (base, base + 1):  # 决策分钟 = 发布当分钟或下一分钟（≤1 分钟）
            if oc.FIRST_ENTRY <= m <= oc.LAST_ENTRY:
                rows.append({"day": d, "minute": m, "release": t.strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows)


def straddle(lk: Lookup, entries: pd.DataFrame, cap: int) -> pd.DataFrame:
    e = entries.drop_duplicates(["day", "minute"]).copy()
    c = lk.get(e["day"], e["minute"], 1, cap)
    p = lk.get(e["day"], e["minute"], -1, cap)
    return e.assign(net_call=c, net_put=p, net=c + p).dropna(subset=["net"])


def baseline_table(
    ds: pd.DataFrame, lk: Lookup, cap: int, n_random: int = 1000, months: set | None = None
) -> tuple[pd.DataFrame, dict]:
    ew = equal_weight(ds, lk, cap, months)
    counts = ew.groupby("day").size()
    rnd = random_entry(lk, counts, cap, n_random)
    cal = calendar_minutes(lk)
    if months is not None and len(cal):
        cal = cal[pd.to_datetime(cal["day"]).dt.to_period("M").astype(str).isin(months)]
    rows = [{"baseline": "等权（全部响铃，去重）", **stats(ew["net"], ew["day"])}]
    rows.append(
        {
            "baseline": f"随机入场（同日同笔数，{n_random} 次）均值",
            "n": int(counts.sum()),
            "sum_net": round(float(rnd.mean()), 2),
            "mean_net": round(float(rnd.mean() / max(1, counts.sum())), 2),
            "rand_p05": round(float(np.percentile(rnd, 5)), 1),
            "rand_p95": round(float(np.percentile(rnd, 95)), 1),
            "ew_pctile_in_random": round(float((rnd <= ew["net"].sum()).mean() * 100), 1),
        }
    )
    for d, nm in ((1, "call"), (-1, "put")):
        if len(cal):
            x = cal.assign(net=lk.get(cal["day"], cal["minute"], d, cap)).dropna(subset=["net"])
            rows.append(
                {"baseline": f"日历入场（发布后 ≤1 分钟）{nm}", **stats(x["net"], x["day"])}
            )
    st_ev = straddle(lk, ew.assign(minute=ew["minute"]), cap)
    rows.append({"baseline": "跨式（等权事件分钟，call+put）", **stats(st_ev["net"], st_ev["day"])})
    if len(cal):
        st_cal = straddle(lk, cal, cap)
        rows.append({"baseline": "跨式（日历分钟）", **stats(st_cal["net"], st_cal["day"])})
    return pd.DataFrame(rows), {"equal_weight": ew, "random_totals": rnd, "calendar": cal}
