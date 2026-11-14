# ruff: noqa: E501  （中文名称较长）
"""R-057 Massive 版回测（Mac）：读 sip_select 的 derived/、VPS pass2 的 bars1/ 与 cand_out/，按预登记（含 2026-11-14 修订与补充）
计算样本内（论文样本内复现检查，至 2023-12-31）与样本外（2024-01-01 起，通过标准）两段的全部结果。

执行一律 1 分钟 K（zr.research.sip_orb.sim_core）：
- SIP 前 N：每日 RelVol 前 50 的 1 分钟 K（bars1/），四档滑点 × 保守 / 乐观 × 自然 / 反向方向（安慰剂 (ii)）。
- 基础版与安慰剂 (i)：VPS 上逐日算好的成交结果（cand_out/，保守 0.01 与乐观 0 两种设定，同一 sim_core）。
写出见 zr.research.sip_massive_write。运行：python -m zr.research.sip_massive_write
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from zr.research import sip_orb as so
from zr.research.sip_orb_report import (
    ALGOS,
    MAIN,
    N_DRAW,
    RMB,
    SLIPS,
    day_groups,
    metrics,
    pick,
    run_sims,
)
from zr.research.sip_select import mroot

IS_END = dt.date(2023, 12, 31)
OOS_START = dt.date(2024, 1, 1)
SEGMENTS = ("样本内", "样本外")
PAPER_START = dt.date(2016, 1, 1)


def year_periods(first: dt.date, last: dt.date) -> list[tuple[str, dt.date, dt.date]]:
    out = [(str(y), dt.date(y, 1, 1), dt.date(y, 12, 31)) for y in range(first.year, last.year + 1)]
    return out + [("2026-06 以后", dt.date(2026, 6, 1), dt.date(2100, 1, 1))]


# ---------------- 数据 ----------------


def load_C(r: Path) -> pd.DataFrame:
    C = pd.read_parquet(r / "derived" / "candidates.parquet")
    C["date"] = pd.to_datetime(C["date"]).dt.date
    return C


def trading_days(r: Path) -> list[dt.date]:
    return sorted(dt.date.fromisoformat(p.stem) for p in (r / "daily").glob("*/*.parquet"))


def spy_returns(r: Path) -> pd.Series:
    rows = []
    for p in sorted((r / "daily").glob("*/*.parquet")):
        x = pd.read_parquet(p, columns=["ticker", "d_close"])
        v = x.loc[x["ticker"] == "SPY", "d_close"]
        if len(v):
            rows.append((dt.date.fromisoformat(p.stem), float(v.iloc[0])))
    s = pd.Series(dict(rows)).sort_index()
    return s.pct_change()


def bars_loader(r: Path):
    cache: dict = {}

    def load_day(day: dt.date):
        if day not in cache:
            cache.clear()
            p = r / "bars1" / f"{day.year}" / f"{day}.parquet"
            cache[day] = pd.read_parquet(p) if p.exists() else None
        return cache[day]

    return load_day


def sims_top50(C: pd.DataFrame, r: Path) -> dict:
    top = C[C["rank"] > 0].copy()
    load = bars_loader(r)
    cfgs = [(a, s_) for a in ALGOS for s_ in SLIPS]
    res = {cfg: [] for cfg in cfgs}
    for day, g in top.groupby("date", sort=True):
        b = load(day)
        if b is None:
            continue
        groups = {s: day_groups(x.sort_values("ts")) for s, x in b.groupby("symbol")}
        rs = run_sims(lambda s, groups=groups: groups.get(s), g, cfgs, both_dirs=True)
        for cfg in cfgs:
            res[cfg].append(rs[cfg])
    return {cfg: pd.concat(v, ignore_index=True) for cfg, v in res.items()}


def base_outcomes(C: pd.DataFrame, r: Path) -> dict:
    parts = [pd.read_parquet(p) for p in sorted((r / "cand_out").glob("*/*.parquet"))]
    oc = pd.concat(parts, ignore_index=True)
    oc["date"] = pd.to_datetime(oc["date"]).dt.date
    keep = ["date", "symbol", "or_high", "or_low", "atr_prev", "relvol", "rank", "eligible"]
    out = {}
    for (algo, slip), g in oc.groupby(["algo", "slip"]):
        x = g.merge(C[keep], on=["date", "symbol"], how="left")
        x["flip"] = 0
        x["atr"] = x["atr_prev"]
        out[(algo, float(slip))] = x.reset_index(drop=True)
    return out


def run() -> dict:
    r = mroot()
    C = load_C(r)
    days_all = trading_days(r)
    top = C[C["rank"] > 0]
    start = min(top["date"])
    days = [d for d in days_all if d >= start]
    T1 = sims_top50(C, r)
    TB = base_outcomes(C, r)
    nbase = C[C["base"]].groupby("date").size().to_dict()
    return {
        "r": r,
        "C": C,
        "T1": T1,
        "TB": TB,
        "nbase": nbase,
        "spy": spy_returns(r),
        "days": days,
        "start": start,
        "last": days[-1],
        "seg_days": {
            "样本内": [d for d in days if d <= IS_END],
            "样本外": [d for d in days if d >= OOS_START],
        },
    }


# ---------------- 组合与指标 ----------------


def seg(T: pd.DataFrame, days: list) -> pd.DataFrame:
    return T[(T["date"] >= days[0]) & (T["date"] <= days[-1])]


def go(R: dict, segname: str, T: pd.DataFrame, eq0: float, n, lev: float, fee: str) -> dict:
    days = R["seg_days"][segname]
    p = so.portfolio(seg(T, days), eq0, n, lev, fee)
    return {"p": p, "m": metrics(p["daily"], days, eq0, p["trades"], R["spy"]), "eq0": eq0}


MAIN_NAME = "主口径（SIP 前 20，保守，D-013，滑点 0.01，10 万美元，4 倍）"
PAPER_SIP = "原文口径 SIP 前 20（乐观，每股 0.0035，零滑点，2.5 万美元，4 倍）"
PAPER_BASE = "原文口径 基础版（全部合格且非十字星，乐观，每股 0.0035，零滑点，2.5 万美元，4 倍）"
MAIN_BASE = "基础版（主口径设定，10 万美元）"


def specs(R: dict) -> list[tuple[str, pd.DataFrame, float, object, float, str]]:
    T1, TB, nb = R["T1"], R["TB"], R["nbase"]
    mainT = pick(T1[(MAIN[0], MAIN[1])], 20)
    out = [
        (MAIN_NAME, mainT, 100_000, 20, 4.0, "d013"),
        (PAPER_SIP, pick(T1[("opt", 0.0)], 20), 25_000, 20, 4.0, "per_share"),
        (PAPER_BASE, TB[("opt", 0.0)].assign(rank=1), 25_000, nb, 4.0, "per_share"),
        (MAIN_BASE, TB[("cons", 0.01)].assign(rank=1), 100_000, nb, 4.0, "d013"),
        ("主口径，2.5 万美元", mainT, 25_000, 20, 4.0, "d013"),
        ("主口径，不加杠杆", mainT, 100_000, 20, 1.0, "d013"),
        (
            "1 万人民币，RelVol 第 1 名（不加杠杆，D-013）",
            pick(T1[(MAIN[0], MAIN[1])], 1),
            RMB,
            1,
            1.0,
            "d013",
        ),
        (
            "1 万人民币，RelVol 前 3 名（不加杠杆，D-013）",
            pick(T1[(MAIN[0], MAIN[1])], 3),
            RMB,
            3,
            1.0,
            "d013",
        ),
        ("敏感性：前 10", pick(T1[(MAIN[0], MAIN[1])], 10), 100_000, 10, 4.0, "d013"),
        ("敏感性：前 50", pick(T1[(MAIN[0], MAIN[1])], 50), 100_000, 50, 4.0, "d013"),
        ("敏感性：乐观算法", pick(T1[("opt", 0.01)], 20), 100_000, 20, 4.0, "d013"),
    ]
    for s_ in SLIPS:
        if s_ != 0.01:
            out.append(
                (f"敏感性：滑点 {s_:.2f}", pick(T1[("cons", s_)], 20), 100_000, 20, 4.0, "d013")
            )
    out += [
        ("敏感性：每股 0.0035 佣金", mainT, 100_000, 20, 4.0, "per_share"),
        (
            "敏感性：只做多",
            pick(T1[(MAIN[0], MAIN[1])], 20, long_only=True),
            100_000,
            20,
            4.0,
            "d013",
        ),
    ]
    return out


def experiments(R: dict) -> dict:
    return {
        sg: {name: go(R, sg, T, eq0, n, lev, fee) for name, T, eq0, n, lev, fee in specs(R)}
        for sg in SEGMENTS
    }


def placebos(R: dict, segname: str, seed: int = 0) -> dict:
    """固定权益 10 万美元的日收益均值（不复利）。(i) 合格 ∩ RelVol ≥ 1 中每日随机 20 只（1 分钟执行，保守 0.01，D-013）vs SIP 前 20；
    (ii) SIP 前 20 每只随机方向。"""
    days = R["seg_days"][segname]
    nd = len(days)
    rng = np.random.default_rng(seed)
    TB = seg(R["TB"][("cons", 0.01)], days)
    pool = TB[TB["relvol"] >= 1]
    p = so.portfolio(pool.assign(rank=1), 100_000, 20, 4.0, "d013", compound=False)["trades"]
    pnl_by_day = {d: g["pnl"].to_numpy() for d, g in p.groupby("date")}
    C = R["C"]
    cand = (
        C[(C["date"] >= days[0]) & (C["date"] <= days[-1]) & C["eligible"] & (C["relvol"] >= 1)]
        .groupby("date")
        .size()
        .to_dict()
    )
    sims = np.zeros(N_DRAW)
    for d, ncand in cand.items():
        v = pnl_by_day.get(d, np.array([]))
        full = np.r_[v, np.zeros(max(ncand - len(v), 0))]
        k = min(20, len(full))
        if k == 0:
            continue
        idx = rng.random((N_DRAW, len(full))).argsort(1)[:, :k]
        sims += full[idx].sum(1) / 100_000
    sims /= nd
    T1 = seg(R["T1"][(MAIN[0], MAIN[1])], days)
    t = T1[T1["rank"] <= 20]
    act_i = (
        so.portfolio(t[t["flip"] == 0], 100_000, 20, 4.0, "d013", compound=False)["daily"][
            "pnl"
        ].sum()
        / 100_000
        / nd
    )
    q = so.portfolio(
        t.assign(rank=t["rank"] * 2 + t["flip"]), 100_000, 20, 4.0, "d013", compound=False
    )["trades"]
    key = q["symbol"] + "|" + q["date"].astype(str)
    a = q[q["flip"] == 0].set_index(key[q["flip"] == 0])["pnl"]
    b = q[q["flip"] == 1].set_index(key[q["flip"] == 1])["pnl"]
    keys = sorted(set(a.index) | set(b.index))
    A, B = a.reindex(keys).fillna(0).to_numpy(), b.reindex(keys).fillna(0).to_numpy()
    flips = rng.random((N_DRAW, len(keys))) < 0.5
    sims2 = np.where(flips, B[None, :], A[None, :]).sum(1) / 100_000 / nd
    return {"i": (act_i, sims), "ii": (A.sum() / 100_000 / nd, sims2)}


def breakeven(R: dict, main: dict) -> pd.DataFrame:
    """保守、D-013、股数固定为主口径实际股数：每边滑点 0～0.10（步长 0.005）重算成交价，线性插值求每笔平均盈亏归零的滑点。"""
    tr = main["p"]["trades"]
    tr = tr[tr["shares"] > 0][
        ["symbol", "date", "shares", "atr_prev", "entry", "dir", "or_high", "or_low"]
    ].reset_index(drop=True)
    grid = np.round(np.arange(0, 0.1001, 0.005), 3)
    load = bars_loader(R["r"])
    pnl = np.zeros((len(tr), len(grid)))
    for day, g in tr.groupby("date"):
        b = load(day)
        groups = {s: day_groups(x.sort_values("ts")) for s, x in b.groupby("symbol")}
        for i in g.index:
            row = tr.loc[i]
            m, o, h, lo, c = groups[row["symbol"]][day]
            for k, s_ in enumerate(grid):
                res = so.sim_core(
                    m,
                    o,
                    h,
                    lo,
                    c,
                    int(row["dir"]),
                    row["or_high"],
                    row["or_low"],
                    row["atr_prev"],
                    s_,
                    True,
                )
                if res is None:
                    continue
                e, _, x, *_ = res
                sh = int(row["shares"])
                pnl[i, k] = sh * row["dir"] * (x - e) - so.trade_fees(
                    sh, e, x, int(row["dir"]), "d013"
                )
    tr["px_b"] = pd.cut(
        tr["entry"], [0, 10, 20, 50, 100, 1e9], labels=["5–10", "10–20", "20–50", "50–100", ">100"]
    )
    tr["atr_b"] = pd.cut(tr["atr_prev"], [0, 1, 2, 5, 1e9], labels=["0.5–1", "1–2", "2–5", ">5"])

    def solve(mask):
        if not mask.any():
            return np.nan, np.nan
        y = pnl[mask].mean(0)
        if y[0] <= 0:
            return 0.0, float(y[0])
        k = np.flatnonzero(y <= 0)
        if not len(k):
            return np.inf, float(y[0])
        k = k[0]
        return float(grid[k - 1] + (grid[k] - grid[k - 1]) * y[k - 1] / (y[k - 1] - y[k])), float(
            y[0]
        )

    rows = []
    for name, col in (("全部", None), ("按股价", "px_b"), ("按 ATR", "atr_b")):
        levels = [None] if col is None else list(tr[col].cat.categories)
        for lv in levels:
            mask = np.ones(len(tr), bool) if col is None else (tr[col] == lv).to_numpy()
            s_, y0 = solve(mask)
            Rm = float((so.STOP_ATR * tr.loc[mask, "atr_prev"]).median()) if mask.any() else np.nan
            rows.append(
                {
                    "分组": name if col is None else f"{name} {lv}",
                    "笔数": int(mask.sum()),
                    "零滑点每笔盈亏（美元）": y0,
                    "盈亏平衡滑点（美元 / 边）": s_,
                    "占 R 中位数": s_ / Rm if np.isfinite(s_) and Rm else np.nan,
                    "R 中位数（美元 / 股）": Rm,
                }
            )
    return pd.DataFrame(rows)
