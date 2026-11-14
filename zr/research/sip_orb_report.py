# ruff: noqa: E501  （中文报告行较长）
"""R-057 回测与报告：读 $ZR_DATA_ROOT/sip（VPS 拉取、经同步任务传回），按预登记跑主口径、原文口径、敏感性、安慰剂、
盈亏平衡滑点，写 docs/research/R057_SIP_ORB.md 与案例图。

运行：python -m zr.research.sip_orb_report [--prelim]   （--prelim = 成交额前 500 只的初步版，标注「初步，标的池不全」）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from zr.research import sip_orb as so

REPO = Path(__file__).resolve().parents[2]
ET = so.ET
SLIPS = (0.0, 0.01, 0.02, 0.05)
ALGOS = {"cons": True, "opt": False}
MAIN = ("cons", 0.01, "d013")
PAPER = ("opt", 0.0, "per_share")
FX = 6.6725
RMB = 10_000 / FX
N_DRAW = 1000
# 原文表 1、表 2（2016-01-01～2023-12-31，初始 25,000 美元，每股 0.0035 美元；SSRN 4729284，HSG 仓库 PDF，2026-11-14 读取）
PAPER_TAB = {
    "ORB Base（原文）": {
        "total": 0.29,
        "irr": 0.032,
        "vol": 0.066,
        "sharpe": 0.48,
        "hit": 0.414,
        "mdd": 0.13,
        "worst": -0.008,
        "alpha": 0.033,
        "beta": 0.01,
    },
    "ORB + Rel Vol（原文）": {
        "total": 16.37,
        "irr": 0.416,
        "vol": 0.148,
        "sharpe": 2.81,
        "hit": 0.484,
        "mdd": 0.12,
        "worst": -0.0161,
        "alpha": 0.358,
        "beta": 0.00,
    },
}
PERIODS = [
    ("2023Q4", dt.date(2023, 10, 1), dt.date(2023, 12, 31)),
    ("2024", dt.date(2024, 1, 1), dt.date(2024, 12, 31)),
    ("2025", dt.date(2025, 1, 1), dt.date(2025, 12, 31)),
    ("2026H1", dt.date(2026, 1, 1), dt.date(2026, 5, 31)),
    ("2026-06 以后", dt.date(2026, 6, 1), dt.date(2100, 1, 1)),
]


# ---------------- 数据 ----------------


def sip_root() -> Path:
    import os

    from zr.config import get_settings

    return Path(os.environ.get("SIP_ROOT") or Path(get_settings().zr_data_root) / "sip")


def load_daily(r: Path) -> pd.DataFrame:
    from zr.data.sip_pull import daily_date

    d = pd.concat(
        [pd.read_parquet(f) for f in sorted((r / "daily").glob("part-*.parquet"))],
        ignore_index=True,
    )
    d["date"] = daily_date(d["ts"])
    return d.drop_duplicates(["symbol", "adj", "date"], keep="last")


def eligibility(d: pd.DataFrame, universe: set) -> pd.DataFrame:
    raw = d[(d["adj"] == "none") & d["symbol"].isin(universe)]
    fwd = d[d["adj"] == "fwd"][["symbol", "date", "open"]].rename(columns={"open": "fwd_open"})
    m = raw.merge(fwd, on=["symbol", "date"], how="left")
    out = []
    for s, g in m.groupby("symbol", sort=False):
        e = so.daily_eligibility(g)
        e["symbol"] = s
        out.append(e)
    return pd.concat(out, ignore_index=True)


def day_groups(b: pd.DataFrame) -> dict:
    """一个标的的 K 线 → {日期: (分钟, o, h, l, c)}。"""
    t = pd.DatetimeIndex(b["ts"]).tz_convert(ET)
    m = np.asarray(t.hour * 60 + t.minute)
    d = np.asarray(t.date)
    arr = {k: b[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    out = {}
    for day, ix in pd.Series(np.arange(len(b))).groupby(d).indices.items():
        out[day] = (m[ix], arr["open"][ix], arr["high"][ix], arr["low"][ix], arr["close"][ix])
    return out


def candidates(r: Path, E: pd.DataFrame, pool: list[str]) -> pd.DataFrame:
    """池内标的：开盘区间 / RelVol / 方向 + 合格标记 + ATR。"""
    parts = []
    for s in pool:
        p = r / "bars5" / f"{s}.parquet"
        if not p.exists():
            continue
        o = so.opening_range(pd.read_parquet(p))
        o["symbol"] = s
        parts.append(o)
    ors = pd.concat(parts, ignore_index=True)
    C = ors.merge(
        E[["symbol", "date", "eligible", "atr_prev", "open"]], on=["symbol", "date"], how="left"
    )
    C["eligible"] = C["eligible"].fillna(False).astype(bool)
    from zr.data.sip_pull import INTRA_FROM

    return C[C["date"] >= INTRA_FROM].reset_index(drop=True)


# ---------------- 执行 ----------------


def run_sims(groups_by_symbol, rows: pd.DataFrame, configs, both_dirs: bool = False) -> dict:
    """rows：symbol / date / dir / or_high / or_low / atr_prev；configs：[(algo, slip)]。返回 {(algo, slip, 方向口径): DataFrame}。"""
    out = {cfg: [] for cfg in configs}
    for s, g in rows.groupby("symbol", sort=False):
        dg = groups_by_symbol(s)
        if dg is None:
            continue
        for r in g.itertuples():
            bars = dg.get(r.date)
            if bars is None:
                continue
            m, o, h, lo, c = bars
            dirs = (r.dir, -r.dir) if both_dirs else (r.dir,)
            for algo, slip in configs:
                for k, dd in enumerate(dirs):
                    res = so.sim_core(
                        m, o, h, lo, c, int(dd), r.or_high, r.or_low, r.atr_prev, slip, ALGOS[algo]
                    )
                    if res is None:
                        continue
                    e, st, x, j, kk, stopped = res
                    out[(algo, slip)].append(
                        (r.Index, k, int(dd), e, x, stopped, int(m[j]), int(m[kk]))
                    )
    res = {}
    for cfg, lst in out.items():
        df = pd.DataFrame(
            lst, columns=["row", "flip", "dir", "entry", "exit", "stopped", "entry_min", "exit_min"]
        )
        res[cfg] = df.merge(rows.drop(columns=["dir"]), left_on="row", right_index=True)
        res[cfg]["atr"] = res[cfg]["atr_prev"]
    return res


def bars1_loader(r: Path):
    cache: dict = {}

    def load_day(day):
        if day not in cache:
            fs = sorted((r / "bars1" / f"day={day}").glob("part-*.parquet"))
            cache.clear()
            cache[day] = (
                pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None
            )
        return cache[day]

    return load_day


# ---------------- 指标 ----------------


def metrics(
    daily: pd.DataFrame, all_days: list, equity0: float, trades: pd.DataFrame, spy: pd.Series
) -> dict:
    D = daily.set_index("date").reindex(all_days)
    r = D["ret"].fillna(0.0).to_numpy()
    eq = equity0 * np.cumprod(1 + r)
    total = eq[-1] / equity0 - 1 if len(eq) else np.nan
    n = len(r)
    irr = (1 + total) ** (252 / n) - 1 if n else np.nan
    sd = r.std(ddof=1)
    peak = np.maximum.accumulate(np.r_[equity0, eq])
    mdd = float(((np.r_[equity0, eq] - peak) / peak).min())
    s = spy.reindex(all_days).to_numpy()
    ok = np.isfinite(s)
    beta, alpha = np.polyfit(s[ok], r[ok], 1) if ok.sum() > 10 else (np.nan, np.nan)
    traded = D["n"].fillna(0).to_numpy() > 0
    t_stat = r.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    rng = np.random.default_rng(0)
    bs = r[rng.integers(0, n, (2000, n))].mean(1) if n else np.array([np.nan])
    tr = trades[trades["shares"] > 0]
    return {
        "days": n,
        "traded_days": int(traded.sum()),
        "trades": len(tr),
        "total": total,
        "irr": irr,
        "vol": sd * np.sqrt(252),
        "sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
        "hit_day": float((r[traded] > 0).mean()) if traded.any() else np.nan,
        "hit_trade": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan,
        "mdd": mdd,
        "worst": float(r.min()),
        "alpha": alpha * 252,
        "beta": beta,
        "mean": r.mean(),
        "t": t_stat,
        "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
        "end_equity": float(eq[-1]) if len(eq) else equity0,
    }


def r_multiples(tr: pd.DataFrame) -> pd.DataFrame:
    tr = tr[tr["shares"] > 0].copy()
    R = so.STOP_ATR * tr["atr_prev"]
    tr["R"] = R
    tr["pnl_ps"] = tr["pnl"] / tr["shares"]
    tr["r_mult"] = tr["pnl_ps"] / R
    tr["pct"] = tr["pnl"] / (tr["shares"] * tr["entry"]) * 100
    return tr


# ---------------- 主流程 ----------------


def run(prelim: bool) -> dict:
    r = sip_root()
    u = pd.read_parquet(r / "universe" / "universe.parquet")
    universe = set(u.loc[u["include"], "symbol"])
    done5 = set((r / "state" / "bars5_done.txt").read_text().split())
    op = r / "derived" / "order5.json"
    order = (
        json.loads(op.read_text()) if op.exists() else sorted(done5)
    )  # 无排序文件（实测数据）时按代码
    pool = [s for s in order if s in done5][: (500 if prelim else None)]
    d = load_daily(r)
    spy = (
        d[(d["symbol"] == "SPY") & (d["adj"] == "none")]
        .sort_values("date")
        .set_index("date")["close"]
        .pct_change()
    )
    E = eligibility(d, universe)
    C = candidates(r, E, pool)
    last = C["date"].max()
    all_days = sorted(C["date"].unique())
    # 时点断言：开盘区间取 09:30 那根，入场不早于 09:35（sim_core 从 m ≥ 575 起）
    assert so.select(C, 50)["relvol"].notna().all()
    top50 = so.select(C, 50)
    top50["rank"] = top50.groupby("date").cumcount() + 1
    load1 = bars1_loader(r)
    # 1 分钟：按日读文件，逐日模拟
    cfgs = [(a, s_) for a in ALGOS for s_ in SLIPS]
    res1 = {cfg: [] for cfg in cfgs}
    for day, g in top50.groupby("date", sort=True):
        b = load1(day)
        if b is None:
            continue
        groups = {s: day_groups(x) for s, x in b.groupby("symbol")}
        rs = run_sims(lambda s, groups=groups: groups.get(s), g, cfgs, both_dirs=True)
        for cfg in cfgs:
            res1[cfg].append(rs[cfg])
    T1 = {cfg: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for cfg, v in res1.items()}
    # 5 分钟：基础版（全部合格且非十字星）、安慰剂 (i) 池（合格 ∩ RelVol ≥ 1）、SIP 前 20
    base_rows = C[C["eligible"] & (C["dir"] != 0)]
    cache5: dict = {}

    def g5(sym):
        if sym not in cache5:
            cache5.clear()
            p = r / "bars5" / f"{sym}.parquet"
            cache5[sym] = day_groups(pd.read_parquet(p)) if p.exists() else None
        return cache5[sym]

    T5 = run_sims(g5, base_rows.sort_values("symbol"), [("cons", 0.01), ("opt", 0.0)])
    return {
        "prelim": prelim,
        "pool": pool,
        "C": C,
        "E": E,
        "top50": top50,
        "T1": T1,
        "T5": T5,
        "spy": spy,
        "all_days": all_days,
        "last": last,
        "universe_n": len(universe),
        "u": u,
    }


def pick(
    T: pd.DataFrame, top: int | None = None, flip: int = 0, long_only: bool = False
) -> pd.DataFrame:
    x = T[T["flip"] == flip]
    if top is not None:
        x = x[x["rank"] <= top]
    if long_only:
        x = x[x["dir"] > 0]
    return x


def experiments(R: dict) -> dict:
    T1, T5, days, spy = R["T1"], R["T5"], R["all_days"], R["spy"]
    out = {}

    def go(name, T, eq0, n, lev, fee, compound=True):
        p = so.portfolio(T, eq0, n, lev, fee, compound)
        out[name] = {"p": p, "m": metrics(p["daily"], days, eq0, p["trades"], spy), "eq0": eq0}

    main_T = pick(T1[(MAIN[0], MAIN[1])], 20)
    go(
        "主口径（SIP 前 20，1 分钟，保守，D-013，滑点 0.01，10 万美元，4 倍）",
        main_T,
        100_000,
        20,
        4.0,
        "d013",
    )
    go(
        "原文口径（SIP 前 20，1 分钟，乐观，每股 0.0035，零滑点，2.5 万美元）",
        pick(T1[("opt", 0.0)], 20),
        25_000,
        20,
        4.0,
        "per_share",
    )
    go("原文口径，10 万美元", pick(T1[("opt", 0.0)], 20), 100_000, 20, 4.0, "per_share")
    go("主口径，2.5 万美元", main_T, 25_000, 20, 4.0, "d013")
    go("主口径，不加杠杆", main_T, 100_000, 20, 1.0, "d013")
    go(
        "1 万人民币，RelVol 第 1 名（不加杠杆，D-013）",
        pick(T1[(MAIN[0], MAIN[1])], 1),
        RMB,
        1,
        1.0,
        "d013",
    )
    go(
        "1 万人民币，RelVol 前 3 名（不加杠杆，D-013）",
        pick(T1[(MAIN[0], MAIN[1])], 3),
        RMB,
        3,
        1.0,
        "d013",
    )
    for top in (10, 50):
        go(f"敏感性：前 {top}", pick(T1[(MAIN[0], MAIN[1])], top), 100_000, top, 4.0, "d013")
    go("敏感性：乐观算法", pick(T1[("opt", 0.01)], 20), 100_000, 20, 4.0, "d013")
    for s_ in SLIPS:
        if s_ != 0.01:
            go(f"敏感性：滑点 {s_:.2f}", pick(T1[("cons", s_)], 20), 100_000, 20, 4.0, "d013")
    go("敏感性：每股 0.0035 佣金", main_T, 100_000, 20, 4.0, "per_share")
    go("敏感性：只做多", pick(T1[(MAIN[0], MAIN[1])], 20, long_only=True), 100_000, 20, 4.0, "d013")
    base_main = T5[("cons", 0.01)].assign(rank=1)
    go(
        "基础版（全部合格，5 分钟执行，主口径设定，10 万美元）",
        base_main,
        100_000,
        None,
        4.0,
        "d013",
    )
    go(
        "基础版（5 分钟执行，原文口径，2.5 万美元）",
        T5[("opt", 0.0)].assign(rank=1),
        25_000,
        None,
        4.0,
        "per_share",
    )
    sip5 = T5[("cons", 0.01)].merge(R["top50"][["symbol", "date", "rank"]], on=["symbol", "date"])
    sip5 = sip5[sip5["rank"] <= 20]
    go("SIP 前 20（5 分钟执行，主口径设定）", sip5, 100_000, 20, 4.0, "d013")
    sip5p = T5[("opt", 0.0)].merge(R["top50"][["symbol", "date", "rank"]], on=["symbol", "date"])
    go(
        "SIP 前 20（5 分钟执行，原文口径，2.5 万美元）",
        sip5p[sip5p["rank"] <= 20],
        25_000,
        20,
        4.0,
        "per_share",
    )
    return out


def placebos(R: dict, seed: int = 0) -> dict:
    """固定权益 10 万美元的日收益均值。(i) 5 分钟执行：合格 ∩ RelVol ≥ 1 中每日随机 20 只 vs SIP 前 20；(ii) 1 分钟：SIP 前 20 随机方向。"""
    days = R["all_days"]
    rng = np.random.default_rng(seed)
    T5 = R["T5"][("cons", 0.01)]
    pool = T5[T5["relvol"] >= 1].copy()
    p = so.portfolio(pool.assign(rank=1), 100_000, 20, 4.0, "d013", compound=False)["trades"]
    pnl_by_day = {d: g["pnl"].to_numpy() for d, g in p.groupby("date")}
    # 候选池里未触发的也算「选中但未交易」：按候选总数抽样
    cand = R["C"][R["C"]["eligible"] & (R["C"]["relvol"] >= 1)].groupby("date").size().to_dict()
    nd = len(days)
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
    sip5 = T5.merge(R["top50"][["symbol", "date", "rank"]], on=["symbol", "date"])
    sip5 = sip5[sip5["rank"] <= 20]
    act_i = (
        so.portfolio(sip5, 100_000, 20, 4.0, "d013", compound=False)["daily"]["pnl"].sum()
        / 100_000
        / nd
    )
    # (ii)
    T1 = R["T1"][(MAIN[0], MAIN[1])]
    t = T1[T1["rank"] <= 20]
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
    act_ii = A.sum() / 100_000 / nd
    return {"i": (act_i, sims), "ii": (act_ii, sims2)}


def breakeven(R: dict, main: dict) -> pd.DataFrame:
    """保守、D-013、股数固定为主口径实际股数：每边滑点 s ∈ [0, 0.10] 重算成交价，求每笔平均盈亏归零的 s（线性插值）。"""
    tr = main["p"]["trades"]
    tr = tr[tr["shares"] > 0][
        ["symbol", "date", "shares", "atr_prev", "entry", "dir", "or_high", "or_low", "rank"]
    ].reset_index(drop=True)
    grid = np.round(np.arange(0, 0.1001, 0.005), 3)
    r = sip_root()
    load1 = bars1_loader(r)
    pnl = np.zeros((len(tr), len(grid)))
    for day, g in tr.groupby("date"):
        b = load1(day)
        groups = {s: day_groups(x) for s, x in b.groupby("symbol")}
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
        y = pnl[mask].mean(0)
        if not mask.any():
            return np.nan, np.nan
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelim", action="store_true")
    a = ap.parse_args(argv)
    from zr.research.sip_orb_write import write_report

    R = run(a.prelim)
    X = experiments(R)
    P = placebos(R)
    B = breakeven(R, X["主口径（SIP 前 20，1 分钟，保守，D-013，滑点 0.01，10 万美元，4 倍）"])
    path = write_report(R, X, P, B)
    print(f"写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
