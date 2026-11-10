# ruff: noqa: E501  （中文文档字符串与注释较长）
"""统计：按笔合并均值 + 按交易日聚类的稳健方差 + 按日整块重抽样；信息系数（全样本排序口径，L-002）；BH。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def cluster_var(x: np.ndarray, days: np.ndarray) -> tuple[float, float]:
    """按笔合并均值 r̄ = Σx / N 与其聚类稳健方差 Σ_d (S_d − r̄·n_d)² / N² × D/(D−1)。"""
    x = np.asarray(x, float)
    N = len(x)
    r = x.mean()
    g = pd.DataFrame({"d": np.asarray(days), "x": x}).groupby("d")["x"].agg(["sum", "count"])
    S, n, D = g["sum"].to_numpy(), g["count"].to_numpy(), len(g)
    var = ((S - r * n) ** 2).sum() / N**2 * (D / (D - 1)) if D > 1 else np.nan
    return float(r), float(var)


def pooled(x, days, n_boot: int = 2000, seed: int = 0) -> dict:
    """按笔合并均值（主口径）、聚类 SE / t / 95% 区间、按日整块重抽样 95% 区间、按天等权均值（副口径）、胜率。"""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, days = x[ok], np.asarray(days)[ok]
    N = len(x)
    if N == 0:
        return {
            "n": 0,
            "days": 0,
            "mean": np.nan,
            "se": np.nan,
            "t": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "b_lo": np.nan,
            "b_hi": np.nan,
            "day_eq": np.nan,
            "win": np.nan,
        }
    r, var = cluster_var(x, days)
    se = float(np.sqrt(var)) if np.isfinite(var) else np.nan
    g = pd.DataFrame({"d": days, "x": x}).groupby("d")["x"].agg(["sum", "count"])
    S, n, D = g["sum"].to_numpy(), g["count"].to_numpy(), len(g)
    if D > 1 and n_boot:
        idx = np.random.default_rng(seed).integers(0, D, (n_boot, D))
        b_lo, b_hi = np.percentile(S[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    else:
        b_lo = b_hi = np.nan
    ok_se = np.isfinite(se)
    return {
        "n": N,
        "days": D,
        "mean": r,
        "se": se,
        "t": r / se if ok_se and se > 0 else np.nan,
        "ci_lo": r - 1.96 * se if ok_se else np.nan,
        "ci_hi": r + 1.96 * se if ok_se else np.nan,
        "b_lo": float(b_lo),
        "b_hi": float(b_hi),
        "day_eq": float((S / n).mean()),
        "win": float((x > 0).mean()),
    }


def fmt_ci(s: dict, nd: int = 4) -> str:
    if not s.get("n"):
        return "—"
    return (
        f"{s['mean']:+.{nd}f}%（{s['ci_lo']:+.{nd}f}～{s['ci_hi']:+.{nd}f}；"
        f"重抽样 {s['b_lo']:+.{nd}f}～{s['b_hi']:+.{nd}f}）"
    )


def daily_ic_global(f, y, day, min_obs: int = 20) -> pd.Series:
    """主口径信息系数：在全部样本上排序并标准化（总体标准差，ddof=0），逐日求 z(f)·z(y) 的均值（跨日等权平均由调用方做）。
    不在日内中心化——日内排序会用到当日 t 之后的价格路径（L-002）。"""
    f, y, day = np.asarray(f, float), np.asarray(y, float), np.asarray(day)
    m = np.isfinite(f) & np.isfinite(y)
    if m.sum() < 100:
        return pd.Series(dtype=float)
    rf, ry = pd.Series(f[m]).rank().to_numpy(), pd.Series(y[m]).rank().to_numpy()
    zf, zy = (rf - rf.mean()) / rf.std(), (ry - ry.mean()) / ry.std()
    s = pd.Series(zf * zy).groupby(day[m])
    return s.mean()[s.size() >= min_obs]


def daily_ic_withinday(f, y, day, min_obs: int = 20) -> pd.Series:
    """日内排序的 Spearman（夜间任务书原口径）。**对水平类特征有机械负偏，只作参照（L-002）。**"""
    f, y, day = np.asarray(f, float), np.asarray(y, float), np.asarray(day)
    m = np.isfinite(f) & np.isfinite(y)
    if m.sum() < 100:
        return pd.Series(dtype=float)
    df = pd.DataFrame({"d": day[m], "f": f[m], "y": y[m]})
    r = df.groupby("d")[["f", "y"]].rank()
    r["d"] = df["d"].to_numpy()
    g = r.groupby("d")
    mf, my = g["f"].transform("mean"), g["y"].transform("mean")
    num = ((r["f"] - mf) * (r["y"] - my)).groupby(r["d"]).sum()
    den = np.sqrt(
        ((r["f"] - mf) ** 2).groupby(r["d"]).sum() * ((r["y"] - my) ** 2).groupby(r["d"]).sum()
    )
    return (num / den.replace(0, np.nan))[g.size() >= min_obs].dropna()


def ic_stats(ic: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    """日 IC 序列 → 均值、t（跨日）、双侧 p、按日重抽样 95% 区间、为正天数占比。"""
    v = np.asarray(ic, float)
    D = len(v)
    if D < 20:
        return {
            "ic": np.nan,
            "t": np.nan,
            "p": 1.0,
            "lo": np.nan,
            "hi": np.nan,
            "pos": np.nan,
            "days": D,
        }
    m, sd = v.mean(), v.std(ddof=1)
    t = m / (sd / np.sqrt(D)) if sd > 0 else np.nan
    p = float(2 * st.t.sf(abs(t), D - 1)) if np.isfinite(t) else 1.0
    bs = v[np.random.default_rng(seed).integers(0, D, (n_boot, D))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"ic": m, "t": t, "p": p, "lo": lo, "hi": hi, "pos": float((v > 0).mean()), "days": D}


def bh(p, q: float = 0.10) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = int(np.flatnonzero(ok).max()) + 1 if ok.any() else 0
    out = np.zeros(m, bool)
    out[o[:k]] = True
    return out
