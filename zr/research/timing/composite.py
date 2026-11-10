# ruff: noqa: E501  （中文文档字符串与注释较长）
"""等权因子合成（不训练模型，R-055 归档）：因子选择、只用过去 20 个交易日的标准化、合成分数、选点。"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_DAY, COOL = 6, 2


def select(
    T: pd.DataFrame,
    X: pd.DataFrame,
    dev: np.ndarray,
    und: str,
    h: int = 15,
    k: int = 10,
    rho_max: float = 0.7,
) -> list[tuple[str, float, float]]:
    """阶段 3 结果 T（列 und / h / feature / ic / t / bh / year_same_sign）中 BH 通过且各年同号的因子，按 |t| 排序，
    两两秩相关 > rho_max 只留 |t| 更高者，最多 k 个。返回 [(特征, 方向 = IC 符号, t)]。"""
    c = T[(T["und"] == und) & (T["h"] == h) & T["bh"] & T["year_same_sign"]]
    c = c.reindex(c["t"].abs().sort_values(ascending=False).index)
    keep: list[tuple[str, float, float]] = []
    samp = X[dev].sample(n=min(50000, int(dev.sum())), random_state=0)
    for r in c.itertuples():
        if len(keep) >= k:
            break
        if all(
            not (abs(samp[[r.feature, f]].rank().corr().iloc[0, 1]) > rho_max) for f, _, _ in keep
        ):
            keep.append((r.feature, float(np.sign(r.ic)), float(r.t)))
    return keep


def zscore_past(x: pd.Series, day: np.ndarray, window: int = 20) -> np.ndarray:
    """用此前 window 个交易日（不含当日）的均值与标准差标准化（日序列 → shift(1) → 映射回行）。"""
    df = pd.DataFrame({"d": day, "x": np.asarray(x, float)})
    g = df.groupby("d")["x"]
    agg = pd.DataFrame({"s": g.sum(), "q": g.apply(lambda v: np.nansum(v**2)), "n": g.count()})
    r = agg.rolling(window, min_periods=window // 2).sum().shift(1)
    mu = r["s"] / r["n"]
    sd = np.sqrt(np.clip(r["q"] / r["n"] - mu**2, 0, None))
    m, s = pd.Series(day).map(mu).to_numpy(), pd.Series(day).map(sd).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (df["x"].to_numpy() - m) / np.where(s > 0, s, np.nan)


def score(X: pd.DataFrame, day: np.ndarray, fac) -> np.ndarray:
    """等权合成：Σ 方向 × z；任一因子缺失 → NaN。"""
    z = np.zeros(len(X))
    cnt = np.zeros(len(X))
    for f, sgn, _ in fac:
        v = zscore_past(X[f], day)
        ok = np.isfinite(v)
        z += np.where(ok, sgn * v, 0)
        cnt += ok
    return np.where(cnt == len(fac), z, np.nan)


def pick(
    sc: np.ndarray,
    cut: float,
    day: np.ndarray,
    mod: np.ndarray,
    mask: np.ndarray,
    max_day: int = MAX_DAY,
    cool: int = COOL,
) -> np.ndarray:
    """|分数| ≥ 切点；每日最多 max_day 笔、相邻两笔间隔 ≥ cool 分钟（按时间顺序贪心）。"""
    cand = np.flatnonzero(mask & np.isfinite(sc) & (np.abs(sc) >= cut))
    out, last_day, last_mod, cnt = [], None, -99, 0
    for k in cand:
        if day[k] != last_day:
            last_day, last_mod, cnt = day[k], -99, 0
        if cnt >= max_day or mod[k] - last_mod < cool:
            continue
        out.append(k)
        last_mod, cnt = mod[k], cnt + 1
    return np.array(out, dtype=int)
