# ruff: noqa: E501  （中文文档字符串与注释较长）
"""事件研究：事件定义是可替换的输入（EventFn），随机对照、入场时点曲线、方向化特征（R-055 归档）。

EventFn：f(e1: 1 分钟 K（口径 C，索引 = UTC 开盘时刻）) → 与 e1 逐行对齐的 Series，取值 ±1 / 0，
只在信号 K 收盘那一分钟（其最后一根 1 分钟 K）的行上非零。必须通过
zr.research.leakcheck.assert_no_lookahead_frame。默认实现 boll_macd_event = 夜间研究阶段 4 的预登记事件；
换成负责人自己的规则时只需另写一个同签名的函数。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from zr.features.bars import resample_bars
from zr.research.timing.data import TICK
from zr.research.timing.labels import l4

EventFn = Callable[[pd.DataFrame], pd.Series]
SESS_EDGES = np.array(
    [572, 600, 690, 840, 946]
)  # 09:32–09:59 / 10:00–11:29 / 11:30–13:59 / 14:00–15:45
KEYS = ("c0", "e1", "e2", "e3", "e6")  # t 收盘价、t+1（主）、t+2、t+3、t+6 开盘
OFFS = {"c0": 1, "e1": 1, "e2": 2, "e3": 3, "e6": 6}


def pbars(e1: pd.DataFrame, P: int) -> pd.DataFrame:
    b = e1 if P == 1 else resample_bars(e1, f"{P}min")
    return b[b["close"].notna()]


def to_rows(e1: pd.DataFrame, b: pd.DataFrame, ev: np.ndarray, P: int, name: str) -> pd.Series:
    """P 分钟 K 上的 ±1 → 该 K 收盘那一分钟的 1 分钟 K 行。"""
    close_t = b.index + pd.Timedelta(minutes=P)
    row_t = pd.Series(np.arange(len(e1)), index=e1.index + pd.Timedelta(minutes=1))
    rows = row_t.reindex(close_t).to_numpy()
    out = np.zeros(len(e1))
    ok = np.isfinite(rows) & (ev != 0)
    out[rows[ok].astype(int)] = ev[ok]
    return pd.Series(out, index=e1.index, name=name)


def boll_macd_event(P: int, thr: float, s: int = 5) -> EventFn:
    """夜间研究预登记事件：P 分钟布林中轨近 s 根斜率（÷ 收盘）> thr 且 MACD(6,13,6) 柱最近 3 根内由 ≤0 变 >0、
    当根柱 > 上一根；做空镜像；条件上升沿触发。"""

    def fn(e1: pd.DataFrame) -> pd.Series:
        b = pbars(e1, P)
        c = b["close"]
        mid = c.rolling(20, min_periods=20).mean()
        sl = ((mid - mid.shift(s)) / c).to_numpy()
        dif = c.ewm(span=6, adjust=False).mean() - c.ewm(span=13, adjust=False).mean()
        h = (dif - dif.ewm(span=6, adjust=False).mean()).to_numpy()
        h1 = np.r_[np.nan, h[:-1]]
        up3 = (
            pd.Series(np.r_[False, (h[1:] > 0) & (h[:-1] <= 0)])
            .rolling(3, min_periods=1)
            .max()
            .to_numpy()
            > 0
        )
        dn3 = (
            pd.Series(np.r_[False, (h[1:] < 0) & (h[:-1] >= 0)])
            .rolling(3, min_periods=1)
            .max()
            .to_numpy()
            > 0
        )
        with np.errstate(invalid="ignore"):
            sig = np.where(
                (sl > thr) & up3 & (h > h1), 1, np.where((sl < -thr) & dn3 & (h < h1), -1, 0)
            )
        prev = np.r_[0, sig[:-1]]
        return to_rows(e1, b, np.where(sig != prev, sig, 0), P, f"boll_macd_p{P}")

    return fn


def random_control(
    vals: dict,
    day: np.ndarray,
    mod: np.ndarray,
    trig: pd.DataFrame,
    mask: np.ndarray,
    n_draw: int = 1000,
    seed: int = 0,
) -> dict:
    """同日、同时段桶、同方向组成，排除离任一触发 ±2 分钟的决策分钟；返回每个指标 n_draw 个按笔合并均值。
    vals：{名: (做多值数组, 做空值数组)}（与决策行对齐）；trig：DataFrame[i（决策行号）, dir]。"""
    rng = np.random.default_rng(seed)
    bucket = np.searchsorted(SESS_EDGES, mod, side="right") - 1
    near = np.zeros(len(day), bool)
    for o in range(-2, 3):
        j = trig["i"].to_numpy() + o
        near[j[(j >= 0) & (j < len(day))]] = True
    cand_mask = mask & ~near
    keys = pd.Series(list(zip(day, bucket, strict=True)))
    cand_idx = (
        pd.Series(np.arange(len(day))[cand_mask])
        .groupby(keys[cand_mask].to_numpy())
        .apply(np.asarray)
    )
    T = trig.assign(day=day[trig["i"]], b=bucket[trig["i"]])
    sums = {k: np.zeros(n_draw) for k in vals}
    cnts = {k: np.zeros(n_draw) for k in vals}
    for (dd, bb), grp in T.groupby(["day", "b"]):
        if (dd, bb) not in cand_idx.index:
            continue
        cand = cand_idx[(dd, bb)]
        k = len(grp)
        pick = (
            cand[rng.random((n_draw, len(cand))).argsort(1)[:, :k]]
            if len(cand) >= k
            else cand[rng.integers(0, len(cand), (n_draw, k))]
        )
        dirs = grp["dir"].to_numpy()[None, :]
        for name, (vl, vs) in vals.items():
            x = np.where(dirs > 0, vl[pick], vs[pick])
            ok = np.isfinite(x)
            sums[name] += np.where(ok, x, 0).sum(1)
            cnts[name] += ok.sum(1)
    return {k: sums[k] / np.where(cnts[k] > 0, cnts[k], np.nan) for k in vals}


def entry_curve(
    G: dict,
    g: pd.DataFrame,
    rows: np.ndarray,
    dirs: np.ndarray,
    P: int,
    m,
    H: int = 15,
    kind: str = "a",
) -> dict:
    """入场时点曲线：t 收盘、t+1（主）、t+2、t+3、t+6 开盘，持有长度相同；返回 {档: dict(entry, exit, ret)}。"""
    c, dend = G["c"], G["day_end"]
    out = {}
    for key in KEYS:
        e = rows + OFFS[key]
        ok = e < dend[rows]
        ec = np.where(ok, e, rows + 1)
        res = {k: np.full(len(rows), np.nan) for k in ("entry", "exit", "ret")}
        for dv in (1, -1):
            s = dirs == dv
            if not s.any():
                continue
            px = (c[rows[s]] + dv * TICK) if key == "c0" else None
            r = l4(
                G,
                ec[s],
                dv,
                m,
                H,
                kind,
                rev_next=g[f"rev{P}_{'long' if dv > 0 else 'short'}"].to_numpy(),
                entry_px=px,
            )
            for k in res:
                res[k][s] = np.where(ok[s], r[k], np.nan)
        out[key] = res
    return out


SIGNED = (
    "slope",
    "dist",
    "dif",
    "dea",
    "hist_atr",
    "hist_chg",
    "side_run",
    "ride_run",
    "ret",
    "zero",
    "_div",
    "di_diff",
    "aroon",
    "supertrend",
    "obv_slope",
    "signed",
    "gap",
    "break",
    "retest",
    "consistency",
    "kd_side",
    "j_ext",
    "updown",
    "vwap_z",
    "rsi14_chg",
)


def directional(X: pd.DataFrame, d: np.ndarray) -> pd.DataFrame:
    """特征方向化：带符号量 × d；%b 取 1 − x、0–100 振荡器取 100 − x（做空时）；无方向量不变。"""
    Y = X.copy()
    for c in X.columns:
        base = c.split("_", 1)[1] if c[:2] in ("p1", "p2", "p5", "f2", "f5") else c
        if base == "pctb":
            Y[c] = np.where(d > 0, X[c], 1 - X[c])
        elif base in ("k", "d", "j", "rsi2", "rsi6", "rsi14", "mfi14"):
            Y[c] = np.where(d > 0, X[c], 100 - X[c])
        elif any(s in base for s in SIGNED):
            Y[c] = X[c] * d
    return Y
