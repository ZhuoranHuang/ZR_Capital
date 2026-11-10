# ruff: noqa: E501  （中文文档字符串与注释较长）
"""趋势持有分支（R-045 噪声区突破，正股；R-055 归档）。

与 R-045 的差异：判定在该 1 分钟 K 收盘确认、下一分钟开盘 ± 1 tick 成交（R-045 按判定 K 收盘价成交）；
仓位按满仓整数股（不用 R-045 的波动率杠杆）。出场：orig（每个判定时刻重算信号，不同则平仓 / 反手）、
fixN（固定持有 N 分钟）、close（持有到收盘）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.research.timing.data import TICK

EXITS = ("orig", "fix30", "fix60", "fix120", "close")


def noise_signal(g: pd.DataFrame, vm: float) -> pd.DataFrame:
    """逐分钟 UB / LB / VWAP 与信号（+1 / −1 / 0），只用 ≤ t 的数据（σ = 前 14 日同一分钟 |C/O−1| 均值，按日 shift）。
    g = RTH 分钟网格（列 day / mod / open / high / low / close / volume）。"""
    day, mod = g["day"].to_numpy(), g["mod"].to_numpy()
    o, h, l_, c, v = (g[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    first = np.r_[True, day[1:] != day[:-1]]
    dopen = pd.Series(np.where(first, o, np.nan)).ffill().to_numpy()
    D = pd.DataFrame({"day": day, "c": c}).groupby("day")["c"].last()
    pc = pd.Series(day).map(D.shift(1)).to_numpy(float)
    mv = np.abs(c / dopen - 1)
    piv = pd.DataFrame({"day": day, "mod": mod, "mv": mv}).pivot_table(
        index="day", columns="mod", values="mv", aggfunc="last"
    )
    sig = piv.rolling(14, min_periods=14).mean().shift(1)
    s = sig.stack().reindex(pd.MultiIndex.from_arrays([day, mod])).to_numpy(float)
    ub = np.maximum(dopen, pc) * (1 + vm * s)
    lb = np.minimum(dopen, pc) * (1 - vm * s)
    tp = (h + l_ + c) / 3
    cv = pd.Series(v).groupby(day).cumsum().to_numpy()
    vw = pd.Series(tp * v).groupby(day).cumsum().to_numpy() / np.where(cv > 0, cv, np.nan)
    with np.errstate(invalid="ignore"):
        sg = np.where((c > ub) & (c > vw), 1, np.where((c < lb) & (c < vw), -1, 0))
    sg = np.where(np.isfinite(ub) & np.isfinite(vw), sg, 0)
    return pd.DataFrame({"ub": ub, "lb": lb, "vwap": vw, "sig": sg}, index=g.index)


def decision_mask(mod: np.ndarray, step: int) -> np.ndarray:
    """判定分钟：10:00 起每 step 分钟，至 15:30（step 30）/ 15:45（step 5）。"""
    last = 930 if step == 30 else 945
    return (mod >= 600) & (mod <= last) & ((mod - 600) % step == 0)


def backtest(g: pd.DataFrame, S: pd.DataFrame, step: int, exit_: str) -> pd.DataFrame:
    """夜间研究的回测语义（一个仓位，收盘前最后一根收盘平仓；不含账户层）。返回逐笔：day / dir / e_row / x_row / entry / exit / hold / reason。"""
    day, mod = g["day"].to_numpy(), g["mod"].to_numpy()
    o, c, dend = g["open"].to_numpy(float), g["close"].to_numpy(float), g["day_end"].to_numpy()
    dec = decision_mask(mod, step)
    sig = S["sig"].to_numpy()
    trades = []
    for s0 in np.flatnonzero(np.r_[True, day[1:] != day[:-1]]):
        e_ = dend[s0]
        pos, ent_row, ent_px = 0, None, None

        def close_at(row, px_raw, reason, _s0=s0):
            nonlocal pos, ent_row, ent_px
            trades.append(
                {
                    "day": day[_s0],
                    "dir": pos,
                    "e_row": ent_row,
                    "x_row": row,
                    "entry": ent_px,
                    "exit": px_raw - pos * TICK,
                    "hold": row - ent_row,
                    "reason": reason,
                }
            )
            pos, ent_row, ent_px = 0, None, None

        for t in np.flatnonzero(dec[s0:e_]) + s0:
            if t + 1 >= e_:
                break
            s = sig[t]
            if exit_ == "orig":
                if pos != 0 and s != pos:
                    close_at(t + 1, o[t + 1], "signal")
                if pos == 0 and s != 0:
                    pos, ent_row, ent_px = s, t + 1, o[t + 1] + s * TICK
            elif pos == 0 and s != 0:
                pos, ent_row, ent_px = s, t + 1, o[t + 1] + s * TICK
                if exit_.startswith("fix"):
                    xr = ent_row + int(exit_[3:])
                    if xr < e_:
                        close_at(xr, o[xr], "time")
        if pos != 0:
            close_at(e_, c[e_ - 1], "close")
    T = pd.DataFrame(trades)
    if exit_.startswith("fix") and len(T):  # 一个仓位：删除与上一笔重叠的交易
        keep, last_x, last_d = [], -1, None
        for r in T.itertuples():
            if r.day == last_d and r.e_row < last_x:
                continue
            keep.append(r.Index)
            last_x, last_d = r.x_row, r.day
        T = T.loc[keep].reset_index(drop=True)
    return T


def random_dirs(d: np.ndarray, n: int, mode: str, seed: int = 0) -> np.ndarray:
    """随机对照的方向矩阵 (n, 笔数)。mode="half"：每笔独立 ±1 各半；mode="permute"：保持策略自身的多空笔数，
    只打乱方向在各笔之间的分配（扣除标的单边漂移）。"""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, int)
    if mode == "half":
        return rng.choice([-1, 1], (n, len(d)))
    if mode == "permute":
        return np.array([rng.permutation(d) for _ in range(n)])
    raise ValueError(mode)


def half_hour(g: pd.DataFrame) -> pd.DataFrame:
    """对照格：首半小时收益（10:00 可知）决定方向，15:30 开盘 ± 1 tick 入场、收盘出场。"""
    rows = []
    for dd, x in g.groupby("day"):
        m = x["mod"].to_numpy()
        if not ((m == 570).any() and (m == 599).any() and (m == 930).any()) or m.max() < 959:
            continue
        o0, c30 = x.loc[m == 570, "open"].iloc[0], x.loc[m == 599, "close"].iloc[0]
        d = int(np.sign(c30 / o0 - 1))
        if d == 0:
            continue
        rows.append(
            {
                "day": dd,
                "dir": d,
                "entry": x.loc[m == 930, "open"].iloc[0] + d * TICK,
                "exit": x.loc[m == 959, "close"].iloc[0] - d * TICK,
                "hold": 30,
                "r1": c30 / o0 - 1,
                "r_last": x.loc[m == 959, "close"].iloc[0] / x.loc[m == 930, "open"].iloc[0] - 1,
            }
        )
    return pd.DataFrame(rows)
