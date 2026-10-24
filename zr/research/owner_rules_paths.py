"""R-049 正股层补报（负责人指令，PROFIT_LABEL_MINING.md 第 1 节）：owner_rule_1..5、
调研三条与随机时刻，触发后 30 秒、1、2、3、5、10、15、30 分钟的方向化收益（%），
同一时间窗内最大有利偏移（MFE）的中位数与分布；「先到 +0.1% / +0.2% / +0.3%」与
「先到 −0.1% / −0.2%」的先后概率（30 分钟内）。

触发 = R-049 第一层落盘的触发（owner_rules_oos stock.parquet：时段内条件由假变真、
同规则同方向 5 分钟去重、ORB 每日 1 次）。随机时刻：每个触发在同日 ±30 分钟内
（09:32–15:40、剔除 ±2 分钟）抽 5 个、同方向（固定种子），汇总为一行「随机」。
价格 = 秒级平价反推价（1 秒网格，10 秒内无更新为空）；不计费用（期权层的扣费后
盈亏见 OWNER_RULES_OOS.md）。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest import intrabar as ib

HORIZONS = (30, 60, 120, 180, 300, 600, 900, 1800)
UPS = (0.001, 0.002, 0.003)
DOWNS = (0.001, 0.002)
N_RAND = 5
RULES = (
    "owner_rule_1",
    "owner_rule_2",
    "owner_rule_3",
    "owner_rule_4",
    "owner_rule_5",
    "survey_rsi2_pullback",
    "survey_ema_pullback",
    "survey_orb_breakout",
)


def _paths(S: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    """每行 = 一个时刻起 0..1800 秒的方向化收益（比例）；越过收盘为 NaN。"""
    w = HORIZONS[-1]
    idx = k[:, None] + np.arange(w + 1)[None, :]
    ok = idx < len(S)
    P = np.where(ok, S[np.clip(idx, 0, len(S) - 1)], np.nan)
    return d[:, None] * (P / P[:, :1] - 1)


def path_metrics(R: np.ndarray) -> pd.DataFrame:
    out = {}
    for h in HORIZONS:
        out[f"ret_{h}"] = R[:, h] * 100
        out[f"mfe_{h}"] = np.nanmax(R[:, 1 : h + 1], axis=1) * 100
    n = R.shape[1]
    big = n + 1
    for u in UPS:
        hit = R >= u
        fu = np.where(hit.any(1), hit.argmax(1), big)
        for dn in DOWNS:
            hd = R <= -dn
            fd = np.where(hd.any(1), hd.argmax(1), big)
            out[f"up{u * 100:g}_first_vs_dn{dn * 100:g}"] = np.where(
                (fu == big) & (fd == big), 0.0, np.where(fu < fd, 1.0, -1.0)
            )
    return pd.DataFrame(out)


def day_rows(S: np.ndarray, t0: int, trig: pd.DataFrame, in_win: np.ndarray, rng) -> pd.DataFrame:
    k = ((trig["t_ns"].to_numpy() - t0) // ib.NS).astype(int)
    d = trig["dir"].to_numpy()
    m = path_metrics(_paths(S, k, d))
    m["rule"] = trig["rule"].to_numpy()
    win = np.flatnonzero(in_win & np.isfinite(S))
    rk, rd = [], []
    for kk, dd in zip(k, d, strict=True):
        pool = win[(np.abs(win - kk) <= 1800) & (np.abs(win - kk) > 120)]
        if len(pool):
            c = rng.choice(pool, size=min(N_RAND, len(pool)), replace=False)
            rk += list(c)
            rd += [dd] * len(c)
    if rk:
        r = path_metrics(_paths(S, np.array(rk), np.array(rd)))
        r["rule"] = "随机"
        m = pd.concat([m, r], ignore_index=True)
    return m


def run(root: Path, stock: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    stock = stock[stock["rule"].isin(RULES)]
    rng = np.random.default_rng(seed)
    parts = []
    for day, trig in stock.groupby("day"):
        dd = dt.date.fromisoformat(str(day))
        g = ib.second_grid(root, "MU", dd)
        if g is None:
            continue
        hm = np.asarray(
            pd.DatetimeIndex(g["t"]).tz_localize("UTC").tz_convert(ib.ET).strftime("%H:%M")
        )
        in_win = (hm >= "09:32") & (hm <= "15:40")
        parts.append(
            day_rows(g["S"].to_numpy(), int(g["t"].iloc[0]), trig, in_win, rng).assign(
                day=str(day), dir=None
            )
        )
    return pd.concat(parts, ignore_index=True)


def summarize(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    g = m.groupby("rule")
    ret = g[[f"ret_{h}" for h in HORIZONS]].mean()
    ret.columns = [f"{h // 60} 分" if h >= 60 else f"{h} 秒" for h in HORIZONS]
    mfe_med = g[[f"mfe_{h}" for h in HORIZONS]].median()
    mfe_med.columns = ret.columns
    q = g["mfe_1800"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = [f"P{int(c * 100)}" for c in q.columns]
    fp = {}
    for c in [x for x in m.columns if x.startswith("up")]:
        fp[c.replace("_first_vs_", " 先于 ")] = g[c].apply(
            lambda s: f"{(s == 1).mean():.0%} / {(s == -1).mean():.0%}"
        )
    n = g.size().rename("n")
    return {
        "ret": pd.concat([n, ret.round(3)], axis=1),
        "mfe_med": mfe_med.round(3),
        "mfe30_q": q.round(3),
        "first": pd.DataFrame(fp),
    }
