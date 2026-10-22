"""E1 回踩深度（D-022）：遍历用的回踩信号 + 进模型的连续深度特征。全部只用 ≤t 的数据。

信号（多头；空头对称）：bar t 处于通道（chan = +1），t 之前至少 n 根阴线（close < open）
且与 t 同日，t 收阳（close > open）；回踩低点 L = min(low[t−n..t])；
布林(P, k) 取 bar t 的值；深度条件二选一：
- pctb：(L − lower) / (upper − lower) ≤ x
- eps：|L − mid| / mid ≤ ε
空头：chan = −1，至少 n 根阳线后收阴，H = max(high[t−n..t])，
pctb 条件为 (upper − H) / (upper − lower) ≤ x，eps 条件为 |H − mid| / mid ≤ ε。

连续特征（按事件方向取值，越小 = 回踩越深）：最近 w 根（含 t）的极值，P ∈ bb_period：
- pb_pctb_P：多头 (L_w − lower)/(upper − lower)，空头 (upper − H_w)/(upper − lower)
- pb_eps_P：多头 (L_w − mid)/mid，空头 (mid − H_w)/mid（带符号）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.features import indicators as ind
from zr.features.bars import ET

CONFIG = Path(__file__).resolve().parents[2] / "config" / "e1_sweep.yaml"


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pullback_columns(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    return [f"pb_{k}_{p}" for p in cfg["bb_period"] for k in ("pctb", "eps")]


def _runs(flag: np.ndarray, day: np.ndarray) -> np.ndarray:
    """截至每根 bar（含）的同日连续 True 根数。"""
    out = np.zeros(len(flag), dtype=int)
    for i in range(len(flag)):
        if flag[i]:
            out[i] = out[i - 1] + 1 if i and day[i] == day[i - 1] else 1
    return out


class PullbackBase:
    """一次性算好与参数无关的部分（阴阳线连续根数、各周期布林），供遍历复用。"""

    def __init__(self, df: pd.DataFrame, periods, k: float = 2.0, max_n: int = 3):
        self.df = df
        self.day = np.asarray(df.index.tz_convert(ET).date)
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        red, green = c < o, c > o
        self.red_run_prev = np.r_[0, _runs(red, self.day)[:-1]]  # t−1 结束的连续阴线根数
        self.green_run_prev = np.r_[0, _runs(green, self.day)[:-1]]
        same_prev = np.r_[False, self.day[1:] == self.day[:-1]]
        self.red_run_prev[~same_prev] = 0
        self.green_run_prev[~same_prev] = 0
        self.red, self.green = red, green
        self.bb = {p: ind.bollinger(df["close"], p, k) for p in periods}
        lo, hi = df["low"], df["high"]
        self.lo_n = {
            n: lo.rolling(n + 1, min_periods=1).min().to_numpy() for n in range(1, max_n + 1)
        }
        self.hi_n = {
            n: hi.rolling(n + 1, min_periods=1).max().to_numpy() for n in range(1, max_n + 1)
        }

    def signal(self, chan: pd.Series, period: int, kind: str, value: float, n: int) -> np.ndarray:
        b = self.bb[period]
        mid, up, lw = (b[c].to_numpy() for c in ("mid", "upper", "lower"))
        width = up - lw
        ch = chan.to_numpy()
        L, H = self.lo_n[n], self.hi_n[n]
        with np.errstate(all="ignore"):
            if kind == "pctb":
                d_long = (L - lw) / width <= value
                d_short = (up - H) / width <= value
            elif kind == "eps":
                d_long = np.abs(L - mid) / mid <= value
                d_short = np.abs(H - mid) / mid <= value
            else:
                raise ValueError(kind)
        long = (ch == 1) & (self.red_run_prev >= n) & self.green & d_long
        short = (ch == -1) & (self.green_run_prev >= n) & self.red & d_short
        return np.where(long, 1, np.where(short, -1, 0))


def depth_features(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """每根 bar 的多 / 空两版深度特征：列 pb_{pctb,eps}_{P}_{long,short}。"""
    cfg = cfg or load_config()
    w = int(cfg["feature_window"])
    lw = df["low"].rolling(w, min_periods=1).min()
    hw = df["high"].rolling(w, min_periods=1).max()
    out = {}
    for p in cfg["bb_period"]:
        b = ind.bollinger(df["close"], p, cfg["bb_k"])
        width = (b["upper"] - b["lower"]).replace(0, np.nan)
        out[f"pb_pctb_{p}_long"] = (lw - b["lower"]) / width
        out[f"pb_pctb_{p}_short"] = (b["upper"] - hw) / width
        out[f"pb_eps_{p}_long"] = (lw - b["mid"]) / b["mid"]
        out[f"pb_eps_{p}_short"] = (b["mid"] - hw) / b["mid"]
    return pd.DataFrame(out, index=df.index)


def directional(feat: pd.DataFrame, ts: pd.Series, direction: np.ndarray, cfg: dict | None = None):
    """按事件方向取多 / 空版本 → 列 pb_{pctb,eps}_{P}。"""
    cfg = cfg or load_config()
    f = feat.reindex(ts)
    out = {}
    for c in pullback_columns(cfg):
        out[c] = np.where(direction == 1, f[f"{c}_long"].to_numpy(), f[f"{c}_short"].to_numpy())
    return pd.DataFrame(out)
