"""通道层（METHOD_V2 补充 1）：通道定义 A/B/C/D/F 的逐根状态（+1 上升 / −1 下降 / 0 无）
与连续的通道质量特征。无前视：只用 ≤t 的 bar。

定义：
- A：MACD 三条件（DIF>DEA、DIF>0、中轨斜率>0；下降对称）
- B：价格在 VWAP 上且 SMA20>SMA60（对称）
- C：两个低点抬高且两个高点抬高（局部极值滞后 w 根确认）
- D：ADX(14)>阈值 且 +DI>−DI（对称）
- F：布林上/中/下轨斜率同号且平行（std/|均值| ≤ parallel_max）、|中轨斜率| ≥ slope_min、
     原始状态连续 ≥ min_len 根、期间方向化最大回撤 ≤ max_dd
质量特征（对每个定义 X 的当前连续段计算，列名 chan_X_*）：
- chan_X_len：当前状态已持续根数（状态为 0 时 0）
- chan_X_slope：中轨斜率（bb_mid_slope，方向化：乘以状态符号）
- chan_X_parallel：三线斜率 std / |均值|
- chan_X_maxdd：通道开始以来的方向化最大回撤（相对通道内极值，>0 为回撤幅度）
F 的 len/maxdd 用其原始状态（未加 min_len/max_dd 过滤）的连续段计算，然后用阈值得到最终 chan_F。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.features import indicators as ind
from zr.research.events import _day
from zr.research.scan import trend_A, trend_B, trend_C, trend_D

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "channel.yaml"
CHANNEL_DEFS = ("A", "B", "C", "D", "F")


def load_channel_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _run_length(state: pd.Series) -> pd.Series:
    """当前非零状态连续根数（状态变化或跨日重置）。"""
    day = _day(state.index)
    s = state.to_numpy()
    d = day.to_numpy()
    out = np.zeros(len(s), dtype=int)
    for i in range(len(s)):
        if s[i] == 0:
            continue
        if i > 0 and s[i] == s[i - 1] and d[i] == d[i - 1]:
            out[i] = out[i - 1] + 1
        else:
            out[i] = 1
    return pd.Series(out, index=state.index)


def _max_drawdown_in_run(close: pd.Series, state: pd.Series, run_len: pd.Series) -> pd.Series:
    """通道开始以来的方向化最大回撤：上升通道 = (段内最高 − 当前) / 段内最高；下降对称。"""
    c = close.to_numpy()
    s = state.to_numpy()
    rl = run_len.to_numpy()
    out = np.zeros(len(c))
    ext = np.nan  # 段内极值
    for i in range(len(c)):
        if rl[i] == 0:
            ext = np.nan
            continue
        if rl[i] == 1:
            ext = c[i]
        if s[i] == 1:
            ext = max(ext, c[i])
            out[i] = (ext - c[i]) / ext if ext > 0 else 0.0
        else:
            ext = min(ext, c[i])
            out[i] = (c[i] - ext) / ext if ext > 0 else 0.0
    return pd.Series(out, index=close.index)


def three_line_slopes(feats: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = feats["bb_mid"]
    band = feats["bb_width"] * mid
    upper, lower = mid + band / 2, mid - band / 2
    return (
        ind.bar_slope(upper, n) / mid,
        ind.bar_slope(mid, n) / mid,
        ind.bar_slope(lower, n) / mid,
    )


def parallelism(su: pd.Series, sm: pd.Series, sl: pd.Series) -> pd.Series:
    stack = pd.concat([su, sm, sl], axis=1)
    return stack.std(axis=1, ddof=0) / stack.mean(axis=1).abs().replace(0, np.nan)


def channel_states(df: pd.DataFrame, feats: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """返回 chan_A..chan_F 状态与 chan_X_len/slope/parallel/maxdd 质量特征。"""
    cfg = cfg or load_channel_config()
    n = cfg.get("slope_n", 5)
    su, sm, sl = three_line_slopes(feats, n)
    par = parallelism(su, sm, sl)
    out = pd.DataFrame(index=df.index)
    raw = {
        "A": trend_A(df, feats, {}),
        "B": trend_B(df, feats, {}),
        "C": trend_C(df, feats, {"swing_w": cfg.get("C", {}).get("swing_w", 5)}),
        "D": trend_D(df, feats, {"adx_min": cfg.get("D", {}).get("adx_min", 25)}),
    }
    f = cfg.get("F", {})
    same_sign = (np.sign(su) == np.sign(sm)) & (np.sign(sm) == np.sign(sl)) & (sm != 0)
    f_raw_ok = (
        same_sign & (par <= f.get("parallel_max", 0.5)) & (sm.abs() >= f.get("slope_min", 5e-5))
    )
    raw["F"] = pd.Series(np.where(f_raw_ok, np.sign(sm), 0).astype(int), index=df.index)
    for k, st in raw.items():
        rl = _run_length(st)
        dd = _max_drawdown_in_run(df["close"], st, rl)
        if k == "F":
            final = st.where((rl >= f.get("min_len", 10)) & (dd <= f.get("max_dd", 0.005)), 0)
        else:
            final = st
        out[f"chan_{k}"] = final.astype(int)
        out[f"chan_{k}_len"] = rl
        out[f"chan_{k}_slope"] = sm * st  # 方向化
        out[f"chan_{k}_parallel"] = par
        out[f"chan_{k}_maxdd"] = dd
    return out


CHANNEL_FEATURE_COLUMNS = [
    c
    for k in CHANNEL_DEFS
    for c in (
        f"chan_{k}",
        f"chan_{k}_len",
        f"chan_{k}_slope",
        f"chan_{k}_parallel",
        f"chan_{k}_maxdd",
    )
]
