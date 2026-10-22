"""结构上下文特征（METHOD_V2 规格 4）。无前视：第 t 行只用 ≤t 的 1 分钟 bar；5 分钟特征只用
在 t 收盘（= t+1 分钟）前已经完成的 5 分钟 bar。

- pullback_idx：本轮通道（通道 A = MACD 三条件状态连续段）内第几次回踩中轨（|low−中轨|/中轨 ≤ ε，
  下降通道用 high）；不在通道内为 0
- bars_since_macd_cross / bars_since_kdj_cross：距最近一次 DIF/DEA（K/D）交叉的根数（首次交叉前 NaN）
- channel_len_bars / channel_max_dd：通道 A 的持续根数与通道内方向化最大回撤（zr/features/channel.py）
- bb_parallel：布林上中下轨斜率的 std/|均值|
- kdj_k / kdj_d / kdj_j
- tf5_regime：5 分钟 K 上的 MACD 状态（+1 DIF>DEA 且 DIF>0；−1 对称；0 其他）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.features import indicators as ind
from zr.features.bars import resample_bars
from zr.features.channel import channel_states, parallelism, three_line_slopes

EPS = 0.0005

STRUCTURE_COLUMNS = [
    "pullback_idx",
    "bars_since_macd_cross",
    "bars_since_kdj_cross",
    "channel_len_bars",
    "channel_max_dd",
    "bb_parallel",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "tf5_regime",
]


def bars_since(event: pd.Series) -> pd.Series:
    """距最近一次 event=True 的根数（当根为 0）；首次之前 NaN。"""
    e = event.to_numpy(dtype=bool)
    out = np.full(len(e), np.nan)
    last = -1
    for i, v in enumerate(e):
        if v:
            last = i
        if last >= 0:
            out[i] = i - last
    return pd.Series(out, index=event.index)


def _cross(a: pd.Series, b: pd.Series) -> pd.Series:
    up = (a > b) & (a.shift(1) <= b.shift(1))
    dn = (a < b) & (a.shift(1) >= b.shift(1))
    return up | dn


def macd_state(close: pd.Series) -> pd.Series:
    m = ind.macd(close)
    up = (m["dif"] > m["dea"]) & (m["dif"] > 0)
    dn = (m["dif"] < m["dea"]) & (m["dif"] < 0)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=close.index)


def tf5_regime(df: pd.DataFrame) -> pd.Series:
    """5 分钟 bar（标签 T 覆盖 [T, T+5)，在 T+5 完成）上的 MACD 状态，按完成时刻对齐到 1 分钟 bar 的收盘时刻 t+1。"""
    b5 = resample_bars(df, "5min")
    st = macd_state(b5["close"])
    avail = pd.DataFrame({"avail": b5.index + pd.Timedelta(minutes=5), "tf5": st.to_numpy()})
    dec = pd.DataFrame({"t": df.index, "dec": df.index + pd.Timedelta(minutes=1)})
    m = pd.merge_asof(
        dec.sort_values("dec"),
        avail.sort_values("avail"),
        left_on="dec",
        right_on="avail",
        direction="backward",
    )
    return pd.Series(m.sort_values("t")["tf5"].to_numpy(), index=df.index)


def compute_structure(
    df: pd.DataFrame, feats: pd.DataFrame, chans: pd.DataFrame | None = None
) -> pd.DataFrame:
    chans = channel_states(df, feats) if chans is None else chans
    out = pd.DataFrame(index=df.index)
    st = chans["chan_A"]
    mid = feats["bb_mid"]
    touch = np.where(
        st == 1,
        (df["low"] - mid).abs() / mid <= EPS,
        np.where(st == -1, (df["high"] - mid).abs() / mid <= EPS, False),
    )
    run_id = (chans["chan_A_len"] == 1).cumsum()
    cnt = pd.Series(touch.astype(int), index=df.index).groupby(run_id.to_numpy()).cumsum()
    out["pullback_idx"] = np.where(st != 0, cnt, 0)
    m = ind.macd(df["close"])
    out["bars_since_macd_cross"] = bars_since(_cross(m["dif"], m["dea"]))
    k = ind.kdj(df["high"], df["low"], df["close"])
    out["bars_since_kdj_cross"] = bars_since(_cross(k["k"], k["d"]))
    out["channel_len_bars"] = chans["chan_A_len"]
    out["channel_max_dd"] = chans["chan_A_maxdd"]
    su, sm, sl = three_line_slopes(feats, 5)
    out["bb_parallel"] = parallelism(su, sm, sl)
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = k["k"], k["d"], k["j"]
    out["tf5_regime"] = tf5_regime(df)
    return out
