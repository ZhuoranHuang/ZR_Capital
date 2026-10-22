"""入场形态库（METHOD_V3 第 3 节）；离场信号库见 zr/research/exits.py。

全部在周期 P 的 K 线上计算，信号在 bar 收盘确认（收盘时刻 = bar 标签 + P 分钟），只用 ≤t 的数据。
做空全部镜像。返回 int8 数组：入场 +1 多 / −1 空 / 0；离场按方向分两条布尔数组（多头离场 /
空头离场）。

入场：
- E10 点火：状态 IGNITION（方向 d）中，连续第 k 根收盘在外轨之外（多 %b > 1，空 %b < 0），k ∈ {2,
  3}。
- E1 回踩：通道 chan_N_M = d（N、M 为第 2 节的通道参数）；信号 bar 为反转 K（多：收阳，
  且前一根收阴），
  回踩极值 L = min(low[t−1], low[t])（空：H = max(high)），L 的 %b ∈ [a, b] 且 |L − 中轨| ≤ c ×
  带半宽。
  注：|L − 中轨| ≤ c × 半宽 等价于 %b ∈ [0.5 − c/2, 0.5 + c/2]，两个条件取交集。
- E11 突破：前一根状态 RANGE，收盘突破前 M 根最高（最低）点，且成交量 ÷ 前 vol_window 根均量 ≥ v。
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.features.regime import IGNITION, RANGE, _consec


def _day(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df.index.tz_convert(ET).date)


def entry_configs(cfg: dict) -> list[dict]:
    e = cfg["entries"]
    ch = cfg["regime"]["channel"]
    out = [{"entry": "E10", "k": k} for k in e["E10"]["k"]]
    for (a, b), c, n, m in product(
        e["E1"]["pctb_ranges"], e["E1"]["mid_dist_hw"], ch["slope_n"], ch["side_m"]
    ):
        lo, hi = max(a, 0.5 - c / 2), min(b, 0.5 + c / 2)
        out.append({"entry": "E1", "pctb": (a, b), "c": c, "N": n, "M": m, "eff": (lo, hi)})
    for m, v in product(e["E11"]["lookback_m"], e["E11"]["vol_ratio"]):
        out.append({"entry": "E11", "M": m, "v": v})
    return out


def entry_name(c: dict) -> str:
    if c["entry"] == "E10":
        return f"E10|k{c['k']}"
    if c["entry"] == "E1":
        return f"E1|b{c['pctb'][0]}-{c['pctb'][1]}|c{c['c']}|N{c['N']}M{c['M']}"
    return f"E11|M{c['M']}|v{c['v']}"


def entry_signal(df: pd.DataFrame, x: pd.DataFrame, reg: pd.DataFrame, c: dict, cfg: dict):
    o, h, lo, cl = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pctb = x["pctb"].to_numpy()
    day = _day(df)
    same_prev = np.r_[False, day[1:] == day[:-1]]
    sig = np.zeros(len(df), dtype=np.int8)
    if c["entry"] == "E10":
        st, d = reg["state"].to_numpy(), reg["dir"].to_numpy()
        with np.errstate(invalid="ignore"):
            up_run, dn_run = _consec(pctb > 1), _consec(pctb < 0)
        ign = st == IGNITION
        sig[ign & (d == 1) & (up_run == c["k"])] = 1
        sig[ign & (d == -1) & (dn_run == c["k"])] = -1
        return sig
    if c["entry"] == "E1":
        chan = reg[f"chan_{c['N']}_{c['M']}"].to_numpy()
        mid, hw, lower = x["mid"].to_numpy(), x["hw"].to_numpy(), x["lower"].to_numpy()
        a, b = c["pctb"]
        prev_red = np.r_[False, cl[:-1] < o[:-1]] & same_prev
        prev_green = np.r_[False, cl[:-1] > o[:-1]] & same_prev
        L = np.minimum(np.r_[np.nan, lo[:-1]], lo)
        H = np.maximum(np.r_[np.nan, h[:-1]], h)
        with np.errstate(invalid="ignore", divide="ignore"):
            pl = (L - lower) / (2 * hw)
            ph = (H - lower) / (2 * hw)
            long = (
                (chan == 1)
                & prev_red
                & (cl > o)
                & (pl >= a)
                & (pl <= b)
                & (np.abs(L - mid) <= c["c"] * hw)
            )
            # 空头镜像：反弹高点 H 的「1 − %b」∈ [a, b]
            short = (
                (chan == -1)
                & prev_green
                & (cl < o)
                & (1 - ph >= a)
                & (1 - ph <= b)
                & (np.abs(H - mid) <= c["c"] * hw)
            )
        sig[long], sig[short] = 1, -1
        return sig
    # E11
    st = reg["state"].to_numpy()
    m, v, vw = c["M"], c["v"], cfg["entries"]["E11"]["vol_window"]
    hh = pd.Series(h).shift(1).rolling(m, min_periods=m).max().to_numpy()
    ll = pd.Series(lo).shift(1).rolling(m, min_periods=m).min().to_numpy()
    vol = df["volume"].to_numpy(float)
    vr = vol / pd.Series(vol).shift(1).rolling(vw, min_periods=vw).mean().to_numpy()
    prev_range = np.r_[False, st[:-1] == RANGE] & same_prev
    with np.errstate(invalid="ignore"):
        sig[prev_range & (cl > hh) & (vr >= v)] = 1
        sig[prev_range & (cl < ll) & (vr >= v)] = -1
    return sig
