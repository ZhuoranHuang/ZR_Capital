"""离场信号库（METHOD_V3 第 4 节）。在周期 P 的 K 线收盘时刻判定；任一触发即出，X6（权利金止盈 /
止损）
与 X7（持仓上限）在 zr/research/sim_v3.py 中按 1 分钟期权快照处理，始终叠加。做空全部镜像。
报告中用「柱变长 / 变短、正 / 负」描述 MACD 柱，不用颜色。

离场（多头；空头镜像）：
- X1 收盘 < MA{5,10}（SMA / EMA）
- X2 MACD：cross = DIF 下穿 DEA；shrink_k = 柱连续 k 根变短（柱值逐根下降）；flip = 柱由正变负
- X3 mid1 = 收盘 < 中轨；mid2 = 连续 2 根收盘 < 中轨
- X4 back_inside（点火专用）= %b 回到 < 0.7（空头 > 0.3）
- X5 higher_low = 收盘 < 当日最近一个「更高低点」（摆动低点左右各 2 根确认，且高于前一个摆动低点）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.features import indicators as ind
from zr.features.bars import ET
from zr.features.regime import _consec


def _day(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df.index.tz_convert(ET).date)


def exit_rules(cfg: dict, entry: str) -> list[str]:
    e = cfg["exits"]
    rules = ["none"] if e.get("include_none") else []
    rules += [f"X1_{k}{n}" for k in e["X1"]["kind"] for n in e["X1"]["ma"]]
    rules += [f"X2_{r}" for r in e["X2"]]
    rules += [f"X3_{r}" for r in e["X3"]]
    if entry == "E10":
        rules += [f"X4_{r}" for r in e["X4"]]
    rules += [f"X5_{r}" for r in e["X5"]]
    return rules


def _swing_level(df: pd.DataFrame, long: bool, w: int = 2) -> np.ndarray:
    """当日最近一个「更高低点」（多头）/「更低高点」（空头）水位，确认滞后 w 根；无则 NaN。"""
    v = df["low" if long else "high"].to_numpy(float)
    day = _day(df)
    n = len(v)
    level = np.full(n, np.nan)
    last = prev = cur = np.nan
    for i in range(n):
        if i and day[i] != day[i - 1]:
            last = prev = cur = np.nan
        j = i - w
        if j - w >= 0 and day[j - w] == day[i]:
            win = v[j - w : i + 1]
            if (long and v[j] == win.min()) or (not long and v[j] == win.max()):
                prev, last = last, v[j]
                if np.isfinite(prev) and ((long and last > prev) or (not long and last < prev)):
                    cur = last
        level[i] = cur
    return level


def exit_signals(
    df: pd.DataFrame, x: pd.DataFrame, cfg: dict
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{规则: (多头离场, 空头离场)}，均为 bar 收盘时刻的布尔数组。"""
    cl = df["close"]
    c = cl.to_numpy(float)
    day = _day(df)
    same_prev = np.r_[False, day[1:] == day[:-1]]
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n = len(c)
    out["none"] = (np.zeros(n, bool), np.zeros(n, bool))
    for kind in cfg["exits"]["X1"]["kind"]:
        for m in cfg["exits"]["X1"]["ma"]:
            ma = (ind.sma(cl, m) if kind == "sma" else ind.ema(cl, m)).to_numpy()
            out[f"X1_{kind}{m}"] = (c < ma, c > ma)
    dif, dea, hist = x["dif"].to_numpy(), x["dea"].to_numpy(), x["hist"].to_numpy()
    with np.errstate(invalid="ignore"):
        dn_cross = np.r_[False, (dif[1:] < dea[1:]) & (dif[:-1] >= dea[:-1])]
        up_cross = np.r_[False, (dif[1:] > dea[1:]) & (dif[:-1] <= dea[:-1])]
        falling = _consec(np.r_[False, hist[1:] < hist[:-1]])
        rising = _consec(np.r_[False, hist[1:] > hist[:-1]])
        flip_dn = np.r_[False, (hist[1:] < 0) & (hist[:-1] >= 0)]
        flip_up = np.r_[False, (hist[1:] > 0) & (hist[:-1] <= 0)]
    for r in cfg["exits"]["X2"]:
        if r == "cross":
            out["X2_cross"] = (dn_cross, up_cross)
        elif r.startswith("shrink"):
            k = int(r[-1])
            out[f"X2_{r}"] = (falling >= k, rising >= k)
        elif r == "flip":
            out["X2_flip"] = (flip_dn, flip_up)
    mid = x["mid"].to_numpy()
    with np.errstate(invalid="ignore"):
        below, above = c < mid, c > mid
    out["X3_mid1"] = (below, above)
    out["X3_mid2"] = (
        below & np.r_[False, below[:-1]] & same_prev,
        above & np.r_[False, above[:-1]] & same_prev,
    )
    pctb = x["pctb"].to_numpy()
    with np.errstate(invalid="ignore"):
        out["X4_back_inside"] = (pctb < 0.7, pctb > 0.3)
    lv_long, lv_short = _swing_level(df, True), _swing_level(df, False)
    with np.errstate(invalid="ignore"):
        out["X5_higher_low"] = (c < lv_long, c > lv_short)
    return out
