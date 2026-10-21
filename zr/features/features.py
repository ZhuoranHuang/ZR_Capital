"""STRATEGY_V1 第 3 层中只依赖正股 1 分钟 K 的特征。

无前视：第 t 行只用 ≤ t 的数据；跨日量只用之前交易日。

输入：`load_bars` 输出（索引 ts UTC，盘中 bar）。输出：同索引的特征 DataFrame。
参数见 `FeatureParams`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from zr.features import indicators as ind

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class FeatureParams:
    macd: tuple[int, int, int] = (12, 26, 9)
    bb_n: int = 20
    bb_k: float = 2.0
    atr_n: int = 14
    slope_n: int = 5
    opening_range_min: int = 30
    ret_windows: tuple[int, ...] = (1, 5, 15, 30)
    rv_windows: tuple[int, ...] = (5, 30)
    same_time_days: int = 5
    vol_ratio_days: int = 20
    extra: dict = field(default_factory=dict)


def _et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(ET)


def _day_key(index: pd.DatetimeIndex) -> np.ndarray:
    return _et(index).date


def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    e = _et(index)
    return (e.hour * 60 + e.minute).to_numpy()


def _within_day_shift_ret(close: pd.Series, n: int, day: np.ndarray) -> pd.Series:
    prev = close.groupby(day).shift(n)
    return close / prev - 1


DEFAULT_PARAMS = FeatureParams()


def compute_features(df: pd.DataFrame, p: FeatureParams = DEFAULT_PARAMS) -> pd.DataFrame:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    idx = df.index
    day = _day_key(idx)
    mod = _minute_of_day(idx)
    out = pd.DataFrame(index=idx)

    atr = ind.atr(high, low, close, p.atr_n).replace(0, np.nan)
    # 趋势
    m = ind.macd(close, *p.macd)
    out["macd_dif_atr"] = m["dif"] / atr
    out["macd_dea_atr"] = m["dea"] / atr
    out["macd_hist_atr"] = m["hist"] / atr
    bb = ind.bollinger(close, p.bb_n, p.bb_k)
    out["bb_pct_b"] = bb["pct_b"]
    out["bb_width"] = bb["width"]
    out["bb_mid_slope"] = ind.bar_slope(bb["mid"], p.slope_n) / bb["mid"]
    out["bb_mid"] = bb["mid"]
    # 位置
    vwap = ind.session_vwap(high, low, close, vol)
    out["vwap"] = vwap
    out["vwap_dist_atr"] = (close - vwap) / atr
    or_mask = mod < 9 * 60 + 30 + p.opening_range_min
    or_high = high.where(or_mask).groupby(day).cummax().groupby(day).ffill()
    or_low = low.where(or_mask).groupby(day).cummin().groupby(day).ffill()
    rng = (or_high - or_low).replace(0, np.nan)
    out["or_high"], out["or_low"] = or_high, or_low
    out["or_pos"] = ((close - or_low) / rng).where(~or_mask)
    # 收益
    for n in p.ret_windows:
        out[f"ret_{n}"] = _within_day_shift_ret(close, n, day)
    day_open = close.groupby(day).transform("first")  # 当日第一根的 close，可换 open
    first_open = df["open"].groupby(day).transform("first")
    out["ret_since_open"] = close / first_open - 1
    prev_close_by_day = close.groupby(day).last().shift(1)
    out["gap_overnight"] = first_open / pd.Series(day, index=idx).map(prev_close_by_day) - 1
    # 同时段历史收益：前 1 日与前 N 日同一分钟的 ret_since_open
    rso = out["ret_since_open"]
    key = pd.MultiIndex.from_arrays([day, mod])
    piv = pd.Series(rso.to_numpy(), index=key).unstack(level=1)  # day × minute
    piv = piv.sort_index()
    yday = piv.shift(1)
    prev_n = piv.shift(1).rolling(p.same_time_days, min_periods=1).mean()
    out["ret_same_time_yday"] = yday.stack().reindex(key).to_numpy()
    out[f"ret_same_time_{p.same_time_days}d"] = prev_n.stack().reindex(key).to_numpy()
    # 时间
    out["min_to_close"] = 16 * 60 - mod
    # 波动
    logret = (
        np.log(close).diff().where(pd.Series(day, index=idx).eq(pd.Series(day, index=idx).shift(1)))
    )
    for n in p.rv_windows:
        out[f"rv_{n}"] = logret.rolling(n, min_periods=n).std(ddof=1)
    rv_prev_day = logret.groupby(day).std(ddof=1).shift(1)
    rv_prev = pd.Series(day, index=idx).map(rv_prev_day)
    for n in p.rv_windows:
        out[f"rv_{n}_vs_day"] = out[f"rv_{n}"] / rv_prev
    out["range_atr"] = (high - low) / atr
    # 量比：同一分钟在前 N 日的均量
    vpiv = pd.Series(vol.to_numpy(), index=key).unstack(level=1).sort_index()
    vbase = vpiv.shift(1).rolling(p.vol_ratio_days, min_periods=5).mean()
    out["vol_ratio"] = vol.to_numpy() / vbase.stack().reindex(key).to_numpy()
    out["atr"] = atr
    out["day_open_close"] = day_open
    return out


FEATURE_COLUMNS = [
    "macd_dif_atr",
    "macd_dea_atr",
    "macd_hist_atr",
    "bb_pct_b",
    "bb_width",
    "bb_mid_slope",
    "vwap_dist_atr",
    "or_pos",
    "ret_1",
    "ret_5",
    "ret_15",
    "ret_30",
    "ret_since_open",
    "gap_overnight",
    "ret_same_time_yday",
    "ret_same_time_5d",
    "min_to_close",
    "rv_5",
    "rv_30",
    "rv_5_vs_day",
    "rv_30_vs_day",
    "range_atr",
    "vol_ratio",
]
