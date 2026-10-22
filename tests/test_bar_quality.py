import datetime as dt

import numpy as np
import pandas as pd

from zr.data.bar_quality import clean_wicks, day_ranges, flag_bars
from zr.features.bars import ET


def _days(n_days: int = 25, n: int = 390, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    d0 = dt.date(2026, 9, 1)
    k = 0
    while len(frames) < n_days:
        day = d0 + dt.timedelta(days=k)
        k += 1
        if day.weekday() >= 5:
            continue
        start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
        idx = pd.date_range(start, periods=n, freq="min", name="ts")
        close = px * np.cumprod(1 + rng.normal(0, 0.0005, n))
        spread = np.full(n, 0.0004)
        spread[0] = 0.003  # 开盘分钟天然振幅大
        frames.append(
            pd.DataFrame(
                {
                    "open": close,
                    "high": close * (1 + spread),
                    "low": close * (1 - spread),
                    "close": close,
                    "volume": 1000,
                },
                index=idx,
            )
        )
        px = close[-1]
    return pd.concat(frames)


def test_opening_bar_not_flagged_but_mid_session_spike_is():
    df = _days()
    last_day = df.index.tz_convert(ET).date[-1]
    i = np.where(df.index.tz_convert(ET).date == last_day)[0][200]
    df.iloc[i, df.columns.get_loc("low")] = df["close"].iloc[i] * 0.97  # 坏影线 −3%
    f = flag_bars(df)
    first_minutes = df.index.tz_convert(ET).time == dt.time(9, 30)
    assert not f.loc[first_minutes, "wick_spike"].iloc[6:].any()  # 有基线后开盘分钟不误报
    assert f["wick_spike"].iloc[i] and f["bar_range_gt_5atr"].iloc[i]
    c = clean_wicks(df, f)
    assert c["low"].iloc[i] == min(df["open"].iloc[i], df["close"].iloc[i])
    assert c["low"].iloc[i - 1] == df["low"].iloc[i - 1]


def test_day_range_outlier_uses_prior_days_only():
    df = _days()
    last_day = df.index.tz_convert(ET).date[-1]
    m = df.index.tz_convert(ET).date == last_day
    df.loc[m, "high"] = df.loc[m, "high"] * 1.05
    d = day_ranges(df)
    assert bool(d["day_range_outlier"].iloc[-1]) is True
    assert not d["day_range_outlier"].iloc[:-1].any()
