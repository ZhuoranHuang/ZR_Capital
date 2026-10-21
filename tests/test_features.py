import datetime as dt

import numpy as np
import pandas as pd
import pytest

from zr.features.bars import ET
from zr.features.features import FEATURE_COLUMNS, compute_features


def _synthetic(days: int = 30, n: int = 390, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    px = 100.0
    d0 = dt.date(2026, 9, 1)
    for i in range(days):
        day = d0 + dt.timedelta(days=i)
        if day.weekday() >= 5:
            continue
        start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
        idx = pd.date_range(start, periods=n, freq="min", name="ts")
        px = px * (1 + rng.normal(0, 0.002))  # 隔夜跳空
        rets = rng.normal(0, 0.0005, n)
        close = px * np.cumprod(1 + rets)
        high = close * (1 + rng.uniform(0, 0.0005, n))
        low = close * (1 - rng.uniform(0, 0.0005, n))
        open_ = np.r_[px, close[:-1]]
        vol = rng.integers(100, 1000, n)
        frames.append(
            pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
            )
        )
        px = close[-1]
    return pd.concat(frames)


@pytest.fixture(scope="module")
def data():
    df = _synthetic()
    return df, compute_features(df)


def test_all_feature_columns_present(data):
    _, f = data
    missing = [c for c in FEATURE_COLUMNS if c not in f]
    assert not missing


@pytest.mark.parametrize("col", FEATURE_COLUMNS)
def test_no_lookahead(data, col):
    """改最后 60 根 bar 不影响之前任何一行的特征。"""
    df, f = data
    bumped = df.copy()
    bumped.iloc[-60:, bumped.columns.get_loc("close")] *= 1.05
    bumped.iloc[-60:, bumped.columns.get_loc("volume")] *= 10
    f2 = compute_features(bumped)
    a, b = f[col].iloc[:-60], f2[col].iloc[:-60]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_returns_and_time_features(data):
    df, f = data
    day = pd.Series(df.index.tz_convert(ET).date, index=df.index)
    first = day != day.shift(1)
    assert f["ret_1"][first].isna().all()  # 日内收益不跨日
    i = 10
    assert np.isclose(f["ret_5"].iloc[i], df["close"].iloc[i] / df["close"].iloc[i - 5] - 1)
    assert f["min_to_close"].iloc[0] == 390 and f["min_to_close"].iloc[389] == 1
    assert np.isclose(f["ret_since_open"].iloc[3], df["close"].iloc[3] / df["open"].iloc[0] - 1)


def test_gap_and_same_time_features(data):
    df, f = data
    day = pd.Series(df.index.tz_convert(ET).date, index=df.index)
    d2_start = np.where(day != day.shift(1))[0][1]
    prev_close = df["close"].iloc[d2_start - 1]
    assert np.isclose(f["gap_overnight"].iloc[d2_start], df["open"].iloc[d2_start] / prev_close - 1)
    # 昨日同一分钟的 ret_since_open
    assert np.isclose(f["ret_same_time_yday"].iloc[d2_start + 7], f["ret_since_open"].iloc[7])
    assert f["ret_same_time_yday"].iloc[:390].isna().all()


def test_opening_range_and_vol_ratio(data):
    df, f = data
    assert f["or_pos"].iloc[:30].isna().all() and f["or_pos"].iloc[30:390].notna().all()
    or_high = df["high"].iloc[:30].max()
    assert np.isclose(f["or_high"].iloc[100], or_high)
    assert f["vol_ratio"].iloc[:390].isna().all()
    assert f["vol_ratio"].iloc[390 * 6 :].notna().mean() > 0.95
