import datetime as dt

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.features.channel import (
    CHANNEL_FEATURE_COLUMNS,
    _max_drawdown_in_run,
    _run_length,
    channel_states,
)
from zr.features.features import compute_features


def _day(day: dt.date, close: np.ndarray) -> pd.DataFrame:
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    idx = pd.date_range(start, periods=len(close), freq="min", name="ts")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": 500,
        },
        index=idx,
    )


def test_run_length_resets_on_state_change_and_day():
    d1 = _day(dt.date(2026, 10, 19), np.full(4, 100.0))
    d2 = _day(dt.date(2026, 10, 20), np.full(2, 100.0))
    idx = pd.concat([d1, d2]).index
    st = pd.Series([1, 1, -1, -1, -1, -1], index=idx)
    assert _run_length(st).tolist() == [1, 2, 1, 2, 1, 2]


def test_max_drawdown_in_run_directional():
    d = _day(dt.date(2026, 10, 19), np.array([100.0, 102.0, 101.0, 103.0, 100.0]))
    st = pd.Series([1, 1, 1, 1, 1], index=d.index)
    rl = _run_length(st)
    dd = _max_drawdown_in_run(d["close"], st, rl).round(4).tolist()
    assert dd == [0.0, 0.0, round(1 / 102, 4), 0.0, round(3 / 103, 4)]


def test_channel_states_columns_and_F_requires_persistence():
    rng = np.random.default_rng(0)
    # 前半天稳步上行（低噪音），后半天震荡
    up = 100 * np.cumprod(
        1 + np.r_[np.full(200, 0.0004) + rng.normal(0, 0.00005, 200), rng.normal(0, 0.001, 190)]
    )
    df = _day(dt.date(2026, 10, 19), up)
    f = compute_features(df)
    cs = channel_states(df, f)
    assert list(cs.columns) == CHANNEL_FEATURE_COLUMNS
    assert set(cs["chan_F"].unique()) <= {-1, 0, 1}
    # 上行段应出现 F=+1 且持续 ≥10 根；F 为 1 的地方 len ≥ 10、maxdd ≤ 0.5%
    on = cs[cs["chan_F"] == 1]
    assert len(on) > 10
    assert (on["chan_F_len"] >= 10).all() and (on["chan_F_maxdd"] <= 0.005).all()
    # 无前视：改最后 5 根不影响之前
    bumped = df.copy()
    bumped.iloc[-5:, bumped.columns.get_loc("close")] *= 1.05
    cs2 = channel_states(bumped, compute_features(bumped))
    pd.testing.assert_frame_equal(cs.iloc[:-5], cs2.iloc[:-5])
