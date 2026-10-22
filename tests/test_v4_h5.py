"""H5 元标签的机械性测试（合成数据）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.features.regime import CHANNEL, IGNITION, OTHER
from zr.research import v4_h5 as h

H5 = h.load_h5()


def test_transition_features_prev_state_and_ignition_to_channel():
    idx = pd.date_range("2026-10-15 14:00", periods=8, freq="2min", tz="UTC")
    st = [OTHER, IGNITION, IGNITION, OTHER, OTHER, CHANNEL, CHANNEL, OTHER]
    di = [0, 1, 1, 0, 0, 1, 1, 0]
    reg = pd.DataFrame({"state": st, "dir": di}, index=idx)
    b = pd.DataFrame({"close": np.arange(100.0, 108.0)}, index=idx)
    f = h.transition_features(b, reg, reset=False)
    assert np.isnan(f["prev_state"].iat[0])
    assert f["prev_state"].iat[1] == OTHER and f["prev_state"].iat[3] == IGNITION
    assert f["ign_to_chan_bars"].iat[5] == 3 and f["ign_to_chan_bars"].iat[6] == 3  # 末次点火在 2
    assert np.isnan(f["ign_to_chan_bars"].iat[4])
    assert np.isnan(f["ma5_pos"].iat[3]) and f["ma5_pos"].iat[4] > 0


def test_walk_forward_expanding_min_days_and_embargo():
    days = pd.bdate_range("2024-01-01", "2024-06-30").date
    day = pd.Series(np.repeat(days, 2))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(day), 3))
    y = (X[:, 0] + rng.normal(scale=0.5, size=len(day)) > 0).astype(int)
    p, folds = h.walk_forward(X, y, day, H5)
    first = pd.Period(folds[0]["month"])
    train_days = {d for d in days if pd.Timestamp(d) < first.start_time - pd.Timedelta(days=1)}
    assert len(train_days) >= 60
    assert np.isnan(p[np.asarray([d < first.start_time.date() for d in day])]).all()
    assert np.isfinite(p[np.asarray([d >= first.start_time.date() for d in day])]).all()
