"""v3 状态层 / 入场 / 离场：无前视、去重、规则集合。"""

import numpy as np
import pandas as pd

from zr.features.regime import indicators, load_config, range_score, regimes, state_features
from zr.research import exits as ex
from zr.research import patterns_v3 as pt
from zr.research.grid_v3 import dedup_names, trial_counts

CFG = load_config()


def _bars(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-03-04 09:30", periods=n, freq="2min", tz="America/New_York")
    idx = idx[(idx.hour * 60 + idx.minute >= 570) & (idx.hour * 60 + idx.minute < 960)]
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.001, len(idx)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.001, len(idx)))
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": rng.uniform(1e3, 5e3, len(idx))},
        index=idx.tz_convert("UTC"),
    )


def _all(df):
    x = indicators(df, CFG)
    reg = regimes(df, x, CFG, 2)
    sig = {pt.entry_name(c): pt.entry_signal(df, x, reg, c, CFG) for c in pt.entry_configs(CFG)}
    exs = ex.exit_signals(df, x, CFG)
    return x, reg, sig, exs, state_features(df, x, reg)


def test_no_lookahead_regime_entries_exits():
    df = _bars()
    cut = 300
    a = _all(df)
    df2 = df.copy()
    df2.iloc[cut:, :4] *= 1.03
    df2.iloc[cut:, 4] *= 5
    b = _all(df2)
    pd.testing.assert_frame_equal(a[1].iloc[:cut], b[1].iloc[:cut])
    for k in a[2]:
        assert (a[2][k][:cut] == b[2][k][:cut]).all(), k
    for k in a[3]:
        for side in (0, 1):
            assert (a[3][k][side][:cut] == b[3][k][side][:cut]).all(), k
    pd.testing.assert_frame_equal(a[4].iloc[:cut], b[4].iloc[:cut])


def test_dedup_and_trial_counts():
    d = dedup_names(CFG)
    assert len(set(d.values())) == 34 and len(d) == 42
    tc = trial_counts(CFG)
    assert tc["rows_total"] == tc["rows_per_period"] * (len(CFG["periods"]) + 1)


def test_exit_rule_sets():
    r10, r1 = ex.exit_rules(CFG, "E10"), ex.exit_rules(CFG, "E1")
    assert "X4_back_inside" in r10 and "X4_back_inside" not in r1
    assert r1[0] == "none" and len(r1) == 13 and len(r10) == 14


def test_bars_per_day_matches_actual_rth_count():
    df = _bars(600)
    from zr.features.regime import bars_per_day

    # 600 根 2 分钟 bar，09:30-16:00 一天最多 195 根（390/2），超出的落到后续「日」
    assert bars_per_day(df) <= 195


def test_range_score_lower_when_flat_and_higher_when_trending():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-03-04 09:30", periods=600, freq="2min", tz="America/New_York")
    idx = idx[(idx.hour * 60 + idx.minute >= 570) & (idx.hour * 60 + idx.minute < 960)]
    flat_close = 100 + rng.normal(0, 0.01, len(idx))
    trend_close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.003, len(idx))))
    for close, label in ((flat_close, "flat"), (trend_close, "trend")):
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": 1000.0,
            },
            index=idx.tz_convert("UTC"),
        )
        x = indicators(df, CFG)
        s = range_score(x, df, CFG["regime"]["range"], 2)
        globals().setdefault("_scores", {})[label] = s.dropna().median()
    assert globals()["_scores"]["flat"] < globals()["_scores"]["trend"]


def test_range_score_no_lookahead():
    df = _bars(400)
    x = indicators(df, CFG)
    s1 = range_score(x, df, CFG["regime"]["range"], 2)
    df2 = df.copy()
    df2.iloc[300:, :4] *= 1.2
    x2 = indicators(df2, CFG)
    s2 = range_score(x2, df2, CFG["regime"]["range"], 2)
    pd.testing.assert_series_equal(s1.iloc[:300], s2.iloc[:300])
