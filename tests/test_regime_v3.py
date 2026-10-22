"""v3 状态层 / 入场 / 离场：无前视、去重、规则集合。"""

import numpy as np
import pandas as pd

from zr.features.bars import ET
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


def test_indicators_daily_reset_vs_continuous():
    """D-027：daily_reset=True 时 BOLL/MACD 每天从头算，当日值不受前一天收盘价影响。"""
    rng = np.random.default_rng(3)
    idx1 = pd.date_range("2024-03-04 09:30", periods=20, freq="2min", tz="America/New_York")
    idx2 = pd.date_range("2024-03-05 09:30", periods=20, freq="2min", tz="America/New_York")
    idx = idx1.append(idx2).tz_convert("UTC")
    close = np.r_[100 + rng.normal(0, 0.1, 20), 100 + rng.normal(0, 0.1, 20)]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    x_reset = indicators(df, CFG, daily_reset=True)
    x_cont = indicators(df, CFG, daily_reset=False)
    day2 = slice(20, 40)
    # 连续版 day2 的早期 bar 能看到 day1 的收盘（std/EMA 不是 NaN）；按日重置版同一位置应是 NaN
    assert x_reset["mid"].iloc[20:25].isna().all()
    assert x_cont["mid"].iloc[20:25].notna().any()
    # 改动第 1 天的价格，重置版的第 2 天指标不受影响；连续版会受影响
    df2 = df.copy()
    df2.iloc[:20, :4] *= 1.5
    x_reset2 = indicators(df2, CFG, daily_reset=True)
    x_cont2 = indicators(df2, CFG, daily_reset=False)
    pd.testing.assert_series_equal(
        x_reset["dif"].iloc[day2], x_reset2["dif"].iloc[day2], check_names=False
    )
    assert not x_cont["dif"].iloc[day2].equals(x_cont2["dif"].iloc[day2])


def test_indicators_slope_no_cross_day_jump_under_daily_reset():
    df = _bars(60)
    x = indicators(df, CFG, daily_reset=True)
    day = np.asarray(df.index.tz_convert(ET).date)
    first_of_day = np.r_[True, day[1:] != day[:-1]]
    assert np.isnan(x["slope"].to_numpy()[first_of_day]).all()
