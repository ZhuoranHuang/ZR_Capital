import datetime as dt

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.features.features import compute_features
from zr.research import events as ev
from zr.research.labels import clustered_t, response_curve, rolling_sigma, triple_barrier


def _day(day: dt.date, close: np.ndarray, vol: int = 500) -> pd.DataFrame:
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    idx = pd.date_range(start, periods=len(close), freq="min", name="ts")
    return pd.DataFrame(
        {"open": close, "high": close + 0.02, "low": close - 0.02, "close": close, "volume": vol},
        index=idx,
    )


def test_e2_breakout_fires_once_per_day_after_range():
    close = np.full(390, 100.0)
    close[31:] = 100.5  # 10:01 突破开盘区间
    df = _day(dt.date(2026, 10, 19), close)
    f = compute_features(df)
    sig = ev.e2_opening_range_breakout(df, f, {"range_min": 30})
    hits = sig[sig != 0]
    assert len(hits) == 1 and hits.iloc[0] == 1
    assert hits.index[0].tz_convert(ET).time() == dt.time(10, 1)


def test_e5_cusum_triggers_on_jump_and_resets():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.0002, 390)
    r[200] = 0.01  # 一次跳升
    close = 100 * np.cumprod(1 + r)
    df = _day(dt.date(2026, 10, 19), close)
    f = compute_features(df)
    sig = ev.e5_cusum(df, f, {"k": 3.0, "sigma_window": 60})
    hits = sig[sig != 0]
    assert 1 in hits.to_numpy()
    assert sig.iloc[200] == 1


def test_e7_fires_at_1530_with_direction():
    close = np.linspace(100, 101, 390)  # 全天上行
    df = _day(dt.date(2026, 10, 19), close)
    f = compute_features(df)
    sig = ev.e7_intraday_momentum(df, f, {"entry_minute": 930})
    hits = sig[sig != 0]
    assert len(hits) == 1 and hits.iloc[0] == 1
    assert hits.index[0].tz_convert(ET).time() == dt.time(15, 30)


def test_detect_events_shape_and_config():
    cfg = ev.load_config()
    assert set(cfg["detectors"]) == {f"E{i}" for i in range(1, 9)}
    close = np.linspace(100, 101, 390)
    df = _day(dt.date(2026, 10, 19), close)
    out = ev.detect_events(df, compute_features(df), "SPY.US", cfg)
    assert list(out.columns) == ["ts", "symbol", "direction", "detector"]
    assert set(out["direction"].unique()) <= {1, -1}


def test_triple_barrier_hits_upper_then_timeout():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.0003, 390)
    close = 100 * np.cumprod(1 + r)
    close[101:111] = close[100] * 1.01  # 事件后立刻 +1%
    df = _day(dt.date(2026, 10, 19), close)
    sigma = rolling_sigma(df["close"], 60)
    events = pd.DataFrame({"ts": [df.index[100], df.index[300]], "direction": [1, -1]})
    lab = triple_barrier(df["close"], events, horizon=10, k=2.0, sigma=sigma)
    assert lab.loc[0, "label"] == 1 and lab.loc[0, "exit_min"] == 1
    assert lab.loc[1, "label"] in (0, 1, -1)
    assert (lab["H"] == 10).all()
    # 跨日截断：最后一根 bar 的事件没有路径 → label 0 且 ret NaN
    tail = triple_barrier(
        df["close"], pd.DataFrame({"ts": [df.index[-1]], "direction": [1]}), 5, 2.0, sigma
    )
    assert tail.loc[0, "label"] == 0 and np.isnan(tail.loc[0, "ret_signed"])


def test_response_curve_and_clustered_t():
    close = np.linspace(100, 110, 390)
    df = _day(dt.date(2026, 10, 19), close)
    events = pd.DataFrame({"ts": df.index[[10, 50, 90]], "direction": [1, 1, -1]})
    rc = response_curve(df["close"], events, horizons=(1, 5, 30))
    assert list(rc["horizon"]) == [1, 5, 30] and (rc["n"] == 3).all()
    assert np.isclose(rc.loc[0, "hit_rate"], 2 / 3)
    t, n = clustered_t(pd.Series([0.1, 0.2, 0.3, 0.4]), pd.Series(["a", "a", "b", "b"]))
    assert n == 2 and t > 0
