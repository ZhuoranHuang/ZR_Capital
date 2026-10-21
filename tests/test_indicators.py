import numpy as np
import pandas as pd
import pytest

from zr.features import indicators as ind


@pytest.fixture
def ohlc() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="min", tz="America/New_York")
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=idx)


def test_ema_recursion():
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    n = 3
    a = 2 / (n + 1)
    expected = [s.iloc[0]]
    for x in s.iloc[1:]:
        expected.append(a * x + (1 - a) * expected[-1])
    np.testing.assert_allclose(ind.ema(s, n).to_numpy(), expected)


def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1.0, 31.0))
    rsi = ind.wilder_rsi(s, 14)
    assert rsi.iloc[:14].isna().all()
    assert (rsi.iloc[14:] == 100.0).all()


def test_rsi_all_losses_is_0():
    s = pd.Series(np.arange(30.0, 0.0, -1.0))
    rsi = ind.wilder_rsi(s, 14)
    np.testing.assert_allclose(rsi.iloc[14:].to_numpy(), 0.0)


def test_rsi_bounded(ohlc):
    rsi = ind.wilder_rsi(ohlc["close"], 14).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_cross_signal_events():
    fast = pd.Series([np.nan, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0])
    slow = pd.Series([np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert ind.cross_signal(fast, slow).tolist() == [0, 0, 0, 1, 0, -1, 0]


def test_cross_signal_no_event_when_line_first_appears():
    fast = pd.Series([3.0, 3.0, 3.0])
    slow = pd.Series([np.nan, 2.0, 2.0])
    assert ind.cross_signal(fast, slow).tolist() == [0, 0, 0]


def test_macd_identity(ohlc):
    m = ind.macd(ohlc["close"])
    c = ohlc["close"]
    np.testing.assert_allclose(m["dif"], ind.ema(c, 12) - ind.ema(c, 26))
    np.testing.assert_allclose(m["dea"], ind.ema(m["dif"], 9))
    np.testing.assert_allclose(m["hist"], m["dif"] - m["dea"])


def test_bollinger_band_order(ohlc):
    bb = ind.bollinger(ohlc["close"], 20, 2.0).dropna()
    assert len(bb) > 0
    assert (bb["upper"] >= bb["mid"]).all()
    assert (bb["mid"] >= bb["lower"]).all()


def test_atr_no_lookahead(ohlc):
    base = ind.atr(ohlc["high"], ohlc["low"], ohlc["close"], 14)
    bumped = ohlc.copy()
    bumped.iloc[-1] = bumped.iloc[-1] * 10
    after = ind.atr(bumped["high"], bumped["low"], bumped["close"], 14)
    pd.testing.assert_series_equal(base.iloc[:-1], after.iloc[:-1])
    assert base.iloc[-1] != after.iloc[-1]


def test_session_vwap_resets_each_day():
    idx = pd.DatetimeIndex(
        [
            "2026-01-05 15:58",
            "2026-01-05 15:59",
            "2026-01-06 09:30",
            "2026-01-06 09:31",
        ],
        tz="America/New_York",
    )
    price = pd.Series([10.0, 20.0, 100.0, 200.0], index=idx)
    volume = pd.Series([1.0, 3.0, 2.0, 2.0], index=idx)
    vwap = ind.session_vwap(price, price, price, volume)
    np.testing.assert_allclose(vwap.to_numpy(), [10.0, 17.5, 100.0, 150.0])


def test_bar_slope_linear():
    s = pd.Series(3.0 + 2.0 * np.arange(10))
    slope = ind.bar_slope(s, 5)
    assert slope.iloc[:4].isna().all()
    np.testing.assert_allclose(slope.iloc[4:].to_numpy(), 2.0)
