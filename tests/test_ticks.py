from decimal import Decimal as D

import pytest

from zr.execution import ticks
from zr.execution.ticks import round_to_tick, set_penny_list, tick_size, ticks_away


@pytest.fixture(autouse=True)
def _reset_penny_list():
    saved = set(ticks._penny)
    yield
    set_penny_list(saved)


def test_full_penny_etfs_are_001_at_any_price():
    for sym in ("SPY261021C774000.US", "QQQ261022P725000.US", "IWM261021C230000.US"):
        assert tick_size(sym, 0.5) == D("0.01")
        assert tick_size(sym, 12.0) == D("0.01")


def test_penny_list_threshold_at_3_dollars():
    assert tick_size("MU261021C920000.US", 2.99) == D("0.01")
    assert tick_size("MU261021C920000.US", 3.00) == D("0.05")
    assert tick_size("MU261021C920000.US", 25.0) == D("0.05")


def test_non_penny_names():
    set_penny_list({"SPY", "QQQ", "IWM"})
    assert tick_size("MU261021C920000.US", 1.0) == D("0.05")
    assert tick_size("MU261021C920000.US", 3.0) == D("0.10")


def test_round_and_offset():
    assert round_to_tick("MU261021C920000.US", 1.234) == D("1.23")
    assert round_to_tick("MU261021C920000.US", 1.235, "up") == D("1.24")
    assert round_to_tick("MU261021C920000.US", 3.12, "down") == D("3.10")
    assert ticks_away("SPY261021C774000.US", 0.81, 1) == D("0.82")
    assert ticks_away("MU261021C920000.US", 3.10, -2) == D("3.00")


def test_negative_premium_rejected():
    with pytest.raises(ValueError):
        tick_size("SPY261021C774000.US", -0.01)
