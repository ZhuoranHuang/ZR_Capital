"""zr/research/popular_confluence.py：费用公式、Pine 口径指标、回测引擎的合成数据检验（不连网）。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from zr.costs.fees import stock_order_fees
from zr.research import popular_confluence as pc


@pytest.mark.parametrize("shares", [1, 3, 17, 150, 1200, 60000])
@pytest.mark.parametrize("px", [0.5, 12.34, 88.88, 456.7, 912.05])
@pytest.mark.parametrize("side", ["Buy", "Sell"])
def test_fast_fee_matches_d013(shares, px, side):
    assert pc.stock_fee(shares, side, px) == pytest.approx(
        float(stock_order_fees(shares, side, px)["total"]), abs=1e-9
    )


def test_linreg_on_line_returns_last_value():
    x = pd.Series(3.0 + 0.5 * np.arange(40))
    r = pc.linreg(x, 20)
    assert r.iloc[:19].isna().all()
    assert np.allclose(r.iloc[19:], x.iloc[19:])


def test_linreg_noise_matches_polyfit():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=60).cumsum())
    r = pc.linreg(x, 20)
    w = x.iloc[-20:].to_numpy()
    b, a = np.polyfit(np.arange(20), w, 1)
    assert r.iloc[-1] == pytest.approx(a + b * 19)


def _ohlc(c):
    c = pd.Series(np.asarray(c, float))
    return c + 0.5, c - 0.5, c


def test_supertrend_flips_with_trend():
    c = np.r_[np.linspace(100, 150, 60), np.linspace(150, 90, 60)]
    h, lo, cc = _ohlc(c)
    st = pc.supertrend(h, lo, cc, 10, 3.0)
    assert st[55] == 1  # 上升段
    assert st[-1] == -1  # 下降段
    assert np.isnan(st[:9]).all()


def test_mfi_bounds_and_monotone_case():
    c = np.linspace(10, 30, 40)
    h, lo, cc = _ohlc(c)
    v = pd.Series(np.full(40, 1000.0))
    m = pc.mfi(h, lo, cc, v, 14)
    assert m.dropna().between(0, 100).all()
    assert m.iloc[-1] == pytest.approx(100.0)  # 只涨不跌：负资金流为 0 → NaN 分母，按 100


def test_wavetrend_shapes():
    rng = np.random.default_rng(1)
    c = 100 + rng.normal(size=300).cumsum()
    h, lo, cc = _ohlc(c)
    wt1, wt2 = pc.wavetrend(h, lo, cc)
    assert len(wt1) == len(wt2) == 300
    assert np.isfinite(wt2.iloc[-1])
    assert wt2.iloc[-1] == pytest.approx(wt1.iloc[-4:].mean())


def test_cross_helpers():
    a = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    assert pc._xup(a, 2.5).tolist() == [False, False, True, False, False]
    assert pc._xdn(a, 2.5).tolist() == [False, False, False, True, False]


def _frame(opens, closes):
    n = len(opens)
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    return pc.Frame(
        np.asarray(opens, float),
        np.asarray(closes, float),
        np.asarray(days),
        np.arange(n, dtype=np.int64),
        np.full(n, np.nan),
        np.ones(n, bool),
        False,
    )


def test_simulate_next_open_fill_and_fees():
    f = _frame([10, 10, 11, 12, 12], [10, 10.5, 11.5, 12, 12])
    z = np.zeros(5, bool)
    sig = {
        "long": np.array([True, False, False, False, False]),
        "short": z,
        "exit_long": np.array([False, False, True, False, False]),
        "exit_short": z,
    }
    r = pc.simulate(f, sig, 0, 4, 1000.0, fees=False)
    t = r["trades"][0]
    # 第 0 根收盘信号 → 第 1 根开盘 10 + 0.01 买入；第 2 根收盘出场 → 第 3 根开盘 12 − 0.01 卖出
    n = int(1000 // 10.01)
    assert (t["i0"], t["i1"], t["dir"]) == (1, 3, 1)
    assert r["final"] == pytest.approx(1000 + n * (11.99 - 10.01))
    rf = pc.simulate(f, sig, 0, 4, 1000.0, fees=True)
    fee = pc.stock_fee(n, "Buy", 10.01) + pc.stock_fee(n, "Sell", 11.99)
    assert rf["final"] == pytest.approx(r["final"] - fee)


def test_simulate_short_wipeout_stops():
    f = _frame([10, 10, 30, 30], [10, 10, 30, 30])
    z = np.zeros(4, bool)
    sig = {
        "long": z,
        "short": np.array([True, False, False, False]),
        "exit_long": z,
        "exit_short": z,
    }
    r = pc.simulate(f, sig, 0, 3, 1000.0, fees=False)
    assert r["final"] == 0.0 and r["wiped"]


def test_long_only_ignores_shorts():
    f = _frame([10] * 5, [10] * 5)
    z = np.zeros(5, bool)
    sig = {"long": z, "short": np.ones(5, bool), "exit_long": z, "exit_short": z}
    assert pc.simulate(f, sig, 0, 4, 1000.0, long_only=True)["trades"] == []


def test_decode_respects_bounds():
    lo = [a for _, a, _, _ in pc.MACD_SPACE]
    p = pc.decode(lo)
    assert p["slow"] >= p["fast"] + 2
    assert p["use_ema"] is False
