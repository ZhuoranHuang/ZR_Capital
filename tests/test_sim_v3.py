"""v3 回放引擎：费用向量化、成交时刻（无前视）、bid=0、D-001、同时一笔。"""

import datetime as dt

import numpy as np
import pandas as pd

from zr.costs.fees import option_order_fees
from zr.features.regime import load_config
from zr.research import sim_v3 as sv

DAY = dt.date(2025, 6, 10)
CFG = load_config()


def test_sell_fee_matches_exact_formula():
    rng = np.random.default_rng(0)
    for p in np.r_[rng.uniform(0.01, 30, 300), [0.01, 0.05, 4.85, 24.27]]:
        for n in (1, 2):
            exact = float(option_order_fees(n, "Sell", round(p, 2))["total"])
            assert abs(sv.sell_fee(n, round(p, 2)) - exact) < 1e-9
    for n in (1, 2):
        assert sv._BUY[n] == float(option_order_fees(n, "Buy", 3.0)["total"])


def _od(bid: np.ndarray, ask: np.ndarray, expiry=DAY) -> sv.OptDay:
    sym = "X"
    return sv.OptDay(
        DAY,
        [expiry],
        {(expiry, "call"): (np.array([101.0]), [sym])},
        {sym: bid},
        {sym: ask},
        {sym: (bid + ask) / 2},
        {sym: np.zeros(sv.MINUTES, bool)},
    )


def _cfg(**exe):
    c = {**CFG, "execution": {**CFG["execution"], **exe}}
    c["exits"] = {
        **CFG["exits"],
        "X6": {"tp": [0.2], "sl": [0.3], "basis": "mid"},
        "X7": {"cap_min": [30]},
    }
    return c


def test_struct_exit_fills_after_trigger_not_before():
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    bid[60] = 5.0  # 触发分钟本身的快照很高：不得用它成交
    bid[61] = 0.5
    od = _od(bid, ask, expiry=DAY + dt.timedelta(days=1))
    ev = pd.DataFrame({"e": [50], "direction": [1], "spot": [100.0]})
    trig = {"X3_mid1": {1: np.array([60]), -1: np.array([], int)}}
    r = sv.evaluate(od, ev, trig, _cfg())
    assert r["hold"][0, 0, 0, 0, 0] == 11  # 61 − 50
    exp = -1.1 * 200 - sv._BUY[2] + 0.5 * 200 - sv.sell_fee(2, 0.5)
    assert np.isclose(r["net"][0, 0, 0, 0, 0], exp, atol=1e-3)


def test_zero_bid_is_a_price_not_skipped():
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    bid[70:] = 0.0
    od = _od(bid, ask, expiry=DAY + dt.timedelta(days=1))
    ev = pd.DataFrame({"e": [60], "direction": [1], "spot": [100.0]})
    trig = {"none": {1: np.array([], int), -1: np.array([], int)}}
    r = sv.evaluate(od, ev, trig, _cfg())
    # mid 在 70 跌到 0.55 ≤ 1.1×0.7 → 71 成交，bid = 0，不回看 69 的 1.0
    assert r["hold"][0, 0, 0, 0, 0] == 11
    assert np.isclose(r["net"][0, 0, 0, 0, 0], -1.1 * 200 - sv._BUY[2], atol=1e-3)


def test_take_profit_judged_on_mid_filled_next_bid():
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    bid[65], ask[65] = 1.3, 1.36  # mid 1.33 ≥ 1.32 → 66 成交
    bid[66] = 1.25
    od = _od(bid, ask, expiry=DAY + dt.timedelta(days=1))
    ev = pd.DataFrame({"e": [60], "direction": [1], "spot": [100.0]})
    trig = {"none": {1: np.array([], int), -1: np.array([], int)}}
    r = sv.evaluate(od, ev, trig, _cfg())
    exp = (
        -1.1 * 200
        - sv._BUY[2]
        + 1.25 * 100
        - sv.sell_fee(1, 1.25)
        + 1.0 * 100
        - sv.sell_fee(1, 1.0)
    )
    assert np.isclose(r["net"][0, 0, 0, 0, 0], exp, atol=1e-3)
    assert r["hold"][0, 0, 0, 0, 0] == 30


def test_d001_zero_dte():
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    od = _od(bid, ask, expiry=DAY)
    trig = {"none": {1: np.array([], int), -1: np.array([], int)}}
    late = pd.DataFrame({"e": [sv.hm("10:16")], "direction": [1], "spot": [100.0]})
    assert sv.evaluate(od, late, trig, _cfg()) is None  # 只有 0DTE → 10:15 后不开
    early = pd.DataFrame({"e": [sv.hm("10:30")], "direction": [1], "spot": [100.0]})
    assert sv.evaluate(od, early, trig, _cfg(d001_last_0dte_entry="10:40")) is not None
    r = sv.evaluate(od, early, trig, _cfg(d001_last_0dte_entry="10:40"))
    assert r["hold"][0, 0, 0, 0, 0] == sv.hm("10:45") - sv.hm("10:30")  # 0DTE 10:45 清仓


def test_greedy_one_position_at_a_time():
    e = np.array([10, 12, 30])
    hold = np.array([[5], [5], [5]])
    take = sv.greedy(e, hold)
    assert take[:, 0].tolist() == [True, False, True]


def test_shift_signal_within_day_no_wraparound():
    day = np.array(["2025-01-02"] * 3 + ["2025-01-03"] * 3)
    sig = np.array([1, 0, -1, 2, 0, 0], dtype=np.int8)
    out = sv.shift_signal_within_day(sig, day, 1)
    # 第 1 天：[1,0,-1] -> 后移 1 -> [0,1,0]；第 2 天：[2,0,0] -> [0,2,0]（不跨日回绕）
    assert out.tolist() == [0, 1, 0, 0, 2, 0]
    assert sv.shift_signal_within_day(sig, day, 0).tolist() == sig.tolist()
