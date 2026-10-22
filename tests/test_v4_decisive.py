"""V4 决定性检验引擎的机械性测试（合成数据，不连网、不读真实结果）。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from zr.costs.fees import option_order_fees
from zr.features.bars import ET, resample_bars
from zr.features.indicators import bollinger
from zr.features.regime import RANGE
from zr.research import v4_decisive as v4d

V4 = v4d.load_v4()
DAY = dt.date(2026, 10, 15)
T0 = pd.Timestamp("2026-10-15 10:30:00", tz=ET).tz_convert("UTC")


def _ns(t: pd.Timestamp) -> int:
    return int(pd.DatetimeIndex([t]).as_unit("ns").asi8[0])


def _ticks(points):
    """[(秒偏移, bid, ask)] → Ticks（相对 T0）。"""
    ts = np.array([_ns(T0 + pd.Timedelta(seconds=s)) for s, _, _ in points], dtype=np.int64)
    return v4d.Ticks(ts, np.array([b for _, b, _ in points], float),
                     np.array([a for _, _, a in points], float))


FLAT = _ns(T0 + pd.Timedelta(hours=5))


def _buy(n):
    return float(option_order_fees(n, "Buy", 1.0)["total"])


def _sell(n, px):
    return float(option_order_fees(n, "Sell", px)["total"])


def test_entry_is_first_ask_at_or_after_decision_plus_2s():
    tk = _ticks([(0, 0.9, 1.0), (1, 0.95, 1.05), (3, 1.0, 1.1), (1900, 1.0, 1.1)])
    r = v4d.simulate(tk, _ns(T0), np.array([], np.int64), FLAT, V4)
    assert r["ask0"] == pytest.approx(1.1)  # 第 3 秒那笔（≥ 0+2 s），不是第 1 秒
    assert r["reason"] == "cap"


def test_tp_sells_half_then_structure_exits_rest_with_exact_fees():
    struct = np.array([_ns(T0 + pd.Timedelta(seconds=120))], np.int64)
    tk = _ticks([(2, 0.9, 1.0), (30, 1.25, 1.27),   # mid 1.26 ≥ 1.2 → 止盈触发 30 s
                 (32, 1.22, 1.24),                   # +2 s 后第一个 bid 1.22 卖 1 张
                 (121, 1.1, 1.12), (122, 1.05, 1.07)])  # 结构 120 s 触发，+2 s → 1.05
    r = v4d.simulate(tk, _ns(T0), struct, FLAT, V4)
    assert r["tp_used"] and r["reason"] == "structure"
    want = -1.0 * 100 * 2 - _buy(2) + (1.22 * 100 - _sell(1, 1.22)) + (1.05 * 100 - _sell(1, 1.05))
    assert r["net_usd"] == pytest.approx(want, abs=1e-9)
    assert r["net_pct"] == pytest.approx(want / 200 * 100)


def test_stop_loss_on_mid_sells_all_and_beats_later_structure():
    struct = np.array([_ns(T0 + pd.Timedelta(seconds=600))], np.int64)
    tk = _ticks([(2, 0.9, 1.0), (50, 0.6, 0.78), (53, 0.55, 0.7), (601, 0.5, 0.6)])
    r = v4d.simulate(tk, _ns(T0), struct, FLAT, V4)  # mid 0.69 ≤ 0.7 → 止损
    assert r["reason"] == "sl" and not r["tp_used"]
    assert r["net_usd"] == pytest.approx(-200 - _buy(2) + 2 * 55 - _sell(2, 0.55))


def test_structure_trigger_must_be_after_entry_bar_close():
    struct = np.array([_ns(T0), _ns(T0 + pd.Timedelta(seconds=240))], np.int64)
    tk = _ticks([(2, 0.9, 1.0), (241, 0.95, 1.0), (243, 0.96, 1.01)])
    r = v4d.simulate(tk, _ns(T0), struct, FLAT, V4)
    assert r["reason"] == "structure"
    assert r["exit_ns"] == _ns(T0 + pd.Timedelta(seconds=243))  # 入场 K 本身的触发不算


def test_forced_flat_and_no_entry_at_or_after_flat():
    tk = _ticks([(2, 0.9, 1.0), (100, 0.95, 1.0), (300, 0.97, 1.02)])
    flat = _ns(T0 + pd.Timedelta(seconds=98))
    r = v4d.simulate(tk, _ns(T0), np.array([], np.int64), flat, V4)
    assert r["reason"] == "flat" and r["exit_ns"] == _ns(T0 + pd.Timedelta(seconds=100))
    assert v4d.simulate(tk, _ns(T0), np.array([], np.int64), _ns(T0), V4) is None


def test_greedy_skips_overlapping():
    rows = pd.DataFrame({"decision_ns": [0, 5, 10, 20], "exit_ns": [10, 8, 15, 25]})
    assert v4d.greedy(rows)["decision_ns"].tolist() == [0, 10, 20]


def test_bh_and_cluster_p():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    assert v4d.bh(p, 0.10).tolist() == [True, True, True, False]
    assert v4d.bh(np.array([0.2, 0.5]), 0.10).tolist() == [False, False]
    x = np.r_[np.full(20, 5.0), np.full(20, 6.0)] + np.tile([0.1, -0.1], 20)
    g = np.repeat(np.arange(20), 2)
    t, pv = v4d.cluster_p(x, g, 10)
    assert t > 10 and pv < 1e-6
    assert v4d.cluster_p(np.ones(5), np.arange(5), 10)[1] == 1.0  # n < min_n


def test_pick_contract_otm_one_strike_and_0dte_cutoff():
    exp0, exp1 = DAY, DAY + dt.timedelta(days=1)
    strikes = np.array([95.0, 100.0, 105.0])
    chain = {(e, r): (strikes, [f"{e}{r}{k}" for k in strikes]) for e in (exp0, exp1)
             for r in ("call", "put")}
    c = v4d.pick_contract(chain, DAY, 1, 100.0, dt.time(10, 0), V4)
    assert c["strike"] == 105.0 and c["dte"] == 0  # 平值 100 不算虚值，取上一档
    p = v4d.pick_contract(chain, DAY, -1, 100.0, dt.time(10, 16), V4)
    assert p["strike"] == 95.0 and p["dte"] == 1  # 10:15 后排除 0DTE
    only0 = {k: v for k, v in chain.items() if k[0] == exp0}
    assert v4d.pick_contract(only0, DAY, 1, 100.0, dt.time(10, 16), V4) is None


def test_e11_delayed_requires_next_close_beyond_breakout():
    idx = pd.date_range("2026-10-15 13:30", periods=40, freq="2min", tz="UTC")
    close = np.full(40, 100.0)
    close[30], close[31] = 103.0, 103.5   # 30 突破，31 仍在突破侧 → 31 入场
    close[35], close[36] = 103.0, 99.0    # 35 突破，36 回落 → 不入场
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
                       "volume": np.r_[np.full(30, 100.0), np.full(10, 1000.0)]}, index=idx)
    reg = pd.DataFrame({"state": np.full(40, RANGE)}, index=idx)
    x = pd.DataFrame({"pctb": np.full(40, 0.5)}, index=idx)
    h = {"M": 20, "v": 1.5, "vol_window": 20, "confirm_bars": 1}
    s = v4d.e11_delayed(df, x, reg, h)
    assert s[31] == 1 and s[30] == 0 and s[36] == 0


def test_resample_anchors_even_minutes_even_if_first_bar_odd():
    idx = pd.date_range("2026-10-21 08:01", periods=10, freq="1min", tz="UTC")  # 04:01 ET
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)
    out = resample_bars(df, "2min")
    assert all(t.minute % 2 == 0 for t in out.index)


def test_bollinger_partial_window():
    s = pd.Series(np.arange(1.0, 26.0))
    assert bollinger(s, 20).iloc[:19]["mid"].isna().all()
    b = bollinger(s, 20, min_periods=1)
    assert b["mid"].iloc[0] == 1.0 and b["mid"].iloc[4] == 3.0


def test_uncovered_segments():
    t = pd.Timestamp("2026-10-15 14:00", tz="UTC")
    m = pd.Timedelta(minutes=1)
    have = [((t + 5 * m).value, (t + 10 * m).value)]
    segs = v4d.uncovered(t, t + 15 * m, have)
    assert segs == [(t, t + 5 * m), (t + 10 * m, t + 15 * m)]
    assert v4d.uncovered(t + 6 * m, t + 9 * m, have) == []
