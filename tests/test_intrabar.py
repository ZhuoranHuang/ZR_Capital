"""盘中引擎（zr/backtest/intrabar.py）的单元测试（合成数据，不连网）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from zr.backtest.fill_limit import Book
from zr.backtest.intrabar import BarState, evaluate, place

SYM = "MU261023C900000.US"
T0 = pd.Timestamp("2026-10-22 15:00:00", tz="UTC")
NS = 1_000_000_000


def _ns(s: float) -> int:
    return int((T0 + pd.Timedelta(seconds=s)).value)


def book(quotes, trades=()):
    q = pd.DataFrame(
        [{"ts": T0 + pd.Timedelta(seconds=s), "bid": b, "ask": a} for s, b, a in quotes]
    )
    t = (
        pd.DataFrame(
            [{"ts": T0 + pd.Timedelta(seconds=s), "price": p, "size": z} for s, p, z in trades]
        )
        if trades
        else None
    )
    return Book.from_frames(q, t)


def test_evaluate_directional_ret_and_extreme():
    s = np.array([100.0, 100.05, 100.10, 100.12])
    feat = {
        "S": s,
        "ret_2": np.array([np.nan, np.nan, 10.0, 7.0]),
        "hi_60": np.array([0.0, 0.0, 0.0, 5.0]),
        "lo_60": np.array([0.0, 5.0, 10.0, 12.0]),
    }
    spec = [
        {"feature": "ret", "window_s": 2, "op": ">=", "value": 8},
        {"feature": "dist_extreme", "window_min": 1, "side": "with", "op": "<=", "value": 50},
    ]
    assert evaluate(feat, spec, 1).tolist() == [False, False, True, False]
    assert not evaluate(feat, spec, -1).any()  # 做空时涨幅为负


def test_trend_side_of_mid():
    bs = BarState.__new__(BarState)
    bs.close_ns = np.arange(1, 13) * 120 * NS
    bs.close = np.array([1.0] * 12)
    bs.mid = np.array([0.0] * 9 + [2.0] * 3)  # 前 9 根收在中轨上方，后 3 根下方
    t = np.array([10 * 120 * NS, 12 * 120 * NS])
    assert bs.trend(t, 10, 8).tolist() == [1, 0]  # 第 1–10 根 9 上 1 下；第 3–12 根 7 上 3 下


def test_place_modes():
    b = book([(0, 1.00, 1.10), (5, 1.00, 1.10)], [(4, 0.94, 3)])
    n, px, _, _ = place(
        b, 1, 2, _ns(0), {"mode": "taker", "lag_s": 2}, SYM, 1.0, None, np.random.default_rng(0)
    )
    assert n == 2 and px == pytest.approx(1.10)
    # passive：中间价 1.05 − 1 × 0.10 = 0.95；第 4 s 成交价 0.94 越过 → 按 0.95 成交
    n, px, _, _ = place(
        b,
        1,
        2,
        _ns(0),
        {"mode": "passive", "k_spread": 1.0, "wait_s": 10, "lag_s": 2},
        SYM,
        1.0,
        None,
        np.random.default_rng(0),
    )
    assert n == 2 and px == pytest.approx(0.95)
    # path：先挂 0.90，第 3 s 改到 0.96，第 4 s 成交价 0.94 越过 0.96 → 0.96
    order = {"mode": "path", "path": [(_ns(0), 0.90), (_ns(3), 0.96)], "end_ns": _ns(9), "lag_s": 0}
    n, px, _, _ = place(b, 1, 2, _ns(0), order, SYM, 1.0, None, np.random.default_rng(0))
    assert n == 2 and px == pytest.approx(0.96)


def test_example_rule_parses():
    r = yaml.safe_load(open("config/intrabar/example_owner_profile.yaml", encoding="utf-8"))
    assert r["direction"] == "trend" and r["trend"] == {"bars": 10, "min_same": 8}
    assert r["order"]["mode"] == "passive" and r["exit"]["tp"] == 0.10
