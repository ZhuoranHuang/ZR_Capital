"""负责人挂单策略模拟（zr/backtest/owner_fill.py）每个分支的合成 tick 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zr.backtest.fill_limit import Book
from zr.backtest.owner_fill import OwnerStrategy, SideParams, execute_owner

SYM = "QQQ260805C712000.US"  # 0.01 tick
T0 = pd.Timestamp("2026-08-04 15:00:00", tz="UTC")


def _ns(s: float) -> int:
    return int((T0 + pd.Timedelta(seconds=s)).value)


def book(quotes, trades=()):
    q = pd.DataFrame(
        [
            {"ts": T0 + pd.Timedelta(seconds=s), "bid": b, "ask": a, "bid_size": 10, "ask_size": 10}
            for s, b, a in quotes
        ]
    )
    t = (
        pd.DataFrame(
            [{"ts": T0 + pd.Timedelta(seconds=s), "price": p, "size": z} for s, p, z in trades]
        )
        if trades
        else None
    )
    return Book.from_frames(q, t)


SP = SideParams((-0.5,), (10.0,), (0,), 4.0, 2.0, 5.0)
ST = OwnerStrategy(SP, SP, queue_fill_prob=1.0, lag_s=2.0)
RNG = np.random.default_rng(0)


def run(b, side=1, mode="entry", draw=None, deadline=None, strat=ST):
    return execute_owner(
        b, side, 2, _ns(0), strat, SYM, np.random.default_rng(0), mode, deadline, draw
    )


def test_initial_price_from_k0_and_fill_on_quote_cross():
    b = book([(0, 1.00, 1.10), (5, 0.99, 1.00)])
    r = run(b, draw={"k0": -0.5, "wait_s": 10, "n_amend": 0})
    assert r.px0 == pytest.approx(1.00)  # 中间价 1.05 − 0.5 × 0.10 = 买一
    assert r.stage == "filled" and r.avg_px == pytest.approx(1.00) and r.first_fill_ns == _ns(5)


def test_marketable_k0_capped_at_ask_and_sell_mirror():
    b = book([(0, 1.00, 1.10)])
    r = run(b, draw={"k0": 2.0, "wait_s": 10, "n_amend": 0})
    assert r.px0 == pytest.approx(1.10) and r.stage == "filled" and r.first_fill_ns == _ns(2)
    s = run(b, side=-1, draw={"k0": 2.0, "wait_s": 10, "n_amend": 0})
    assert s.px0 == pytest.approx(1.00) and s.stage == "filled"


def test_wait_expiry_entry_missed_exit_crosses():
    b = book([(0, 1.00, 1.10), (30, 1.00, 1.10)])
    m = run(b, draw={"k0": -0.5, "wait_s": 6, "n_amend": 0})
    assert (
        m.stage == "missed" and m.missed_qty == 2 and m.end_reason == "wait" and m.end_ns == _ns(8)
    )
    x = run(b, side=-1, mode="exit", draw={"k0": -0.5, "wait_s": 6, "n_amend": 0})
    assert x.stage == "cross" and x.avg_px == pytest.approx(1.00)  # 卖出吃买一


def test_amendment_steps_toward_market_then_fills_on_trade_through():
    # +2 s 到达挂 1.00；+6 s 改到 1.02；+10 s 改到 1.04。+9 s 成交价 1.03 没越过 1.02；
    # +11 s 成交价 1.03 越过 1.04 → 按 1.04 成交
    b = book([(0, 1.00, 1.10)], [(5, 1.01, 5), (9, 1.03, 5), (11, 1.03, 5)])
    r = run(b, draw={"k0": -0.5, "wait_s": 10, "n_amend": 2})
    assert r.stage == "filled" and r.avg_px == pytest.approx(1.04) and r.first_fill_ns == _ns(11)


def test_cancel_when_price_runs_away():
    b = book([(0, 1.00, 1.10), (4, 1.03, 1.13)])  # 中间价 +3 tick < 5
    r0 = run(b, draw={"k0": -0.5, "wait_s": 10, "n_amend": 0})
    assert r0.end_reason == "wait"
    b2 = book([(0, 1.00, 1.10), (4, 1.06, 1.16)])  # +6 tick ≥ 5 → 撤单
    r = run(b2, draw={"k0": -0.5, "wait_s": 10, "n_amend": 0})
    assert r.stage == "missed" and r.end_reason == "away" and r.end_ns == _ns(4)


def test_deadline_and_sampling_reproducible():
    b = book([(0, 1.00, 1.10)])
    r = run(b, mode="exit", deadline=_ns(5), draw={"k0": -0.5, "wait_s": 10, "n_amend": 0})
    assert r.end_reason == "deadline" and r.stage == "cross"
    sp = SideParams((-1.0, 0.0), (5.0, 60.0), (0, 1), 4.0, 2.0, None)
    st = OwnerStrategy(sp, sp)
    a = execute_owner(b, 1, 1, _ns(0), st, SYM, np.random.default_rng(7))
    c = execute_owner(b, 1, 1, _ns(0), st, SYM, np.random.default_rng(7))
    assert (a.k0, a.wait_s, a.n_amend) == (c.k0, c.wait_s, c.n_amend)
