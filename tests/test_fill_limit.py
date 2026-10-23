"""限价单成交模拟的每条分支（合成 tick）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zr.backtest.fill_limit import Book, LimitRule, cross_now, execute, limit_price

SYM = "QQQ260805C712000.US"  # 全系列 0.01 tick（D-015）
T0 = pd.Timestamp("2026-08-04 15:00:00", tz="UTC")


def _ns(s: float) -> int:
    return int((T0 + pd.Timedelta(seconds=s)).value)


def book(quotes, trades=()):
    q = pd.DataFrame(
        [
            {
                "ts": T0 + pd.Timedelta(seconds=s),
                "bid": b,
                "ask": a,
                "bid_size": bs,
                "ask_size": as_,
            }
            for s, b, a, bs, as_ in quotes
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


def test_limit_price_rules_and_cap():
    b = book([(0, 1.00, 1.10, 10, 10)])
    assert limit_price(b, 0, 1, "passive", SYM) == pytest.approx(1.01)
    assert limit_price(b, 0, 1, "mid", SYM) == pytest.approx(1.05)
    assert limit_price(b, 0, 1, "aggressive", SYM) == pytest.approx(1.09)
    assert limit_price(b, 0, -1, "passive", SYM) == pytest.approx(1.09)
    assert limit_price(b, 0, -1, "aggressive", SYM) == pytest.approx(1.01)
    tight = book([(0, 1.00, 1.01, 10, 10)])
    assert limit_price(tight, 0, 1, "passive", SYM) == pytest.approx(1.01)  # 越过对手价 → 对手价


def test_quote_cross_fills_at_limit_with_size_cap():
    b = book([(0, 1.00, 1.10, 10, 10), (4, 1.02, 1.05, 10, 1), (5, 1.02, 1.04, 10, 5)])
    r = execute(b, 1, 3, _ns(0), LimitRule(price="mid", T=10, ladder="abandon"), SYM)
    assert r.limit_px == pytest.approx(1.05) and r.avg_px == pytest.approx(1.05)
    assert r.filled_qty == 3 and r.first_fill_ns == _ns(4)  # 先 1 张（卖一量 1），再 2 张


def test_trade_through_fills_and_at_limit_depends_on_queue():
    q = [(0, 1.00, 1.10, 10, 10)]
    through = book(q, [(3, 1.04, 2)])
    r = execute(through, 1, 2, _ns(0), LimitRule(price="mid", T=10, ladder="abandon"), SYM)
    assert r.filled_qty == 2 and r.avg_px == pytest.approx(1.05)
    at = book(q, [(3, 1.05, 2)])
    yes = execute(
        at, 1, 2, _ns(0), LimitRule(price="mid", T=10, ladder="abandon", queue_fill_prob=1.0), SYM
    )
    no = execute(
        at, 1, 2, _ns(0), LimitRule(price="mid", T=10, ladder="abandon", queue_fill_prob=0.0), SYM
    )
    assert yes.filled_qty == 2 and no.filled_qty == 0 and no.stage == "missed"


def test_events_before_post_time_are_ignored():
    b = book([(0, 1.00, 1.10, 10, 10)], [(1, 1.00, 5)])  # 挂单在 +2 s 才到
    r = execute(b, 1, 1, _ns(0), LimitRule(price="mid", T=10, ladder="abandon"), SYM)
    assert r.filled_qty == 0 and r.missed_qty == 1


def test_timeout_reprice_then_cross():
    b = book([(0, 1.00, 1.10, 10, 10), (20, 1.00, 1.10, 10, 10), (30, 1.02, 1.12, 10, 10)])
    r = execute(b, 1, 2, _ns(0), LimitRule(price="passive", T=5, ladder="reprice"), SYM)
    assert r.stage == "cross" and r.filled_qty == 2
    assert r.avg_px == pytest.approx(1.10)  # 两段都没成交，最后按对手价（第 12 s 时的卖一）
    b2 = book([(0, 1.00, 1.10, 10, 10)], [(9, 1.05, 5)])
    r2 = execute(
        b2,
        1,
        2,
        _ns(0),
        LimitRule(price="passive", T=5, ladder="reprice", queue_fill_prob=0.0),
        SYM,
    )
    # 第一段挂 1.01 未成交；改到中间价 +1 tick = 1.06，第 9 s 成交价 1.05 越过 → 成交
    assert r2.stage == "reprice" and r2.avg_px == pytest.approx(1.06)


def test_abandon_records_miss_and_sell_side_mirrors():
    b = book([(0, 2.00, 2.10, 10, 10), (3, 2.06, 2.12, 4, 10)])
    r = execute(b, -1, 3, _ns(0), LimitRule(price="mid", T=10, ladder="abandon"), SYM)
    # 卖出挂 2.05（中间价向上取整到己方），买一 2.06 ≥ 2.05 → 成交，数量受买一量 4 限制 → 全成
    assert r.limit_px == pytest.approx(2.05) and r.filled_qty == 3
    none = book([(0, 2.00, 2.10, 10, 10)])
    m = execute(none, -1, 3, _ns(0), LimitRule(price="passive", T=2, ladder="abandon"), SYM)
    assert m.stage == "missed" and m.missed_qty == 3 and np.isnan(m.avg_px)


def test_cross_now_and_deadline():
    b = book([(0, 1.00, 1.10, 10, 10), (3, 0.98, 1.08, 10, 10)])
    assert cross_now(b, -1, _ns(0)) == (_ns(3), 0.98)
    r = execute(
        b, 1, 1, _ns(0), LimitRule(price="passive", T=60, ladder="reprice"), SYM, deadline_ns=_ns(4)
    )
    assert r.stage == "cross" and r.last_fill_ns >= _ns(4)  # 截止前没成交 → 截止时吃对手价


def test_reprice_abandon_misses_after_two_stages_and_partial_keeps_fill():
    b = book([(0, 1.00, 1.10, 10, 10)])
    r = execute(b, 1, 2, _ns(0), LimitRule(price="passive", T=5, ladder="reprice_abandon"), SYM)
    assert r.stage == "missed" and r.missed_qty == 2 and r.filled_qty == 0
    assert r.end_ns == _ns(2 + 5 + 5)  # +2 s 到达，两段各 5 s
    # 第二段（中间价 +1 tick = 1.06）只成交 1 张，剩 1 张放弃
    b2 = book([(0, 1.00, 1.10, 10, 10)], [(9, 1.05, 1)])
    r2 = execute(
        b2, 1, 2, _ns(0), LimitRule(price="passive", T=5, ladder="reprice_abandon", seed=1), SYM
    )
    assert r2.filled_qty == 1 and r2.missed_qty == 1 and r2.avg_px == pytest.approx(1.06)


def test_default_queue_prob_is_calibrated_value():
    assert LimitRule().queue_fill_prob == 0.9
