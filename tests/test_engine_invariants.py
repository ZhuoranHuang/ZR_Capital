"""带账户层回测引擎（zr/sim/engine.py）的 8 条不变量测试（任务书：回测引擎收敛 v1）。"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from zr.costs.fees import option_order_fees
from zr.sim.engine import Account, SimPolicy, Trade, simulate_day, size_position

ET = ZoneInfo("America/New_York")
DAY = dt.date(2026, 3, 3)
NS = 1_000_000_000
SYM = "MU260306C00100000.US"


def ts(hhmmss: str) -> int:
    t = dt.time.fromisoformat(hhmmss)
    return int(pd.Timestamp(dt.datetime.combine(DAY, t), tz=ET).value)


def quotes(rows: list[tuple[int, float, float]], symbol: str = SYM) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ns": [r[0] for r in rows],
            "symbol": symbol,
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
        }
    )


def every_second(start: str, end: str, fn) -> list[tuple[int, float, float]]:
    """start..end（含两端）每秒一条；fn(秒偏移) → (bid, ask)。"""
    a, b = ts(start), ts(end)
    return [(t, *fn((t - a) // NS)) for t in range(a, b + 1, NS)]


def fees_total(n: int, side: str, px: float) -> float:
    return float(option_order_fees(n, side, px)["total"])


# ---------------- T1 现金恒等 ----------------


def test_t1_cash_identity():
    q = quotes(
        every_second("10:00:00", "10:10:00", lambda s: (1.00, 1.10) if s < 300 else (1.20, 1.30))
    )
    session = {"first_entry": "10:00", "last_entry": "10:09", "flat": "10:10"}
    acct = Account(equity_usd=10000, sizing_basis="premium")
    pol = SimPolicy(lag_s=0, min_coverage=0.0)
    r = simulate_day(
        q, [(ts("10:00:00"), 1, SYM, {"kind": "time", "minutes": 5})], acct, session, pol
    )
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t.qty == 4
    assert t.entry_px == pytest.approx(1.10, abs=1e-9)
    assert t.exit_px == pytest.approx(1.20, abs=1e-9)
    expect_fees = fees_total(4, "Buy", 1.10) + fees_total(4, "Sell", 1.20)
    assert t.fees_usd == pytest.approx(expect_fees, abs=1e-9)
    assert t.pnl_usd == pytest.approx((1.20 - 1.10) * 100 * 4 - t.fees_usd, abs=1e-9)
    assert r.equity_end == pytest.approx(r.equity_start + t.pnl_usd, abs=1e-9)


# ---------------- T2 仓位 ----------------


def test_t2_sizing():
    a5 = Account(equity_usd=2000, max_risk_pct=0.05, sizing_basis="premium")
    assert size_position(a5, 1.47, None) == 0
    assert size_position(a5, 0.92, None) == 1
    assert size_position(a5, 25.0, None) == 0
    a8 = Account(equity_usd=2000, max_risk_pct=0.08, sizing_basis="premium")
    assert size_position(a8, 1.47, None) == 1
    s5 = Account(equity_usd=2000, max_risk_pct=0.05, sizing_basis="stop")
    assert size_position(s5, 0.92, 0.30) == 3
    assert size_position(s5, 1.47, 0.30) == 2
    assert size_position(s5, 2.15, 0.30) == 1
    assert size_position(s5, 25.0, 0.30) == 0
    q = quotes(every_second("10:00:00", "10:10:00", lambda s: (24.90, 25.00)))
    session = {"first_entry": "10:00", "last_entry": "10:09", "flat": "10:10"}
    r = simulate_day(
        q,
        [(ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 5})],
        a5,
        session,
        SimPolicy(lag_s=0, min_coverage=0.0),
    )
    assert r.trades == []
    assert r.skipped["size_zero"] == 1


# ---------------- T3 时间顺序 ----------------


def test_t3_time_order_and_flat():
    q = quotes(every_second("15:40:00", "15:55:00", lambda s: (1.00, 1.05)))
    session = {"first_entry": "15:40", "last_entry": "15:46", "flat": "15:50"}
    acct = Account(equity_usd=10000, sizing_basis="premium")
    r = simulate_day(
        q,
        [(ts("15:45:00"), 1, SYM, {"kind": "time", "minutes": 30})],
        acct,
        session,
        SimPolicy(lag_s=0, min_coverage=0.0),
    )
    assert len(r.trades) == 1
    flat_ns = ts("15:50:00")
    for t in r.trades:
        # 清仓容差：flat 之后 120 秒内取第一条可清仓报价，所以出场可晚于 flat 至多 120 秒
        assert t.entry_ts_ns <= t.exit_ts_ns <= flat_ns + 120 * 10**9
    assert r.trades[0].exit_ts_ns == flat_ns
    assert r.trades[0].reason == "flat"


# ---------------- T4 冷却 ----------------


def _two_signals(second_at: str):
    q = quotes(every_second("10:00:00", "10:20:00", lambda s: (1.00, 1.05)))
    session = {"first_entry": "10:00", "last_entry": "10:19", "flat": "10:20"}
    sig = [
        (ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 5}),
        (ts(second_at), 1, SYM, {"kind": "time", "minutes": 5}),
    ]
    return q, session, sig


def test_t4_cooldown():
    q, session, sig = _two_signals("10:06:03")  # 第一笔 10:06:00 平仓后 3 秒
    acct = Account(equity_usd=10000, sizing_basis="premium")
    r60 = simulate_day(q, sig, acct, session, SimPolicy(lag_s=0, min_coverage=0.0, cooldown_s=60))
    assert len(r60.trades) == 1
    assert r60.skipped["cooldown"] == 1
    r0 = simulate_day(q, sig, acct, session, SimPolicy(lag_s=0, min_coverage=0.0, cooldown_s=0))
    assert len(r0.trades) == 2


# ---------------- T5 持仓期间忽略 ----------------


def test_t5_in_position_ignored():
    q, session, sig = _two_signals("10:03:00")  # 第一笔持仓期间
    acct = Account(equity_usd=10000, sizing_basis="premium")
    for cd in (0, 60):
        r = simulate_day(q, sig, acct, session, SimPolicy(lag_s=0, min_coverage=0.0, cooldown_s=cd))
        assert len(r.trades) == 1
        assert r.skipped["in_position"] == 1
        assert r.skipped["cooldown"] == 0


# ---------------- T6 确定性 ----------------


def test_t6_deterministic():
    q, session, sig = _two_signals("10:12:00")
    sig.append(
        (ts("10:14:00"), -1, SYM, {"kind": "bracket", "tp": 0.1, "sl": 0.3, "max_hold_min": 3})
    )
    acct = Account(equity_usd=10000)
    pol = SimPolicy(lag_s=2, min_coverage=0.0)
    a = simulate_day(q, sig, acct, session, pol)
    b = simulate_day(q, sig, acct, session, pol)
    assert a == b
    assert all(isinstance(t, Trade) for t in a.trades)


# ---------------- T7 无效报价 ----------------


def _t7_run(f):
    q = quotes(every_second("10:00:00", "10:10:00", f))
    session = {"first_entry": "10:00", "last_entry": "10:09", "flat": "10:10"}
    return simulate_day(
        q,
        [(ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 1})],
        Account(equity_usd=10000, sizing_basis="premium"),
        session,
        SimPolicy(lag_s=0, min_coverage=0.0),
    )


def test_t7_invalid_quotes():
    # (a) 入场 10:01:00；time 1 分钟 → 目标 10:02:00；
    #     10:02:00–10:02:09 bid=0（连续 10 秒），10:02:10 起有效
    ra = _t7_run(lambda s: (0.0, 1.05) if 120 <= s < 130 else (1.00, 1.05))
    assert ra.trades[0].exit_ts_ns == ts("10:02:10")
    assert ra.trades[0].exit_px > 0
    # (b) 10:01:00–10:01:01 bid=0.68 / ask=1.05（价差约 43%），2 秒后（10:01:02）起 0.98 / 1.05
    rb = _t7_run(lambda s: (0.68, 1.05) if 60 <= s < 62 else (0.98, 1.05))
    assert rb.trades[0].entry_ts_ns == ts("10:01:02")
    assert rb.trades[0].entry_px == pytest.approx(1.05, abs=1e-9)
    # (c) ask == bid 视为有效，可在该报价成交
    rc = _t7_run(lambda s: (1.00, 1.00) if s == 60 else (0.98, 1.05))
    assert rc.trades[0].entry_ts_ns == ts("10:01:00")
    assert rc.trades[0].entry_px == pytest.approx(1.00, abs=1e-9)


# ---------------- T8 覆盖率 ----------------


def test_t8_coverage():
    session = {"first_entry": "09:32", "last_entry": "15:40", "flat": "15:50"}
    sig = [(ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 1})]
    acct = Account(equity_usd=10000, sizing_basis="premium")
    with pytest.raises(ValueError):
        simulate_day(quotes([]), sig, acct, session, SimPolicy(min_coverage=0.95))
    q = quotes(every_second("10:00:00", "10:09:59", lambda s: (1.00, 1.05)))  # 只覆盖 10 分钟
    with pytest.raises(ValueError):
        simulate_day(q, sig, acct, session, SimPolicy(min_coverage=0.95))
    r = simulate_day(q, sig, acct, session, SimPolicy(min_coverage=0.0))
    assert r.equity_start == pytest.approx(10000, abs=1e-9)


# ---------------- T9 清仓断流 ----------------


def test_t9_flat_no_quotes_raises():
    # 报价到 10:09:59 为止；flat=10:10；持有 30 分钟 → 需要清仓，但 flat 后 120 秒内没有任何报价行
    q = quotes(every_second("10:00:00", "10:09:59", lambda s: (1.00, 1.05)))
    session = {"first_entry": "10:00", "last_entry": "10:09", "flat": "10:10"}
    with pytest.raises(ValueError):
        simulate_day(
            q,
            [(ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 30})],
            Account(equity_usd=10000, sizing_basis="premium"),
            session,
            SimPolicy(lag_s=0, min_coverage=0.0),
        )


# ---------------- T10 清仓时报价无效 ----------------


def _t10_run(after_flat: list[tuple[int, float, float]]):
    """10:00:00–10:09:59 每秒 1.00 / 1.05；flat=10:10；after_flat 为 flat 及之后的报价行。"""
    q = quotes(every_second("10:00:00", "10:09:59", lambda s: (1.00, 1.05)) + after_flat)
    session = {"first_entry": "10:00", "last_entry": "10:09", "flat": "10:10"}
    return simulate_day(
        q,
        [(ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 30})],
        Account(equity_usd=10000, sizing_basis="premium"),
        session,
        SimPolicy(lag_s=0, min_coverage=0.0),
    )


def test_t10_flat_with_bad_quotes():
    # (a) flat 起 2 分钟内只有价差 40% 的报价（bid=0.60 / ask=0.90）
    ra = _t10_run(every_second("10:10:00", "10:12:00", lambda s: (0.60, 0.90)))
    ta = ra.trades[0]
    assert ta.exit_ts_ns == ts("10:10:00")
    assert ta.exit_px == pytest.approx(0.60, abs=1e-9)
    assert ta.reason == "flat"
    assert ra.skipped["flat_wide"] == 1
    # (b) flat 后 1 分钟才出现第一条报价
    rb = _t10_run(every_second("10:11:00", "10:12:00", lambda s: (1.00, 1.05)))
    assert rb.trades[0].exit_ts_ns == ts("10:11:00")
    assert rb.trades[0].reason == "flat"
    assert rb.skipped["flat_late"] == 1
    # (c) flat 后 10 分钟才有报价
    with pytest.raises(ValueError):
        _t10_run(every_second("10:20:00", "10:21:00", lambda s: (1.00, 1.05)))
    # (d) flat 时报价 bid=0 / ask=0.05 → 按 0 成交，pnl = −(入场成本 + 费用)
    rd = _t10_run([(ts("10:10:00"), 0.0, 0.05)])
    td = rd.trades[0]
    assert td.exit_px == pytest.approx(0.0, abs=1e-9)
    expect_fees = fees_total(td.qty, "Buy", td.entry_px) + fees_total(td.qty, "Sell", 0.0)
    assert td.fees_usd == pytest.approx(expect_fees, abs=1e-9)
    assert td.pnl_usd == pytest.approx((0 - td.entry_px) * 100 * td.qty - td.fees_usd, abs=1e-9)
    # (e) flat 时刻唯一报价为 0 / 0 → 成交价 0，flat_wide 不加，flat_zero += 1
    re_ = _t10_run([(ts("10:10:00"), 0.0, 0.0)])
    assert re_.trades[0].exit_px == pytest.approx(0.0, abs=1e-9)
    assert re_.skipped["flat_wide"] == 0
    assert re_.skipped["flat_zero"] == 1


# ---------------- T12 入场等待上限 ----------------


def test_t12_late_quote():
    # lag 2 秒；信号 10:01:00；10:01:00–10:01:11 只有宽报价，
    # 第一条紧报价在 10:01:12（= lag + 10 秒 > 5 秒上限）
    q = quotes(
        every_second(
            "10:00:00", "10:10:00", lambda s: (0.68, 1.05) if 60 <= s < 72 else (0.98, 1.05)
        )
    )
    session = {"first_entry": "10:00", "last_entry": "10:09", "flat": "10:10"}
    r = simulate_day(
        q,
        [(ts("10:01:00"), 1, SYM, {"kind": "time", "minutes": 1})],
        Account(equity_usd=10000, sizing_basis="premium"),
        session,
        SimPolicy(lag_s=2, min_coverage=0.0, max_entry_wait_s=5.0),
    )
    assert r.trades == []
    assert r.skipped["late_quote"] == 1
