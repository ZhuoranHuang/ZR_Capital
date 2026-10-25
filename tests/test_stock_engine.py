"""正股模式引擎（zr/sim/stock.py）的 6 条测试（任务书：规则回测最小流水线）。"""

from __future__ import annotations

import dataclasses
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from zr.costs.fees import stock_order_fees
from zr.sim.engine import Account
from zr.sim.stock import StockPolicy, simulate_day, size_shares

ET = ZoneInfo("America/New_York")
DAY = dt.date(2026, 3, 3)


def ts(hhmm: str) -> int:
    return int(pd.Timestamp(dt.datetime.combine(DAY, dt.time.fromisoformat(hhmm)), tz=ET).value)


def bars(start: str, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """1 分钟 K，start 起每分钟一根；rows = (open, high, low, close)。"""
    t0 = ts(start)
    return pd.DataFrame(
        {
            "ts_ns": [t0 + i * 60 * 10**9 for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
        }
    )


def flat_bars(start: str, n: int) -> pd.DataFrame:
    return bars(start, [(100.00, 100.01, 99.99, 100.00)] * n)


def fee(n: int, side: str, px: float) -> float:
    return float(stock_order_fees(n, side, px)["total"])


SESSION = {"first_entry": "10:00", "last_entry": "10:30", "flat": "10:40"}
TIME2 = {"kind": "time", "bars": 2}


def _inv(r, n_signals: int) -> None:
    assert sum(r.skipped.values()) + len(r.trades) == n_signals


# ---------------- 1 现金恒等 ----------------


def test_1_cash_identity():
    b = flat_bars("10:00", 6)
    acct = Account(equity_usd=10000)  # 名义上限 max_notional_pct 默认 0.5
    r = simulate_day(b, [(1, +1)], acct, SESSION, StockPolicy(slippage_ticks=1), exit_rule=TIME2)
    _inv(r, 1)
    (t,) = r.trades
    assert t.qty == 49
    assert t.entry_px == pytest.approx(100.01)
    assert t.exit_px == pytest.approx(99.99)
    fees = fee(49, "Buy", 100.01) + fee(49, "Sell", 99.99)
    assert t.fees_usd == pytest.approx(fees)
    assert t.pnl_usd == pytest.approx((99.99 - 100.01) * 49 - fees)
    assert r.equity_end == pytest.approx(10000 + t.pnl_usd)


# ---------------- 2 仓位 ----------------


def test_2_sizing():
    a = Account(equity_usd=10000, max_risk_pct=0.05)
    assert size_shares(a, 100.0, 99.0) == 100  # 500 / 1 = 500，受「不超过全部权益」约束
    assert size_shares(a, 100.0, 90.0) == 50
    assert size_shares(a, 100.0, 80.0) == 25
    assert size_shares(a, 100.0, None, max_notional_pct=0.5) == 50
    assert size_shares(Account(equity_usd=2000), 500.0, None) == 2


# ---------------- 3 时间顺序 ----------------


def test_3_time_order_and_flat():
    b = flat_bars("15:55", 5)  # 15:55..15:59
    session = {"first_entry": "15:55", "last_entry": "15:58", "flat": "15:59"}
    r = simulate_day(
        b, [(3, +1)], Account(equity_usd=10000), session, exit_rule={"kind": "time", "bars": 5}
    )
    _inv(r, 1)
    (t,) = r.trades
    assert t.entry_ts_ns == ts("15:58")
    assert t.exit_ts_ns == ts("15:59")
    assert t.reason == "flat"
    assert t.entry_ts_ns <= t.exit_ts_ns <= ts("15:59")


# ---------------- 4 不重叠与冷却 ----------------


def test_4_no_overlap_and_cooldown():
    b = flat_bars("10:00", 12)
    acct = Account(equity_usd=10000)
    r = simulate_day(b, [(1, +1), (2, +1)], acct, SESSION, exit_rule=TIME2)
    _inv(r, 2)
    assert len(r.trades) == 1 and r.skipped["in_position"] == 1
    # 入场 1 → 第 3 根开盘平仓；平仓后 1 根（第 4 根）再入场
    r = simulate_day(
        b, [(1, +1), (4, +1)], acct, SESSION, StockPolicy(cooldown_bars=2), exit_rule=TIME2
    )
    _inv(r, 2)
    assert len(r.trades) == 1 and r.skipped["cooldown"] == 1
    r = simulate_day(
        b, [(1, +1), (4, +1)], acct, SESSION, StockPolicy(cooldown_bars=0), exit_rule=TIME2
    )
    _inv(r, 2)
    assert len(r.trades) == 2


# ---------------- 5 确定性 ----------------


def test_5_deterministic():
    b = bars(
        "10:00",
        [(100 + 0.1 * i, 100.3 + 0.1 * i, 99.8 + 0.1 * i, 100.1 + 0.1 * i) for i in range(20)],
    )
    rule = {"kind": "bracket", "tp_pct": 0.004, "sl_pct": 0.002, "max_bars": 5}
    sig = [(1, +1), (3, -1), (8, -1), (12, +1)]
    acct = Account(equity_usd=10000)
    r1 = simulate_day(b, sig, acct, SESSION, exit_rule=rule)
    r2 = simulate_day(b, sig, acct, SESSION, exit_rule=rule)
    assert [dataclasses.asdict(t) for t in r1.trades] == [dataclasses.asdict(t) for t in r2.trades]
    assert r1.skipped == r2.skipped and r1.equity_end == r2.equity_end
    _inv(r1, len(sig))


# ---------------- 6 跳空止损 ----------------


def test_6_gap_stop():
    b = bars(
        "10:00",
        [
            (99.99, 100.10, 99.90, 100.00),  # 入场根：开盘 99.99 + 1 tick = 100.00
            (93.00, 93.50, 92.50, 93.20),  # 跳空低开，开盘已越过止损 95
            (93.20, 93.40, 93.00, 93.10),
        ],
    )
    rule = {"kind": "bracket", "tp_px": 110.0, "sl_px": 95.0, "max_bars": 10}
    r = simulate_day(b, [(0, +1)], Account(equity_usd=10000), SESSION, exit_rule=rule)
    _inv(r, 1)
    (t,) = r.trades
    assert t.entry_px == pytest.approx(100.00)
    assert t.exit_px == pytest.approx(92.99)
    assert t.reason == "sl"
    assert t.exit_ts_ns == ts("10:02")  # 触发根（10:01）收盘时刻
