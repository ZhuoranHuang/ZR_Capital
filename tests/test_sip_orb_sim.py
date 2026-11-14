# ruff: noqa: E501  （测试数据为长行）
"""R-057 回测执行：入场 / 止损 / 平仓成交价、同根歧义两种算法、仓位、费用、按日复利。"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from zr.costs.fees import stock_order_fees
from zr.research import sip_orb as so

DAY = dt.date(2024, 3, 4)


def bars1(rows, start="09:30"):
    """rows = [(open, high, low, close)]，从 start 起每分钟一根。"""
    t0 = pd.Timestamp(f"{DAY} {start}", tz="America/New_York")
    return pd.DataFrame(
        {
            "ts": [(t0 + pd.Timedelta(minutes=i)).tz_convert("UTC") for i in range(len(rows))],
            **{k: [r[j] for r in rows] for j, k in enumerate(("open", "high", "low", "close"))},
        }
    )


FLAT = (10.0, 10.05, 9.95, 10.0)


def test_long_entry_stop_and_close_exit():
    # 区间高 10.20，ATR 1.0 → 止损距 0.10；09:35 那根最高 10.25 触发，开盘 10.10 < 10.20 → 成交 10.20 + 0.01
    rows = (
        [FLAT] * 5
        + [(10.10, 10.25, 10.15, 10.22)]
        + [(10.22, 10.30, 10.18, 10.28)] * 3
        + [(10.28, 10.40, 10.25, 10.35)]
    )
    t = so.simulate_trade(bars1(rows), 1, 10.20, 9.80, 1.0, slip=0.01, conservative=True)
    assert t["entry"] == pytest.approx(10.21) and t["stop"] == pytest.approx(10.11)
    assert t["reason"] == "close" and t["exit"] == pytest.approx(10.34)
    assert t["entry_ts"] == bars1(rows)["ts"].iloc[5]


def test_entry_bar_stop_conservative_vs_optimistic():
    rows = [FLAT] * 5 + [(10.10, 10.25, 10.05, 10.22)] + [(10.22, 10.30, 10.18, 10.28)] * 4
    c = so.simulate_trade(bars1(rows), 1, 10.20, 9.80, 1.0, slip=0.01, conservative=True)
    o = so.simulate_trade(bars1(rows), 1, 10.20, 9.80, 1.0, slip=0.01, conservative=False)
    assert c["reason"] == "stop" and c["exit"] == pytest.approx(
        10.11 - 0.01
    )  # 入场那根按止损价，不用开盘价
    assert o["reason"] == "close"


def test_gap_through_stop_fills_at_open_and_short_mirror():
    rows = [FLAT] * 5 + [(10.25, 10.30, 10.24, 10.26), (9.90, 9.95, 9.85, 9.90)] + [FLAT] * 3
    t = so.simulate_trade(bars1(rows), 1, 10.20, 9.80, 1.0, slip=0.01, conservative=True)
    assert (
        t["entry"] == pytest.approx(10.26)
        and t["exit"] == pytest.approx(9.89)
        and t["reason"] == "stop"
    )  # 开盘 10.25 > 区间高 → 按开盘
    rows = [FLAT] * 5 + [(9.85, 9.86, 9.70, 9.75)] + [(9.75, 9.76, 9.60, 9.65)] * 3
    s = so.simulate_trade(bars1(rows), -1, 10.20, 9.80, 1.0, slip=0.01, conservative=True)
    assert (
        s["entry"] == pytest.approx(9.79)
        and s["stop"] == pytest.approx(9.89)
        and s["exit"] == pytest.approx(9.66)
    )


def test_no_trigger_and_no_entry_before_0935():
    rows = [(10.0, 10.5, 9.9, 10.0)] * 5 + [FLAT] * 5  # 09:30–09:34 的高点不算
    assert so.simulate_trade(bars1(rows), 1, 10.20, 9.80, 1.0, slip=0.01, conservative=True) is None


def test_sizing_risk_cap_and_unlevered():
    # 权益 100,000、N = 20：风险 50 美元；R = 0.10 × ATR 0.5 = 0.05 → 1,000 股；名义上限 4 × 5,000 / 10 = 2,000 股
    assert so.shares(100_000, 20, atr=0.5, entry=10.0, lev=4.0) == 1000
    assert (
        so.shares(100_000, 20, atr=0.05, entry=10.0, lev=4.0) == 2000
    )  # 风险算出 10,000 股，受杠杆上限
    assert so.shares(100_000, 20, atr=0.05, entry=10.0, lev=1.0) == 500
    assert (
        so.shares(1498.69, 1, atr=1.0, entry=50.0, lev=1.0) == 29
    )  # min(floor(14.99 / 0.1) = 149, floor(1498.69 / 50) = 29)


def test_fees_two_schemes():
    assert so.trade_fees(100, 10.0, 10.5, 1, "per_share") == pytest.approx(0.70)
    want = float(
        stock_order_fees(100, "Sell", 10.0)["total"] + stock_order_fees(100, "Buy", 9.5)["total"]
    )
    assert so.trade_fees(100, 10.0, 9.5, -1, "d013") == pytest.approx(
        want
    )  # 做空开仓 Sell、平仓 Buy


def test_portfolio_compounding_identity():
    T = pd.DataFrame(
        {
            "date": [DAY, DAY, DAY + dt.timedelta(1)],
            "rank": [1, 2, 1],
            "dir": [1, -1, 1],
            "atr": [0.5, 0.5, 0.5],
            "entry": [10.0, 20.0, 10.0],
            "exit": [10.2, 19.9, 9.9],
        }
    )
    r = so.portfolio(T, equity0=100_000, n=20, lev=4.0, fee="d013")
    assert r["equity"].iloc[-1] == pytest.approx(100_000 + r["trades"]["pnl"].sum())
    d1 = r["daily"].iloc[0]
    assert d1["ret"] == pytest.approx(r["trades"]["pnl"].iloc[:2].sum() / 100_000)
    assert r["trades"]["shares"].iloc[2] == so.shares(
        100_000 + r["trades"]["pnl"].iloc[:2].sum(), 20, 0.5, 10.0, 4.0
    )
