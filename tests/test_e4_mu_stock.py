"""E4 正股专项回测：入场 / 止损 / 平仓 / 加仓 / 费用的规则单测（合成 5 分钟 K）。"""

import datetime as dt

import numpy as np
import pandas as pd

from zr.research import e4_mu_stock as e4

P = e4.Params(ma=3, stop_mult=1.0, max_adds=2, lot=100, tick=0.01)


def _bars(closes, day=dt.date(2024, 3, 5), start="09:30"):
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq="5min", tz="America/New_York")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + 0.05,
            "low": np.minimum(o, c) - 0.05,
            "close": c,
            "volume": 1000.0,
        },
        index=idx.tz_convert("UTC"),
    )


def _arrays(df, long_at=(), short_at=(), atr=0.5):
    a = e4.prepare(df, P)
    n = len(df)
    a["long"] = np.isin(np.arange(n), long_at)
    a["short"] = np.isin(np.arange(n), short_at)
    a["atr"] = np.full(n, atr)
    a["sw_lo"] = np.full(n, np.nan)
    a["sw_hi"] = np.full(n, np.nan)
    return a


def test_entry_next_open_with_slippage_and_eod_flat():
    df = _bars([100 + 0.01 * i for i in range(78)])  # 09:30…15:55，缓慢上涨，不触止损
    a = _arrays(df, long_at=[10])
    t = e4.simulate_entry(a, 10, 1, P)
    assert np.isclose(t["entry_px"], a["open"][11] + 0.01)
    # 15:45 那根 bar 收盘平仓（= 15:50）
    assert t["exit_ts"].tz_convert("America/New_York").strftime("%H:%M") == "15:45"
    assert t["reason"] == "eod"
    assert np.isclose(t["exit_px"], a["close"][t["exit_i"]] - 0.01)
    fees = e4.fee(100, "Buy", t["entry_px"]) + e4.fee(100, "Sell", t["exit_px"])
    assert np.isclose(t["net"], 100 * (t["exit_px"] - t["entry_px"]) - fees)


def test_stop_fill_at_stop_or_gap_open():
    closes = [100.0] * 12 + [99.0, 98.0] + [98.0] * 10
    df = _bars(closes)
    a = _arrays(df, long_at=[5], atr=0.5)
    t = e4.simulate_entry(a, 5, 1, P)
    stop = t["entry_px"] - 0.5
    k = t["exit_i"]
    assert t["reason"] == "stop"
    assert np.isclose(t["exit_px"], min(a["open"][k], stop) - 0.01)


def test_short_mirror_and_no_entry_after_1545():
    df = _bars([100 - 0.01 * i for i in range(78)])
    a = _arrays(df, short_at=[20, 75])
    t = e4.simulate_entry(a, 20, -1, P)
    assert np.isclose(t["entry_px"], a["open"][21] - 0.01)
    assert t["gross"] > 0
    # bar 75 = 15:45 收盘信号 → 下一根 15:50 开盘不允许入场
    assert e4.simulate_entry(a, 75, -1, P) is None


def test_adds_only_in_profit_and_at_most_two():
    df = _bars([100 + 0.05 * i for i in range(60)])
    a = _arrays(df, long_at=[5, 10, 15, 20, 25], atr=5.0)
    t = e4.simulate_entry(a, 5, 1, P)
    assert t["adds"] == 2 and t["shares"] == 300
    # 下跌行情中的同向信号不加仓（浮亏）
    df2 = _bars([100 - 0.01 * i for i in range(60)])
    a2 = _arrays(df2, long_at=[5, 10, 15], atr=50.0)
    assert e4.simulate_entry(a2, 5, 1, P)["adds"] == 0


def test_backtest_one_position_at_a_time():
    df = _bars([100 + 0.01 * i for i in range(78)])
    a = _arrays(df, long_at=[5, 6, 7], atr=50.0)
    tr = e4.backtest(a, e4.Params(ma=3, stop_mult=1.0, max_adds=0))
    assert len(tr) == 1  # 5 入场后持有到收盘，6/7 的信号被忽略（max_adds=0）


def test_random_direction_contains_real_when_all_same():
    df = _bars([100 + 0.01 * i for i in range(78)])
    a = _arrays(df, long_at=[5], atr=50.0)
    tr = e4.backtest(a, P)
    rnd = e4.random_direction(a, tr, P, n=200, seed=1)
    assert set(np.round(rnd, 6)) <= {
        round(tr["net"].iloc[0], 6),
        round(e4.simulate_entry(a, 5, -1, P)["net"], 6),
    }
