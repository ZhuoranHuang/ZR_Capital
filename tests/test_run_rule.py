"""规则 runner（scripts/run_rule.py）：无前视检查、入场映射、计数恒等、期权模式。"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from zr.sim.engine import Account
from zr.sim.stock import StockPolicy

ET = ZoneInfo("America/New_York")
REPO = Path(__file__).resolve().parents[1]
DAY = dt.date(2026, 3, 3)
SESSION = {"first_entry": "09:30", "last_entry": "15:40", "flat": "15:50"}


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass 需要模块已登记
    spec.loader.exec_module(mod)
    return mod


RR = _load(REPO / "scripts" / "run_rule.py")


def assert_no_lookahead(signal_fn, bars: pd.DataFrame, cuts: int = 3) -> None:
    """对每个截断点 k：signal_fn(bars.iloc[:k]) 的前 k 个值必须与 signal_fn(bars) 的前 k 个值
    相同。"""
    full = np.asarray(signal_fn(bars), float)
    n = len(bars)
    for k in np.linspace(n // 4, n - 1, cuts).astype(int):
        part = np.asarray(signal_fn(bars.iloc[:k]), float)
        assert len(part) == k
        msg = f"截断点 k={k} 的信号与全量不同（前视）"
        np.testing.assert_array_equal(part, full[:k], err_msg=msg)


def bars_1m(day: dt.date, closes: np.ndarray, opens: np.ndarray | None = None) -> pd.DataFrame:
    """RTH 1 分钟 K（09:30 起），索引 = UTC 开盘时刻（与 load_bars 相同）。"""
    t0 = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET)
    idx = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=i) for i in range(len(closes))])
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]] if opens is None else np.asarray(opens, float)
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + 0.01,
            "low": np.minimum(o, c) - 0.01,
            "close": c,
            "volume": 100.0,
        },
        index=idx.tz_convert("UTC"),
    )


def walk(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))


def at_label(rule_bars: pd.DataFrame, hhmm: str) -> pd.Series:
    """在标签 = hhmm（美东）的那根 PERIOD K 上给 +1。"""
    lab = rule_bars.index.tz_convert(ET).strftime("%H:%M")
    return pd.Series(np.where(lab == hhmm, 1, 0), index=rule_bars.index)


def rule_ns(period: int, exit_rule: dict, fn, direction: str = "both", contract=None):
    return types.SimpleNamespace(
        PERIOD=period,
        DIRECTION=direction,
        EXIT=exit_rule,
        CONTRACT=contract or {"dte_max": 1, "otm_steps": 0},
        signal=fn,
    )


# ---------------- 无前视 ----------------


def test_example_rule_no_lookahead():
    ex = _load(REPO / "rules" / "example_boll_macd.py")
    b = pd.concat([bars_1m(d, walk(390, s)) for s, d in enumerate([DAY, DAY + dt.timedelta(1)])])
    rb = RR.resample_rule_bars(b, ex.PERIOD)
    assert_no_lookahead(ex.signal, rb, cuts=3)
    assert set(np.unique(ex.signal(rb))) <= {-1, 0, 1}


def test_lookahead_is_caught():
    b = bars_1m(DAY, walk(390))

    def peek(x: pd.DataFrame) -> pd.Series:
        return np.sign(x["close"].shift(-1) - x["close"]).fillna(0).astype(int)

    with pytest.raises(AssertionError):
        assert_no_lookahead(peek, b, cuts=3)


# ---------------- 正股模式 ----------------


def test_stock_entry_is_minute_open_at_signal_close():
    n = 390
    closes = 100 + 0.01 * np.arange(n)
    opens = 50 + np.arange(n) * 0.5  # 开盘价与收盘无关，便于定位
    b = bars_1m(DAY, closes, opens)
    rule = rule_ns(2, {"kind": "time", "bars": 3}, lambda x: at_label(x, "10:00"))
    res = RR.run_rule(b, rule, Account(equity_usd=10000), SESSION, "stock")
    t = res.trades[res.trades["skip_reason"] == ""]
    assert len(t) == 1
    row = t.iloc[0]
    # 2 分钟 K 10:00–10:02 收盘 → 入场 = 10:02 那根 1 分钟 K 的开盘 + 1 tick
    assert row["signal_time"] == "10:02:00"
    assert row["stock_entry_time"] == "10:02:00"
    assert row["stock_entry_px"] == pytest.approx(opens[32] + 0.01)
    # time 3 根 × PERIOD 2 = 6 根 1 分钟 K → 10:08 开盘
    assert row["stock_exit_time"] == "10:08:00"
    assert row["stock_exit_px"] == pytest.approx(opens[38] - 0.01)
    assert res.equity_end == pytest.approx(10000 + row["pnl_usd"])


def test_stock_counts_identity_and_no_bar():
    b = bars_1m(DAY, walk(390))

    def sig(x):
        s = pd.Series(0, index=x.index)
        s.iloc[::7] = 1
        s.iloc[3::11] = -1
        s.iloc[-1] = 1  # 最后一根收盘 = 16:00，没有入场 K → no_bar
        return s

    rule = rule_ns(2, {"kind": "bracket", "tp_pct": 0.002, "sl_pct": 0.002, "max_bars": 5}, sig)
    res = RR.run_rule(b, rule, Account(equity_usd=10000), SESSION, "stock")
    n_sig = int((sig(RR.resample_rule_bars(b, 2)) != 0).sum())
    n_tr = int((res.trades["skip_reason"] == "").sum())
    assert sum(res.skipped.values()) + n_tr == n_sig == len(res.trades)
    assert res.skipped["no_bar"] >= 1


def test_direction_filter():
    b = bars_1m(DAY, walk(390))
    rule = rule_ns(2, {"kind": "time", "bars": 2}, lambda x: -at_label(x, "10:00"), "long")
    res = RR.run_rule(b, rule, Account(equity_usd=10000), SESSION, "stock")
    assert len(res.trades) == 0


# ---------------- 期权模式 ----------------

EXP = DAY  # 0DTE


def chain(day: dt.date, strikes, minutes: int = 390, spread: float = 0.02, drop=()) -> pd.DataFrame:
    """合成 1 分钟期权报价（09:30 起每分钟）：mid 恒定，call 价随行权价递减。"""
    rows = []
    t0 = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET)
    for k in strikes:
        for right in ("call", "put"):
            sym = f"MU{EXP:%y%m%d}{'C' if right == 'call' else 'P'}{int(k * 1000)}.US"
            mid = max(0.5, (100 - k if right == "call" else k - 100) + 1.0)
            for m in range(minutes + 1):
                if (sym, m) in drop:
                    continue
                rows.append(
                    {
                        "ts": t0 + pd.Timedelta(minutes=m),
                        "symbol": sym,
                        "expiry": EXP,
                        "strike": float(k),
                        "right": right,
                        "bid": mid - spread / 2,
                        "ask": mid + spread / 2,
                    }
                )
    q = pd.DataFrame(rows)
    q["ts"] = pd.DatetimeIndex(q["ts"]).tz_convert("UTC")
    return q


def test_option_exit_at_stock_exit():
    b = bars_1m(DAY, np.full(390, 100.0))
    rule = rule_ns(
        2,
        {"kind": "time", "bars": 5},
        lambda x: at_label(x, "10:00"),
        contract={"dte_max": 1, "otm_steps": 1},
    )
    q = chain(DAY, [98, 99, 100, 101, 102])
    res = RR.run_rule(b, rule, Account(equity_usd=10000), SESSION, "option", quotes=q)
    t = res.trades[res.trades["skip_reason"] == ""]
    assert len(t) == 1
    row = t.iloc[0]
    assert row["contract"] == "MU260303C101000.US"  # 平值 100 → 虚值 1 档 = 101 call
    assert row["stock_exit_time"] == "10:12:00"  # 10:02 入场 + 5 × 2 分钟（time = 开盘时刻）
    # 引擎 at = 10:12 + 2 秒延迟 → 1 分钟报价下为 10:13
    assert row["option_entry_time"] == "10:03:00"
    assert row["option_exit_time"] == "10:13:00"
    assert row["option_exit_time"] >= row["stock_exit_time"]


def test_option_no_contract_wide_spread_no_fallback():
    b = bars_1m(DAY, np.full(390, 100.0))
    rule = rule_ns(2, {"kind": "time", "bars": 5}, lambda x: at_label(x, "10:00"))
    acct = Account(equity_usd=10000)
    # 到期日不在 dte_max 内 → no_contract
    q = chain(DAY, [99, 100, 101])
    q["expiry"] = DAY + dt.timedelta(days=7)
    res = RR.run_rule(b, rule, acct, SESSION, "option", quotes=q)
    assert res.skipped["no_contract"] == 1
    # 价差过宽 → wide_spread
    q = chain(DAY, [99, 100, 101], spread=1.5)
    res = RR.run_rule(b, rule, acct, SESSION, "option", quotes=q)
    assert res.skipped["wide_spread"] == 1
    # 选中合约入场分钟缺报价 → no_quote，不回退到别的行权价
    q = chain(DAY, [99, 100, 101], drop={("MU260303C100000.US", 33)})
    res = RR.run_rule(b, rule, acct, SESSION, "option", quotes=q)
    assert res.skipped["no_quote"] == 1
    assert (res.trades["skip_reason"] == "").sum() == 0


def test_option_daily_loss_and_max_entries():
    b = bars_1m(DAY, np.full(390, 100.0))

    def sig(x):
        s = pd.Series(0, index=x.index)
        s.iloc[5:150:10] = 1
        return s

    q = chain(DAY, [99, 100, 101], spread=0.1)  # 每笔亏价差 + 费用
    rule = rule_ns(2, {"kind": "time", "bars": 2}, sig)
    n_sig = int((sig(RR.resample_rule_bars(b, 2)) != 0).sum())
    acct = Account(equity_usd=10000)
    res = RR.run_rule(
        b, rule, acct, SESSION, "option", quotes=q, stock_policy=StockPolicy(max_entries_per_day=3)
    )
    assert (res.trades["skip_reason"] == "").sum() == 3
    assert res.skipped["max_entries"] == n_sig - 3
    # 已实现亏损 ≥ 0.1% × 期初权益后不再开仓：第一笔亏损后其余全部 daily_loss
    res = RR.run_rule(b, rule, acct, SESSION, "option", quotes=q, max_daily_loss_pct=0.001)
    t = res.trades[res.trades["skip_reason"] == ""]
    assert len(t) == 1 and t["pnl_usd"].iloc[0] < -10
    assert res.skipped["daily_loss"] == n_sig - 1
    assert sum(res.skipped.values()) + len(t) == n_sig
