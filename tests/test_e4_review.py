"""E4 复核：前视截断测试（SMA120、KDJ、摆动点、梯形出场、ATR 止损）与口径测试；合成数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zr.research import e4_mu_stock as old
from zr.research import e4_review as er


def _bars(days: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2026-03-02", periods=days):
        start = pd.Timestamp(d.date().isoformat() + " 09:30", tz="America/New_York")
        idx += list(pd.date_range(start, periods=78, freq="5min"))
    idx = pd.DatetimeIndex(idx).tz_convert("UTC")
    c = 100 + np.cumsum(rng.normal(0, 0.25, len(idx)))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.05, len(idx))
    h = np.maximum(o, c) + rng.uniform(0, 0.3, len(idx))
    lo = np.minimum(o, c) - rng.uniform(0, 0.3, len(idx))
    return pd.DataFrame(
        {
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": rng.integers(1000, 5000, len(idx)).astype(float),
        },
        index=idx,
    )


KEYS = ("ma", "k", "d", "atr", "sw_lo", "sw_hi", "long", "short")


@pytest.mark.parametrize("p", [er.BASELINE, er.VARIANTS["V1b n=5"], er.VARIANTS["V7 组合"]])
@pytest.mark.parametrize("cut", [300, 517, 801])
def test_truncation_indicators_and_closed_trades(p, cut):
    df = _bars()
    full, part = er.prepare(df, p), er.prepare(df.iloc[:cut], p)
    for k in KEYS:
        a, b = np.asarray(full[k][:cut], float), np.asarray(part[k], float)
        assert np.array_equal(a, b, equal_nan=True), k
    tf, tp = er.backtest(full, p), er.backtest(part, p)
    keep = lambda t: t[t["exit_i"] < cut - 1].reset_index(drop=True) if len(t) else t  # noqa: E731
    a, b = keep(tf), keep(tp)
    assert len(a) == len(b) and len(a) > 0
    for col in ("signal_i", "exit_i", "entry_px", "exit_px", "net", "reason", "stop"):
        assert (a[col].to_numpy() == b[col].to_numpy()).all(), col


def test_old_config_reproduces_original_engine():
    df = _bars(seed=3)
    for ma, m in ((120, 1.5), (127, 1.0)):
        po = old.Params(ma=ma, stop_mult=m)
        to = old.backtest(old.prepare(df, po), po)
        pn = er.P(ma=ma, stop_mult=m, ladder="ratchet", strict=False)
        tn = er.backtest(er.prepare(df, pn), pn)
        assert len(to) == len(tn) and len(to) > 0
        assert np.allclose(to["net"], tn["net"]) and (to["exit_i"] == tn["exit_i"]).all()


def test_swing_strict_and_recent_ladder():
    day = np.zeros(9, dtype=object)
    day[:] = "d"
    low = np.array([5, 4, 3, 4, 5, 4, 3.5, 4, 5], float)
    high = low + 1
    lo_s, _ = er.swings(low, high, day, 2, strict=True)
    assert lo_s[4] == 3 and lo_s[8] == 3.5  # bar 2、bar 6 分别在 bar 4、bar 8 收盘确认
    eq = np.array([5, 4, 3, 3, 5, 6, 7, 8, 9], float)  # 并列低点：strict 不确认，原口径确认
    assert np.isnan(er.swings(eq, eq + 1, day, 2, True)[0]).all()
    assert np.isfinite(er.swings(eq, eq + 1, day, 2, False)[0]).any()
