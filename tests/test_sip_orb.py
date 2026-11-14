# ruff: noqa: E501  （测试数据为长行）
"""R-057 SIP 开盘区间突破：标的名单过滤、合格集合、拆股剔除、RelVol、选股、无前视。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from zr.research import leakcheck as lc
from zr.research import sip_orb as so

NASDAQ = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZWZZT|NASDAQ TEST STOCK|Q|Y|N|100|N|N
ABCDW|ABC Corp. - Warrant|S|N|N|100|N|N
ABCDU|ABC Corp. - Unit|S|N|N|100|N|N
ABCDR|ABC Corp. - Rights|S|N|N|100|N|N
BABC|Baidu ADR - American Depositary Shares|Q|N|N|100|N|N
XYZ|XYZ Ltd. - Class A Ordinary Shares|Q|N|N|100|N|N
File Creation Time: 1113202618:01|||||||
"""
OTHER = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
A|Agilent Technologies, Inc. Common Stock|N|A|N|100|N|A
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK=B
PFX$A|PFX Corp 6% Preferred Series A|N|PFX-A|N|100|N|PFX-A
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
AMX|American Small Co Common Stock|A|AMX|N|100|N|AMX
File Creation Time: 1113202618:01||||||
"""


def test_parse_symbol_directory_keeps_nyse_nasdaq_common_only():
    u, listed = so.parse_symbol_directory(NASDAQ, OTHER)
    keep = set(u.loc[u["include"], "symbol"])
    assert keep == {"AAPL", "XYZ", "A", "BRK.B"}
    assert listed == "2026-11-13"
    r = u.set_index("symbol")["reason"]
    assert r["QQQ"] == "etf" and r["ZWZZT"] == "test" and r["ABCDW"] == "warrant"
    assert r["BABC"] == "depositary" and r["AMX"] == "exchange" and r["PFX$A"] == "preferred"
    assert so.lb_symbol("BRK.B") == "BRK.B.US"


def _daily(n=60, seed=0, split_at=None):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n).date
    c = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.005, n))
    h = np.maximum(o, c) * 1.02
    lo = np.minimum(o, c) * 0.98
    v = rng.integers(1_500_000, 3_000_000, n).astype(float)
    fwd = o.copy()
    if split_at is not None:  # 2:1 拆股：拆股前的前复权价 = 原始价 / 2
        fwd[:split_at] = o[:split_at] / 2
    return pd.DataFrame(
        {"date": days, "open": o, "high": h, "low": lo, "close": c, "volume": v, "fwd_open": fwd}
    )


def test_eligibility_uses_only_prior_days():
    d = _daily()
    e = so.daily_eligibility(d)
    # 首日没有前收，TR = H − L（fmax 忽略缺失）
    tr = np.fmax(
        d["high"] - d["low"],
        np.fmax((d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()),
    )
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    assert e["atr_prev"].iloc[30] == pytest.approx(atr.iloc[29])
    assert e["vol14_prev"].iloc[30] == pytest.approx(d["volume"].iloc[16:30].mean())
    assert not e["eligible"].iloc[:14].any()  # 不足 14 个前日
    # 截断：改动第 k 天及之后的数据，第 k 天之前的合格结果不变；第 k 天只用当日开盘
    d2 = d.copy()
    d2.loc[40:, ["high", "low", "close", "volume"]] *= 3
    e2 = so.daily_eligibility(d2)
    pd.testing.assert_frame_equal(e.iloc[:41], e2.iloc[:41])


def test_split_blocks_fifteen_days_from_split():
    d = _daily(split_at=30)
    e = so.daily_eligibility(d)
    assert (
        e["split_block"].iloc[30:45].all()
        and not e["split_block"].iloc[29]
        and not e["split_block"].iloc[45]
    )
    assert not e["eligible"].iloc[30:45].any()


def _bars5(days, rng, base=50.0):
    rows = []
    for k, day in enumerate(days):
        t0 = pd.Timestamp(f"{day} 09:30", tz="America/New_York")
        p = base + k * 0.1
        for j in range(78):
            o = p + rng.normal(0, 0.05)
            c = o + rng.normal(0, 0.05)
            rows.append(
                {
                    "ts": (t0 + pd.Timedelta(minutes=5 * j)).tz_convert("UTC"),
                    "open": o,
                    "high": max(o, c) + 0.02,
                    "low": min(o, c) - 0.02,
                    "close": c,
                    "volume": float(rng.integers(1000, 5000) * (10 if j == 0 else 1)),
                }
            )
            p = c
    return pd.DataFrame(rows)


def test_opening_range_relvol_and_doji():
    rng = np.random.default_rng(1)
    days = list(pd.bdate_range("2024-03-01", periods=20).date)
    b = _bars5(days, rng)
    i0 = b.index[(b["ts"].dt.tz_convert("America/New_York").dt.date == days[-1])][0]
    b.loc[i0, "close"] = b.loc[i0, "open"]  # 最后一天首根十字星
    o = so.opening_range(b)
    assert len(o) == 20 and np.isnan(o["relvol"].iloc[13]) and np.isfinite(o["relvol"].iloc[14])
    assert o["relvol"].iloc[15] == pytest.approx(
        o["or_vol"].iloc[15] / o["or_vol"].iloc[1:15].mean()
    )
    assert o["dir"].iloc[-1] == 0
    # 首根 K 之后的数据不影响 RelVol 与方向
    b2 = b.copy()
    later = b2["ts"].dt.tz_convert("America/New_York").dt.time > dt.time(9, 30)
    b2.loc[later, "volume"] *= 7
    pd.testing.assert_frame_equal(o, so.opening_range(b2))


def test_selection_top_n_order_and_filters():
    c = pd.DataFrame(
        {
            "date": [dt.date(2024, 1, 2)] * 5,
            "symbol": list("ABCDE"),
            "eligible": [True, True, True, False, True],
            "relvol": [3.0, 0.9, 5.0, 9.0, 3.0],
            "dir": [1, 1, -1, 1, 0],
        }
    )
    s = so.select(c, 2)
    assert s["symbol"].tolist() == ["C", "A"]  # D 不合格、B RelVol < 1、E 十字星


def test_opening_range_truncation_leakcheck():
    rng = np.random.default_rng(3)
    b = _bars5(list(pd.bdate_range("2024-03-01", periods=20).date), rng)
    b.index = pd.DatetimeIndex(b["ts"])

    def fn(x):
        o = so.opening_range(x.reset_index(drop=True))
        day = x["ts"].dt.tz_convert("America/New_York").dt.date.to_numpy()
        return (
            o.set_index("date")
            .reindex(day)[["relvol", "dir", "or_high", "or_low"]]
            .set_axis(x.index)
        )

    lc.assert_no_lookahead_frame(fn, b, cuts=8)
