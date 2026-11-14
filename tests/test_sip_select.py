# ruff: noqa: E501
"""R-057 Massive 版选股：拆股接口日期的剔除期、开盘区间特征与截断测试（无前视）。"""

import datetime as dt

import numpy as np
import pandas as pd

from zr.research import sip_orb as so
from zr.research import sip_select as sel
from zr.research.leakcheck import assert_no_lookahead_frame


def _daily(n=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-02", periods=n).date
    c = 20 + rng.normal(0, 0.5, n).cumsum()
    o = c + rng.normal(0, 0.2, n)
    h = np.maximum(o, c) + rng.uniform(0.2, 1.0, n)
    lo = np.minimum(o, c) - rng.uniform(0.2, 1.0, n)
    v = rng.integers(500_000, 3_000_000, n).astype(float)
    oo = o + rng.normal(0, 0.05, n)
    oc = oo + np.where(rng.random(n) < 0.1, 0.0, rng.normal(0, 0.1, n))
    df = pd.DataFrame(
        {
            "date": days,
            "d_open": o,
            "d_high": h,
            "d_low": lo,
            "d_close": c,
            "d_volume": v,
            "or_open": oo,
            "or_high": np.maximum(oo, oc) + 0.05,
            "or_low": np.minimum(oo, oc) - 0.05,
            "or_close": oc,
            "or_volume": rng.integers(10_000, 300_000, n).astype(float),
            "or_bars": 5,
        }
    )
    df.loc[[7, 30], "or_bars"] = 0  # 两天没有开盘区间
    df.loc[[7, 30], ["or_open", "or_high", "or_low", "or_close", "or_volume"]] = np.nan
    return df


def test_split_dates_block():
    d = _daily()
    s = d["date"].iloc[25]
    f = sel.ticker_features(d, [s])
    assert f.loc[:24, "split_block"].eq(False).all()
    assert f.loc[25:39, "split_block"].all() and not f.loc[40:, "split_block"].any()
    # 执行日不是交易日：取之后第一个交易日
    f2 = sel.ticker_features(d, [s - dt.timedelta(days=1) if s.weekday() > 0 else s])
    assert f2.loc[25, "split_block"]


def test_or_stats_relvol_excludes_days_without_or():
    d = _daily()
    f = sel.ticker_features(d, [])
    x = d[d["or_bars"] > 0].reset_index(drop=True)
    base = x["or_volume"].rolling(14).mean().shift(1)
    exp = x["or_volume"] / base
    got = f.loc[f["or_open"].notna(), "relvol"].reset_index(drop=True)
    pd.testing.assert_series_equal(got, exp, check_names=False)
    assert f.loc[d["or_bars"] == 0, "relvol"].isna().all()
    assert set(f["dir"].dropna().unique()) <= {-1, 0, 1}


def test_ticker_features_truncation_no_lookahead():
    d = _daily()
    s = d["date"].iloc[25]
    cols = ["atr_prev", "vol14_prev", "split_block", "eligible", "relvol", "dir"]
    assert_no_lookahead_frame(lambda x: sel.ticker_features(x, [s])[cols], d, cuts=12)


def test_eligibility_matches_wilder_and_thresholds():
    d = _daily()
    f = sel.ticker_features(d, [])
    atr = so.wilder_atr(d["d_high"], d["d_low"], d["d_close"]).shift(1)
    pd.testing.assert_series_equal(f["atr_prev"], atr, check_names=False)
    v14 = d["d_volume"].rolling(14).mean().shift(1)
    exp = (d["d_open"] > 5) & (v14 >= 1e6) & (atr > 0.5)
    assert (f["eligible"] == exp.fillna(False)).all()
