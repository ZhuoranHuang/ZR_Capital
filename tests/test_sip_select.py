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


def test_gap_resets_history_ticker_reuse():
    """代码复用：断档超过 30 个自然日后，新标的的前 14 天不得合格，RelVol 只用断档后的开盘区间。"""
    a = _daily(n=40, seed=1)
    b = _daily(n=40, seed=2)
    b["date"] = pd.bdate_range("2025-06-02", periods=40).date  # 一年多以后同一代码重新出现
    b[["d_volume", "or_volume"]] = b[["d_volume", "or_volume"]] * 50  # 新公司成交量大得多
    b["or_bars"] = 5
    b[["or_open", "or_high", "or_low", "or_close"]] = (
        b[["or_open", "or_high", "or_low", "or_close"]].ffill().bfill()
    )
    b["or_volume"] = b["or_volume"].fillna(b["or_volume"].median())
    g = pd.concat([a, b], ignore_index=True)
    f = sel.ticker_features(g, [])
    new = f[f["date"] >= b["date"].iloc[0]].reset_index(drop=True)
    assert not new.loc[:13, "eligible"].any()  # 前 14 天没有自己的历史
    assert new.loc[:13, "relvol"].isna().all()
    assert (
        new["segment"].eq(1).all() and f.loc[f["date"] < b["date"].iloc[0], "segment"].eq(0).all()
    )
    alone = sel.ticker_features(b, [])  # 单独计算新标的，结果应完全一致
    cols = ["atr_prev", "vol14_prev", "eligible", "relvol", "dir", "split_block"]
    pd.testing.assert_frame_equal(new[cols], alone[cols], check_dtype=False)
    # 截断测试仍通过
    assert_no_lookahead_frame(lambda x: sel.ticker_features(x, [])[cols], g, cuts=12)


def test_short_gap_keeps_history():
    a = _daily(n=60, seed=3)
    days = list(a["date"])
    a["date"] = days[:30] + [
        d + dt.timedelta(days=21) for d in days[30:]
    ]  # 21 个自然日的断档：不切段
    f = sel.ticker_features(a, [])
    assert f["segment"].eq(0).all() and f["atr_prev"].notna().sum() == 60 - 14


def test_split_before_history_does_not_block():
    d = _daily()
    f0 = sel.ticker_features(d, [])
    f1 = sel.ticker_features(d, [d["date"].iloc[0] - dt.timedelta(days=200)])  # 数据起点之前的拆股
    assert not f1["split_block"].any()
    pd.testing.assert_series_equal(f0["eligible"], f1["eligible"])
