"""R-054：规则触发、无前视（截断）、入场对齐、随机基线排除。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.research.r054_index_stock_layer import (
    MIN_NS,
    _cut,
    compute_signals,
    evaluate,
    load_rules,
    random_pool,
)

D1, D2 = dt.date(2025, 3, 5), dt.date(2025, 3, 6)


def _bars(day: dt.date, start: str, closes: np.ndarray) -> pd.DataFrame:
    """1 分钟 K：open = 上一根 close，high / low 各外扩 0.01。"""
    t0 = pd.Timestamp(f"{day} {start}", tz=ET)
    idx = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=i) for i in range(len(closes))])
    idx = idx.tz_convert("UTC")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + 0.01,
            "low": np.minimum(o, c) - 0.01,
            "close": c,
            "volume": 100.0,
        },
        index=idx,
    )


def _ext_rth(series_by_day: dict[dt.date, np.ndarray]):
    """每天 04:00 起 960 分钟（到 20:00）；RTH = 09:30–15:59。"""
    ext = pd.concat([_bars(d, "04:00", s) for d, s in series_by_day.items()])
    t = ext.index.tz_convert(ET)
    mod = t.hour * 60 + t.minute
    rth = ext[(mod >= 570) & (mod < 960)]
    return ext, rth


def _count(ext, rth, name):
    rules = load_rules()
    s = compute_signals(ext, rth, "C", {name: rules[name]})
    return s


def test_rule_units():
    n = 960
    # 开盘区间突破：09:30–09:34 上涨，之后区间内横盘，10:00 起站上区间高点
    orb_pos = np.full(n, 100.2)
    k0 = 330  # 09:30
    orb_pos[k0 : k0 + 5] = np.linspace(100.0, 100.4, 5)
    orb_pos[k0 + 30 :] = 101.0
    orb_neg = orb_pos.copy()
    orb_neg[k0 + 30 :] = 100.2
    ext, rth = _ext_rth({D2: orb_pos})
    s = _count(ext, rth, "survey_orb_breakout")
    assert len(s) == 1 and s["dir"].iloc[0] == 1
    ext, rth = _ext_rth({D2: orb_neg})
    assert len(_count(ext, rth, "survey_orb_breakout")) == 0

    # RSI(2) 回调：两天稳步上涨，第二天 10:00 起急跌 4 分钟后继续上涨
    up = 100 + 0.01 * np.arange(2 * n)
    dip = up.copy()
    j = n + 360  # 第二天 10:00
    dip[j : j + 4] = dip[j - 1] - 0.15 * np.arange(1, 5)
    dip[j + 4 :] = dip[j + 3] + 0.01 * np.arange(1, 2 * n - j - 3)
    for series, want in ((dip, 1), (up, 0)):
        ext, rth = _ext_rth({D1: series[:n], D2: series[n:]})
        s = _count(ext, rth, "survey_rsi2_pullback")
        assert len(s) == want and (want == 0 or s["dir"].iloc[0] == 1)

    # 均线回踩：上涨趋势中跌破 EMA50 三根 2 分钟 K 后收回
    pb = up.copy()
    pb[j : j + 6] = pb[j - 1] - 1.0
    pb[j + 6 :] = up[j + 6 :]
    for series, want in ((pb, 1), (up, 0)):
        ext, rth = _ext_rth({D1: series[:n], D2: series[n:]})
        s = _count(ext, rth, "survey_ema_pullback")
        assert len(s) == want and (want == 0 or s["dir"].iloc[0] == 1)


def test_truncation_no_lookahead():
    rng = np.random.default_rng(7)
    n = 960
    walk = 100 * np.exp(np.cumsum(rng.normal(0, 0.0015, 3 * n)))
    days = [dt.date(2025, 3, 4), D1, D2]
    ext, rth = _ext_rth({d: walk[i * n : (i + 1) * n] for i, d in enumerate(days)})
    rules = load_rules()
    compared = 0
    for conv in ("C", "D027"):
        full = compute_signals(ext, rth, conv, rules)
        for hh, mm in ((9, 41), (10, 7), (11, 30), (13, 3), (15, 20)):
            t = pd.Timestamp(dt.datetime.combine(D2, dt.time(hh, mm)), tz=ET).value
            tr = compute_signals(_cut(ext, t), _cut(rth, t), conv, rules)
            a = full[full["sig_ns"] <= t].reset_index(drop=True)
            b = tr[tr["sig_ns"] <= t].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)
            compared += len(a)
    assert compared > 0


def test_entry_is_next_minute_open():
    n = 390
    c = 100 + 0.1 * np.arange(n)
    rth = _bars(D2, "09:30", c)
    rth["open"] = 50 + np.arange(n) * 1.0  # 开盘价与收盘价无关，便于区分
    sig_ns = pd.Timestamp(f"{D2} 10:00", tz=ET).value  # 信号 K（09:58–10:00）收盘时刻
    sig = pd.DataFrame([{"day": D2, "rule": "survey_rsi2_pullback", "dir": 1, "sig_ns": sig_ns}])
    ev, cnt = evaluate(sig, rth, load_rules(), 0, "t")
    assert cnt["entry_dropped"] == 0
    assert ev["entry"].iloc[0] == rth.loc[pd.Timestamp(f"{D2} 10:00", tz=ET), "open"] == 80.0
    # 5 分钟收益 = 10:04 那根 K 的收盘 / 入场价 − 1
    assert ev["ret5"].iloc[0] == (c[34] / 80.0 - 1) * 100
    assert (sig_ns - pd.Timestamp(f"{D2} 09:30", tz=ET).value) // MIN_NS == 30


def test_random_baseline_excludes_triggers():
    valid = np.ones(390, bool)
    trig = np.array([100, 110])
    pool = random_pool(100, trig, 2, 370, valid)
    d = np.min(np.abs(pool[:, None] - trig[None, :]), axis=1)
    assert len(pool) > 0 and (d > 2).all()
    assert (np.abs(pool - 100) <= 30).all()
    for j in (98, 99, 100, 101, 102, 108, 112):
        assert j not in pool
