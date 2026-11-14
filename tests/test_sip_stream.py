# ruff: noqa: E501
"""zr.data.sip_stream 纯函数：常规时段 / 开盘区间聚合、全时段标签、逐日成交结果与 simulate_trade 一致。"""

import numpy as np
import pandas as pd

from zr.data import sip_stream as ss
from zr.research import sip_orb as so


def _minute(day="2024-04-15", tickers=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{day} 04:00", tz="America/New_York")
    idx = pd.date_range(start, periods=16 * 60, freq="1min")  # 04:00–19:59
    rows = []
    for t in tickers:
        c = 50 + rng.normal(0, 0.05, len(idx)).cumsum()
        o = np.r_[c[0], c[:-1]]
        for i, ts in enumerate(idx):
            if rng.random() < 0.1:  # 无成交的分钟没有行
                continue
            rows.append(
                (
                    t,
                    float(rng.integers(100, 5000)),
                    o[i],
                    c[i],
                    max(o[i], c[i]) + 0.02,
                    min(o[i], c[i]) - 0.02,
                    ts.value,
                    3,
                )
            )
    df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "volume",
            "open",
            "close",
            "high",
            "low",
            "window_start",
            "transactions",
        ],
    )
    return df.sample(frac=1, random_state=1).reset_index(drop=True)  # 打乱顺序


def test_day_tables_rth_and_or():
    m = _minute()
    t = ss.day_tables(m, 960)
    x = m[m.ticker == "AAA"].sort_values("window_start")
    et = pd.to_datetime(x.window_start, unit="ns", utc=True).dt.tz_convert("America/New_York")
    mm = et.dt.hour * 60 + et.dt.minute
    r = x[(mm >= 570) & (mm < 960)]
    o = x[(mm >= 570) & (mm < 575)]
    a = t.loc["AAA"]
    assert a.r_open == r.open.iloc[0] and a.r_close == r.close.iloc[-1]
    assert a.r_high == r.high.max() and a.r_low == r.low.min() and a.r_volume == r.volume.sum()
    assert (
        a.or_open == o.open.iloc[0]
        and a.or_close == o.close.iloc[-1]
        and a.or_volume == o.volume.sum()
    )
    assert a.or_bars == len(o) and a.or_first_min == mm[o.index[0]]


def test_day_tables_early_close():
    m = _minute()
    t = ss.day_tables(m, 13 * 60)
    x = m[m.ticker == "BBB"]
    et = pd.to_datetime(x.window_start, unit="ns", utc=True).dt.tz_convert("America/New_York")
    mm = et.dt.hour * 60 + et.dt.minute
    assert t.loc["BBB"].r_bars == int(((mm >= 570) & (mm < 780)).sum())


def test_extra_bars_sessions():
    m = _minute(tickers=("MU", "XYZ"))
    e = ss.extra_bars(m, 960)
    assert set(e.symbol) == {"MU"} and e.ts.is_monotonic_increasing
    et = e.ts.dt.tz_convert("America/New_York")
    mm = et.dt.hour * 60 + et.dt.minute
    assert (e.session[mm < 570] == "Pre").all() and (e.session[mm >= 960] == "Post").all()
    assert (e.session[(mm >= 570) & (mm < 960)] == "Intraday").all()


def test_outcomes_match_simulate_trade():
    m = _minute(seed=3)
    t = ss.day_tables(m, 960)
    groups = ss.rth_groups(m, 960, {"AAA", "BBB"})
    rows = []
    for s in ("AAA", "BBB"):
        a = t.loc[s]
        d = int(np.sign(round(a.or_close - a.or_open, 6))) or 1
        rows.append(
            {"symbol": s, "dir": d, "or_high": a.or_high, "or_low": a.or_low, "atr_prev": 0.6}
        )
    rows = pd.DataFrame(rows)
    out = ss.outcomes(groups, rows)
    for r in rows.itertuples():
        g = groups[r.symbol][5]
        bars = pd.DataFrame(
            {
                "ts": pd.to_datetime(g.window_start.to_numpy(), unit="ns", utc=True),
                "open": g.open,
                "high": g.high,
                "low": g.low,
                "close": g.close,
            }
        )
        for algo, slip in ss.PASS2_CFGS:
            ref = so.simulate_trade(
                bars, r.dir, r.or_high, r.or_low, r.atr_prev, slip, algo == "cons"
            )
            got = out[(out.symbol == r.symbol) & (out.algo == algo) & (out.slip == slip)]
            if ref is None:
                assert got.empty
            else:
                assert len(got) == 1
                assert np.isclose(got.entry.iloc[0], ref["entry"]) and np.isclose(
                    got.exit.iloc[0], ref["exit"]
                )
