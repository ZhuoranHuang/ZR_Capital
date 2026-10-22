"""METHOD_V2 规格 3/4 特征的单元测试：结构上下文、日历、跨资产、共振。每组都带无前视测试。"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from zr.features.bars import ET
from zr.features.calendar_features import compute_calendar
from zr.features.channel import channel_states
from zr.features.cross_asset import compute_cross_asset
from zr.features.features import compute_features
from zr.features.structure import STRUCTURE_COLUMNS, bars_since, compute_structure, tf5_regime
from zr.research import events as ev


def _bars(days: int = 6, n: int = 390, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    d0, k = dt.date(2026, 9, 14), 0
    while len(frames) < days:
        day = d0 + dt.timedelta(days=k)
        k += 1
        if day.weekday() >= 5:
            continue
        start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
        idx = pd.date_range(start, periods=n, freq="min", name="ts")
        close = px * np.cumprod(1 + drift + rng.normal(0, 0.0008, n))
        hi = close * (1 + rng.uniform(0, 0.0006, n))
        lo = close * (1 - rng.uniform(0, 0.0006, n))
        op = np.r_[px, close[:-1]]
        frames.append(
            pd.DataFrame(
                {
                    "open": op,
                    "high": np.maximum(hi, op),
                    "low": np.minimum(lo, op),
                    "close": close,
                    "volume": rng.integers(100, 1000, n),
                },
                index=idx,
            )
        )
        px = close[-1]
    return pd.concat(frames)


@pytest.fixture(scope="module")
def data():
    df = _bars()
    f = compute_features(df)
    return df, f, channel_states(df, f)


def _bump_tail(df: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    b = df.copy()
    for c in ("open", "high", "low", "close"):
        b.iloc[-k:, b.columns.get_loc(c)] *= 1.03
    return b


def test_bars_since():
    s = pd.Series([False, True, False, False, True, False])
    out = bars_since(s).tolist()
    assert np.isnan(out[0]) and out[1:] == [0, 1, 2, 0, 1]


def test_structure_columns_and_no_lookahead(data):
    df, f, ch = data
    s = compute_structure(df, f, ch)
    assert list(s.columns) == STRUCTURE_COLUMNS
    assert (s["pullback_idx"] >= 0).all()
    b = _bump_tail(df)
    fb = compute_features(b)
    s2 = compute_structure(b, fb, channel_states(b, fb))
    pd.testing.assert_frame_equal(s.iloc[:-30], s2.iloc[:-30])


def test_tf5_uses_only_completed_5min_bars(data):
    df, _, _ = data
    r = tf5_regime(df)
    # 改动最后一根 1 分钟 bar 只能影响它自己以后（它所在 5 分钟 bar 的完成时刻 ≥ 自身收盘）
    b = df.copy()
    b.iloc[-1, b.columns.get_loc("close")] *= 1.2
    r2 = tf5_regime(b)
    pd.testing.assert_series_equal(r.iloc[:-1], r2.iloc[:-1])
    # 09:30–09:33 收盘时没有任何完成的当日 5 分钟 bar，状态来自前一日或 NaN；09:34 收盘（=09:35）才看到 09:30 那根
    first_day = df.index.tz_convert(ET).date[0]
    m = df.index.tz_convert(ET).date == first_day
    assert r[m].iloc[:4].isna().all()


def test_calendar_minutes_to_and_since():
    idx = pd.date_range(
        pd.Timestamp("2026-09-16 13:50", tz=ET).tz_convert("UTC"), periods=20, freq="min"
    )
    rel = pd.DatetimeIndex([pd.Timestamp("2026-09-16 14:00", tz=ET)]).tz_convert("UTC")
    c = compute_calendar(idx, releases=rel)
    # bar 13:50 收盘 13:51 → 距 14:00 还有 9 分钟；bar 13:59 收盘 14:00 → 0；bar 14:05 收盘 14:06 → 已过 6
    assert c["min_to_next_release"].iloc[0] == 9
    assert c["min_to_next_release"].iloc[9] == 0
    assert c["min_since_last_release"].iloc[15] == 6
    assert np.isnan(c["min_since_last_release"].iloc[0])
    assert c["is_earnings_week"].isna().all()


def test_cross_asset_alignment_and_no_lookahead():
    a = _bars(days=2, seed=1)
    refs = {
        "SPY": _bars(days=2, seed=2),
        "QQQ": _bars(days=2, seed=3),
        "SMH": _bars(days=2, seed=4),
    }
    x = compute_cross_asset(a, "MU", refs=refs)
    assert {"xa_SPY_ret1", "xa_QQQ_rv5", "xa_SMH_ret1", "xa_n_same_sign"} <= set(x.columns)
    i = 100
    assert np.isclose(
        x["xa_SPY_ret1"].iloc[i],
        refs["SPY"]["close"].iloc[i] / refs["SPY"]["close"].iloc[i - 1] - 1,
    )
    refs2 = {k: _bump_tail(v, 10) for k, v in refs.items()}
    x2 = compute_cross_asset(a, "MU", refs=refs2)
    pd.testing.assert_frame_equal(x.iloc[:-10], x2.iloc[:-10])


def test_confluence_counts_same_direction_within_window():
    idx = pd.date_range(
        pd.Timestamp("2026-09-16 10:00", tz=ET).tz_convert("UTC"), periods=10, freq="min"
    )
    sig = pd.DataFrame(0, index=idx, columns=["E1", "E3", "E5", "E9"])
    sig.iloc[2, 0] = 1  # 趋势
    sig.iloc[4, 1] = 1  # 反转
    sig.iloc[5, 2] = 1  # 波动
    sig.iloc[5, 3] = -1  # 反向，不计入同向
    c = ev.confluence(sig, window=5)
    assert c["confl_n_up"].iloc[5] == 3 and c["confl_fam_up"].iloc[5] == 3
    assert c["confl_n_up"].iloc[7] == 2  # E1（第 2 根）已出窗口
    events = ev.events_from_matrix(sig, "MU.US", c)
    e5 = events[events["detector"] == "E5"].iloc[0]
    assert e5["confl_n_same_dir_5m"] == 2 and e5["confl_families_5m"] == 3
    # 无前视：改未来的信号不影响当前
    sig2 = sig.copy()
    sig2.iloc[8, 0] = 1
    pd.testing.assert_frame_equal(c.iloc[:8], ev.confluence(sig2, 5).iloc[:8])
