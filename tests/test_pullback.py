"""D-022 回踩信号与深度特征：形态、深度阈值、同日约束、无前视。"""

import datetime as dt

import numpy as np
import pandas as pd

from zr.features import pullback as pb


def _df(rows, day=dt.date(2024, 3, 5), start="09:30"):
    idx = pd.date_range(f"{day} {start}", periods=len(rows), freq="min", tz="America/New_York")
    o, h, lo, c = map(np.asarray, zip(*rows, strict=True))
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx.tz_convert("UTC")
    )


def _flat(n, px=100.0):
    return [(px, px + 0.1, px - 0.1, px)] * n


# 平盘 20 根后：阴、阴、阳（低点 99.6），再阴、阳（只有 1 根回调）
ROWS = _flat(20) + [
    (100.3, 100.35, 100.0, 100.1),
    (100.1, 100.15, 99.6, 99.8),
    (99.8, 100.4, 99.75, 100.3),  # 22：收阳，前 2 根阴
    (100.3, 100.35, 100.0, 100.1),
    (100.1, 100.5, 100.05, 100.4),  # 24：收阳，前 1 根阴
]


def test_pattern_requires_n_pullback_bars():
    df = _df(ROWS)
    base = pb.PullbackBase(df, [20], 2.0, 3)
    chan = pd.Series(1, index=df.index)
    s2 = base.signal(chan, 20, "eps", 0.01, 2)
    assert s2[22] == 1 and s2[24] == 0
    assert base.signal(chan, 20, "eps", 0.01, 3)[22] == 0  # 只有 2 根阴线
    # 不在通道（chan=0）不触发
    assert (base.signal(pd.Series(0, index=df.index), 20, "eps", 0.01, 2) == 0).all()


def test_depth_thresholds_pctb_and_eps():
    df = _df(ROWS)
    base = pb.PullbackBase(df, [20], 2.0, 3)
    chan = pd.Series(1, index=df.index)
    b = base.bb[20].iloc[22]
    low = 99.6
    pctb = (low - b["lower"]) / (b["upper"] - b["lower"])
    eps = abs(low - b["mid"]) / b["mid"]
    assert base.signal(chan, 20, "pctb", pctb + 1e-6, 2)[22] == 1
    assert base.signal(chan, 20, "pctb", pctb - 1e-6, 2)[22] == 0
    assert base.signal(chan, 20, "eps", eps + 1e-9, 2)[22] == 1
    assert base.signal(chan, 20, "eps", eps - 1e-9, 2)[22] == 0


def test_short_is_mirror():
    mirror = [(200 - o, 200 - lo, 200 - h, 200 - c) for o, h, lo, c in ROWS]
    df = _df(mirror)
    base = pb.PullbackBase(df, [20], 2.0, 3)
    s = base.signal(pd.Series(-1, index=df.index), 20, "eps", 0.01, 2)
    assert s[22] == -1 and s[24] == 0


def test_pullback_bars_must_be_same_day():
    d1 = _df(_flat(18) + [ROWS[20], ROWS[21]], start="15:40")  # 前一日最后 2 根阴线
    d2 = _df([ROWS[22]] + _flat(5), day=dt.date(2024, 3, 6))
    df = pd.concat([d1, d2])
    base = pb.PullbackBase(df, [20], 2.0, 3)
    assert base.signal(pd.Series(1, index=df.index), 20, "eps", 0.05, 2)[20] == 0


def test_no_lookahead_signal_and_features():
    df = _df(ROWS + _flat(10))
    chan = pd.Series(1, index=df.index)
    cfg = {"bb_period": [10, 20], "bb_k": 2.0, "feature_window": 3}
    a = pb.PullbackBase(df, [20], 2.0, 3).signal(chan, 20, "eps", 0.01, 2)
    fa = pb.depth_features(df, cfg)
    df2 = df.copy()
    df2.iloc[23:, :4] = df2.iloc[23:, :4] * 1.05
    b = pb.PullbackBase(df2, [20], 2.0, 3).signal(chan, 20, "eps", 0.01, 2)
    fb = pb.depth_features(df2, cfg)
    assert (a[:23] == b[:23]).all()
    pd.testing.assert_frame_equal(fa.iloc[:23], fb.iloc[:23])


def test_directional_features_pick_side():
    df = _df(ROWS)
    cfg = {"bb_period": [20], "bb_k": 2.0, "feature_window": 3}
    f = pb.depth_features(df, cfg)
    ts = pd.Series([df.index[22], df.index[22]])
    d = pb.directional(f, ts, np.array([1, -1]), cfg)
    assert list(d.columns) == ["pb_pctb_20", "pb_eps_20"]
    assert d["pb_eps_20"].iloc[0] == f["pb_eps_20_long"].iloc[22]
    assert d["pb_eps_20"].iloc[1] == f["pb_eps_20_short"].iloc[22]
    # 多头版：低点 99.6 在中轨下方 → 带符号距离为负
    assert d["pb_eps_20"].iloc[0] < 0
