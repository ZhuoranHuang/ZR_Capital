"""增量计算路径（zr/strategy/incremental.py）的固定测试（audit/base-final 第 5 节）。

1. 同一组 K 线：增量引擎逐根计算的状态 / 信号 / 离场 / 护栏与批量向量化路径逐根相同
   （冻结的 MU 2026-10-21 含盘前盘后 1 分钟 K → 2 分钟 K，口径 C）。
2. BarBuilder：只有常规成交计价、全部成交计量；空箱补占位 K；迟到成交丢弃；时钟收线。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zr.features.bars import resample_bars
from zr.features.regime import bars_per_day, indicators, regimes
from zr.research import v4_decisive as v4d
from zr.strategy.incremental import NS_MIN, Bar, BarBuilder, IncrementalEngine

FIX = Path(__file__).parent / "fixtures"


def test_incremental_matches_batch_bar_by_bar():
    df = pd.read_csv(FIX / "mu_2026_10_21_min1_ext.csv", parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    b = resample_bars(df.set_index("ts").sort_index(), "2min")
    v4 = v4d.load_v4()
    cfg = v4d.state_cfg(v4, "C")
    day = b.index[0].tz_convert("America/New_York").date()
    x = indicators(b, cfg, daily_reset=False, warmup="nan")
    reg = regimes(b, x, cfg, 2)
    sig = v4d.hypothesis_signals(b, x, reg, cfg, v4)
    ex = v4d.exit_arrays(b, x, cfg)
    tok, okl, oks = v4d.guard_mask(b, reg, v4, {day})
    eng = IncrementalEngine(cfg, v4, max(1, round(bars_per_day(b))), {day})
    recs = [
        eng.on_bar(Bar(int(t), o, h, lo, c, v))
        for t, o, h, lo, c, v in zip(
            v4d.ns(b.index), b["open"], b["high"], b["low"], b["close"], b["volume"], strict=True
        )
    ]
    assert [r["state"] for r in recs] == reg["state"].tolist()
    assert [r["dir"] for r in recs] == reg["dir"].tolist()
    for h in ("H1", "H2", "H3"):
        assert [r[h] for r in recs] == sig[h].tolist(), h
    for rule in ("X3_mid2", "X2_shrink2", "X4_inside_0.1_0.9", "X5_higher_low"):
        for side in (0, 1):
            assert np.array_equal([bool(r["ex"][rule][side]) for r in recs], ex[rule][side]), rule
    assert [r["ok_long"] for r in recs] == (tok & okl).tolist()
    assert [r["ok_short"] for r in recs] == (tok & oks).tolist()
    assert np.allclose([r["mid"] for r in recs], x["mid"], equal_nan=True, atol=1e-9)
    assert np.allclose([r["dea"] for r in recs], x["dea"], atol=1e-9)
    assert sum(r["H1"] != 0 for r in recs) > 0  # 这一天确有 H1 信号（09:30 后急跌）


T0 = int(pd.Timestamp("2026-10-22 14:00", tz="UTC").value)


def test_bar_builder_price_filter_placeholder_late_and_clock():
    bb = BarBuilder(2, lambda tt: tt == "")
    s = 1_000_000_000
    out = []
    out += bb.on_trade(T0 + 1 * s, 100.0, 100, "")
    out += bb.on_trade(T0 + 2 * s, 105.0, 1, "I")  # 零股：只计量，不计价
    out += bb.on_trade(T0 + 3 * s, 99.0, 100, "")
    out += bb.on_trade(T0 + 6 * NS_MIN + 1 * s, 101.0, 50, "")  # 跳过 14:02、14:04 两箱
    assert len(out) == 3
    b0, b1, b2 = out
    assert (b0.open, b0.high, b0.low, b0.close, b0.volume) == (100.0, 100.0, 99.0, 99.0, 201)
    assert (b1.open, b1.close, b1.volume) == (99.0, 99.0, 0) and b2.ts == T0 + 4 * NS_MIN
    assert bb.on_trade(T0 + 3 * NS_MIN, 98.0, 10, "") == [] and bb.late == 1  # 已收线分箱
    assert bb.on_clock(T0 + 8 * NS_MIN + 500_000_000) == []  # 终点 + grace 之前不收线
    last = bb.on_clock(T0 + 8 * NS_MIN + 2 * s)
    assert len(last) == 1 and last[0].ts == T0 + 6 * NS_MIN and last[0].close == 101.0
