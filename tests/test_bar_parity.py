"""K 线一致性固定测试（audit/base-final 第 1 节，见 BASE_FINAL_AUDIT.md）。

冻结的 MU 2026-10-21：本地 1 分钟 K（tests/fixtures/mu_2026_10_21_min1_rth.csv）经 resample_bars
合成的 2 分钟 K，与长桥官方 2 分钟 K（mu_2026_10_21_official_2m_rth.csv，
scripts/diag_pull_candles.py 实拉）逐根比对，全部 195 根 OHLCV 完全相等。
（数据目录里 10-21 的 1 分钟 K 是当日拉取的，09:30 的 open 是长桥事后修订前的 925.42；
fixture 是次日拉取的修订值 923.735。）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zr.features.bars import ET, resample_bars
from zr.research.bar_parity import compare, synth

FIX = Path(__file__).parent / "fixtures"


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / name, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def test_synthesized_2min_matches_official_bar_by_bar():
    local = resample_bars(_read("mu_2026_10_21_min1_rth.csv"), "2min")
    off = _read("mu_2026_10_21_official_2m_rth.csv")
    assert local.index.equals(off.index) and len(off) == 195
    for c in ("open", "high", "low", "close", "volume"):
        assert np.allclose(local[c], off[c], atol=1e-9), c


def test_compare_classifies_causes():
    one = _read("mu_2026_10_21_min1_rth.csv")
    off = _read("mu_2026_10_21_official_2m_rth.csv")
    j = compare(synth(one, 2), off, 2)
    c = j["cause"].value_counts().to_dict()
    assert c == {"match": 194}  # 最后一个分箱不比（本地可能不完整）
    stale = one.copy()
    stale.iloc[0, stale.columns.get_loc("open")] = 925.42  # 修订前的 09:30 open
    js = compare(synth(stale, 2), off, 2)
    assert js["cause"].value_counts().to_dict() == {"match": 193, "price": 1}
    gap = synth(one.drop(one.index[5]), 2)  # 去掉 09:35 那分钟 → 09:34 分箱只有 1 根
    j2 = compare(gap, off, 2)
    hm = j2.index.tz_convert(ET).strftime("%H:%M")
    assert j2.loc[hm == "09:34", "cause"].item() == "missing_1m"


def test_extended_session_matches_official_except_partial_last_bin():
    """口径 C 的数据：含盘前盘后的 1 分钟 K（04:00 起）合成 2 分钟 K vs 官方 TradeSessions.All
    2 分钟 K。fixture 的 1 分钟 K 到 15:37 为止，最后一个分箱（15:36）不完整，不比；这份
    1 分钟 K 是长桥修订 09:30 open 之前拉的（925.42，修订后 923.735），只有这一个值不同。"""
    local = resample_bars(_read("mu_2026_10_21_min1_ext.csv"), "2min").iloc[:-1]
    off = _read("mu_2026_10_21_official_2m_ext.csv").iloc[:-1]
    assert local.index.equals(off.index) and len(off) == 348
    for c in ("high", "low", "close", "volume"):
        assert np.allclose(local[c], off[c], atol=1e-9), c
    diff = local.index[~np.isclose(local["open"], off["open"], atol=1e-9)]
    assert [t.tz_convert(ET).strftime("%H:%M") for t in diff] == ["09:30"]
    assert local["open"][diff].item() == 925.42 and off["open"][diff].item() == 923.735


def test_zero_volume_placeholder_minutes_do_not_set_open_high_low():
    """长桥在无成交的分钟插入 O=H=L=C=前收、量 0 的占位 K；官方 2 分钟 K 的开高低不含它们。"""
    idx = pd.date_range("2025-03-03 09:02", periods=4, freq="1min", tz="UTC")  # 04:02 ET 起
    one = pd.DataFrame(
        {
            "open": [95.805, 94.44, 94.44, 94.44],
            "high": [95.805, 94.44, 94.44, 94.44],
            "low": [95.805, 94.44, 94.44, 94.44],
            "close": [95.805, 94.44, 94.44, 94.44],
            "volume": [0, 318, 0, 0],
        },
        index=idx,
    )
    b = resample_bars(one, "2min")
    # 04:02 分箱：占位 + 成交 → 只用有成交那分钟（官方 94.44 / 94.44 / 94.44 / 94.44 / 318）
    assert b.iloc[0][["open", "high", "low", "close", "volume"]].tolist() == [
        94.44,
        94.44,
        94.44,
        94.44,
        318,
    ]
    # 04:04 分箱全是占位 → 仍输出（价格 = 前收，量 0）
    assert b.iloc[1]["open"] == 94.44 and b.iloc[1]["volume"] == 0
