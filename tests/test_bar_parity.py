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
