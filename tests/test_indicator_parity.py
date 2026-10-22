"""D-027 指标标准口径的固定回归测试（diag/indicator-parity 第 4 节「指标一致性作为固定测试」）。

用冻结的 MU 2026-10-21 RTH 1 分钟 K（tests/fixtures/mu_2026_10_21_min1_rth.csv，取自
scripts/diag_pull_indicator_parity.py 实拉的长桥数据，不连网）核对：RTH-only、按日重置、
ddof=0、MACD 标准 EMA 首值做种、时间戳偏移 +2 分钟，在负责人截图的 3 个独立时刻上复现
docs/research/INDICATOR_PARITY.md 的最佳口径结果。数值来源与误差计算见该文档与
zr/research/indicator_parity.py；此处只锁定「不会被后续改动悄悄破坏」。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zr.features.bars import resample_bars
from zr.research.indicator_parity import APP, boll, ema_first_seed, macd, nearest_close, pct_err

FIXTURE = Path(__file__).parent / "fixtures" / "mu_2026_10_21_min1_rth.csv"
BOLL_TOL_PCT = 0.01  # 实测 ≤5e-5%，放宽到 0.01% 留余量
MACD_TOL_PCT = 0.2  # 实测 ≤0.155%，全部可用 App 只显示 3 位小数的量化误差解释


def _load_2min_close() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return resample_bars(df, "2min")


def test_boll_matches_app_screenshots_within_tolerance():
    close = _load_2min_close()["close"]
    b = boll(close, 20, 2.0, ddof=0)
    for hm, want in APP.items():
        ts = nearest_close(close.index, hm, label_offset_min=2)
        assert ts is not None, f"{hm} 找不到对应 bar"
        for k in ("mid", "upper", "lower"):
            err = pct_err(float(b.loc[ts, k]), want[k])
            assert err < BOLL_TOL_PCT, f"{hm} {k} 误差 {err}% 超出 {BOLL_TOL_PCT}%"


def test_macd_matches_app_screenshots_within_display_rounding():
    close = _load_2min_close()["close"]
    d = macd(close, 6, 13, 6, ema_first_seed)
    for hm, want in APP.items():
        ts = nearest_close(close.index, hm, label_offset_min=2)
        assert ts is not None, f"{hm} 找不到对应 bar"
        for k in ("dif", "dea"):
            err = pct_err(float(d.loc[ts, k]), want[k])
            assert err < MACD_TOL_PCT, f"{hm} {k} 误差 {err}% 超出 {MACD_TOL_PCT}%"


EXT_FIXTURE = Path(__file__).parent / "fixtures" / "mu_2026_10_21_min1_ext.csv"


def test_convention_c_continuous_extended_matches_app_exactly_at_display_precision():
    """D-028 口径 C：含盘前盘后的长桥原始 1 分钟 K（04:00 起，不清洗）聚合 2 分钟、不重置，
    3 个截图时刻 15 个值四舍五入到 3 位小数后与 App 完全相等。"""
    df = pd.read_csv(EXT_FIXTURE, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    close = resample_bars(df.set_index("ts").sort_index(), "2min")["close"]
    b, d = boll(close, 20, 2.0, ddof=0), macd(close, 6, 13, 6, ema_first_seed)
    for hm, want in APP.items():
        ts = nearest_close(close.index, hm, label_offset_min=2)
        for k in ("mid", "upper", "lower"):
            assert round(float(b.loc[ts, k]), 3) == want[k], (hm, k)
        for k in ("dif", "dea"):
            assert round(float(d.loc[ts, k]), 3) == want[k], (hm, k)
