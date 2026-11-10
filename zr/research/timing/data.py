# ruff: noqa: E501  （中文文档字符串与注释较长）
"""时间切分、留出期封存、数据加载、账户与成本（夜间择时研究 R-055 归档）。

留出期封存：loaders 默认截到验证期末，读取之后的日期必须显式 allow_holdout=True（只在预登记后的一次性
检验里使用）。费用一律经 zr.costs.fees.stock_order_fees（D-013 / D-035，不另写公式），按
(股数, 方向, 价格) 缓存。
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from zr.costs.fees import stock_order_fees
from zr.features.bars import ET, load_bars, load_extended_bars

DEV = (dt.date(2023, 10, 2), dt.date(2025, 12, 31))
VAL = (dt.date(2026, 1, 1), dt.date(2026, 5, 31))
HOLD_A = (dt.date(2026, 6, 1), dt.date(2026, 8, 31))
HOLD_B_FROM = dt.date(2026, 9, 1)
VAL_END = VAL[1]
TICK = 0.01
FX = 6.6725  # 人民币 / 美元，2026-10-23（夜间任务书给定）
ACCT = {"A": 10_000 / FX, "B": 100_000.0}  # A = 1 万人民币 ≈ 1,498.69 美元
HS = (1, 2, 3, 5, 10, 15, 30)
MS = (0.0010, 0.0015, 0.0020, 0.0030)
CAPS = (5, 15, 30)
GRID = (572, 945)  # 决策分钟标签 09:32–15:45（含），在该分钟 K 收盘时决策
REF = {"QQQ": "SPY", "SPY": "QQQ", "MU": "SMH"}


def _end(end: dt.date | None, allow_holdout: bool) -> dt.date:
    end = end or VAL_END
    if end > VAL_END and not allow_holdout:
        raise AssertionError(f"留出期封存：不得读取 {VAL_END} 之后的数据（end={end}）")
    return end


def rth(
    root: Path, und: str, end: dt.date | None = None, allow_holdout: bool = False
) -> pd.DataFrame:
    """RTH 1 分钟 K（索引 = UTC 开盘时刻），截到 end（默认验证期末）。"""
    end = _end(end, allow_holdout)
    b = load_bars(root, und, None, end)
    return b[np.asarray(b.index.tz_convert(ET).date) <= end]


def ext(
    root: Path, und: str, end: dt.date | None = None, allow_holdout: bool = False
) -> pd.DataFrame:
    """口径 C：含盘前盘后的长桥原始 1 分钟 K（clean=False），截到 end。"""
    end = _end(end, allow_holdout)
    b = load_extended_bars(root, und, None, end, clean=False)
    return b[np.asarray(b.index.tz_convert(ET).date) <= end]


def et_parts(idx: pd.DatetimeIndex):
    t = idx.tz_convert(ET)
    return np.asarray(t.date), np.asarray(t.hour * 60 + t.minute)


def split_of(d) -> np.ndarray:
    d = np.asarray(d)
    out = np.full(len(d), "", dtype=object)
    out[(d >= DEV[0]) & (d <= DEV[1])] = "dev"
    out[(d >= VAL[0]) & (d <= VAL[1])] = "val"
    out[(d >= HOLD_A[0]) & (d <= HOLD_A[1])] = "hold_a"
    out[d >= HOLD_B_FROM] = "hold_b"
    return out


@lru_cache(maxsize=1_000_000)
def _fee(shares: int, side: str, px: float) -> float:
    return float(stock_order_fees(shares, side, px)["total"]) if shares > 0 else 0.0


def fees(shares, px, side: str) -> np.ndarray:
    """D-013 正股单边费用（逐笔调用 stock_order_fees，按 (股数, 方向, 价格) 缓存；价格保留 6 位小数作缓存键）。"""
    sh = np.asarray(shares, float).astype(np.int64)
    p6 = np.round(np.asarray(px, float), 6)
    return np.array([_fee(int(a), side, float(b)) for a, b in zip(sh, p6, strict=True)], float)


def round_trip_fees(shares, entry, exit_, d) -> np.ndarray:
    """开仓 + 平仓两侧费用：多头 Buy 入 / Sell 出，空头 Sell 入 / Buy 出。"""
    d = np.asarray(d, float)
    f_long = fees(shares, entry, "Buy") + fees(shares, exit_, "Sell")
    f_short = fees(shares, entry, "Sell") + fees(shares, exit_, "Buy")
    return np.where(d > 0, f_long, f_short)


def net_pct(entry, exit_, d, acct: str) -> tuple[np.ndarray, np.ndarray]:
    """满仓整数股（固定初始权益）的扣费后每笔收益（%）与股数。股数 0 → NaN。"""
    entry, exit_, d = (np.asarray(a, float) for a in (entry, exit_, d))
    sh = np.floor(ACCT[acct] / entry + 1e-9)
    pnl = sh * d * (exit_ - entry) - round_trip_fees(sh, entry, exit_, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sh > 0, pnl / (sh * entry) * 100, np.nan), sh
