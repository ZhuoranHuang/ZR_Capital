"""日历特征（METHOD_V2 规格 4），时间按 bar 收盘时刻 t+1 计算。

数据：config/calendar.yaml（只收录有来源的条目；CPI / 非农 / 2026 前 PCE / 财报日为 pending，
对应特征在这些事件上没有信息——见该文件 pending 段）。
- min_to_next_release：距下一个定时发布的分钟数（无则 NaN）
- min_since_last_release：距上一个定时发布的分钟数（无则 NaN）
- is_earnings_week：财报所在周（财报日 pending → 全部 NaN）
- is_expiry_day：当日是否为该标的的期权到期日（来自 ThetaData 已落盘的期权数据；无期权数据的日期 NaN）
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

ET = ZoneInfo("America/New_York")
CAL = Path(__file__).resolve().parents[2] / "config" / "calendar.yaml"
CALENDAR_COLUMNS = [
    "min_to_next_release",
    "min_since_last_release",
    "is_earnings_week",
    "is_expiry_day",
]


def release_times(path: Path = CAL) -> pd.DatetimeIndex:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    ts = [
        pd.Timestamp(
            dt.datetime.combine(pd.Timestamp(e["date"]).date(), dt.time.fromisoformat(e["time"])),
            tz=ET,
        )
        for e in cfg.get("events", [])
    ]
    return pd.DatetimeIndex(sorted(ts)).tz_convert("UTC")


def expiry_days(root: Path, und: str) -> dict[dt.date, bool]:
    """期权日 → 当日是否有到期合约（读 quotes.parquet 的 expiry 列）。"""
    base = root / "options_1m" / f"underlying={und}"
    out: dict[dt.date, bool] = {}
    if not base.is_dir():
        return out
    for p in base.glob("date=*"):
        f = p / "quotes.parquet"
        if not f.exists():
            continue
        day = dt.date.fromisoformat(p.name[5:])
        try:
            e = pd.read_parquet(f, columns=["expiry"])["expiry"]
        except Exception:  # noqa: BLE001 - 空日文件无列
            continue
        out[day] = bool((e == day).any())
    return out


def compute_calendar(
    index: pd.DatetimeIndex,
    root: Path | None = None,
    und: str | None = None,
    releases: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    rel = release_times() if releases is None else releases
    dec = pd.DatetimeIndex(index) + pd.Timedelta(minutes=1)  # bar 收盘时刻
    out = pd.DataFrame(index=index)
    # 两边统一到 ns：DatetimeIndex 的 asi8 单位随精度（s/ms/us/ns）变化
    r = pd.DatetimeIndex(rel).as_unit("ns").asi8
    d = pd.DatetimeIndex(dec).as_unit("ns").asi8
    j = np.searchsorted(r, d, side="left")  # 第一个 ≥ dec
    nxt = np.where(j < len(r), r[np.minimum(j, len(r) - 1)], np.nan)
    i = np.searchsorted(r, d, side="right") - 1  # 最后一个 ≤ dec
    prv = np.where(i >= 0, r[np.maximum(i, 0)], np.nan)
    out["min_to_next_release"] = (nxt - d) / 6e10
    out["min_since_last_release"] = (d - prv) / 6e10
    out["is_earnings_week"] = np.nan
    if root is not None and und is not None:
        ed = expiry_days(root, und)
        days = pd.Series(pd.DatetimeIndex(index).tz_convert(ET).date, index=index)
        out["is_expiry_day"] = days.map(lambda x: float(ed[x]) if x in ed else np.nan).to_numpy()
    else:
        out["is_expiry_day"] = np.nan
    return out
