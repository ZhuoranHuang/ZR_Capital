"""美股交易日历（全天休市 / 提前收盘），研究流水线统一按此过滤。

来源：ThetaData `/v3/calendar/year_holidays?year=YYYY`（本地终端，交易所日程），落到
`config/us_equity_calendar.json`（入库，离线可用；VPS 也用同一份）。
刷新：uv run python -m zr.data.calendar --refresh 2023 2024 2025 2026

背景（2026-10-22 实测）：长桥 1 分钟 K 在全天休市日也有 390 根 session=Intraday、volume=0、
OHLC 全等的记录（例 2023-11-23），提前收盘日 13:00 后也有记录；不过滤会污染特征与标签。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
CAL_PATH = Path(__file__).resolve().parents[2] / "config" / "us_equity_calendar.json"
OPEN = dt.time(9, 30)
CLOSE = dt.time(16, 0)


@lru_cache(maxsize=1)
def _load(path: str = str(CAL_PATH)) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"交易日历缺失：{p}（运行 python -m zr.data.calendar --refresh）")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {r["date"]: r for r in data["days"]}


def covered_years() -> set[int]:
    data = json.loads(CAL_PATH.read_text(encoding="utf-8"))
    return set(data["years"])


def is_trading_day(day: dt.date) -> bool:
    if day.weekday() >= 5:
        return False
    if day.year not in covered_years():
        raise ValueError(f"交易日历未覆盖 {day.year} 年（刷新 config/us_equity_calendar.json）")
    r = _load().get(day.isoformat())
    return r is None or r["type"] != "full_close"


def session_close(day: dt.date) -> dt.time:
    r = _load().get(day.isoformat())
    if r and r["type"] == "early_close" and r.get("close"):
        return dt.time.fromisoformat(r["close"])
    return CLOSE


def rth_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """bar 时间戳（开盘时刻）落在交易日 [09:30, 收盘) 内为 True。"""
    et = pd.DatetimeIndex(index).tz_convert(ET)
    days = et.date
    uniq = {d: (is_trading_day(d), session_close(d)) for d in set(days)}
    minute = et.hour * 60 + et.minute
    ok = np.zeros(len(et), dtype=bool)
    for d, (trading, close) in uniq.items():
        if not trading:
            continue
        m = days == d
        ok[m] = (minute[m] >= OPEN.hour * 60 + OPEN.minute) & (
            minute[m] < close.hour * 60 + close.minute
        )
    return ok


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and is_trading_day(d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def refresh(years: list[int], base_url: str = "http://127.0.0.1:25503/v3") -> Path:
    days = []
    for y in years:
        with urllib.request.urlopen(
            f"{base_url}/calendar/year_holidays?year={y}&format=json", timeout=30
        ) as r:  # noqa: S310
            body = json.loads(r.read().decode("utf-8"))
        rows = body.get("response", body) if isinstance(body, dict) else body
        days += [
            {"date": x["date"], "type": x["type"], "open": x.get("open"), "close": x.get("close")}
            for x in rows
        ]
    data = {
        "source": "ThetaData /v3/calendar/year_holidays（交易所日程，经本地 Theta Terminal）",
        "fetched_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "years": sorted(years),
        "days": sorted(days, key=lambda r: r["date"]),
    }
    CAL_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _load.cache_clear()
    return CAL_PATH


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", nargs="+", type=int, metavar="YEAR")
    a = p.parse_args(argv)
    if a.refresh:
        print(f"写入 {refresh(a.refresh)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
