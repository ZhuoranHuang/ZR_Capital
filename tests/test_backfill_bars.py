import datetime as dt
from types import SimpleNamespace

import pandas as pd

from zr.data.backfill_bars import backfill_symbol, bars_to_frame, existing_min_ts, write_merged


def _bar(ts: dt.datetime, px: float = 100.0):
    return SimpleNamespace(
        timestamp=ts,
        open=px,
        high=px,
        low=px,
        close=px,
        volume=1,
        turnover=px,
        trade_session="TradeSession.Intraday",
    )


class FakeQuoteCtx:
    """向过去翻页：返回 cursor 之前的 count 根，最早到 floor。"""

    def __init__(self, newest: dt.datetime, floor: dt.datetime):
        self.newest, self.floor = newest, floor
        self.calls: list[dt.datetime | None] = []

    def history_candlesticks_by_offset(
        self, symbol, period, adjust, forward, count, time, sessions
    ):
        self.calls.append(time)
        end = self.newest if time is None else time
        bars = []
        t = end
        while len(bars) < count and t >= self.floor:
            bars.append(_bar(t))
            t -= dt.timedelta(minutes=1)
        return list(reversed(bars))


def test_bars_to_frame_localizes_naive_utc():
    df = bars_to_frame("SPY.US", [_bar(dt.datetime(2026, 10, 21, 14, 0))])
    assert str(df["ts"].iloc[0]) == "2026-10-21 14:00:00+00:00"
    assert df["session"].iloc[0] == "Intraday"


def test_backfill_pages_until_empty_and_dedups(tmp_path):
    newest = dt.datetime(2026, 1, 1, 0, 59)
    floor = dt.datetime(2025, 12, 31, 22, 0)  # 跨年，共 180 根
    q = FakeQuoteCtx(newest, floor)
    import zr.data.backfill_bars as bb

    bb.PAGE_SIZE = 50
    try:
        res = backfill_symbol(q, tmp_path, "SPY.US", dt.date(2020, 1, 1), sleep=lambda s: None)
    finally:
        bb.PAGE_SIZE = 1000
    assert res["stop_reason"] == "empty_page"
    assert res["rows_fetched"] == 180
    y25 = pd.read_parquet(tmp_path / "bars_1m/underlying=SPY/year=2025/bars.parquet")
    y26 = pd.read_parquet(tmp_path / "bars_1m/underlying=SPY/year=2026/bars.parquet")
    assert len(y25) + len(y26) == 180
    assert y25["ts"].is_monotonic_increasing and y25["ts"].is_unique
    # 断点续传：再跑一次从最早 ts 之前继续，不重复写
    res2 = backfill_symbol(q, tmp_path, "SPY.US", dt.date(2020, 1, 1), sleep=lambda s: None)
    assert res2["rows_fetched"] == 0 and res2["stop_reason"] == "empty_page"
    assert len(pd.read_parquet(tmp_path / "bars_1m/underlying=SPY/year=2025/bars.parquet")) == len(
        y25
    )


def test_backfill_stops_at_until(tmp_path):
    q = FakeQuoteCtx(dt.datetime(2026, 1, 3), dt.datetime(2025, 1, 1))
    res = backfill_symbol(q, tmp_path, "QQQ.US", dt.date(2026, 1, 2), sleep=lambda s: None)
    assert res["stop_reason"] == "reached_until"
    assert res["pages"] == 2


def test_write_merged_dedup_and_min_ts(tmp_path):
    df = bars_to_frame("MU.US", [_bar(dt.datetime(2026, 3, 1, 10, i)) for i in range(3)])
    write_merged(tmp_path, "MU.US", df)
    write_merged(tmp_path, "MU.US", df)  # 重复写
    out = pd.read_parquet(tmp_path / "bars_1m/underlying=MU/year=2026/bars.parquet")
    assert len(out) == 3
    assert str(existing_min_ts(tmp_path, "MU.US")) == "2026-03-01 10:00:00+00:00"


def test_write_merged_new_pull_overwrites_revised_bar(tmp_path):
    """长桥会事后修订 bar（MU 2026-10-21 09:30 open）：再次拉取时以新值为准。"""
    t = dt.datetime(2026, 3, 1, 14, 30)
    old = bars_to_frame("MU.US", [_bar(t)])
    write_merged(tmp_path, "MU.US", old)
    new = old.assign(open=old["open"] - 1.0)
    write_merged(tmp_path, "MU.US", new)
    out = pd.read_parquet(tmp_path / "bars_1m/underlying=MU/year=2026/bars.parquet")
    assert len(out) == 1 and out["open"].iloc[0] == new["open"].iloc[0]
