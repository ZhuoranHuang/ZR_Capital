"""R-057 拉取脚本的纯逻辑：限频间隔、错误分类与退避、美东时间换算、日线日期映射。"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from zr.data import sip_pull as sp


class FakeQ:
    def __init__(self, fail):
        self.fail = list(fail)

    def history_candlesticks_by_offset(self, *a):
        if self.fail:
            raise RuntimeError(self.fail.pop(0))
        return ["ok"]


def _client(fail, now):
    t = {"v": 0.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        t["v"] += s

    cl = sp.Client(
        FakeQ(fail),
        clock=lambda: t["v"],
        sleep=sleep,
        now=lambda: now,
        is_trading_day=lambda d: d.weekday() < 5,
    )
    return cl, sleeps


def test_interval_by_session():
    in_rth = dt.datetime(2026, 11, 16, 15, 0, tzinfo=dt.UTC)  # 周一 10:00 ET
    off = dt.datetime(2026, 11, 14, 15, 0, tzinfo=dt.UTC)  # 周六
    assert _client([], in_rth)[0].interval() == 0.5
    assert _client([], off)[0].interval() == 0.2


def test_rate_limit_backoff_then_success_and_notfound():
    cl, sleeps = _client(
        ["OpenApiException: (code=301606) too many requests", "429 rate limit"],
        dt.datetime(2026, 11, 14, tzinfo=dt.UTC),
    )
    assert cl.call("history_candlesticks_by_offset") == ["ok"]
    assert [s for s in sleeps if s >= 2] == [2.0, 4.0] and cl.err["rate"] == 2
    cl, _ = _client(["symbol not found"], dt.datetime(2026, 11, 14, tzinfo=dt.UTC))
    with pytest.raises(sp.NotFound):
        cl.call("history_candlesticks_by_offset")
    cl, _ = _client(["boom", "boom", "boom"], dt.datetime(2026, 11, 14, tzinfo=dt.UTC))
    with pytest.raises(RuntimeError):
        cl.call("history_candlesticks_by_offset")


def test_et_naive_and_daily_date():
    assert sp.et_naive(pd.Timestamp("2024-03-04 14:30", tz="UTC")) == dt.datetime(2024, 3, 4, 9, 30)
    ts = pd.Series(pd.to_datetime(["2024-03-04 00:00", "2024-03-04 05:00"], utc=True))
    assert sp.daily_date(ts).tolist() == [dt.date(2024, 3, 4)] * 2
