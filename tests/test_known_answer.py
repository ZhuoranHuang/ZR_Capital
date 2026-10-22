"""已知答案审计的常规测试：夏令时、半日市、backtrader 成交时点（合成数据，不读真实数据）。"""

from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import pandas as pd

from zr.data.calendar import is_trading_day, rth_mask, session_close
from zr.features.bars import ET, resample_bars
from zr.research import e4_review as er

warnings.filterwarnings("ignore", category=SyntaxWarning)


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_dst_open_maps_to_0930_on_both_sides_of_switches():
    cases = {  # (UTC 时刻, 是否在 RTH 内)
        "2026-03-06 14:30": True,  # 冬令时 EST：14:30 UTC = 09:30 ET
        "2026-03-06 13:30": False,  # = 08:30 ET
        "2026-03-09 13:30": True,  # 3-08 切换后 EDT：13:30 UTC = 09:30 ET
        "2026-03-09 20:00": False,  # = 16:00 ET
        "2025-10-31 13:30": True,  # 11-02 切换前 EDT
        "2025-11-03 13:30": False,  # 切换后 EST：= 08:30 ET
        "2025-11-03 14:30": True,
    }
    idx = pd.DatetimeIndex([_utc(k) for k in cases])
    assert rth_mask(idx).tolist() == list(cases.values())
    assert [t.tz_convert(ET).strftime("%H:%M") for t in idx[[0, 2, 6]]] == ["09:30"] * 3


def test_resample_on_dst_switch_day_keeps_0930_label():
    idx = pd.date_range("2026-03-09 13:30", periods=10, freq="1min", tz="UTC")  # 09:30 EDT
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx)
    out = resample_bars(df, "2min")
    assert out.index[0].tz_convert(ET).strftime("%H:%M") == "09:30"


def test_half_day_calendar_and_rth_mask():
    d = dt.date(2025, 11, 28)
    assert session_close(d) == dt.time(13, 0)
    assert not is_trading_day(dt.date(2025, 11, 27))  # 感恩节
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp(f"2025-11-28 {t}", tz=ET).tz_convert("UTC")
            for t in ("12:59", "13:00", "15:00")
        ]
    )
    assert rth_mask(idx).tolist() == [True, False, False]


def test_e4_exits_on_last_bar_of_half_day():
    idx = []
    for d, n in (("2025-11-26", 78), ("2025-11-28", 42)):  # 11-28 只到 12:55（42 根 5 分钟）
        start = pd.Timestamp(f"{d} 09:30", tz=ET)
        idx += list(pd.date_range(start, periods=n, freq="5min"))
    idx = pd.DatetimeIndex(idx).tz_convert("UTC")
    a = {
        "day": np.asarray(idx.tz_convert(ET).date),
        "mod": np.asarray(idx.tz_convert(ET).hour * 60 + idx.tz_convert(ET).minute),
        "close": np.zeros(len(idx)),
    }
    last = len(idx) - 1
    assert er._eod(a, last, er.BASELINE)  # 半日市最后一根（12:55）触发清仓
    assert not er._eod(a, last - 1, er.BASELINE)


def test_backtrader_fill_timing_semantics():
    import backtrader as bt

    from zr.research import bt_independent as bi

    idx = pd.date_range("2026-01-05 14:30", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [100.5, 101.5, 102.2, 103.5, 104.5, 105.5],
            "low": [99.5, 100.5, 99.0, 102.5, 103.5, 104.5],
            "close": [100, 101, 101.5, 103, 104, 105],
            "volume": 1000,
        },
        index=idx,
    )
    fills = []

    class S(bt.Strategy):
        def next_open(self):
            if self.data.open[0] == 102:
                self.buy(size=100, exectype=bt.Order.Limit, price=self.data.open[0] + 0.01)
                self.sell(size=100, exectype=bt.Order.Stop, price=99.8)
            if self.data.open[0] == 103:
                self.buy(size=100, exectype=bt.Order.Limit, price=self.data.open[0] + 0.01)

        def next(self):
            if self.data.close[0] == 104:
                self.sell(size=100)

        def notify_order(self, o):
            if o.status == o.Completed:
                fills.append(
                    (
                        bt.num2date(o.executed.dt).strftime("%H:%M"),
                        o.getordername(),
                        round(o.executed.price, 2),
                    )
                )

    c = bi._cerebro()
    c.adddata(bt.feeds.PandasData(dataname=df))
    c.addstrategy(S)
    c.run()
    assert fills == [
        ("14:40", "Limit", 102.01),
        ("14:40", "Stop", 99.79),
        ("14:45", "Limit", 103.01),
        ("14:50", "Market", 103.99),
    ]


def test_v4_day_limits_follow_early_close():
    from zr.research import v4_decisive as v4d

    exe = v4d.load_v4()["execution"]
    assert v4d.day_limits(dt.date(2025, 11, 26), exe) == (dt.time(15, 40), dt.time(15, 50))
    assert v4d.day_limits(dt.date(2025, 11, 28), exe) == (dt.time(12, 40), dt.time(12, 50))
