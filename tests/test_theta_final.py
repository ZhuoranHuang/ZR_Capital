# ruff: noqa: E501
"""zr.data.theta_final / zr.data.inventory / zr.data.massive_seconds：假 HTTP，不连终端、不联网。"""

import datetime as dt
import json

import pandas as pd
import pytest

from zr.data import inventory
from zr.data import massive_seconds as ms
from zr.data import theta_final as tf
from zr.data import thetadata as td

DAY = dt.date(2026, 11, 16)


def _client(routes: dict):
    def getter(url: str):
        for k, v in routes.items():
            if k in url:
                return v if isinstance(v, tuple) else (200, json.dumps({"response": v}))
        return 472, "No data"

    return td.ThetaClient(getter=getter, sleep=lambda s: None)


def _contract(strike, right, data):
    return {
        "contract": {"symbol": "SPY", "strike": strike, "right": right, "expiration": "2026-11-20"},
        "data": [data],
    }


def test_fetch_eod_prefers_greeks_and_falls_back():
    g = _contract(
        725.0,
        "CALL",
        {
            "close": 18.18,
            "volume": 354,
            "count": 37,
            "bid": 18.07,
            "ask": 18.35,
            "implied_vol": 0.27,
            "delta": 0.76,
            "gamma": 0.01,
            "theta": -0.8,
            "vega": 23.6,
            "rho": 6.0,
            "underlying_price": 739.95,
            "timestamp": "2026-11-16T16:11:50.005",
            "zomma": 1,
        },
    )
    df = tf.fetch_eod(_client({"/option/history/greeks/eod": [g]}), "SPY", DAY)
    assert df["source"].iloc[0] == "greeks_eod" and df["symbol"].iloc[0] == "SPY261120C725000.US"
    assert {
        "implied_vol",
        "delta",
        "volume",
        "bid",
        "ask",
        "underlying_price",
        "last_trade",
    } <= set(df.columns) and "zomma" not in df
    e = _contract(
        725.0,
        "PUT",
        {
            "close": 3.0,
            "volume": 10,
            "count": 2,
            "bid": 2.9,
            "ask": 3.1,
            "last_trade": "2026-11-16T16:14:11.674",
        },
    )
    df2 = tf.fetch_eod(_client({"/option/history/eod": [e]}), "SPY", DAY)
    assert (
        df2["source"].iloc[0] == "eod"
        and "implied_vol" not in df2
        and df2["right"].iloc[0] == "put"
    )
    assert tf.fetch_eod(_client({}), "SPY", DAY).empty


def test_fetch_oi_full_chain():
    r = _contract(825.0, "CALL", {"open_interest": 680, "timestamp": "2026-11-16T06:30:00.000"})
    df = tf.fetch_oi(_client({"/option/history/open_interest": [r]}), "SPY", DAY)
    assert list(df["open_interest"]) == [680] and df["expiry"].iloc[0] == dt.date(2026, 11, 20)


def test_run_daily_writes_resumes_and_stops_on_403(tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "trading_days", lambda a, b: [dt.date(2026, 11, 13), DAY])
    r = _contract(825.0, "CALL", {"open_interest": 680, "timestamp": "2026-11-16T06:30:00.000"})
    c = _client({"/option/history/open_interest": [r]})
    jobs = tf.daily_jobs(tmp_path, ("SPY",), dt.date(2026, 11, 13), DAY, ("oi",))
    assert len(jobs) == 2 and tf.run_daily(c, tmp_path, jobs, "t")["done"] == 2
    assert tf.daily_jobs(tmp_path, ("SPY",), dt.date(2026, 11, 13), DAY, ("oi",)) == []  # 断点续传
    c403 = _client({"/option/history/open_interest": (403, "requires a STANDARD subscription")})
    with pytest.raises(tf.Stop):
        tf.run_daily(c403, tmp_path, [("QQQ", DAY, "oi")], "t")
    assert (tmp_path / "theta_final" / "STOPPED.md").exists()


def test_p3_day_filters_dte_and_band(tmp_path):
    rows = [
        {
            "contract": {
                "symbol": "NVDA",
                "strike": s,
                "right": "CALL",
                "expiration": "2026-11-20",
            },
            "data": [{"bid": 1.0, "ask": 1.1, "timestamp": "2026-11-16T09:31:00.000"}],
        }
        for s in (100.0, 150.0, 200.0)
    ]
    c = _client({"/option/history/quote": rows})
    exps = [dt.date(2026, 11, 20), dt.date(2026, 12, 18)]
    rec = tf.p3_day(c, tmp_path, "NVDA", DAY, exps, (150.0, 148.0, 152.0))
    assert rec["expirations"] == ["2026-11-20"]  # 12-18 超过 7 天
    q = pd.read_parquet(
        tmp_path / "options_1m" / "underlying=NVDA" / f"date={DAY}" / "quotes.parquet"
    )
    assert list(q["strike"]) == [150.0]  # 只留 [148×0.97, 152×1.03]


def test_prices_from_stock_eod():
    rows = [
        {"open": 749.64, "high": 749.82, "low": 738.07, "last_trade": "2026-11-16T17:14:54.083"}
    ]
    px = tf.Prices(_client({"/stock/history/eod": rows}))
    fn = px.fn("SPY", DAY, DAY)
    assert fn(DAY) == (749.64, 738.07, 749.82) and fn(dt.date(2026, 11, 13)) is None


def test_inventory_groups_by_underlying_not_by_date(tmp_path):
    for p in (
        "options_1m/underlying=SPY/date=2026-11-16/quotes.parquet",
        "recorder/date=2026-11-16/underlying=MU/q_1.parquet",
        "bars_1m/underlying=MU/year=2026/bars.parquet",
    ):
        f = tmp_path / p
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x" * 10)
    rows = {r["dataset"]: r for r in inventory.scan(tmp_path)}
    assert set(rows) == {"options_1m/underlying=SPY", "recorder", "bars_1m/underlying=MU"}
    assert (
        rows["recorder"]["last"] == "2026-11-16"
        and rows["bars_1m/underlying=MU"]["first"] == "2026"
    )
    assert "| `recorder` |" in inventory.to_md(list(rows.values()))


def test_massive_seconds_frame_and_bisect():
    class R:
        calls = 0

        def paged(self, path, params):
            yield from [
                {
                    "t": 1763296200000,
                    "o": 1.0,
                    "h": 2.0,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 10,
                    "vw": 1.2,
                    "n": 3,
                },
                {
                    "t": 1763296201000,
                    "o": 1.5,
                    "h": 1.6,
                    "l": 1.4,
                    "c": 1.5,
                    "v": 5,
                    "vw": 1.5,
                    "n": 1,
                },
            ]

        def get(self, path, params=None):
            from zr.data import massive as mv

            day = path.split("/")[-1]
            if day < "2021-11-18":
                raise mv.MassiveError(403, "NOT_AUTHORIZED", path)
            return {}

    d = ms.fetch_day(R(), "SPY", dt.date(2025, 11, 16))
    assert len(d) == 2 and str(d["ts"].dt.tz) == "UTC" and d["volume"].sum() == 15
    days = [dt.date(2021, 11, 15) + dt.timedelta(days=i) for i in range(10)]
    assert ms.first_accessible(R(), days) == dt.date(2021, 11, 18)
