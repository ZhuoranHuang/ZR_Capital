import datetime as dt
import json
import urllib.parse

import pandas as pd
import pytest

from zr.data import thetadata as td


def _fake_getter(calls: list[str]):
    def getter(url: str) -> tuple[int, str]:
        calls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        path = urllib.parse.urlparse(url).path
        if path.endswith("/option/list/expirations"):
            # 终端真实形状：{"response": [...]}（2026-10-22 实测）
            dates = ("2026-09-01", "2026-09-02", "2026-09-03", "2026-09-30")
            return 200, json.dumps(
                {"response": [{"symbol": "SPY", "expiration": d} for d in dates]}
            )
        if path.endswith("/option/history/quote"):
            exp = q["expiration"][0]
            exp_iso = f"{exp[:4]}-{exp[4:6]}-{exp[6:]}"
            if q.get("interval", [""])[0] == "tick":
                return 200, json.dumps(
                    [
                        {
                            "symbol": "SPY",
                            "expiration": exp_iso,
                            "strike": 640.0,
                            "right": "call",
                            "timestamp": "2026-09-01T10:00:00.123",
                            "bid": 1.0,
                            "ask": 1.02,
                            "bid_size": 5,
                            "ask_size": 7,
                        }
                    ]
                )
            rows = []
            for strike in (600.0, 630.0, 640.0, 650.0, 700.0):
                for right in ("call", "put"):
                    rows.append(
                        {
                            "symbol": "SPY",
                            "expiration": exp_iso,
                            "strike": strike,
                            "right": right,
                            "timestamp": "2026-09-01T09:30:00.000",
                            "bid": 1.0,
                            "ask": 1.05,
                            "bid_size": 10,
                            "ask_size": 12,
                            "bid_exchange": 1,
                            "ask_exchange": 1,
                            "bid_condition": 0,
                            "ask_condition": 0,
                        }
                    )
            return 200, json.dumps(rows)
        if path.endswith("/option/history/ohlc"):
            exp = q["expiration"][0]
            return 200, json.dumps(
                [
                    {
                        "symbol": "SPY",
                        "expiration": f"{exp[:4]}-{exp[4:6]}-{exp[6:]}",
                        "strike": 640.0,
                        "right": "call",
                        "timestamp": "2026-09-01T09:30:00.000",
                        "open": 1,
                        "high": 1.1,
                        "low": 0.9,
                        "close": 1.05,
                        "volume": 100,
                        "count": 3,
                        "vwap": 1.02,
                    }
                ]
            )
        if path.endswith("/option/history/open_interest"):
            assert q["expiration"][0] != "*", "OI 必须按到期日请求"
            if q["expiration"][0] != "20260902":
                return 472, "NO_DATA"
            return 200, json.dumps(
                [
                    {
                        "symbol": "SPY",
                        "expiration": "2026-09-02",
                        "strike": 640.0,
                        "right": "put",
                        "timestamp": "2026-09-01T06:30:00.000",
                        "open_interest": 1234,
                    },
                    {
                        "symbol": "SPY",
                        "expiration": "2026-09-02",
                        "strike": 900.0,
                        "right": "put",
                        "timestamp": "2026-09-01T06:30:00.000",
                        "open_interest": 1,
                    },
                ]
            )
        return 472, "NO_DATA"

    return getter


def test_lb_symbol_roundtrip():
    s = td.lb_symbol("SPY", dt.date(2026, 10, 21), "call", 774.0)
    assert s == "SPY261021C774000.US"
    c = td.parse_contract(s)
    assert c == {"und": "SPY", "expiry": dt.date(2026, 10, 21), "right": "call", "strike": 774.0}
    assert td.lb_symbol("MU", dt.date(2026, 10, 23), "P", 912.5) == "MU261023P912500.US"


def test_select_expirations_dte_window():
    exps = [
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
        dt.date(2026, 9, 6),
        dt.date(2026, 9, 7),
        dt.date(2026, 8, 31),
    ]
    assert td.select_expirations(exps, dt.date(2026, 9, 1), max_dte=5) == [
        dt.date(2026, 9, 1),
        dt.date(2026, 9, 2),
        dt.date(2026, 9, 6),
    ]


def test_frame_timestamps_are_et_converted_to_utc():
    rows = [
        {
            "symbol": "SPY",
            "expiration": "2026-09-01",
            "strike": 640,
            "right": "call",
            "timestamp": "2026-09-01T09:30:00.000",
            "bid": 1.0,
            "ask": 1.1,
        }
    ]
    df = td._frame(rows, "SPY", dt.date(2026, 9, 1))
    assert str(df["ts"].iloc[0]) == "2026-09-01 13:30:00+00:00"  # EDT = UTC-4
    assert df["symbol"].iloc[0] == "SPY260901C640000.US"
    assert list(df.columns[:7]) == [
        "ts",
        "trade_date",
        "symbol",
        "underlying",
        "expiry",
        "strike",
        "right",
    ]


def test_bulk_day_filters_strikes_writes_parquet_and_resumes(tmp_path):
    calls: list[str] = []
    client = td.ThetaClient(getter=_fake_getter(calls), concurrency=2)
    day = dt.date(2026, 9, 1)
    recs = td.bulk_quotes_1m(client, tmp_path, "SPY", [day], px_fn=lambda d: (640.0, 638.0, 642.0))
    assert len(recs) == 1 and "error" not in recs[0]
    assert recs[0]["expirations"] == ["2026-09-01", "2026-09-02", "2026-09-03"]  # 09-30 超过 DTE 5
    d = tmp_path / "options_1m/underlying=SPY/date=2026-09-01"
    q = pd.read_parquet(d / "quotes.parquet")
    # 带宽按当日高低点：[638×0.97, 642×1.03] = [618.9, 661.3] → 只剩 630/640/650，
    # 三个到期日 × 3 档 × call+put
    assert sorted(q["strike"].unique()) == [630.0, 640.0, 650.0]
    assert len(q) == 3 * 3 * 2
    oi = pd.read_parquet(d / "oi.parquet")
    assert len(oi) == 1 and oi["open_interest"].iloc[0] == 1234
    assert (tmp_path / "options_1m/manifest.jsonl").exists()
    n_calls = len(calls)
    # 断点续传：已完成的日期不再请求
    assert (
        td.bulk_quotes_1m(client, tmp_path, "SPY", [day], px_fn=lambda d: (640.0, 638.0, 642.0))
        == []
    )
    assert len(calls) == n_calls


def test_http_error_raises_and_no_data_is_empty():
    def getter(url):
        return (473, "INVALID_PARAMS") if "expiration=20260903" in url else (472, "NO_DATA")

    client = td.ThetaClient(getter=getter)
    assert client.history_quote("SPY", dt.date(2026, 9, 2), dt.date(2026, 9, 1)).empty
    with pytest.raises(td.ThetaError) as e:
        client.history_quote("SPY", dt.date(2026, 9, 3), dt.date(2026, 9, 1))
    assert e.value.status == 473


def test_ticks_window_uses_tick_interval_and_contract_fields(tmp_path):
    calls: list[str] = []
    client = td.ThetaClient(getter=_fake_getter(calls))
    s = dt.datetime(2026, 9, 1, 9, 45, tzinfo=td.ET)
    e = dt.datetime(2026, 9, 1, 10, 15, tzinfo=td.ET)
    p = td.ticks_window(client, tmp_path, "SPY260901C640000.US", s, e)
    assert p.name == "SPY260901C640000.US_0945-1015.parquet"
    q = urllib.parse.parse_qs(urllib.parse.urlparse(calls[-1]).query)
    assert q["interval"] == ["tick"] and q["strike"] == ["640.000"] and q["right"] == ["call"]
    assert q["start_time"] == ["09:45:00.000"] and q["end_time"] == ["10:15:00.000"]
    assert len(pd.read_parquet(p)) == 1


def test_client_waits_through_terminal_restart_then_succeeds():
    calls = []
    seq = [
        (599, "refused"),
        (503, "io exception"),
        (571, "SERVER_STARTING"),
        (474, "DISCONNECTED"),
        (200, json.dumps({"response": [{"symbol": "SPY", "expiration": "2026-09-02"}]})),
    ]

    def getter(url):
        calls.append(url)
        return seq[len(calls) - 1]

    slept = []
    c = td.ThetaClient(getter=getter, outage_wait_s=60, outage_poll_s=10, sleep=slept.append)
    assert c.list_expirations("SPY") == [dt.date(2026, 9, 2)]
    assert len(calls) == 5 and slept == [10, 10, 10, 10]


def test_client_gives_up_after_outage_budget():
    c = td.ThetaClient(
        getter=lambda url: (599, "refused"),
        outage_wait_s=30,
        outage_poll_s=10,
        sleep=lambda s: None,
    )
    with pytest.raises(td.ThetaError) as e:
        c.list_expirations("SPY")
    assert e.value.status == 599


def test_coverage_report_flags_days_filtered_too_narrow(tmp_path, monkeypatch):
    day = dt.date(2026, 9, 1)
    d = tmp_path / "options_1m/underlying=SPY/date=2026-09-01"
    d.mkdir(parents=True)
    pd.DataFrame({"strike": [630.0, 640.0, 650.0]}).to_parquet(d / "quotes.parquet", index=False)
    # 当日最低 600 → 需要 582 起；最高 642 → 需要到 661.3；存的只有 630–650 → 不覆盖
    monkeypatch.setattr(td, "day_price_range", lambda root, und, day: (640.0, 600.0, 642.0))
    cov = td.coverage_report(tmp_path, "SPY", [day])
    assert cov.loc[0, "covered"] is False or cov.loc[0, "covered"] == False  # noqa: E712
    # 需要 [620.3, 659.7]，存的 630–650，一档容差 10 → 覆盖
    monkeypatch.setattr(td, "day_price_range", lambda root, und, day: (640.0, 639.5, 640.5))
    cov = td.coverage_report(tmp_path, "SPY", [day])
    assert bool(cov.loc[0, "covered"]) is True
    assert td.drop_days(tmp_path, "SPY", [day]) == 1 and not d.exists()


def test_coverage_uses_edge_spacing_and_chain_extent(tmp_path, monkeypatch):
    day = dt.date(2026, 9, 1)
    d = tmp_path / "options_1m/underlying=MU/date=2026-09-01"
    d.mkdir(parents=True)
    # 近 ATM 0.5 档，远端 1 档：存 62–74
    pd.DataFrame({"strike": [62.0, 63.0, 67.5, 68.0, 68.5, 73.0, 74.0]}).to_parquet(
        d / "quotes.parquet", index=False
    )
    # 需要 [61.7, 74.81]：低侧 62 ≤ 61.7+1 ✓，高侧 74 ≥ 74.81−1 ✓ → 覆盖（全局最小档距 0.5 会误判）
    monkeypatch.setattr(td, "day_price_range", lambda r, u, dd: (68.0, 67.06, 69.27))
    assert bool(td.coverage_report(tmp_path, "MU", [day]).loc[0, "covered"]) is True
    # 需要高到 80：存的只到 74 → 不覆盖；但若 manifest 说链本来就只到 74 → 覆盖
    monkeypatch.setattr(td, "day_price_range", lambda r, u, dd: (68.0, 67.06, 74.0))
    assert bool(td.coverage_report(tmp_path, "MU", [day]).loc[0, "covered"]) is False
    (tmp_path / "options_1m/manifest.jsonl").write_text(
        json.dumps({"underlying": "MU", "date": "2026-09-01", "chain_min": 62.0, "chain_max": 74.0})
        + "\n",
        encoding="utf-8",
    )
    assert bool(td.coverage_report(tmp_path, "MU", [day]).loc[0, "covered"]) is True


def test_429_exponential_backoff_then_success():
    seq = [(429, "OS_LIMIT")] * 3 + [(200, json.dumps({"response": []}))]
    calls = []

    def getter(url):
        calls.append(url)
        return seq[len(calls) - 1]

    slept = []
    c = td.ThetaClient(getter=getter, sleep=slept.append)
    assert c.get_json("/option/list/expirations", symbol="SPY") == []
    assert slept == [1.0, 2.0, 4.0]
