# ruff: noqa: E501
import numpy as np
import pandas as pd

from zr.data import final_audit as fa


def _sec(rows):
    d = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "transactions"])
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    return d


def test_sec_to_min_aggregates_ohlcv():
    s = _sec(
        [
            ("2024-01-02 14:30:01", 10.0, 10.5, 9.9, 10.2, 100, 3),
            ("2024-01-02 14:30:40", 10.2, 10.8, 10.1, 10.6, 50, 2),
            ("2024-01-02 14:31:05", 10.6, 10.6, 10.0, 10.1, 10, 1),
        ]
    )
    m = fa.sec_to_min(s)
    assert len(m) == 2
    r = m.iloc[0]
    assert (r.open, r.high, r.low, r.close, r.volume, r.transactions) == (10.0, 10.8, 9.9, 10.6, 150, 5)


def test_compare_day_counts_mismatch_and_volume_ratio():
    s = _sec(
        [
            ("2024-01-02 14:30:01", 10.0, 10.5, 9.9, 10.2, 100, 3),
            ("2024-01-02 14:31:05", 10.6, 10.6, 10.0, 10.1, 10, 1),
        ]
    )
    m1 = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-02 14:30", "2024-01-02 14:31", "2024-01-02 14:32"], utc=True),
            "open": [10.0, 10.6, 10.1],
            "high": [10.5, 10.6, 10.1],
            "low": [9.9, 10.0, 10.1],
            "close": [10.2, 10.1, 10.1],
            "volume": [100.0, 20.0, 5.0],
            "transactions": [3, 2, 1],
        }
    )
    r = fa.compare_day(s, m1)
    assert r["min_both"] == 2 and r["only_1m"] == 1 and r["only_sec"] == 0
    assert r["eq_px"] == 2 and r["eq_vol"] == 1 and r["eq_all"] == 1
    assert r["vol_sec"] == 110 and r["vol_1m"] == 125


def test_chain_gap_lists_listed_strikes_missing_inside_band():
    listed = np.array([95.0, 100.0, 105.0, 110.0, 115.0])
    have = np.array([100.0, 110.0])
    g = fa._chain_gap(listed, have, 98.0, 112.0)
    assert g == {"chain_in_band": 3, "missing_strikes": 1, "missing_list": "105"}
    assert fa._chain_gap(None, have, 98.0, 112.0)["missing_strikes"] is None
