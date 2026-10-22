import datetime as dt

import numpy as np
import pandas as pd

from zr.research.seconds_v2 import detect_moves, option_response, pick_contract

S = 1_000_000_000


def test_detect_moves_threshold_and_spacing():
    t = np.arange(0, 60) * S
    v = np.full(60, 100.0)
    v[20:] = 100.2  # 第 20 秒起 +0.2%
    v[40:] = 99.9
    px = pd.Series(v, index=t)
    ev = detect_moves(px, 5, 0.001)
    assert ev["direction"].tolist() == [1, -1]
    assert ev["ts"].tolist() == [20 * S, 40 * S]


def test_pick_contract_and_response():
    day = dt.date(2026, 10, 21)
    idx = np.array([0, 10, 30, 60, 120]) * S
    books = {
        "MU261021C920000.US": pd.DataFrame(
            {"bid": [1.0, 1.1, 1.2, 1.0, 0.9], "bid_vol": [5, 5, 5, 0, 5], "ask": 1.2}, index=idx
        ),
        "MU261023C920000.US": pd.DataFrame(
            {"bid": [2.0] * 5, "bid_vol": [5] * 5, "ask": 2.2}, index=idx
        ),
        "MU261021P910000.US": pd.DataFrame(
            {"bid": [1.0] * 5, "bid_vol": [5] * 5, "ask": 1.1}, index=idx
        ),
    }
    assert pick_contract(books, day, 915.0, 1) == "MU261021C920000.US"  # 当日到期优先
    assert pick_contract(books, day, 915.0, -1) == "MU261021P910000.US"
    ev = pd.DataFrame({"ts": [0], "price": [915.0], "direction": [1]})
    r = option_response(ev, books, day).iloc[0]
    assert np.isclose(r["bid_chg_10s"], 0.1) and np.isclose(r["bid_chg_120s"], -0.1)
    assert r["tradable_30s"] and not r["tradable_60s"]  # 60 秒时 bid 量为 0
