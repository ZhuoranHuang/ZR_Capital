import datetime as dt

import numpy as np
import pandas as pd

from zr.journal import analyze_history_2 as ah

ET = ah.ET


def _trip(
    entry: str, exit_: str, entry_px: float = 2.0, exit_px: float = 1.0, qty: int = 2
) -> pd.Series:
    e = pd.Timestamp(entry, tz=ET)
    x = pd.Timestamp(exit_, tz=ET)
    return pd.Series(
        {
            "symbol": "MU260722P850000.US",
            "und": "MU",
            "expiry": dt.date(2026, 7, 22),
            "right": "P",
            "strike": 850.0,
            "entry_et": e,
            "exit_et": x,
            "hold_min": (x - e).total_seconds() / 60,
            "qty": qty,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "fees_actual": 4.0,
            "pnl_net": 0.0,
        }
    )


def _path(start: str, mids: list[float], spread: float = 0.02) -> pd.DataFrame:
    ts = pd.date_range(pd.Timestamp(start, tz=ET).tz_convert("UTC"), periods=len(mids), freq="min")
    m = np.array(mids)
    return pd.DataFrame({"ts": ts, "bid": m - spread / 2, "ask": m + spread / 2, "mid": m})


def test_time_stop_uses_bid_30_minutes_after_entry():
    tr = _trip("2026-07-20 10:00:20", "2026-07-20 11:30:00")
    path = _path("2026-07-20 10:00", list(np.linspace(2.0, 1.0, 100)))
    px, note = ah.time_stop_exit(tr, path)
    assert note == "time_stop"
    # 10:00:20 + 30 分钟 → 向上取整到 10:31，对应 path 第 31 个点的 bid
    assert abs(px - (path["bid"].iloc[31])) < 1e-12
    short = _trip("2026-07-20 10:00", "2026-07-20 10:20")
    assert ah.time_stop_exit(short, path) == (short["exit_px"], "unchanged")


def test_premium_stop_triggers_on_mid_and_exits_at_bid():
    tr = _trip("2026-07-20 10:00:00", "2026-07-20 11:00:00", entry_px=2.0)
    mids = [2.0, 1.9, 1.8, 1.5, 1.39, 1.2] + [1.0] * 60
    path = _path("2026-07-20 10:00", mids)
    px, note = ah.premium_stop_exit(tr, path)
    assert note == "premium_stop"
    assert abs(px - (1.39 - 0.01)) < 1e-12  # 第一个 mid ≤ 1.40 的分钟，按 bid
    calm = _path("2026-07-20 10:00", [2.0] * 70)
    assert ah.premium_stop_exit(tr, calm) == (tr["exit_px"], "unchanged")


def test_recompute_and_stats():
    trips = pd.DataFrame([_trip("2026-07-20 10:00", "2026-07-20 10:10", 2.0, 1.0, 2)])
    pnl = ah.recompute(trips, pd.Series([2.5]))
    assert pnl.iloc[0] == (2.5 - 2.0) * 100 * 2 - 4.0
    s = ah.stats(pd.Series([100.0, -50.0, 30.0, -10.0]))
    assert s["win_rate"] == 0.5 and s["payoff"] == round(65 / 30, 2) and s["net_usd"] == 70.0


def test_longest_win_streak():
    win = pd.Series([True, False, True, True, True, False, True, True])
    assert ah.longest_win_streak(win) == (2, 4)
    assert ah.longest_win_streak(pd.Series([False, False])) == (0, -1)


def test_random_control_is_reproducible_and_reports_percentile():
    idx = pd.date_range(
        pd.Timestamp("2026-07-20 09:30", tz=ET).tz_convert("UTC"), periods=390 * 3, freq="min"
    )
    close = pd.Series(
        100 + np.cumsum(np.random.default_rng(1).normal(0, 0.05, len(idx))), index=idx
    )
    bars = {"MU": pd.DataFrame({"open": close, "close": close})}
    trips = pd.DataFrame(
        [
            _trip("2026-07-20 10:00", "2026-07-20 10:30"),
            _trip("2026-07-21 11:00", "2026-07-21 11:05"),
        ]
    )
    trips["pnl_net"] = [10.0, -5.0]
    a = ah.random_control(trips, bars, n=50, seed=3)
    b = ah.random_control(trips, bars, n=50, seed=3)
    assert a == b
    assert a["covered_trips"] == 2 and 0 <= a["real_model_percentile"] <= 100
    assert a["random_p05"] <= a["random_p50"] <= a["random_p95"]
