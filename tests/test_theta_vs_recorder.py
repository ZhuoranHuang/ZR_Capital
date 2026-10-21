import datetime as dt
import json

import pandas as pd

from zr.data.theta_vs_recorder import compare, recorder_minute_quotes, theta_minute_quotes


def _depth(symbol: str, ts: pd.Timestamp, bid: float, ask: float) -> dict:
    return {
        "recv_ts_utc_ns": int(ts.value),
        "push_ts": int(ts.timestamp()),
        "symbol": symbol,
        "kind": "depth",
        "payload": json.dumps(
            {
                "bids": [{"price": bid, "volume": 5, "order_num": 0, "position": 1}],
                "asks": [{"price": ask, "volume": 7, "order_num": 0, "position": 1}],
            }
        ),
    }


def test_compare_minute_alignment(tmp_path):
    day = dt.date(2026, 10, 21)
    sym = "SPY261021C774000.US"
    t0 = pd.Timestamp("2026-10-21 18:30:00", tz="UTC")
    rec_dir = tmp_path / "recorder/date=2026-10-21/underlying=SPY"
    rec_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            _depth(sym, t0 + pd.Timedelta(seconds=10), 0.80, 0.82),
            _depth(sym, t0 + pd.Timedelta(seconds=50), 0.81, 0.82),  # 该分钟最后一条
            _depth(sym, t0 + pd.Timedelta(seconds=70), 0.83, 0.85),
            _depth("SPY.US", t0, 774.1, 774.2),  # 正股不参与
        ]
    ).to_parquet(rec_dir / "depth_x.parquet", index=False)
    th_dir = tmp_path / "options_1m/underlying=SPY/date=2026-10-21"
    th_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts": [t0, t0 + pd.Timedelta(minutes=1)],
            "symbol": [sym, sym],
            "bid": [0.81, 0.84],
            "ask": [0.82, 0.85],
            "bid_size": [5, 5],
            "ask_size": [7, 7],
        }
    ).to_parquet(th_dir / "quotes.parquet", index=False)

    rec = recorder_minute_quotes(tmp_path, day, "SPY")
    assert len(rec) == 2 and set(rec["symbol"]) == {sym}
    theta = theta_minute_quotes(tmp_path, day, "SPY")
    m, stats = compare(rec, theta)
    assert stats["matched_minutes"] == 2
    assert stats["both_equal_pct"] == 50.0
    assert stats["within_1_tick_pct"] == 100.0
