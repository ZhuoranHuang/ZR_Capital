import datetime as dt
import os
from zoneinfo import ZoneInfo

from zr.ops.healthcheck import evaluate, in_session, newest_file_age_s

ET = ZoneInfo("America/New_York")


def _et(h: int, m: int, day: dt.date = dt.date(2026, 10, 21)) -> dt.datetime:  # 周三
    return dt.datetime.combine(day, dt.time(h, m), tzinfo=ET)


def test_in_session_bounds():
    assert not in_session(_et(9, 29))
    assert in_session(_et(9, 30))
    assert in_session(_et(15, 59))
    assert not in_session(_et(16, 0))
    assert not in_session(_et(10, 0, dt.date(2026, 10, 24)))  # 周六


def test_evaluate_outside_session_is_silent():
    assert evaluate(_et(8, 0), recorder_active=False, file_age_s=None) == []


def test_evaluate_flags_missing_process_and_stale_files():
    assert evaluate(_et(10, 0), True, 30.0) == []
    assert evaluate(_et(10, 0), False, 30.0) == ["recorder_not_running"]
    assert evaluate(_et(10, 0), True, None) == ["no_parquet_today"]
    assert evaluate(_et(10, 0), True, 301.0) == ["stale_parquet_301s"]
    assert evaluate(_et(10, 0), False, 900.0) == ["recorder_not_running", "stale_parquet_900s"]


def test_newest_file_age(tmp_path):
    now = dt.datetime(2026, 10, 21, 14, 0, tzinfo=dt.UTC)
    today = dt.date(2026, 10, 21)
    assert newest_file_age_s(tmp_path, now, today) is None
    part = tmp_path / "date=2026-10-21" / "underlying=SPY"
    part.mkdir(parents=True)
    old = part / "quote_a.parquet"
    new = part / "quote_b.parquet"
    old.write_bytes(b"")
    new.write_bytes(b"")
    os.utime(old, (now.timestamp() - 600, now.timestamp() - 600))
    os.utime(new, (now.timestamp() - 40, now.timestamp() - 40))
    assert round(newest_file_age_s(tmp_path, now, today)) == 40
