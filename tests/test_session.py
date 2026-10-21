import datetime as dt
from zoneinfo import ZoneInfo

from zr.data.recorder import SESSION_END, SUBSCRIBE_AT, session_action

ET = ZoneInfo("America/New_York")


def _et(h: int, m: int, day: dt.date = dt.date(2026, 10, 21)) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(h, m), tzinfo=ET)


def test_non_trading_day_skips():
    assert session_action(_et(9, 0), is_trading_day=False) == ("skip", 0.0)


def test_before_subscribe_time_waits_until_0925():
    action, wait = session_action(_et(8, 20), is_trading_day=True)
    assert action == "wait"
    assert wait == 65 * 60


def test_timer_fire_in_edt_and_est_both_wait():
    # 13:20 UTC：夏令时 = 09:20 EDT，冬令时 = 08:20 EST
    edt = dt.datetime(2026, 10, 21, 13, 20, tzinfo=dt.UTC).astimezone(ET)
    est = dt.datetime(2026, 12, 10, 13, 20, tzinfo=dt.UTC).astimezone(ET)
    assert edt.time() == dt.time(9, 20) and est.time() == dt.time(8, 20)
    assert session_action(edt, True) == ("wait", 5 * 60)
    assert session_action(est, True) == ("wait", 65 * 60)


def test_during_session_runs():
    assert session_action(_et(9, 25), True) == ("run", 0.0)
    assert session_action(_et(15, 59), True) == ("run", 0.0)


def test_after_end_is_done():
    assert session_action(_et(16, 5), True) == ("done", 0.0)
    assert session_action(_et(20, 0), True) == ("done", 0.0)


def test_defaults():
    assert SUBSCRIBE_AT == dt.time(9, 25) and SESSION_END == dt.time(16, 5)
