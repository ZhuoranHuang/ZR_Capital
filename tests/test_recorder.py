import datetime as dt
import re

import pyarrow.parquet as pq

from zr.data.recorder import (
    ParquetSink,
    StrikeInfo,
    build_subscription_set,
    diff_subscriptions,
    underlying_of,
)
from zr.data.universe import ChainRule

TODAY = dt.date(2026, 10, 21)
_SYM_RE = re.compile(r"^(?P<base>[A-Z]+)(?P<exp>\d{6})[CP](?P<strike>\d+)\.US$")


def _strike(symbol: str) -> float:
    return int(_SYM_RE.match(symbol).group("strike")) / 1000


def _chain(base: str, spot: float, n_each_side: int, expiries: list[dt.date], step: float = 1.0):
    chain = {}
    for exp in expiries:
        tag = exp.strftime("%y%m%d")
        chain[exp] = [
            StrikeInfo(s, f"{base}{tag}C{int(s * 1000)}.US", f"{base}{tag}P{int(s * 1000)}.US")
            for s in (spot + step * i for i in range(-n_each_side, n_each_side + 1))
        ]
    return chain


def test_underlying_of():
    assert underlying_of("SPY.US") == "SPY"
    assert underlying_of("SPY261021C500000.US") == "SPY"
    assert underlying_of("BRK.B261021P400000.US") == "BRK.B"


def test_subscription_set_strike_window_and_expiries():
    exps = [TODAY + dt.timedelta(days=k) for k in (-1, 0, 1, 2)]
    chains = {"SPY.US": _chain("SPY", 500.0, 30, exps)}
    rules = {"SPY.US": ChainRule(n_expiries=2, strikes_each_side=10)}
    subs = build_subscription_set(chains, {"SPY.US": 500.0}, TODAY, rules, cap=450)
    assert subs[0] == "SPY.US"
    opts = subs[1:]
    # 2 个到期日 × 21 档 × call+put
    assert len(opts) == 2 * 21 * 2
    # 只含今日与次日到期；昨日（已过期）与第三个到期日不在内
    tags = {_SYM_RE.match(s).group("exp") for s in opts}
    assert tags == {TODAY.strftime("%y%m%d"), (TODAY + dt.timedelta(days=1)).strftime("%y%m%d")}
    strikes = {_strike(s) for s in opts}
    assert min(strikes) == 490.0 and max(strikes) == 510.0


def test_subscription_set_cap_trims_farthest():
    exps = [TODAY, TODAY + dt.timedelta(days=1)]
    chains = {
        "SPY.US": _chain("SPY", 500.0, 40, exps),
        "QQQ.US": _chain("QQQ", 430.0, 40, exps),
    }
    rules = {"SPY.US": ChainRule(2, 40), "QQQ.US": ChainRule(2, 40)}
    spots = {"SPY.US": 500.0, "QQQ.US": 430.0}
    full = build_subscription_set(chains, spots, TODAY, rules, cap=10_000)
    assert len(full) == 2 + 2 * 2 * 81 * 2
    capped = build_subscription_set(chains, spots, TODAY, rules, cap=100)
    assert len(capped) == 100
    assert capped[:2] == ["SPY.US", "QQQ.US"]

    # 留下的都是离 ATM 最近的
    def rel_dist(s: str) -> float:
        spot = spots[f"{_SYM_RE.match(s).group('base')}.US"]
        return abs(_strike(s) - spot) / spot

    kept_far = max(rel_dist(s) for s in capped[2:])
    dropped_near = min(rel_dist(s) for s in set(full[2:]) - set(capped[2:]))
    assert kept_far <= dropped_near


def test_diff_subscriptions():
    add, rm = diff_subscriptions({"A", "B"}, {"B", "C"})
    assert add == ["C"] and rm == ["A"]


def _row(symbol: str, kind: str, push_ts: int, payload: str, recv_ns: int):
    return {
        "recv_ts_utc_ns": recv_ns,
        "push_ts": push_ts,
        "symbol": symbol,
        "kind": kind,
        "payload": payload,
    }


def test_parquet_sink_writes_partitions_and_dedups(tmp_path):
    sink = ParquetSink(tmp_path, flush_interval_s=9999, max_buffer=9999)
    # 2026-10-21 14:00 UTC = 10:00 ET
    recv = int(dt.datetime(2026, 10, 21, 14, 0, tzinfo=dt.UTC).timestamp() * 1e9)
    sink.add(_row("SPY.US", "quote", 1, '{"p":1}', recv))
    sink.add(_row("SPY.US", "quote", 1, '{"p":1}', recv + 5))  # 重复
    sink.add(_row("SPY261021C500000.US", "depth", 2, '{"a":[]}', recv))
    sink.add(_row("QQQ.US", "trade", 3, '{"t":1}', recv))
    files = sink.flush()
    assert len(files) == 3
    assert sink.rows_dropped_dup == 1
    assert sink.rows_written == 3
    dirs = {p.parent.relative_to(tmp_path).as_posix() for p in files}
    assert dirs == {
        "date=2026-10-21/underlying=SPY",
        "date=2026-10-21/underlying=QQQ",
    }
    spy_quote = next(p for p in files if p.name.startswith("quote_"))
    t = pq.read_table(spy_quote).to_pylist()
    assert t == [_row("SPY.US", "quote", 1, '{"p":1}', recv)]
    assert sink.flush() == []


def test_parquet_sink_auto_flush_on_buffer_size(tmp_path):
    sink = ParquetSink(tmp_path, flush_interval_s=9999, max_buffer=3)
    recv = int(dt.datetime(2026, 10, 21, 14, 0, tzinfo=dt.UTC).timestamp() * 1e9)
    for i in range(3):
        sink.add(_row("SPY.US", "quote", i, "{}", recv + i))
    assert sink.files_written == 1
    assert sink.rows_written == 3
