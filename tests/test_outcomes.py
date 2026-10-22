import datetime as dt

import numpy as np
import pandas as pd

from zr.research import outcomes as oc

DAY = dt.date(2026, 9, 15)  # 周二


def _day(close: np.ndarray, mid: np.ndarray, quotes: dict) -> oc.Day:
    """quotes: {(expiry, right, strike): (bid_array, ask_array)}"""
    bid, ask, rows = {}, {}, []
    for (e, r, k), (b, a) in quotes.items():
        sym = f"MU{e:%y%m%d}{'C' if r == 'call' else 'P'}{int(k * 1000)}.US"
        bid[sym], ask[sym] = b, a
        rows.append({"symbol": sym, "expiry": e, "strike": k, "right": r})
    con = pd.DataFrame(rows)
    expiries = sorted(e for e in con["expiry"].unique() if 0 <= (e - DAY).days <= oc.MAX_DTE)
    chains = {
        (e, r): g.sort_values("strike")[["strike", "symbol"]].reset_index(drop=True)
        for (e, r), g in con.groupby(["expiry", "right"])
    }
    return oc.Day(DAY, close, mid, expiries, chains, bid, ask)


def _flat(v: float) -> np.ndarray:
    return np.full(oc.MINUTES, v)


def test_pick_nearest_expiry_slightly_otm_and_rolls_when_no_ask():
    close = _flat(100.0)
    e0, e1 = DAY, DAY + dt.timedelta(days=2)
    q = {
        (e0, "call", 100.0): (_flat(1.0), _flat(1.1)),
        (e0, "call", 101.0): (_flat(0.5), _flat(0.6)),
        (e0, "put", 99.0): (_flat(0.5), _flat(0.6)),
        (e1, "call", 101.0): (_flat(1.5), _flat(1.6)),
    }
    d = _day(close, _flat(99.0), q)
    sym, exp = oc.pick(d, 1, 100.0, 50)
    assert exp == e0 and sym.endswith("C101000.US")  # 严格大于现价的最近一档
    sym, exp = oc.pick(d, -1, 100.0, 50)
    assert sym.endswith("P99000.US")
    d.ask[sym := [s for s in d.ask if s.endswith("C101000.US") and "260915" in s][0]][50] = np.nan
    assert oc.pick(d, 1, 100.0, 50)[1] == e1  # 当日合约该分钟无 ask → 顺延下一到期日


def test_simulate_take_profit_half_then_struct_stop():
    close = _flat(100.0)
    mid = _flat(99.0)
    close[60:] = 98.0  # 第 60 分钟收盘跌破中轨 → 结构止损
    bid, ask = _flat(1.0), _flat(1.02)
    bid[55] = 1.25  # ≥ 1.02×1.2 = 1.224 → 减半
    bid[60] = 0.9
    d = _day(close, mid, {(DAY, "call", 101.0): (bid, ask)})
    sym = next(iter(d.bid))
    r = oc.simulate(d, 50, 1, sym, cap=30)
    assert r["half_tp"] and r["reason"] == "struct_stop" and r["exit"] == 60
    gross = 1.25 * 100 + 0.9 * 100 - 1.02 * 100 * 2
    assert abs(r["gross"] - gross) < 1e-9 and r["net"] < r["gross"]


def test_simulate_cap_and_eod():
    d = _day(_flat(100.0), _flat(99.0), {(DAY, "call", 101.0): (_flat(1.0), _flat(1.05))})
    sym = next(iter(d.bid))
    r = oc.simulate(d, 100, 1, sym, cap=10)
    assert r["reason"] == "cap" and r["exit"] == 110 and r["hold"] == 10
    r = oc.simulate(d, 380, 1, sym, cap=30)
    assert r["reason"] == "eod" and r["exit"] == oc.FORCE_EXIT


def test_day_outcomes_flags_d001_and_covers_both_directions():
    close = _flat(100.0)
    q = {
        (DAY, "call", 101.0): (_flat(1.0), _flat(1.05)),
        (DAY, "put", 99.0): (_flat(1.0), _flat(1.05)),
    }
    out = oc.day_outcomes(_day(close, _flat(100.0), q))
    assert set(out["direction"]) == {1, -1} and set(out["cap"]) == {10, 30}
    assert out["minute"].min() == oc.FIRST_ENTRY and out["minute"].max() == oc.LAST_ENTRY
    assert (out.loc[out["minute"] > 45, "d001_violation"]).all()
    assert not out.loc[out["minute"] <= 45, "d001_violation"].any()


def test_bid_max_ret_windows():
    bid, ask = _flat(1.0), _flat(1.1)
    bid[53] = 1.5
    q = {(DAY, "call", 101.0): (bid, ask), (DAY, "put", 99.0): (_flat(1.0), _flat(1.1))}
    out = oc.day_outcomes(_day(_flat(100.0), _flat(100.0), q))
    r = out[(out["minute"] == 50) & (out["direction"] == 1)].iloc[0]
    assert np.isclose(r["bid_max_ret_2"], 1.0 / 1.1 - 1)  # 51–52 分钟
    assert np.isclose(r["bid_max_ret_3"], 1.5 / 1.1 - 1)  # 含 53 分钟
    assert np.isclose(r["bid_max_ret_30"], 1.5 / 1.1 - 1)
