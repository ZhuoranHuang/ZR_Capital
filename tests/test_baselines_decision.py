import datetime as dt

import numpy as np
import pandas as pd

from zr.research import outcomes as oc
from zr.research.baselines_v2 import equal_weight, random_entry, straddle
from zr.research.decision_sim import _trades
from zr.research.replay_rule import Lookup

D1, D2 = dt.date(2026, 9, 15), dt.date(2026, 9, 16)


def _lookup() -> Lookup:
    rows = []
    for day in (D1, D2):
        for m in range(oc.FIRST_ENTRY, oc.LAST_ENTRY + 1):
            for d in (1, -1):
                for cap in oc.CAPS:
                    rows.append(
                        {
                            "day": day,
                            "minute": m,
                            "direction": d,
                            "cap": cap,
                            "net": float(m % 7 - 3) * d,
                            "dte": 0,
                        }
                    )
    return Lookup(pd.DataFrame(rows))


def test_equal_weight_dedupes_and_maps_decision_minute():
    lk = _lookup()
    ds = pd.DataFrame({"day": [D1, D1, D1], "bar_minute": [9, 9, 20], "direction": [1, 1, -1]})
    ew = equal_weight(ds, lk, 10)
    assert len(ew) == 2
    assert set(ew["minute"]) == {10, 21}
    assert ew.loc[ew["minute"] == 10, "net"].iloc[0] == float(10 % 7 - 3)


def test_random_entry_is_reproducible_and_bounded():
    lk = _lookup()
    counts = pd.Series({D1: 3, D2: 2})
    a, b = random_entry(lk, counts, 10, n=200, seed=1), random_entry(lk, counts, 10, n=200, seed=1)
    assert np.array_equal(a, b)
    assert a.min() >= -3 * 5 and a.max() <= 3 * 5


def test_straddle_sums_both_legs():
    lk = _lookup()
    e = pd.DataFrame({"day": [D1], "minute": [10]})
    s = straddle(lk, e, 10)
    assert s["net"].iloc[0] == s["net_call"].iloc[0] + s["net_put"].iloc[0] == 0.0


def test_decision_trades_take_highest_p_per_minute():
    lk = _lookup()
    res = pd.DataFrame(
        {
            "day": [D1, D1],
            "bar_minute": [9, 9],
            "direction": [1, 1],
            "p": [0.7, 0.6],
            "detector": ["E1", "E3"],
        }
    )
    t = _trades(res, lk, 10)
    assert len(t) == 1 and t["p"].iloc[0] == 0.7
