"""R-056 账户层薄封装（zr.sim.stock 逐笔调用）：单一仓位、120 分钟固定持有、15:50 清仓、
满仓股数。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.research import leakcheck as lc
from zr.research.r056_mu_trend_holdout import complete_days, simulate
from zr.research.timing.data import ACCT


def _grid(n_days: int = 3) -> pd.DataFrame:
    b = lc.synthetic_bars(n_days=n_days, seed=2)
    t = b.index.tz_convert("America/New_York")
    return pd.DataFrame(
        {
            "day": np.asarray(t.date),
            "mod": np.asarray(t.hour * 60 + t.minute),
            "ts": b.index,
            "filled": False,
            **{c: b[c].to_numpy() for c in ("open", "high", "low", "close", "volume")},
        }
    )


def test_one_position_fixed_hold_and_flat():
    g = _grid()
    sig = np.ones(len(g), int)  # 每个判定时刻都做多
    T = simulate(g, sig, sorted(set(g["day"])), "B")
    et = (
        pd.to_datetime(T["entry_ts"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    xt = (
        pd.to_datetime(T["exit_ts"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    for _, x in T.assign(et=et, xt=xt).groupby("day"):
        assert x["et"].tolist() == ["10:01", "12:01", "14:01"]
        assert x["xt"].tolist() == ["12:01", "14:01", "15:50"]
        assert (x["exit_ts"].to_numpy()[:-1] <= x["entry_ts"].to_numpy()[1:]).all()
    first = T.iloc[0]
    assert first["qty"] == int(np.floor(ACCT["B"] / first["entry"]))
    assert first["equity_before"] == ACCT["B"]


def test_long_only_skips_shorts_and_complete_days():
    g = _grid(2)
    T = simulate(g, -np.ones(len(g), int), sorted(set(g["day"])), "B", long_only=True)
    assert T.empty
    g.loc[g.index[-30:], "filled"] = True  # 最后一天 15:30 以后缺 K
    assert complete_days(g) == {sorted(set(g["day"]))[0]}
