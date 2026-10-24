"""case_charts 的选取规则：盈利 / 亏损各取最近 N 笔，按时间倒序，0 不计。"""

import pandas as pd

from zr.research import case_charts as cc


def test_pick_latest_n_each():
    df = pd.DataFrame(
        {
            "entry_ns": range(20),
            "ret_pct": [0.1, -0.1, 0.0] * 6 + [0.2, -0.2],
        }
    )
    s = cc.pick(df)
    w, lo = s[s["group"] == "盈利"], s[s["group"] == "亏损"]
    assert len(w) == cc.N_EACH and len(lo) == cc.N_EACH
    assert (w["ret_pct"] > 0).all() and (lo["ret_pct"] < 0).all()
    assert list(w["entry_ns"]) == sorted(w["entry_ns"], reverse=True)
    assert w["entry_ns"].max() == 18 and lo["entry_ns"].max() == 19
