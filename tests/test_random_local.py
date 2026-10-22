"""时段匹配随机对照：只在真实入场 ±window 分钟内抽样。"""

import datetime as dt

import numpy as np
import pandas as pd

from zr.research import outcomes as oc
from zr.research.replay_rule import Lookup, random_control, random_control_local


def _lookup():
    day = dt.date(2024, 3, 5)
    mins = np.arange(oc.MINUTES)
    net = np.where(mins < 60, -50.0, 1.0)  # 开盘一小时很贵，之后 +1
    rows = [
        {"day": day, "minute": m, "direction": d, "cap": c, "net": net[m], "dte": 1}
        for m in mins
        for d in (1, -1)
        for c in oc.CAPS
    ]
    return Lookup(pd.DataFrame(rows)), day


def test_local_control_draws_near_real_minute():
    lk, day = _lookup()
    real = pd.DataFrame({"day": [day] * 3, "minute": [200, 250, 300], "net": [0.0, 0.0, 0.0]})
    loc = random_control_local(lk, real, 1, oc.CAPS[0], window=15, n=200)
    # 附近分钟全是 +1 → 随机合计恒为 3 > 真实 0
    assert loc["local_p50"] == 3.0 and loc["real_pctile_local"] == 0.0
    elig = np.zeros((1, oc.MINUTES), dtype=bool)
    elig[:, oc.FIRST_ENTRY : oc.LAST_ENTRY + 1] = True
    glob = random_control(lk, real, 1, oc.CAPS[0], elig, n=200)
    # 全天随机会抽到开盘高成本分钟 → 真实合计显得「好」
    assert glob["real_pctile"] > loc["real_pctile_local"]
