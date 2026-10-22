"""日内单腿筛选规则（D-030）的判定逻辑；合成持仓段，不读真实交易史。"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from zr.journal.intraday_list import classify


def _ep(rows):
    out = []
    for sym, right, o, c, side in rows:
        ot = pd.Timestamp(o, tz="America/New_York")
        ct = pd.Timestamp(c, tz="America/New_York") if c else pd.NaT
        out.append(
            {
                "symbol": sym,
                "und": "MU",
                "right": right,
                "side": side,
                "open_ts": ot,
                "close_ts": ct,
                "closed": c is not None,
                "interval_end": ct
                if c
                else pd.Timestamp("2026-08-14 16:00", tz="America/New_York"),
                "expiry": dt.date(2026, 8, 14),
            }
        )
    return pd.DataFrame(out)


def test_rules():
    ep = _ep(
        [
            ("A_C", "C", "2026-08-10 10:00", "2026-08-11 10:00", "long"),  # 过夜
            (
                "B_C",
                "C",
                "2026-08-12 10:00",
                "2026-08-12 10:30",
                "long",
            ),  # 与 B_P 重叠 20 分钟 → 剔除
            ("B_P", "P", "2026-08-12 10:10", "2026-08-12 10:40", "long"),
            ("C_C", "C", "2026-08-12 11:00", "2026-08-12 11:03", "long"),  # 开仓相差 1 分钟 → 剔除
            ("C_P", "P", "2026-08-12 11:01", "2026-08-12 11:02", "long"),
            ("D_C", "C", "2026-08-12 12:00", "2026-08-12 12:10", "long"),  # 先平后开 → 都保留
            ("D_P", "P", "2026-08-12 12:10", "2026-08-12 12:20", "long"),
            (
                "E_C",
                "C",
                "2026-08-12 13:00",
                "2026-08-12 13:10",
                "long",
            ),  # 重叠 3 分钟、开仓差 7 分钟 → 待确认
            ("E_P", "P", "2026-08-12 13:07", "2026-08-12 13:20", "long"),
            ("F_P", "P", "2026-08-13 09:50", None, "short"),  # 卖出开仓 → 其他
        ]
    )
    cat = dict(zip(ep["symbol"], classify(ep)["category"], strict=True))
    assert cat["A_C"] == "剔除" and cat["B_C"] == cat["B_P"] == "剔除"
    assert cat["C_C"] == cat["C_P"] == "剔除"
    assert cat["D_C"] == cat["D_P"] == "入选"
    assert cat["E_C"] == cat["E_P"] == "待确认"
    assert cat["F_P"] == "其他"
