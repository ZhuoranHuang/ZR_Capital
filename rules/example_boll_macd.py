"""示例规则：BOLL 中轨穿越 + MACD 柱同向（接口示例，参数未检验，不是研究结论）。

多：2 分钟 K 收盘上穿 BOLL(20, 2) 中轨，且 MACD(6, 13, 6) 柱 > 0；空：镜像。
指标口径同 CLAUDE.md「指标标准口径」：BOLL 标准差 ddof=0，MACD 标准 EMA（span=n、adjust=False）。
runner 传入的是 RTH 2 分钟 K（跨日连续、不按日重置），只用当前及之前的 K。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.features.indicators import bollinger, macd

PERIOD = 2
DIRECTION = "both"
EXIT = {"kind": "bracket", "tp_pct": 0.004, "sl_pct": 0.003, "max_bars": 15}
CONTRACT = {"dte_max": 7, "otm_steps": 1}  # 最近可交易到期日（7 天内），虚值 1 档


def signal(bars: pd.DataFrame) -> pd.Series:
    c = bars["close"]
    mid = bollinger(c, 20, 2.0)["mid"]
    hist = macd(c, 6, 13, 6)["hist"]
    above, prev_above = c > mid, (c.shift(1) > mid.shift(1))
    below, prev_below = c < mid, (c.shift(1) < mid.shift(1))
    long_ = above & ~prev_above & mid.shift(1).notna() & (hist > 0)
    short = below & ~prev_below & mid.shift(1).notna() & (hist < 0)
    return pd.Series(np.where(long_, 1, np.where(short, -1, 0)), index=bars.index)
