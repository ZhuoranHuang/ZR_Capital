"""美股期权最小报价单位（tick）。

规则来源：长桥帮助中心 https://support.longbridge.hk/topics/us-trade/c4los9（2026-10-22 读取）与
OCC Penny Interval Program（名单由 OCC/交易所维护，运行时须核对，见 D-015）：
- Penny 名单内：权利金 < 3.00 美元 tick 0.01；≥ 3.00 美元 tick 0.05
- SPY / QQQ / IWM：全部价位 0.01
- 名单外：权利金 < 3.00 美元 tick 0.05；≥ 3.00 美元 tick 0.10

本模块只做纯计算；名单 `PENNY_UNDERLYINGS` 是运行时可覆盖的默认值（`set_penny_list`），
不是权威来源。
"""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

# 全价位 0.01 的 ETF（长桥 FAQ 明确列出）
FULL_PENNY = frozenset({"SPY", "QQQ", "IWM"})

# Penny 名单默认值：仅含本项目标的；完整名单运行时从 OCC 核对后用 set_penny_list 覆盖
_penny: set[str] = {"SPY", "QQQ", "IWM", "MU", "AMD"}

_SYM_RE = re.compile(r"^(?P<und>[A-Z.\-]+?)\d{6}[CP]\d+\.US$")
THRESHOLD = Decimal("3.00")


def underlying_of(symbol: str) -> str:
    m = _SYM_RE.match(symbol)
    return m["und"] if m else symbol.split(".")[0]


def set_penny_list(underlyings: set[str] | list[str]) -> None:
    """用核对后的 OCC Penny 名单覆盖默认值（大写、不带 .US）。"""
    _penny.clear()
    _penny.update(u.upper() for u in underlyings)


def is_penny(symbol: str) -> bool:
    return underlying_of(symbol).upper() in _penny


def tick_size(symbol: str, premium: float | Decimal) -> Decimal:
    """给定合约与权利金，返回 tick。premium 为每股价格（美元）。"""
    und = underlying_of(symbol).upper()
    p = Decimal(str(premium))
    if p < 0:
        raise ValueError("premium 不能为负")
    if und in FULL_PENNY:
        return Decimal("0.01")
    if und in _penny:
        return Decimal("0.01") if p < THRESHOLD else Decimal("0.05")
    return Decimal("0.05") if p < THRESHOLD else Decimal("0.10")


def round_to_tick(symbol: str, price: float | Decimal, mode: str = "nearest") -> Decimal:
    """把价格对齐到 tick。mode: nearest / up / down。"""
    tick = tick_size(symbol, price)
    p = Decimal(str(price))
    rounding = {"nearest": ROUND_HALF_UP, "up": ROUND_CEILING, "down": ROUND_FLOOR}[mode]
    return (p / tick).quantize(Decimal("1"), rounding=rounding) * tick


def ticks_away(symbol: str, price: float | Decimal, n: int) -> Decimal:
    """价格偏移 n 个 tick（n 可为负），结果再对齐 tick。"""
    p = Decimal(str(price))
    return round_to_tick(symbol, p + tick_size(symbol, p) * n)
