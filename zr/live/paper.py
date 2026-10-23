"""纸面离场单：在实时盘口上模拟限价单与改价阶梯（不向券商下任何单）。

规则与 zr/backtest/fill_limit.py 的 execute(ladder="reprice") 一致，改成逐事件推进：
- kind="limit"：按 rule.price（passive / mid / aggressive，卖出镜像、tick 按 D-015、
  越过对手价按对手价）在下单时的盘口挂单；T 秒内 买一 ≥ 挂单价（量 ≤ 买一挂单量）
  或期权成交价 > 挂单价（量 ≤ 该笔成交量）即按挂单价成交；
  超时改到「中间价按 tick 向上取整 − 1 tick」（不低于买一）再等 T 秒；
  仍未全成则剩余按当时买一成交。成交价恰好等于挂单价的期权成交不算成交（实时没有排队位置信息，
  按保守处理；回测的 queue_fill_prob 不用在这里）。
- kind="stop"：止损类离场，下单后第一个有效盘口直接按买一成交。
只做卖出（离场）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from zr.backtest.fill_limit import LimitRule, _round_passive, _tick


def valid(bid: float | None, ask: float | None) -> bool:
    return (
        bid is not None
        and ask is not None
        and math.isfinite(bid)
        and math.isfinite(ask)
        and ask > 0
        and bid >= 0
        and ask >= bid
    )


def limit_px(bid: float, ask: float, rule: str, symbol: str) -> float:
    """卖出挂单价（fill_limit.limit_price 的卖出分支，单个盘口）。"""
    tick = _tick(symbol, (bid + ask) / 2)
    if rule == "passive":
        px = ask - tick
    elif rule == "mid":
        px = _round_passive((bid + ask) / 2, tick, -1)
    elif rule == "aggressive":
        px = bid + tick
    else:
        raise ValueError(rule)
    return max(round(px, 6), bid)


@dataclass
class PaperSell:
    symbol: str
    qty: int
    reason: str
    kind: str  # limit / stop
    created_ns: int
    rule: LimitRule
    px: float = math.nan
    stage: str = "new"  # new / limit / reprice / done
    stage_end: int = 0
    fills: list = field(default_factory=list)  # [(ns, px, qty, stage)]
    bid_at_create: float = math.nan

    @property
    def left(self) -> int:
        return self.qty - sum(f[2] for f in self.fills)

    @property
    def done(self) -> bool:
        return self.left <= 0

    @property
    def avg_px(self) -> float:
        q = sum(f[2] for f in self.fills)
        return sum(f[1] * f[2] for f in self.fills) / q if q else math.nan

    def _fill(self, ns: int, px: float, q: int) -> None:
        q = min(q, self.left)
        if q > 0:
            self.fills.append((ns, px, q, self.stage))
        if self.done:
            self.stage = "done"

    def on_quote(self, ns: int, bid: float, ask: float, bid_size: float | None = None) -> None:
        if self.done or not valid(bid, ask) or ns < self.created_ns:
            return
        if self.stage == "new":
            self.bid_at_create = bid
            if self.kind == "stop":
                self._fill(ns, bid, self.left)
                return
            self.px = limit_px(bid, ask, self.rule.price, self.symbol)
            self.stage, self.stage_end = "limit", ns + int(self.rule.T * 1e9)
        elif ns > self.stage_end:
            if self.stage == "limit":
                tick = _tick(self.symbol, (bid + ask) / 2)
                self.px = max(_round_passive((bid + ask) / 2, tick, -1) - tick, bid)
                self.stage, self.stage_end = "reprice", ns + int(self.rule.T * 1e9)
            else:  # 改价后仍未全成：按买一
                self.stage = "cross"
                self._fill(ns, bid, self.left)
                return
        if bid >= self.px:
            avail = int(bid_size) if bid_size and bid_size > 0 else self.left
            self._fill(ns, self.px, avail)

    def on_trade(self, ns: int, price: float, size: float) -> None:
        if self.done or self.stage not in ("limit", "reprice") or ns < self.created_ns:
            return
        if price > self.px:
            self._fill(ns, self.px, int(size) if size and size > 0 else self.left)

    def force_cross(self, ns: int, bid: float) -> None:
        """强平 / kill switch：剩余部分立即按买一。"""
        if not self.done:
            self.stage = "cross"
            self._fill(ns, bid, self.left)
