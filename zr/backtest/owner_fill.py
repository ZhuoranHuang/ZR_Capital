"""负责人挂单策略的逐笔模拟（成交模型校准修正版；参数见 config/owner_strategy.yaml，
由 zr/research/owner_strategy.py 从 D-030 日内单腿订单提取）。

从决策时刻 + lag 起（委托到达）：
1. 初始挂单价 = 中间价 + 方向 × k0 × spread（k0 从负责人同方向委托的经验分布抽样），按 tick 向己方
   取整，不越过对手价（越过则按对手价，可立即成交）；
2. 等待时长 W 从负责人委托总时长的经验分布抽样（已截到 5–120 秒）；
3. 改价：次数从经验分布抽样，每隔 amend_interval_s 秒改一次；幅度 = chase_a + chase_b × 上次挂单以来
   中间价朝对手方向移动的 tick 数（负责人改价「追价」的线性拟合；无拟合时用 amend_step_ticks），
   不越过当时的对手价；
4. 撤单条件：中间价相对委托到达时朝对手方向跑开 ≥ cancel_away_ticks 个 tick，或等满 W 秒；
5. 成交判定同 zr/backtest/fill_limit（卖一 ≤ 挂单价按卖一量成交、OPRA 成交越过挂单价成交、
   恰好等于时以概率 q 成交）。
结束时未成交的张数：入场（mode="entry"）放弃，记为错过；离场（mode="exit"）按 +0 s 后第一个对手价
成交（一定出得去）。止损类离场不用本策略，直接 fill_limit.cross_now 吃买一。
抽样用 numpy Generator，种子由调用方给定（同一事件结果可复现）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import yaml

from zr.backtest.fill_limit import Book, _rest, _round_passive, _tick, cross_now


@dataclass(frozen=True)
class SideParams:
    k0_spread: tuple
    wait_s: tuple
    amend_counts: tuple
    amend_interval_s: float
    amend_step_ticks: float
    cancel_away_ticks: float | None
    chase_a: float | None = None  # 改价幅度 = a + b × 上次挂单以来中间价朝对手方向的移动（tick）
    chase_b: float | None = None


@dataclass(frozen=True)
class OwnerStrategy:
    buy: SideParams
    sell: SideParams
    queue_fill_prob: float = 0.9
    lag_s: float = 2.0

    @classmethod
    def from_yaml(cls, path, queue_fill_prob: float = 0.9, lag_s: float = 2.0) -> OwnerStrategy:
        d = yaml.safe_load(open(path, encoding="utf-8"))

        def sp(x: dict) -> SideParams:
            return SideParams(
                tuple(x["k0_spread"]),
                tuple(x["wait_s"]),
                tuple(x["amend_counts"]),
                float(x["amend_interval_s"]),
                float(x["amend_step_ticks"]),
                None if x.get("cancel_away_ticks") is None else float(x["cancel_away_ticks"]),
                x.get("chase_a"),
                x.get("chase_b"),
            )

        return cls(sp(d["Buy"]), sp(d["Sell"]), queue_fill_prob, lag_s)


@dataclass
class OwnerResult:
    filled_qty: float
    avg_px: float
    first_fill_ns: int | None
    last_fill_ns: int | None
    missed_qty: float
    stage: str  # filled / missed / cross
    end_reason: str  # filled / wait / away / deadline
    end_ns: int
    k0: float
    wait_s: float
    n_amend: int
    px0: float


def execute_owner(
    book: Book,
    side: int,
    qty: float,
    decision_ns: int,
    strat: OwnerStrategy,
    symbol: str,
    rng: np.random.Generator,
    mode: str = "entry",
    deadline_ns: int | None = None,
    draw: dict | None = None,
) -> OwnerResult | None:
    """draw 可指定 {k0, wait_s, n_amend}（测试与验证用），否则按经验分布抽样。"""
    p = strat.buy if side == 1 else strat.sell
    k0 = float(draw["k0"]) if draw and "k0" in draw else float(rng.choice(p.k0_spread))
    w = float(draw["wait_s"]) if draw and "wait_s" in draw else float(rng.choice(p.wait_s))
    n_am = int(draw["n_amend"]) if draw and "n_amend" in draw else int(rng.choice(p.amend_counts))
    t0 = decision_ns + int(strat.lag_s * 1e9)
    i0 = book.asof(t0)
    if i0 is None:
        i0 = book.first_at(t0)
    if i0 is None:
        return None
    bid, ask = book.bid[i0], book.ask[i0]
    mid0, spr = (bid + ask) / 2, ask - bid
    tick = _tick(symbol, mid0)
    px = _round_passive(mid0 + side * k0 * spr, tick, side)
    px = max(px, tick) if side == 1 else px
    px = min(px, ask) if side == 1 else max(px, bid)
    px0 = px
    end = t0 + int(w * 1e9)
    reason = "wait"
    if deadline_ns is not None and deadline_ns < end:
        end, reason = deadline_ns, "deadline"
    if p.cancel_away_ticks is not None:
        j0 = int(np.searchsorted(book.q_ns, t0, side="left"))
        j1 = int(np.searchsorted(book.q_ns, end, side="right"))
        mids = (book.bid[j0:j1] + book.ask[j0:j1]) / 2
        away = np.flatnonzero(side * (mids - mid0) >= p.cancel_away_ticks * tick - 1e-9)
        if len(away):
            end, reason = int(book.q_ns[j0 + away[0]]), "away"
    marks = [t0 + int(k * p.amend_interval_s * 1e9) for k in range(1, n_am + 1)]
    marks = [m for m in marks if m < end]
    bounds = [t0, *marks, end]
    fills, left = [], float(qty)
    mid_last = mid0
    for k in range(len(bounds) - 1):
        if k > 0:  # 改价：追价幅度，不越过当时对手价
            j = book.asof(bounds[k])
            step = p.amend_step_ticks
            if j is not None and p.chase_b is not None:
                move = side * (book.mid(j) - mid_last) / tick
                step = p.chase_a + p.chase_b * move
            px = _round_passive(px + side * step * tick, tick, side)
            px = max(px, tick)
            if j is not None:
                px = min(px, book.ask[j]) if side == 1 else max(px, book.bid[j])
                mid_last = book.mid(j)
        f, left = _rest(book, side, px, bounds[k], bounds[k + 1], left, rng, strat.queue_fill_prob)
        fills += f
        if left <= 0:
            break
    stage = "filled" if left <= 0 else "missed"
    end_ns = end
    if left > 0 and mode == "exit":
        c = cross_now(book, side, end, lag_s=0.0)
        if c is not None:
            fills.append((c[0], c[1], left))
            left = 0.0
            stage = "cross"
            end_ns = c[0]
    if left <= 0 and fills:
        end_ns = max(end_ns if stage == "cross" else 0, fills[-1][0])
        reason = "filled" if stage == "filled" else reason
    missed = left if mode == "entry" else 0.0
    if not fills:
        return OwnerResult(
            0.0, math.nan, None, None, missed, "missed", reason, end_ns, k0, w, n_am, px0
        )
    q = sum(f[2] for f in fills)
    avg = sum(f[1] * f[2] for f in fills) / q
    return OwnerResult(
        q, avg, fills[0][0], fills[-1][0], missed, stage, reason, end_ns, k0, w, n_am, px0
    )
