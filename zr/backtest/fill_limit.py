"""限价单成交模拟（feat/fill-model 第 2 节）。逐笔 NBBO + OPRA 成交，单合约。

挂单价规则（参数，买入；卖出镜像）：
- passive：己方最优价 + 1 tick（买一 + 1 tick；卖出为卖一 − 1 tick）
- mid：中间价（按 tick 向己方取整，不越过中间价）
- aggressive：对手价 − 1 tick（卖一 − 1 tick；卖出为买一 + 1 tick）
任一规则若算出的价格已越过对手价，则按对手价（可立即成交）。
tick 大小按 D-015（zr/execution/ticks）。

成交判定（挂单后 T 秒内，按时间顺序处理事件）：
- 盘口越过：买单 卖一 ≤ 挂单价 → 按挂单价成交，数量 ≤ 当时卖一挂单量
  （无挂单量数据时视为全量）；
- OPRA 成交价严格越过挂单价（买：成交价 < 挂单价）→ 按挂单价成交，数量 ≤ 该笔成交量；
- OPRA 成交价恰好等于挂单价 → 以概率 queue_fill_prob 成交（排队，校准值来自负责人订单），
  数量 ≤ 该笔成交量；
  随机数用固定种子，同一输入结果可复现。
超时（改价阶梯）：
- ladder="reprice"：改到中间价 ± 1 tick（向对手方向，不越过对手价）再等 T 秒，仍未全成则剩余部分
  以对手价成交（离场用：一定出得去）；
- ladder="reprice_abandon"：同样改价再等 T 秒，仍未全成则剩余部分放弃，记为「错过」（入场用）；
- ladder="abandon"：不改价，T 秒后剩余部分放弃。
queue_fill_prob 默认 0.9：负责人 264 个挂单段逐段重放的 Brier 最小值
（docs/research/FILL_MODEL_CALIBRATION.md，zr/research/fill_segments.py）。
止损类离场用 cross_now：+lag 后第一个有效 bid（卖出）直接成交。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from zr.execution.ticks import tick_size

RULES = ("passive", "mid", "aggressive")


@dataclass(frozen=True)
class LimitRule:
    price: str = "mid"
    T: float = 5.0
    ladder: str = "reprice"  # reprice / reprice_abandon / abandon
    queue_fill_prob: float = 0.9
    lag_s: float = 2.0
    seed: int = 0


@dataclass
class Book:
    """单合约的有效报价与成交（时间为 int64 纳秒，升序）。"""

    q_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    t_ns: np.ndarray = field(default_factory=lambda: np.array([], np.int64))
    t_px: np.ndarray = field(default_factory=lambda: np.array([], float))
    t_sz: np.ndarray = field(default_factory=lambda: np.array([], float))

    @classmethod
    def from_frames(cls, q: pd.DataFrame, t: pd.DataFrame | None = None) -> Book:
        q = q[(q["ask"] > 0) & (q["bid"] > 0) & (q["ask"] >= q["bid"])]
        q = q.sort_values("ts", kind="stable")
        ns = pd.DatetimeIndex(pd.to_datetime(q["ts"], utc=True)).as_unit("ns").asi8
        bs = q["bid_size"].to_numpy(float) if "bid_size" in q else np.full(len(q), np.inf)
        as_ = q["ask_size"].to_numpy(float) if "ask_size" in q else np.full(len(q), np.inf)
        b = cls(ns, q["bid"].to_numpy(float), q["ask"].to_numpy(float), bs, as_)
        if t is not None and len(t):
            t = t.sort_values("ts", kind="stable")
            b.t_ns = pd.DatetimeIndex(pd.to_datetime(t["ts"], utc=True)).as_unit("ns").asi8
            b.t_px = t["price"].to_numpy(float)
            b.t_sz = t["size"].to_numpy(float) if "size" in t else np.full(len(t), np.inf)
        return b

    def asof(self, ns: int) -> int | None:
        i = int(np.searchsorted(self.q_ns, ns, side="right")) - 1
        return i if i >= 0 else None

    def first_at(self, ns: int) -> int | None:
        i = int(np.searchsorted(self.q_ns, ns, side="left"))
        return i if i < len(self.q_ns) else None

    def mid(self, i: int) -> float:
        return (self.bid[i] + self.ask[i]) / 2


@dataclass
class Result:
    filled_qty: float
    avg_px: float  # 已成交部分的均价（含改价阶梯与最后吃对手价）
    first_fill_ns: int | None
    last_fill_ns: int | None
    missed_qty: float
    stage: str  # limit / reprice / cross / missed
    limit_px: float
    end_ns: int | None = None  # 挂单结束时刻（最后一段的终点；吃对手价时为成交时刻）


@dataclass
class PriceImprovement:
    """价格改善模型（R-046，参数 config/price_improvement.yaml 由
    zr/research/price_improvement.py 生成）：可立即成交的订单成交价 = 对手价 − 方向 × 改善量；
    改善量 = 价差 × 比例，比例按「类别（market / limit）× 价差档」从负责人订单的经验值抽样
    （样本不足的格子退回同类别、再退回全部），按 tick 取整；成交价不优于己方最优价，
    限价单不越过自己的限价。"""

    params: dict

    @classmethod
    def from_yaml(cls, path) -> PriceImprovement:
        import yaml

        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def frac(self, kind: str, spread_ticks: float, rng: np.random.Generator) -> float:
        p = self.params
        bins, labels = p["spread_bins"], p["spread_labels"]
        i = int(np.searchsorted(bins, spread_ticks, side="left")) - 1
        lab = labels[min(max(i, 0), len(labels) - 1)]
        ent = p["by_kind"].get(kind) or {}
        pool = (ent.get("by_spread") or {}).get(lab)
        if not pool:
            pool = ent.get("all") if len(ent.get("all") or []) >= p["min_n"] else p["all"]
        return float(rng.choice(pool)) if pool else 0.0

    def price(
        self,
        side: int,
        bid: float,
        ask: float,
        kind: str,
        symbol: str,
        rng: np.random.Generator,
        limit: float | None = None,
    ) -> float:
        spr = ask - bid
        tick = _tick(symbol, (ask + bid) / 2)
        imp = round(self.frac(kind, spr / tick if tick > 0 else 0.0, rng) * spr / tick) * tick
        px = ask - imp if side == 1 else bid + imp
        px = max(px, bid) if side == 1 else min(px, ask)
        if limit is not None:
            px = min(px, limit) if side == 1 else max(px, limit)
        return round(px, 6)


def _tick(symbol: str, px: float) -> float:
    return float(tick_size(symbol, max(px, 0.01)))


def _round_passive(px: float, tick: float, side: int) -> float:
    """按 tick 向己方取整（买向下、卖向上），避免浮点误差。"""
    k = px / tick
    k = math.floor(k + 1e-9) if side == 1 else math.ceil(k - 1e-9)
    return round(k * tick, 6)


def limit_price(book: Book, i: int, side: int, rule: str, symbol: str) -> float:
    b, a = book.bid[i], book.ask[i]
    tick = _tick(symbol, (a + b) / 2)
    if rule == "passive":
        px = b + tick if side == 1 else a - tick
    elif rule == "mid":
        px = _round_passive((a + b) / 2, tick, side)
    elif rule == "aggressive":
        px = a - tick if side == 1 else b + tick
    else:
        raise ValueError(rule)
    px = round(px, 6)
    return min(px, a) if side == 1 else max(px, b)  # 越过对手价则按对手价


def _rest(
    book: Book,
    side: int,
    px: float,
    start: int,
    end: int,
    want: float,
    rng: np.random.Generator,
    qprob: float,
    pi: PriceImprovement | None = None,
    symbol: str | None = None,
) -> tuple[list[tuple[int, float, float]], float]:
    """挂单 [start, end] 纳秒内的成交事件：返回 [(ns, 价格, 数量)]、剩余数量。"""
    fills = []
    left = want
    # 到达时有效的报价已越过挂单价（可立即成交的挂单）→ 在到达时刻按对手价成交（价格改善：
    # 吃掉的是盘口上的挂单，成交价是对手价而不是自己的限价）
    i0 = book.asof(start)
    if i0 is not None and (book.ask[i0] <= px if side == 1 else book.bid[i0] >= px):
        avail = book.ask_size[i0] if side == 1 else book.bid_size[i0]
        q = min(left, avail if np.isfinite(avail) and avail > 0 else left)
        opp = float(book.ask[i0] if side == 1 else book.bid[i0])
        if pi is not None and symbol is not None:
            opp = pi.price(side, float(book.bid[i0]), float(book.ask[i0]), "limit", symbol, rng, px)
        fills.append((int(start), opp, q))
        left -= q
    qi = int(np.searchsorted(book.q_ns, start, side="left"))
    qe = int(np.searchsorted(book.q_ns, end, side="right"))
    ti = int(np.searchsorted(book.t_ns, start, side="left"))
    te = int(np.searchsorted(book.t_ns, end, side="right"))
    ev = [(int(book.q_ns[k]), 0, k) for k in range(qi, qe)] + [
        (int(book.t_ns[k]), 1, k) for k in range(ti, te)
    ]
    ev.sort()
    for ns, kind, k in ev:
        if left <= 0:
            break
        if kind == 0:
            cross = book.ask[k] <= px if side == 1 else book.bid[k] >= px
            if cross:
                avail = book.ask_size[k] if side == 1 else book.bid_size[k]
                q = min(left, avail if np.isfinite(avail) and avail > 0 else left)
                fills.append((ns, px, q))
                left -= q
        else:
            p = book.t_px[k]
            through = p < px if side == 1 else p > px
            at = abs(p - px) < 1e-9
            if through or (at and rng.random() < qprob):
                sz = book.t_sz[k]
                q = min(left, sz if np.isfinite(sz) and sz > 0 else left)
                fills.append((ns, px, q))
                left -= q
    return fills, left


def cross_now(
    book: Book,
    side: int,
    decision_ns: int,
    lag_s: float = 2.0,
    pi: PriceImprovement | None = None,
    symbol: str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, float] | None:
    """直接吃对手价：+lag 后第一个有效报价的 ask（买）/ bid（卖）；之后再无报价则用当时有效的
    最后一笔报价，成交时刻记为 +lag 那一刻。"""
    t = decision_ns + int(lag_s * 1e9)
    i = book.first_at(t)
    ts = None
    if i is not None:
        ts = int(book.q_ns[i])
    else:
        i = book.asof(t)
        if i is None:
            return None
        ts = t
    if pi is not None and symbol is not None:
        r = rng if rng is not None else np.random.default_rng(0)
        return ts, pi.price(side, float(book.bid[i]), float(book.ask[i]), "market", symbol, r)
    return ts, float(book.ask[i] if side == 1 else book.bid[i])


def execute(
    book: Book,
    side: int,
    qty: float,
    decision_ns: int,
    rule: LimitRule,
    symbol: str,
    deadline_ns: int | None = None,
) -> Result | None:
    """从决策时刻起按规则挂单。deadline_ns：最晚必须结束的时刻（如强平），阶梯在此之前截断。"""
    rng = np.random.default_rng(rule.seed + (decision_ns % 1_000_003))
    t0 = decision_ns + int(rule.lag_s * 1e9)
    i0 = book.asof(t0)
    if i0 is None:
        i0 = book.first_at(t0)
    if i0 is None:
        return None
    px = limit_price(book, i0, side, rule.price, symbol)
    T = int(rule.T * 1e9)
    end1 = t0 + T if deadline_ns is None else min(t0 + T, deadline_ns)
    fills, left = _rest(book, side, px, t0, end1, qty, rng, rule.queue_fill_prob)
    stage = "limit"
    end_last = end1
    if left > 0 and rule.ladder in ("reprice", "reprice_abandon"):
        i1 = book.asof(end1)
        tick = _tick(symbol, book.mid(i1))
        px2 = _round_passive(book.mid(i1), tick, side) + side * tick
        px2 = min(px2, book.ask[i1]) if side == 1 else max(px2, book.bid[i1])
        end2 = end1 + T if deadline_ns is None else min(end1 + T, deadline_ns)
        f2, left = _rest(book, side, px2, end1, end2, left, rng, rule.queue_fill_prob)
        end_last = end2
        fills += f2
        stage = "reprice" if f2 else stage
        if left > 0 and rule.ladder == "reprice":
            c = cross_now(book, side, end2, lag_s=0.0)
            if c is not None:
                fills.append((c[0], c[1], left))
                left = 0
                stage = "cross"
                end_last = c[0]
    missed = left if rule.ladder in ("abandon", "reprice_abandon") else 0.0
    if not fills:
        return Result(0.0, float("nan"), None, None, missed, "missed", px, end_last)
    q = sum(f[2] for f in fills)
    avg = sum(f[1] * f[2] for f in fills) / q
    if not left:
        end_last = fills[-1][0]
    return Result(q, avg, fills[0][0], fills[-1][0], missed, stage, px, end_last)
