"""纸面护栏助手的核心逻辑（事件驱动、不做 I/O）：实时状态层、护栏、风控、系统接管离场。

实时运行（zr/live/runner.py）与录制数据回放（同文件 replay）走同一个 Assistant：所有事件都带
时刻（UTC 纳秒），本模块不读系统时钟、不连网、不下单。离场全部是纸面单（zr/live/paper.py）。

状态层 = zr/strategy/incremental.py（与批量回测逐根一致，BASE_FINAL_AUDIT 第 5 节）；
另算 MA5 / MA10 供显示。护栏的证据见 docs/research/GUARDRAIL_EVIDENCE.md。
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from zr.backtest.fill_limit import LimitRule
from zr.costs.fees import option_order_fees
from zr.data.thetadata import parse_contract
from zr.features.bars import ET
from zr.features.regime import STATE_NAMES
from zr.live.paper import PaperSell, valid
from zr.strategy.incremental import NS_MIN, Bar, BarBuilder, IncrementalEngine

LEVELS = ("info", "notice", "warn", "block")


def et_of(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, tz="UTC").tz_convert(ET)


def hm(s: str) -> dt.time:
    h, m = (int(x) for x in s.split(":"))
    return dt.time(h, m)


def price_ok(trade_type: str) -> bool:
    """只有常规成交计开高低收（与长桥官方 K 一致，BASE_FINAL_AUDIT 第 5 节）。"""
    return trade_type == ""


def buy_fee(n: int, px: float) -> float:
    return float(option_order_fees(n, "Buy", px)["total"]) if n > 0 else 0.0


def sell_fee(n: int, px: float) -> float:
    return float(option_order_fees(n, "Sell", px)["total"]) if n > 0 and px > 0 else 0.0


@dataclass
class Alert:
    ns: int
    level: str
    code: str
    msg: str
    symbol: str = ""


@dataclass
class Entry:
    ns: int
    symbol: str
    qty: int
    price: float
    source: str = "account"
    order_id: str = ""


# ---------------- 标的实时状态 ----------------


class UnderlyingFeed:
    """一个标的：逐笔 → 2 分钟 K → 增量状态层。先 seed 历史 K，再从 start_bin 起接逐笔。"""

    def __init__(self, und: str, engine: IncrementalEngine, period_min: int, rth_only: bool):
        self.und = und
        self.engine = engine
        self.builder = BarBuilder(period_min, price_ok)
        self.period_ns = period_min * NS_MIN
        self.rth_only = rth_only
        self.closes: deque = deque(maxlen=10)
        self.last: dict | None = None
        self.start_bin: int | None = None
        self.first_trade_ns: int | None = None
        self.buffer: list = []

    @property
    def live(self) -> bool:
        return self.start_bin is not None

    def _rth(self, ts: int) -> bool:
        t = et_of(ts).time()
        return dt.time(9, 30) <= t < dt.time(16, 0)

    def _push(self, b: Bar, done_ns: int) -> dict | None:
        if self.rth_only and not self._rth(b.ts):
            return None
        rec = self.engine.on_bar(b)
        self.closes.append(b.close)
        c = list(self.closes)
        rec["ma5"] = sum(c[-5:]) / 5 if len(c) >= 5 else math.nan
        rec["ma10"] = sum(c) / 10 if len(c) >= 10 else math.nan
        rec["completed_ns"] = done_ns
        self.last = rec
        return rec

    def seed(self, bars: list[Bar], start_bin: int) -> None:
        for b in bars:
            if b.ts < start_bin:
                self._push(b, b.ts + self.period_ns)
                self.builder.last_close = b.close
        self.start_bin = start_bin
        self.builder.cur = start_bin  # 从 start_bin 起逐箱收线（空箱补占位 K）
        pending, self.buffer = self.buffer, []
        for recv, trades in pending:
            self.on_trades(recv, trades)

    def on_trades(self, recv: int, trades: list) -> list[dict]:
        """trades = [(成交时刻 ns, 价, 量, 类型)]。未 seed 前先缓存。"""
        if self.first_trade_ns is None and trades:
            self.first_trade_ns = trades[0][0]
        if not self.live:
            self.buffer.append((recv, trades))
            return []
        out = []
        for t, px, vol, tt in trades:
            if t < self.start_bin:
                continue
            for b in self.builder.on_trade(t, px, vol, tt):
                r = self._push(b, recv)
                if r is not None:
                    out.append(r)
        return out

    def on_clock(self, now: int) -> list[dict]:
        if not self.live:
            return []
        out = []
        for b in self.builder.on_clock(now):
            r = self._push(b, now)
            if r is not None:
                out.append(r)
        return out


# ---------------- 持仓与离场 ----------------


@dataclass
class Position:
    pid: int
    symbol: str
    und: str
    direction: int  # call +1（看多正股）/ put −1
    expiry: dt.date
    qty: int
    entry_ns: int
    entry_px: float
    source: str
    bid0: float = math.nan
    ask0: float = math.nan
    state: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    tp_done: bool = False
    macd_done: bool = False
    orders: list = field(default_factory=list)
    owner_sells: list = field(default_factory=list)  # [(ns, px, qty)]
    closed_ns: int | None = None

    @property
    def dte(self) -> int:
        return (self.expiry - et_of(self.entry_ns).date()).days

    @property
    def sold(self) -> int:
        return sum(o.qty - o.left for o in self.orders)

    @property
    def committed(self) -> int:
        return sum(o.qty for o in self.orders)

    @property
    def free(self) -> int:
        return self.qty - self.committed

    @property
    def open_qty(self) -> int:
        return self.qty - self.sold

    def realized(self) -> float:
        """已成交部分的盈亏（含买入费按成交比例分摊、卖出费）。"""
        cash = 0.0
        for o in self.orders:
            for _ns, px, q, _st in o.fills:
                cash += px * q * 100 - sell_fee(q, px)
        cost_share = self.sold / self.qty if self.qty else 0.0
        return (
            cash - (self.entry_px * self.qty * 100 + buy_fee(self.qty, self.entry_px)) * cost_share
        )

    def unrealized(self, bid: float | None) -> float:
        if bid is None or not math.isfinite(bid) or self.open_qty <= 0:
            return 0.0
        share = self.open_qty / self.qty
        cost = (self.entry_px * self.qty * 100 + buy_fee(self.qty, self.entry_px)) * share
        return bid * self.open_qty * 100 - sell_fee(self.open_qty, bid) - cost


class Assistant:
    def __init__(
        self,
        cfg: dict,
        state_cfg: dict,
        v4: dict,
        day: dt.date,
        notify: Callable[[str], None] = lambda m: None,
        log: Callable[[dict], None] = lambda r: None,
        week_realized_before: float = 0.0,
    ):
        self.cfg, self.day = cfg, day
        self.notify, self.log = notify, log
        conv = cfg["convention"]
        if conv not in ("A", "C"):
            raise ValueError(f"convention 只能是 A / C：{conv}")
        self.rth_only = conv == "A"
        range_w = (
            195 if self.rth_only else 480
        )  # 口径 A RTH 195 根 / 日，口径 C 480 根 / 日（D-031）
        self.feeds = {
            u: UnderlyingFeed(
                u,
                IncrementalEngine(state_cfg, v4, range_w, {day}, daily_reset=self.rth_only),
                cfg["period_min"],
                self.rth_only,
            )
            for u in cfg["underlyings"]
        }
        self.quotes: dict[str, tuple] = {}
        self.positions: dict[int, Position] = {}
        self.alerts: list[Alert] = []
        self.equity = float(cfg["paper_equity_usd"])
        self.week_before = week_realized_before
        self.rule = LimitRule(price=cfg["fill"]["price"], T=float(cfg["fill"]["T"]), lag_s=0.0)
        self.killed = False
        self.quotes_at_flat: dict[str, tuple] | None = (
            None  # 15:50 那一刻的盘口（「若无接管」估算）
        )
        self._pid = 0
        ex = cfg["exits"]
        self.flat_t, self.zdte_flat_t = hm(ex["flat"]), hm(ex["zero_dte_flat"])
        self.first_bar_t = hm(cfg["session"]["first_bar_close"])
        self.zdte_after = hm(cfg["guardrails"]["zero_dte_late"]["after"])

    # ---------- 事件 ----------

    def alert(self, a: Alert, push: bool = True) -> None:
        self.alerts.append(a)
        self.log({"type": "alert", **a.__dict__})
        if push:
            self.notify(f"[{a.level.upper()}] {a.code} {a.symbol}：{a.msg}")

    def on_tap(self, row: dict) -> None:
        k, s, r = row["k"], row["s"], int(row["r"])
        if k == "t" and s in self.feeds:
            from zr.live.tap import ts_ns

            trades = [(ts_ns(t), float(p), float(v), tt or "") for t, p, v, tt in row["x"]]
            for rec in self.feeds[s].on_trades(r, trades):
                self.on_bar(s, rec)
        elif k == "d":
            self.on_quote(r, s, row.get("b"), row.get("a"), row.get("bs"))
        elif k == "ot":
            for _t, p, v in row["x"]:
                for pos in self._open(s):
                    for o in pos.orders:
                        o.on_trade(r, float(p), float(v))

    def on_quote(self, ns: int, sym: str, bid, ask, bid_size=None) -> None:
        if not valid(bid, ask):
            return
        bid, ask = float(bid), float(ask)
        self.quotes[sym] = (ns, bid, ask, bid_size)
        for pos in self._open(sym):
            for o in pos.orders:
                o.on_quote(ns, bid, ask, bid_size)
            self._premium_rules(pos, ns, bid, ask)
            self._close_if_done(pos, ns)

    def on_bar(self, und: str, rec: dict) -> None:
        prev = getattr(self, "_last_state", {}).get(und)
        cur = (rec["state"], rec["dir"])
        if prev is not None and prev != cur:
            self.alert(
                Alert(
                    rec["completed_ns"],
                    "info",
                    "状态变化",
                    f"{STATE_NAMES[prev[0]]}{prev[1]:+d} → {STATE_NAMES[cur[0]]}{cur[1]:+d}",
                    und,
                ),
                push=cur[0] != prev[0],
            )
        self._last_state = {**getattr(self, "_last_state", {}), und: cur}
        self.log({"type": "bar", "und": und, **{k: v for k, v in rec.items() if k != "ex"}})
        for pos in [p for p in self.positions.values() if p.und == und and p.closed_ns is None]:
            side = 0 if pos.direction == 1 else 1
            if rec["ex"][self.cfg["exits"]["structure"]][side]:
                self._sell(
                    pos, pos.free, "结构止损（跌破前一更高低点）", "stop", rec["completed_ns"]
                )
            elif not pos.macd_done and rec["ex"]["X2_shrink2"][side]:
                pos.macd_done = True
                self._sell(
                    pos, pos.free // 2, "MACD 柱连续 2 根缩短：减半", "limit", rec["completed_ns"]
                )

    def on_clock(self, now: int) -> None:
        for f in self.feeds.values():
            for rec in f.on_clock(now):
                self.on_bar(f.und, rec)
        t = et_of(now).time()
        if self.quotes_at_flat is None and t >= self.flat_t:
            self.quotes_at_flat = dict(self.quotes)
        cap = int(self.cfg["exits"]["cap_min"] * 60e9)
        for pos in [p for p in self.positions.values() if p.closed_ns is None]:
            if t >= self.flat_t:
                self._sell(pos, pos.free, "15:50 全部清仓", "stop", now)
            elif pos.dte == 0 and t >= self.zdte_flat_t:
                self._sell(pos, pos.free, "0DTE 15:30 前清仓", "limit", now)
            elif now >= pos.entry_ns + cap:
                self._sell(pos, pos.free, "持仓 30 分钟上限", "limit", now)
            q = self.quotes.get(pos.symbol)
            if q is not None:  # 改价 / 吃价靠盘口推动；盘口静止时用时钟推进
                for o in pos.orders:
                    o.on_quote(now, q[1], q[2], q[3])
                self._close_if_done(pos, now)

    def on_entry(self, e: Entry) -> Position | None:
        try:
            c = parse_contract(e.symbol)
        except ValueError:
            self.alert(Alert(e.ns, "info", "非期权成交", "忽略", e.symbol), push=False)
            return None
        und = f"{c['und']}.US"
        self._pid += 1
        pos = Position(
            self._pid,
            e.symbol,
            und,
            1 if c["right"] == "call" else -1,
            c["expiry"],
            int(e.qty),
            e.ns,
            float(e.price),
            e.source,
        )
        q = self.quotes.get(e.symbol)
        if q is not None:
            pos.bid0, pos.ask0 = q[1], q[2]
        feed = self.feeds.get(und)
        rec = feed.last if feed else None
        if rec is not None:
            pos.state = {
                k: rec.get(k)
                for k in (
                    "ts",
                    "state",
                    "dir",
                    "close",
                    "mid",
                    "upper",
                    "lower",
                    "pctb",
                    "bw",
                    "dif",
                    "dea",
                    "hist",
                    "ma5",
                    "ma10",
                    "range_score",
                )
            }
        self.positions[pos.pid] = pos
        rel = (
            (pos.entry_px - pos.ask0) / (pos.ask0 - pos.bid0)
            if math.isfinite(pos.ask0) and pos.ask0 > pos.bid0
            else math.nan
        )
        self.log(
            {
                "type": "entry",
                "pid": pos.pid,
                "symbol": pos.symbol,
                "qty": pos.qty,
                "price": pos.entry_px,
                "ns": e.ns,
                "source": e.source,
                "order_id": e.order_id,
                "bid": pos.bid0,
                "ask": pos.ask0,
                "vs_ask_spreads": rel,
                "state": pos.state,
            }
        )
        self.notify(
            f"[入场] {pos.symbol} ×{pos.qty} @ {pos.entry_px:.2f}（卖一 {pos.ask0:.2f}，"
            f"状态 {STATE_NAMES.get(pos.state.get('state'), '未知')}{pos.state.get('dir', 0):+d}）"
        )
        self._guardrails(pos, rec, e.ns)
        return pos

    def on_owner_sell(self, ns: int, sym: str, qty: int, px: float) -> None:
        for pos in self._open(sym) or [p for p in self.positions.values() if p.symbol == sym]:
            pos.owner_sells.append((ns, float(px), int(qty)))
            self.log({"type": "owner_sell", "pid": pos.pid, "symbol": sym, "qty": qty, "px": px})
            return

    def restore(self, rows: list[dict]) -> int:
        """重启恢复：按当日 trades_*.jsonl 的 entry / exit_fill 重建持仓
        （不重发通知、不重跑护栏）。"""
        n = 0
        for r in rows:
            if r.get("type") == "entry":
                c = parse_contract(r["symbol"])
                pos = Position(
                    int(r["pid"]),
                    r["symbol"],
                    f"{c['und']}.US",
                    1 if c["right"] == "call" else -1,
                    c["expiry"],
                    int(r["qty"]),
                    int(r["ns"]),
                    float(r["price"]),
                    r.get("source", "account"),
                    float(r.get("bid") or math.nan),
                    float(r.get("ask") or math.nan),
                    r.get("state") or {},
                )
                self.positions[pos.pid] = pos
                self._pid = max(self._pid, pos.pid)
                n += 1
            elif r.get("type") == "exit_fill" and int(r["pid"]) in self.positions:
                pos = self.positions[int(r["pid"])]
                o = PaperSell(
                    pos.symbol, int(r["qty"]), r["reason"], r["kind"], int(r["ns"]), self.rule
                )
                o.fills.append((int(r["ns"]), float(r["avg_px"]), int(r["qty"]), "restored"))
                o.stage, o._reported = "done", True
                pos.orders.append(o)
                pos.tp_done |= "+20%" in r["reason"]
                pos.macd_done |= "MACD" in r["reason"]
                if pos.open_qty <= 0:
                    pos.closed_ns = int(r["ns"])
        return n

    def kill(self, now: int) -> None:
        """kill switch：全部纸面持仓立即按买一平仓。"""
        self.killed = True
        for pos in [p for p in self.positions.values() if p.closed_ns is None]:
            q = self.quotes.get(pos.symbol)
            self._sell(pos, pos.free, "kill switch", "stop", now)
            if q is not None:
                for o in pos.orders:
                    o.force_cross(now, q[1])
                self._close_if_done(pos, now)
        self.alert(Alert(now, "block", "kill switch", "全部纸面持仓已平仓，助手退出"))

    # ---------- 内部 ----------

    def _open(self, sym: str) -> list[Position]:
        return [p for p in self.positions.values() if p.symbol == sym and p.closed_ns is None]

    def _sell(self, pos: Position, n: int, reason: str, kind: str, now: int) -> None:
        if n <= 0:
            return
        o = PaperSell(pos.symbol, int(n), reason, kind, now, self.rule)
        pos.orders.append(o)
        self.log({"type": "exit_order", "pid": pos.pid, "qty": n, "reason": reason, "kind": kind})
        q = self.quotes.get(pos.symbol)
        if q is not None and q[0] <= now:
            o.on_quote(now, q[1], q[2], q[3])
            self._close_if_done(pos, now)

    def _premium_rules(self, pos: Position, ns: int, bid: float, ask: float) -> None:
        mid = (bid + ask) / 2
        ex = self.cfg["exits"]
        if mid <= pos.entry_px * (1 - ex["sl"]):
            self._sell(pos, pos.free, "权利金 −30% 全出", "stop", ns)
        elif not pos.tp_done and mid >= pos.entry_px * (1 + ex["tp"]):
            pos.tp_done = True
            self._sell(pos, pos.free // 2, "权利金 +20% 减半", "limit", ns)

    def _close_if_done(self, pos: Position, ns: int) -> None:
        for o in pos.orders:
            if o.done and not getattr(o, "_reported", False):
                o._reported = True
                self.log(
                    {
                        "type": "exit_fill",
                        "pid": pos.pid,
                        "symbol": pos.symbol,
                        "qty": o.qty,
                        "avg_px": o.avg_px,
                        "reason": o.reason,
                        "kind": o.kind,
                        "stage": o.fills[-1][3] if o.fills else "",
                        "bid_at_order": o.bid_at_create,
                        "ns": o.fills[-1][0] if o.fills else ns,
                    }
                )
                self.notify(
                    f"[系统离场] {pos.symbol} 卖 {o.qty} @ {o.avg_px:.2f}（{o.reason}）"
                    "——纸面，请自行在账户对应操作"
                )
        if pos.open_qty <= 0 and pos.closed_ns is None:
            pos.closed_ns = ns
            self.log({"type": "closed", "pid": pos.pid, "realized": pos.realized()})

    def day_pnl(self) -> float:
        tot = 0.0
        for p in self.positions.values():
            q = self.quotes.get(p.symbol)
            tot += p.realized() + p.unrealized(q[1] if q else None)
        return tot

    def _guardrails(self, pos: Position, rec: dict | None, ns: int) -> None:
        g, rk = self.cfg["guardrails"], self.cfg["risk"]
        t = et_of(ns).time()
        if t < self.first_bar_t:
            self.alert(
                Alert(ns, g["before_first_bar"], "首根 K 收盘前开仓", "09:32 前", pos.symbol)
            )
        if rec is None:
            self.alert(Alert(ns, "notice", "状态未就绪", "该标的状态层尚无 K 线", pos.symbol))
        else:
            if STATE_NAMES[rec["state"]] == "IGNITION" and rec["dir"] == -pos.direction:
                self.alert(
                    Alert(
                        ns,
                        g["counter_ignition"],
                        "逆点火开仓",
                        "IGNITION 方向与持仓相反",
                        pos.symbol,
                    )
                )
            if STATE_NAMES[rec["state"]] == "RANGE":
                self.alert(Alert(ns, g["range"], "RANGE 开仓", "当前为 RANGE 状态", pos.symbol))
        if pos.dte == 0 and t > self.zdte_after:
            half = pos.free // 2 if g["zero_dte_late"]["action"] == "notice_half" else 0
            msg = f"提示 + 仓位减半（纸面卖出 {half} 张）" if half else "提示（1 张无法减半）"
            self.alert(Alert(ns, "notice", "0DTE 10:15 后开仓", msg, pos.symbol))
            self._sell(pos, half, "0DTE 10:15 后：仓位减半", "limit", ns)
        cost = pos.entry_px * pos.qty * 100
        if cost > self.equity * rk["per_trade_pct"] / 100:
            self.alert(
                Alert(
                    ns,
                    "warn",
                    "超单笔上限",
                    f"开仓成本 {cost:.0f} > 权益 {rk['per_trade_pct']}%（{self.equity:.0f}）",
                    pos.symbol,
                )
            )
        day_loss = -(self.day_pnl() - pos.unrealized(pos.bid0 if math.isfinite(pos.bid0) else None))
        week_loss = day_loss - self.week_before
        if (
            day_loss >= self.equity * rk["day_loss_pct"] / 100
            or week_loss >= self.equity * rk["week_loss_pct"] / 100
            or self.killed
        ):
            self.alert(
                Alert(
                    ns,
                    "block",
                    "禁止开仓",
                    f"日亏 {day_loss:.0f} / 周亏 {week_loss:.0f} 已到线",
                    pos.symbol,
                )
            )
            self._sell(pos, pos.free, "风控禁止开仓：立即平仓", "stop", ns)

    # ---------- 输出 ----------

    def snapshot(self, now: int) -> dict:
        und = {}
        for u, f in self.feeds.items():
            r = f.last or {}
            und[u] = {
                "live": f.live,
                "bar": et_of(r["ts"]).strftime("%H:%M") if r else None,
                "state": STATE_NAMES.get(r.get("state")) if r else None,
                "dir": r.get("dir") if r else None,
                **{
                    k: r.get(k)
                    for k in (
                        "close",
                        "mid",
                        "upper",
                        "lower",
                        "pctb",
                        "dif",
                        "dea",
                        "hist",
                        "ma5",
                        "ma10",
                    )
                },
            }
        pos = []
        for p in self.positions.values():
            q = self.quotes.get(p.symbol)
            pos.append(
                {
                    "pid": p.pid,
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "open": p.open_qty,
                    "entry_px": p.entry_px,
                    "entry_et": et_of(p.entry_ns).strftime("%H:%M:%S"),
                    "bid": q[1] if q else None,
                    "ask": q[2] if q else None,
                    "realized": round(p.realized(), 2),
                    "unrealized": round(p.unrealized(q[1] if q else None), 2),
                    "closed": p.closed_ns is not None,
                }
            )
        return {
            "now_et": et_of(now).strftime("%Y-%m-%d %H:%M:%S"),
            "convention": self.cfg["convention"],
            "underlyings": und,
            "positions": pos,
            "day_pnl": round(self.day_pnl(), 2),
            "equity": self.equity,
            "alerts": [a.__dict__ for a in self.alerts[-10:]],
            "killed": self.killed,
        }
