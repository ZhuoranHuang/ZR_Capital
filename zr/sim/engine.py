"""带账户层的期权回测引擎 v1（任务书：回测引擎收敛 v1）。

单标的期权报价 + 信号 → 逐笔交易与当日权益。语义见任务书「语义（写死）」：
- 仓位：size_position（premium / stop 两种口径，取整向下，0 就是 0）；
- 报价：有效 = bid > 0、ask > 0、ask >= bid、相对价差 <= max_spread_pct；无效报价视为不存在；
- 覆盖率：signals 中每个标的，first_entry 至 flat（含两端）的美东分钟里至少有 1 行报价的分钟占比；
  任一标的 < min_coverage 或 quotes 为空 → ValueError；
- 入场：信号 + lag_s 之后（>=）第一条有效报价的 ask，且不晚于信号 + lag_s + max_entry_wait_s
  （否则 late_quote）；出场：time / bracket（mid 判断，触发后 > 的第一条有效报价的 bid）；
  清仓：这些出场的有效报价落在 flat 之后或不存在时，flat 之后（>=）120 秒内第一条可清仓报价
  （bid >= 0 且 ask >= bid，不按价差过滤）按 bid 成交；窗口内无报价 → ValueError；
  DayResult.counters 计 flat_late（成交晚于 flat）、flat_wide（价差 > max_spread_pct）、
  flat_zero（bid == 0）；
- 信号过滤顺序：outside_session → daily_loss → max_entries → in_position → cooldown → no_quote
  → late_quote → size_zero；daily_loss = 已实现亏损 >= max_daily_loss_pct × 期初权益；max_entries =
  已成交开仓笔数 >= max_entries_per_day（None = 关闭）；
- 出场：time / bracket / at（指定时刻，可带保护性 sl），延迟按类型（见 _exit）；
- 费用：zr.costs.fees.option_order_fees，买卖两侧分别计算后相加；权益按笔累加，
  下一笔按新权益定张数。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from zr.costs.fees import option_order_fees

ET = ZoneInfo("America/New_York")
NS = 1_000_000_000
# 信号过滤（按此顺序判定，每个信号至多计一次）
SKIP_KEYS = (
    "outside_session",
    "daily_loss",
    "max_entries",
    "in_position",
    "cooldown",
    "no_quote",
    "late_quote",
    "size_zero",
)
# 清仓计数（每笔走清仓规则的交易计入，不参与信号过滤）
COUNTER_KEYS = ("flat_late", "flat_wide", "flat_zero")
FLAT_WINDOW_NS = 120 * NS  # 清仓：flat 之后（>=）120 秒内取第一条可清仓报价


@dataclass(frozen=True)
class Account:
    equity_usd: float
    max_risk_pct: float = 0.05  # 单笔风险预算 / 权益（D-003）
    max_premium_pct: float = 0.15  # 单笔权利金 / 权益 上限（D-011）
    max_contracts: int = 10
    multiplier: int = 100
    sizing_basis: str = "stop"  # "stop"（D-011：按止损距离）| "premium"（全额权利金）


@dataclass(frozen=True)
class SimPolicy:
    cooldown_s: float = 60.0
    min_coverage: float = 0.95
    max_spread_pct: float = 0.15
    lag_s: float = 2.0  # 信号到下单的延迟
    max_entry_wait_s: float = 5.0  # 入场报价最晚 = 信号 + lag + 该值，超时计 late_quote
    max_daily_loss_pct: float | None = (
        0.10  # 已实现亏损 >= 该比例 × 期初权益后不再开仓；None = 关闭
    )
    max_entries_per_day: int | None = 20  # 已成交开仓笔数上限；None = 关闭


@dataclass(frozen=True)
class Trade:
    symbol: str
    direction: int  # +1 call / -1 put
    entry_ts_ns: int
    entry_px: float  # 买入价 = 当时 ask
    qty: int  # >= 1
    exit_ts_ns: int
    exit_px: float  # 卖出价 = 当时 bid
    reason: str  # "tp" | "sl" | "time" | "flat" | "signal"
    fees_usd: float  # 买卖两侧合计，来自 zr.costs.fees.option_order_fees
    pnl_usd: float  # 已扣费


@dataclass(frozen=True)
class DayResult:
    trades: list
    equity_start: float
    equity_end: float
    skipped: dict  # 键 = SKIP_KEYS（信号过滤）；sum(skipped) + len(trades) == len(signals)
    counters: dict  # 键 = COUNTER_KEYS（清仓计数）


def size_position(account: Account, premium_px: float, stop_frac: float | None) -> int:
    per_contract = premium_px * account.multiplier
    if not (per_contract > 0):
        return 0
    if stop_frac is not None and stop_frac <= 0:
        return 0
    eq = account.equity_usd
    if account.sizing_basis == "stop" and stop_frac is not None:
        by_risk = _floor(eq * account.max_risk_pct / (per_contract * stop_frac))
    else:  # premium 口径；或 stop 口径但没有止损距离（time 类出场）→ 退回 premium 口径
        by_risk = _floor(eq * account.max_risk_pct / per_contract)
    by_premium = _floor(eq * account.max_premium_pct / per_contract)
    return int(max(0, min(by_risk, by_premium, account.max_contracts)))


def _floor(x: float) -> int:
    # 1e-9 容差：避免 3.0 的浮点表示 2.9999999999 被向下取成 2
    return int(math.floor(x + 1e-9))


def _ns_at(day: dt.date, hhmm: str) -> int:
    return int(pd.Timestamp(dt.datetime.combine(day, dt.time.fromisoformat(hhmm)), tz=ET).value)


def _coverage(q: pd.DataFrame, symbols: set, day: dt.date, session: dict, policy: SimPolicy):
    fe, fl = _ns_at(day, session["first_entry"]), _ns_at(day, session["flat"])
    denom = (fl - fe) // (60 * NS) + 1
    for sym in sorted(symbols):
        x = q[(q["symbol"] == sym) & q["bid"].notna() & q["ask"].notna()]
        minute = (x["ts_ns"].to_numpy(np.int64) - fe) // (60 * NS)
        n = len(np.unique(minute[(minute >= 0) & (minute < denom)]))
        cov = n / denom if denom > 0 else 0.0
        if cov < policy.min_coverage:
            raise ValueError(f"{sym} 报价覆盖率 {cov:.3f} < {policy.min_coverage}")


class _Book:
    """单标的的报价（按时间升序）。默认 = 有效报价（bid > 0、ask > 0、ask >= bid、
    价差 <= max_spread）；clearable=True = 可清仓报价（bid >= 0 且 ask >= bid，含 0 / 0），
    只供 flat 使用。"""

    def __init__(self, q: pd.DataFrame, max_spread: float, clearable: bool = False):
        bid, ask = q["bid"].to_numpy(float), q["ask"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            mid = (ask + bid) / 2
            if clearable:
                ok = (bid >= 0) & (ask >= bid)
            else:
                ok = (bid > 0) & (ask > 0) & (ask >= bid) & ((ask - bid) / mid <= max_spread)
        self.ts = q["ts_ns"].to_numpy(np.int64)[ok]
        self.bid, self.ask, self.mid = bid[ok], ask[ok], mid[ok]

    def first_ge(self, t: int) -> int | None:
        i = int(np.searchsorted(self.ts, t, side="left"))
        return i if i < len(self.ts) else None

    def first_gt(self, t: int) -> int | None:
        i = int(np.searchsorted(self.ts, t, side="right"))
        return i if i < len(self.ts) else None


def _exit(book: _Book, i_entry: int, entry_px: float, rule: dict, flat_ns: int, lag_ns: int):
    """非清仓出场：返回 (strict 报价下标, 原因)；成交时刻 >= flat 或没有报价返回 None（走清仓）。
    成交报价：bracket = first_gt(触发 + lag)；time = first_ge(到期 + lag)；
    at = first_ge(at + lag)（at <= 入场时为 first_gt(入场 + lag)），保护性 sl 只在触发时刻 < at 的
    报价上判定、first_gt(触发 + lag) 成交；at 与 sl 取成交更早者，同时则按 sl。"""
    t_entry = int(book.ts[i_entry])
    kind = rule["kind"]
    cand: tuple[int, str] | None = None
    if kind == "at":
        at = int(rule["ts_ns"])
        x_at = book.first_gt(t_entry + lag_ns) if at <= t_entry else book.first_ge(at + lag_ns)
        opts = [(x_at, "signal")] if x_at is not None else []
        sl = rule.get("sl")
        if sl is not None:
            j0 = i_entry + 1
            hit = np.flatnonzero((book.mid[j0:] <= entry_px * (1 - sl)) & (book.ts[j0:] < at))
            if len(hit):
                x = book.first_gt(int(book.ts[j0 + int(hit[0])]) + lag_ns)
                if x is not None:
                    opts.append((x, "sl"))
        if opts:
            cand = min(opts, key=lambda c: (int(book.ts[c[0]]), 0 if c[1] == "sl" else 1))
    else:
        hold_min = rule["minutes"] if kind == "time" else rule.get("max_hold_min")
        t_time = t_entry + int(hold_min * 60 * NS) if hold_min is not None else None
        if kind == "bracket":
            j0 = i_entry + 1
            mid = book.mid[j0:]
            hit = np.flatnonzero(
                (mid >= entry_px * (1 + rule["tp"])) | (mid <= entry_px * (1 - rule["sl"]))
            )
            if len(hit):
                k = j0 + int(hit[0])
                if t_time is None or book.ts[k] < t_time:
                    x = book.first_gt(int(book.ts[k]) + lag_ns)
                    if x is not None:
                        cand = (x, "tp" if book.mid[k] >= entry_px * (1 + rule["tp"]) else "sl")
        if cand is None and t_time is not None:
            x = book.first_ge(t_time + lag_ns)
            if x is not None:
                cand = (x, "time")
    if cand is None or book.ts[cand[0]] >= flat_ns:
        return None
    return cand


def _flat(raw_ts: np.ndarray, loose: _Book, flat_ns: int, symbol: str) -> int:
    """清仓：flat 之后（>=）120 秒内第一条可清仓报价（loose 下标）。窗口内没有任何报价行或没有
    可清仓报价 → ValueError。"""
    j = loose.first_ge(flat_ns)
    if j is not None and loose.ts[j] <= flat_ns + FLAT_WINDOW_NS:
        return j
    any_row = bool(((raw_ts >= flat_ns) & (raw_ts <= flat_ns + FLAT_WINDOW_NS)).any())
    raise ValueError(
        f"{symbol} 清仓：flat 后 120 秒内"
        + (
            "有报价行但没有可清仓报价（ask < bid 或缺值）"
            if any_row
            else "没有任何报价行（数据缺失）"
        )
    )


def simulate_day(
    quotes,
    signals,
    account: Account,
    session: dict,
    policy: SimPolicy = SimPolicy(),  # noqa: B008 - 任务书规定的签名；SimPolicy 为 frozen
) -> DayResult:
    if quotes is None or len(quotes) == 0:
        raise ValueError("quotes 为空")
    q = quotes.sort_values("ts_ns", kind="stable")
    day = pd.Timestamp(int(q["ts_ns"].iloc[0]), tz="UTC").tz_convert(ET).date()
    _coverage(q, {s[2] for s in signals}, day, session, policy)
    fe = _ns_at(day, session["first_entry"])
    le = _ns_at(day, session["last_entry"])
    flat_ns = _ns_at(day, session["flat"])
    books = {sym: _Book(g, policy.max_spread_pct) for sym, g in q.groupby("symbol")}  # strict
    loose = {sym: _Book(g, float("inf"), clearable=True) for sym, g in q.groupby("symbol")}
    raw_ts = {sym: g["ts_ns"].to_numpy(np.int64) for sym, g in q.groupby("symbol")}
    skipped = dict.fromkeys(SKIP_KEYS, 0)
    counters = dict.fromkeys(COUNTER_KEYS, 0)
    trades: list[Trade] = []
    equity0 = float(account.equity_usd)
    equity = equity0
    last_exit: int | None = None
    for ts_ns, direction, symbol, rule in signals:
        ts_ns = int(ts_ns)
        if ts_ns < fe or ts_ns > le:
            skipped["outside_session"] += 1
            continue
        if (
            policy.max_daily_loss_pct is not None
            and equity0 - equity >= policy.max_daily_loss_pct * equity0
        ):
            skipped["daily_loss"] += 1
            continue
        if policy.max_entries_per_day is not None and len(trades) >= policy.max_entries_per_day:
            skipped["max_entries"] += 1
            continue
        if last_exit is not None and ts_ns < last_exit:
            skipped["in_position"] += 1
            continue
        if last_exit is not None and ts_ns < last_exit + int(policy.cooldown_s * NS):
            skipped["cooldown"] += 1
            continue
        book = books.get(symbol)
        i = book.first_ge(ts_ns + int(policy.lag_s * NS)) if book is not None else None
        if i is None or book.ts[i] > le:
            skipped["no_quote"] += 1
            continue
        if book.ts[i] > ts_ns + int(policy.lag_s * NS) + int(policy.max_entry_wait_s * NS):
            skipped["late_quote"] += 1
            continue
        entry_px = float(book.ask[i])
        stop = (
            float(rule["sl"])
            if rule.get("kind") in ("bracket", "at") and rule.get("sl") is not None
            else None
        )
        qty = size_position(dataclasses.replace(account, equity_usd=equity), entry_px, stop)
        if qty == 0:
            skipped["size_zero"] += 1
            continue
        ex = _exit(book, i, entry_px, rule, flat_ns, int(policy.lag_s * NS))
        if ex is not None:
            x, reason = ex
            exit_ts, exit_px = int(book.ts[x]), float(book.bid[x])
        else:
            lb = loose[symbol]
            j = _flat(raw_ts[symbol], lb, flat_ns, symbol)
            exit_ts, exit_px, reason = int(lb.ts[j]), float(lb.bid[j]), "flat"
            if exit_ts > flat_ns:
                counters["flat_late"] += 1
            mid = (lb.ask[j] + lb.bid[j]) / 2
            if mid > 0 and (lb.ask[j] - lb.bid[j]) / mid > policy.max_spread_pct:
                counters["flat_wide"] += 1
            if exit_px == 0:
                counters["flat_zero"] += 1
        fees = float(option_order_fees(qty, "Buy", entry_px)["total"]) + float(
            option_order_fees(qty, "Sell", exit_px)["total"]
        )
        pnl = (exit_px - entry_px) * account.multiplier * qty - fees
        trades.append(
            Trade(
                symbol=symbol,
                direction=int(direction),
                entry_ts_ns=int(book.ts[i]),
                entry_px=entry_px,
                qty=qty,
                exit_ts_ns=exit_ts,
                exit_px=exit_px,
                reason=reason,
                fees_usd=fees,
                pnl_usd=pnl,
            )
        )
        equity += pnl
        last_exit = exit_ts
    return DayResult(
        trades=trades,
        equity_start=equity0,
        equity_end=equity0 + sum(t.pnl_usd for t in trades),
        skipped=skipped,
        counters=counters,
    )
