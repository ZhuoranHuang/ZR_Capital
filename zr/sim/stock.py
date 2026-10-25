"""正股模式回测（任务书：规则回测最小流水线）。

单标的当日 RTH 1 分钟 K + 入场信号 → 逐笔交易与当日权益。复用 zr.sim.engine 的
Account / Trade / DayResult（权益按笔累加，下一笔按新权益定股数），不另建账户。
- 信号 = (入场 K 下标, 方向 ±1)，在该 K 的开盘价成交：open + 方向 × slippage_ticks × tick。
  「信号 K 收盘确认 → 下一根 1 分钟 K 开盘入场」的映射由 runner（scripts/run_rule.py）
  完成，引擎不管周期。
- 出场的唯一实现是 plan_exit（simulate_day 与 runner 的期权模式都调用它）：
  time = 入场 K 之后第 n 根的开盘（时刻 = 该根开盘）；
  bracket = 从入场 K 的下一根起逐根判，多头 high 判止盈、low 判止损（空头镜像），同根同时
  触发按止损；成交价 = 触发价，开盘已越过触发价则按开盘；到 max_bars 按该根收盘
  （时刻 = 该根收盘）；
  到 flat 仍持仓 = flat 那根的开盘，该根不存在则用 flat 之前最后一根的收盘。
  所有出场价含滑点（多头卖出 − 滑点，空头买回 + 滑点）。
- 股数：size_shares（有止损按风险预算，无止损按名义上限；不超过全部权益 / 入场价，
  不加杠杆）。
- 费用：zr.costs.fees.stock_order_fees，开仓、平仓两侧各按其方向（多头 Buy 入 / Sell 出，
  空头 Sell 入 / Buy 出）计算后相加。允许做空；**不建模借券费**。
- 信号过滤（按此顺序，每个信号至多计一次）：no_bar（下标越界或入场 K 不早于 flat）
  → outside_session → in_position → cooldown → max_entries → size_zero；
  sum(skipped) + len(trades) == len(signals)。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from zr.costs.fees import stock_order_fees
from zr.sim.engine import ET, NS, Account, DayResult, Trade, _floor, _ns_at

SKIP_KEYS = ("outside_session", "no_bar", "in_position", "cooldown", "max_entries", "size_zero")
MIN_NS = 60 * NS


@dataclass(frozen=True)
class StockPolicy:
    slippage_ticks: float = 1.0
    tick: float = 0.01
    cooldown_bars: int = 0  # 平仓后至少隔多少根 1 分钟 K 才能再开仓
    max_entries_per_day: int | None = None


def size_shares(
    account: Account,
    entry_px: float,
    stop_px: float | None,
    max_notional_pct: float = 0.5,
    max_shares: int = 100_000,
) -> int:
    """有止损：floor(权益 × max_risk_pct / |入场 − 止损|)；
    无止损：floor(权益 × max_notional_pct / 入场)；
    再取 min(…, floor(权益 / 入场), max_shares)，不加杠杆。"""
    eq = float(account.equity_usd)
    if not (entry_px > 0) or not (eq > 0):
        return 0
    if stop_px is not None:
        dist = abs(entry_px - stop_px)
        if not (dist > 0):
            return 0
        n = _floor(eq * account.max_risk_pct / dist)
    else:
        n = _floor(eq * max_notional_pct / entry_px)
    return int(max(0, min(n, _floor(eq / entry_px), max_shares)))


def _arrays(bars: pd.DataFrame):
    ts = bars["ts_ns"].to_numpy(np.int64)
    if len(ts) > 1 and not (np.diff(ts) > 0).all():
        raise ValueError("bars 必须按 ts_ns 严格升序")
    return ts, *(bars[c].to_numpy(float) for c in ("open", "high", "low", "close"))


def _day(ts_ns: int):
    return pd.Timestamp(int(ts_ns), tz="UTC").tz_convert(ET).date()


def _levels(rule: dict, entry_px: float, d: int) -> tuple[float | None, float | None]:
    """bracket 的 (止盈价, 止损价)；pct 相对入场成交价（含滑点）。"""
    tp = rule.get("tp_px")
    if tp is None and rule.get("tp_pct") is not None:
        tp = entry_px * (1 + d * rule["tp_pct"])
    sl = rule.get("sl_px")
    if sl is None and rule.get("sl_pct") is not None:
        sl = entry_px * (1 - d * rule["sl_pct"])
    return tp, sl


def plan_exit(
    bars: pd.DataFrame,
    entry_index: int,
    rule: dict,
    session: dict,
    direction: int = 1,
    policy: StockPolicy = StockPolicy(),  # noqa: B008 - frozen
) -> tuple[int, float, int, str] | None:
    """出场的唯一实现。返回 (出场 K 下标, 出场价（含滑点）, 出场时刻 ns, 原因)；
    入场 K 越界或不早于 flat → None。bars 为单日 1 分钟 K（ts_ns = 开盘时刻）。"""
    ts, o, h, lo, c = _arrays(bars)
    n, i, d = len(ts), int(entry_index), int(direction)
    if not 0 <= i < n:
        return None
    flat_ns = _ns_at(_day(ts[i]), session["flat"])
    if ts[i] >= flat_ns:
        return None
    slip = policy.slippage_ticks * policy.tick
    entry_px = o[i] + d * slip

    def out(px: float) -> float:
        return px - d * slip

    def flat_exit() -> tuple[int, float, int, str]:
        j = int(np.searchsorted(ts, flat_ns, side="left"))
        if j < n and ts[j] == flat_ns:
            return j, out(o[j]), int(ts[j]), "flat"
        k = j - 1  # flat 那根不存在：flat 之前最后一根的收盘
        return k, out(c[k]), int(ts[k]) + MIN_NS, "flat"

    kind = rule["kind"]
    if kind == "signal":
        raise NotImplementedError("signal 出场尚未实现")
    if kind == "time":
        j = i + int(rule["bars"])
        if j < n and ts[j] < flat_ns:
            return j, out(o[j]), int(ts[j]), "time"
        return flat_exit()
    if kind != "bracket":
        raise ValueError(f"未知出场类型 {kind}")
    tp, sl = _levels(rule, entry_px, d)
    max_bars = rule.get("max_bars")
    for j in range(i + 1, n):
        if ts[j] >= flat_ns:
            break
        hi, low = (h[j], lo[j]) if d == 1 else (-lo[j], -h[j])  # 空头取负后按多头判
        op = o[j] if d == 1 else -o[j]
        if sl is not None and low <= d * sl:
            px = o[j] if op <= d * sl else sl
            return j, out(px), int(ts[j]) + MIN_NS, "sl"
        if tp is not None and hi >= d * tp:
            px = o[j] if op >= d * tp else tp
            return j, out(px), int(ts[j]) + MIN_NS, "tp"
        if max_bars is not None and j >= i + int(max_bars):
            return j, out(c[j]), int(ts[j]) + MIN_NS, "time"
    return flat_exit()


def run_day(
    bars: pd.DataFrame,
    signals,
    account: Account,
    session: dict,
    policy: StockPolicy = StockPolicy(),  # noqa: B008 - frozen
    *,
    exit_rule: dict,
    symbol: str = "",
) -> tuple[DayResult, list]:
    """simulate_day 的实现，另返回逐信号结果：("trade", Trade) 或 ("skip", 键)，
    供 runner 写明细。"""
    skipped = dict.fromkeys(SKIP_KEYS, 0)
    trades: list[Trade] = []
    outcomes: list = []
    equity0 = float(account.equity_usd)
    equity = equity0
    n_flat = 0
    if len(bars) == 0:
        skipped["no_bar"] = len(signals)
        outcomes = [("skip", "no_bar")] * len(signals)
        return DayResult([], equity0, equity0, skipped, {"flat": 0}), outcomes
    ts, o, _, _, _ = _arrays(bars)
    day = _day(ts[0])
    fe, le, fl = (_ns_at(day, session[k]) for k in ("first_entry", "last_entry", "flat"))
    slip = policy.slippage_ticks * policy.tick
    # 上一笔平仓后第一根可开仓的 K（平仓在开盘 = 该根；在收盘 = 下一根）
    free_idx: int | None = None

    def skip(key: str) -> None:
        skipped[key] += 1
        outcomes.append(("skip", key))

    for idx, d in signals:
        idx, d = int(idx), int(d)
        if not 0 <= idx < len(ts) or ts[idx] >= fl:
            skip("no_bar")
            continue
        if ts[idx] < fe or ts[idx] > le:
            skip("outside_session")
            continue
        if free_idx is not None and idx < free_idx:
            skip("in_position")
            continue
        if free_idx is not None and idx < free_idx + policy.cooldown_bars:
            skip("cooldown")
            continue
        if policy.max_entries_per_day is not None and len(trades) >= policy.max_entries_per_day:
            skip("max_entries")
            continue
        entry_px = float(o[idx] + d * slip)
        stop = _levels(exit_rule, entry_px, d)[1] if exit_rule["kind"] == "bracket" else None
        qty = size_shares(dataclasses.replace(account, equity_usd=equity), entry_px, stop)
        if qty == 0:
            skip("size_zero")
            continue
        j, exit_px, exit_ts, reason = plan_exit(bars, idx, exit_rule, session, d, policy)
        open_side, close_side = ("Buy", "Sell") if d == 1 else ("Sell", "Buy")
        fees = float(stock_order_fees(qty, open_side, entry_px)["total"]) + float(
            stock_order_fees(qty, close_side, exit_px)["total"]
        )
        pnl = d * (exit_px - entry_px) * qty - fees
        t = Trade(
            symbol=symbol,
            direction=d,
            entry_ts_ns=int(ts[idx]),
            entry_px=entry_px,
            qty=qty,
            exit_ts_ns=int(exit_ts),
            exit_px=float(exit_px),
            reason=reason,
            fees_usd=fees,
            pnl_usd=pnl,
        )
        trades.append(t)
        outcomes.append(("trade", t))
        n_flat += reason == "flat"
        equity += pnl
        free_idx = j if exit_ts == ts[j] else j + 1
    # 做空不建模借券费（DayResult.trades 的空头盈亏未扣借券成本）
    return (
        DayResult(
            trades=trades,
            equity_start=equity0,
            equity_end=equity0 + sum(t.pnl_usd for t in trades),
            skipped=skipped,
            counters={"flat": n_flat},
        ),
        outcomes,
    )


def simulate_day(
    bars: pd.DataFrame,
    signals,
    account: Account,
    session: dict,
    policy: StockPolicy = StockPolicy(),  # noqa: B008 - frozen
    *,
    exit_rule: dict,
    symbol: str = "",
) -> DayResult:
    """bars：DataFrame[ts_ns, open, high, low, close]，1 分钟、单标的、升序、当日 RTH；
    signals：[(入场 K 下标, 方向 ±1)]；exit_rule：plan_exit 的出场规则。做空不建模借券费。"""
    return run_day(bars, signals, account, session, policy, exit_rule=exit_rule, symbol=symbol)[0]
