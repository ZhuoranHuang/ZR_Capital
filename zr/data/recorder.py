"""Task 1.2：分钟级期权行情录制器。

订阅 4 正股 + 近端期权链的 Quote / Depth / Trade 推送，原始事件落 Parquet：
  ZR_DATA_ROOT/recorder/date=YYYY-MM-DD/underlying=XXX/<kind>_<ts>.parquet
QA 报告：ZR_DATA_ROOT/recorder_qa/YYYY-MM-DD.md

运行：uv run python -m zr.data.recorder [--dry-run] [--now]
时段：启动后用 trading_days 判断交易日（非交易日退出 0）；9:25 ET 前等待；16:05 ET 写 QA 后退出 0。
时间计算只用 ZoneInfo("America/New_York")。
--dry-run 不连接 API，用假期权链打印订阅集构造结果。--now 跳过交易日判断与等待。

纯逻辑（订阅集构造、缓冲落盘）与 I/O（Recorder）分离，前者有单元测试且不触网。

SDK 已核实（4.3.7 内省）：subscribe(symbols, sub_types)、unsubscribe(symbols, sub_types)、
set_on_quote / set_on_depth / set_on_trades(callback(symbol, event))；
PushQuote / PushDepth{asks,bids} / PushTrades{trades[]: Trade{direction,...}}。
行情限制 ≤500 订阅、10 次/秒（来自任务书，未在 SDK 内验证）。重连由 SDK 内部处理，
本模块只能通过回调中断统计近似记录（未验证 SDK 是否暴露重连事件）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from zr.broker.serialize import to_json
from zr.data.universe import ChainRule, load_universe

log = logging.getLogger("zr.recorder")
ET = ZoneInfo("America/New_York")

OPTION_RE = re.compile(r"^(?P<underlying>[A-Z.\-]+?)\d{6}[CP]\d+\.US$")
KINDS = ("quote", "depth", "trade")


# ---------- 订阅集构造（纯函数） ----------


@dataclass(frozen=True)
class StrikeInfo:
    strike: float
    call: str
    put: str


# 规则（ChainRule）来自 config/universe.yaml，见 docs/UNIVERSE_SPEC.md。
# SPY/QQQ「今日与次日到期」与 MU/AMD「最近 2 个到期」都等价于
# 「≥今日的最近 n_expiries 个到期日」，差别只在档数。
Chain = dict[dt.date, list[StrikeInfo]]


def underlying_of(symbol: str) -> str:
    m = OPTION_RE.match(symbol)
    if m:
        return m.group("underlying")
    return symbol.split(".")[0]


def select_option_symbols(
    chain: Chain, spot: float, today: dt.date, rule: ChainRule
) -> list[tuple[float, str]]:
    """返回 [(相对距离, 期权代码)]，覆盖最近 n 个到期日、ATM ±N 档、call+put。"""
    expiries = sorted(d for d in chain if d >= today)[: rule.n_expiries]
    out: list[tuple[float, str]] = []
    for exp in expiries:
        strikes = sorted(chain[exp], key=lambda s: s.strike)
        if not strikes:
            continue
        atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i].strike - spot))
        lo = max(0, atm_idx - rule.strikes_each_side)
        hi = min(len(strikes), atm_idx + rule.strikes_each_side + 1)
        for s in strikes[lo:hi]:
            dist = abs(s.strike - spot) / spot
            out.append((dist, s.call))
            out.append((dist, s.put))
    return out


def build_subscription_set(
    chains: dict[str, Chain],
    spots: dict[str, float],
    today: dt.date,
    rules: dict[str, ChainRule],
    cap: int,
) -> list[str]:
    """正股永远保留；期权按距 ATM 由远及近裁剪到 cap。"""
    underlyings = list(rules)
    candidates: list[tuple[float, str]] = []
    for u in underlyings:
        if u not in chains or u not in spots:
            continue
        candidates.extend(select_option_symbols(chains[u], spots[u], today, rules[u]))
    # 去重（同一代码可能被多个到期日规则重复选中——理论上不会，防御）
    best: dict[str, float] = {}
    for dist, sym in candidates:
        best[sym] = min(dist, best.get(sym, dist))
    room = max(0, cap - len(underlyings))
    ranked = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:room]
    return underlyings + sorted(sym for sym, _ in ranked)


def diff_subscriptions(current: set[str], desired: set[str]) -> tuple[list[str], list[str]]:
    return sorted(desired - current), sorted(current - desired)


# ---------- 缓冲与落盘 ----------

SCHEMA = pa.schema(
    [
        ("recv_ts_utc_ns", pa.int64()),
        ("push_ts", pa.int64()),
        ("symbol", pa.string()),
        ("kind", pa.string()),
        ("payload", pa.string()),
    ]
)


def make_event(symbol: str, kind: str, event: Any, recv_ts_ns: int | None = None) -> dict:
    """把推送对象转成落盘行。push_ts 取事件 timestamp（epoch 秒，整数）；没有则 0。"""
    recv = recv_ts_ns if recv_ts_ns is not None else time.time_ns()
    payload = to_json(event)
    ts = getattr(event, "timestamp", None)
    if ts is None and kind == "trade":
        trades = getattr(event, "trades", None) or []
        ts = getattr(trades[0], "timestamp", None) if trades else None
    push_ts = int(ts.timestamp()) if isinstance(ts, dt.datetime) else 0
    return {
        "recv_ts_utc_ns": recv,
        "push_ts": push_ts,
        "symbol": symbol,
        "kind": kind,
        "payload": payload,
    }


class ParquetSink:
    """线程安全缓冲；每 flush_interval_s 秒或 max_buffer 条落盘。

    去重键 (symbol, kind, push_ts, payload)，窗口 dedup_window 条。
    """

    def __init__(
        self,
        root: Path,
        flush_interval_s: float = 30.0,
        max_buffer: int = 5000,
        dedup_window: int = 200_000,
    ) -> None:
        self.root = Path(root)
        self.flush_interval_s = flush_interval_s
        self.max_buffer = max_buffer
        self._buf: list[dict] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._seen: OrderedDict[tuple, None] = OrderedDict()
        self._dedup_window = dedup_window
        self.files_written = 0
        self.rows_written = 0
        self.rows_dropped_dup = 0

    def add(self, row: dict) -> None:
        key = (row["symbol"], row["kind"], row["push_ts"], row["payload"])
        with self._lock:
            if key in self._seen:
                self.rows_dropped_dup += 1
                return
            self._seen[key] = None
            if len(self._seen) > self._dedup_window:
                self._seen.popitem(last=False)
            self._buf.append(row)
            need_flush = len(self._buf) >= self.max_buffer
        if need_flush:
            self.flush()

    def maybe_flush(self) -> None:
        if time.monotonic() - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> list[Path]:
        with self._lock:
            rows, self._buf = self._buf, []
            self._last_flush = time.monotonic()
        if not rows:
            return []
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for r in rows:
            date = dt.datetime.fromtimestamp(r["recv_ts_utc_ns"] / 1e9, tz=dt.UTC).astimezone(ET)
            groups[(date.strftime("%Y-%m-%d"), underlying_of(r["symbol"]), r["kind"])].append(r)
        written: list[Path] = []
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        for (date, und, kind), grp in groups.items():
            d = self.root / f"date={date}" / f"underlying={und}"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{kind}_{stamp}.parquet"
            table = pa.Table.from_pylist(grp, schema=SCHEMA)
            pq.write_table(table, path)
            written.append(path)
            self.rows_written += len(grp)
        self.files_written += len(written)
        return written


# ---------- 统计与 QA ----------


@dataclass
class SymbolStats:
    count: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    last_recv_ns: dict[str, int] = field(default_factory=dict)
    max_gap_s: dict[str, float] = field(default_factory=lambda: dict.fromkeys(KINDS, 0.0))
    gaps_over_60s: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))

    def record(self, kind: str, recv_ns: int) -> None:
        self.count[kind] += 1
        last = self.last_recv_ns.get(kind)
        if last is not None:
            gap = (recv_ns - last) / 1e9
            self.max_gap_s[kind] = max(self.max_gap_s[kind], gap)
            if gap > 60:
                self.gaps_over_60s[kind] += 1
        self.last_recv_ns[kind] = recv_ns


def write_qa_report(
    path: Path,
    date: dt.date,
    stats: dict[str, SymbolStats],
    reconnects: int,
    callback_errors: int,
    sink: ParquetSink,
    forced_resubscribes: int = 0,
) -> None:
    lines = [
        f"# 录制器 QA {date.isoformat()}",
        "",
        f"- 重连次数（近似，见模块说明）：{reconnects}",
        f"- 回调异常次数：{callback_errors}",
        f"- 静默强制重订阅次数：{forced_resubscribes}",
        f"- 落盘文件：{sink.files_written}，行：{sink.rows_written}，"
        f"去重丢弃：{sink.rows_dropped_dup}",
        "",
        "| symbol | quote | depth | trade | 最大间隔 s (q/d/t) | >60s 缺口 (q/d/t) |",
        "|---|---|---|---|---|---|",
    ]
    for sym in sorted(stats):
        s = stats[sym]
        gaps = "/".join(f"{s.max_gap_s[k]:.0f}" for k in KINDS)
        holes = "/".join(str(s.gaps_over_60s[k]) for k in KINDS)
        lines.append(
            f"| {sym} | {s.count['quote']} | {s.count['depth']} | {s.count['trade']} "
            f"| {gaps} | {holes} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- 交易日与时段（只用 America/New_York） ----------

SUBSCRIBE_AT = dt.time(9, 25)
SESSION_END = dt.time(16, 5)


def session_action(
    now_et: dt.datetime,
    is_trading_day: bool,
    start: dt.time = SUBSCRIBE_AT,
    end: dt.time = SESSION_END,
) -> tuple[str, float]:
    """返回 (动作, 等待秒数)。动作 ∈ skip / wait / run / done。"""
    if not is_trading_day:
        return "skip", 0.0
    t = now_et.timetz().replace(tzinfo=None)
    if t >= end:
        return "done", 0.0
    if t < start:
        target = now_et.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
        return "wait", (target - now_et).total_seconds()
    return "run", 0.0


def is_trading_day(qctx: Any, day: dt.date) -> bool:
    """交易日判断：优先用入库的交易所日历（config/us_equity_calendar.json，zr/data/calendar.py）；
    日历未覆盖该年份时才退回长桥 trading_days（2026-10-22 实测长桥对历史休市日不可靠）。"""
    from zr.data.calendar import covered_years
    from zr.data.calendar import is_trading_day as cal_is_trading_day

    if day.year in covered_years():
        return cal_is_trading_day(day)
    from longport.openapi import Market

    res = qctx.trading_days(Market.US, day, day)
    days = set(res.trading_days) | set(res.half_trading_days)
    return day in days


# ---------- 主循环 ----------


class Recorder:
    def __init__(
        self,
        qctx: Any,
        sink: ParquetSink,
        rules: dict[str, ChainRule],
        cap: int,
        resub_interval_s: float = 300.0,
    ) -> None:
        self.qctx = qctx
        self.sink = sink
        self.underlyings = list(rules)
        self.rules = rules
        self.cap = cap
        self.resub_interval_s = resub_interval_s
        self.current: set[str] = set()
        self.stats: dict[str, SymbolStats] = defaultdict(SymbolStats)
        self.reconnects = 0
        self.callback_errors = 0
        self.forced_resubscribes = 0
        self.silence_s = 120.0
        self._last_forced: float | None = None
        self._lock = threading.Lock()
        self._last_any_recv: float | None = None

    # 回调：任何异常只计数，不抛出
    def _handle(self, kind: str, symbol: str, event: Any) -> None:
        try:
            recv = time.time_ns()
            self.sink.add(make_event(symbol, kind, event, recv))
            with self._lock:
                self.stats[symbol].record(kind, recv)
                now = time.monotonic()
                # 全局静默超过 120s 后再次收到推送，近似记为一次重连（未验证 SDK 重连语义）
                if self._last_any_recv is not None and now - self._last_any_recv > 120:
                    self.reconnects += 1
                self._last_any_recv = now
        except Exception:  # noqa: BLE001
            self.callback_errors += 1
            log.exception("回调异常 kind=%s symbol=%s", kind, symbol)

    def install_callbacks(self) -> None:
        self.qctx.set_on_quote(lambda s, e: self._handle("quote", s, e))
        self.qctx.set_on_depth(lambda s, e: self._handle("depth", s, e))
        self.qctx.set_on_trades(lambda s, e: self._handle("trade", s, e))

    def fetch_chains(self, today: dt.date) -> tuple[dict[str, Chain], dict[str, float]]:
        chains: dict[str, Chain] = {}
        spots: dict[str, float] = {}
        quotes = self.qctx.quote(self.underlyings)
        for q in quotes:
            spots[q.symbol] = float(q.last_done)
        for u in self.underlyings:
            chain: Chain = {}
            expiries = sorted(d for d in self.qctx.option_chain_expiry_date_list(u) if d >= today)
            for exp in expiries[: self.rules[u].n_expiries]:
                infos = self.qctx.option_chain_info_by_date(u, exp)
                chain[exp] = [
                    StrikeInfo(float(i.price), i.call_symbol, i.put_symbol) for i in infos
                ]
            chains[u] = chain
        return chains, spots

    def resubscribe(self, today: dt.date) -> None:
        from longport.openapi import SubType

        chains, spots = self.fetch_chains(today)
        desired = set(build_subscription_set(chains, spots, today, self.rules, self.cap))
        add, remove = diff_subscriptions(self.current, desired)
        sub_types = [SubType.Quote, SubType.Depth, SubType.Trade]
        if remove:
            self.qctx.unsubscribe(remove, sub_types)
        if add:
            self.qctx.subscribe(add, sub_types)
        self.current = desired
        log.info("订阅集 %d（+%d −%d）", len(desired), len(add), len(remove))

    def run(
        self,
        until_et: dt.time = SESSION_END,
        qa_dir: Path | None = None,
        now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(ET),
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.install_callbacks()
        today = now_fn().date()
        last_resub: float | None = None
        while True:
            now_et = now_fn()
            if now_et.time() >= until_et:
                break
            # 静默看门狗：盘中超过 silence_s 没有任何推送，清空本地订阅状态强制全量重订
            with self._lock:
                last_recv = self._last_any_recv
            silent = last_recv is not None and time.monotonic() - last_recv > self.silence_s
            if silent and (
                self._last_forced is None or time.monotonic() - self._last_forced > self.silence_s
            ):
                log.warning("超过 %.0fs 无推送，强制全量重订阅", self.silence_s)
                self.forced_resubscribes += 1
                self._last_forced = time.monotonic()
                self.current = set()
                last_resub = None
            if last_resub is None or time.monotonic() - last_resub >= self.resub_interval_s:
                try:
                    self.resubscribe(today)
                except Exception:  # noqa: BLE001
                    log.exception("重算订阅集失败")
                last_resub = time.monotonic()
            self.sink.maybe_flush()
            sleep_fn(1.0)
        self.sink.flush()
        if qa_dir is not None:
            write_qa_report(
                qa_dir / f"{today.isoformat()}.md",
                today,
                dict(self.stats),
                self.reconnects,
                self.callback_errors,
                self.sink,
                self.forced_resubscribes,
            )


# ---------- CLI ----------


def _fake_chains(
    today: dt.date, underlyings: list[str]
) -> tuple[dict[str, Chain], dict[str, float]]:
    """--dry-run 用：合成期权链，不代表真实代码格式。"""
    spots = {"SPY.US": 500.0, "QQQ.US": 430.0, "MU.US": 120.0, "AMD.US": 160.0}
    chains: dict[str, Chain] = {}
    for u in underlyings:
        spot = spots.get(u, 100.0)
        base = u.split(".")[0]
        step = 1.0 if spot > 200 else 0.5
        chain: Chain = {}
        for k in range(3):
            exp = today + dt.timedelta(days=k)
            tag = exp.strftime("%y%m%d")
            chain[exp] = [
                StrikeInfo(
                    s,
                    f"{base}{tag}C{int(s * 1000)}.US",
                    f"{base}{tag}P{int(s * 1000)}.US",
                )
                for s in (spot + step * i for i in range(-40, 41))
            ]
        chains[u] = chain
    return chains, spots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分钟级期权行情录制器")
    parser.add_argument("--dry-run", action="store_true", help="不连接 API，打印订阅集")
    parser.add_argument("--until", default="16:05", help="美东停止时间 HH:MM")
    parser.add_argument("--now", action="store_true", help="跳过交易日判断与 9:25 等待，立即订阅")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from zr.config import get_settings

    settings = get_settings()
    uni = load_universe()
    underlyings = uni.symbols
    rules = uni.chain_rules()
    today = dt.datetime.now(ET).date()

    if args.dry_run:
        chains, spots = _fake_chains(today, underlyings)
        subs = build_subscription_set(chains, spots, today, rules, uni.subscription_cap)
        print(f"订阅集 {len(subs)} 个（上限 {uni.subscription_cap}）")
        by_und: dict[str, int] = defaultdict(int)
        for s in subs:
            by_und[underlying_of(s)] += 1
        print(json.dumps(by_und, ensure_ascii=False))
        print("\n".join(subs[:12]) + "\n...")
        return 0

    from zr.broker.longport_client import make_quote_ctx

    hh, mm = (int(x) for x in args.until.split(":"))
    until = dt.time(hh, mm)
    qctx = make_quote_ctx()

    if not args.now:
        try:
            trading = is_trading_day(qctx, today)
        except Exception:  # noqa: BLE001
            # 日历查询失败时按交易日处理（宁可空跑，不可漏录）
            log.exception("trading_days 查询失败，按交易日处理")
            trading = True
        action, wait_s = session_action(dt.datetime.now(ET), trading, end=until)
        log.info("今日 %s 交易日=%s 动作=%s 等待=%.0fs", today, trading, action, wait_s)
        if action in ("skip", "done"):
            return 0
        if action == "wait":
            time.sleep(wait_s)

    sink = ParquetSink(settings.zr_data_root / "recorder")
    rec = Recorder(qctx, sink, rules, uni.subscription_cap)
    rec.run(until_et=until, qa_dir=settings.zr_data_root / "recorder_qa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
