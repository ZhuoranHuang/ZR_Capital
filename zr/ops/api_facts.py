"""Task 1.1：只读核验长桥 API 行为，结果写入 docs/API_FACTS.md。

每个检查项独立 try/except；异常连错误码一起记录，绝不伪造结果。
本模块不包含任何下单、撤单、改单调用。

运行：uv run python -m zr.ops.api_facts [--out docs/API_FACTS.md]
需要 .env 中的 LONGPORT_* 密钥；本脚本不读取或打印密钥本身。

方法签名来源：对已安装 longport 4.3.7 的 __text_signature__ 内省（2026-10-21）。
行为（返回内容、是否接受期权代码、行情包限制）在实际运行前均为「未验证」。
"""

from __future__ import annotations

import argparse
import datetime as dt
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from zr.broker.serialize import to_json, to_py
from zr.config import get_settings

ET = ZoneInfo("America/New_York")
PAGE_SLEEP_S = 0.15
MAX_PAGES = 30
SUBSCRIBE_SECONDS = 20


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.started = dt.datetime.now(dt.UTC)

    def section(self, title: str) -> None:
        self.lines.append(f"\n## {title}\n")

    def ok(self, name: str, value: Any, note: str = "") -> None:
        body = to_json(value)
        if len(body) > 4000:
            body = body[:4000] + " …(截断)"
        self.lines.append(f"### {name}\n\n{note}\n\n```json\n{body}\n```\n")

    def fail(self, name: str, exc: BaseException) -> None:
        code = getattr(exc, "code", None)
        self.lines.append(
            f"### {name}\n\n**失败** `{type(exc).__name__}` code={code!r}\n\n"
            f"```\n{exc}\n```\n"
        )

    def write(self, path: Path) -> None:
        header = (
            "# 长桥 API 核验结果\n\n"
            f"生成时间（UTC）：{self.started.isoformat(timespec='seconds')}  \n"
            "生成脚本：`zr/ops/api_facts.py`，不手改。\n"
            "每项为实际调用返回或实际异常；未出现在本文件的 API 行为视为「未验证」。\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n".join(self.lines), encoding="utf-8")


def _run(report: Report, name: str, fn: Callable[[], Any], note: str = "") -> Any | None:
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - 核验脚本必须记录任何异常
        report.fail(name, exc)
        return None
    report.ok(name, value, note)
    return value


# ---------- 行情 ----------


def _earliest_minute_bar(qctx: Any, symbol: str, period: Any, adjust: Any, sessions: Any):
    """向前翻页找最早可回溯的 1 分钟 bar；最多 MAX_PAGES 页。"""
    cursor: dt.datetime | None = None
    earliest = None
    pages = 0
    total = 0
    while pages < MAX_PAGES:
        # forward=False 表示向过去取（未验证：参数语义以运行结果为准）
        bars = qctx.history_candlesticks_by_offset(
            symbol, period, adjust, False, 1000, cursor, sessions
        )
        pages += 1
        if not bars:
            break
        total += len(bars)
        first = bars[0]
        if earliest is None or first.timestamp < earliest.timestamp:
            earliest = first
        if cursor is not None and first.timestamp >= cursor:
            break
        cursor = first.timestamp - dt.timedelta(minutes=1)
        time.sleep(PAGE_SLEEP_S)
    return {
        "pages": pages,
        "bars_seen": total,
        "earliest": to_py(earliest, datetime_as_str=True),
        "cursor_at_stop": cursor.isoformat() if cursor else None,
    }


def _nearest_atm_call(qctx: Any, symbol: str) -> dict[str, Any]:
    # 已观测（2026-10-21）：expiry_date_list 含已过期日期，必须过滤 >= 今日
    today = dt.datetime.now(dt.UTC).astimezone(ET).date()
    expiries = [d for d in qctx.option_chain_expiry_date_list(symbol) if d >= today]
    expiry = sorted(expiries)[0]
    chain = qctx.option_chain_info_by_date(symbol, expiry)
    spot = float(qctx.quote([symbol])[0].last_done)
    atm = min(chain, key=lambda s: abs(float(s.price) - spot))
    return {
        "expiry": expiry.isoformat(),
        "spot": spot,
        "strike": float(atm.price),
        "call_symbol": atm.call_symbol,
        "put_symbol": atm.put_symbol,
    }


def _subscribe_probe(qctx: Any, symbols: list[str]) -> dict[str, Any]:
    from longport.openapi import SubType

    counts = {"quote": 0, "depth": 0, "trade": 0}
    depth_sample: dict[str, Any] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def on_quote(sym: str, ev: Any) -> None:
        with lock:
            counts["quote"] += 1

    def on_depth(sym: str, ev: Any) -> None:
        with lock:
            counts["depth"] += 1
            if sym not in depth_sample:
                depth_sample[sym] = to_py(ev, datetime_as_str=True)

    def on_trades(sym: str, ev: Any) -> None:
        with lock:
            counts["trade"] += 1

    qctx.set_on_quote(on_quote)
    qctx.set_on_depth(on_depth)
    qctx.set_on_trades(on_trades)
    sub_types = [SubType.Quote, SubType.Depth, SubType.Trade]
    try:
        qctx.subscribe(symbols, sub_types)
        time.sleep(SUBSCRIBE_SECONDS)
    finally:
        try:
            qctx.unsubscribe(symbols, sub_types)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"unsubscribe: {exc!r}")
    return {
        "symbols": symbols,
        "seconds": SUBSCRIBE_SECONDS,
        "counts": counts,
        "depth_sample": depth_sample,
        "errors": errors,
    }


def run_quote_checks(
    report: Report, qctx: Any, symbols: list[str], skip_bars: bool = False
) -> dict[str, Any] | None:
    from longport.openapi import AdjustType, Market, Period, TradeSessions

    report.section("行情包与交易日")
    _run(report, "quote_level", qctx.quote_level)  # 未验证：返回结构
    _run(report, "quote_package_details", qctx.quote_package_details)  # 未验证
    _run(report, "trading_session", qctx.trading_session)  # 未验证
    today = dt.date.today()
    _run(
        report,
        "trading_days(US, 近 20 天)",
        lambda: qctx.trading_days(Market.US, today - dt.timedelta(days=20), today),
    )

    report.section("正股 quote / depth")
    _run(report, "quote(正股)", lambda: qctx.quote(symbols))
    for s in symbols:
        _run(report, f"depth({s})", lambda s=s: qctx.depth(s))

    report.section("1 分钟 bar 可回溯起点")
    if skip_bars:
        report.lines.append("（--skip-bars，跳过翻页；结果见上一次运行）\n")
    for s in [] if skip_bars else symbols:
        _run(
            report,
            f"earliest_minute_bar({s})",
            lambda s=s: _earliest_minute_bar(
                qctx, s, Period.Min_1, AdjustType.NoAdjust, TradeSessions.All
            ),
            note=f"翻页 ≤{MAX_PAGES} 页，每页 1000 根，间隔 {PAGE_SLEEP_S}s。",
        )

    report.section("期权链到期日")
    for s in symbols:

        def expiries(s=s):
            dates = sorted(qctx.option_chain_expiry_date_list(s))[:6]
            return [{"date": d.isoformat(), "weekday": d.strftime("%A")} for d in dates]

        _run(report, f"option_chain_expiry_date_list({s}) 最近 6 个", expiries,
             note="核对 MU/AMD 是否为周一 / 周三到期。")

    report.section("SPY 最近到期 ATM call")
    atm = _run(report, "nearest_atm_call(SPY.US)", lambda: _nearest_atm_call(qctx, "SPY.US"))
    if atm:
        c = atm["call_symbol"]
        _run(report, f"option_quote({c})", lambda: qctx.option_quote([c]),
             note="OptionQuote 无 bid/ask（已核实）。")
        _run(report, f"depth({c})", lambda: qctx.depth(c),
             note="关键测试：期权代码是否返回盘口；每档 price/volume/order_num/position。")
        _run(
            report,
            f"candlesticks({c}, Min_1, 100)",
            lambda: qctx.candlesticks(c, Period.Min_1, 100, AdjustType.NoAdjust),
        )
        _run(
            report,
            f"history_candlesticks_by_offset({c}, Min_1, 1000)",
            lambda: qctx.history_candlesticks_by_offset(
                c, Period.Min_1, AdjustType.NoAdjust, False, 1000, None, TradeSessions.All
            ),
        )
        report.section("订阅推送计数")
        _run(
            report,
            "subscribe Quote+Depth+Trade 20s",
            lambda: _subscribe_probe(qctx, ["SPY.US", c]),
            note="按 symbol 计数；保存首条 Depth 样例。",
        )
    return atm


# ---------- 交易（只读） ----------


def run_trade_checks(report: Report, tctx: Any, atm: dict[str, Any] | None) -> None:
    from longport.openapi import OrderSide, OrderType

    report.section("交易账户（只读）")
    _run(report, "account_balance", tctx.account_balance)
    _run(report, "stock_positions", tctx.stock_positions)
    _run(report, "today_orders", tctx.today_orders)
    now = dt.datetime.now(dt.UTC)
    _run(
        report,
        "history_executions(近 30 天)",
        lambda: tctx.history_executions(None, now - dt.timedelta(days=30), now),
    )
    if atm:
        c = atm["call_symbol"]
        # 未验证：估算接口对期权代码的支持；price 取 0.01 为极小值仅作可行性探测
        _run(
            report,
            f"estimate_max_purchase_quantity({c}, LO, Buy, 0.01)",
            lambda: tctx.estimate_max_purchase_quantity(c, OrderType.LO, OrderSide.Buy, 0.01),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="长桥 API 只读核验")
    parser.add_argument("--out", default="docs/API_FACTS.md")
    parser.add_argument("--skip-trade", action="store_true", help="不创建 TradeContext")
    parser.add_argument("--skip-bars", action="store_true", help="跳过 1 分钟 bar 翻页")
    args = parser.parse_args(argv)

    settings = get_settings()
    report = Report()
    symbols = settings.symbols

    try:
        from zr.broker.longport_client import make_quote_ctx, make_trade_ctx
    except ImportError as exc:
        report.fail("import longport", exc)
        report.write(Path(args.out))
        return 1

    qctx = _run(report, "make_quote_ctx", make_quote_ctx)
    atm = None
    if qctx is not None:
        try:
            atm = run_quote_checks(report, qctx, symbols, skip_bars=args.skip_bars)
        except Exception as exc:  # noqa: BLE001
            report.fail("run_quote_checks 外层", exc)
            report.lines.append(f"```\n{traceback.format_exc()}\n```\n")

    if not args.skip_trade:
        tctx = _run(report, "make_trade_ctx", make_trade_ctx)
        if tctx is not None:
            try:
                run_trade_checks(report, tctx, atm)
            except Exception as exc:  # noqa: BLE001
                report.fail("run_trade_checks 外层", exc)

    report.write(Path(args.out))
    print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
