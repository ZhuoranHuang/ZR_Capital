"""Task 1.3 第一步：导出长桥历史订单、成交与费用到 Parquet。

运行：uv run python -m zr.journal.trade_history --since 2026-05-01 --options-only
输出：ZR_DATA_ROOT/journal/{orders,executions,charges}.parquet

只读：history_orders / history_executions / order_detail。无任何下单调用。
签名来源：longport 4.3.7 内省——
  history_orders(symbol=None, status=None, side=None, market=None, start_at=None, end_at=None)
  history_executions(symbol=None, start_at=None, end_at=None)
  order_detail(order_id)
单次查询时间跨度上限、charge_detail 字段结构：未验证，以运行结果为准。
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from zr.broker.serialize import to_py
from zr.config import get_settings

OPTION_RE = re.compile(r"^[A-Z.\-]+\d{6}[CP]\d+\.US$")
CHUNK_DAYS = 28
MIN_CALL_GAP_S = 1.05


def is_option_symbol(symbol: str) -> bool:
    return bool(OPTION_RE.match(symbol))


def date_chunks(
    since: dt.datetime, until: dt.datetime, days: int = CHUNK_DAYS
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    """[since, until) 按 days 切成连续区间，最后一段截到 until。"""
    start = since
    step = dt.timedelta(days=days)
    while start < until:
        end = min(start + step, until)
        yield start, end
        start = end


class Throttle:
    """保证相邻调用间隔 ≥ gap 秒。"""

    def __init__(self, gap_s: float = MIN_CALL_GAP_S, sleep: Callable[[float], None] = time.sleep):
        self.gap = gap_s
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self._last is not None:
            remaining = self.gap - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = time.monotonic()


def _flatten_charges(order_id: str, detail: Any) -> list[dict[str, Any]]:
    """从 order_detail.charge_detail.items[].fees[] 拉平为行。字段名未验证。"""
    rows: list[dict[str, Any]] = []
    cd = to_py(getattr(detail, "charge_detail", None))
    if not isinstance(cd, dict):
        return rows
    total = cd.get("total_amount")
    currency = cd.get("currency")
    for item in cd.get("items") or []:
        for fee in item.get("fees") or []:
            rows.append(
                {
                    "order_id": order_id,
                    "item_code": item.get("code"),
                    "item_name": item.get("name"),
                    "fee_code": fee.get("code"),
                    "fee_name": fee.get("name"),
                    "amount": fee.get("amount"),
                    "currency": fee.get("currency") or currency,
                    "total_amount": total,
                }
            )
    if not rows and total is not None:
        rows.append(
            {
                "order_id": order_id,
                "item_code": None,
                "item_name": None,
                "fee_code": None,
                "fee_name": None,
                "amount": None,
                "currency": currency,
                "total_amount": total,
            }
        )
    return rows


def fetch(
    tctx: Any,
    since: dt.datetime,
    until: dt.datetime,
    *,
    options_only: bool,
    throttle: Throttle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from longport.openapi import Market

    orders: dict[str, dict[str, Any]] = {}
    executions: dict[str, dict[str, Any]] = {}

    for start, end in date_chunks(since, until):
        throttle.wait()
        for o in tctx.history_orders(None, None, None, Market.US, start, end):
            row = to_py(o)
            if options_only and not is_option_symbol(row.get("symbol", "")):
                continue
            orders[row["order_id"]] = row
        throttle.wait()
        for e in tctx.history_executions(None, start, end):
            row = to_py(e)
            if options_only and not is_option_symbol(row.get("symbol", "")):
                continue
            executions[row["trade_id"]] = row

    filled_ids = {e["order_id"] for e in executions.values()}
    charges: list[dict[str, Any]] = []
    for oid in sorted(filled_ids):
        throttle.wait()
        detail = tctx.order_detail(oid)
        charges.extend(_flatten_charges(oid, detail))

    # 已观测（2026-10-21）：history_orders 的 msg 对被拒订单为空，拒绝原因只在 order_detail.msg
    for oid, row in orders.items():
        if row.get("status") == "Rejected":
            throttle.wait()
            row["reject_reason"] = to_py(tctx.order_detail(oid)).get("msg")

    return (
        pd.DataFrame(list(orders.values())),
        pd.DataFrame(list(executions.values())),
        pd.DataFrame(charges),
    )


def write_outputs(root: Path, orders: pd.DataFrame, execs: pd.DataFrame, charges: pd.DataFrame):
    out = root / "journal"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in [("orders", orders), ("executions", execs), ("charges", charges)]:
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} 行 -> {out / f'{name}.parquet'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出长桥交易史（只读）")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD")
    parser.add_argument("--until", default=None, help="YYYY-MM-DD，默认今天")
    parser.add_argument("--options-only", action="store_true")
    args = parser.parse_args(argv)

    since = dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.UTC)
    until = (
        dt.datetime.fromisoformat(args.until).replace(tzinfo=dt.UTC)
        if args.until
        else dt.datetime.now(dt.UTC)
    )

    from zr.broker.longport_client import make_trade_ctx

    tctx = make_trade_ctx()
    orders, execs, charges = fetch(
        tctx, since, until, options_only=args.options_only, throttle=Throttle()
    )
    write_outputs(get_settings().zr_data_root, orders, execs, charges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
