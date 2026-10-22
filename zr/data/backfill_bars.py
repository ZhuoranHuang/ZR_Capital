"""正股 1 分钟 K 回填（在 VPS 上跑）。

history_candlesticks_by_offset(symbol, Min_1, NoAdjust, forward=False, 1000, cursor,
                               TradeSessions.All)
向过去翻页到空页或早于 --until（默认 2023-01-01）。落盘：
  ZR_DATA_ROOT/bars_1m/underlying=XXX/year=YYYY/bars.parquet
按 (ts, symbol) 去重；断点续传：从已有数据的最早 ts 继续往前。

运行：uv run python -m zr.data.backfill_bars --symbols SPY.US [--until 2023-01-01]
      uv run python -m zr.data.backfill_bars --inventory   # 打印 markdown 清单

已验证（2026-10-21 api_facts）：cursor 传 naive UTC datetime；返回 Candlestick{open,high,low,close,
volume,turnover,timestamp(naive UTC),trade_session}。forward=False 的语义按运行结果为「向过去」。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from zr.config import get_settings

log = logging.getLogger("zr.backfill")

PAGE_SIZE = 1000
PAGE_SLEEP_S = 0.15
FLUSH_EVERY_PAGES = 40
COLUMNS = ["ts", "symbol", "open", "high", "low", "close", "volume", "turnover", "session"]


def bars_to_frame(symbol: str, bars: list[Any]) -> pd.DataFrame:
    rows = [
        {
            "ts": pd.Timestamp(b.timestamp).tz_localize("UTC")
            if pd.Timestamp(b.timestamp).tzinfo is None
            else pd.Timestamp(b.timestamp).tz_convert("UTC"),
            "symbol": symbol,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),
            "turnover": float(b.turnover),
            "session": str(b.trade_session).split(".")[-1],
        }
        for b in bars
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def _underlying_dir(root: Path, symbol: str) -> Path:
    return root / "bars_1m" / f"underlying={symbol.split('.')[0]}"


def existing_min_ts(root: Path, symbol: str) -> pd.Timestamp | None:
    files = sorted(_underlying_dir(root, symbol).glob("year=*/bars.parquet"))
    if not files:
        return None
    # 最早的年份文件就够了
    df = pd.read_parquet(files[0], columns=["ts"])
    return None if df.empty else df["ts"].min()


def write_merged(root: Path, symbol: str, df: pd.DataFrame) -> dict[int, int]:
    """按年合并写入，去重（同一 ts 以本次拉到的为准：长桥会事后修订 bar，例如 MU 2026-10-21
    09:30 的 open 由 925.42 改为 923.735，见 BASE_FINAL_AUDIT.md 第 1 节）；返回 {year: 合并后行数}。"""
    out: dict[int, int] = {}
    if df.empty:
        return out
    for year, grp in df.groupby(df["ts"].dt.year):
        path = _underlying_dir(root, symbol) / f"year={int(year)}" / "bars.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_parquet(path)
            grp = pd.concat([old, grp], ignore_index=True)
        grp = (
            grp.drop_duplicates(["ts", "symbol"], keep="last")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        grp.to_parquet(path, index=False)
        out[int(year)] = len(grp)
    return out


def backfill_symbol(
    qctx: Any,
    root: Path,
    symbol: str,
    until: dt.date,
    *,
    page_sleep_s: float = PAGE_SLEEP_S,
    max_pages: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    from longport.openapi import AdjustType, Period, TradeSessions

    resume = existing_min_ts(root, symbol)
    cursor: dt.datetime | None = (
        None
        if resume is None
        else resume.tz_convert("UTC").tz_localize(None).to_pydatetime() - dt.timedelta(minutes=1)
    )
    log.info("%s 从 %s 开始回填", symbol, cursor or "最新")
    pages = 0
    rows = 0
    earliest: pd.Timestamp | None = resume
    buffer: list[pd.DataFrame] = []
    stop_reason = "max_pages"
    while max_pages is None or pages < max_pages:
        bars = qctx.history_candlesticks_by_offset(
            symbol, Period.Min_1, AdjustType.NoAdjust, False, PAGE_SIZE, cursor, TradeSessions.All
        )
        pages += 1
        if not bars:
            stop_reason = "empty_page"
            break
        df = bars_to_frame(symbol, bars)
        buffer.append(df)
        rows += len(df)
        page_min = df["ts"].min()
        if earliest is None or page_min < earliest:
            earliest = page_min
        if cursor is not None and page_min.tz_localize(None) >= cursor:
            stop_reason = "no_progress"
            break
        cursor = page_min.tz_localize(None).to_pydatetime() - dt.timedelta(minutes=1)
        if page_min.date() < until:
            stop_reason = "reached_until"
            break
        if len(buffer) >= FLUSH_EVERY_PAGES:
            write_merged(root, symbol, pd.concat(buffer, ignore_index=True))
            buffer = []
            log.info("%s 已到 %s（%d 页 %d 行）", symbol, page_min, pages, rows)
        sleep(page_sleep_s)
    if buffer:
        write_merged(root, symbol, pd.concat(buffer, ignore_index=True))
    return {
        "symbol": symbol,
        "pages": pages,
        "rows_fetched": rows,
        "earliest": None if earliest is None else earliest.isoformat(),
        "stop_reason": stop_reason,
    }


def inventory(root: Path) -> str:
    lines = [
        "# 数据清单：正股 1 分钟 K",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC，"
        "`zr/data/backfill_bars.py --inventory`",
        "",
        "| 标的 | 最早 (UTC) | 最晚 (UTC) | 行数 | 按年 |",
        "|---|---|---|---|---|",
    ]
    for d in sorted((root / "bars_1m").glob("underlying=*")):
        files = sorted(d.glob("year=*/bars.parquet"))
        if not files:
            continue
        parts = [pd.read_parquet(f, columns=["ts"]) for f in files]
        by_year = ", ".join(
            f"{f.parent.name[5:]}: {len(p)}" for f, p in zip(files, parts, strict=True)
        )
        allts = pd.concat(parts)["ts"]
        lines.append(
            f"| {d.name[11:]} | {allts.min()} | {allts.max()} | {len(allts)} | {by_year} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="正股 1 分钟 K 回填")
    parser.add_argument("--symbols", default=None, help="逗号分隔，默认 universe 全部")
    parser.add_argument("--until", default="2023-01-01")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    root = get_settings().zr_data_root
    if args.inventory:
        print(inventory(root))
        return 0

    from zr.broker.longport_client import make_quote_ctx
    from zr.data.universe import load_universe

    symbols = args.symbols.split(",") if args.symbols else load_universe().symbols
    until = dt.date.fromisoformat(args.until)
    qctx = make_quote_ctx()
    for s in symbols:
        res = backfill_symbol(qctx, root, s, until, max_pages=args.max_pages)
        log.info("完成 %s", res)
        print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
