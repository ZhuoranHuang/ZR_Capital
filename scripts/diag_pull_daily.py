"""一次性诊断拉取（audit/known-answer 第 3 节）：MU / QQQ / SPY 长桥日线
（NoAdjust，RTH），
2023-09-01 起，只读不下单。写到 $ZR_DATA_ROOT/diag/daily/<SYMBOL>.parquet，
供与 1 分钟 K 合成日线比对。
只在 VPS 跑（长桥 API，本机 DNS 污染，见 CLAUDE.md）。

uv run python scripts/diag_pull_daily.py
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from longport.openapi import AdjustType, Period, TradeSessions

from zr.broker.longport_client import make_quote_ctx
from zr.config import get_settings
from zr.data.backfill_bars import bars_to_frame

log = logging.getLogger("diag_pull_daily")
SYMBOLS = ["MU.US", "QQQ.US", "SPY.US"]
UNTIL = dt.date(2023, 9, 1)


def pull(qctx, symbol: str) -> pd.DataFrame:
    cursor, frames = None, []
    for _ in range(10):
        bars = qctx.history_candlesticks_by_offset(
            symbol, Period.Day, AdjustType.NoAdjust, False, 1000, cursor, TradeSessions.Intraday
        )
        if not bars:
            break
        df = bars_to_frame(symbol, bars)
        frames.append(df)
        page_min = df["ts"].min()
        if page_min.date() < UNTIL or (cursor is not None and page_min.tz_localize(None) >= cursor):
            break
        cursor = page_min.tz_localize(None).to_pydatetime() - dt.timedelta(days=1)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["ts", "symbol"])
    return out.sort_values("ts")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = get_settings().zr_data_root / "diag" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    qctx = make_quote_ctx()
    for s in SYMBOLS:
        df = pull(qctx, s)
        p = out_dir / f"{s.split('.')[0]}.parquet"
        df.to_parquet(p, index=False)
        log.info("%s：%d 行，%s ~ %s -> %s", s, len(df), df["ts"].min(), df["ts"].max(), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
