"""一次性诊断拉取（audit/base-final 第 1 节）：长桥官方 2 / 5 分钟 K（NoAdjust），MU、QQQ，
全部可得日期（向前翻页到 2023-09-01 或无数据为止），只读不下单。
默认 RTH（TradeSessions.Intraday）2 / 5 分钟；--all 拉含盘前盘后（TradeSessions.All）的 2 分钟 K
（口径 C 用）。写到 $ZR_DATA_ROOT/diag/official_bars/<SYM>_<period>[_all].parquet。
只在 VPS 跑（长桥 API）。

uv run python scripts/diag_pull_candles.py [--all]
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import time

import pandas as pd
from longport.openapi import AdjustType, Period, TradeSessions

from zr.broker.longport_client import make_quote_ctx
from zr.config import get_settings
from zr.data.backfill_bars import bars_to_frame
from zr.features.bars import ET

log = logging.getLogger("diag_pull_candles")
UNTIL = dt.date(2023, 9, 1)


def pull(qctx, symbol: str, period, sessions) -> pd.DataFrame:
    # 2026-10-22 VPS 实测：time 参数按美东时间解释（返回的 timestamp 是 UTC），游标要换成 ET
    cursor, frames = None, []
    for _ in range(3000):
        bars = qctx.history_candlesticks_by_offset(
            symbol, period, AdjustType.NoAdjust, False, 1000, cursor, sessions
        )
        if not bars:
            break
        df = bars_to_frame(symbol, bars)
        frames.append(df)
        page_min = df["ts"].min()
        nxt = page_min.tz_convert(ET).tz_localize(None).to_pydatetime() - dt.timedelta(minutes=1)
        if page_min.date() < UNTIL or (cursor is not None and nxt >= cursor):
            break
        cursor = nxt
        time.sleep(0.25)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["ts", "symbol"])
    return out.sort_values("ts")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = get_settings().zr_data_root / "diag" / "official_bars"
    out.mkdir(parents=True, exist_ok=True)
    qctx = make_quote_ctx()
    full = "--all" in sys.argv[1:]
    sessions = TradeSessions.All if full else TradeSessions.Intraday
    periods = (("2m", Period.Min_2),) if full else (("2m", Period.Min_2), ("5m", Period.Min_5))
    for sym in ("MU.US", "QQQ.US"):
        for name, period in periods:
            df = pull(qctx, sym, period, sessions)
            p = out / f"{sym.split('.')[0]}_{name}{'_all' if full else ''}.parquet"
            df.to_parquet(p, index=False)
            log.info("%s %s：%d 根，%s ~ %s", sym, name, len(df), df["ts"].min(), df["ts"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
