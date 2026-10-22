"""一次性诊断拉取（diag/indicator-parity 第 1 节）：MU 2026-10-19～10-21，4 种口径各拉一份，
只读不下单。写到 $ZR_DATA_ROOT/diag/indicator_parity/<config>.parquet，供指标口径比对用。
只在 VPS 跑（Longbridge API，本机 DNS 污染，见 CLAUDE.md）。

uv run python scripts/diag_pull_indicator_parity.py
"""

from __future__ import annotations

import datetime as dt
import logging

from longport.openapi import AdjustType, Period, TradeSessions

from zr.broker.longport_client import make_quote_ctx
from zr.config import get_settings
from zr.data.backfill_bars import bars_to_frame

log = logging.getLogger("diag_pull")
SYMBOL = "MU.US"
UNTIL = dt.date(2026, 10, 19)
CONFIGS = [
    ("min1_noadjust", Period.Min_1, AdjustType.NoAdjust),
    ("min1_fwdadjust", Period.Min_1, AdjustType.ForwardAdjust),
    ("min2_noadjust", Period.Min_2, AdjustType.NoAdjust),
    ("min2_fwdadjust", Period.Min_2, AdjustType.ForwardAdjust),
]


def pull(qctx, period, adjust, until: dt.date, max_pages: int = 6):
    cursor = None
    frames = []
    for _ in range(max_pages):
        bars = qctx.history_candlesticks_by_offset(
            SYMBOL, period, adjust, False, 1000, cursor, TradeSessions.All
        )
        if not bars:
            break
        df = bars_to_frame(SYMBOL, bars)
        frames.append(df)
        page_min = df["ts"].min()
        if cursor is not None and page_min.tz_localize(None) >= cursor:
            break
        cursor = page_min.tz_localize(None).to_pydatetime() - dt.timedelta(minutes=1)
        if page_min.date() < until:
            break
    import pandas as pd

    out = pd.concat(frames, ignore_index=True).drop_duplicates(["ts", "symbol"])
    return out[out["ts"].dt.tz_convert("UTC").dt.date >= until].sort_values("ts")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    out_dir = root / "diag" / "indicator_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    qctx = make_quote_ctx()
    for name, period, adjust in CONFIGS:
        df = pull(qctx, period, adjust, UNTIL)
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        log.info("%s: %d 行，%s ~ %s -> %s", name, len(df), df["ts"].min(), df["ts"].max(), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
