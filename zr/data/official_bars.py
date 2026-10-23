"""长桥官方 2 分钟 K（TradeSessions.All，NoAdjust）的增量更新：拉最近若干页并入
$ZR_DATA_ROOT/diag/official_bars/<SYM>_2m_all.parquet（同一 ts 以新拉的为准）。只在 VPS 跑。

`history_candlesticks_by_offset` 的 time 参数按美东时间解释（docs/API_FACTS.md 补充）。
运行：uv run python -m zr.data.official_bars [--pages 2]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.data.backfill_bars import bars_to_frame
from zr.features.bars import ET

log = logging.getLogger("zr.data.official_bars")
SYMBOLS = ("MU.US", "QQQ.US")


def path(root: Path, symbol: str) -> Path:
    return root / "diag" / "official_bars" / f"{symbol.split('.')[0]}_2m_all.parquet"


def merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


def update(qctx, root: Path, symbol: str, pages: int = 2) -> int:
    from longport.openapi import AdjustType, Period, TradeSessions

    cursor, frames = None, []
    for _ in range(pages):
        bars = qctx.history_candlesticks_by_offset(
            symbol, Period.Min_2, AdjustType.NoAdjust, False, 1000, cursor, TradeSessions.All
        )
        if not bars:
            break
        df = bars_to_frame(symbol, bars)
        frames.append(df)
        cursor = df["ts"].min().tz_convert(ET).tz_localize(None).to_pydatetime() - dt.timedelta(
            minutes=1
        )
        time.sleep(0.25)
    if not frames:
        return 0
    p = path(root, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(p) if p.exists() else None
    out = merge(old, pd.concat(frames, ignore_index=True))
    out.to_parquet(p, index=False)
    return len(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from zr.broker.longport_client import make_quote_ctx

    root = get_settings().zr_data_root
    q = make_quote_ctx()
    for s in SYMBOLS:
        log.info("%s：合并后 %d 根", s, update(q, root, s, args.pages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
