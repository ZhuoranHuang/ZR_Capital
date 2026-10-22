"""事件 × 特征 × 标签矩阵 v2（METHOD_V2 规格 3/4/5）。

每行一个事件（E1–E9 + E1T，config/events_v2.yaml），列 = 正股特征 + 通道质量 + 结构上下文 +
期权特征（D-019 前的
口径，OI 变化待改）+ 日历 + 跨资产 + 共振 + 检测器/方向 + 四套标签。全部只用 ≤t
的数据（各模块有无前视测试）。
运行：uv run python -m zr.research.dataset_v2 --symbol MU.US →
ZR_DATA_ROOT/research/v2/dataset_<UND>.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.calendar_features import CALENDAR_COLUMNS, compute_calendar
from zr.features.channel import CHANNEL_FEATURE_COLUMNS, channel_states
from zr.features.cross_asset import compute_cross_asset, cross_asset_columns
from zr.features.features import FEATURE_COLUMNS, compute_features
from zr.features.option_features import OPTION_FEATURE_COLUMNS, compute_option_features
from zr.features.structure import STRUCTURE_COLUMNS, compute_structure
from zr.research import outcomes as oc
from zr.research.events import confluence, events_from_matrix, load_config, signal_matrix
from zr.research.labels_v2 import option_labels, stock_labels

log = logging.getLogger("zr.research.dataset_v2")
CONFIG = Path(__file__).resolve().parents[2] / "config" / "events_v2.yaml"


def feature_columns(und: str) -> list[str]:
    return (
        FEATURE_COLUMNS
        + CHANNEL_FEATURE_COLUMNS
        + STRUCTURE_COLUMNS
        + OPTION_FEATURE_COLUMNS
        + CALENDAR_COLUMNS
        + cross_asset_columns(und)
        + ["confl_n_same_dir_5m", "confl_families_5m"]
    )


def build(root: Path, und: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    t0 = time.monotonic()
    df = load_bars(root, und, start, end)
    feats = compute_features(df)
    chans = channel_states(df, feats)
    struct = compute_structure(df, feats, chans)
    xa = compute_cross_asset(df, und, root)
    cal = compute_calendar(df.index, root, und)
    opt = compute_option_features(root, und, df)
    log.info("%s 特征完成 %.0fs", und, time.monotonic() - t0)
    sig = signal_matrix(df, feats, load_config(CONFIG), extra={"_chans": chans})
    conf = confluence(sig, 5)
    ev = events_from_matrix(sig, f"{und}.US", conf)
    allf = pd.concat([feats, chans, struct, opt, cal, xa], axis=1)
    allf = allf.loc[:, ~allf.columns.duplicated()]
    fe = allf.reindex(ev["ts"])
    cols = [c for c in feature_columns(und) if c in fe.columns]
    ev = pd.concat([ev.reset_index(drop=True), fe[cols].reset_index(drop=True)], axis=1)
    ev["day"] = ev["ts"].dt.tz_convert(ET).dt.date
    ev["bar_minute"] = oc.minute_index(ev["ts"])
    labs = stock_labels(df, ev[["ts", "direction"]])
    outc = oc.build(root, und, start, end)
    olab = option_labels(ev[["ts", "direction"]], outc, ev["bar_minute"].to_numpy())
    ds = pd.concat([ev, labs, olab], axis=1)
    log.info(
        "%s 事件 %d，期权标签非空 %d，%.0fs",
        und,
        len(ds),
        int(ds.get("net_cap10", pd.Series()).notna().sum()),
        time.monotonic() - t0,
    )
    return ds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MU.US")
    p.add_argument("--start", default="2023-10-02")
    p.add_argument("--end", default="2026-10-21")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    und = a.symbol.split(".")[0]
    ds = build(root, und, dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end))
    out = root / "research" / "v2" / f"dataset_{und}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    log.info("写入 %s：%d 行 × %d 列", out, len(ds), ds.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
