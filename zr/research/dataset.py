"""元标签数据集：事件 × 特征矩阵（R-013）。

每行一个事件：第 3 层全部正股特征（事件时刻）、quote_slope_1（有盘口数据的日期，否则 NaN）、
检测器 ID、方向、标的，以及 triple-barrier 标签 H=5 与 H=15（label、ret_signed、pnl_fee_adj）。
运行：uv run python -m zr.research.dataset  → ZR_DATA_ROOT/research/w2/dataset.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from zr.config import get_settings
from zr.features.bars import load_bars
from zr.features.features import FEATURE_COLUMNS, compute_features
from zr.features.option_features import OPTION_FEATURE_COLUMNS, compute_option_features
from zr.research.events import detect_events, load_config
from zr.research.labels import rolling_sigma, triple_barrier
from zr.research.run import fee_adjusted_pnl

log = logging.getLogger("zr.research.dataset")
ET = ZoneInfo("America/New_York")
H_LIST = (5, 15)


def quote_slope_1(root: Path, und: str, days: list[dt.date]) -> pd.Series:
    """录制器 Depth：正股中间价在每分钟内的斜率（末价 − 首价，bp）；只有录制过的日期有值。"""
    rows = []
    for day in days:
        d = root / "recorder" / f"date={day.isoformat()}" / f"underlying={und}"
        files = sorted(d.glob("depth_*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df[df["symbol"] == f"{und}.US"]
        if df.empty:
            continue
        mids = []
        for r in df.itertuples(index=False):
            p = json.loads(r.payload)
            b, a = p.get("bids") or [], p.get("asks") or []
            if b and a and b[0]["price"] and a[0]["price"]:
                mids.append(
                    (
                        pd.Timestamp(r.recv_ts_utc_ns, unit="ns", tz="UTC"),
                        (b[0]["price"] + a[0]["price"]) / 2,
                    )
                )
        m = pd.DataFrame(mids, columns=["ts", "mid"]).sort_values("ts")
        m["minute"] = m["ts"].dt.floor("min")
        g = m.groupby("minute")["mid"]
        slope = (g.last() / g.first() - 1) * 1e4
        rows.append(slope)
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows)


def build_symbol(symbol: str, root: Path, det_cfg: dict, cfg: dict) -> pd.DataFrame:
    und = symbol.split(".")[0]
    df = load_bars(
        root, und, dt.date.fromisoformat(str(cfg["start"])), dt.date.fromisoformat(str(cfg["end"]))
    )
    feats = compute_features(df)
    events = detect_events(df, feats, symbol, det_cfg)
    sigma = rolling_sigma(df["close"], cfg["labels"]["sigma_window"])
    out = events.copy()
    fe = feats.reindex(events["ts"])
    for c in FEATURE_COLUMNS:
        out[c] = fe[c].to_numpy()
    out["spot"] = df["close"].reindex(events["ts"]).to_numpy()
    for h in H_LIST:
        lab = triple_barrier(df["close"], events, h, cfg["labels"]["k"], sigma)
        out[f"label_h{h}"] = lab["label"].to_numpy()
        out[f"ret_h{h}"] = lab["ret_signed"].to_numpy()
        out[f"exit_h{h}"] = lab["exit_min"].to_numpy()
        out[f"pnl_h{h}"] = fee_adjusted_pnl(
            lab["ret_signed"], out["spot"], und, cfg["fee_model"]
        ).to_numpy()
    out["day"] = out["ts"].dt.tz_convert(ET).dt.date
    days_with_depth = (
        sorted({p.name[5:] for p in (root / "recorder").glob("date=*")})
        if (root / "recorder").exists()
        else []
    )
    qs = quote_slope_1(root, und, [dt.date.fromisoformat(d) for d in days_with_depth])
    # 事件所在分钟的上一分钟斜率（无前视）
    prev_min = out["ts"].dt.floor("min") - pd.Timedelta(minutes=1)
    out["quote_slope_1"] = qs.reindex(prev_min).to_numpy() if len(qs) else float("nan")
    # 期权特征（PLAN_W3 线 a）：只在有 ThetaData 的交易日有值，其余 NaN
    if cfg.get("with_options", True):
        of = compute_option_features(root, und, df)
        ofe = of.reindex(events["ts"])
        for c in OPTION_FEATURE_COLUMNS:
            out[c] = ofe[c].to_numpy(dtype=float)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_w2.yaml")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--start", default=None, help="覆盖配置的起始日")
    parser.add_argument("--end", default=None, help="覆盖配置的结束日")
    parser.add_argument("--out", default=None, help="默认 ZR_DATA_ROOT/research/w2/dataset.parquet")
    parser.add_argument("--no-options", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.start:
        cfg["start"] = args.start
    if args.end:
        cfg["end"] = args.end
    cfg["with_options"] = not args.no_options
    det_cfg = load_config(Path(cfg["detectors_config"]))
    root = get_settings().zr_data_root
    symbols = args.symbols.split(",") if args.symbols else cfg["symbols"]
    parts = []
    for s in symbols:
        d = build_symbol(s, root, det_cfg, cfg)
        log.info(
            "%s 事件 %d 行，quote_slope_1 非空 %d", s, len(d), int(d["quote_slope_1"].notna().sum())
        )
        parts.append(d)
    ds = pd.concat(parts, ignore_index=True)
    out = Path(args.out) if args.out else root / "research" / "w2" / "dataset.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    log.info("写入 %s：%d 行 × %d 列", out, len(ds), ds.shape[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
