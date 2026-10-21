"""W2 第一步：按 YAML 跑「检测器 × 标的 × H」，输出事件 × 标签表与响应曲线。

运行：uv run python -m zr.research.run [--config config/research_w2.yaml] [--symbols SPY.US]
输出：docs/research/EVENT_LABEL_TABLE.md、docs/research/response_<detector>.png、
      ZR_DATA_ROOT/research/w2/{events,labels,curves}.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.execution.ticks import tick_size
from zr.features.bars import load_bars
from zr.features.features import compute_features
from zr.ops.markdown import df_to_md
from zr.research.events import detect_events, load_config
from zr.research.labels import clustered_t, response_curve, rolling_sigma, triple_barrier

log = logging.getLogger("zr.research.run")
ET = ZoneInfo("America/New_York")


def fee_adjusted_pnl(ret_signed: pd.Series, spot: pd.Series, und: str, fm: dict) -> pd.Series:
    """单张 ATM 期权近似 P&L（USD）：r × S × 100 × delta − 往返费用 − 点差。

    假设见 research_w2.yaml。
    """
    delta = fm.get("delta", 0.5)
    prem = spot * fm.get("premium_pct_of_spot", 0.005)
    sym = f"{und}260101C100000.US"  # 只用于 tick 规则判断
    tick = prem.map(lambda p: float(tick_size(sym, p)))
    spread = tick * fm.get("spread_ticks", 2) * 100
    fees = prem.map(
        lambda p: float(
            option_order_fees(1, "Buy", p)["total"] + option_order_fees(1, "Sell", p)["total"]
        )
    )
    return ret_signed * spot * 100 * delta - fees - spread


def run_symbol(symbol: str, cfg: dict, det_cfg: dict, root: Path) -> dict[str, pd.DataFrame]:
    und = symbol.split(".")[0]
    t0 = time.monotonic()
    df = load_bars(
        root, und, dt.date.fromisoformat(str(cfg["start"])), dt.date.fromisoformat(str(cfg["end"]))
    )
    feats = compute_features(df)
    events = detect_events(df, feats, symbol, det_cfg)
    sigma = rolling_sigma(df["close"], cfg["labels"]["sigma_window"])
    labels = []
    for h in cfg["labels"]["horizons"]:
        lab = triple_barrier(df["close"], events, h, cfg["labels"]["k"], sigma)
        labels.append(lab)
    labels = pd.concat(labels, ignore_index=True)
    labels["spot"] = df["close"].reindex(labels["ts"]).to_numpy()
    labels["pnl_fee_adj"] = fee_adjusted_pnl(
        labels["ret_signed"], labels["spot"], und, cfg["fee_model"]
    )
    labels["day"] = labels["ts"].dt.tz_convert(ET).dt.date
    curves = []
    for det, g in events.groupby("detector"):
        rc = response_curve(df["close"], g, tuple(cfg["response_horizons"]))
        rc["detector"], rc["symbol"] = det, symbol
        curves.append(rc)
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    n_days = df.index.tz_convert(ET).normalize().nunique()
    log.info(
        "%s bars=%d days=%d events=%d 用时 %.1fs",
        symbol,
        len(df),
        n_days,
        len(events),
        time.monotonic() - t0,
    )
    return {
        "events": events,
        "labels": labels,
        "curves": curves,
        "meta": pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "bars": len(df),
                    "days": n_days,
                    "seconds": round(time.monotonic() - t0, 1),
                }
            ]
        ),
    }


def summarize(labels: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    days = meta.set_index("symbol")["days"]
    rows = []
    for (det, sym, h), g in labels.groupby(["detector", "symbol", "H"]):
        g = g.dropna(subset=["ret_signed"])
        if g.empty:
            continue
        t, nd = clustered_t(g["ret_signed"], g["day"])
        rows.append(
            {
                "detector": det,
                "symbol": sym,
                "H": h,
                "n": len(g),
                "per_day": round(len(g) / days[sym], 2),
                "hit_rate": round(float((g["ret_signed"] > 0).mean()), 3),
                "label_up": round(float((g["label"] == 1).mean()), 3),
                "label_dn": round(float((g["label"] == -1).mean()), 3),
                "mean_ret_bp": round(float(g["ret_signed"].mean() * 1e4), 2),
                "t_daily": round(t, 2),
                "days": nd,
                "mean_pnl_fee_adj_usd": round(float(g["pnl_fee_adj"].mean()), 2),
                "mean_exit_min": round(float(g["exit_min"].mean()), 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["detector", "symbol", "H"]).reset_index(drop=True)


def peak_horizon(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (det, sym), g in curves.groupby(["detector", "symbol"]):
        g = g.sort_values("horizon")
        i = int(g["mean"].to_numpy().argmax())
        rows.append(
            {
                "detector": det,
                "symbol": sym,
                "peak_horizon": int(g["horizon"].iloc[i]),
                "peak_mean_bp": round(float(g["mean"].iloc[i] * 1e4), 2),
                "n": int(g["n"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(curves: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for det, g in curves.groupby("detector"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for sym, gg in g.groupby("symbol"):
            gg = gg.sort_values("horizon")
            ax.plot(
                gg["horizon"],
                gg["mean"] * 1e4,
                marker="o",
                label=f"{sym} (n={int(gg['n'].iloc[0])})",
            )
            ax.fill_between(gg["horizon"], gg["q25"] * 1e4, gg["q75"] * 1e4, alpha=0.1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{det} response curve: mean signed return (bp), band = q25-q75")
        ax.set_xlabel("minutes after event")
        ax.set_ylabel("bp")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = out_dir / f"response_{det}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def r004_r005(labels: pd.DataFrame) -> list[str]:
    """R-004：H=30 vs H=5 扣费后期望差；R-005：按实际离场时长分层的期望。"""
    out = ["## R-004 持仓时长：H=5 vs H=30（扣费后近似 P&L，USD/张）", ""]
    piv = labels.groupby(["detector", "symbol", "H"])["pnl_fee_adj"].mean().unstack("H")
    if 5 in piv and 30 in piv:
        tab = piv[[5, 30]].copy()
        tab["diff_30_minus_5"] = tab[30] - tab[5]
        out.append(df_to_md(tab.round(2).reset_index()))
    out += ["", "## R-005 按实际离场时长分桶（H=30 标签）", ""]
    g = labels[labels["H"] == 30].dropna(subset=["exit_min"]).copy()
    bins = [0, 5, 10, 15, 30]
    g["exit_bucket"] = pd.cut(g["exit_min"], bins=bins, labels=["≤5", "5–10", "10–15", "15–30"])
    tab = (
        g.groupby(["detector", "exit_bucket"], observed=True)
        .agg(
            n=("ret_signed", "size"),
            hit_rate=("ret_signed", lambda s: round(float((s > 0).mean()), 3)),
            mean_ret_bp=("ret_signed", lambda s: round(float(s.mean() * 1e4), 2)),
            mean_pnl_fee_adj=("pnl_fee_adj", lambda s: round(float(s.mean()), 2)),
        )
        .reset_index()
    )
    out.append(df_to_md(tab))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research_w2.yaml")
    parser.add_argument("--symbols", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    det_cfg = load_config(Path(cfg["detectors_config"]))
    root = get_settings().zr_data_root
    symbols = args.symbols.split(",") if args.symbols else cfg["symbols"]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    res = [run_symbol(s, cfg, det_cfg, root) for s in symbols]
    events = pd.concat([r["events"] for r in res], ignore_index=True)
    labels = pd.concat([r["labels"] for r in res], ignore_index=True)
    curves = pd.concat([r["curves"] for r in res], ignore_index=True)
    meta = pd.concat([r["meta"] for r in res], ignore_index=True)
    store = root / "research" / "w2"
    store.mkdir(parents=True, exist_ok=True)
    events.to_parquet(store / "events.parquet", index=False)
    labels.to_parquet(store / "labels.parquet", index=False)
    curves.to_parquet(store / "curves.parquet", index=False)

    table = summarize(labels, meta)
    peaks = peak_horizon(curves)
    pngs = plot_curves(curves, out_dir)
    total = time.monotonic() - t0
    lines = [
        "# 事件 × 标签表（W2 第一步）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"配置 `{args.config}` + `{cfg['detectors_config']}` · 总耗时 {total:.0f}s",
        f"数据：长桥正股 1 分钟 K（盘中），{cfg['start']} ~ {cfg['end']}。",
        "标签：triple-barrier，障碍 ±k·σ_H（k={k}，σ 窗口 {w} 分钟），H ∈ {hs}。".format(
            k=cfg["labels"]["k"], w=cfg["labels"]["sigma_window"], hs=cfg["labels"]["horizons"]
        ),
        "hit_rate = 方向化收益 > 0 的比例（原始成功率）；t_daily = 按日聚合后的 t 值；"
        "mean_pnl_fee_adj_usd = 单张 ATM 期权近似 P&L（delta {d}、权利金≈{p:.1%} 现价、"
        "点差 {s} tick、D-013 费用），只是量级参考。".format(
            d=cfg["fee_model"]["delta"],
            p=cfg["fee_model"]["premium_pct_of_spot"],
            s=cfg["fee_model"]["spread_ticks"],
        ),
        "E8 为 ungated（无期权 OI 数据，未按做市商 gamma 门控）。",
        "注意：mean_ret_bp 是事件等权均值，t_daily 是日等权（先按日平均再做 t）；"
        "事件数与当日收益相关时两者可能异号"
        "（例如 E8 在高波动日事件多且为正，安静日为负），两者都要看。",
        "注意：R-005 的离场时长分桶按结果分组（早触障碍的必然是大波动），"
        "存在选择偏差，只作描述不作证据。",
        "",
        "## 数据与耗时",
        "",
        df_to_md(meta),
        "",
        "## 检测器 × 标的 × H",
        "",
        df_to_md(table),
        "",
        "## 响应曲线见顶（均值最高的 horizon）",
        "",
        df_to_md(peaks),
        "",
        "## 响应曲线（方向化收益，bp）",
        "",
        df_to_md(
            curves.assign(
                **{
                    c: (curves[c] * 1e4).round(2)
                    for c in ["mean", "median", "q10", "q25", "q75", "q90"]
                }
            ).round(3)
        ),
        "",
        "图：" + "、".join(f"`{p.name}`" for p in pngs),
        "",
        *r004_r005(labels),
        "",
        "## 复现",
        "",
        "```bash",
        "bash deploy/sync_from_vps.sh --now   # 需要 bars_1m",
        f"uv run python -m zr.research.run --config {args.config}",
        "```",
        "",
    ]
    (out_dir / "EVENT_LABEL_TABLE.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s，总耗时 %.0fs", out_dir / "EVENT_LABEL_TABLE.md", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
