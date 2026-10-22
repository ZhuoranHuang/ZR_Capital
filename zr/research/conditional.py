"""条件分层表（R-012）：每个检测器 × 标的，按状态分层后重算成功率、均值、t、样本数。

分层维度：入场时段、当日实现波动三分位（用事件时刻之前的 rv_30 相对前一日的比值分位，无前视）、
隔夜跳空方向、事件方向与开盘至今收益是否同向、星期几。
运行：uv run python -m zr.research.conditional [--H 5]
输入：ZR_DATA_ROOT/research/w2/labels.parquet（run.py 产出）+ 重算的特征。
输出：docs/research/CONDITIONAL_TABLE.md，只报告样本 ≥ min_n 的格子并写明格子总数。
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import load_bars
from zr.features.features import compute_features
from zr.ops.markdown import df_to_md
from zr.research.labels import clustered_t

ET = ZoneInfo("America/New_York")
SESSIONS = [
    (570, 600, "09:30–10:00"),
    (600, 690, "10:00–11:30"),
    (690, 840, "11:30–14:00"),
    (840, 930, "14:00–15:30"),
    (930, 961, "15:30–16:00"),
]


def session_bucket(minute_of_day: pd.Series) -> pd.Series:
    out = pd.Series("other", index=minute_of_day.index)
    for lo, hi, name in SESSIONS:
        out[(minute_of_day >= lo) & (minute_of_day < hi)] = name
    return out


def attach_conditions(
    labels: pd.DataFrame, feats_by_symbol: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    parts = []
    for sym, g in labels.groupby("symbol"):
        f = feats_by_symbol[sym]
        fe = f.reindex(g["ts"])
        et = g["ts"].dt.tz_convert(ET)
        mod = et.dt.hour * 60 + et.dt.minute
        g = g.copy()
        g["session"] = session_bucket(mod.reset_index(drop=True)).to_numpy()
        # 实现波动分位：rv_30_vs_day（事件前 30 分钟 vs 前一日），按标的分三分位
        rv = fe["rv_30_vs_day"].to_numpy()
        g["rv_tercile"] = (
            pd.qcut(pd.Series(rv), 3, labels=["低波动", "中波动", "高波动"]).astype(str).to_numpy()
        )
        gap = fe["gap_overnight"].to_numpy()
        g["gap_dir"] = np.where(gap > 0, "跳空向上", np.where(gap < 0, "跳空向下", "无跳空"))
        rso = fe["ret_since_open"].to_numpy()
        g["with_day_trend"] = np.where(
            np.sign(rso) == g["direction"].to_numpy(), "顺开盘至今方向", "逆开盘至今方向"
        )
        g["weekday"] = et.dt.day_name().str[:3].to_numpy()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def cell_table(df: pd.DataFrame, dim: str, min_n: int) -> tuple[pd.DataFrame, int]:
    rows = []
    total = 0
    for (det, sym, val), g in df.groupby(["detector", "symbol", dim]):
        total += 1
        g = g.dropna(subset=["ret_signed"])
        if len(g) < min_n:
            continue
        t, nd = clustered_t(g["ret_signed"], g["day"])
        rows.append(
            {
                "dim": dim,
                "detector": det,
                "symbol": sym,
                "cell": val,
                "n": len(g),
                "hit_rate": round(float((g["ret_signed"] > 0).mean()), 3),
                "mean_ret_bp": round(float(g["ret_signed"].mean() * 1e4), 2),
                "t_daily": round(t, 2),
                "days": nd,
                "mean_pnl_fee_adj_usd": round(float(g["pnl_fee_adj"].mean()), 2),
            }
        )
    return pd.DataFrame(rows), total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=5)
    parser.add_argument("--min-n", type=int, default=200)
    parser.add_argument("--out", default="docs/research/CONDITIONAL_TABLE.md")
    args = parser.parse_args(argv)
    root = get_settings().zr_data_root
    labels = pd.read_parquet(root / "research" / "w2" / "labels.parquet")
    labels = labels[labels["H"] == args.H]
    feats = {}
    for sym in labels["symbol"].unique():
        df = load_bars(root, sym.split(".")[0])
        feats[sym] = compute_features(df)
    df = attach_conditions(labels, feats)
    dims = ["session", "rv_tercile", "gap_dir", "with_day_trend", "weekday"]
    tables, totals = [], {}
    for d in dims:
        t, n = cell_table(df, d, args.min_n)
        tables.append(t)
        totals[d] = n
    tab = pd.concat(tables, ignore_index=True)
    n_cells = sum(totals.values())
    top = tab.sort_values("t_daily", ascending=False).head(20)
    lines = [
        "# 条件分层表（R-012）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"标签 H={args.H}（triple-barrier，同 EVENT_LABEL_TABLE）",
        f"格子总数 {n_cells}（{', '.join(f'{k} {v}' for k, v in totals.items())}）；"
        f"只报告样本 ≥ {args.min_n} 的格子，共 {len(tab)} 个。",
        f"多重比较：{n_cells} 个格子下按 Bonferroni 的 t 阈值 ≈ 3.9（α=5%）；t_daily 为日聚合 t。",
        "波动分位用事件前 30 分钟实现波动相对前一日的比值（无前视），不是全天波动。",
        "",
        "## t_daily 最高的 20 个格子",
        "",
        df_to_md(top),
        "",
    ]
    for d in dims:
        sub = tab[tab["dim"] == d].drop(columns=["dim"])
        lines += [f"## 按 {d}（{totals[d]} 格，报告 {len(sub)}）", "", df_to_md(sub), ""]
    lines += [
        "## 复现",
        "",
        "```bash",
        f"uv run python -m zr.research.conditional --H {args.H} --min-n {args.min_n}",
        "```",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {args.out}：格子总数 {n_cells}，报告 {len(tab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
