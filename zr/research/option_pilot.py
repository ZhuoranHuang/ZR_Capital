"""期权特征流水线试运行（PLAN_W3 线 a，验收 1）：只验证流水线，不下研究结论。

在有 ThetaData 的日期上构建「事件 × 特征（含 opt_*）」矩阵，报告覆盖率、分布与和标签的秩相关。
运行：uv run python -m zr.research.option_pilot --symbol SPY.US --start 2026-09-01 --end 2026-10-21
输出：docs/research/OPTION_FEATURES_PILOT.md，ZR_DATA_ROOT/research/w3/dataset_pilot.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from zr.config import get_settings
from zr.features.option_features import OPTION_FEATURE_COLUMNS, option_days
from zr.ops.markdown import df_to_md
from zr.research.dataset import build_symbol
from zr.research.events import load_config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SPY.US")
    p.add_argument("--start", default="2026-09-01")
    p.add_argument("--end", default="2026-10-21")
    p.add_argument("--config", default="config/research_w2.yaml")
    p.add_argument("--out", default="docs/research/OPTION_FEATURES_PILOT.md")
    a = p.parse_args(argv)
    root = get_settings().zr_data_root
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg.update({"start": a.start, "end": a.end, "with_options": True})
    det_cfg = load_config(Path(cfg["detectors_config"]))
    und = a.symbol.split(".")[0]

    t0 = time.monotonic()
    ds = build_symbol(a.symbol, root, det_cfg, cfg)
    secs = time.monotonic() - t0
    store = root / "research" / "w3"
    store.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(store / "dataset_pilot.parquet", index=False)

    s, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    odays = [d for d in option_days(root, und) if s <= d <= e]
    has_opt = ds["day"].isin(odays)
    sub = ds[has_opt]
    cov = pd.DataFrame(
        {
            "feature": OPTION_FEATURE_COLUMNS,
            "non_null_in_option_days": [
                round(float(sub[c].notna().mean()), 3) for c in OPTION_FEATURE_COLUMNS
            ],
            "non_null_other_days": [
                round(float(ds.loc[~has_opt, c].notna().mean()), 3) if (~has_opt).any() else np.nan
                for c in OPTION_FEATURE_COLUMNS
            ],
        }
    )
    desc = (
        sub[OPTION_FEATURE_COLUMNS].describe(percentiles=[0.1, 0.5, 0.9]).T.round(4).reset_index()
    )
    desc = desc.rename(columns={"index": "feature"})
    rows = []
    for c in OPTION_FEATURE_COLUMNS:
        for h in (5, 15):
            m = sub[[c, f"ret_h{h}"]].dropna()
            if len(m) < 30 or m[c].nunique() < 3:
                continue
            rho, pval = spearmanr(m[c], m[f"ret_h{h}"])
            rows.append(
                {
                    "feature": c,
                    "H": h,
                    "n": len(m),
                    "spearman": round(float(rho), 4),
                    "p": round(float(pval), 4),
                }
            )
    corr = pd.DataFrame(rows)
    lines = [
        "# 期权特征流水线试运行（PLAN_W3 线 a 验收 1）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · {a.symbol} · "
        f"{a.start}～{a.end} · 构建耗时 {secs:.0f}s",
        f"有期权数据的交易日 {len(odays)} 个"
        f"（{odays[0] if odays else '-'} ~ {odays[-1] if odays else '-'}）；"
        f"窗口内事件 {len(ds)} 行，其中落在期权日的 {int(has_opt.sum())} 行。",
        "**只验证流水线是否正确**：22 天样本太小，下表的相关系数不构成任何研究结论；"
        "全量数据到位后在 2 年数据上重跑元标签"
        "（`uv run python -m zr.research.dataset` + `meta_model`）。",
        "opt_near_vol_ratio20 需要前 5 个交易日才有值，所以期权日的前 5 天为 NaN。",
        "已知问题：opt_oi_chg 均值 +40% 且从不为负——DTE≤5 窗口内的合约越接近到期 OI 越大，"
        "「同一批合约日变化」主要反映这一期限效应而非持仓变化；"
        "建议改为同期限桶对比（PLAN_W3 线 a）。"
        "0DTE 的 ATM IV 在临近收盘时随剩余时间 → 0 急剧放大（最大值 >1），属正常现象。",
        "",
        "## 覆盖率",
        "",
        df_to_md(cov),
        "",
        "## 分布（期权日内的事件）",
        "",
        df_to_md(desc),
        "",
        "## 与方向化收益的 Spearman（仅作流水线健全性检查）",
        "",
        df_to_md(corr),
        "",
        "## 复现",
        "",
        "```bash",
        f"uv run python -m zr.research.option_pilot --symbol {a.symbol} "
        f"--start {a.start} --end {a.end}",
        "```",
        "",
    ]
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {a.out}：事件 {len(ds)}，期权日事件 {int(has_opt.sum())}，耗时 {secs:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
