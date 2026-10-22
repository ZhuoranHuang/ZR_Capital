"""方法论 v2 第一轮报告（METHOD_V2 规格 11）→ docs/research/METHOD_V2_ROUND1.md。

顺序：覆盖矩阵 → 回放结果 → E1 回踩遍历（D-022）→ 四条基线 → 标签与分层（R-018～R-022）
→ 模型 → 决策模拟 → 审计 → 秒级。
每节附边界声明（CLAUDE.md 规则 12）。前置：
  uv run python -m zr.research.outcomes --symbol MU.US
  uv run python -m zr.research.dataset_v2 --symbol MU.US
运行：uv run python -m zr.research.method_v2_round1 --symbol MU.US
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.calendar import trading_days
from zr.features.bars import ET, load_bars
from zr.features.option_features import option_days
from zr.ops.markdown import df_to_md
from zr.research import (
    audit_v2,
    baselines_v2,
    decision_sim,
    e1_sweep,
    model_v2,
    outcomes,
    replay_rule,
    seconds_v2,
)
from zr.research.labels_v2 import H_MIN, X_SIGMA

log = logging.getLogger("zr.research.method_v2_round1")
START, END = dt.date(2023, 10, 2), dt.date(2026, 10, 21)


def coverage(root: Path) -> pd.DataFrame:
    """每月：交易日数（交易日历）、各标的正股 1 分钟日数（RTH 过滤后实有）、期权日数。"""
    tds = trading_days(START, END)
    month = pd.Series(pd.to_datetime(tds).to_period("M").astype(str), index=tds)
    out = month.value_counts().sort_index().rename("trading_days").to_frame()
    for und in ("MU", "QQQ", "SPY"):
        try:
            b = load_bars(root, und)
            sd = set(np.asarray(b.index.tz_convert(ET).date))
        except (FileNotFoundError, ValueError):
            sd = set()
        od = set(option_days(root, und))
        out[f"{und}_stock"] = [sum(d in sd for d in month.index[month == m]) for m in out.index]
        out[f"{und}_opt"] = [sum(d in od for d in month.index[month == m]) for m in out.index]
    return out.rename_axis("month").reset_index()


def strat(ds: pd.DataFrame, key: pd.Series, name: str) -> pd.DataFrame:
    d = ds.assign(_k=key.astype(str))
    g = d.groupby("_k")
    out = (
        pd.DataFrame(
            {
                "n": g.size(),
                "ever_up_0.5_10": g["ever_up_0.5_10"].mean().round(3),
                "mfe_10_bp": (g["mfe_10"].mean() * 1e4).round(2),
                "mae_10_bp": (g["mae_10"].mean() * 1e4).round(2),
                "n_opt": g["net_cap30"].count(),
                "mean_net_cap30": g["net_cap30"].mean().round(2),
            }
        )
        .reset_index()
        .rename(columns={"_k": name})
    )
    return out


def label_overview(ds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in H_MIN:
        r = {
            "H_min": h,
            "n": int(ds[f"mfe_{h}"].notna().sum()),
            "mfe_med_bp": round(float(ds[f"mfe_{h}"].median() * 1e4), 2),
            "mae_med_bp": round(float(ds[f"mae_{h}"].median() * 1e4), 2),
        }
        for x in X_SIGMA:
            r[f"ever_up_{x}σ"] = round(float(ds[f"ever_up_{x}_{h}"].mean()), 3)
        r["tb_up"] = round(float((ds[f"tb_label_{h}"] == 1).mean()), 3)
        r["tb_dn"] = round(float((ds[f"tb_label_{h}"] == -1).mean()), 3)
        c = f"bid_max_ret_{h}"
        if c in ds:
            r["opt_bid_max_med"] = round(float(ds[c].median()), 4)
            r["opt_bid_max_ge20%"] = round(float((ds[c] >= 0.2).mean()), 3)
        rows.append(r)
    return pd.DataFrame(rows)


def _thin_note(tab: pd.DataFrame) -> str:
    """阈值组笔数 < 30 时提示样本过少（目标正例率远低于阈值时常见）。"""
    thr = tab[tab["group"].str.startswith("模型")]
    thin = thr[thr["n"].fillna(0) < 30]
    if thin.empty:
        return ""
    return (
        f"注：{'、'.join(sorted(set(thin['group'])))} 有配置笔数 < 30，样本过少，不作结论；"
        "固定阈值高于该目标的正例率时几乎不触发，参考第 4 节十分位表。\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MU.US")
    p.add_argument("--n-random", type=int, default=1000)
    p.add_argument("--out", default="docs/research/METHOD_V2_ROUND1.md")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    und = a.symbol.split(".")[0]
    t0 = time.monotonic()
    repo = Path(__file__).resolve().parents[2]

    cov = coverage(root)
    out = outcomes.build(root, und, START, END)
    lk = replay_rule.Lookup(out)
    ds = pd.read_parquet(root / "research" / "v2" / f"dataset_{und}.parquet")
    n_opt_days = len(lk.days)
    first_opt, last_opt = (min(lk.days), max(lk.days)) if n_opt_days else (None, None)
    bound = (
        f"标的 {und}；事件样本 {ds['day'].min()}～{ds['day'].max()}（RTH、交易日历过滤）；期权日 "
        f"{n_opt_days}（{first_opt}～{last_opt}）；"
        f"事件 {len(ds)}（含多检测器重复）；费用 D-013 + 买 ask 卖 bid"
    )

    # 1 回放
    rtext, rtr = replay_rule.run(root, und, START, END, a.n_random)
    (repo / "docs" / "research" / f"REPLAY_RULE_{und}.md").write_text(rtext, encoding="utf-8")
    summ = rtext.split("## 16 组配置")[1].split("## 按")[0] if "## 16 组配置" in rtext else ""
    log.info("回放完成 %.0fs", time.monotonic() - t0)

    # 1b E1 回踩深度遍历（D-022）
    stext, _, srp = e1_sweep.run(root, und, START, END, a.n_random)
    (repo / "docs" / "research" / f"E1_PULLBACK_SWEEP_{und}.md").write_text(stext, encoding="utf-8")
    scfg = e1_sweep.load_config()
    ssum = e1_sweep.summary(srp, scfg) if len(srp) else {}
    smarg = {cap: e1_sweep.marginals(srp, cap) for cap in outcomes.CAPS} if len(srp) else {}
    stop = (
        srp[srp["n"] >= 30].sort_values("mean_net", ascending=False).head(10)
        if len(srp)
        else pd.DataFrame()
    )
    log.info("D-022 遍历完成 %.0fs", time.monotonic() - t0)

    # 2 基线
    base_tabs = []
    for cap in outcomes.CAPS:
        bt, _ = baselines_v2.baseline_table(ds, lk, cap, a.n_random)
        base_tabs.append((cap, bt))
    log.info("基线完成 %.0fs", time.monotonic() - t0)

    # 3 标签与分层
    lab = label_overview(ds)
    strats = [
        (
            "R-018 共振：过去 5 分钟同向其他检测器数",
            strat(ds, ds["confl_n_same_dir_5m"].clip(upper=3), "confl_n_same_dir_5m（3=≥3）"),
        ),
        ("R-018 共振：同向家族数", strat(ds, ds["confl_families_5m"], "confl_families_5m")),
        (
            "R-019 结构：本轮通道第几次回踩",
            strat(ds, ds["pullback_idx"].clip(upper=3), "pullback_idx（3=≥3）"),
        ),
        (
            "R-019 结构：5 分钟 MACD 状态与事件方向",
            strat(ds, np.sign(ds["tf5_regime"]) * ds["direction"], "tf5×方向（1=同向）"),
        ),
        (
            "R-019 结构：通道 A 内最大回撤三分位",
            strat(
                ds,
                pd.qcut(ds["channel_max_dd"].rank(method="first"), 3, labels=["低", "中", "高"]),
                "channel_max_dd",
            ),
        ),
        (
            "R-020 日历：距下一个定时发布 ≤ 30 分钟",
            strat(ds, ds["min_to_next_release"] <= 30, "≤30 分钟"),
        ),
        (
            "R-020 日历：是否到期日",
            strat(ds, ds["is_expiry_day"], "is_expiry_day（NaN=无期权数据）"),
        ),
        (
            "R-021 跨资产：SPY/QQQ/SMH 同分钟同向个数",
            strat(ds, ds["xa_n_same_sign"], "xa_n_same_sign"),
        ),
    ]
    log.info("分层完成 %.0fs", time.monotonic() - t0)

    # 4 模型 + 5 决策模拟
    model_sections, dec_sections = [], []
    for target in model_v2.TARGETS:
        r = model_v2.run(root, und, target)
        folds = r["folds"]
        model_sections.append(
            (target, r, folds, model_v2.calibration(r["res"]), model_v2.decile_table(r["res"]))
        )
        tab, curves = decision_sim.simulate(r["res"], lk)
        png = repo / "docs" / "research" / f"decision_{und}_{target}.png"
        decision_sim.plot_curves(
            curves, png, f"{und} {target}: cumulative net option P&L by threshold"
        )
        months = decision_sim.months_of(r["res"])
        bt_oos, _ = baselines_v2.baseline_table(ds, lk, 30, a.n_random, months)
        dec_sections.append((target, tab, png.name, bt_oos, sorted(months)))
        log.info("模型 %s 完成 %.0fs", target, time.monotonic() - t0)

    # 6 审计
    planted = audit_v2.planted_signal(ds, und)
    shuffled = audit_v2.shuffled_labels(ds, und)
    checklist = audit_v2.lookahead_checklist(repo)
    manual = audit_v2.manual_recompute(root, und, ds, out)
    tf = audit_v2.timeframe_table(root, und)
    log.info("审计完成 %.0fs", time.monotonic() - t0)

    # 7 秒级
    sec_summary, _, sec_meta = seconds_v2.run(root, und)
    sec_labels = seconds_v2.second_labels(root, und, ds[["ts", "detector", "direction"]])

    L = [
        f"# 方法论 v2 第一轮（{und}）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 总耗时 "
        f"{time.monotonic() - t0:.0f}s · "
        "规格见 docs/METHOD_V2.md，假设预登记见 docs/RESEARCH_LOG.md（R-017～R-027）",
        "",
        "## 0. 覆盖矩阵（各月交易日 / 正股 1 分钟日 / 期权日）",
        "",
        df_to_md(cov),
        "",
        f"边界：{bound}。QQQ / SPY 期权未到位的月份不参与任何结论。",
        "",
        "## 1. 规则直接回放（规格 2，R-017）",
        "",
        f"完整表见 `docs/research/REPLAY_RULE_{und}.md`（DTE、时段、星期、D-001、DTE 0–1 时代、"
        "通道三步）。16 组汇总：",
        summ.strip(),
        "",
        f"边界：{bound}；随机对照 = 同日同笔数、同方向同持仓上限的随机决策分钟，{a.n_random} 次。",
        "",
    ]
    L += [
        "## 1b. E1 回踩深度遍历（D-022，R-029）",
        "",
        f"完整表见 `docs/research/E1_PULLBACK_SWEEP_{und}.md`（事件表 + 回放全表）。网格 "
        "config/e1_sweep.yaml；试验计数与 Bonferroni 门槛见要点表。",
        "",
    ]
    if ssum:
        L += ["要点：", "", df_to_md(pd.DataFrame([ssum])), ""]
        for cap, m in smarg.items():
            L += [f"回放边际（持仓上限 {cap}，按维度取中位数）：", "", df_to_md(m), ""]
        L += ["回放前 10 行（n ≥ 30，按 mean_net）：", "", df_to_md(stop), ""]
    else:
        L += [f"待 {und} 全量（无期权数据）。", ""]
    L += [
        f"边界：{bound}；回踩信号定义见 zr/features/pullback.py（深度 %b / ε 二选一、"
        "至少 n 根同日回调后收阳、布林值取信号 bar）；随机对照同第 1 节。",
        "",
        "## 2. 四条基线（规格 8，R-025）",
        "",
    ]
    for cap, bt in base_tabs:
        L += [f"### 持仓上限 {cap} 分钟（全部期权日）", "", df_to_md(bt), ""]
    L += [
        f"边界：{bound}；日历入场只有 RTH 内的发布（FOMC 14:00 与个别 PCE 10:00），CPI / "
        "非农待接入（config/calendar.yaml pending）。",
        "",
    ]
    L += [
        "## 3. 标签层概览与预登记分层（规格 5，R-018～R-022）",
        "",
        "### 标签随 H 的变化（全部事件）",
        "",
        df_to_md(lab),
        "",
    ]
    for title, t in strats:
        L += [f"### {title}", "", df_to_md(t), ""]
    L += [
        f"边界：{bound}；ever_up / MFE 用正股 1 分钟高低点（日内）；net_cap30 只在期权日有值；"
        "分层为描述性，未做多重比较修正。",
        "",
    ]
    L += [
        "## 4. 模型（规格 6，R-023）",
        "",
        "GBDT 参数固定（叶子 31、300 轮、学习率 0.03，不调参），唯一性加权，训练 12 月 / 测试 1 月 "
        "/ embargo 1 天滚动，**无验证集**。",
        "",
    ]
    for target, r, folds, cal, dec in model_sections:
        auc_m = folds["auc"] if len(folds) else pd.Series(dtype=float)
        L += [
            f"### 目标 {target}（后端 {r['backend']}，特征 {r['n_features']} 个，正例率 "
            f"{r['base_rate']:.3f}）",
            "",
            f"事件 {r['n']}，样本外 {r['n_oos']}（{len(folds)} 个测试月），**样本外 AUC "
            f"{r['auc']:.4f}**；"
            f"逐月 AUC 均值 {auc_m.mean():.4f}，>0.5 的月份 {(auc_m > 0.5).mean():.0%}。",
            "",
            "十分位（样本外）：",
            "",
            df_to_md(dec),
            "",
            "校准：",
            "",
            df_to_md(cal),
            "",
            "逐月 AUC：",
            "",
            df_to_md(folds),
            "",
        ]
    L += [f"边界：{bound}；样本外月份见逐月表。", ""]
    L += ["## 5. 决策模拟（规格 7，R-024）", ""]
    for target, tab, png, bt_oos, months in dec_sections:
        L += [
            f"### 目标 {target}（样本外 {len(months)} 个月："
            f"{months[0] if months else '-'}～{months[-1] if months else '-'}）",
            "",
            df_to_md(tab),
            "",
            _thin_note(tab),
            f"![decision](./{png})",
            "",
            "同一批月份的四条基线（持仓上限 30）：",
            "",
            df_to_md(bt_oos),
            "",
        ]
    L += [
        "「模型过滤有没有附加价值」= 阈值组的 mean_net 与无模型组之差（delta_mean_vs_no_model 列）"
        "；"
        "附加价值为正但绝对值仍为负时，模型只是少亏，不构成可交易的边。",
        "",
        f"边界：{bound}；只在样本外月份。",
        "",
    ]
    L += [
        "## 6. 审计（规格 10，R-026）",
        "",
        "### 植入信号（随机 5% 事件 MFE_10 +20 bp，附指示特征；最近 8 个测试月）",
        "",
        df_to_md(pd.DataFrame([planted])),
        "",
        "### 标签打乱（月内打乱 ever_up_0.5_10；最近 8 个测试月）",
        "",
        df_to_md(pd.DataFrame([shuffled])),
        "",
        "### 前视检查清单（实际运行 pytest）",
        "",
        df_to_md(checklist),
        "",
        "### 手工复算 10 个事件（独立读原始 parquet）",
        "",
        df_to_md(manual),
        "",
        "### 1 / 2 / 5 分钟分周期（E1T：通道 A 回踩，10 分钟方向化收益）",
        "",
        df_to_md(tf),
        "",
        f"边界：{bound}。",
        "",
    ]
    L += [
        "## 7. 秒级（规格 10，R-027）",
        "",
        f"录制器数据：{sec_meta}。**只有 1 个不完整交易日（约 78 分钟），只验证流水线，不下结论。"
        "**",
        "",
        df_to_md(sec_summary),
        "",
        f"数据集事件在录制日的 10 / 30 秒标签：样本 {len(sec_labels)}"
        + (
            f"；ret_10s 均值 {sec_labels['ret_10s'].mean() * 1e4:.2f} bp、ret_30s 均值 "
            f"{sec_labels['ret_30s'].mean() * 1e4:.2f} bp"
            if len(sec_labels)
            else ""
        )
        + "。",
        "",
        "边界：MU 2026-10-21 14:46–16:05 ET；长桥 Depth 一档、非 NBBO；时间为本地接收时间。",
        "",
    ]
    L += [
        "## 复现",
        "",
        "```bash",
        f"uv run python -m zr.research.outcomes --symbol {und}.US",
        f"uv run python -m zr.research.dataset_v2 --symbol {und}.US",
        f"uv run python -m zr.research.method_v2_round1 --symbol {und}.US",
        "```",
        "",
    ]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    log.info("写入 %s，总耗时 %.0fs", a.out, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
