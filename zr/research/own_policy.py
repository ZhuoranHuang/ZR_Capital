"""负责人打法提取（METHOD_V3 第 7 节）→ docs/research/OWN_POLICY_EXTRACTED.md。

- 样本：负责人 157 笔往返（zr/journal/analyze_history.fifo_round_trips）中有正股 1 分钟 K 的 MU /
  QQQ。
  入场时刻 → 决策 K = 入场前最后一根已收盘的 2 分钟 K（收盘分钟 ≤ 入场分钟）。同一 (日, K, 方向)
  只算一次。
- 正样本：上述决策 K；未标注样本：同日同标的其余 2 分钟 K。按 1:20 负采样训练「入场倾向」分类器
  （HistGradientBoosting，按日留一交叉验证 AUC），call 与 put 分开。
- 规则：深度 ≤ 3 的决策树（class_weight=balanced）；取正样本占比最高、且含 ≥ 5 个正样本的叶子，
  路径条件
  即规则；规则在两年 MU 2 分钟 K 上作为入场信号，用 v3 引擎（zr/research/sim_v3.py）全出场组合回放，
  与第 6 节网格对比。
- 状态分类：157 笔按决策 K 的状态层标签（第 2 节）分组，报告笔数 / 胜率 / 扣费后盈亏（实际成交）。
运行：uv run python -m zr.research.own_policy
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from zr.config import get_settings
from zr.features.bars import ET, load_bars, resample_bars
from zr.features.option_features import option_days
from zr.features.regime import STATE_NAMES, indicators, load_config, regimes, state_features
from zr.journal.analyze_history import fifo_round_trips, load
from zr.ops.markdown import df_to_md
from zr.research import grid_v3
from zr.research import patterns_v3 as pt
from zr.research import sim_v3 as sv

log = logging.getLogger("zr.research.own_policy")
PERIOD = 2
NEG_RATIO = 20


def feature_frame(
    bars: pd.DataFrame, x: pd.DataFrame, reg: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    f = state_features(bars, x, reg)
    f["pctb"] = x["pctb"].to_numpy()
    f["bw"] = x["bw"].to_numpy()
    f["slope"] = x["slope"].to_numpy()
    f["dif_n"] = (x["dif"] / bars["close"]).to_numpy()
    f["ret_1"] = bars["close"].pct_change().to_numpy()
    f["ret_5"] = bars["close"].pct_change(5).to_numpy()
    for c in pt.entry_configs(cfg):
        if c["entry"] == "E1" and (c["N"], c["M"]) != tuple(cfg["regime"]["channel"]["default"]):
            continue
        f[f"sig_{pt.entry_name(c)}"] = pt.entry_signal(bars, x, reg, c, cfg)
    et = bars.index.tz_convert(ET)
    f["minute"] = (et.hour * 60 + et.minute - 570).to_numpy()
    return f


def owner_trades(root: Path) -> pd.DataFrame:
    _, ex, ch = load(root)
    rt, _ = fifo_round_trips(ex, ch)
    rt["entry_ts"] = pd.to_datetime(rt["entry_et"].astype(str))
    rt["day"] = rt["entry_ts"].dt.date
    rt["dir"] = np.where(rt["right"] == "C", 1, -1)
    rt["entry_min"] = rt["entry_ts"].dt.hour * 60 + rt["entry_ts"].dt.minute - 570
    return rt


def decision_bars(rt: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    """每笔 → 决策 K 在 bars 中的位置（收盘分钟 ≤ 入场分钟的最后一根同日 K）；无则 −1。"""
    et = bars.index.tz_convert(ET)
    day = np.asarray(et.date)
    cm = (et.hour * 60 + et.minute - 570).to_numpy() + PERIOD
    out = []
    for d, m in zip(rt["day"], rt["entry_min"], strict=True):
        idx = np.flatnonzero((day == d) & (cm <= m))
        out.append(int(idx[-1]) if len(idx) else -1)
    return pd.Series(out, index=rt.index)


def tree_rule(tree: DecisionTreeClassifier, cols: list[str], min_pos: int = 5):
    """正样本占比最高的叶子（≥ min_pos 个正样本）→ [(列, '<='/'>', 阈值)]、叶子统计。"""
    t = tree.tree_
    best, best_path, best_stat = -1.0, None, None

    def walk(node, path):
        nonlocal best, best_path, best_stat
        if t.children_left[node] == -1:
            v = t.value[node][0]
            w = t.weighted_n_node_samples[node]
            pos_share = v[1] / v.sum() if v.sum() else 0
            n_pos = t.n_node_samples[node] * pos_share
            if n_pos >= min_pos and pos_share > best:
                best, best_path, best_stat = pos_share, list(path), (int(t.n_node_samples[node]), w)
            return
        f, thr = cols[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], path + [(f, "<=", thr)])
        walk(t.children_right[node], path + [(f, ">", thr)])

    walk(0, [])
    return best_path, best, best_stat


def apply_rule(f: pd.DataFrame, path) -> np.ndarray:
    m = np.ones(len(f), dtype=bool)
    for col, op, thr in path:
        v = np.nan_to_num(f[col].to_numpy(float), nan=-999.0)  # 与训练树时的缺失值填充一致
        with np.errstate(invalid="ignore"):
            m &= (v <= thr) if op == "<=" else (v > thr)
    return m


def run(und: str = "MU") -> str:
    cfg = load_config()
    root = get_settings().zr_data_root
    rt = owner_trades(root)
    L = [
        "# 负责人打法提取（METHOD_V3 第 7 节）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.research.own_policy`",
        "",
    ]
    # ---- 157 笔按状态分类 ----
    cls_rows = []
    for u in sorted(rt["und"].unique()):
        sub = rt[rt["und"] == u]
        try:
            stock = load_bars(root, u)
        except (FileNotFoundError, ValueError):
            stock = pd.DataFrame()
        if stock.empty:
            for _, r in sub.iterrows():
                cls_rows.append(
                    {"und": u, "state": "未覆盖（无正股 1 分钟 K）", "pnl": r["pnl_net"]}
                )
            continue
        bars = resample_bars(stock, f"{PERIOD}min")
        x = indicators(bars, cfg)
        reg = regimes(bars, x, cfg, PERIOD)
        pos = decision_bars(sub, bars)
        for (_, r), p in zip(sub.iterrows(), pos, strict=True):
            if p < 0:
                st = "未覆盖（入场在首根 K 收盘前）"
            else:
                s, d = int(reg["state"].iat[p]), int(reg["dir"].iat[p])
                st = STATE_NAMES[s] + ("" if d == 0 else (" 同向" if d == r["dir"] else " 反向"))
            cls_rows.append({"und": u, "state": st, "pnl": r["pnl_net"]})
    cls = pd.DataFrame(cls_rows)
    g = cls.groupby("state")["pnl"]
    tab = pd.DataFrame(
        {
            "n": g.size(),
            "win_rate": g.apply(lambda s: round(float((s > 0).mean()), 3)),
            "sum_pnl": g.sum().round(2),
            "mean_pnl": g.mean().round(2),
        }
    ).reset_index()
    L += [
        "## 157 笔按状态分类（决策 K = 入场前最后一根已收盘 2 分钟 K；「同向 / 反向」= 状态方向与"
        "所买 call / put 方向是否一致；盈亏为实际成交扣费后）",
        "",
        df_to_md(tab),
        "",
        "边界：157 笔 = 负责人 2026-06-09～2026-08-18 的全部期权往返（FIFO 配对）；只有 MU / QQQ "
        "有正股 1 分钟 K；样本小，分组为描述性。",
        "",
    ]
    # ---- 倾向模型与规则（MU） ----
    stock = load_bars(root, und)
    bars = resample_bars(stock, f"{PERIOD}min")
    x = indicators(bars, cfg)
    reg = regimes(bars, x, cfg, PERIOD)
    f = feature_frame(bars, x, reg, cfg)
    sub = rt[rt["und"] == und].copy()
    sub["pos"] = decision_bars(sub, bars).to_numpy()
    sub = sub[sub["pos"] >= 0]
    day = np.asarray(bars.index.tz_convert(ET).date)
    cols = [c for c in f.columns if c != "minute"] + ["minute"]
    rng = np.random.default_rng(0)
    rules = {}
    for d, name in ((1, "call"), (-1, "put")):
        p_idx = sorted(set(sub.loc[sub["dir"] == d, "pos"]))
        if len(p_idx) < 10:
            L += [f"### {name}：正样本 {len(p_idx)} 个，不足 10，跳过", ""]
            continue
        pdays = sorted({day[i] for i in p_idx})
        unl = np.flatnonzero(np.isin(day, pdays))
        unl = np.setdiff1d(unl, p_idx)
        neg = rng.choice(unl, size=min(len(unl), NEG_RATIO * len(p_idx)), replace=False)
        idx = np.r_[p_idx, neg]
        y = np.r_[np.ones(len(p_idx)), np.zeros(len(neg))]
        X = f.iloc[idx][cols].astype(float).to_numpy()
        dd = day[idx]
        oof = np.full(len(y), np.nan)
        for dv in pdays:
            te = dd == dv
            if y[~te].sum() < 3:
                continue
            m = HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                class_weight="balanced",
                random_state=0,
            )
            m.fit(X[~te], y[~te])
            oof[te] = m.predict_proba(X[te])[:, 1]
        ok = np.isfinite(oof)
        auc = roc_auc_score(y[ok], oof[ok]) if ok.any() and len(set(y[ok])) == 2 else np.nan
        tree = DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", min_samples_leaf=5, random_state=0
        )
        Xf = np.nan_to_num(X, nan=-999.0)
        tree.fit(Xf, y)
        path, share, stat = tree_rule(tree, cols)
        rules[name] = (d, path)
        rule_txt = " 且 ".join(f"{c} {op} {thr:.4g}" for c, op, thr in path) if path else "无"
        L += [
            f"### {name}（正样本 {len(p_idx)} 个决策 K，{len(pdays)} 天；负采样 1:{NEG_RATIO}）",
            "",
            f"倾向模型按日留一 AUC：{auc:.3f}。深度 ≤ 3 决策树最佳叶子：正样本占比（加权）"
            f"{share:.3f}，"
            f"叶子样本 {stat[0] if stat else 0}。",
            "",
            f"规则：**{rule_txt}**",
            "",
        ]
    # ---- 规则回放 ----
    if rules:

        def mk(d, path):
            def fn(b, xx, rr):
                ff = feature_frame(b, xx, rr, cfg)
                return np.where(apply_rule(ff, path), d, 0).astype(np.int8)

            return fn

        extra = {f"OWN_{k}": mk(d, p) for k, (d, p) in rules.items() if p}
        days_opt = option_days(root, und)
        res = sv.run_grid(root, und, stock, cfg, [PERIOD], days_opt, extra_entries=extra)
        lt = grid_v3.long_table_any(res, cfg, "6,13,6", list(extra))
        grid = pd.read_parquet(root / "research" / "v3" / f"grid_{und}.parquet")
        gp = grid[(grid["period"] == PERIOD) & (grid["macd"] == "6,13,6") & (grid["n"] >= 30)]
        comp = []
        for k in extra:
            r = lt[(lt["entry"] == k) & (lt["n"] >= 30)]
            side = "long" if rules[k[4:]][0] == 1 else "short"
            g2 = gp[gp["side"] == side]
            comp.append(
                {
                    "rule": k,
                    "side": side,
                    "rows_n_ge_30": len(r),
                    "events_max_n": int(lt.loc[lt["entry"] == k, "n"].max()) if len(lt) else 0,
                    "median_per_contract": round(float(r["mean_net"].median() / 2), 2)
                    if len(r)
                    else np.nan,
                    "best_per_contract": round(float(r["mean_net"].max() / 2), 2)
                    if len(r)
                    else np.nan,
                    "rows_mean_gt_0": int((r["mean_net"] > 0).sum()),
                    "grid_median_per_contract": round(float(g2["mean_net"].median() / 2), 2),
                    "grid_best_per_contract": round(float(g2["mean_net"].max() / 2), 2),
                }
            )
        L += [
            "## 提取规则在两年 MU 上的真实期权回放（2 分钟，全部出场组合）"
            "vs 第 6 节网格（2 分钟、同方向、n ≥ 30）",
            "",
            df_to_md(pd.DataFrame(comp)),
            "",
            "边界：规则从 2026-07～08 的 20 个交易日学出，再放到 2023-10～2026-10 "
            "全样本回放——学习期包含在回放期内，"
            "不是独立验证；正样本只有几十个决策 K，树的阈值不稳定。成交口径同第 6 节（D-023）。"
            "",
            "",
        ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MU.US")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    text = run(a.symbol.split(".")[0])
    out = Path(__file__).resolve().parents[2] / "docs" / "research" / "OWN_POLICY_EXTRACTED.md"
    out.write_text(text, encoding="utf-8")
    log.info("写入 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
