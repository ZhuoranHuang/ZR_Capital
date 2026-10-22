"""V4 补充 H5：元标签模型（RESEARCH_LOG R-034；参数 config/v4_h5.yaml）→ docs/research/V4_H5.md。

事件、合约、tick 成交、离场与 R-033（zr/research/v4_decisive.py）完全相同，只读已缓存的 tick，
不再请求。每个（口径 × 标的 × 离场组合）一个 HistGradientBoosting，扩展窗口滚动、按月样本外。
运行：uv run python -m zr.research.v4_h5
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.calendar_features import compute_calendar
from zr.features.channel import channel_states
from zr.features.cross_asset import compute_cross_asset
from zr.features.features import compute_features
from zr.features.option_features import compute_option_features
from zr.features.pullback import depth_features, directional
from zr.features.regime import CHANNEL, IGNITION, indicators, regimes, state_features
from zr.features.structure import compute_structure
from zr.ops.markdown import df_to_md
from zr.research import v4_decisive as v4d
from zr.research.dataset_v2 import CONFIG as EV2_CONFIG
from zr.research.dataset_v2 import feature_columns
from zr.research.events import confluence, signal_matrix
from zr.research.events import load_config as load_ev2

log = logging.getLogger("zr.research.v4_h5")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "v4_h5.yaml"
OUT = REPO / "docs" / "research" / "V4_H5.md"
EXITS = ("a", "b", "c")


def load_h5(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------- 事件与标签 ----------------


def event_outcomes(root: Path, und: str, conv: str, v4: dict):
    """H1–H3 全部信号行 × 所属假设，三种离场组合的 tick 级结果（不贪心）。"""
    ev, aux = v4d.build_events(root, und, conv, v4)
    trig = aux["trig"]
    sig = ev[ev[[f"hyp_{h}" for h in v4d.HYPS]].any(axis=1)].reset_index(drop=True)
    sig["decision_ns"] = v4d.ns(sig["decision_ts"])
    sig["flat_ns"] = v4d.ns(sig["flat_ts"])
    cache: dict = {}
    rows = []
    for _, r in sig.iterrows():
        key = (r["symbol"], r["day"])
        if key not in cache:
            cache[key] = v4d.load_ticks(root, und, *key)
        tk = cache[key]
        if tk is None:
            continue
        for h in v4d.HYPS:
            if not r[f"hyp_{h}"]:
                continue
            out = {
                "day": r["day"],
                "bar_i": int(r["bar_i"]),
                "decision_ns": int(r["decision_ns"]),
                "direction": int(r["direction"]),
                "dte": int(r["dte"]),
                "hyp": h,
                **{f"hyp_{k}": bool(r[f"hyp_{k}"]) for k in v4d.HYPS},
            }
            ok = True
            for x in EXITS:
                res = v4d.simulate(
                    tk,
                    int(r["decision_ns"]),
                    trig[v4["exits"][x][h]][int(r["direction"])],
                    int(r["flat_ns"]),
                    v4,
                )
                if res is None:
                    ok = False
                    break
                out.update(
                    {
                        f"net_usd_{x}": res["net_usd"],
                        f"net_pct_{x}": res["net_pct"],
                        f"entry_ns_{x}": res["entry_ns"],
                        f"exit_ns_{x}": res["exit_ns"],
                    }
                )
            if ok:
                rows.append(out)
    return pd.DataFrame(rows)


# ---------------- 特征 ----------------


def v2_frame(root: Path, und: str, stock: pd.DataFrame, window: int) -> pd.DataFrame:
    """v2 全部特征（1 分钟 K），共振窗口改为 window 根 1 分钟 K。"""
    feats = compute_features(stock)
    chans = channel_states(stock, feats)
    struct = compute_structure(stock, feats, chans)
    xa = compute_cross_asset(stock, und, root)
    cal = compute_calendar(stock.index, root, und)
    opt = compute_option_features(root, und, stock)
    sig = signal_matrix(stock, feats, load_ev2(EV2_CONFIG), extra={"_chans": chans})
    conf = confluence(sig, window)
    allf = pd.concat([feats, chans, struct, opt, cal, xa, conf], axis=1)
    return allf.loc[:, ~allf.columns.duplicated()]


def transition_features(b: pd.DataFrame, reg: pd.DataFrame, reset: bool) -> pd.DataFrame:
    """上一状态及方向、IGNITION→CHANNEL 根数、MA5/MA10 位置（2 分钟 K；只用 ≤t）。"""
    st, d = reg["state"].to_numpy(), reg["dir"].to_numpy()
    day = np.asarray(b.index.tz_convert(ET).date)
    n = len(st)
    prev_s = np.full(n, np.nan)
    prev_d = np.full(n, np.nan)
    i2c = np.full(n, np.nan)
    run_start, last_ign = 0, -1
    ps, pd_ = np.nan, np.nan
    for i in range(n):
        if i and (st[i] != st[i - 1] or d[i] != d[i - 1] or day[i] != day[i - 1]):
            ps, pd_ = (st[i - 1], d[i - 1]) if day[i] == day[i - 1] else (np.nan, np.nan)
            run_start = i
        if i and day[i] != day[i - 1]:
            last_ign = -1
        prev_s[i], prev_d[i] = ps, pd_
        if st[i] == CHANNEL and 0 <= last_ign < run_start:
            i2c[i] = run_start - last_ign
        if st[i] == IGNITION:
            last_ign = i
    c = b["close"]
    if reset:
        g = c.groupby(day)
        ma5 = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
        ma10 = g.transform(lambda s: s.rolling(10, min_periods=10).mean())
    else:
        ma5, ma10 = c.rolling(5, min_periods=5).mean(), c.rolling(10, min_periods=10).mean()
    return pd.DataFrame(
        {
            "prev_state": prev_s,
            "prev_dir": prev_d,
            "ign_to_chan_bars": i2c,
            "ma5_pos": ((c - ma5) / c).to_numpy(),
            "ma10_pos": ((c - ma10) / c).to_numpy(),
        },
        index=b.index,
    )


def build_features(
    root: Path,
    und: str,
    conv: str,
    v4: dict,
    h5: dict,
    ev: pd.DataFrame,
    v2: pd.DataFrame,
    depth: pd.DataFrame,
) -> pd.DataFrame:
    c = v4["conventions"][conv]
    cfg = v4d.state_cfg(v4, conv)
    days = v4d.sample_days(root, und, v4)
    b = v4d.build_bars(root, und, c, max(days))
    x = indicators(b, cfg, daily_reset=c["reset"], warmup=c["warmup"])
    reg = regimes(b, x, cfg, v4["period"])
    sf = state_features(b, x, reg)
    tf = transition_features(b, reg, c["reset"])
    pos = ev["bar_i"].to_numpy()
    S = pd.concat([sf, tf], axis=1).iloc[pos].reset_index(drop=True)
    ts1 = pd.DatetimeIndex(pd.to_datetime(ev["decision_ns"], utc=True)) - pd.Timedelta(minutes=1)
    cols = [
        k
        for k in feature_columns(und)
        if k in v2.columns and not k.startswith("confl_") and not k.startswith("pb_")
    ]
    F = v2[cols].reindex(ts1).reset_index(drop=True)
    up = ev["direction"].to_numpy() == 1
    conf = v2[["confl_n_up", "confl_n_dn", "confl_fam_up", "confl_fam_dn"]].reindex(ts1)
    F["confl_n_same_dir"] = np.where(up, conf["confl_n_up"], conf["confl_n_dn"])
    F["confl_families"] = np.where(up, conf["confl_fam_up"], conf["confl_fam_dn"])
    pb = directional(depth, pd.Series(ts1), ev["direction"].to_numpy()).reset_index(drop=True)
    E = pd.DataFrame(
        {
            "direction": ev["direction"].to_numpy(),
            "dte": ev["dte"].to_numpy(),
            "hyp_row": ev["hyp"].map({"H1": 1, "H2": 2, "H3": 3}).to_numpy(),
            **{f"hyp_{h}": ev[f"hyp_{h}"].astype(int).to_numpy() for h in v4d.HYPS},
        }
    )
    return pd.concat([F, pb, S, E], axis=1)


# ---------------- 训练与评估 ----------------


def make_model(h5: dict) -> HistGradientBoostingClassifier:
    m = h5["model"]
    return HistGradientBoostingClassifier(
        max_leaf_nodes=m["max_leaf_nodes"],
        max_iter=m["max_iter"],
        learning_rate=m["learning_rate"],
        early_stopping=m["early_stopping"],
        random_state=m["random_state"],
    )


def walk_forward(X: np.ndarray, y: np.ndarray, day: pd.Series, h5: dict):
    """扩展窗口：训练 = 测试月首日 − embargo 之前的全部样本，且样本日数 ≥ min_train_days。"""
    wf = h5["walk_forward"]
    d = pd.to_datetime(day.astype(str))
    month = d.dt.to_period("M")
    p = np.full(len(y), np.nan)
    folds = []
    for m in sorted(month.unique()):
        tr = (d < m.start_time - pd.Timedelta(days=wf["embargo_days"])).to_numpy()
        te = (month == m).to_numpy()
        if d[tr].dt.date.nunique() < wf["min_train_days"] or len(set(y[tr])) < 2 or not te.any():
            continue
        mdl = make_model(h5).fit(X[tr], y[tr])
        p[te] = mdl.predict_proba(X[te])[:, 1]
        folds.append(
            {
                "month": str(m),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "model": mdl,
                "te": te,
            }
        )
    return p, folds


def mda(X: np.ndarray, y: np.ndarray, folds: list, cols: list[str], seed: int = 0) -> pd.Series:
    """置换重要性：每折把一列打乱后样本外 AUC 的下降，按测试样本数加权平均。"""
    rng = np.random.default_rng(seed)
    drops = np.zeros(len(cols))
    w = 0
    for f in folds:
        te = f["te"]
        if len(set(y[te])) < 2:
            continue
        Xt = X[te].copy()
        base = roc_auc_score(y[te], f["model"].predict_proba(Xt)[:, 1])
        for j in range(len(cols)):
            keep = Xt[:, j].copy()
            Xt[:, j] = rng.permutation(keep)
            drops[j] += te.sum() * (base - roc_auc_score(y[te], f["model"].predict_proba(Xt)[:, 1]))
            Xt[:, j] = keep
        w += te.sum()
    return pd.Series(drops / max(w, 1), index=cols).sort_values(ascending=False)


def simulate_filter(o: pd.DataFrame, x: str, v4: dict) -> dict:
    """一个账户（标的 × 离场）按时间贪心执行；返回 v4d.cell_stats。"""
    t = o.rename(
        columns={
            f"net_usd_{x}": "net_usd",
            f"net_pct_{x}": "net_pct",
            f"entry_ns_{x}": "entry_ns",
            f"exit_ns_{x}": "exit_ns",
        }
    )
    t = t.drop_duplicates(["decision_ns", "direction"])  # 同一 K 同方向只下一单
    t = v4d.greedy(t) if len(t) else t
    return (
        v4d.cell_stats(t, v4)
        if len(t)
        else {"n": 0, "mean_pct": np.nan, "p": 1.0, "mean_pct_ex_top5": np.nan}
    )


def rule_cards(o: pd.DataFrame, cols: list[str], x: str, h5: dict) -> pd.DataFrame:
    e = h5["eval"]
    tgt = (o["p"] >= e["tree_target_prob"]).astype(int).to_numpy()
    if tgt.sum() < e["tree_min_leaf"] or tgt.sum() == len(tgt):
        return pd.DataFrame()
    Xo = np.nan_to_num(o[cols].astype(float).to_numpy(), nan=-999.0)
    tree = DecisionTreeClassifier(
        max_depth=e["tree_depth"], min_samples_leaf=e["tree_min_leaf"], random_state=0
    ).fit(Xo, tgt)
    t = tree.tree_
    leaf = tree.apply(Xo)
    rows = []

    def walk(node, path):
        if t.children_left[node] == -1:
            m = leaf == node
            share = tgt[m].mean() if m.any() else 0
            if share > 0.5:
                s = o[m]
                rows.append(
                    {
                        "规则": " 且 ".join(f"{c} {op} {v:.4g}" for c, op, v in path),
                        "事件数": int(m.sum()),
                        "高概率占比": round(float(share), 2),
                        "胜率": round(float((s[f"net_usd_{x}"] > 0).mean()), 3),
                        "扣费后均值%": round(float(s[f"net_pct_{x}"].mean()), 2),
                    }
                )
            return
        c, thr = cols[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], path + [(c, "<=", thr)])
        walk(t.children_right[node], path + [(c, ">", thr)])

    walk(0, [])
    return pd.DataFrame(rows)


def run_one(root: Path, und: str, conv: str, v4: dict, h5: dict, v2, depth) -> dict:
    ev = event_outcomes(root, und, conv, v4)
    F = build_features(root, und, conv, v4, h5, ev, v2, depth)
    cols = [c for c in F.columns if F[c].notna().any()]
    X = F[cols].astype(float).to_numpy()
    out = {"n_events": len(ev), "by_exit": {}}
    for x in EXITS:
        y = (ev[f"net_usd_{x}"] > 0).astype(int).to_numpy()
        p, folds = walk_forward(X, y, ev["day"], h5)
        o = pd.concat([ev, F[cols]], axis=1).assign(p=p, y=y)
        o = o[np.isfinite(p)].reset_index(drop=True)
        r = {
            "oos": o,
            "n_oos": len(o),
            "folds": len(folds),
            "oos_from": str(o["day"].min()) if len(o) else "—",
            "auc": round(float(roc_auc_score(o["y"], o["p"])), 4)
            if o["y"].nunique() == 2
            else np.nan,
        }
        if len(o) >= 10:
            o2 = o.assign(
                dec=pd.qcut(
                    o["p"].rank(method="first"),
                    h5["eval"]["deciles"],
                    labels=range(1, h5["eval"]["deciles"] + 1),
                )
            )
            g = o2.groupby("dec", observed=True)
            r["deciles"] = pd.DataFrame(
                {
                    "n": g.size(),
                    "p_mean": g["p"].mean().round(3),
                    "胜率": g["y"].mean().round(3),
                    "扣费后均值%": g[f"net_pct_{x}"].mean().round(2),
                }
            ).reset_index()
        r["base"] = simulate_filter(o, x, v4)
        r["thr"] = {t: simulate_filter(o[o["p"] >= t], x, v4) for t in h5["eval"]["thresholds"]}
        r["mda"] = mda(X, y, folds, cols).head(h5["eval"]["mda_top"])
        r["cards"] = rule_cards(o, cols, x, h5)
        out["by_exit"][x] = r
        log.info("%s %s %s：事件 %d、样本外 %d、AUC %s", conv, und, x, len(ev), len(o), r["auc"])
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--unds", default=None, help="逗号分隔；给定时为部分结果，不算 FDR")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4, h5 = v4d.load_v4(), load_h5()
    convs = (h5["base"], h5["contrast"])
    unds = args.unds.split(",") if args.unds else v4["sample"]["underlyings"]
    partial = args.unds is not None
    sfx = f"_{'_'.join(unds)}" if partial else ""
    res: dict = {}
    for und in unds:
        stock = load_bars(root, und)
        v2 = v2_frame(root, und, stock, h5["features"]["confluence_window_1m"])
        depth = depth_features(stock)
        for conv in convs:
            res[(conv, und)] = run_one(root, und, conv, v4, h5, v2, depth)
    # 判定与 FDR
    cells = []
    for und in unds:
        for x in EXITS:
            for t in h5["eval"]["thresholds"]:
                s = {c: res[(c, und)]["by_exit"][x]["thr"][t] for c in convs}
                means = [s[c]["mean_pct"] for c in convs]
                ex5 = [s[c]["mean_pct_ex_top5"] for c in convs]
                ok = (
                    all(np.isfinite(means))
                    and all(m > 0 for m in means)
                    and all(np.isfinite(ex5))
                    and all(e > 0 for e in ex5)
                )
                cells.append(
                    {
                        "und": und,
                        "exit": x,
                        "thr": t,
                        **{f"n_{c}": s[c]["n"] for c in convs},
                        **{f"mean%_{c}": s[c]["mean_pct"] for c in convs},
                        **{f"ex5%_{c}": s[c]["mean_pct_ex_top5"] for c in convs},
                        **{f"p_{c}": s[c]["p"] for c in convs},
                        "p": max(s[c]["p"] for c in convs),
                        "条件满足": ok,
                    }
                )
    cells = pd.DataFrame(cells)
    if partial:
        r33, r33_joint = None, None
        cells["joint_fdr"] = None
        cells["通过"] = np.where(cells["条件满足"], "条件满足，FDR 待定", "否")
    else:
        r33 = pd.read_parquet(root / "research" / "v4" / f"cells_{v4['primary']}.parquet")
        joint = np.r_[r33["p"].to_numpy(), cells["p"].to_numpy()]
        rej = v4d.bh(joint, h5["judge"]["fdr_q"])
        cells["joint_fdr"] = rej[len(r33) :]
        cells["通过"] = cells["条件满足"] & cells["joint_fdr"]
        r33_joint = rej[: len(r33)]
    write_report(res, cells, r33, r33_joint, v4, h5)
    store = root / "research" / "v4"
    cells.to_parquet(store / f"h5_cells{sfx}.parquet", index=False)
    for (c, u), r in res.items():
        for x, rr in r["by_exit"].items():
            rr["oos"].to_parquet(store / f"h5_oos_{c}_{u}_{x}.parquet", index=False)
    return 0


def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "0 笔"
    return (
        f"{s['n']} 笔 · 胜率 {s['win_rate']} · 盈亏比 {s['payoff']} · 均值 {s['mean_pct']}% · "
        f"去前5 {s['mean_pct_ex_top5']}% · 合计 {s['sum_usd']:.0f} · 回撤 {s['max_dd_usd']:.0f}"
    )


def write_report(res, cells, r33, r33_joint, v4, h5) -> None:
    convs = (h5["base"], h5["contrast"])
    L = [
        "# V4 补充 H5：元标签模型（R-034）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.research.v4_h5`（需先跑 v4_decisive）",
        "",
        "预登记：RESEARCH_LOG R-034、`config/v4_h5.yaml`（commit 2715743，早于 R-033 任何结果）。"
        f"底座口径 {convs[0]}（D-027），对照 {convs[1]}（D-028）。",
        "",
        "## 1. 判定（18 格；两口径过滤后均值 > 0、去前 5 笔 > 0、符号一致，且联合 FDR 通过）",
        "",
    ]
    if r33 is None:
        n_ok = int((cells["条件满足"]).sum())
        L += [
            f"**部分结果（{', '.join(sorted(cells['und'].unique()))}）："
            "FDR 待 QQQ 完成后统一计算。**"
            f"下表 p 为未校正值；满足「两口径均值 > 0、去前 5 笔 > 0」条件的格子 {n_ok} 个。",
            "",
            df_to_md(cells.drop(columns=["joint_fdr"]).round(4)),
            "",
        ]
    else:
        passed = cells[cells["通过"]]
        L += [
            "**没有格子通过。**" if passed.empty else f"**通过 {len(passed)} 格。**",
            "",
            df_to_md(cells.round(4)),
            "",
            f"联合 FDR（R-033 口径 C 36 格 + H5 18 格 = {len(r33) + len(cells)} 格，q="
            f"{h5['judge']['fdr_q']}）：R-033 的格子在联合族中通过 {int(r33_joint.sum())} 个；"
            f"R-033 预登记的 36 格一族结果见 V4_DECISIVE.md 第 1 节，不因加入 H5 改写。",
            "",
        ]
    for (conv, und), r in res.items():
        L += [f"## {conv} · {und}（H1–H3 事件 {r['n_events']} 行）", ""]
        for x, rr in r["by_exit"].items():
            L += [
                f"### 离场 ({x})：样本外 {rr['n_oos']} 行"
                f"（{rr['folds']} 折，自 {rr['oos_from']}），"
                f"AUC {rr['auc']}",
                "",
                f"- 不用模型、H1–H3 全做（同期样本外）：{_fmt(rr['base'])}",
            ]
            L += [f"- 阈值 {t}：{_fmt(s)}" for t, s in rr["thr"].items()]
            L += [""]
            if "deciles" in rr:
                L += ["概率十分位：", "", df_to_md(rr["deciles"]), ""]
            L += [
                "MDA 前 15（样本外 AUC 下降）：",
                "",
                df_to_md(
                    rr["mda"]
                    .round(4)
                    .rename("auc_drop")
                    .reset_index()
                    .rename(columns={"index": "特征"})
                ),
                "",
            ]
            L += [
                "规则卡（深度 ≤3 树翻译「样本外概率 ≥ 0.60」；同一批样本外数据上提取与评估，"
                "只是描述，不是独立检验）：",
                "",
                df_to_md(rr["cards"]) if len(rr["cards"]) else "无概率 ≥ 0.60 的叶子。",
                "",
            ]
    L += [
        "## 边界",
        "",
        "- 事件、合约、成交、离场与 R-033 相同（见 V4_DECISIVE.md 边界）；标签是每笔自身结果，"
        "逐笔模拟才贪心。",
        "- 训练样本少（每模型几百到两千行），扩展窗口自 60 个样本交易日起，样本外月份不足 3 年。",
        "- 规则卡在同一批样本外数据上提取并评估，存在选择偏差，只作描述。",
        "- LightGBM 本机不可用，用 sklearn HistGradientBoosting（叶子 31、300 轮、学习率 0.03）。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    raise SystemExit(main())
