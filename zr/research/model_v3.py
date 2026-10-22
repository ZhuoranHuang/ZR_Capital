"""模型 v3（METHOD_V3 第 8 节）。

- 样本：2 分钟周期上全部入场形态的唯一事件（日, 入场分钟, 方向）× 结构离场规则（出场规则 ID
  作为特征），
  X6 固定 (+20% 减半, −30% 止损)、X7 固定 15 分钟（R-032 预登记的口径）。
- 目标 = 主标签：该事件在该出场规则下的真实期权扣费后盈亏 > 0（引擎 zr/research/sim_v3.py，D-023
  口径）。
- 特征：v2 全部（正股 23 列、通道质量、结构上下文、期权、日历、跨资产、共振；
  取决策时刻前最后一根已收盘的
  1 分钟 K）+ 回踩深度 pb_*（按方向）+ v3 状态特征（状态、方向、持续根数、带宽扩张率、外轨连续根数、
  MACD(6,13,6) 柱及变化、回踩 %b 与带半宽比，2 分钟决策 K）+ 入场家族标记 + 出场规则 ID + 方向。
- 训练 12 月 / 测试 1 月 / embargo 1 天滚动。两套参数：固定（meta_model.make_model）与 Optuna
  嵌套调参
  （每个滚动窗口内部，用训练窗最后 2 个月作内层验证，调叶子数、学习率、轮数、最小叶样本，按内层 AUC
  选，
  再用整个训练窗重训）。神经网络不做（登记为备选）。
- 样本外高概率事件（最高十分位）用深度 ≤ 3 决策树提取规则，并入 RULE_CARDS。
运行：uv run python -m zr.research.model_v3 --symbol MU.US
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from zr.config import get_settings
from zr.features.bars import ET, load_bars
from zr.features.calendar_features import compute_calendar
from zr.features.channel import channel_states
from zr.features.cross_asset import compute_cross_asset
from zr.features.features import compute_features
from zr.features.option_features import compute_option_features, option_days
from zr.features.pullback import depth_features, directional
from zr.features.regime import load_config, state_features
from zr.features.structure import compute_structure
from zr.ops.markdown import df_to_md
from zr.research import sim_v3 as sv
from zr.research.dataset_v2 import CONFIG as EV2_CONFIG
from zr.research.dataset_v2 import feature_columns
from zr.research.events import confluence, signal_matrix
from zr.research.events import load_config as load_ev2
from zr.research.meta_model import EMBARGO_DAYS, TRAIN_MONTHS, make_model
from zr.research.own_policy import apply_rule, tree_rule

log = logging.getLogger("zr.research.model_v3")
PERIOD = 2
TP, SL, CAP = 0.20, 0.30, 15
NAN_NOTE = "\n\n规则中的缺失值按 −999 参与比较（例如 `x <= 阈值` 包含 x 缺失的行）。"


def v2_frame(root: Path, und: str, df: pd.DataFrame) -> pd.DataFrame:
    feats = compute_features(df)
    chans = channel_states(df, feats)
    struct = compute_structure(df, feats, chans)
    xa = compute_cross_asset(df, und, root)
    cal = compute_calendar(df.index, root, und)
    opt = compute_option_features(root, und, df)
    sig = signal_matrix(df, feats, load_ev2(EV2_CONFIG), extra={"_chans": chans})
    conf = confluence(sig, 5)
    allf = pd.concat([feats, chans, struct, opt, cal, xa, conf], axis=1)
    return allf.loc[:, ~allf.columns.duplicated()]


def build_dataset(root: Path, und: str, cfg: dict) -> pd.DataFrame:
    t0 = time.monotonic()
    stock = load_bars(root, und)  # RTH-only；v2_frame / depth_features / v3 状态层都用这个
    # （D-027 取代 D-026：核实发现 App 的 BOLL/MACD 不含盘前盘后，也不跨日累计，见
    # docs/research/INDICATOR_PARITY.md；regime.indicators 默认 daily_reset=True）
    pdat = sv.prepare_period(stock, PERIOD, cfg)
    x6, x7 = cfg["exits"]["X6"], cfg["exits"]["X7"]
    ti, si, ci = x6["tp"].index(TP), x6["sl"].index(SL), x7["cap_min"].index(CAP)
    fam_of = {n: n.split("|")[0] for n in pdat.entries}
    last_entry = sv.hm(cfg["execution"]["last_entry"])
    rows = []
    for day in option_days(root, und):
        mask = pdat.day == day
        if not mask.any():
            continue
        od = sv.load_day(root, und, day, cfg["execution"]["stale_snapshots"])
        if od is None:
            continue
        ev, by_cfg = sv.day_events(pdat, mask, last_entry)
        if ev.empty:
            continue
        res = sv.evaluate(od, ev, sv.day_triggers(pdat, mask), cfg)
        if res is None:
            continue
        fams = {i: set() for i in range(len(ev))}
        for name, lst in by_cfg.items():
            for j in lst:
                fams[j].add(fam_of[name])
        key = {(e, d): j for j, (e, d) in enumerate(zip(ev["e"], ev["direction"], strict=True))}
        meta = res["meta"]
        for i, r in meta.iterrows():
            j = key[(r["e"], r["direction"])]
            for ri, rule in enumerate(res["rules"]):
                rows.append(
                    {
                        "day": day,
                        "e": int(r["e"]),
                        "direction": int(r["direction"]),
                        "dte": int(r["dte"]),
                        "rule": rule,
                        "rule_id": ri,
                        "net": float(res["net"][i, ri, ti, si, ci]),
                        "hold": int(res["hold"][i, ri, ti, si, ci]),
                        "fam_E10": "E10" in fams[j],
                        "fam_E1": "E1" in fams[j],
                        "fam_E11": "E11" in fams[j],
                    }
                )
    ds = pd.DataFrame(rows)
    log.info("%s 事件×规则 %d 行，%.0fs", und, len(ds), time.monotonic() - t0)
    # 特征：决策时刻 = 入场分钟 − 1（2 分钟 K 收盘）；1 分钟特征取标签 = 决策分钟 − 1 的那根
    base = pd.to_datetime(ds["day"].astype(str)).dt.tz_localize(ET)
    ts1 = (base + pd.to_timedelta(570 + ds["e"] - 2, unit="min")).dt.tz_convert("UTC")
    ts2 = (base + pd.to_timedelta(570 + ds["e"] - 1 - PERIOD, unit="min")).dt.tz_convert("UTC")
    v2 = v2_frame(root, und, stock)
    cols = [c for c in feature_columns(und) if c in v2.columns and not c.startswith("confl_")]
    F = v2[cols].reindex(ts1.to_numpy()).reset_index(drop=True)
    up = ds["direction"].to_numpy() == 1
    conf = v2[["confl_n_up", "confl_n_dn", "confl_fam_up", "confl_fam_dn"]].reindex(ts1.to_numpy())
    F["confl_n_same_dir_5m"] = np.where(up, conf["confl_n_up"], conf["confl_n_dn"])
    F["confl_families_5m"] = np.where(up, conf["confl_fam_up"], conf["confl_fam_dn"])
    pb = directional(depth_features(stock), pd.Series(ts1.to_numpy()), ds["direction"].to_numpy())
    pdx = sv.prepare_period.__globals__["indicators"](pdat.bars, cfg)
    reg = sv.prepare_period.__globals__["regimes"](pdat.bars, pdx, cfg, PERIOD)
    sf = state_features(pdat.bars, pdx, reg).reindex(ts2.to_numpy()).reset_index(drop=True)
    out = pd.concat([ds.reset_index(drop=True), F, pb.reset_index(drop=True), sf], axis=1)
    log.info("%s 特征完成 %.0fs", und, time.monotonic() - t0)
    return out


def feature_list(ds: pd.DataFrame, und: str) -> list[str]:
    drop = {"day", "e", "net", "hold", "rule", "y"}
    return [
        c
        for c in ds.columns
        if c not in drop
        and ds[c].notna().any()
        and pd.api.types.is_numeric_dtype(ds[c])
        or c in ("fam_E10", "fam_E1", "fam_E11")
    ]


def _hgb(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        early_stopping=False, l2_regularization=1.0, random_state=0, **params
    )


def tune(Xtr, ytr, dtr, n_trials: int, seed: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    months = pd.to_datetime(pd.Series(dtr)).dt.to_period("M")
    inner = months >= sorted(months.unique())[-2]
    if inner.sum() < 200 or (~inner).sum() < 500 or len(set(ytr[inner.to_numpy()])) < 2:
        return {}
    it, iv = (~inner).to_numpy(), inner.to_numpy()

    def obj(trial):
        p = {
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 400, log=True),
        }
        m = _hgb(p).fit(Xtr[it], ytr[it])
        return roc_auc_score(ytr[iv], m.predict_proba(Xtr[iv])[:, 1])

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params


def walk_forward(ds: pd.DataFrame, cols: list[str], n_trials: int) -> tuple[pd.DataFrame, list]:
    X = ds[cols].astype(float).to_numpy()
    y = (ds["net"] > 0).astype(int).to_numpy()
    day = pd.to_datetime(ds["day"].astype(str))
    month = day.dt.to_period("M")
    months = sorted(month.unique())
    p_fix = np.full(len(ds), np.nan)
    p_opt = np.full(len(ds), np.nan)
    folds = []
    for i, m in enumerate(months):
        if i < TRAIN_MONTHS:
            continue
        tr = (
            (day >= (m - TRAIN_MONTHS).start_time)
            & (day < m.start_time - pd.Timedelta(days=EMBARGO_DAYS))
        ).to_numpy()
        te = (month == m).to_numpy()
        if tr.sum() < 1000 or te.sum() < 50 or len(set(y[tr])) < 2:
            continue
        _, mf = make_model()
        mf.fit(X[tr], y[tr])
        p_fix[te] = mf.predict_proba(X[te])[:, 1]
        best = tune(X[tr], y[tr], ds["day"].to_numpy()[tr], n_trials, seed=i)
        mo = _hgb(best).fit(X[tr], y[tr]) if best else mf
        p_opt[te] = mo.predict_proba(X[te])[:, 1]
        auc = lambda p, t=te: roc_auc_score(y[t], p[t]) if len(set(y[t])) == 2 else np.nan  # noqa: E731
        folds.append(
            {
                "month": str(m),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "auc_fixed": round(float(auc(p_fix)), 4),
                "auc_optuna": round(float(auc(p_opt)), 4),
                **{f"best_{k}": v for k, v in best.items()},
            }
        )
        log.info(
            "fold %s fixed %.4f optuna %.4f", m, folds[-1]["auc_fixed"], folds[-1]["auc_optuna"]
        )
    return ds.assign(p_fixed=p_fix, p_optuna=p_opt, y=y), folds


def deciles(res: pd.DataFrame, col: str) -> pd.DataFrame:
    """按样本外概率分十分位汇总（标准评估报表写法）。

    分位边界用整个样本外集合算，仅作描述性报告，不是因果特征、不作为交易阈值
    （交易用的是固定阈值，不是相对同批样本外数据的分位）。
    """
    r = res[res[col].notna()].copy()
    r["decile"] = pd.qcut(r[col].rank(method="first"), 10, labels=range(1, 11))
    g = r.groupby("decile", observed=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "p_mean": g[col].mean().round(3),
            "y_rate": g["y"].mean().round(3),
            "mean_per_contract": (g["net"].mean() / 2).round(2),
        }
    ).reset_index()


def extract_rule(oos: pd.DataFrame, cols: list[str]) -> dict:
    """样本外最高十分位 → 深度 ≤ 3 决策树 → 正样本占比最高的叶子规则及其真实期权盈亏。

    90% 分位用整个样本外集合算，是事后对「模型高置信区域长什么样」的描述，不是因果特征、
    不是可独立复现的交易规则（提取与评估用的是同一批样本外数据）。
    """
    top = (oos["p_optuna"] >= oos["p_optuna"].quantile(0.9)).astype(int).to_numpy()
    Xo = np.nan_to_num(oos[cols].astype(float).to_numpy(), nan=-999.0)
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0).fit(Xo, top)
    path, share, _ = tree_rule(tree, cols, min_pos=50)
    if not path:
        return {}
    sel = oos[apply_rule(oos[cols], path)]
    yr = pd.to_datetime(sel["day"].astype(str)).dt.year
    return {
        "rule": " 且 ".join(f"{c} {op} {t:.4g}" for c, op, t in path),
        "n_rows": int(len(sel)),
        "n_events": int(sel[["day", "e", "direction"]].drop_duplicates().shape[0]),
        "mean_per_contract": round(float(sel["net"].mean() / 2), 2) if len(sel) else np.nan,
        "win_rate": round(float((sel["net"] > 0).mean()), 3) if len(sel) else np.nan,
        "mean_per_contract_2026": round(float(sel.loc[yr == 2026, "net"].mean() / 2), 2)
        if (yr == 2026).any()
        else np.nan,
        "top_decile_share_in_leaf": round(float(share), 3),
    }


def rules_only(und: str) -> None:
    """不重训：从已保存的样本外预测重算规则，替换 model_<UND>.md 的规则表与 model_rules parquet。"""
    root = get_settings().zr_data_root
    store = root / "research" / "v3"
    oos = pd.read_parquet(store / f"model_oos_{und}.parquet")
    oos = oos[oos["p_optuna"].notna()]
    drop = {"p_fixed", "p_optuna", "y"}
    cols = [c for c in feature_list(oos.drop(columns=list(drop)), und) if c != "direction"] + [
        "direction"
    ]
    rs = extract_rule(oos, cols)
    md = (store / f"model_{und}.md").read_text(encoding="utf-8")
    head, rest = md.split("样本外最高十分位的深度 ≤ 3 决策树规则（并入 RULE_CARDS 附录）：", 1)
    tail = rest[rest.index("边界：") :]
    body = (df_to_md(pd.DataFrame([rs])) + NAN_NOTE) if rs else "无满足条件的叶子。"
    (store / f"model_{und}.md").write_text(
        head
        + "样本外最高十分位的深度 ≤ 3 决策树规则（并入 RULE_CARDS 附录）：\n\n"
        + body
        + "\n\n"
        + tail,
        encoding="utf-8",
    )
    (pd.DataFrame([rs]) if rs else pd.DataFrame()).to_parquet(
        store / f"model_rules_{und}.parquet", index=False
    )


def run(und: str, n_trials: int) -> tuple[str, pd.DataFrame]:
    t0 = time.monotonic()
    cfg = load_config()
    root = get_settings().zr_data_root
    ds = build_dataset(root, und, cfg)
    cols = [c for c in feature_list(ds, und) if c != "direction"] + ["direction"]
    res, folds = walk_forward(ds, cols, n_trials)
    store = root / "research" / "v3"
    store.mkdir(parents=True, exist_ok=True)
    res.to_parquet(store / f"model_oos_{und}.parquet", index=False)
    oos = res[res["p_optuna"].notna()]
    auc_f = roc_auc_score(oos["y"], oos["p_fixed"]) if len(oos) else np.nan
    auc_o = roc_auc_score(oos["y"], oos["p_optuna"]) if len(oos) else np.nan
    df_f, df_o = deciles(oos, "p_fixed"), deciles(oos, "p_optuna")
    rule_stats = extract_rule(oos, cols)
    months = sorted(oos["day"].astype(str).str[:7].unique())
    L = [
        f"## 模型（第 8 节，{und}）",
        "",
        f"样本：2 分钟周期唯一事件 × 结构离场规则，X6 = (+{TP:.0%} 减半, −{SL:.0%})、X7 = {CAP} "
        "分钟；"
        f"{len(ds)} 行，特征 {len(cols)} 个，正例率 {ds['net'].gt(0).mean():.3f}；样本外 "
        f"{len(oos)} 行"
        f"（{months[0] if months else '-'}～{months[-1] if months else '-'}，{len(folds)} "
        "个测试月）。"
        f"Optuna 每窗 {n_trials} 次试验。耗时 {time.monotonic() - t0:.0f}s。",
        "",
        f"**样本外 AUC：固定参数 {auc_f:.4f}，Optuna 嵌套调参 {auc_o:.4f}（差 "
        f"{auc_o - auc_f:+.4f}）。**",
        "",
        "十分位（固定参数）：",
        "",
        df_to_md(df_f),
        "",
        "十分位（Optuna）：",
        "",
        df_to_md(df_o),
        "",
        "逐月：",
        "",
        df_to_md(pd.DataFrame(folds)),
        "",
        "样本外最高十分位的深度 ≤ 3 决策树规则（并入 RULE_CARDS 附录）：",
        "",
        (df_to_md(pd.DataFrame([rule_stats])) + NAN_NOTE) if rule_stats else "无满足条件的叶子。",
        "",
        f"边界：标的 {und}；期权日 2023-10-02～2026-10-21；成交口径 D-023；只评估一组 X6/X7；"
        "十分位与规则都在样本外月份上计算，但规则是在同一批样本外数据上提取的（描述性，"
        "不是独立验证）。",
        "",
    ]
    return "\n".join(L), pd.DataFrame([rule_stats]) if rule_stats else pd.DataFrame()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MU.US")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--rules-only", action="store_true", help="从已保存的样本外预测重算规则")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    und = a.symbol.split(".")[0]
    if a.rules_only:
        rules_only(und)
        return 0
    text, rules = run(und, a.trials)
    root = get_settings().zr_data_root
    (root / "research" / "v3" / f"model_{und}.md").write_text(text, encoding="utf-8")
    rules.to_parquet(root / "research" / "v3" / f"model_rules_{und}.parquet", index=False)
    log.info("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
