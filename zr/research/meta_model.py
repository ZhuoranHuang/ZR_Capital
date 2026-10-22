"""元标签模型（R-013）与全样本基线（R-014）。

- 元标签：GBDT 二分类预测 P(方向化收益 > 0 | 特征)，按日期的 purged walk-forward：
  训练 12 个月 → 测试 1 个月，embargo 1 天（训练集剔除测试期开始前 1 天内的事件），逐月滚动。
  报告样本外 AUC（总体与逐月）、校准曲线、按预测概率十分位的实际成功率与扣费后均值、MDA 特征重要性。
- 基线（--baseline）：不经事件，对所有 K 线（每 5 根抽 1 根控制规模）用同一特征集
  预测未来 5/15 分钟方向，
  报告月度 Spearman IC 的均值、正月占比。
缺失值（quote_slope_1、期权特征在无数据日期）原样传入：两个后端都原生支持 NaN，不做 0 填充。
后端：LightGBM 可用时用 LightGBM，否则 scikit-learn HistGradientBoosting（同为直方图 GBDT；
本机无 libomp 时自动回退，报告里注明）。
运行：uv run python -m zr.research.meta_model [--H 5] [--baseline]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from zr.config import get_settings
from zr.features.bars import load_bars
from zr.features.features import FEATURE_COLUMNS, compute_features
from zr.features.option_features import OPTION_FEATURE_COLUMNS
from zr.ops.markdown import df_to_md

log = logging.getLogger("zr.research.meta_model")
ET = ZoneInfo("America/New_York")
TRAIN_MONTHS, EMBARGO_DAYS = 12, 1


def make_model():
    try:
        import lightgbm as lgb

        return "lightgbm", lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            verbose=-1,
            n_jobs=4,
        )
    except Exception:  # noqa: BLE001 - 缺 libomp 等
        from sklearn.ensemble import HistGradientBoostingClassifier

        return "sklearn_hgb", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
        )


def design_matrix(ds: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    opt_cols = [c for c in OPTION_FEATURE_COLUMNS if c in ds and ds[c].notna().any()]
    X = ds[FEATURE_COLUMNS + ["quote_slope_1"] + opt_cols].copy()
    X["direction"] = ds["direction"].astype(float)
    for d in sorted(ds["detector"].unique()):
        X[f"det_{d}"] = (ds["detector"] == d).astype(float)
    for s in sorted(ds["symbol"].unique()):
        X[f"sym_{s.split('.')[0]}"] = (ds["symbol"] == s).astype(float)
    return X, list(X.columns)


def month_key(days: pd.Series) -> pd.Series:
    return pd.to_datetime(days).dt.to_period("M")


def walk_forward(
    X: pd.DataFrame, y: pd.Series, days: pd.Series
) -> tuple[pd.Series, list[dict], object, pd.Index]:
    months = sorted(month_key(days).unique())
    pdays = pd.to_datetime(days)
    oos = pd.Series(np.nan, index=X.index)
    folds = []
    last_model, last_test_idx = None, None
    for i, m in enumerate(months):
        if i < TRAIN_MONTHS:
            continue
        test_start = m.start_time
        train_start = (m - TRAIN_MONTHS).start_time
        train_mask = (pdays >= train_start) & (pdays < test_start - pd.Timedelta(days=EMBARGO_DAYS))
        test_mask = month_key(days) == m
        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue
        backend, model = make_model()
        model.fit(X[train_mask], y[train_mask])
        p = model.predict_proba(X[test_mask])[:, 1]
        oos[test_mask] = p
        auc = roc_auc_score(y[test_mask], p) if y[test_mask].nunique() == 2 else np.nan
        folds.append(
            {
                "month": str(m),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "auc": round(float(auc), 3),
            }
        )
        last_model, last_test_idx = model, X.index[test_mask]
    return oos, folds, last_model, last_test_idx


def decile_table(
    p: pd.Series, y: pd.Series, ret: pd.Series, pnl: pd.Series, days: pd.Series
) -> pd.DataFrame:
    ok = p.notna()
    q = pd.qcut(p[ok].rank(method="first"), 10, labels=False) + 1
    n_days = days[ok].nunique()
    rows = []
    for d, idx in q.groupby(q).groups.items():
        rows.append(
            {
                "decile": int(d),
                "n": len(idx),
                "per_day": round(len(idx) / n_days, 2),
                "p_mean": round(float(p[idx].mean()), 3),
                "hit_rate": round(float(y[idx].mean()), 3),
                "mean_ret_bp": round(float(ret[idx].mean() * 1e4), 2),
                "mean_pnl_fee_adj_usd": round(float(pnl[idx].mean()), 2),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(p: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    ok = p.notna()
    b = pd.cut(p[ok], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    g = pd.DataFrame({"bin": b, "p": p[ok], "y": y[ok]}).groupby("bin", observed=True)
    return (
        g.agg(n=("y", "size"), p_mean=("p", "mean"), y_rate=("y", "mean"))
        .round(3)
        .reset_index()
        .astype({"bin": str})
    )


def run_meta(ds: pd.DataFrame, H: int, out_dir: Path) -> list[str]:
    ds = ds.dropna(subset=[f"ret_h{H}"]).reset_index(drop=True)
    X, cols = design_matrix(ds)
    y = (ds[f"ret_h{H}"] > 0).astype(int)
    days = ds["day"]
    t0 = time.monotonic()
    oos, folds, model, test_idx = walk_forward(X, y, days)
    backend, _ = make_model()
    ok = oos.notna()
    auc_all = roc_auc_score(y[ok], oos[ok]) if ok.sum() else np.nan
    dec = decile_table(oos, y, ds[f"ret_h{H}"], ds[f"pnl_h{H}"], days)
    cal = calibration_table(oos, y)
    mda = pd.DataFrame()
    if model is not None:
        r = permutation_importance(
            model,
            X.loc[test_idx],
            y.loc[test_idx],
            scoring="roc_auc",
            n_repeats=5,
            random_state=0,
            n_jobs=1,
        )
        mda = (
            pd.DataFrame({"feature": cols, "mda_auc_drop": r.importances_mean.round(4)})
            .sort_values("mda_auc_drop", ascending=False)
            .head(20)
        )
    top = dec[dec["decile"] == 10].iloc[0] if len(dec) else None
    per_month_auc = pd.DataFrame(folds)
    lines = [
        f"## 元标签 H={H}（后端 {backend}，耗时 {time.monotonic() - t0:.0f}s）",
        "",
        f"事件 {len(ds)}，样本外事件 {int(ok.sum())}（{len(folds)} 个测试月），"
        f"**样本外 AUC {auc_all:.3f}**。",
        f"月度 AUC：均值 {per_month_auc['auc'].mean():.3f}，"
        f">0.5 的月份 {(per_month_auc['auc'] > 0.5).mean():.0%}。",
        "",
        "### 按预测概率十分位（样本外）",
        "",
        df_to_md(dec),
        "",
        (
            f"**判定**：最高十分位扣费后期望 {top['mean_pnl_fee_adj_usd']:+.2f} USD/张，"
            f"每日 {top['per_day']} 个事件，hit_rate {top['hit_rate']}。"
        )
        if top is not None
        else "无样本外结果",
        "",
        "### 校准（预测概率 vs 实际成功率）",
        "",
        df_to_md(cal),
        "",
        "### 逐月 AUC",
        "",
        df_to_md(per_month_auc),
        "",
        "### 特征重要性（MDA，最后一个测试月，AUC 下降）",
        "",
        df_to_md(mda),
        "",
    ]
    return lines


def _monthly_ic_table(tab: pd.DataFrame) -> pd.DataFrame:
    piv = tab.pivot_table(index="month", columns=["symbol", "H"], values="ic").round(3)
    piv.columns = [f"{s.split('.')[0]}_H{h}" for s, h in piv.columns]
    return piv.reset_index()


def run_baseline(root: Path, symbols: list[str], step: int = 5) -> list[str]:
    rows = []
    t0 = time.monotonic()
    for sym in symbols:
        und = sym.split(".")[0]
        df = load_bars(root, und)
        f = compute_features(df)
        day = pd.Series(df.index.tz_convert(ET).date, index=df.index)
        X = f[FEATURE_COLUMNS].iloc[::step]
        d = day.iloc[::step]
        for H in (5, 15):
            fut = (df["close"].groupby(day.to_numpy()).shift(-H) / df["close"] - 1).iloc[::step]
            ok = fut.notna() & X.notna().all(axis=1)
            Xo, yo, fo, do = X[ok], (fut[ok] > 0).astype(int), fut[ok], d[ok]
            months = sorted(month_key(do).unique())
            pdays = pd.to_datetime(do)
            for i, m in enumerate(months):
                if i < TRAIN_MONTHS:
                    continue
                tr = (pdays >= (m - TRAIN_MONTHS).start_time) & (
                    pdays < m.start_time - pd.Timedelta(days=EMBARGO_DAYS)
                )
                te = month_key(do) == m
                if tr.sum() < 1000 or te.sum() < 100:
                    continue
                _, model = make_model()
                model.fit(Xo[tr], yo[tr])
                p = model.predict_proba(Xo[te])[:, 1]
                ic = spearmanr(p, fo[te]).correlation
                rows.append(
                    {
                        "symbol": sym,
                        "H": H,
                        "month": str(m),
                        "n": int(te.sum()),
                        "ic": round(float(ic), 4),
                    }
                )
        log.info("基线 %s 完成 %.0fs", sym, time.monotonic() - t0)
    tab = pd.DataFrame(rows)
    summ = (
        tab.groupby(["symbol", "H"])
        .agg(
            months=("ic", "size"),
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            pos_share=("ic", lambda s: float((s > 0).mean())),
        )
        .round(4)
        .reset_index()
    )
    summ["ic_t"] = (summ["ic_mean"] / (summ["ic_std"] / np.sqrt(summ["months"]))).round(2)
    return [
        f"## 全样本基线（R-014，每 {step} 根抽 1 根，训练 12 月 / 测试 1 月，"
        f"耗时 {time.monotonic() - t0:.0f}s）",
        "",
        df_to_md(summ),
        "",
        "逐月 IC：",
        "",
        df_to_md(_monthly_ic_table(tab)),
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--out", default="docs/research/META_MODEL.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    backend, _ = make_model()
    lines = [
        "# 元标签模型（R-013）与全样本基线（R-014）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 后端 {backend} · "
        f"purged walk-forward：训练 {TRAIN_MONTHS} 月 / 测试 1 月 / embargo {EMBARGO_DAYS} 天。",
        "特征：FEATURE_COLUMNS + quote_slope_1 + 检测器 one-hot + 方向 + 标的 one-hot；"
        "标签 = 方向化收益 > 0。",
        "扣费后 P&L 口径同 EVENT_LABEL_TABLE（单张 ATM 近似）。",
        "",
    ]
    if not args.baseline:
        ds = pd.read_parquet(root / "research" / "w2" / "dataset.parquet")
        for H in args.H:
            lines += run_meta(ds, H, out_dir)
    else:
        lines += run_baseline(root, get_settings().symbols)
        args.out = str(Path(args.out).with_name("BASELINE_IC.md"))
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
