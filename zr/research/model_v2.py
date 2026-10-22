"""模型 v2（METHOD_V2 规格 6，R-023）：GBDT，参数固定不调参，按唯一性加权，滚动 walk-forward。

- 后端：LightGBM（缺 libomp 时回退 sklearn HistGradientBoosting），叶子 31、300 轮、学习率 0.03，见
  meta_model.make_model。
- 训练 12 个月 / 测试 1 个月 / embargo 1 天，逐月滚动；**没有验证集**（不做早停、不选参数）。
- 唯一性加权（López de Prado 2018 第 4 章）：事件标签窗口 [t+1, t+H] 覆盖的每分钟，并发数 c =
  覆盖该分钟的
  事件数；权重 = 窗口内 1/c 的平均。
- 目标：(b) ever_up_0.5_10（H=10 分钟内 MFE ≥ 0.5σ_H）；(c) net_cap30 > 0（真实期权扣费后盈利，
  持仓上限 30）。
- 输出：样本外概率（ZR_DATA_ROOT/research/v2/oos_<UND>_<target>.parquet）、AUC（总体 + 逐月）、
  校准、十分位。
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from zr.config import get_settings
from zr.research.dataset_v2 import feature_columns
from zr.research.meta_model import EMBARGO_DAYS, TRAIN_MONTHS, make_model

log = logging.getLogger("zr.research.model_v2")
TARGETS = {
    "b_ever_up_0.5_10": {"label": "ever_up_0.5_10", "H": 10},
    "c_option_profit_cap30": {"label": "net_cap30", "H": 30, "binarize": True},
}


def uniqueness_weights(day: pd.Series, minute: np.ndarray, H: int) -> np.ndarray:
    """窗口 [m+1, m+H]（日内，分钟索引 0–390）的平均唯一性。"""
    w = np.ones(len(day))
    for idx in day.groupby(day.to_numpy()).groups.values():
        ii = np.asarray(list(idx))
        pos = day.index.get_indexer(ii)
        m = minute[pos]
        conc = np.zeros(400 + H + 2)
        np.add.at(conc, m + 1, 1)
        np.add.at(conc, m + H + 1, -1)
        conc = np.cumsum(conc)
        inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1), 0.0)
        csum = np.concatenate([[0.0], np.cumsum(inv)])
        w[pos] = (csum[m + H + 1] - csum[m + 1]) / H
    return w


def design(ds: pd.DataFrame, und: str) -> pd.DataFrame:
    cols = [c for c in feature_columns(und) if c in ds.columns and ds[c].notna().any()]
    X = ds[cols].astype(float).copy()
    X["direction"] = ds["direction"].astype(float)
    for d in sorted(ds["detector"].unique()):
        X[f"det_{d}"] = (ds["detector"] == d).astype(float)
    return X


def walk_forward(
    X: pd.DataFrame, y: pd.Series, w: np.ndarray, day: pd.Series
) -> tuple[pd.Series, list[dict]]:
    month = pd.to_datetime(day).dt.to_period("M")
    pday = pd.to_datetime(day)
    months = sorted(month.unique())
    oos = pd.Series(np.nan, index=X.index)
    folds = []
    for i, m in enumerate(months):
        if i < TRAIN_MONTHS:
            continue
        tr = (pday >= (m - TRAIN_MONTHS).start_time) & (
            pday < m.start_time - pd.Timedelta(days=EMBARGO_DAYS)
        )
        te = month == m
        if tr.sum() < 500 or te.sum() < 50 or y[tr].nunique() < 2:
            continue
        backend, model = make_model()
        try:
            model.fit(X[tr], y[tr], sample_weight=w[tr.to_numpy()])
        except TypeError:
            model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oos[te] = p
        auc = roc_auc_score(y[te], p) if y[te].nunique() == 2 else np.nan
        folds.append(
            {
                "month": str(m),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "auc": round(float(auc), 4),
            }
        )
    return oos, folds


def run(root: Path, und: str, target: str) -> dict:
    t0 = time.monotonic()
    ds = pd.read_parquet(root / "research" / "v2" / f"dataset_{und}.parquet")
    spec = TARGETS[target]
    lab = ds[spec["label"]]
    keep = lab.notna()
    ds = ds[keep].reset_index(drop=True)
    y = (
        (ds[spec["label"]] > 0).astype(int)
        if spec.get("binarize")
        else ds[spec["label"]].astype(int)
    )
    X = design(ds, und)
    w = uniqueness_weights(ds["day"], ds["bar_minute"].to_numpy(), spec["H"])
    oos, folds = walk_forward(X, y, w, ds["day"])
    backend, _ = make_model()
    res = ds[["ts", "day", "bar_minute", "direction", "detector"]].copy()
    res["y"] = y
    res["p"] = oos
    for c in ("net_cap10", "net_cap30", "ever_up_0.5_10", "dte", "mfe_10"):
        if c in ds:
            res[c] = ds[c].to_numpy()
    out = root / "research" / "v2" / f"oos_{und}_{target}.parquet"
    res.to_parquet(out, index=False)
    ok = res["p"].notna()
    auc = (
        roc_auc_score(res.loc[ok, "y"], res.loc[ok, "p"])
        if ok.sum() and res.loc[ok, "y"].nunique() == 2
        else np.nan
    )
    return {
        "target": target,
        "backend": backend,
        "n": int(len(res)),
        "n_oos": int(ok.sum()),
        "auc": auc,
        "folds": pd.DataFrame(folds),
        "res": res,
        "seconds": time.monotonic() - t0,
        "n_features": X.shape[1],
        "base_rate": float(y.mean()),
    }


def decile_table(res: pd.DataFrame) -> pd.DataFrame:
    r = res[res["p"].notna()].copy()
    if r.empty:
        return pd.DataFrame()
    r["decile"] = pd.qcut(r["p"].rank(method="first"), 10, labels=False) + 1
    nd = r["day"].nunique()
    g = r.groupby("decile")
    return pd.DataFrame(
        {
            "n": g.size(),
            "per_day": (g.size() / nd).round(2),
            "p_mean": g["p"].mean().round(3),
            "y_rate": g["y"].mean().round(3),
            "mean_net_cap30": g["net_cap30"].mean().round(2) if "net_cap30" in r else np.nan,
            "mean_net_cap10": g["net_cap10"].mean().round(2) if "net_cap10" in r else np.nan,
        }
    ).reset_index()


def calibration(res: pd.DataFrame) -> pd.DataFrame:
    r = res[res["p"].notna()]
    b = pd.cut(r["p"], np.linspace(0, 1, 11), include_lowest=True)
    g = pd.DataFrame({"bin": b.astype(str), "p": r["p"], "y": r["y"]}).groupby("bin", observed=True)
    return g.agg(n=("y", "size"), p_mean=("p", "mean"), y_rate=("y", "mean")).round(3).reset_index()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MU.US")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    for t in TARGETS:
        r = run(root, a.symbol.split(".")[0], t)
        log.info(
            "%s %s AUC %.4f（OOS %d / %d，%d 折，%.0fs）",
            a.symbol,
            t,
            r["auc"],
            r["n_oos"],
            r["n"],
            len(r["folds"]),
            r["seconds"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
