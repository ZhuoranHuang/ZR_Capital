"""负责人入场规律反推（第一条）→ docs/research/OWNER_ENTRY_MINING.md、
config/intrabar/owner_rule_*.yaml。

样本：D-030 入选段中 MU 的开仓买入委托（日内单腿），事件时刻 = 委托提交时刻（长桥历史委托，
整秒向下取整；特征取 ≤ 该秒的数据）。分组：盈利入场（该段净盈亏 > 0）、亏损入场、随机对照（每张
委托 50 个时刻：同日、提交前后 30 分钟内、09:31–15:59、剔除提交前后 2 分钟；方向 = 所配对委托的
方向）。任务 A：盈利入场 vs 其随机对照；B：盈利 vs 亏损入场；C：全部入场 vs 随机对照。
特征：zr/features/entry_features.py（方向化、逐秒、截断检验）。不使用以前的事件检测器 / 状态
标签作特征（状态层只作报告列）。

单特征：组内百分位（入场时刻在自身 50 个随机时刻中的位置，A / C）与合并 AUC；置换检验 500 次
（A / C：每组随机指定一个成员为「入场」；B：打乱盈亏标签），经验 p（双侧），BH q = 0.10。
模型：梯度提升树（sklearn HistGradientBoosting，本机 LightGBM 缺 libomp、无 XGBoost；叶 ≤ 8、
深度 ≤ 3、叶最少 10、L2 = 10、≤ 200 轮、学习率 0.05）与 L1 逻辑回归；按交易日留一交叉验证；
样本外 AUC 与置换 p（500 次，标签按同一方式打乱后整套交叉验证重做）；样本外置换重要性前 20；
深度 ≤ 3 决策树规则（覆盖、盈利入场占比、随机占比）。对称性（多 / 空分别做 A）、稳健性（月份、
所买合约 DTE 0 / ≥1、10:15 前后）。候选规则：「趋势方向判定 + 触发条件 + 不做条件」模板搜索
（阈值取盈利入场的分位数），要求覆盖 ≥ 8 个盈利入场且盈利占比 > 随机占比；规则搜索流程本身
按交易日留一检验（每折在其余日子上选规则，在留出日上看覆盖）。
运行：uv run python -m zr.research.owner_entry_mining [--stage sample|stats|models|rules|report|all]
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from zr.config import get_settings
from zr.features import entry_features as ef
from zr.features.bars import ET
from zr.features.regime import STATE_NAMES, indicators, regimes
from zr.research import entry_timing_diag as etd
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.owner_entry_mining")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "OWNER_ENTRY_MINING.md"
RULE_DIR = REPO / "config" / "intrabar"
NS = 1_000_000_000
N_RAND = 50
N_PERM = 500
SEED = 0
Q = 0.10
MIN_WIN_COVER = 8
TRIGGER_FAMS = ("sec", "live", "brute")
TREND_FAMS = ("bar",)
# 趋势方向判定只从这些「正值 = 处于方向 d 的趋势中」的特征里选，阈值 > 0（双向对称）
TREND_DEF = tuple(
    f"b{p}_{k}"
    for p in (1, 2, 5, 15)
    for k in (
        "side_count10",
        "side_count20",
        "ma_align",
        "boll_mid_slope3",
        "boll_mid_slope5",
        "m6_dif_side",
        "m6_cross_side",
        "m12_dif_side",
        "reg_slope20_atr",
        "di_diff",
        "eff_ratio20",
        "struct10",
        "dist_ma20_atr",
        "aroon_osc",
    )
) + ("live2_dist_mid_atr", "live5_dist_mid_atr")
NOTRADE_FAMS = ("opt", "optvol", "cal", "vol", "cross")


def store(root: Path) -> Path:
    p = root / "research" / "owner_entry_mining"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- 样本 ----------------


def events(root: Path) -> pd.DataFrame:
    b = etd.buy_orders(root)
    b = b[b["und"] == "MU"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for k, o in b.iterrows():
        sub = int(o["submit_raw_ns"])
        day = pd.Timestamp(sub, tz="UTC").tz_convert(ET).date()
        et = pd.Timestamp(sub, tz="UTC").tz_convert(ET)
        base = {
            "group": k,
            "day": day,
            "dir": int(o["dir"]),
            "pnl": float(o["pnl"]),
            "win": bool(o["pnl"] > 0),
            "month": day.month,
            "dte": (pd.Timestamp(o["expiry"]).date() - day).days,
            "before_1015": et.time() < dt.time(10, 15),
            "symbol": o["symbol"],
        }
        rows.append({**base, "kind": "entry", "t_ns": sub})
        lo = max(sub - 30 * 60 * NS, pd.Timestamp(f"{day} 09:31", tz=ET).value)
        hi = min(sub + 30 * 60 * NS, pd.Timestamp(f"{day} 15:59", tz=ET).value)
        cand = rng.integers(lo, hi, size=N_RAND * 4)
        cand = cand[np.abs(cand - sub) > 120 * NS][:N_RAND]
        rows += [{**base, "kind": "random", "t_ns": int(c)} for c in np.sort(cand)]
    return pd.DataFrame(rows)


def state_at(root: Path, t_ns: np.ndarray, end: dt.date) -> list[str]:
    """报告列：最后一根已完成 2 分钟 K（口径 C）的状态层标签（不作特征）。"""
    v4 = v4d.load_v4()
    conv = v4["conventions"]["C"]
    cfg = v4d.state_cfg(v4, "C")
    bars = v4d.build_bars(root, "MU", conv, end)
    x = indicators(bars, cfg, daily_reset=False, warmup="nan")
    reg = regimes(bars, x, cfg, v4["period"])
    cns = v4d.ns(bars.index) + v4["period"] * 60 * NS
    i = np.searchsorted(cns, t_ns, side="right") - 1
    st, dr = reg["state"].to_numpy(), reg["dir"].to_numpy()
    return [f"{STATE_NAMES[int(st[j])]}{int(dr[j]):+d}" if j >= 0 else "" for j in i]


def build_sample(root: Path) -> pd.DataFrame:
    ev = events(root)
    ctx = ef.Ctx.load(root)
    parts, trunc = [], []
    for day, g in ev.groupby("day"):
        d = ef.load_day(ctx, day)
        if d is None:
            log.warning("%s 没有秒级正股价", day)
            continue
        fs = ef.day_features(d)
        rows = ef.grid_rows(d, g["t_ns"].to_numpy())
        x = ef.sample(fs, rows, g["dir"].to_numpy())
        x.index = g.index
        parts.append(pd.concat([g, x], axis=1))
        # 截断检验：该日全部入场时刻 + 每张委托 1 个随机时刻
        tt = np.r_[
            g.loc[g["kind"] == "entry", "t_ns"].to_numpy(),
            g[g["kind"] == "random"].groupby("group").head(1)["t_ns"].to_numpy(),
        ]
        tc = ef.truncation_check(ctx, day, tt)
        trunc.append(tc.assign(day=str(day), n_times=len(tt)))
        log.info("%s：%d 行，截断不一致 %d", day, len(g), int((tc["mismatch"] > 0).sum()))
    s = pd.concat(parts).sort_index()
    s["state"] = state_at(root, s["t_ns"].to_numpy(), max(s["day"]))
    tr = pd.concat(trunc, ignore_index=True)
    st = store(root)
    s.assign(day=s["day"].astype(str)).to_parquet(st / "sample.parquet")
    tr.to_parquet(st / "truncation.parquet", index=False)
    return s


def load_sample(root: Path) -> pd.DataFrame:
    s = pd.read_parquet(store(root) / "sample.parquet")
    s["day"] = pd.to_datetime(s["day"]).dt.date
    return s


META = (
    "group",
    "day",
    "dir",
    "pnl",
    "win",
    "month",
    "dte",
    "before_1015",
    "symbol",
    "kind",
    "t_ns",
    "state",
)


def feature_cols(s: pd.DataFrame) -> list[str]:
    """去掉元数据、全空、常数、缺失 > 30% 的列。"""
    out = []
    for c in s.columns:
        if c in META:
            continue
        v = s[c].astype(float)
        if v.isna().mean() > 0.3 or v.nunique(dropna=True) < 2:
            continue
        out.append(c)
    return out


# ---------------- 单特征 ----------------


def group_pct(s: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每组（入场 + 50 个随机）内每个成员相对组内其他成员的百分位。返回
    (P: 组 × 成员 × 特征, entry_idx: 每组入场的位置, groups)。"""
    gs = sorted(s["group"].unique())
    m = max(s.groupby("group").size())
    P = np.full((len(gs), m, len(cols)), np.nan)
    eidx = np.zeros(len(gs), dtype=int)
    for a, g in enumerate(gs):
        sub = s[s["group"] == g]
        x = sub[cols].astype(float)
        r = x.rank(method="average")
        n = x.notna().sum()
        P[a, : len(sub)] = ((r - 1) / (n - 1).replace(0, np.nan)).to_numpy()
        eidx[a] = int(np.flatnonzero((sub["kind"] == "entry").to_numpy())[0])
    return P, eidx, np.array(gs)


def perm_pct(P: np.ndarray, eidx: np.ndarray, sizes: np.ndarray, rng) -> tuple:
    obs = np.nanmean(P[np.arange(len(eidx)), eidx], axis=0)
    null = np.empty((N_PERM, P.shape[2]))
    for k in range(N_PERM):
        pick = (rng.random(len(sizes)) * sizes).astype(int)
        null[k] = np.nanmean(P[np.arange(len(pick)), pick], axis=0)
    p = (1 + (np.abs(null - 0.5) >= np.abs(obs - 0.5) - 1e-12).sum(0)) / (N_PERM + 1)
    z = (obs - null.mean(0)) / null.std(0, ddof=1)
    return obs, p, z


def auc(y: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(x)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return math.nan
    return float(roc_auc_score(y[m], x[m]))


def univariate(s: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    from scipy import stats

    rng = np.random.default_rng(SEED)
    out = {}
    for task, sub in (
        ("A", s[s["win"]]),
        ("C", s),
        ("A_long", s[s["win"] & (s["dir"] == 1)]),
        ("A_short", s[s["win"] & (s["dir"] == -1)]),
    ):
        P, eidx, gs = group_pct(sub, cols)
        sizes = sub.groupby("group").size().reindex(gs).to_numpy()
        obs, p, z = perm_pct(P, eidx, sizes, rng)
        y = (sub["kind"] == "entry").to_numpy().astype(int)
        X = sub[cols].astype(float).to_numpy()
        a = np.array([auc(y, X[:, j]) for j in range(len(cols))])
        ent, rnd = sub[sub["kind"] == "entry"], sub[sub["kind"] == "random"]
        df = pd.DataFrame(
            {
                "feature": cols,
                "family": [ef.family_of(c) for c in cols],
                "n_entry": len(ent),
                "pct_mean": obs,
                "auc": a,
                "med_entry": ent[cols].median().to_numpy(),
                "med_random": rnd[cols].median().to_numpy(),
                "p_perm": p,
                "z_perm": z,
                "p_z": 2 * stats.norm.sf(np.abs(z)),
            }
        )
        df["bh_perm"] = v4d.bh(df["p_perm"].to_numpy(), Q)
        df["bh_z"] = v4d.bh(df["p_z"].fillna(1).to_numpy(), Q)
        out[task] = df
    # B：盈利 vs 亏损入场（打乱标签）
    e = s[s["kind"] == "entry"]
    y = e["win"].to_numpy().astype(int)
    X = e[cols].astype(float).to_numpy()
    obs = np.array([auc(y, X[:, j]) for j in range(len(cols))])
    null = np.empty((N_PERM, len(cols)))
    for k in range(N_PERM):
        yp = rng.permutation(y)
        null[k] = [auc(yp, X[:, j]) for j in range(len(cols))]
    p = (1 + (np.abs(null - 0.5) >= np.abs(obs - 0.5) - 1e-12).sum(0)) / (N_PERM + 1)
    z = (obs - np.nanmean(null, 0)) / np.nanstd(null, 0, ddof=1)
    df = pd.DataFrame(
        {
            "feature": cols,
            "family": [ef.family_of(c) for c in cols],
            "auc": obs,
            "med_win": e.loc[e["win"], cols].median().to_numpy(),
            "med_lose": e.loc[~e["win"], cols].median().to_numpy(),
            "p_perm": p,
            "z_perm": z,
            "p_z": 2 * stats.norm.sf(np.abs(z)),
        }
    )
    df["bh_perm"] = v4d.bh(df["p_perm"].to_numpy(), Q)
    df["bh_z"] = v4d.bh(df["p_z"].fillna(1).to_numpy(), Q)
    out["B"] = df
    return out


# ---------------- 模型 ----------------


def _hgb():
    return HistGradientBoostingClassifier(
        max_leaf_nodes=8,
        max_depth=3,
        min_samples_leaf=10,
        l2_regularization=10.0,
        max_iter=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=SEED,
    )


def _l1():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            l1_ratio=1.0, C=0.05, solver="liblinear", class_weight="balanced", max_iter=2000
        ),
    )


MODELS = {"hgb": _hgb, "l1": _l1}


def lodo(X: np.ndarray, y: np.ndarray, days: np.ndarray, model: str) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for d in np.unique(days):
        te = days == d
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        m = MODELS[model]()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def task_data(s: pd.DataFrame, cols: list[str], task: str):
    if task == "A":
        sub = s[s["win"]]
        y = (sub["kind"] == "entry").to_numpy().astype(int)
    elif task == "C":
        sub = s
        y = (sub["kind"] == "entry").to_numpy().astype(int)
    else:  # B
        sub = s[s["kind"] == "entry"]
        y = sub["win"].to_numpy().astype(int)
    return sub, sub[cols].astype(float).to_numpy(), y, np.array([str(d) for d in sub["day"]])


def within_pct(pred: np.ndarray, sub: pd.DataFrame, y: np.ndarray) -> float:
    """入场时刻的模型分数在自身随机对照中的平均百分位（A / C）。"""
    out = []
    g = sub["group"].to_numpy()
    for k in np.unique(g):
        m = g == k
        pe, pr = pred[m][y[m] == 1], pred[m][y[m] == 0]
        if len(pe) and len(pr) and np.isfinite(pe[0]):
            out.append(((pr < pe[0]).sum() + 0.5 * (pr == pe[0]).sum()) / len(pr))
    return float(np.mean(out)) if out else math.nan


def _shuffle(sub: pd.DataFrame, y: np.ndarray, task: str, rng) -> np.ndarray:
    if task == "B":
        return rng.permutation(y)
    yp = np.zeros_like(y)
    g = sub["group"].to_numpy()
    for k in np.unique(g):
        idx = np.flatnonzero(g == k)
        yp[rng.choice(idx)] = 1
    return yp


def _perm_one(X, y, days, sub, task, model, seed):
    yp = _shuffle(sub, y, task, np.random.default_rng(seed))
    pred = lodo(X, yp, days, model)
    return auc(yp, pred), (within_pct(pred, sub, yp) if task != "B" else math.nan)


def models(s: pd.DataFrame, cols: list[str], n_jobs: int = 12) -> dict:
    out = {}
    for task in ("A", "B", "C"):
        sub, X, y, days = task_data(s, cols, task)
        for model in MODELS:
            pred = lodo(X, y, days, model)
            a, wp = auc(y, pred), (within_pct(pred, sub, y) if task != "B" else math.nan)
            null = Parallel(n_jobs=n_jobs)(
                delayed(_perm_one)(X, y, days, sub, task, model, SEED + 1 + k)
                for k in range(N_PERM)
            )
            na = np.array([x[0] for x in null])
            nw = np.array([x[1] for x in null])
            rec = {
                "task": task,
                "model": model,
                "n_pos": int(y.sum()),
                "n": len(y),
                "oos_auc": a,
                "p_auc": float((1 + (na >= a - 1e-12).sum()) / (N_PERM + 1)),
                "null_auc_mean": float(np.nanmean(na)),
                "null_auc_p95": float(np.nanquantile(na, 0.95)),
            }
            if task != "B":
                rec.update(
                    {
                        "oos_within_pct": wp,
                        "p_within": float((1 + (nw >= wp - 1e-12).sum()) / (N_PERM + 1)),
                    }
                )
            out[(task, model)] = {"summary": rec, "pred": pred}
            log.info("模型 %s %s：%s", task, model, rec)
    return out


def oof_importance(s: pd.DataFrame, cols: list[str], task: str = "A", reps: int = 3):
    """留一日交叉验证中，逐特征在测试折内打乱，看样本外 AUC 下降（HGB）。"""
    sub, X, y, days = task_data(s, cols, task)
    fitted = {}
    for d in np.unique(days):
        tr = days != d
        m = _hgb()
        m.fit(X[tr], y[tr])
        fitted[d] = m
    base = np.full(len(y), np.nan)
    for d, m in fitted.items():
        te = days == d
        base[te] = m.predict_proba(X[te])[:, 1]
    a0 = auc(y, base)
    rng = np.random.default_rng(SEED)
    drops = np.zeros(len(cols))
    for j in range(len(cols)):
        acc = []
        for _ in range(reps):
            pred = np.full(len(y), np.nan)
            for d, m in fitted.items():
                te = np.flatnonzero(days == d)
                Xt = X[te].copy()
                Xt[:, j] = rng.permutation(Xt[:, j])
                pred[te] = m.predict_proba(Xt)[:, 1]
            acc.append(a0 - auc(y, pred))
        drops[j] = np.mean(acc)
    return pd.DataFrame({"feature": cols, "auc_drop": drops}).sort_values(
        "auc_drop", ascending=False
    ), a0


def tree_rules(s: pd.DataFrame, cols: list[str], task: str = "A") -> tuple[pd.DataFrame, float]:
    sub, X, y, days = task_data(s, cols, task)
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    t = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=SEED
    ).fit(Xi, y)
    # 留一日
    pred = np.full(len(y), np.nan)
    for d in np.unique(days):
        te = days == d
        imp = SimpleImputer(strategy="median").fit(X[~te])
        m = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=SEED
        ).fit(imp.transform(X[~te]), y[~te])
        pred[te] = m.predict_proba(imp.transform(X[te]))[:, 1]
    tr = t.tree_
    leaf = t.apply(Xi)
    rows = []

    def walk(node, path):
        if tr.children_left[node] == -1:
            m = leaf == node
            rows.append(
                {
                    "rule": " 且 ".join(path) or "（全部）",
                    "入场覆盖": int(y[m].sum()),
                    "入场占比": y[m].sum() / y.sum(),
                    "随机占比": (m & (y == 0)).sum() / (y == 0).sum(),
                }
            )
            return
        f, thr = cols[tr.feature[node]], tr.threshold[node]
        walk(tr.children_left[node], [*path, f"{f} ≤ {thr:.4g}"])
        walk(tr.children_right[node], [*path, f"{f} > {thr:.4g}"])

    walk(0, [])
    df = pd.DataFrame(rows)
    df["倍数"] = df["入场占比"] / df["随机占比"].replace(0, np.nan)
    return df.sort_values("倍数", ascending=False), auc(y, pred)


# ---------------- 候选规则 ----------------


def _cands(uni: pd.DataFrame, fams: tuple, k: int, q_lo=(0.2, 0.3), q_hi=(0.7, 0.8)):
    """按任务 A 置换 z 绝对值选前 k 个特征；方向由 z 的符号决定（入场偏高 → ≥）。"""
    u = uni[uni["family"].isin(fams)].copy()
    u = u.reindex(u["z_perm"].abs().sort_values(ascending=False).index).head(k)
    return [
        (r.feature, ">=" if r.z_perm > 0 else "<=", q_lo if r.z_perm > 0 else q_hi)
        for r in u.itertuples()
    ]


def _cond(x: np.ndarray, op: str, v: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return x >= v if op == ">=" else x <= v


def search_rules(s: pd.DataFrame, uni: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """模板：趋势（TREND_DEF 中按任务 A 的 z 取前 k 个，阈值 > 0）+ 触发（秒级 / 当前 K /
    暴力因子）+ 可选不做条件（期权 / 日历 / 量 / 跨标的）。触发与不做的阈值 = 盈利入场该特征的
    分位数（≥ 用 20% / 30% 分位，≤ 用 70% / 80% 分位）。"""
    A = s[s["win"]]
    ent, rnd = A[A["kind"] == "entry"], A[A["kind"] == "random"]
    lose = s[(s["kind"] == "entry") & ~s["win"]]
    ut = uni[uni["feature"].isin(TREND_DEF)].sort_values("z_perm", ascending=False).head(k)
    tg, nt = (_cands(uni, f, k) for f in (TRIGGER_FAMS, NOTRADE_FAMS))

    def opts(lst):
        for f, op, qs in lst:
            for q in qs:
                yield f, op, float(ent[f].quantile(q))

    T = []
    for f in ut["feature"]:  # 趋势：> 0 或盈利入场 20% / 30% 分位（取较大者）
        for v in sorted({1e-9, *(max(float(ent[f].quantile(q)), 1e-9) for q in (0.2, 0.3))}):
            T.append((f, ">=", v))
    G, N = list(opts(tg)), [None, *opts(nt)]
    cache = {}

    def m(df, c):
        key = (id(df), c)
        if key not in cache:
            cache[key] = _cond(df[c[0]].astype(float).to_numpy(), c[1], c[2])
        return cache[key]

    rows = []
    for a, b, c in itertools.product(T, G, N):
        if a[0] == b[0]:
            continue
        parts = [x for x in (a, b, c) if x is not None]
        me = np.logical_and.reduce([m(ent, p) for p in parts])
        mr = np.logical_and.reduce([m(rnd, p) for p in parts])
        ml = np.logical_and.reduce([m(lose, p) for p in parts])
        rows.append(
            {
                "trend": a,
                "trigger": b,
                "no_trade": c,
                "win_cover": int(me.sum()),
                "win_share": me.mean(),
                "rand_share": mr.mean(),
                "lose_cover": int(ml.sum()),
                "lose_share": ml.mean() if len(ml) else math.nan,
            }
        )
    r = pd.DataFrame(rows)
    r["lift"] = (r["win_share"] + 0.5 / len(ent)) / (r["rand_share"] + 0.5 / len(rnd))
    return r


def pick_rules(r: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    ok = r[(r["win_cover"] >= MIN_WIN_COVER) & (r["win_share"] > r["rand_share"])]
    ok = ok.sort_values(["lift", "win_cover"], ascending=False)
    out, used = [], set()
    for row in ok.itertuples():
        key = row.trigger[0]
        if key in used or row.trend[0] in {x.trend[0] for x in out}:
            continue
        out.append(row)
        used.add(key)
        if len(out) >= n:
            break
    return pd.DataFrame(out)


def rule_cv(s: pd.DataFrame, cols: list[str], k: int = 8) -> pd.DataFrame:
    """规则搜索流程的留一日检验：在其余日子上做单特征统计与模板搜索、取倍数最高的规则，
    在留出日上看盈利入场与随机时刻的触发比例。"""
    rows = []
    for d in sorted(s["day"].unique()):
        tr, te = s[s["day"] != d], s[s["day"] == d]
        if not (te["win"] & (te["kind"] == "entry")).any():
            continue
        uni = _uni_fast(tr, cols)
        r = search_rules(tr, uni, k)
        ok = r[(r["win_cover"] >= MIN_WIN_COVER) & (r["win_share"] > r["rand_share"])]
        if ok.empty:
            continue
        best = ok.sort_values(["lift", "win_cover"], ascending=False).iloc[0]
        A = te[te["win"]]
        ent, rnd = A[A["kind"] == "entry"], A[A["kind"] == "random"]
        parts = [x for x in (best["trend"], best["trigger"], best["no_trade"]) if x is not None]
        me = np.logical_and.reduce(
            [_cond(ent[p[0]].astype(float).to_numpy(), p[1], p[2]) for p in parts]
        )
        mr = np.logical_and.reduce(
            [_cond(rnd[p[0]].astype(float).to_numpy(), p[1], p[2]) for p in parts]
        )
        rows.append(
            {
                "day": str(d),
                "rule": rule_text(best),
                "in_lift": best["lift"],
                "test_wins": len(ent),
                "test_win_cover": int(me.sum()),
                "test_rand": len(rnd),
                "test_rand_cover": int(mr.sum()),
            }
        )
    return pd.DataFrame(rows)


def _uni_fast(s: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """rule_cv 用：任务 A 组内百分位均值的 z（按组内百分位的抽样分布近似，不做置换）。"""
    sub = s[s["win"]]
    P, eidx, gs = group_pct(sub, cols)
    obs = np.nanmean(P[np.arange(len(eidx)), eidx], axis=0)
    n = np.sum(np.isfinite(P[np.arange(len(eidx)), eidx]), axis=0)
    z = (obs - 0.5) / np.sqrt(1 / 12 / np.maximum(n, 1))
    return pd.DataFrame({"feature": cols, "family": [ef.family_of(c) for c in cols], "z_perm": z})


def rule_text(r) -> str:
    def one(c):
        return f"{c[0]} {c[1]} {c[2]:.4g}"

    s = f"趋势：{one(r['trend'])}；触发：{one(r['trigger'])}"
    if r["no_trade"] is not None:
        op = "<" if r["no_trade"][1] == ">=" else ">"
        s += f"；不做：{r['no_trade'][0]} {op} {r['no_trade'][2]:.4g}"
    return s


def rule_yaml(k: int, r, s: pd.DataFrame) -> dict:
    conds = []
    for role, c in (("trend", r["trend"]), ("trigger", r["trigger"]), ("no_trade", r["no_trade"])):
        if c is None:
            continue
        conds.append(
            {
                "feature": "ef",
                "name": c[0],
                "op": c[1],
                "value": round(float(c[2]), 6),
                "role": role,
            }
        )
    return {
        "name": f"owner_rule_{k}",
        "description": "负责人入场挖掘（OWNER_ENTRY_MINING.md）模板搜索所得；双向对称：特征已按"
        "方向化，"
        "同一组条件对看涨（d=+1）与看跌（d=−1）分别评估。no_trade 条件写成「满足才做」的形式。",
        "underlying": "MU",
        "convention": "C",
        "direction": "both",
        "entry": {"all": conds},
        "session": {"first_entry": "09:32", "last_entry": "15:40", "flat": "15:50"},
        "contract": {"nearest_expiry": True, "otm_steps": 1, "no_0dte_after": "10:15"},
        "exit": {"tp": 0.10, "sl": 0.15, "max_hold_min": 30, "basis": "mid"},
        "executions": {
            "owner": {"entry": {"mode": "owner"}, "exit": {"mode": "owner"}},
            "taker": {
                "entry": {"mode": "taker", "lag_s": 2.0},
                "exit": {"mode": "taker", "lag_s": 2.0},
            },
        },
        "qty": 1,
        "in_sample": {
            "data": "MU D-030 开仓买入委托 2026-07-07～2026-08-06（不得用于检验，D-033）",
            "win_cover": int(r["win_cover"]),
            "win_share": round(float(r["win_share"]), 4),
            "random_share": round(float(r["rand_share"]), 4),
            "lose_cover": int(r["lose_cover"]),
            "lift": round(float(r["lift"]), 3),
        },
    }


# ---------------- 主流程 ----------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    st = store(root)
    stages = (
        ["sample", "stats", "models", "rules", "report"] if args.stage == "all" else [args.stage]
    )
    if "sample" in stages:
        build_sample(root)
    s = load_sample(root)
    cols = feature_cols(s)
    if "stats" in stages:
        uni = univariate(s, cols)
        for k, v in uni.items():
            v.to_parquet(st / f"uni_{k}.parquet", index=False)
    if "models" in stages:
        res = models(s, cols, args.jobs)
        pd.DataFrame([v["summary"] for v in res.values()]).to_parquet(
            st / "models.parquet", index=False
        )
        imp, a0 = oof_importance(s, cols)
        imp.assign(base_auc=a0).to_parquet(st / "importance.parquet", index=False)
        for task in ("A", "B", "C"):
            tr, ta = tree_rules(s, cols, task)
            tr.assign(lodo_auc=ta).to_parquet(st / f"tree_{task}.parquet", index=False)
    if "rules" in stages:
        uni = pd.read_parquet(st / "uni_A.parquet")
        r = search_rules(s, uni)
        r.to_pickle(st / "rule_search.pkl")
        pk = pick_rules(r)
        pk.to_pickle(st / "rule_pick.pkl")
        cv = rule_cv(s, cols)
        cv.to_parquet(st / "rule_cv.parquet", index=False)
        for old in RULE_DIR.glob("owner_rule_*.yaml"):
            old.unlink()
        for k, row in enumerate(pk.to_dict("records"), start=1):
            (RULE_DIR / f"owner_rule_{k}.yaml").write_text(
                yaml.safe_dump(rule_yaml(k, row, s), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        log.info("规则：%d 条（搜索 %d 条）", len(pk), len(r))
    if "report" in stages:
        from zr.research import owner_entry_report

        owner_entry_report.write(root, s, cols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
