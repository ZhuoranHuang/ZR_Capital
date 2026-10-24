"""R-050：以扣费后盈亏为标签的机会挖掘（预登记见 RESEARCH_LOG R-050）→
PROFIT_LABEL_MINING.md 第 3 节。

候选：09:32–15:40 内当前秒级价相对 [t−120 s, t−30 s] 的最低（看涨）/ 最高（看跌）已朝该方向走了
≥ 0.05% 的秒；同方向 120 秒内只取第一个。每个候选独立模拟一笔（不受持仓占用限制）：R-041 负责人
挂单策略入场，最近到期日虚值一档（D-001），1 张，费用 D-013；出场 (a) 负责人习惯（+10% / −15% /
30 分钟，止损吃买一）；(b) 出场挖掘选出的规则（exit_mining，选择标准预先固定）。
标签 = 扣费后净盈亏占权利金 %；未成交 = 未入场。训练 2025；验证 2026-01～05（只报告）；测试
2026-09～10（只跑一次）。模型 = HGB 回归 / 分类（本机无 LightGBM）。
运行：uv run python -m zr.research.profit_label_mining [--stage sim|models|report|all]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier

from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.data.calendar import trading_days
from zr.features import entry_features as ef
from zr.features.bars import ET
from zr.research import exit_mining as xm

log = logging.getLogger("zr.research.profit_label_mining")
NS = 1_000_000_000
MOVE = 0.0005
W_LO, W_HI = 30, 120
REFRACTORY = 120
SEED = 0
RANGES = {
    "train": (dt.date(2025, 1, 2), dt.date(2025, 12, 31)),
    "val": (dt.date(2026, 1, 2), dt.date(2026, 5, 31)),
    "test": (dt.date(2026, 9, 1), dt.date(2026, 10, 21)),
}
CONTRACT = {"contract": {"nearest_expiry": True, "otm_steps": 1, "no_0dte_after": "10:15"}}
EXITS = ("habit", "rule")
META = (
    "day",
    "t_ns",
    "dir",
    "symbol",
    "split",
    *(f"{k}_{x}" for x in EXITS for k in ("filled", "net_usd", "net_pct", "entry_px", "hold_s")),
)


def store(root: Path) -> Path:
    p = root / "research" / "profit_label_mining"
    (p / "days").mkdir(parents=True, exist_ok=True)
    return p


def split_of(day: dt.date) -> str | None:
    for k, (a, b) in RANGES.items():
        if a <= day <= b:
            return k
    return None


def all_days(root: Path) -> list[dt.date]:
    base = root / "intrabar" / "underlying=MU"
    out = []
    for a, b in RANGES.values():
        out += [d for d in trading_days(a, b) if (base / f"date={d}" / "S.parquet").exists()]
    ib.check_out_of_sample(out)
    return out


def candidates(S: np.ndarray, in_win: np.ndarray) -> list[tuple[int, int]]:
    s = pd.Series(S)
    w = W_HI - W_LO + 1
    lo = s.rolling(w, min_periods=w // 2).min().shift(W_LO).to_numpy()
    hi = s.rolling(w, min_periods=w // 2).max().shift(W_LO).to_numpy()
    with np.errstate(invalid="ignore"):
        up = (S / lo - 1 >= MOVE) & in_win
        dn = (1 - S / hi >= MOVE) & in_win
    out = []
    for d, m in ((1, up), (-1, dn)):
        last = -(10**9)
        for k in np.flatnonzero(m):
            if k - last > REFRACTORY:
                out.append((int(k), d))
                last = k
    return sorted(out)


def simulate(
    book, sym: str, t: np.ndarray, k: int, d: int, day: dt.date, flat_k: int, fs: dict, rule, rng
) -> dict:
    """一笔：R-041 入场 → 出场 (a) 习惯 / (b) 规则。返回两版的成交与扣费后盈亏。"""
    out = {}
    t0 = int(t[k])
    flat_ns = int(t[flat_k])
    n, px0, te, _ = ib.place(book, 1, 1.0, t0, {"mode": "owner"}, sym, 0.9, flat_ns, rng)
    if n <= 0 or te is None:
        for x in EXITS:
            out.update({f"filled_{x}": 0.0, f"net_usd_{x}": 0.0, f"net_pct_{x}": math.nan})
        return out
    i0 = int(np.searchsorted(t, te, side="right"))
    stop = flat_k
    if ib.parse_contract(sym)["expiry"] == day:  # D-001：0DTE 10:45 前清仓
        stop = min(stop, int(np.searchsorted(t, pd.Timestamp(f"{day} 10:45", tz=ET).value)))
    cols = xm.state_arrays(book, t, i0, px0)
    for name in {c[0] for c in rule.get("conds", []) if c[0] not in xm.STATE}:
        cols[name] = fs[name].directional(np.full(len(t), d))
    for x, r in (("habit", xm.HABIT), ("rule", rule)):
        j = xm.fires(r, cols, i0, stop) if i0 < stop else None
        forced = j is None
        j = stop if forced else j
        tdec = int(t[min(j, len(t) - 1)])
        is_sl = (
            x == "habit"
            and np.isfinite(cols["st_float_pct"][j])
            and cols["st_float_pct"][j] <= -r["sl"]
        )
        order = {"mode": "taker", "lag_s": 2.0} if (forced or is_sl) else {"mode": "owner"}
        xn, px1, _, _ = ib.place(book, -1, 1.0, tdec, order, sym, 0.9, None, rng)
        if xn < 1:
            c = ib.cross_now(book, -1, tdec, 2.0)
            px1 = c[1] if c else (px1 if np.isfinite(px1) else 0.0)
        cash = -px0 * 100 - ib.fees(1, px0, "Buy") + px1 * 100 - ib.fees(1, px1, "Sell")
        out.update(
            {
                f"filled_{x}": 1.0,
                f"net_usd_{x}": cash,
                f"net_pct_{x}": cash / (px0 * 100) * 100,
                f"entry_px_{x}": px0,
                f"hold_s_{x}": (tdec - te) / NS,
            }
        )
    return out


def run_day(root: Path, ctx, od, day: dt.date, rule: dict) -> pd.DataFrame:
    g = ib.second_grid(root, "MU", day)
    dd = ef.load_day(ctx, day)
    if g is None or dd is None:
        return pd.DataFrame()
    S = g["S"].to_numpy()
    t = g["t"].to_numpy()
    hm = np.asarray(pd.DatetimeIndex(t).tz_localize("UTC").tz_convert(ET).strftime("%H:%M"))
    in_win = (hm >= "09:32") & (hm <= "15:40")
    cand = candidates(S, in_win)
    if not cand:
        return pd.DataFrame()
    fs = ef.day_features(dd)
    flat_k = int(np.searchsorted(t, pd.Timestamp(f"{day} 15:50", tz=ET).value))
    rng = np.random.default_rng([SEED, day.toordinal()])
    rows = []
    et = pd.DatetimeIndex(t).tz_localize("UTC").tz_convert(ET)
    for k, d in cand:
        sym = od.pick(day, et[k].time(), d, float(S[k]), CONTRACT)
        rec = {"day": str(day), "t_ns": int(t[k]), "dir": d, "symbol": sym, "split": split_of(day)}
        bk = None
        if sym is not None:
            bk = od.book(
                sym,
                day,
                pd.Timestamp(int(t[k]), tz="UTC") - pd.Timedelta(minutes=1),
                pd.Timestamp(int(t[flat_k]), tz="UTC") + pd.Timedelta(minutes=3),
            )
        if bk is None:
            for x in EXITS:
                rec.update({f"filled_{x}": 0.0, f"net_usd_{x}": 0.0, f"net_pct_{x}": math.nan})
        else:
            rec.update(simulate(bk, sym, t, k, d, day, flat_k, fs, rule, rng))
        rows.append(rec)
    x = ef.sample(fs, np.array([k for k, _ in cand]), np.array([d for _, d in cand]))
    return pd.concat([pd.DataFrame(rows), x], axis=1)


def _shard(root: Path, days: list, rule: dict) -> int:
    ctx = ef.Ctx.load(root)
    od = ib.OptionData(root, "MU")
    n = 0
    for day in days:
        p = store(root) / "days" / f"{day}.parquet"
        if p.exists():
            continue
        df = run_day(root, ctx, od, day, rule)
        df.to_parquet(p, index=False)
        n += len(df)
        log.info("%s：候选 %d", day, len(df))
    return n


def run_sim(root: Path, rule: dict, jobs: int) -> None:
    days = all_days(root)
    shards = [days[k::jobs] for k in range(jobs)]
    Parallel(n_jobs=jobs)(delayed(_shard)(root, sh, rule) for sh in shards)


def load(root: Path) -> pd.DataFrame:
    fs = sorted((store(root) / "days").glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs if f.stat().st_size > 0], ignore_index=True)


# ---------------- 模型与评判 ----------------


def _models():
    kw = {
        "max_leaf_nodes": 8,
        "max_depth": 3,
        "min_samples_leaf": 10,
        "l2_regularization": 10.0,
        "max_iter": 200,
        "learning_rate": 0.05,
        "random_state": SEED,
    }
    return {"reg": HistGradientBoostingRegressor(**kw), "clf": HistGradientBoostingClassifier(**kw)}


def feature_cols(df: pd.DataFrame) -> list[str]:
    tr = df[df["split"] == "train"]
    return [
        c
        for c in df.columns
        if c not in META
        and tr[c].astype(float).isna().mean() <= 0.3
        and tr[c].nunique(dropna=True) >= 2
    ]


def _max_dd(x: pd.DataFrame) -> float:
    c = x.sort_values("t_ns")["pnl"].cumsum().to_numpy()
    if not len(c):
        return 0.0
    c = np.r_[0.0, c]
    return float((np.maximum.accumulate(c) - c).max())


def evaluate(test: pd.DataFrame, score: np.ndarray, exit_: str, rng) -> pd.DataFrame:
    """测试期：按分数取前 10% / 20%，对照全部候选与随机同数量（1,000 次）。未入场按 0 计。"""
    t = test.assign(score=score, pnl=test[f"net_usd_{exit_}"].fillna(0.0))
    n_days = t["day"].nunique()
    rows = []
    for grp, sub in (("合并", t), ("多", t[t["dir"] == 1]), ("空", t[t["dir"] == -1])):
        if sub.empty:
            continue
        for lab, frac in (("前 10%", 0.1), ("前 20%", 0.2), ("全部候选", 1.0)):
            m = max(1, int(round(len(sub) * frac)))
            pick = sub.nlargest(m, "score")
            filled = pick[f"filled_{exit_}"] > 0
            pnl = pick["pnl"].to_numpy()
            rnd = np.array(
                [rng.choice(sub["pnl"].to_numpy(), m, replace=False).mean() for _ in range(1000)]
            )
            rows.append(
                {
                    "exit": exit_,
                    "组": grp,
                    "选取": lab,
                    "候选数": m,
                    "成交率": float(filled.mean()),
                    "每笔平均 USD（含未入场 0）": float(pnl.mean()),
                    "成交笔平均 % 权利金": float(pick.loc[filled, f"net_pct_{exit_}"].mean()),
                    "合计 USD": float(pnl.sum()),
                    "胜率（成交笔）": float((pick.loc[filled, "pnl"] > 0).mean())
                    if filled.any()
                    else math.nan,
                    "最大回撤 USD": _max_dd(pick),
                    "每日笔数": float(filled.sum() / n_days),
                    "随机同数量每笔均值 P50": float(np.median(rnd)),
                    "随机 P95": float(np.quantile(rnd, 0.95)),
                    "相对随机分位": float((rnd < pnl.mean()).mean()),
                }
            )
    return pd.DataFrame(rows)


def fit_eval(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {"eval": [], "val": [], "importance": [], "tree": []}
    rng = np.random.default_rng(SEED)
    for x in EXITS:
        tr = df[(df["split"] == "train") & (df[f"filled_{x}"] > 0)]
        va, te = df[df["split"] == "val"], df[df["split"] == "test"]
        Xtr, ytr = tr[cols].astype(float).to_numpy(), tr[f"net_pct_{x}"].to_numpy()
        for kind, mdl in _models().items():
            y = ytr if kind == "reg" else (ytr > 0).astype(int)
            mdl.fit(Xtr, y)

            def sc(d, mdl=mdl, kind=kind):
                X = d[cols].astype(float).to_numpy()
                return mdl.predict(X) if kind == "reg" else mdl.predict_proba(X)[:, 1]

            out["val"].append(evaluate(va, sc(va), x, rng).assign(model=kind))
            out["eval"].append(evaluate(te, sc(te), x, rng).assign(model=kind))
            vf = va[va[f"filled_{x}"] > 0]
            yv = vf[f"net_pct_{x}"].to_numpy()
            pi = permutation_importance(
                mdl,
                vf[cols].astype(float).to_numpy(),
                yv if kind == "reg" else (yv > 0).astype(int),
                n_repeats=3,
                random_state=SEED,
                n_jobs=4,
            )
            imp = pd.DataFrame({"feature": cols, "drop": pi.importances_mean}).nlargest(10, "drop")
            out["importance"].append(imp.assign(exit=x, model=kind))
        # 深度 ≤ 3 决策树（训练期，标签 = 盈利），叶规则在测试期的扣费后盈亏
        imp_ = SimpleImputer(strategy="median").fit(Xtr)
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=SEED).fit(
            imp_.transform(Xtr), (ytr > 0).astype(int)
        )
        out["tree"].append(_leaves(tree, cols, imp_, tr, te, x))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def _leaves(tree, cols, imp_, tr, te, x) -> pd.DataFrame:
    T = tree.tree_
    rows = []

    def walk(node, path):
        if T.children_left[node] == -1:
            rows.append({"node": node, "rule": " 且 ".join(path) or "（全部）"})
            return
        f, thr = cols[T.feature[node]], T.threshold[node]
        walk(T.children_left[node], [*path, f"{f} ≤ {thr:.4g}"])
        walk(T.children_right[node], [*path, f"{f} > {thr:.4g}"])

    walk(0, [])
    lt = tree.apply(imp_.transform(tr[cols].astype(float).to_numpy()))
    lte = tree.apply(imp_.transform(te[cols].astype(float).to_numpy()))
    out = []
    for r in rows:
        a, b = tr[lt == r["node"]], te[lte == r["node"]]
        bf = b[b[f"filled_{x}"] > 0]
        out.append(
            {
                "exit": x,
                "规则": r["rule"],
                "训练期成交笔": len(a),
                "训练期均值 %": float(a[f"net_pct_{x}"].mean()),
                "测试期候选": len(b),
                "测试期成交": len(bf),
                "测试期每笔 USD": float(bf[f"net_usd_{x}"].mean()) if len(bf) else math.nan,
                "测试期合计 USD": float(bf[f"net_usd_{x}"].sum()),
                "测试期胜率": float((bf[f"net_usd_{x}"] > 0).mean()) if len(bf) else math.nan,
            }
        )
    return pd.DataFrame(out).sort_values("训练期均值 %", ascending=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    st = store(root)
    stages = ["sim", "models"] if args.stage == "all" else [args.stage]
    if "sim" in stages:
        rule = pd.read_pickle(xm.store(root) / "chosen_rule.pkl")
        run_sim(root, rule, args.jobs)
    if "models" in stages:
        df = load(root)
        cols = feature_cols(df)
        res = fit_eval(df, cols)
        for k, v in res.items():
            v.to_parquet(st / f"{k}.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
