"""出场规律挖掘（负责人指令四步队列第 2 步；PROFIT_LABEL_MINING.md 第 2 节）。

事件 = D-030 入选段中 MU 持仓的每张卖出委托的成交时刻（「出场事件」；51 张卖单中
11 张挂单超过 60 秒才成交，提交时刻不是卖出时刻）；对照 = 同一
持仓期间（买入成交后、最后一张卖单成交前）每 10 秒一个仍持有、未卖出的时刻（剔除卖单
成交前后 10 秒）。盈利段与亏损段分开（任务 W / L），另报合并（任务 ALL）。
特征：入场挖掘同一套方向化特征（zr/features/entry_features.py，方向 = 持仓方向）+
持仓状态（该段所持合约 ThetaData 逐笔中间价）：已持有秒数、当前浮盈 %（中间价 /
买入均价 − 1）、最高浮盈 %、已回吐比例（(最高中间价 − 当前) / (最高中间价 − 买入价)，
最高未超过买入价时为 0）、距最高点秒数。
统计：单特征 AUC、段内百分位均值（卖出时刻在本段持有时刻中的位置）+ 段内打乱标签的
置换检验 500 次（经验 p，BH q = 0.10）；模型同入场挖掘（HGB / L1，按交易日留一，置换
500 次）。规则：模板「持仓状态条件 +（可选）市场条件」，阈值取出场事件的分位数，按
出场事件覆盖与持有时刻占比之比取 3–5 条；在 48 笔真实入场上用规则替换实际出场（真实
买入成交价与张数不变；规则首次成立的秒按负责人挂单策略 R-041 卖出全部张数，另报吃
买一版本；15:50 清仓兜底），比较扣费后盈亏。
这是 2026-06～08（D-033 校准窗口）上的样本内挖掘与替换，不是检验；规则的样本外表现
见 R-050。
运行：uv run python -m zr.research.exit_mining [--stage sample|stats|models|rules|all]
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.features import entry_features as ef
from zr.features.bars import ET
from zr.research import owner_entry_mining as oem
from zr.research import owner_limit as ol

log = logging.getLogger("zr.research.exit_mining")
NS = 1_000_000_000
STEP = 10
EXCL = 10
N_PERM = 500
N_PERM_TREE = 200  # 出场模型的梯度提升树只置换 200 次（负责人 2026-10-24 指令；入场挖掘仍为 500）
SEED = 0
STATE = ("st_hold_s", "st_float_pct", "st_max_float_pct", "st_giveback", "st_since_max_s")
HABIT = {"name": "habit", "tp": 10.0, "sl": 15.0, "cap_min": 30.0}


def store(root: Path) -> Path:
    p = root / "research" / "exit_mining"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- 持仓与状态 ----------------


def episodes(root: Path) -> list[dict]:
    sel, orders = ol.episode_orders(root)
    out = []
    for ep, g in orders[orders["symbol"].str.startswith("MU")].groupby("episode"):
        b, s = g[g["side"] == "Buy"], g[g["side"] == "Sell"].sort_values("submit_ns")
        if b.empty or s.empty:
            continue
        sym = b["symbol"].iloc[0]
        out.append(
            {
                "episode": int(ep),
                "symbol": sym,
                "dir": 1 if ib.parse_contract(sym)["right"] == "call" else -1,
                "day": pd.Timestamp(int(b["fill_ns"].iloc[0]), tz="UTC").tz_convert(ET).date(),
                "entry_ns": int(b["fill_ns"].max()),
                "entry_px": float((b["act_px"] * b["qty"]).sum() / b["qty"].sum()),
                "qty": float(b["qty"].sum()),
                "buy_fees": float(
                    sum(ib.fees(q, p, "Buy") for q, p in zip(b["qty"], b["act_px"], strict=True))
                ),
                "sells": [int(x) for x in s["fill_ns"]],  # 出场事件 = 卖单成交时刻（卖出那一刻）
                "sell_qty": [float(x) for x in s["qty"]],
                "sell_px": [float(x) for x in s["act_px"]],
                "last_sell_fill_ns": int(s["fill_ns"].max()),
                "pnl": float(sel.loc[ep, "pnl_net"]),
            }
        )
    for e in out:
        e["win"] = e["pnl"] > 0
    return out


def state_arrays(book, t: np.ndarray, i0: int, entry_px: float) -> dict[str, np.ndarray]:
    """网格 t[i0:] 上的持仓状态（i0 = 买入成交所在秒的下一秒）。"""
    tt = t[i0:]
    j = np.searchsorted(book.q_ns, tt, side="right") - 1
    mid = np.where(
        j >= 0, (book.bid[np.clip(j, 0, None)] + book.ask[np.clip(j, 0, None)]) / 2, np.nan
    )
    mid = pd.Series(mid).ffill().to_numpy()
    mx = np.fmax.accumulate(np.where(np.isfinite(mid), mid, -np.inf))
    arg = pd.Series(np.where(mid >= mx - 1e-12, np.arange(len(mid)), np.nan)).ffill().to_numpy()
    gain = mx - entry_px
    out = {
        "st_hold_s": np.arange(len(tt), dtype=float) + 1,
        "st_float_pct": (mid / entry_px - 1) * 100,
        "st_max_float_pct": (mx / entry_px - 1) * 100,
        "st_giveback": np.where(gain > 0, (mx - mid) / np.where(gain > 0, gain, 1), 0.0),
        "st_since_max_s": np.arange(len(tt)) - arg,
    }
    pad = {k: np.r_[np.full(i0, np.nan), v] for k, v in out.items()}
    pad["mid"] = np.r_[np.full(i0, np.nan), mid]
    return pad


# ---------------- 样本 ----------------


def build_sample(root: Path) -> pd.DataFrame:
    eps = episodes(root)
    ctx = ef.Ctx.load(root)
    od = ib.OptionData(root, "MU")
    rows = []
    for day in sorted({e["day"] for e in eps}):
        d = ef.load_day(ctx, day)
        fs = ef.day_features(d, tuple(f for f in ef.FAMILIES if f != "svy"))
        for e in [x for x in eps if x["day"] == day]:
            bk = od.book(
                e["symbol"],
                day,
                pd.Timestamp(e["entry_ns"], tz="UTC") - pd.Timedelta(minutes=1),
                pd.Timestamp(e["last_sell_fill_ns"], tz="UTC") + pd.Timedelta(minutes=5),
            )
            if bk is None:
                log.warning("段 %d 无报价", e["episode"])
                continue
            i0 = int(np.searchsorted(d.t, e["entry_ns"], side="right"))
            st = state_arrays(bk, d.t, i0, e["entry_px"])
            last = int(np.searchsorted(d.t, max(e["sells"]), side="right")) - 1
            sells = ef.grid_rows(d, np.array(e["sells"]))
            ctrl = np.arange(i0 + STEP, last, STEP)
            ctrl = (
                ctrl[np.min(np.abs(ctrl[:, None] - sells[None, :]), axis=1) > EXCL]
                if len(ctrl)
                else ctrl
            )
            idx = np.r_[sells, ctrl].astype(int)
            lab = np.r_[np.ones(len(sells)), np.zeros(len(ctrl))]
            x = ef.sample(fs, idx, np.full(len(idx), e["dir"]))
            for k in STATE:
                x[k] = st[k][idx]
            x["label"] = lab
            for k in ("episode", "dir", "win", "pnl", "symbol"):
                x[k] = e[k]
            x["day"] = str(day)
            x["t_ns"] = d.t[idx]
            rows.append(x)
        log.info("%s 完成", day)
    s = pd.concat(rows, ignore_index=True)
    s.to_parquet(store(root) / "sample.parquet", index=False)
    return s


META = ("label", "episode", "dir", "win", "pnl", "symbol", "day", "t_ns")


def feature_cols(s: pd.DataFrame) -> list[str]:
    return [
        c
        for c in s.columns
        if c not in META
        and s[c].astype(float).isna().mean() <= 0.3
        and s[c].nunique(dropna=True) >= 2
    ]


# ---------------- 统计与模型 ----------------


def _shuffle_within(y: np.ndarray, g: np.ndarray, rng) -> np.ndarray:
    yp = y.copy()
    for k in np.unique(g):
        m = np.flatnonzero(g == k)
        yp[m] = rng.permutation(y[m])
    return yp


def univariate(s: pd.DataFrame, cols: list[str], task: str) -> pd.DataFrame:
    from zr.research import v4_decisive as v4d

    sub = _task(s, task)
    y, g = sub["label"].to_numpy().astype(int), sub["episode"].to_numpy()
    X = sub[cols].astype(float).to_numpy()
    obs = np.array([oem.auc(y, X[:, j]) for j in range(len(cols))])
    rng = np.random.default_rng(SEED)
    null = np.empty((N_PERM, len(cols)))
    for k in range(N_PERM):
        yp = _shuffle_within(y, g, rng)
        null[k] = [oem.auc(yp, X[:, j]) for j in range(len(cols))]
    p = (1 + (np.abs(null - 0.5) >= np.abs(obs - 0.5) - 1e-12).sum(0)) / (N_PERM + 1)
    df = pd.DataFrame(
        {
            "feature": cols,
            "auc": obs,
            "med_sell": sub.loc[sub["label"] == 1, cols].median().to_numpy(),
            "med_hold": sub.loc[sub["label"] == 0, cols].median().to_numpy(),
            "p_perm": p,
            "z_perm": (obs - np.nanmean(null, 0)) / np.nanstd(null, 0, ddof=1),
        }
    )
    pct = []
    for _, gg in sub.groupby("episode"):
        r = gg[cols].astype(float).rank(pct=False)
        n = gg[cols].notna().sum()
        pp = (r - 1) / (n - 1).replace(0, np.nan)
        pct.append(pp[gg["label"].to_numpy() == 1])
    df["pct_in_episode"] = pd.concat(pct).mean().to_numpy()
    df["bh"] = v4d.bh(df["p_perm"].to_numpy(), 0.10)
    return df


def _task(s: pd.DataFrame, task: str) -> pd.DataFrame:
    return {"W": s[s["win"]], "L": s[~s["win"]], "ALL": s}[task]


def _perm_model(X, y, days, g, model, seed):
    yp = _shuffle_within(y, g, np.random.default_rng(seed))
    return oem.auc(yp, oem.lodo(X, yp, days, model))


def models(s: pd.DataFrame, cols: list[str], jobs: int) -> pd.DataFrame:
    rows = []
    for task in ("W", "L", "ALL"):
        sub = _task(s, task)
        X, y = sub[cols].astype(float).to_numpy(), sub["label"].to_numpy().astype(int)
        days, g = sub["day"].to_numpy(), sub["episode"].to_numpy()
        for model in ("hgb", "l1"):
            a = oem.auc(y, oem.lodo(X, y, days, model))
            n_perm = N_PERM_TREE if model == "hgb" else N_PERM
            null = np.array(
                Parallel(n_jobs=jobs)(
                    delayed(_perm_model)(X, y, days, g, model, SEED + 1 + k) for k in range(n_perm)
                )
            )
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "n_sell": int(y.sum()),
                    "n": len(y),
                    "oos_auc": a,
                    "n_perm": n_perm,
                    "p": float((1 + (null >= a - 1e-12).sum()) / (n_perm + 1)),
                    "null_p95": float(np.nanquantile(null, 0.95)),
                }
            )
            log.info("出场模型 %s", rows[-1])
    return pd.DataFrame(rows)


# ---------------- 规则 ----------------


def fires(rule: dict, cols: dict[str, np.ndarray], start: int, stop: int) -> int | None:
    """规则在 [start, stop) 内首次成立的网格行；cols = 该持仓的状态 + 方向化市场特征（逐秒）。"""
    if rule.get("name") == "habit":
        f = cols["st_float_pct"][start:stop]
        hit = np.flatnonzero((f >= rule["tp"]) | (f <= -rule["sl"]))
        cap = int(rule["cap_min"] * 60)
        k = int(hit[0]) if len(hit) else None
        if k is None or k >= cap:
            return start + cap if start + cap < stop else None
        return start + k
    groups = rule.get("any") or [rule["conds"]]  # 任一组条件全部成立即出场（止盈组 或 止损组）
    best = None
    for conds in groups:
        ok = np.ones(stop - start, bool)
        for name, op, v in conds:
            x = cols[name][start:stop]
            with np.errstate(invalid="ignore"):
                ok &= (x >= v) if op == ">=" else (x <= v)
        h = np.flatnonzero(ok)
        if len(h) and (best is None or h[0] < best):
            best = int(h[0])
    return None if best is None else start + best


def candidates(s: pd.DataFrame, uni: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """模板：持仓状态条件 1–2 个（阈值 = 出场事件 30% / 50% 分位，方向按 AUC）+ 可选 1 个
    市场条件（单特征 z 绝对值前 k 个，阈值 = 出场事件 30% / 70% 分位）。按「出场事件覆盖 /
    持有时刻占比」排序。"""
    ev, ho = s[s["label"] == 1], s[s["label"] == 0]
    u = uni.set_index("feature")

    def opts(f, qs):
        op = ">=" if u.loc[f, "auc"] >= 0.5 else "<="
        for q in qs:
            yield (f, op, float(ev[f].quantile(q if op == ">=" else 1 - q)))

    st = [c for f in STATE if f in u.index for c in opts(f, (0.3, 0.5))]
    mk = uni[~uni["feature"].isin(STATE)]
    mk = mk.reindex(mk["z_perm"].abs().sort_values(ascending=False).index).head(k)["feature"]
    mc = [None, *(c for f in mk for c in opts(f, (0.3,)))]
    rows = []
    combos = [(a,) for a in st] + [(a, b) for a, b in itertools.combinations(st, 2) if a[0] != b[0]]
    for sc, m in itertools.product(combos, mc):
        conds = list(sc) + ([m] if m else [])

        def hit(df, conds=conds):
            out = np.ones(len(df), bool)
            for f, op, v in conds:
                x = df[f].astype(float).to_numpy()
                out &= (x >= v) if op == ">=" else (x <= v)
            return out

        ce, ch = hit(ev).mean(), hit(ho).mean()
        rows.append(
            {"conds": conds, "sell_cover": ce, "hold_share": ch, "lift": (ce + 0.01) / (ch + 0.01)}
        )
    return pd.DataFrame(rows)


def pick(c: pd.DataFrame, n: int = 4) -> list[dict]:
    ok = c[(c["sell_cover"] >= 0.3)].sort_values("lift", ascending=False)
    out, seen = [], set()
    for r in ok.itertuples():
        key = tuple(sorted(x[0] for x in r.conds))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": f"exit_rule_{len(out) + 1}",
                "conds": r.conds,
                "sell_cover": r.sell_cover,
                "hold_share": r.hold_share,
                "lift": r.lift,
            }
        )
        if len(out) >= n:
            break
    return out


def replace(root: Path, rules: list[dict], seed: int = 0) -> pd.DataFrame:
    """48 笔真实入场上用规则替换实际出场（R-041 卖出 / 吃买一），扣费后盈亏。"""
    eps = episodes(root)
    ctx = ef.Ctx.load(root)
    od = ib.OptionData(root, "MU")
    strat_rng = np.random.default_rng(seed)
    out = []
    need = {c[0] for r in rules for c in r.get("conds", []) if c[0] not in STATE}
    for day in sorted({e["day"] for e in eps}):
        d = ef.load_day(ctx, day)
        fams = tuple(sorted({ef.family_of(n) for n in need})) or ("cal",)
        fs = ef.day_features(d, fams)
        flat = int(np.searchsorted(d.t, pd.Timestamp(f"{day} 15:50", tz=ET).value))
        for e in [x for x in eps if x["day"] == day]:
            bk = od.book(
                e["symbol"],
                day,
                pd.Timestamp(e["entry_ns"], tz="UTC") - pd.Timedelta(minutes=1),
                pd.Timestamp(d.t[flat], tz="UTC") + pd.Timedelta(minutes=3),
            )
            if bk is None:
                continue
            i0 = int(np.searchsorted(d.t, e["entry_ns"], side="right"))
            cols = state_arrays(bk, d.t, i0, e["entry_px"])
            dd = np.full(len(d.t), e["dir"])
            for n in need:
                cols[n] = fs[n].directional(dd)
            base = {
                "episode": e["episode"],
                "day": day,
                "dir": e["dir"],
                "win": e["win"],
                "actual": e["pnl"],
            }
            for r in rules:
                k = fires(r, cols, i0, flat)
                k = flat if k is None else k
                tdec = int(d.t[k])
                for mode in ("owner", "taker"):
                    reason_sl = r.get("name") == "habit" and cols["st_float_pct"][k] <= -r["sl"]
                    order = (
                        {"mode": "taker", "lag_s": 2.0}
                        if (mode == "taker" or reason_sl or k == flat)
                        else {"mode": "owner"}
                    )
                    n, px, tl, _ = ib.place(
                        bk, -1, e["qty"], tdec, order, e["symbol"], 0.9, None, strat_rng
                    )
                    if n < e["qty"]:
                        c = ib.cross_now(bk, -1, tdec, 2.0)
                        px = (
                            ((px * n if n else 0.0) + c[1] * (e["qty"] - n)) / e["qty"] if c else px
                        )
                    cash = (
                        -e["entry_px"] * 100 * e["qty"]
                        - e["buy_fees"]
                        + px * 100 * e["qty"]
                        - ib.fees(e["qty"], px, "Sell")
                    )
                    out.append(
                        {
                            **base,
                            "rule": r["name"],
                            "exec": mode,
                            "exit_s": (tdec - e["entry_ns"]) / NS,
                            "sim": cash,
                        }
                    )
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    st = store(root)
    stages = ["sample", "stats", "models", "rules"] if args.stage == "all" else [args.stage]
    if "sample" in stages:
        build_sample(root)
    s = pd.read_parquet(st / "sample.parquet")
    cols = feature_cols(s)
    if "stats" in stages:
        for task in ("W", "L", "ALL"):
            univariate(s, cols, task).to_parquet(st / f"uni_{task}.parquet", index=False)
    if "models" in stages:
        models(s, cols, args.jobs).to_parquet(st / "models.parquet", index=False)
    if "rules" in stages:
        # 盈利段与亏损段分开挖：止盈组来自盈利段的卖出，止损组来自亏损段的卖出；规则 = 两组任一成立
        cw = candidates(s[s["win"]], pd.read_parquet(st / "uni_W.parquet"))
        cl = candidates(s[~s["win"]], pd.read_parquet(st / "uni_L.parquet"))
        pd.concat([cw.assign(src="W"), cl.assign(src="L")]).to_pickle(st / "candidates.pkl")
        tp, sl = pick(cw, 2), pick(cl, 2)
        rules = [
            {
                "name": f"exit_rule_{i * 2 + j + 1}",
                "any": [a["conds"], b["conds"]],
                "conds": a["conds"] + b["conds"],
                "tp": a,
                "sl": b,
            }
            for i, a in enumerate(tp)
            for j, b in enumerate(sl)
        ]
        pd.to_pickle(rules, st / "rules.pkl")
        rep = replace(root, [HABIT, *rules])
        rep.to_parquet(st / "replace.parquet", index=False)
        # R-050 预登记的选择标准：替换后扣费合计（R-041 卖出）最高的一条挖掘规则
        tot = rep[(rep["exec"] == "owner") & (rep["rule"] != "habit")].groupby("rule")["sim"].sum()
        best = next(r for r in rules if r["name"] == tot.idxmax())
        pd.to_pickle(best, st / "chosen_rule.pkl")
        log.info("R-050 出场规则：%s（合计 %.2f）", best["name"], tot.max())
        log.info("规则 %d 条（止盈候选 %d、止损候选 %d）", len(rules), len(cw), len(cl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
