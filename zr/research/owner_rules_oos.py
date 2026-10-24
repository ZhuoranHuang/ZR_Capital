"""R-049：负责人规则 / 调研规则 / 融合规则的样本外检验 → docs/research/OWNER_RULES_OOS.md。

规则：config/intrabar/owner_rule_*.yaml、survey_*.yaml 中 oos_pick: true 的、fused_rule_*.yaml；
全部双向（看涨 d = +1 / 看跌 d = −1 分别评估同一组方向化条件）。条件逐秒评估（intrabar 引擎的
entry_conditions，特征来自 zr/features/entry_features.py 与引擎原生特征）。
样本外区间：MU 2025-01-02～2026-05-31、2026-09-01～2026-10-21（有秒级平价价的交易日；D-033 校准
窗口 2026-06～08 由 intrabar.check_out_of_sample 拒绝）。

第一层（正股）：触发 = 入场时段（规则 session）内条件由假变真的那一秒，同一规则同一方向 5 分钟内
不重复计（预登记的约定）。每个触发：方向化正股收益（平价反推价，bp）在 1 / 2 / 5 / 10 / 15 / 30
分钟；30 分钟内最大顺向 / 逆向偏移（MFE / MAE）；同日、触发前后 30 分钟内（入场时段内、剔除前后
2 分钟）随机 50 秒、同方向的同口径收益，触发在其中的百分位与「触发 − 随机均值」超额；超额按交易
日聚类的单侧 t（H0：均值 ≤ 0）。多空分开与合并报告。不计费用。
第二层（期权，盘中引擎）：合约 = 最近到期日、虚值一档（看涨取现价之上第一个行权价，看跌取之下
第一个）；0DTE 按 D-001（10:15 后不开 0DTE、改用次一到期日；0DTE 仓位 10:45 前清仓）与「不限」
敏感性两列；执行 (i) 负责人挂单策略（R-041）入场 + 负责人离场习惯（中间价 +10% 全卖 / −15% 全卖 /
30 分钟上限；止盈与时间上限用负责人挂单策略卖出，止损吃买一）；(ii) 吃价差对照（入场离场都
+2 秒后对手价）。同一时间只持一个仓位，15:50 全部清仓。费用 D-013。
多重比较：主格 = 每条规则 × 6 个持有期的正股合并格 + 每条规则 × 2 种执行的期权格（D-001），
BH q = 0.10。
运行：uv run python -m zr.research.owner_rules_oos [--stage stock|option|report|all]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.data.calendar import trading_days
from zr.features import entry_features as ef
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.owner_rules_oos")
REPO = Path(__file__).resolve().parents[2]
RULE_DIR = REPO / "config" / "intrabar"
OUT = REPO / "docs" / "research" / "OWNER_RULES_OOS.md"
NS = 1_000_000_000
HORIZONS = (60, 120, 300, 600, 900, 1800)
REFRACTORY = 300
N_RAND = 50
RAND_WIN = 1800
RAND_EXCL = 120
RANGES = ((dt.date(2025, 1, 2), dt.date(2026, 5, 31)), (dt.date(2026, 9, 1), dt.date(2026, 10, 21)))
EXECUTIONS = ("owner", "taker")
ZERO_DTE = ("d001", "free")
Q = 0.10


def store(root: Path) -> Path:
    p = root / "research" / "owner_rules_oos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_rules() -> dict[str, dict]:
    out = {}
    for pat in ("owner_rule_*.yaml", "survey_*.yaml", "fused_rule_*.yaml"):
        for f in sorted(RULE_DIR.glob(pat)):
            r = yaml.safe_load(f.read_text(encoding="utf-8"))
            if pat.startswith("survey") and not r.get("oos_pick"):
                continue
            out[f.stem] = normalize(r)
    return out


def normalize(r: dict) -> dict:
    """R-049 预登记的统一口径：合约最近到期、虚值一档；离场 +10% / −15% / 30 分钟（中间价）；
    执行 owner（R-041）与 taker 对照。0DTE 口径由 intrabar.with_execution 设定。"""
    r = dict(r)
    r["contract"] = {"nearest_expiry": True, "otm_steps": 1}
    r["exit"] = {"tp": 0.10, "sl": 0.15, "max_hold_min": 30, "basis": "mid"}
    r["executions"] = {
        "owner": {"entry": {"mode": "owner"}, "exit": {"mode": "owner"}},
        "taker": {
            "entry": {"mode": "taker", "lag_s": 2.0},
            "exit": {"mode": "taker", "lag_s": 2.0},
        },
    }
    r["qty"] = 1
    r["direction"] = "both"
    return r


def oos_days(root: Path) -> list[dt.date]:
    base = root / "intrabar" / "underlying=MU"
    days = []
    for a, b in RANGES:
        days += [d for d in trading_days(a, b) if (base / f"date={d}" / "S.parquet").exists()]
    ib.check_out_of_sample(days)
    return days


# ---------------- 第一层 ----------------


def _edges(x: np.ndarray, refractory: int) -> np.ndarray:
    on = np.flatnonzero(x & ~np.r_[False, x[:-1]])
    out, last = [], -(10**9)
    for k in on:
        if k - last >= refractory:
            out.append(k)
            last = k
    return np.array(out, dtype=int)


def _fwd(S: np.ndarray) -> dict:
    n = len(S)
    out = {}
    for h in HORIZONS:
        fut = np.r_[S[h:], np.full(min(h, n), np.nan)][:n]
        out[h] = fut
    rs = pd.Series(S[::-1])
    w = HORIZONS[-1] + 1
    out["max"] = rs.rolling(w, min_periods=1).max().to_numpy()[::-1]
    out["min"] = rs.rolling(w, min_periods=1).min().to_numpy()[::-1]
    return out


def _metrics(S, fw, k: np.ndarray, d: int) -> dict:
    s0 = S[k]
    m = {f"ret{h}": d * (fw[h][k] / s0 - 1) * 1e4 for h in HORIZONS}
    up, dn = (fw["max"][k] / s0 - 1) * 1e4, (fw["min"][k] / s0 - 1) * 1e4
    m["mfe"] = up if d == 1 else -dn
    m["mae"] = dn if d == 1 else -up
    return m


def stock_day(root: Path, ctx, bs, rules: dict, day: dt.date, seed: int) -> tuple:
    g = ib.second_grid(root, "MU", day)
    dd = ef.load_day(ctx, day)
    if g is None or dd is None:
        return [], {}
    assert len(g) == len(dd.t)
    fs = ef.day_features(dd)
    S = g["S"].to_numpy()
    fw = _fwd(S)
    rng = np.random.default_rng([seed, day.toordinal()])
    rows, conds = [], {}
    for name, rule in rules.items():
        specs = rule["entry"]["all"]
        feat = ib.features(g, bs, specs)
        feat["ef"] = {c["name"]: fs[c["name"]] for c in specs if c["feature"] == "ef"}
        cond, in_win = ib.entry_conditions(rule, g, feat, bs)
        conds[name] = cond
        win_idx = np.flatnonzero(in_win)
        trig = {d: _edges(c & in_win & np.isfinite(S), REFRACTORY) for d, c in cond.items()}
        cap = rule.get("max_entries_per_day")
        if cap:  # 每天只取最早的 cap 个触发（不分方向）
            allk = sorted((k, d) for d, ks in trig.items() for k in ks)[: int(cap)]
            trig = {d: np.array([k for k, dd in allk if dd == d], dtype=int) for d in trig}
        for d, ks in trig.items():
            if not len(ks):
                continue
            mt = _metrics(S, fw, ks, d)
            for j, k in enumerate(ks):
                pool = win_idx[
                    (np.abs(win_idx - k) <= RAND_WIN) & (np.abs(win_idx - k) > RAND_EXCL)
                ]
                pool = pool[np.isfinite(S[pool])]
                rk = rng.choice(pool, size=min(N_RAND, len(pool)), replace=False)
                mr = _metrics(S, fw, rk, d) if len(rk) else None
                row = {"rule": name, "day": day, "t_ns": int(g["t"].iloc[k]), "dir": d}
                for key, v in mt.items():
                    row[key] = float(v[j])
                    if mr is not None and key.startswith("ret"):
                        rv = mr[key][np.isfinite(mr[key])]
                        row[f"rand_{key}"] = float(rv.mean()) if len(rv) else math.nan
                        x = row[key]
                        row[f"pct_{key}"] = (
                            float(((rv < x).sum() + 0.5 * (rv == x).sum()) / len(rv))
                            if len(rv) and np.isfinite(x)
                            else math.nan
                        )
                rows.append(row)
    return rows, conds


def _shard_stock(root: Path, rules: dict, days: list, seed: int) -> list:
    ctx = ef.Ctx.load(root)
    bs = ib.BarState(root, "MU", max(days))
    cdir = store(root) / "conds"
    cdir.mkdir(exist_ok=True)
    rows = []
    for day in days:
        r, conds = stock_day(root, ctx, bs, rules, day, seed)
        rows += r
        if conds:
            np.savez_compressed(
                cdir / f"{day}.npz",
                **{f"{n}|{d}": np.packbits(v) for n, c in conds.items() for d, v in c.items()},
            )
        log.info("%s：触发 %d", day, len(r))
    return rows


def run_stock(root: Path, rules: dict, days: list, jobs: int, seed: int = 0) -> pd.DataFrame:
    shards = [days[k::jobs] for k in range(jobs)]
    parts = Parallel(n_jobs=jobs)(delayed(_shard_stock)(root, rules, sh, seed) for sh in shards)
    df = pd.DataFrame([x for p in parts for x in p])
    df.assign(day=df["day"].astype(str)).to_parquet(store(root) / "stock.parquet", index=False)
    return df


def load_conds(root: Path, day: dt.date, name: str, n: int) -> dict | None:
    p = store(root) / "conds" / f"{day}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    out = {}
    for d in (1, -1):
        k = f"{name}|{d}"
        if k in z:
            out[d] = np.unpackbits(z[k])[:n].astype(bool)
    return out or None


def stock_stats(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    rows = []
    for (rule, grp), sub in _groups(df):
        for h in HORIZONS:
            x = sub[f"ret{h}"].to_numpy(float)
            rr = sub[f"rand_ret{h}"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(rr)
            ex = x[ok] - rr[ok]
            t, p = v4d.cluster_p(ex, sub["day"].to_numpy()[ok], 10) if ok.sum() else (np.nan, 1.0)
            rows.append(
                {
                    "rule": rule,
                    "dir": grp,
                    "h_min": h // 60,
                    "n": int(ok.sum()),
                    "days": int(sub["day"].nunique()),
                    "per_day": len(sub) / n_days,
                    "ret_mean_bp": float(np.nanmean(x)),
                    "rand_mean_bp": float(np.nanmean(rr)),
                    "excess_bp": float(ex.mean()) if len(ex) else math.nan,
                    "pct_mean": float(sub[f"pct_ret{h}"].mean()),
                    "hit": float((x[ok] > 0).mean()) if ok.any() else math.nan,
                    "mfe_med": float(sub["mfe"].median()),
                    "mae_med": float(sub["mae"].median()),
                    "t_day": t,
                    "p": p,
                }
            )
    return pd.DataFrame(rows)


def _groups(df: pd.DataFrame):
    for rule, sub in df.groupby("rule"):
        yield (rule, "合并"), sub
        yield (rule, "多"), sub[sub["dir"] == 1]
        yield (rule, "空"), sub[sub["dir"] == -1]


# ---------------- 第二层 ----------------


def prefetch(root: Path, days: list, concurrency: int = 4) -> pd.DataFrame:
    """第二层要用到的合约的全天逐笔 NBBO 与 OPRA 成交预先拉进引擎缓存（diag/v4_ticks、v4_trades）：
    当日秒级价区间内任一时刻的虚值一档看涨 / 看跌，最近到期日；最近到期日是当日（0DTE）时另加次一
    到期日（D-001 10:15 后改用）。"""
    od = ib.OptionData(root, "MU", concurrency)
    exps = od.expirations()
    rows = []
    for day in days:
        g = ib.second_grid(root, "MU", day)
        if g is None or not g["S"].notna().any():
            continue
        lo, hi = float(g["S"].min()), float(g["S"].max())
        fut = [e for e in exps if e >= day][:2]
        use = fut[:2] if fut and fut[0] == day else fut[:1]
        for e in use:
            ks = od.strikes(e)
            calls = ks[(ks > lo) & (ks <= (ks[ks > hi][0] if (ks > hi).any() else ks[-1]))]
            puts = ks[(ks < hi) & (ks >= (ks[ks < lo][-1] if (ks < lo).any() else ks[0]))]
            for right, kk in (("call", calls), ("put", puts)):
                for k in kk:
                    rows.append(
                        {
                            "symbol": ib.lb_symbol("MU", e, right, float(k)),
                            "day": day,
                            "start": pd.Timestamp(f"{day} 09:29", tz=ib.ET).tz_convert("UTC"),
                            "end": pd.Timestamp(f"{day} 16:01", tz=ib.ET).tz_convert("UTC"),
                            "expiry": e,
                            "strike": float(k),
                            "right": right,
                        }
                    )
    w = pd.DataFrame(rows)
    log.info("预取 %d 个合约日", len(w))
    for k in range(0, len(w), 200):
        part = w.iloc[k : k + 200]
        st = v4d.pull_ticks(root, "MU", part, concurrency)
        for r in part.itertuples(index=False):
            od._trades(r.symbol, r.day)
        log.info("预取 %d / %d：%s", min(k + 200, len(w)), len(w), st)
    return w


def _shard_option(root: Path, rules: dict, days: list, seed: int) -> list:
    bs = ib.BarState(root, "MU", max(days))
    od = ib.OptionData(root, "MU")
    out = []
    for day in days:
        g = ib.second_grid(root, "MU", day)
        if g is None:
            continue
        for name, rule in rules.items():
            co = load_conds(root, day, name, len(g))
            if co is None or not any(v.any() for v in co.values()):
                continue
            for ex in EXECUTIONS:
                for z in ZERO_DTE:
                    r = ib.with_execution(rule, ex, z)
                    for tr in ib.run_day(root, r, day, bs, od, seed, None, co):
                        out.append({"rule": name, "exec": ex, "zdte": z, **tr})
        log.info("%s 期权层完成（累计 %d 笔）", day, len(out))
    return out


def run_option(root: Path, rules: dict, days: list, jobs: int, seed: int = 0) -> pd.DataFrame:
    ib.check_out_of_sample(days)
    shards = [days[k::jobs] for k in range(jobs)]
    parts = Parallel(n_jobs=jobs)(delayed(_shard_option)(root, rules, sh, seed) for sh in shards)
    df = pd.DataFrame([x for p in parts for x in p])
    df.assign(day=df["day"].astype(str)).to_parquet(store(root) / "option.parquet", index=False)
    return df


def _max_dd(t: pd.DataFrame) -> float:
    if t.empty:
        return 0.0
    c = t.sort_values("entry_ns")["net_usd"].cumsum().to_numpy()
    return float((np.maximum.accumulate(np.r_[0.0, c]) - np.r_[0.0, c]).max())


def option_cell(t: pd.DataFrame) -> dict:
    att = len(t)
    f = t[t["filled"] > 0]
    usd = f["net_usd"].to_numpy(float)
    win, loss = usd[usd > 0], usd[usd <= 0]
    tval, p = (
        v4d.cluster_p(f["net_pct"].to_numpy(float), f["day"].to_numpy(), 10)
        if len(f)
        else (
            np.nan,
            1.0,
        )
    )
    return {
        "attempts": att,
        "trades": len(f),
        "fill_rate": len(f) / att if att else math.nan,
        "win_rate": float((usd > 0).mean()) if len(usd) else math.nan,
        "payoff": float(win.mean() / -loss.mean())
        if len(win) and len(loss) and loss.mean() < 0
        else math.nan,
        "net_pct_mean": float(f["net_pct"].mean()) if len(f) else math.nan,
        "total_usd": float(usd.sum()),
        "max_dd_usd": _max_dd(f),
        "t_day": tval,
        "p": p,
    }


def option_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["year"] = pd.to_datetime(df["day"]).dt.year
    for (rule, ex, z), sub in df.groupby(["rule", "exec", "zdte"]):
        for label, part in (
            ("全部", sub),
            ("多", sub[sub["dir"] == 1]),
            ("空", sub[sub["dir"] == -1]),
            *((str(y), sub[sub["year"] == y]) for y in sorted(sub["year"].unique())),
        ):
            rows.append({"rule": rule, "exec": ex, "zdte": z, "slice": label, **option_cell(part)})
    return pd.DataFrame(rows)


def fdr(stock: pd.DataFrame, opt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sm = (stock["dir"] == "合并").to_numpy()
    om = ((opt["slice"] == "全部") & (opt["zdte"] == "d001")).to_numpy()
    p = np.r_[stock.loc[sm, "p"].to_numpy(float), opt.loc[om, "p"].to_numpy(float)]
    rej = v4d.bh(np.nan_to_num(p, nan=1.0), Q)
    stock, opt = stock.copy(), opt.copy()
    stock["bh"], opt["bh"] = None, None
    stock.loc[sm, "bh"] = rej[: sm.sum()]
    opt.loc[om, "bh"] = rej[sm.sum() :]
    return stock, opt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--option-jobs", type=int, default=3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    rules = load_rules()
    days = oos_days(root)
    log.info("规则 %d 条：%s；样本外交易日 %d", len(rules), list(rules), len(days))
    st = store(root)
    stages = ["stock", "prefetch", "option", "report"] if args.stage == "all" else [args.stage]
    if "stock" in stages:
        run_stock(root, rules, days, args.jobs)
    if "prefetch" in stages:
        prefetch(root, days)
    if "option" in stages:
        run_option(root, rules, days, args.option_jobs)
    if "report" in stages:
        from zr.research import owner_rules_oos_report

        stock = pd.read_parquet(st / "stock.parquet")
        opt = pd.read_parquet(st / "option.parquet")
        owner_rules_oos_report.write(root, rules, days, stock, opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
