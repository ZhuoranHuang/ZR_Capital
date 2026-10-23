"""R-040 纪律类规则（config/v4_discipline.yaml）→ docs/research/DISCIPLINE_RULES.md。

B1 负责人 D-030 入选样本按真实入场叠加紧止损；B2 V4 H1、H2 × 原离场 × 5 种紧止损；B3 共振规则。
成交（补登）：R-041 负责人挂单策略（种子 0；入场按策略入场口径、(a)(c) 离场按策略离场口径），
止损类吃买一。族：与 R-041 合为 183 格 BH q = 0.10（读取 research/v4_owner/family_*.parquet）。
运行：uv run python -m zr.research.discipline
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.backtest.fill_limit import cross_now
from zr.backtest.owner_fill import execute_owner
from zr.config import get_settings
from zr.features.regime import RANGE
from zr.journal.intraday_list import build as build_list
from zr.ops.markdown import df_to_md
from zr.research import fill_model as fm
from zr.research import v4_decisive as v4d
from zr.research import v4_limit as vl
from zr.research import v4_owner as vo
from zr.research.sim_v3 import sell_fee

log = logging.getLogger("zr.research.discipline")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "v4_discipline.yaml"
OUT = REPO / "docs" / "research" / "DISCIPLINE_RULES.md"
VARIANTS = ("a", "b10", "b15", "c5", "c10")


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def strategy(dc: dict):
    return vo.load_strategy(vo.load_cfg())


def adverse_arrays(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """逆向 K 收盘时刻（ns）：多头 收盘 < 开盘，空头 收盘 > 开盘。"""
    cns = v4d.ns(df.index + pd.Timedelta(minutes=2))
    c, o = df["close"].to_numpy(float), df["open"].to_numpy(float)
    return {1: cns[c < o], -1: cns[c > o]}


def variant(v4: dict, name: str) -> dict:
    v = copy.deepcopy(v4)
    if name.startswith("b"):
        v["premium"]["sl"] = int(name[1:]) / 100
    if name.startswith("c"):
        v["cap_min"] = int(name[1:])
    return v


# ---------------- B2 / B3：V4 事件 ----------------


def run_events(root, und, conv, v4, lc, dc, hyps, sig_fn, exit_plan) -> pd.DataFrame:
    """exit_plan(h, d, trig, adv) → [(变体名, v4 变体, 结构离场触发数组)]。"""
    ev, aux = v4d.build_events(root, und, conv, v4, sig_fn=sig_fn, hyps=hyps)
    ev = ev[ev[[f"hyp_{h}" for h in hyps]].any(axis=1)]  # 不做随机对照：只拉信号事件的 tick
    log.info("%s %s 拉 tick：%s", und, conv, v4d.pull_ticks(root, und, v4d.needed_windows(ev, v4)))
    log.info("%s OPRA：%s", und, fm.pull_v4_trades(root, und))
    days = aux["meta"]["days"]
    df = v4d.build_bars(root, und, v4["conventions"][conv], max(days))
    adv = adverse_arrays(df)
    trig = aux["trig"]
    sig = ev[ev[[f"hyp_{h}" for h in hyps]].any(axis=1)].reset_index(drop=True)
    sig["decision_ns"] = v4d.ns(sig["decision_ts"])
    sig["flat_ns"] = v4d.ns(sig["flat_ts"])
    strat = strategy(dc)
    seed = dc["fill"]["seed"]
    rows, books = [], {}
    for _, r in sig.iterrows():
        key = (r["symbol"], r["day"])
        if key not in books:
            books[key] = vl.load_book(root, und, *key)
        bk = books[key]
        if bk is None:
            continue
        d = int(r["direction"])
        for h in hyps:
            if not r[f"hyp_{h}"]:
                continue
            for name, v, st in exit_plan(h, d, trig, adv):
                res = vo.simulate_owner(
                    bk,
                    r["symbol"],
                    int(r["decision_ns"]),
                    st,
                    int(r["flat_ns"]),
                    v,
                    strat,
                    vo.rng_for(seed, int(r["decision_ns"]), d, f"{h}|{name}"),
                )
                if res is not None:
                    rows.append(
                        {
                            "day": r["day"],
                            "decision_ns": int(r["decision_ns"]),
                            "direction": d,
                            "hyp": h,
                            "variant": name,
                            "dte": int(r["dte"]),
                            **res,
                        }
                    )
    return pd.DataFrame(rows)


def cells_from(sims: pd.DataFrame, v4: dict, keys: list[str], extra: dict) -> pd.DataFrame:
    out = []
    for k, g in sims.groupby(keys + ["direction"]):
        g = g.drop_duplicates(["decision_ns"])
        t = vl.greedy_attempts(g)
        f = t[t["filled"] > 0]
        s = v4d.cell_stats(f, v4) if len(f) else {"n": 0, "mean_pct": np.nan, "p": 1.0}
        s.update(dict(zip(keys + ["direction"], k if isinstance(k, tuple) else (k,), strict=True)))
        s.update(
            {
                "attempts": len(t),
                "fill_rate": round(len(f) / len(t), 3) if len(t) else np.nan,
                **extra,
            }
        )
        out.append(s)
    return pd.DataFrame(out)


def b2(root: Path, v4: dict, lc: dict, dc: dict, conv: str = "C") -> pd.DataFrame:
    def plan(h, d, trig, adv):
        out = []
        for xk in v4d.EXIT_KEYS:
            base = trig[v4["exits"][xk][h]][d]
            for name in VARIANTS:
                st = np.unique(np.r_[base, adv[d]]) if name == "a" else base
                out.append((f"{xk}|{name}", variant(v4, name), st))
        return out

    parts = []
    for und in v4["sample"]["underlyings"]:
        sims = run_events(root, und, conv, v4, lc, dc, ("H1", "H2"), None, plan)
        sims[["exit", "tight"]] = sims["variant"].str.split("|", expand=True)
        parts.append(cells_from(sims, v4, ["hyp", "exit", "tight"], {"und": und, "conv": conv}))
    return pd.concat(parts, ignore_index=True)


def conf_signal(dc: dict):
    vr_v, vw = dc["confluence"]["vol_ratio"]["v"], dc["confluence"]["vol_ratio"]["window"]

    def fn(df, x, reg, cfg, conv):
        h = x["hist"].to_numpy()
        hp = np.r_[np.nan, h[:-1]]
        c, mid = df["close"].to_numpy(float), x["mid"].to_numpy()
        vol = df["volume"].to_numpy(float)
        vr = vol / pd.Series(vol).shift(1).rolling(vw, min_periods=vw).mean().to_numpy()
        nr = reg["state"].to_numpy() != RANGE
        with np.errstate(invalid="ignore", divide="ignore"):
            lg = (h > 0) & (h > hp) & (c > mid) & (vr >= vr_v) & nr
            sh = (h < 0) & (h < hp) & (c < mid) & (vr >= vr_v) & nr
        return {"CONF": (lg, sh)}

    return fn


def b3(root: Path, v4: dict, lc: dict, dc: dict, conv: str) -> pd.DataFrame:
    v = copy.deepcopy(v4)
    v["premium"]["tp"], v["premium"]["sl"], v["cap_min"] = 1e9, 1.0, 10**6  # 只用 (a) + 强平

    def plan(h, d, trig, adv):
        return [("a_only", v, adv[d])]

    parts = []
    for und in v4["sample"]["underlyings"]:
        sims = run_events(root, und, conv, v4, lc, dc, ("CONF",), conf_signal(dc), plan)
        parts.append(cells_from(sims, v4, ["hyp"], {"und": und, "conv": conv}))
    return pd.concat(parts, ignore_index=True)


# ---------------- B1：负责人 ----------------


def b1(root: Path, v4: dict, lc: dict, dc: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ep = build_list(root)
    sel = ep[ep["category"] == "入选"].reset_index(drop=True)
    _, ex, _ = fm.load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False))
    win = []
    for _, r in sel.iterrows():
        c = fm.parse_contract(r["symbol"])
        win.append(
            {
                "und": c["und"],
                "symbol": r["symbol"],
                "day": r["open_ts"].date(),
                "start": r["open_ts"].tz_convert("UTC") - pd.Timedelta(minutes=1),
                "end": r["close_ts"].tz_convert("UTC") + pd.Timedelta(minutes=3),
                "expiry": c["expiry"],
                "strike": c["strike"],
                "right": c["right"],
            }
        )
    w = pd.DataFrame(win)
    for und, g in w.groupby("und"):
        log.info("B1 %s tick：%s", und, v4d.pull_ticks(root, und, g.drop(columns="und")))
        log.info("B1 %s OPRA：%s", und, fm.pull_v4_trades(root, und))
    conv = v4["conventions"]["C"]
    bars = v4d.build_bars(root, "MU", conv, max(w["day"]))
    adv = adverse_arrays(bars)
    strat = strategy(dc)
    rows = []
    for k, r in sel.iterrows():
        und = w.loc[k, "und"]
        bk = vl.load_book(root, und, r["symbol"], w.loc[k, "day"])
        if bk is None:
            continue
        sym = r["symbol"]
        fills = ex[ex["symbol"] == sym]
        fills = fills[
            (fills["ts"] >= r["open_ts"].tz_convert("UTC"))
            & (fills["ts"] <= r["close_ts"].tz_convert("UTC"))
        ]
        buys, sells = fills[fills["side"] == "Buy"], fills[fills["side"] == "Sell"]
        te = int(v4d.ns(pd.DatetimeIndex([buys["ts"].min()]))[0])
        tx = int(v4d.ns(pd.DatetimeIndex([sells["ts"].min()]))[0])
        n = float(buys["quantity"].sum())
        px0 = float((buys["price"] * buys["quantity"]).sum() / n)
        sq = float(sells["quantity"].sum())
        spx = float((sells["price"] * sells["quantity"]).sum() / sq)
        act_sell_net = spx * 100 * sq - float(sell_fee(int(sq), spx))
        i0 = int(np.searchsorted(bk.q_ns, te, side="right"))
        mids, qts = (bk.bid[i0:] + bk.ask[i0:]) / 2, bk.q_ns[i0:]
        d = 1  # 负责人全部是买入期权；看跌期权的「逆向 K」= 正股收盘 > 开盘
        dirn = 1 if r["right"] == "C" else -1
        for name in VARIANTS:
            if name == "a":
                if und != "MU":
                    continue
                a = adv[dirn]
                j = int(np.searchsorted(a, te, side="right"))
                t_r = int(a[j]) if j < len(a) else None
            elif name.startswith("b"):
                hit = qts[mids <= px0 * (1 - int(name[1:]) / 100)]
                t_r = int(hit[0]) if len(hit) else None
            else:
                t_r = te + int(int(name[1:]) * 60e9)
            used = t_r is not None and t_r < tx
            if used:
                if name.startswith("b"):
                    c = cross_now(bk, -d, t_r, strat.lag_s)
                    px = c[1] if c else float(bk.bid[-1])
                else:
                    rng = vo.rng_for(dc["fill"]["seed"], t_r, -1, f"owner|{name}")
                    e = execute_owner(bk, -d, n, t_r, strat, sym, rng, "exit")
                    px = e.avg_px if e and e.filled_qty > 0 else float(bk.bid[-1])
                sim_sell_net = px * 100 * n - float(sell_fee(int(n), px))
                diff = sim_sell_net - act_sell_net
            else:
                diff = 0.0
            rows.append(
                {
                    "episode": k,
                    "symbol": sym,
                    "und": und,
                    "variant": name,
                    "day": r["open_ts"].date(),
                    "triggered": used,
                    "act_pnl": float(r["pnl_net"]),
                    "diff_usd": diff,
                    "rule_pnl": float(r["pnl_net"]) + diff,
                }
            )
    t = pd.DataFrame(rows)
    cells = []
    for name, g in t.groupby("variant"):
        tv, p = v4d.cluster_p(g["diff_usd"].to_numpy(), g["day"].to_numpy(), v4["stats"]["min_n"])
        cells.append(
            {
                "variant": name,
                "n": len(g),
                "触发": int(g["triggered"].sum()),
                "实际合计": round(g["act_pnl"].sum(), 2),
                "规则合计": round(g["rule_pnl"].sum(), 2),
                "差额合计": round(g["diff_usd"].sum(), 2),
                "t": round(tv, 2) if np.isfinite(tv) else np.nan,
                "p": p,
            }
        )
    return t, pd.DataFrame(cells)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4, lc, dc = v4d.load_v4(), vl.load_limit(), load_cfg()
    store = root / "research" / "discipline"
    store.mkdir(parents=True, exist_ok=True)
    t1, c1 = b1(root, v4, lc, dc)
    t1.to_parquet(store / "b1_episodes.parquet", index=False)
    c1.to_parquet(store / "b1_cells.parquet", index=False)
    c2 = b2(root, v4, lc, dc, "C")
    c2.to_parquet(store / "b2_cells_C.parquet", index=False)
    c3 = b3(root, v4, lc, dc, "C")
    c3.to_parquet(store / "b3_cells_C.parquet", index=False)
    c3a = b3(root, v4, lc, dc, "A")
    c3a.to_parquet(store / "b3_cells_A.parquet", index=False)
    write_report(root, v4, lc, c1, t1, c2, c3, c3a)
    return 0


def write_report(root, v4, lc, c1, t1, c2, c3, c3a) -> None:
    lim = root / "research" / "v4_owner"
    fa = pd.read_parquet(lim / "family_v4_C.parquet")
    fh = pd.read_parquet(lim / "family_h5.parquet")
    p = np.r_[
        fa["p"].to_numpy(float),
        fh["p"].to_numpy(float),
        c1["p"].to_numpy(float),
        c2["p"].to_numpy(float),
        c3["p"].to_numpy(float),
    ]
    rej = v4d.bh(p, lc["stats"]["fdr_q"])
    n_a, n_h, n1, n2 = len(fa), len(fh), len(c1), len(c2)
    c1 = c1.assign(fdr_183=rej[n_a + n_h : n_a + n_h + n1])
    c2 = c2.assign(fdr_183=rej[n_a + n_h + n1 : n_a + n_h + n1 + n2])
    c3 = c3.assign(fdr_183=rej[n_a + n_h + n1 + n2 :])
    fam = pd.DataFrame(
        [
            {
                "部分": "R-041 V4（口径 C，负责人挂单策略）",
                "格子": n_a,
                "通过": int(rej[:n_a].sum()),
            },
            {"部分": "R-041 H5", "格子": n_h, "通过": int(rej[n_a : n_a + n_h].sum())},
            {"部分": "R-040 B1 负责人紧止损", "格子": n1, "通过": int(c1["fdr_183"].sum())},
            {"部分": "R-040 B2 V4 紧止损", "格子": n2, "通过": int(c2["fdr_183"].sum())},
            {"部分": "R-040 B3 共振规则", "格子": len(c3), "通过": int(c3["fdr_183"].sum())},
        ]
    )
    pd.DataFrame({"p": p, "reject": rej}).to_parquet(
        root / "research" / "discipline" / "family_615.parquet"
    )
    c2v = (
        c2.groupby(["und", "hyp", "tight"])
        .agg(
            格子=("n", "size"),
            笔数=("n", "sum"),
            均值=("mean_pct", "mean"),
            最好=("mean_pct", "max"),
            最小p=("p", "min"),
            通过=("fdr_183", "sum"),
        )
        .round(3)
        .reset_index()
    )
    base = fa[fa["hyp"].isin(["H1", "H2"])]
    bsum = (
        base.groupby(["und", "hyp"])
        .agg(格子=("n", "size"), 均值=("mean_pct", "mean"), 最好=("mean_pct", "max"))
        .round(3)
        .reset_index()
    )
    cols3 = [
        "und",
        "conv",
        "direction",
        "n",
        "attempts",
        "fill_rate",
        "win_rate",
        "mean_pct",
        "median_pct",
        "sum_usd",
        "mean_pct_ex_top5",
        "p",
    ]
    lines = [
        "# 纪律类规则（R-040）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} · 复现 `uv run python -m "
        "zr.research.discipline`。预登记 RESEARCH_LOG R-040、`config/v4_discipline.yaml`。成交："
        "R-041 负责人挂单策略（种子 0），止损类吃买一。",
        "",
        "## 族（R-041 + R-040，BH q = 0.10）",
        "",
        df_to_md(fam),
        "",
        "## B1 负责人 D-030 样本叠加紧止损（真实入场；规则先触发则按规则全部离场，否则用实际离场）",
        "",
        df_to_md(c1),
        "",
        "差额 = 规则离场的卖出净额 − 实际卖出净额（每段）；p 为单侧（H0：差额均值 ≤ 0），按日聚类。"
        "(a) 只有 MU（需要正股 K）。",
        "",
        "## B2 V4 H1、H2 加紧止损（口径 C）",
        "",
        "对照：同一成交口径（负责人挂单策略，种子 0）、不加紧止损（R-041）：",
        "",
        df_to_md(bsum),
        "",
        df_to_md(c2v),
        "",
        "## B3 共振规则（离场只用 (a) + 强平）",
        "",
        df_to_md(pd.concat([c3, c3a], ignore_index=True)[[c for c in cols3 if c in c3]]),
        "",
        "## 边界",
        "",
        "- V4：MU、QQQ，日期同 R-033；负责人：2026-06-09～2026-08-19 D-030 入选 51 段（(a) 只 M"
        "U 48 段）。",
        "- 成交模型同 R-041（负责人挂单策略；q = 0.9；V4 tick 无挂单量；OPRA 含负责人自己的成交"
        "）。",
        "- B1 用负责人的真实入场，只回答「叠加这些纪律后负责人会更好还是更差」，不是可独立交易的"
        "规则。",
        "- 逆向 K 定义为收盘 < 开盘（多头），不是「收盘低于前一根收盘」。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
