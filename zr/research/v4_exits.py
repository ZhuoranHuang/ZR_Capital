"""R-043 第 2–3 节：V4 出场修正重跑与 MFE 诊断 → docs/research/EXIT_REVISION.md。

出场组（一律一次全卖；止损与强平直接吃买一，其余按成交口径卖出）：
x1_tp{10,15,20}：tick 中间价 ≥ 入场均价 ×（1 + x）；x2_n{2,3}：入场后正股 2 分钟 K（口径 C）
连续 N 根逆向收盘（多头 收盘 < 开盘，空头镜像；只数入场后收盘的 K）；x3_tp{..}_n{..}：x1 或 x2
任一，另加 −15% 止损与 30 分钟上限；x4：config/v4_exits_x4.yaml（负责人习惯提炼的
c1 + c2 + c3，另加 30 分钟上限）。
x1、x2 单独使用时只有强平兜底。成交：owner = R-041 负责人挂单策略（种子 0），taker = 吃价差。
样本：V4 H1–H3 信号（MU、QQQ，口径 C，R-033 样本与护栏、合约）；负责人 D-030 51 段真实入场
（规则先于负责人第一笔平仓成交触发则按规则全卖，否则用实际离场）。
运行：uv run python -m zr.research.v4_exits
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.backtest.fill_limit import cross_now
from zr.backtest.owner_fill import execute_owner
from zr.config import get_settings
from zr.ops.markdown import df_to_md
from zr.research import v4_decisive as v4d
from zr.research import v4_limit as vl
from zr.research import v4_owner as vo

log = logging.getLogger("zr.research.v4_exits")
REPO = Path(__file__).resolve().parents[2]
BIG = np.iinfo(np.int64).max


def exit_specs() -> dict[str, dict]:
    sp = {f"x1_tp{t}": {"tp": t / 100} for t in (10, 15, 20)}
    sp |= {f"x2_n{n}": {"n": n} for n in (2, 3)}
    for t in (10, 15, 20):
        for n in (2, 3):
            sp[f"x3_tp{t}_n{n}"] = {"tp": t / 100, "n": n, "sl": 0.15, "cap": 30}
    x4 = REPO / "config" / "v4_exits_x4.yaml"
    if x4.exists():
        c = {d["id"]: d for d in yaml.safe_load(x4.read_text(encoding="utf-8"))["x4"]}
        sp["x4"] = {
            "tp": c["c1_tp"]["tp_pct"] / 100,
            "n": int(c["c2_adverse"]["n"]),
            "sl": -c["c3_sl"]["sl_pct"] / 100,
            "cap": 30,
        }
    return sp


def adverse_trigger(close_ns: np.ndarray, adverse: np.ndarray, after_ns: int, n: int) -> int:
    i0 = int(np.searchsorted(close_ns, after_ns, side="right"))
    run = 0
    for i in range(i0, len(close_ns)):
        run = run + 1 if adverse[i] else 0
        if run >= n:
            return int(close_ns[i])
    return BIG


def first_trigger(book, te: int, px0: float, spec: dict, adv_t: int, flat: int, cap_from: int):
    i0 = int(np.searchsorted(book.q_ns, te, side="right"))
    ts, mid = book.q_ns[i0:], (book.bid[i0:] + book.ask[i0:]) / 2
    c = {"flat": int(flat)}
    if "tp" in spec:
        h = ts[mid >= px0 * (1 + spec["tp"])]
        c["tp"] = int(h[0]) if len(h) else BIG
    if "sl" in spec:
        h = ts[mid <= px0 * (1 - spec["sl"])]
        c["sl"] = int(h[0]) if len(h) else BIG
    if "n" in spec:
        c["adverse"] = adv_t
    if "cap" in spec:
        c["cap"] = cap_from + int(spec["cap"] * 60e9)
    reason = min(c, key=c.get)
    return reason, c[reason]


def sell(book, sym, n, t, reason, fill, strat, rng, flat):
    if reason in ("sl", "flat") or fill == "taker":
        cc = cross_now(book, -1, t, 2.0)
        return cc if cc is not None else (t, float(book.bid[-1]))
    e = execute_owner(book, -1, n, t, strat, sym, rng, "exit", flat)
    if e is None or e.filled_qty < n:
        cc = cross_now(book, -1, t, 2.0)
        return cc if cc is not None else (t, float(book.bid[-1]))
    return int(e.last_fill_ns), float(e.avg_px)


def v4_sims(root: Path, und: str, v4: dict, strat, specs: dict) -> pd.DataFrame:
    ev, aux = v4d.build_events(root, und, "C", v4)
    sig = ev[ev[[f"hyp_{h}" for h in v4d.HYPS]].any(axis=1)].reset_index(drop=True)
    sig["decision_ns"] = v4d.ns(sig["decision_ts"])
    sig["flat_ns"] = v4d.ns(sig["flat_ts"])
    bars = v4d.build_bars(root, und, v4["conventions"]["C"], max(aux["meta"]["days"]))
    cns = v4d.ns(bars.index + pd.Timedelta(minutes=2))
    o, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    adv = {1: c < o, -1: c > o}
    qty = v4["execution"]["qty"]
    rows, books = [], {}
    for _, r in sig.iterrows():
        key = (r["symbol"], r["day"])
        if key not in books:
            books[key] = vl.load_book(root, und, *key)
        bk = books[key]
        if bk is None:
            continue
        d, dec, flat = int(r["direction"]), int(r["decision_ns"]), int(r["flat_ns"])
        hyps = [h for h in v4d.HYPS if r[f"hyp_{h}"]]
        for fill in ("owner", "taker"):
            rng = vo.rng_for(0, dec, d, f"x|{fill}")
            if fill == "taker":
                cc = cross_now(bk, 1, dec, 2.0)
                if cc is None or cc[0] >= flat:
                    continue
                n, px0, te = qty, float(cc[1]), int(cc[0])
                miss_end = None
            else:
                e = execute_owner(bk, 1, qty, dec, strat, r["symbol"], rng, "entry", flat)
                if e is None:
                    continue
                if e.filled_qty <= 0:
                    for name in specs:
                        rows.append(
                            {
                                "day": r["day"],
                                "decision_ns": dec,
                                "direction": d,
                                "hyps": ",".join(hyps),
                                "fill": fill,
                                "exit": name,
                                "filled": 0,
                                "exit_ns": e.end_ns,
                            }
                        )
                    continue
                n, px0, te = int(e.filled_qty), float(e.avg_px), int(e.last_fill_ns)
                miss_end = None
            for name, spec in specs.items():
                adv_t = adverse_trigger(cns, adv[d], te, spec["n"]) if "n" in spec else BIG
                reason, t = first_trigger(bk, te, px0, spec, adv_t, flat, te)
                x_ns, x_px = sell(bk, r["symbol"], n, t, reason, fill, strat, rng, flat)
                cash = -px0 * 100 * n - vl._buy_fee(n, px0) + x_px * 100 * n - vl._sell_fee(n, x_px)
                rows.append(
                    {
                        "day": r["day"],
                        "decision_ns": dec,
                        "direction": d,
                        "hyps": ",".join(hyps),
                        "fill": fill,
                        "exit": name,
                        "filled": n,
                        "entry_ns": te,
                        "exit_ns": int(x_ns),
                        "reason": reason,
                        "net_usd": cash,
                        "net_pct": cash / (px0 * 100 * n) * 100,
                    }
                )
            _ = miss_end
    return pd.DataFrame(rows)


def v4_cells(sims: pd.DataFrame, v4: dict, und: str) -> pd.DataFrame:
    out = []
    for (fill, ex), g0 in sims.groupby(["fill", "exit"]):
        for h in v4d.HYPS:
            for d in (1, -1):
                g = g0[g0["hyps"].str.contains(h) & (g0["direction"] == d)]
                g = g.drop_duplicates(["decision_ns"])
                t = vl.greedy_attempts(g) if len(g) else g
                f = t[t["filled"] > 0]
                s = v4d.cell_stats(f, v4) if len(f) else {"n": 0, "mean_pct": np.nan, "p": 1.0}
                s.update(
                    {
                        "und": und,
                        "fill": fill,
                        "exit": ex,
                        "hyp": h,
                        "dir": "long" if d == 1 else "short",
                        "attempts": len(t),
                        "fill_rate": round(len(f) / len(t), 3) if len(t) else np.nan,
                    }
                )
                out.append(s)
    return pd.DataFrame(out)


def mfe_diag(root: Path, und: str | None = None) -> pd.DataFrame:
    """R-033 吃价差口径的 V4 交易（原出场）：持仓期间 tick 中间价相对入场 ask 的最大有利偏移。"""
    t = pd.read_parquet(root / "research" / "v4" / "trades_C.parquet")
    if und is not None:
        t = t[t["und"] == und]
    rows = []
    for (und, sym, day), g in t.groupby(["und", "symbol", "day"]):
        tk = v4d.load_ticks(root, und, sym, day)
        if tk is None:
            continue
        for _, r in g.iterrows():
            i0, i1 = (
                np.searchsorted(tk.ts, r["entry_ns"]),
                np.searchsorted(tk.ts, r["exit_ns"], "right"),
            )
            if i1 <= i0:
                continue
            m = tk.mid[i0:i1] / r["ask0"] - 1
            rec = {"und": und, "hyp": r["hyp"], "exit": r["exit"], "net_pct": r["net_pct"]}
            for lv in (5, 10, 15, 20):
                hit = np.flatnonzero(m >= lv / 100)
                rec[f"hit{lv}"] = len(hit) > 0
                rec[f"t{lv}_min"] = (
                    (tk.ts[i0 + hit[0]] - r["entry_ns"]) / 60e9 if len(hit) else np.nan
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def owner_sims(root: Path, strat, specs: dict) -> pd.DataFrame:
    from zr.journal.intraday_list import build as build_list
    from zr.research import fill_model as fm

    sel = build_list(root)
    sel = sel[sel["category"] == "入选"].reset_index(drop=True)
    _, ex, _ = fm.load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False))
    v4 = v4d.load_v4()
    bars = v4d.build_bars(root, "MU", v4["conventions"]["C"], max(sel["open_ts"]).date())
    cns = v4d.ns(bars.index + pd.Timedelta(minutes=2))
    o, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    adv = {1: c < o, -1: c > o}
    rows = []
    for k, r in sel.iterrows():
        sym, und = r["symbol"], fm.parse_contract(r["symbol"])["und"]
        d = 1 if r["right"] == "C" else -1
        f = ex[
            (ex["symbol"] == sym)
            & (ex["ts"] >= r["open_ts"].tz_convert("UTC"))
            & (ex["ts"] <= r["close_ts"].tz_convert("UTC"))
        ]
        b, s = f[f["side"] == "Buy"], f[f["side"] == "Sell"]
        n = float(b["quantity"].sum())
        px0 = float((b["price"] * b["quantity"]).sum() / n)
        te = int(v4d.ns(pd.DatetimeIndex([b["ts"].max()]))[0])
        tx = int(v4d.ns(pd.DatetimeIndex([s["ts"].min()]))[0])
        sq = float(s["quantity"].sum())
        spx = float((s["price"] * s["quantity"]).sum() / sq)
        act_net = spx * 100 * sq - vl._sell_fee(int(sq), spx)
        bk = vl.load_book(root, und, sym, r["open_ts"].date())
        if bk is None:
            continue
        flat = (
            v4d.et_ts(r["open_ts"].date(), v4d.day_limits(r["open_ts"].date(), v4["execution"])[1])
            - pd.Timedelta(seconds=60)
        ).value
        for name, spec in specs.items():
            if "n" in spec and und != "MU":
                continue
            adv_t = adverse_trigger(cns, adv[d], te, spec["n"]) if "n" in spec else BIG
            reason, t = first_trigger(bk, te, px0, spec, adv_t, flat, te)
            for fill in ("owner", "taker"):
                if t < tx:
                    rng = vo.rng_for(0, te, d, f"own|{name}|{fill}")
                    _, x_px = sell(bk, sym, n, t, reason, fill, strat, rng, flat)
                    diff = x_px * 100 * n - vl._sell_fee(int(n), x_px) - act_net
                else:
                    diff = 0.0
                rows.append(
                    {
                        "episode": k,
                        "und": und,
                        "day": r["open_ts"].date(),
                        "exit": name,
                        "fill": fill,
                        "triggered": t < tx,
                        "reason": reason,
                        "act_pnl": float(r["pnl_net"]),
                        "diff_usd": diff,
                        "rule_pnl": float(r["pnl_net"]) + diff,
                    }
                )
    return pd.DataFrame(rows)


def owner_cells(o: pd.DataFrame, v4: dict) -> pd.DataFrame:
    out = []
    for (ex, fill), g in o.groupby(["exit", "fill"]):
        tv, p = v4d.cluster_p(g["diff_usd"].to_numpy(), g["day"].to_numpy(), v4["stats"]["min_n"])
        out.append(
            {
                "exit": ex,
                "fill": fill,
                "n": len(g),
                "触发": int(g["triggered"].sum()),
                "实际合计": round(g["act_pnl"].sum(), 2),
                "规则合计": round(g["rule_pnl"].sum(), 2),
                "差额合计": round(g["diff_usd"].sum(), 2),
                "规则胜率": round(float((g["rule_pnl"] > 0).mean()), 3),
                "实际胜率": round(float((g["act_pnl"] > 0).mean()), 3),
                "p": p,
            }
        )
    return pd.DataFrame(out)


OUT = REPO / "docs" / "research" / "EXIT_REVISION.md"


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--unds", default="MU,QQQ", help="逗号分隔；只给部分标的时不算 FDR、不写报告")
    ap.add_argument("--skip-owner", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    strat = vo.load_strategy(vo.load_cfg())
    specs = exit_specs()
    store = root / "research" / "exit_revision"
    store.mkdir(parents=True, exist_ok=True)
    unds = args.unds.split(",")
    for und in unds:
        sims = v4_sims(root, und, v4, strat, specs)
        sims.to_parquet(store / f"sims_{und}.parquet", index=False)
        v4_cells(sims, v4, und).to_parquet(store / f"cells_{und}.parquet", index=False)
        mfe_diag(root, und).to_parquet(store / f"mfe_{und}.parquet", index=False)
        log.info("%s：%d 行", und, len(sims))
    if not args.skip_owner:
        own = owner_sims(root, strat, specs)
        own.to_parquet(store / "owner.parquet", index=False)
        owner_cells(own, v4).to_parquet(store / "owner_cells.parquet", index=False)
    all_u = v4["sample"]["underlyings"]
    if all((store / f"cells_{u}.parquet").exists() for u in all_u) and set(unds) >= set(all_u):
        cells = pd.concat([pd.read_parquet(store / f"cells_{u}.parquet") for u in all_u])
        mfe = pd.concat([pd.read_parquet(store / f"mfe_{u}.parquet") for u in all_u])
        oc = pd.read_parquet(store / "owner_cells.parquet")
        write_report(root, v4, cells, oc, mfe, specs)
    return 0


def write_report(root, v4, cells, oc, mfe, specs) -> None:
    disc = root / "research" / "discipline" / "family_183.parquet"
    prev = pd.read_parquet(disc)["p"].to_numpy(float) if disc.exists() else np.array([])
    vo_cells = cells[cells["fill"] == "owner"].reset_index(drop=True)
    oo = oc[oc["fill"] == "owner"].reset_index(drop=True)
    p = np.r_[prev, vo_cells["p"].to_numpy(float), oo["p"].to_numpy(float)]
    rej = v4d.bh(p, 0.10)
    n0 = len(prev)
    vo_cells["fdr"] = rej[n0 : n0 + len(vo_cells)]
    oo["fdr"] = rej[n0 + len(vo_cells) :]
    fam = pd.DataFrame(
        [
            {"部分": "R-041 + R-040（183 格）", "格子": n0, "通过": int(rej[:n0].sum())},
            {
                "部分": "R-043 V4 出场组（负责人挂单策略）",
                "格子": len(vo_cells),
                "通过": int(vo_cells["fdr"].sum()),
            },
            {"部分": "R-043 负责人入场 + 出场组", "格子": len(oo), "通过": int(oo["fdr"].sum())},
        ]
    )
    summ = (
        cells.groupby(["fill", "exit", "und"])
        .agg(
            格子=("n", "size"),
            笔数=("n", "sum"),
            胜率=("win_rate", "mean"),
            盈亏比=("payoff", "mean"),
            每笔均值=("mean_pct", "mean"),
            合计=("sum_usd", "sum"),
            正均值格=("mean_pct", lambda x: int((x > 0).sum())),
            最小p=("p", "min"),
        )
        .round(3)
        .reset_index()
    )
    base_o = pd.concat(
        [
            pd.read_parquet(f)
            for f in (root / "research" / "v4_owner").glob("cells_base_C_*.parquet")
        ]
    )
    base = (
        base_o.groupby(["cfg", "und"])
        .agg(
            格子=("n", "size"),
            笔数=("n", "sum"),
            胜率=("win_rate", "mean"),
            盈亏比=("payoff", "mean"),
            每笔均值=("mean_pct", "mean"),
            合计=("sum_usd", "sum"),
        )
        .round(3)
        .reset_index()
    )
    mf = (
        mfe.groupby("und")
        .agg(
            交易=("net_pct", "size"),
            **{f"达到+{lv}%": (f"hit{lv}", "mean") for lv in (5, 10, 15, 20)},
            **{f"达到+{lv}%所需分钟中位数": (f"t{lv}_min", "median") for lv in (5, 10, 15, 20)},
        )
        .round(3)
        .reset_index()
    )
    h10 = mfe[mfe["hit10"]]
    lose10 = h10.groupby("und")["net_pct"].apply(lambda x: round(float((x < 0).mean()), 3))
    lines = [
        "# V4 出场修正重跑（R-043）",
        "",
        "`uv run python -m zr.research.exit_habits` → `uv run python -m zr.research.v4_exits`。"
        "预登记 R-043。"
        "成交：owner = R-041 负责人挂单策略（种子 0），taker = 吃价差；出场一律一次全卖。",
        "",
        "## 出场组",
        "",
        df_to_md(pd.DataFrame([{"出场组": k, **v} for k, v in specs.items()])),
        "",
        "## 族（R-041 + R-040 + R-043，BH q = 0.10）",
        "",
        df_to_md(fam),
        "",
        "## V4 H1–H3（口径 C）各出场组汇总",
        "",
        df_to_md(summ),
        "",
        "对照：原出场（R-041 负责人挂单策略 owner_s0 / 吃价差 taker）：",
        "",
        df_to_md(base[base["cfg"].isin(["owner_s0", "taker"])]),
        "",
        "## 负责人 D-030 51 段真实入场 + 出场组（规则先触发则全卖，否则用实际离场）",
        "",
        df_to_md(oc),
        "",
        "## MFE 诊断（R-033 吃价差口径的 V4 交易、原出场；不进族）",
        "",
        df_to_md(mf),
        "",
        "达到 +10% 后最终亏损离场的比例："
        + "；".join(f"{u} {v:.1%}" for u, v in lose10.items())
        + "。",
        "",
        "## 边界",
        "",
        "- V4：MU、QQQ，口径 C，样本同 R-033；负责人：D-030 51 段（x2 / x3 / x4 需要正股 K，只有"
        " MU 段）。",
        "- 负责人挂单策略参数在负责人样本上提取（样本内）；q = 0.9；V4 tick 无挂单量。",
        "- x4 参数由负责人样本的描述统计算出，用在同一负责人样本上是样本内。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", OUT)


if __name__ == "__main__":
    raise SystemExit(main())
