"""【R-039 已作废（负责人 2026-10-22 指令）：固定挂单价 × 固定 T 的口径不再使用。】本模块只保留
load_book / greedy_attempts / cells / 费用等公共函数供 R-041（zr/research/v4_owner.py）复用。

原说明：R-039：V4 H1–H3（与 H5 阈值）在限价成交模型下重跑（config/v4_limit.yaml）。

每个 H1–H3 信号事件（R-033 的 build_events：护栏、0/1 DTE 合约、强平时刻都不变）× 9 组挂单
（买一 + 1 tick / 中间价 / 卖一 − 1 tick × T ∈ {2, 5, 10} 秒）× 所需结构离场规则，逐笔模拟：
入场 zr/backtest/fill_limit.execute（ladder reprice_abandon，部分成交持有已成交张数，0 张 = 错过）；
持仓中 +20%（卖 1 张；只成交 1 张时不减仓）/ −30%（全卖）以入场均价为基准、按 tick 中间价判定；
结构 / 30 分钟上限 / +20% 减仓用同一挂单规则（卖出镜像，ladder reprice，两段未成交吃买一）；
止损与强平直接按触发 +2 s 后第一个买一。同格按时间贪心：上一笔离场（错过则为挂单结束）前的信号跳过。
错过的入场记录同一事件在 R-033 吃价差口径下的结果（事后表现）。

行情：diag/v4_ticks（NBBO，无挂单量 → 盘口越过时视为全量可成交）+ diag/v4_trades（OPRA 成交）。
运行：uv run python -m zr.research.v4_limit [--conv C|A|both] [--unds MU,QQQ] [--variant base|0dte]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.backtest.fill_limit import Book, LimitRule, cross_now, execute
from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.v4_limit")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "v4_limit.yaml"


def load_limit(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def configs(lc: dict) -> list[tuple[str, float]]:
    return [(r, float(t)) for r in lc["entry"]["rules"] for t in lc["entry"]["T_s"]]


def load_book(root: Path, und: str, sym: str, day) -> Book | None:
    base = root / "diag" / "v4_ticks" / f"underlying={und}" / f"date={day.isoformat()}"
    qf = sorted(base.glob(f"{sym}_*.parquet"))
    if not qf:
        return None
    q = pd.concat([pd.read_parquet(f) for f in qf], ignore_index=True).drop_duplicates()
    q = q[(q["ask"] > 0) & (q["bid"] > 0) & (q["ask"] >= q["bid"])]
    if q.empty:
        return None
    tb = root / "diag" / "v4_trades" / f"underlying={und}" / f"date={day.isoformat()}"
    tf = [tb / f.name for f in qf if (tb / f.name).exists()]
    parts = [pd.read_parquet(f) for f in tf]
    parts = [x for x in parts if len(x)]
    t = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else None
    return Book.from_frames(q, t)


def _buy_fee(n: int, px: float) -> float:
    return float(option_order_fees(n, "Buy", px)["total"])


def _sell_fee(n: int, px: float) -> float:
    # D-013（D-035 统一）；权利金 <= 0 时不卖出、不计费（保留原规则，见 D-035 歧义）
    return float(option_order_fees(n, "Sell", px)["total"]) if px > 0 else 0.0


def simulate_limit(
    book: Book, sym: str, dec: int, struct: np.ndarray, flat: int, v4: dict, lr: LimitRule
) -> dict | None:
    exe, prem = v4["execution"], v4["premium"]
    qty, lag = exe["qty"], int(lr.lag_s * 1e9)
    if dec + lag >= flat:
        return None
    ent = execute(book, 1, qty, dec, dataclasses.replace(lr, ladder="reprice_abandon"), sym, flat)
    if ent is None:
        return None
    if ent.filled_qty <= 0:
        return {"filled": 0, "attempt_end": ent.end_ns, "exit_ns": ent.end_ns}
    n, px0, te = int(ent.filled_qty), float(ent.avg_px), int(ent.last_fill_ns)
    i0 = int(np.searchsorted(book.q_ns, te, side="right"))
    ts = book.q_ns[i0:]
    mid = (book.bid[i0:] + book.ask[i0:]) / 2
    big = np.iinfo(np.int64).max
    tp = ts[mid >= px0 * (1 + prem["tp"])]
    sl = ts[mid <= px0 * (1 - prem["sl"])]
    t_tp = int(tp[0]) if len(tp) else big
    j = int(np.searchsorted(struct, dec, side="right"))
    t_st = max(int(struct[j]), te) if j < len(struct) else big
    cand = {
        "structure": t_st,
        "sl": int(sl[0]) if len(sl) else big,
        "cap": te + int(v4["cap_min"] * 60e9),
        "flat": int(flat),
    }
    reason = min(cand, key=cand.get)
    t_f = cand[reason]
    cash = -px0 * 100 * n - _buy_fee(n, px0)
    left, tp_used = n, False
    exit_lr = dataclasses.replace(lr, ladder="reprice")
    if t_tp < t_f and n >= 2:
        e = execute(book, -1, 1, t_tp, exit_lr, sym, flat)
        if e is not None and e.filled_qty > 0:
            cash += e.avg_px * 100 - _sell_fee(1, e.avg_px)
            left -= 1
            tp_used = True
    if reason in ("sl", "flat"):
        c = cross_now(book, -1, t_f, lr.lag_s)
        x_ns, x_px = c if c is not None else (t_f, float(book.bid[-1]))
    else:
        e = execute(book, -1, left, t_f, exit_lr, sym, flat)
        if e is None or e.filled_qty < left:
            c = cross_now(book, -1, t_f, lr.lag_s)
            x_ns, x_px = c if c is not None else (t_f, float(book.bid[-1]))
        else:
            x_ns, x_px = int(e.last_fill_ns), float(e.avg_px)
    cash += left * x_px * 100 - _sell_fee(left, x_px)
    return {
        "filled": n,
        "entry_px": px0,
        "entry_ns": te,
        "entry_stage": ent.stage,
        "attempt_end": ent.end_ns,
        "exit_ns": int(x_ns),
        "reason": reason,
        "tp_used": tp_used,
        "net_usd": cash,
        "net_pct": cash / (px0 * 100 * n) * 100,
    }


def event_sims(root: Path, und: str, conv: str, v4: dict, lc: dict) -> pd.DataFrame:
    """每个信号事件 × 结构离场规则：吃价差（R-033 simulate）与 9 组挂单的结果（不贪心）。"""
    ev, aux = v4d.build_events(root, und, conv, v4)
    trig = aux["trig"]
    sig = ev[ev[[f"hyp_{h}" for h in v4d.HYPS]].any(axis=1)].reset_index(drop=True)
    sig["decision_ns"] = v4d.ns(sig["decision_ts"])
    sig["flat_ns"] = v4d.ns(sig["flat_ts"])
    rules = sorted({r for v in v4["exits"].values() for r in v.values()})
    fm = lc["fill_model"]
    rows, books, ticks = [], {}, {}
    for _, r in sig.iterrows():
        key = (r["symbol"], r["day"])
        if key not in books:
            books[key] = load_book(root, und, *key)
            ticks[key] = v4d.load_ticks(root, und, *key)
        bk, tk = books[key], ticks[key]
        if bk is None or tk is None:
            continue
        hyps = [h for h in v4d.HYPS if r[f"hyp_{h}"]]
        need = {v4["exits"][x][h] for h in hyps for x in v4d.EXIT_KEYS}
        for rule in rules:
            if rule not in need:
                continue
            st = trig[rule][int(r["direction"])]
            dec, flat = int(r["decision_ns"]), int(r["flat_ns"])
            base = {
                "day": r["day"],
                "decision_ns": dec,
                "direction": int(r["direction"]),
                "symbol": r["symbol"],
                "dte": int(r["dte"]),
                "rule": rule,
                **{f"hyp_{h}": bool(r[f"hyp_{h}"]) for h in v4d.HYPS},
            }
            tak = v4d.simulate(tk, dec, st, flat, v4)
            if tak is None:
                continue
            rows.append(
                {
                    **base,
                    "cfg": "taker",
                    "filled": v4["execution"]["qty"],
                    "attempt_end": tak["entry_ns"],
                    **tak,
                }
            )
            for pr, T in configs(lc):
                lr = LimitRule(
                    price=pr, T=T, queue_fill_prob=fm["queue_fill_prob"], lag_s=fm["lag_s"], seed=0
                )
                res = simulate_limit(bk, r["symbol"], dec, st, flat, v4, lr)
                if res is not None:
                    rows.append(
                        {**base, "cfg": f"{pr}_T{int(T)}", **res, "taker_net_pct": tak["net_pct"]}
                    )
    return pd.DataFrame(rows)


def greedy_attempts(t: pd.DataFrame) -> pd.DataFrame:
    """按决策时刻贪心：上一笔（成交则到离场、错过则到挂单结束）之前的信号跳过。"""
    t = t.sort_values("decision_ns", kind="stable")
    take, last = [], -1
    for dn, xn in zip(t["decision_ns"], t["exit_ns"], strict=True):
        if dn >= last:
            take.append(True)
            last = xn
        else:
            take.append(False)
    return t[np.array(take, bool)]


def cells(sims: pd.DataFrame, v4: dict, conv: str, und: str) -> pd.DataFrame:
    out = []
    for cfg, g0 in sims.groupby("cfg"):
        for h in v4d.HYPS:
            for xk in v4d.EXIT_KEYS:
                rule = v4["exits"][xk][h]
                for d in (1, -1):
                    g = g0[g0[f"hyp_{h}"] & (g0["rule"] == rule) & (g0["direction"] == d)]
                    g = g.drop_duplicates(["decision_ns", "direction"])
                    t = greedy_attempts(g) if len(g) else g
                    f = t[t["filled"] > 0]
                    m = t[t["filled"] == 0]
                    s = (
                        v4d.cell_stats(f, v4)
                        if len(f)
                        else {"n": 0, "mean_pct": np.nan, "p": 1.0, "mean_pct_ex_top5": np.nan}
                    )
                    s.update(
                        {
                            "conv": conv,
                            "und": und,
                            "cfg": cfg,
                            "hyp": h,
                            "exit": xk,
                            "rule": rule,
                            "dir": "long" if d == 1 else "short",
                            "attempts": len(t),
                            "fill_rate": round(len(f) / len(t), 3) if len(t) else np.nan,
                            "miss_rate": round(len(m) / len(t), 3) if len(t) else np.nan,
                            "partial_rate": round(
                                float((f["filled"] < v4["execution"]["qty"]).mean()), 3
                            )
                            if len(f) and cfg != "taker"
                            else 0.0,
                            "missed_taker_mean_pct": round(float(m["taker_net_pct"].mean()), 2)
                            if len(m)
                            else np.nan,
                            "filled_taker_mean_pct": round(float(f["taker_net_pct"].mean()), 2)
                            if len(f) and "taker_net_pct" in f and f["taker_net_pct"].notna().any()
                            else np.nan,
                        }
                    )
                    out.append(s)
    return pd.DataFrame(out)


def variant_v4(v4: dict, lc: dict, variant: str) -> dict:
    v = copy.deepcopy(v4)
    if variant == "0dte":
        v["execution"]["d001_last_0dte_entry"] = lc["sensitivity_0dte"]["d001_last_0dte_entry"]
        v["execution"]["d001_flat_0dte"] = lc["sensitivity_0dte"]["d001_flat_0dte"]
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="both", choices=["C", "A", "both"])
    ap.add_argument("--unds", default="MU,QQQ")
    ap.add_argument("--variant", default="base", choices=["base", "0dte"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    lc = load_limit()
    v4 = variant_v4(v4d.load_v4(), lc, args.variant)
    store = root / "research" / "v4_limit"
    store.mkdir(parents=True, exist_ok=True)
    convs = ["C", "A"] if args.conv == "both" else [args.conv]
    for conv in convs:
        for und in args.unds.split(","):
            if args.variant == "0dte":
                ev, _ = v4d.build_events(root, und, conv, v4)
                log.info(
                    "0DTE 变体 %s %s 拉 tick：%s",
                    conv,
                    und,
                    v4d.pull_ticks(root, und, v4d.needed_windows(ev, v4)),
                )
                from zr.research.fill_model import pull_v4_trades

                log.info("0DTE 变体 OPRA：%s", pull_v4_trades(root, und))
            sims = event_sims(root, und, conv, v4, lc)
            tag = f"{args.variant}_{conv}_{und}"
            sims.to_parquet(store / f"sims_{tag}.parquet", index=False)
            c = cells(sims, v4, conv, und)
            c.to_parquet(store / f"cells_{tag}.parquet", index=False)
            log.info("%s：事件模拟 %d 行，格子 %d", tag, len(sims), len(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
