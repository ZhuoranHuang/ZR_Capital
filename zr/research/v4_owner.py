"""R-041：V4 H1–H3（与 0DTE 敏感性变体）在「负责人挂单策略」成交下重跑（config/v4_owner.yaml）。

每个 H1–H3 信号事件（R-033 build_events：护栏、0/1 DTE 合约、强平不变）× 所需结构离场规则：
吃价差（R-033 simulate）与负责人策略（种子 0 / 1 / 2）各模拟一次（不贪心），
再按格子贪心统计
（上一笔离场——错过则为挂单结束——之前的信号跳过）。策略模拟：zr/backtest/owner_fill.execute_owner
入场口径（未成交的张数放弃）；持仓中 +20%（2 张卖 1 张）/ −30% 以入场均价为基准、
按 tick 中间价判定；
结构 / 30 分钟上限 / +20% 减仓用策略卖出（结束未成交吃买一），止损与强平直接吃买一。
运行：uv run python -m zr.research.v4_owner [--conv C|A|both] [--unds MU,QQQ] [--variant base|0dte]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.backtest.fill_limit import Book, cross_now
from zr.backtest.owner_fill import OwnerStrategy, execute_owner
from zr.config import get_settings
from zr.research import v4_decisive as v4d
from zr.research import v4_limit as vl

log = logging.getLogger("zr.research.v4_owner")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "v4_owner.yaml"


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_strategy(oc: dict) -> OwnerStrategy:
    fm = oc["fill_model"]
    return OwnerStrategy.from_yaml(REPO / oc["strategy"], fm["queue_fill_prob"], fm["lag_s"])


def rng_for(seed: int, dec: int, direction: int, rule: str) -> np.random.Generator:
    return np.random.default_rng([seed, dec % (2**31), direction + 2, sum(map(ord, rule))])


def simulate_owner(
    book: Book,
    sym: str,
    dec: int,
    struct: np.ndarray,
    flat: int,
    v4: dict,
    strat: OwnerStrategy,
    rng: np.random.Generator,
    tp_full_single: bool = False,
) -> dict | None:
    """tp_full_single（R-043 第 2 节修正）：入场只成交 1 张时，+20% 即全部卖出（原实现：不减仓）。"""
    exe, prem = v4["execution"], v4["premium"]
    qty, lag = exe["qty"], int(strat.lag_s * 1e9)
    if dec + lag >= flat:
        return None
    ent = execute_owner(book, 1, qty, dec, strat, sym, rng, "entry", flat)
    if ent is None:
        return None
    if ent.filled_qty <= 0:
        return {
            "filled": 0,
            "attempt_end": ent.end_ns,
            "exit_ns": ent.end_ns,
            "end_reason": ent.end_reason,
        }
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
    if tp_full_single and n == 1:
        cand["tp"] = t_tp
    reason = min(cand, key=cand.get)
    t_f = cand[reason]
    cash = -px0 * 100 * n - vl._buy_fee(n, px0)
    left, tp_used = n, False
    if t_tp < t_f and n >= 2:
        e = execute_owner(book, -1, 1, t_tp, strat, sym, rng, "exit", flat)
        if e is not None and e.filled_qty > 0:
            cash += e.avg_px * 100 - vl._sell_fee(1, e.avg_px)
            left -= 1
            tp_used = True
    if reason in ("sl", "flat"):
        c = cross_now(book, -1, t_f, strat.lag_s)
        x_ns, x_px = c if c is not None else (t_f, float(book.bid[-1]))
    else:
        e = execute_owner(book, -1, left, t_f, strat, sym, rng, "exit", flat)
        if e is None or e.filled_qty < left:
            c = cross_now(book, -1, t_f, strat.lag_s)
            x_ns, x_px = c if c is not None else (t_f, float(book.bid[-1]))
        else:
            x_ns, x_px = int(e.last_fill_ns), float(e.avg_px)
    cash += left * x_px * 100 - vl._sell_fee(left, x_px)
    return {
        "filled": n,
        "entry_px": px0,
        "entry_ns": te,
        "attempt_end": ent.end_ns,
        "exit_ns": int(x_ns),
        "reason": reason,
        "tp_used": tp_used,
        "end_reason": ent.end_reason,
        "net_usd": cash,
        "net_pct": cash / (px0 * 100 * n) * 100,
    }


def event_sims(
    root: Path,
    und: str,
    conv: str,
    v4: dict,
    oc: dict,
    strat: OwnerStrategy,
    tp_full_single: bool = False,
):
    ev, aux = v4d.build_events(root, und, conv, v4)
    trig = aux["trig"]
    sig = ev[ev[[f"hyp_{h}" for h in v4d.HYPS]].any(axis=1)].reset_index(drop=True)
    sig["decision_ns"] = v4d.ns(sig["decision_ts"])
    sig["flat_ns"] = v4d.ns(sig["flat_ts"])
    rules = sorted({r for v in v4["exits"].values() for r in v.values()})
    seeds = [oc["seeds"]["primary"], *oc["seeds"]["robustness"]]
    rows, books, ticks = [], {}, {}
    for _, r in sig.iterrows():
        key = (r["symbol"], r["day"])
        if key not in books:
            books[key] = vl.load_book(root, und, *key)
            ticks[key] = v4d.load_ticks(root, und, *key)
        bk, tk = books[key], ticks[key]
        if bk is None or tk is None:
            continue
        hyps = [h for h in v4d.HYPS if r[f"hyp_{h}"]]
        need = {v4["exits"][x][h] for h in hyps for x in v4d.EXIT_KEYS}
        dec, flat, d = int(r["decision_ns"]), int(r["flat_ns"]), int(r["direction"])
        for rule in rules:
            if rule not in need:
                continue
            st = trig[rule][d]
            base = {
                "day": r["day"],
                "decision_ns": dec,
                "direction": d,
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
            for sd in seeds:
                res = simulate_owner(
                    bk,
                    r["symbol"],
                    dec,
                    st,
                    flat,
                    v4,
                    strat,
                    rng_for(sd, dec, d, rule),
                    tp_full_single,
                )
                if res is not None:
                    rows.append(
                        {**base, "cfg": f"owner_s{sd}", **res, "taker_net_pct": tak["net_pct"]}
                    )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="both", choices=["C", "A", "both"])
    ap.add_argument("--unds", default="MU,QQQ")
    ap.add_argument("--variant", default="base", choices=["base", "0dte"])
    ap.add_argument("--tp-full-single", action="store_true", help="R-043 第 2 节：1 张时 +20% 全卖")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    oc = load_cfg()
    strat = load_strategy(oc)
    v4 = v4d.load_v4()
    if args.variant == "0dte":
        v4["execution"]["d001_last_0dte_entry"] = oc["sensitivity_0dte"]["d001_last_0dte_entry"]
        v4["execution"]["d001_flat_0dte"] = oc["sensitivity_0dte"]["d001_flat_0dte"]
    store = root / "research" / "v4_owner"
    store.mkdir(parents=True, exist_ok=True)
    for conv in ["C", "A"] if args.conv == "both" else [args.conv]:
        for und in args.unds.split(","):
            if args.variant == "0dte":
                ev, _ = v4d.build_events(root, und, conv, v4)
                ev = ev[ev[[f"hyp_{h}" for h in v4d.HYPS]].any(axis=1)]
                log.info(
                    "0DTE 变体 %s %s tick：%s",
                    conv,
                    und,
                    v4d.pull_ticks(root, und, v4d.needed_windows(ev, v4)),
                )
                from zr.research.fill_model import pull_v4_trades

                log.info("0DTE 变体 OPRA：%s", pull_v4_trades(root, und))
            sims = event_sims(root, und, conv, v4, oc, strat, args.tp_full_single)
            tag = f"{args.variant}{'_fix1' if args.tp_full_single else ''}_{conv}_{und}"
            sims.to_parquet(store / f"sims_{tag}.parquet", index=False)
            c = vl.cells(sims, v4, conv, und)
            c.to_parquet(store / f"cells_{tag}.parquet", index=False)
            log.info("%s：事件模拟 %d 行，格子 %d", tag, len(sims), len(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
