"""R-041（负责人部分）：负责人 D-030 日内单腿在「负责人挂单策略」成交下重放 vs 对账单。

决策时刻 = 每张委托的提交时刻（盘前提交取 09:30；委托在提交时刻到达，lag 0）。开仓买单按策略入场口径
（结束未成交的张数放弃），平仓卖单按策略离场口径（结束未成交吃买一；按模拟持有张数依次卖出，最后一张
平仓委托卖完剩余）。种子 0 为主、1 / 2 看范围；吃价差（提交 +2 s 后第一个对手价）对照。
撤单机会：zr/research/calib_sample 纳入的撤单 / 过期开仓限价单（MU），策略模拟成交后按离场 (c)
（X5 + 20% / −30% + 30 分钟，强平 15:49、0DTE 15:29）管理（v4_owner.simulate_owner）。
运行：uv run python -m zr.research.owner_replay
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest.fill_limit import cross_now
from zr.backtest.owner_fill import execute_owner
from zr.config import get_settings
from zr.execution.ticks import tick_size
from zr.features.regime import indicators
from zr.research import fill_model as fm
from zr.research import owner_limit as ol
from zr.research import v4_decisive as v4d
from zr.research import v4_limit as vl
from zr.research import v4_owner as vo
from zr.research.calib_sample import classify

log = logging.getLogger("zr.research.owner_replay")
SEEDS = (0, 1, 2)


def replay(root: Path, sel: pd.DataFrame, orders: pd.DataFrame, strat) -> pd.DataFrame:
    st0 = dataclasses.replace(strat, lag_s=0.0)
    rows = []
    for k, g in orders.groupby("episode"):
        sym = sel.loc[k, "symbol"]
        bk = ol.order_book(root, g["order_id"].tolist())
        if bk is None:
            continue
        for cfg in ("taker", *[f"owner_s{s}" for s in SEEDS]):
            held = n_buy = 0.0
            cash = 0.0
            per = []
            buys, sells = g[g["side"] == "Buy"], g[g["side"] == "Sell"]
            for _, o in buys.iterrows():
                if cfg == "taker":
                    c = cross_now(bk, 1, int(o["submit_ns"]), 2.0)
                    f, px = (o["qty"], c[1]) if c else (0.0, np.nan)
                else:
                    rng = np.random.default_rng([int(cfg[-1]), int(o["submit_ns"]) % 2**31, 1])
                    e = execute_owner(bk, 1, o["qty"], int(o["submit_ns"]), st0, sym, rng, "entry")
                    f, px = (e.filled_qty, e.avg_px) if e else (0.0, np.nan)
                if f > 0:
                    held += f
                    n_buy += f
                    cash -= px * 100 * f + vl._buy_fee(int(f), px)
                per.append(("Buy", o["act_px"], px, f))
            for j, (_, o) in enumerate(sells.iterrows()):
                want = held if j == len(sells) - 1 else min(held, o["qty"])
                if want <= 0:
                    continue
                if cfg == "taker":
                    c = cross_now(bk, -1, int(o["submit_ns"]), 2.0)
                    px = c[1] if c else float(bk.bid[-1])
                else:
                    rng = np.random.default_rng([int(cfg[-1]), int(o["submit_ns"]) % 2**31, 2])
                    e = execute_owner(bk, -1, want, int(o["submit_ns"]), st0, sym, rng, "exit")
                    px = e.avg_px if e and e.filled_qty > 0 else float(bk.bid[-1])
                held -= want
                cash += px * 100 * want - vl._sell_fee(int(want), px)
                per.append(("Sell", o["act_px"], px, want))
            tick = float(tick_size(sym, float(np.nanmedian([x[1] for x in per]))))
            for side, act, simp, f in per:
                if f > 0 and np.isfinite(simp):
                    sg = 1 if side == "Buy" else -1
                    rows.append(
                        {
                            "kind": "order",
                            "episode": k,
                            "cfg": cfg,
                            "side": side,
                            "bias_ticks": sg * (simp - act) / tick,
                            "bias_pct": sg * (simp / act - 1) * 100,
                        }
                    )
            rows.append(
                {
                    "kind": "episode",
                    "episode": k,
                    "cfg": cfg,
                    "symbol": sym,
                    "sim_qty": n_buy,
                    "act_qty": float(sel.loc[k, "qty"]),
                    "sim_pnl": cash if n_buy > 0 else 0.0,
                    "act_pnl": float(sel.loc[k, "pnl_net"]),
                    "missed": n_buy == 0,
                    "open_et": sel.loc[k, "open_ts"].strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    return pd.DataFrame(rows)


def canceled(root: Path, strat) -> pd.DataFrame:
    oc = vo.load_cfg()
    v4 = v4d.load_v4()
    v4["execution"]["d001_last_0dte_entry"] = oc["sensitivity_0dte"]["d001_last_0dte_entry"]
    v4["execution"]["d001_flat_0dte"] = oc["sensitivity_0dte"]["d001_flat_0dte"]
    s = classify(root)
    s = s[s["calib"] & s["group"].isin(["canceled", "expired"]) & (s["side"] == "Buy")]
    s = s[(s["type"] == "LO") & (s["und"] == "MU")].copy()
    s["sub"] = [
        max(r["submit"], w[0] + fm.PRE) if (w := fm.window(r)) else pd.NaT for _, r in s.iterrows()
    ]
    s = s[s["sub"].notna()]
    win = pd.DataFrame(
        {
            "symbol": s["symbol"],
            "day": s["day"],
            "start": s["sub"] - pd.Timedelta(minutes=1),
            "end": s["sub"] + pd.Timedelta(minutes=36),
            "expiry": s["expiry"],
            "strike": s["strike"],
            "right": s["right"],
        }
    )
    log.info("撤单机会 tick：%s", v4d.pull_ticks(root, "MU", win))
    log.info("撤单机会 OPRA：%s", fm.pull_v4_trades(root, "MU"))
    conv = v4["conventions"]["C"]
    cfg = v4d.state_cfg(v4, "C")
    b = v4d.build_bars(root, "MU", conv, max(s["day"]))
    x = indicators(b, cfg, daily_reset=False, warmup="nan")
    struct = v4d.ns(
        b.index[v4d.exit_arrays(b, x, cfg)["X5_higher_low"][0]] + pd.Timedelta(minutes=2)
    )
    exe = v4["execution"]
    rows = []
    for _, r in s.iterrows():
        bk = vl.load_book(root, "MU", r["symbol"], r["day"])
        if bk is None:
            continue
        flat_t = v4d.day_limits(r["day"], exe)[1]
        if r["expiry"] == r["day"]:
            flat_t = min(flat_t, v4d._hm(exe["d001_flat_0dte"]))
        flat = (v4d.et_ts(r["day"], flat_t) - pd.Timedelta(seconds=exe["flat_lead_s"])).value
        dec = int(pd.Timestamp(r["sub"]).value) - int(strat.lag_s * 1e9)  # 委托在提交时刻到达
        for sd in SEEDS:
            res = vo.simulate_owner(
                bk, r["symbol"], dec, struct, flat, v4, strat, vo.rng_for(sd, dec, 1, "cancel")
            )
            if res is not None:
                rows.append({"order_id": str(r["order_id"]), "cfg": f"owner_s{sd}", **res})
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    strat = vo.load_strategy(vo.load_cfg())
    sel, orders = ol.episode_orders(root)
    rep = replay(root, sel, orders, strat)
    can = canceled(root, strat)
    store = root / "research" / "v4_owner"
    store.mkdir(parents=True, exist_ok=True)
    rep.to_parquet(store / "owner_replay.parquet", index=False)
    can.to_parquet(store / "owner_canceled.parquet", index=False)
    e = rep[rep["kind"] == "episode"]
    print(
        e.groupby("cfg")
        .agg(
            n=("episode", "size"),
            missed=("missed", "sum"),
            sim=("sim_pnl", "sum"),
            act=("act_pnl", "sum"),
        )
        .round(2)
    )
    if len(can):
        print(
            can.groupby("cfg")
            .agg(
                n=("order_id", "size"),
                filled=("filled", lambda x: (x > 0).sum()),
                net=("net_usd", "sum"),
            )
            .round(2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
