"""R-039（负责人部分，描述性）：负责人 D-030 日内单腿在限价成交模型下的模拟 vs 对账单。

每个入选持仓段的每张委托（开仓买入、平仓卖出）在其实际提交时刻（盘前提交取 09:30）按三种挂单 × T
重新执行（zr/backtest/fill_limit.execute，+2 s 到达）：
开仓 ladder reprice_abandon（没成交的张数放弃），
平仓 ladder reprice（两段未成交吃买一；按开仓实际模拟成交张数依次卖出，最后一张平仓委托卖完剩余）。
盈亏扣 zr.costs 费用，与对账单 pnl_net 比较；逐张委托的模拟均价 vs 实际成交均价（tick、%）。
吃价差对照：提交 +2 s 后第一个 ask（买）/ bid（卖）。
10:15 后开仓、V4 规则无合约的 0DTE 持仓段（BASE_FINAL_AUDIT 第 6 节的 14 段）单列。
撤单 / 过期的开仓限价单（MU）：模拟成交则按 V4 离场 (c)（X5 跌破更高低点 + 20% 减半 / −30% 全出 /
30 分钟上限，强平 15:49、0DTE 15:29）管理（tick 由 v4_decisive.pull_ticks 补拉）。
运行：uv run python -m zr.research.owner_limit
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest.fill_limit import Book, LimitRule, cross_now, execute
from zr.config import get_settings
from zr.execution.ticks import tick_size
from zr.journal.analyze_history import load
from zr.journal.intraday_list import build as build_list
from zr.research import fill_model as fm
from zr.research import v4_decisive as v4d
from zr.research import v4_limit as vl

log = logging.getLogger("zr.research.owner_limit")


def _ns(x) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(x, utc=True)).as_unit("ns").asi8


def order_book(root: Path, oids: list[str]) -> Book | None:
    qs, ts = [], []
    for oid in oids:
        d = fm._load_ticks(root, oid)
        if d is None:
            continue
        qs.append(d[0])
        if len(d[1]):
            ts.append(d[1])
    if not qs:
        return None
    q = pd.concat(qs, ignore_index=True).drop_duplicates(["ts", "bid", "ask"])
    t = pd.concat(ts, ignore_index=True).drop_duplicates() if ts else None
    return Book.from_frames(q, t)


def episode_orders(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(入选持仓段, 委托表：order_id、episode、side、submit_ns、filled_qty、实际均价)。"""
    ep = build_list(root)
    sel = ep[ep["category"] == "入选"].reset_index(drop=True)
    o, ex, _ = load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False), order_id=ex["order_id"].astype(str))
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
    rows = []
    for k, r in sel.iterrows():
        m = (
            (ex["symbol"] == r["symbol"])
            & (ex["ts"] >= r["open_ts"].tz_convert("UTC"))
            & (ex["ts"] <= r["close_ts"].tz_convert("UTC"))
        )
        for oid, g in ex[m].groupby("order_id"):
            so = s[s["order_id"] == oid]
            if so.empty:
                continue
            so = so.iloc[0]
            w = fm.window(so)
            sub = so["submit"] if w is None else max(so["submit"], w[0] + fm.PRE)
            rows.append(
                {
                    "episode": k,
                    "order_id": oid,
                    "symbol": r["symbol"],
                    "side": so["side"],
                    "type": so["type"],
                    "submit_ns": int(_ns([sub])[0]),
                    "fill_ns": int(_ns([g["ts"].min()])[0]),
                    "limit": float(so["limit"]) if pd.notna(so["limit"]) else np.nan,
                    "qty": float(g["quantity"].sum()),
                    "act_px": float((g["price"] * g["quantity"]).sum() / g["quantity"].sum()),
                }
            )
    return sel, pd.DataFrame(rows).sort_values(["episode", "submit_ns"]).reset_index(drop=True)


def _fees(n: float, px: float, side: str) -> float:
    return vl._buy_fee(int(n), px) if side == "Buy" else vl._sell_fee(int(n), px)


def replay(
    root: Path, sel: pd.DataFrame, orders: pd.DataFrame, lc: dict, ref: str = "submit"
) -> pd.DataFrame:
    """ref = submit（预登记：委托提交时刻）或 fill（该委托实际首笔成交时刻，未预登记的补充口径：
    负责人常把限价挂在远离市价处等价格过来，提交时刻不是执行决策时刻）。"""
    fmc = lc["fill_model"]
    tcol = "submit_ns" if ref == "submit" else "fill_ns"
    cfgs = [("taker", 0.0), *vl.configs(lc)]
    rows = []
    for k, g in orders.groupby("episode"):
        sym = sel.loc[k, "symbol"]
        bk = order_book(root, g["order_id"].tolist())
        if bk is None:
            continue
        for pr, T in cfgs:
            lr = LimitRule(
                price=pr if pr != "taker" else "mid",
                T=T,
                queue_fill_prob=fmc["queue_fill_prob"],
                lag_s=fmc["lag_s"],
            )
            held, cash, n_buy = 0.0, 0.0, 0.0
            per = []
            buys, sells = g[g["side"] == "Buy"], g[g["side"] == "Sell"]
            for _, o in buys.iterrows():
                if pr == "taker":
                    c = cross_now(bk, 1, int(o[tcol]), lr.lag_s)
                    f, px = (o["qty"], c[1]) if c else (0.0, np.nan)
                else:
                    e = execute(
                        bk,
                        1,
                        o["qty"],
                        int(o[tcol]),
                        dataclasses.replace(lr, ladder="reprice_abandon"),
                        sym,
                    )
                    f, px = (e.filled_qty, e.avg_px) if e else (0.0, np.nan)
                if f > 0:
                    held += f
                    n_buy += f
                    cash -= px * 100 * f + _fees(f, px, "Buy")
                per.append((o["order_id"], "Buy", o["act_px"], px, f, o["qty"]))
            for j, (_, o) in enumerate(sells.iterrows()):
                want = held if j == len(sells) - 1 else min(held, o["qty"])
                if want <= 0:
                    per.append((o["order_id"], "Sell", o["act_px"], np.nan, 0.0, o["qty"]))
                    continue
                if pr == "taker":
                    c = cross_now(bk, -1, int(o[tcol]), lr.lag_s)
                    px = c[1] if c else float(bk.bid[-1])
                else:
                    e = execute(
                        bk,
                        -1,
                        want,
                        int(o[tcol]),
                        dataclasses.replace(lr, ladder="reprice"),
                        sym,
                    )
                    px = e.avg_px if e and e.filled_qty > 0 else float(bk.bid[-1])
                held -= want
                cash += px * 100 * want - _fees(want, px, "Sell")
                per.append((o["order_id"], "Sell", o["act_px"], px, want, o["qty"]))
            tick = float(tick_size(sym, float(np.nanmedian([x[2] for x in per]))))
            for oid, side, act, simp, f, q in per:
                if f > 0 and np.isfinite(simp):
                    sgn = 1 if side == "Buy" else -1
                    rows.append(
                        {
                            "kind": "order",
                            "ref": ref,
                            "episode": k,
                            "cfg": pr if pr == "taker" else f"{pr}_T{int(T)}",
                            "order_id": oid,
                            "side": side,
                            "act_px": act,
                            "sim_px": simp,
                            "fill_ratio": f / q,
                            "bias_ticks": sgn * (simp - act) / tick,
                            "bias_pct": sgn * (simp / act - 1) * 100,
                        }
                    )
            rows.append(
                {
                    "kind": "episode",
                    "ref": ref,
                    "episode": k,
                    "cfg": pr if pr == "taker" else f"{pr}_T{int(T)}",
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


# ---------------- 撤单 / 过期的开仓限价单 ----------------


def canceled_opps(root: Path, lc: dict) -> pd.DataFrame:
    v4 = vl.variant_v4(v4d.load_v4(), lc, "0dte")  # 强平 15:49、0DTE 15:29
    s = fm.order_sample(root)
    s = s[s["group"].isin(["canceled", "expired"]) & (s["side"] == "Buy") & (s["type"] == "LO")]
    s = s[s["und"] == "MU"].copy()
    s["sub"] = [
        max(r["submit"], (w[0] + fm.PRE)) if (w := fm.window(r)) else pd.NaT
        for _, r in s.iterrows()
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
            "right": s["right"].map({"call": "call", "put": "put", "C": "call", "P": "put"}),
        }
    )
    log.info("撤单机会补拉 tick：%s", v4d.pull_ticks(root, "MU", win))
    log.info("撤单机会补拉 OPRA：%s", fm.pull_v4_trades(root, "MU"))
    conv = v4["conventions"]["C"]
    b = v4d.build_bars(root, "MU", conv, max(s["day"]))
    from zr.features.regime import indicators

    x = indicators(b, v4d.state_cfg(v4, "C"), daily_reset=False, warmup="nan")
    lg = v4d.exit_arrays(b, x, v4d.state_cfg(v4, "C"))["X5_higher_low"][0]
    struct = v4d.ns(b.index[lg] + pd.Timedelta(minutes=2))
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
        dec = int(_ns([r["sub"]])[0]) - int(lc["fill_model"]["lag_s"] * 1e9)  # 委托提交即到达
        for pr, T in vl.configs(lc):
            lr = LimitRule(price=pr, T=T, queue_fill_prob=lc["fill_model"]["queue_fill_prob"])
            res = vl.simulate_limit(bk, r["symbol"], dec, struct, flat, v4, lr)
            if res is None:
                continue
            rows.append(
                {
                    "order_id": str(r["order_id"]),
                    "cfg": f"{pr}_T{int(T)}",
                    "owner_limit": r["limit"],
                    **res,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    lc = vl.load_limit()
    sel, orders = episode_orders(root)
    rep = pd.concat(
        [replay(root, sel, orders, lc, "submit"), replay(root, sel, orders, lc, "fill")],
        ignore_index=True,
    )
    orders.to_parquet(root / "research" / "v4_limit" / "owner_orders.parquet", index=False)
    can = canceled_opps(root, lc)
    store = root / "research" / "v4_limit"
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
    print(
        can.groupby("cfg")
        .agg(
            n=("order_id", "size"),
            filled=("filled", lambda x: (x > 0).sum()),
            net=("net_usd", "sum"),
        )
        .round(2)
        if len(can)
        else "无"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
