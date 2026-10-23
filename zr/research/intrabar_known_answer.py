"""盘中引擎（zr/backtest/intrabar.py）的已知答案检验 → docs/research/INTRABAR_KNOWN_ANSWER.md。

(1) 负责人 D-030 入选持仓段中的 MU 段：每张委托按真实提交时刻（盘前提交取 09:30）、真实挂单价路径
（改价历史的 New / Replaced，到终结时刻 + 1 秒止；市价单与触价单在改价历史首个 New——
触价单即触发生效时刻——吃对手价）在 tick 上重放（到达不加延迟；恰好等于挂单价的
OPRA 成交按确定成交，因为负责人自己的成交就在 OPRA 数据里），平仓卖单同样按真实卖单的价格路径；
逐张委托比较成交张数与成交均价（tick），逐段比较扣费后盈亏（D-013 费用）与对账单。
通过标准：成交张数全部一致，成交均价差 ≤ 1 tick 的委托 ≥ 95%，段盈亏差 ≤ Σ（张数 × 1 tick × 100）
的段 ≥ 95%。
(2) 同一批段：买单在提交时刻 + 2 秒吃卖一，卖单在卖单提交时刻 + 2 秒吃买一（数量同实际），
与此前 R-041 负责人重放的吃价差口径（research/v4_owner/owner_replay.parquet 的 taker）逐段比较。
通过标准：逐段差 ≤ 0.01 美元的段 ≥ 95%。
运行：uv run python -m zr.research.intrabar_known_answer
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.execution.ticks import tick_size
from zr.ops.markdown import df_to_md
from zr.research import fill_model as fm
from zr.research import fill_segments as fs
from zr.research import owner_limit as ol

log = logging.getLogger("zr.research.intrabar_known_answer")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "INTRABAR_KNOWN_ANSWER.md"
NS = 1_000_000_000


def add_force(root: Path, exact: list[dict]) -> list[dict]:
    """R-047：平仓卖单加 force_ns = 该卖单最后一笔真实成交的时刻（限价路径单才需要）。"""
    import copy

    from zr.journal.analyze_history import load

    _, ex, _ = load(root)
    ex = ex.assign(ts=fm._utc(ex["ts_et"], naive_is_utc=False), order_id=ex["order_id"].astype(str))
    last = ex.groupby("order_id")["ts"].max()
    out = copy.deepcopy(exact)
    for lg in out:
        if lg["side"] == -1 and lg["order"]["mode"] == "path" and lg["order_id"] in last.index:
            lg["order"]["force_ns"] = int(fs._ns(pd.Series([last[lg["order_id"]]]))[0])
    return out


def legs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    sel, orders = ol.episode_orders(root)
    orders = orders[orders["symbol"].str.startswith("MU")].copy()
    s = fm.order_sample(root)
    s["order_id"] = s["order_id"].astype(str)
    h = fs.load_history(root)
    exact, taker = [], []
    for _, o in orders.iterrows():
        so = s[s["order_id"] == o["order_id"]].iloc[0]
        day = pd.Timestamp(o["submit_ns"], tz="UTC").tz_convert(fm.ET).date()
        side = 1 if o["side"] == "Buy" else -1
        hh = h[h["order_id"] == o["order_id"]]
        bnd = hh[hh["status"].isin(fs.BOUNDARY)]
        term = hh[hh["status"].isin(fs.TERMINAL)]
        open_ns = int(o["submit_ns"])  # 已截到 09:30（盘前提交）
        base = {
            "episode": int(o["episode"]),
            "order_id": o["order_id"],
            "symbol": o["symbol"],
            "day": day,
            "side": side,
            "qty": float(o["qty"]),
            "act_px": float(o["act_px"]),
            "type": o["type"],
        }
        if so["type"] == "LO" and len(bnd):
            ts = fs._ns(bnd["ts"])
            path = [(max(int(t), open_ns), float(p)) for t, p in zip(ts, bnd["price"], strict=True)]
            end = int(fs._ns(term["ts"])[-1]) + NS if len(term) else path[-1][0] + 600 * NS
            exact.append(
                {
                    **base,
                    "t_ns": path[0][0],
                    "order": {"mode": "path", "path": path, "end_ns": end, "lag_s": 0.0},
                }
            )
        else:  # 市价 / 触价单：改价历史的首个 New（触价单的触发、生效时刻）吃对手价（不加延迟）
            t_live = max(int(fs._ns(bnd["ts"])[0]), open_ns) if len(bnd) else open_ns
            exact.append({**base, "t_ns": t_live, "order": {"mode": "taker", "lag_s": 0.0}})
        taker.append({**base, "t_ns": open_ns, "order": {"mode": "taker", "lag_s": 2.0}})
    return sel, orders, exact, taker


def pnl(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep, g in df.groupby("episode"):
        b, s = g[g["side"] == 1], g[g["side"] == -1]
        nb = float(b["filled"].sum())
        cash = 0.0
        for _, r in b.iterrows():
            if r["filled"] > 0:
                cash -= r["px"] * 100 * r["filled"] + ib.fees(r["filled"], r["px"], "Buy")
        held = nb
        for j, (_, r) in enumerate(s.iterrows()):
            n = min(held, r["filled"]) if j < len(s) - 1 else held
            if n > 0 and np.isfinite(r["px"]):
                cash += r["px"] * 100 * n - ib.fees(n, r["px"], "Sell")
                held -= n
        rows.append({"episode": ep, "sim_pnl": cash, "bought": nb})
    return pd.DataFrame(rows)


def acceptance(root: Path, sel, exact: list[dict], pi) -> tuple[pd.DataFrame, dict]:
    """R-046：(a)(b)(c) 用种子 0 的一次重放，(d) 种子 0–99。"""
    cache: dict = {}
    act_total = float(sum(float(sel.loc[k, "pnl_net"]) for k in {lg["episode"] for lg in exact}))
    totals = []
    for seed in range(100):
        d = pd.DataFrame(ib.replay(root, "MU", exact, q=1.0, pi=pi, seed=seed, cache=cache))
        totals.append(float(pnl(d)["sim_pnl"].sum()))
        if seed == 0:
            d0 = d
    d0["tick"] = [float(tick_size(s, p)) for s, p in zip(d0["symbol"], d0["act_px"], strict=True)]
    qok = np.isclose(d0["filled"], d0["qty"])
    qok_nf = np.isclose(d0["filled"] - d0.get("forced", 0.0), d0["qty"])  # 不计补卖
    both = d0["filled"] > 0
    diff = (d0["side"] * (d0["px"] - d0["act_px"]) / d0["tick"])[both]  # 正 = 模拟对负责人更差
    t0 = totals[0]
    lo, hi = np.quantile(totals, [0.05, 0.95])
    res = {
        "a_qty_rate": float(qok.mean()),
        "a_qty_rate_no_forced": float(qok_nf.mean()),
        "n_forced_orders": int((d0.get("forced", 0.0) > 0).sum()) if "forced" in d0 else 0,
        "b_median_ticks": float(diff.median()),
        "c_total_sim": t0,
        "c_total_act": act_total,
        "c_rel": abs(t0 - act_total) / abs(act_total),
        "d_p05": float(lo),
        "d_p95": float(hi),
        "d_pct_of_actual": float((np.array(totals) < act_total).mean()),
    }
    res["a"] = res["a_qty_rate"] >= 0.95
    res["b"] = abs(res["b_median_ticks"]) <= 1.0
    res["c"] = res["c_rel"] <= 0.10
    res["d"] = lo <= act_total <= hi
    tab = pd.DataFrame(
        [
            {
                "项": "(a) 委托成交张数一致率 ≥ 95%",
                "结果": f"{res['a_qty_rate']:.1%}",
                "是否通过": res["a"],
            },
            {
                "项": "(b) 成交价差（按方向，tick）中位数在 ±1 内",
                "结果": f"{res['b_median_ticks']:+.2f}",
                "是否通过": res["b"],
            },
            {
                "项": "(c) 48 段合计与对账单相差 ≤ 10%",
                "结果": f"模拟 {t0:+,.2f} / 对账单 {act_total:+,.2f}（差 {res['c_rel']:.1%}）",
                "是否通过": res["c"],
            },
            {
                "项": "(d) 对账单合计落在 100 次模拟的 5%–95% 分位",
                "结果": f"P5 {lo:+,.2f} / P95 {hi:+,.2f}（对账单在第 {res['d_pct_of_actual']:.0%}"
                " 分位）",
                "是否通过": res["d"],
            },
        ]
    )
    # 事后描述（不改变判定）：差额来源
    pe = pnl(d0)
    pe["act"] = [float(sel.loc[k, "pnl_net"]) for k in pe["episode"]]
    pe["diff"] = pe["sim_pnl"] - pe["act"]
    qbad = set(d0.loc[~qok, "episode"])
    usd = -d0["side"] * (d0["px"] - d0["act_px"]) * 100 * d0["filled"]
    res["gap_total"] = float(pe["diff"].sum())
    res["gap_qty_eps"] = float(pe.loc[pe["episode"].isin(qbad), "diff"].sum())
    res["n_qty_eps"] = len(qbad)
    res["gap_px_usd"] = float(usd.fillna(0).sum())
    res["worst_ep_diff"] = float(pe["diff"].min())
    return tab, {**res, "totals": totals, "orders": d0}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    sel, orders, exact, taker = legs(root)
    ex = pd.DataFrame(ib.replay(root, "MU", exact, q=1.0))
    tk = pd.DataFrame(ib.replay(root, "MU", taker, q=1.0))
    # 将 taker 的成交张数设为实际张数（吃价差一定成交）；平仓同样按实际张数依次卖出
    for d in (ex, tk):
        d["tick"] = [float(tick_size(s, p)) for s, p in zip(d["symbol"], d["act_px"], strict=True)]
    ex["qty_ok"] = np.isclose(ex["filled"], ex["qty"])
    ex["px_diff_ticks"] = (ex["px"] - ex["act_px"]) / ex["tick"]
    ex["px_ok"] = ex["px_diff_ticks"].abs() <= 1 + 1e-9
    pe = pnl(ex).merge(
        pnl(tk).rename(columns={"sim_pnl": "taker_pnl", "bought": "tb"}), on="episode"
    )
    pe["act_pnl"] = [float(sel.loc[k, "pnl_net"]) for k in pe["episode"]]
    tol = ex.groupby("episode").apply(lambda g: float((g["qty"] * g["tick"] * 100).sum()))
    pe["tol"] = pe["episode"].map(tol)
    pe["pnl_diff"] = pe["sim_pnl"] - pe["act_pnl"]
    pe["pnl_ok"] = pe["pnl_diff"].abs() <= pe["tol"] + 1e-6
    prev = pd.read_parquet(root / "research" / "v4_owner" / "owner_replay.parquet")
    prev = prev[(prev["kind"] == "episode") & (prev["cfg"] == "taker")][["episode", "sim_pnl"]]
    pe = pe.merge(prev.rename(columns={"sim_pnl": "prev_taker"}), on="episode", how="left")
    pe["taker_diff"] = pe["taker_pnl"] - pe["prev_taker"]
    pe["taker_ok"] = pe["taker_diff"].abs() <= 0.01
    q_ok, p_ok = float(ex["qty_ok"].mean()), float(ex.loc[ex["qty_ok"], "px_ok"].mean())
    e_ok, t_ok = float(pe["pnl_ok"].mean()), float(pe["taker_ok"].mean())
    pass1 = q_ok == 1.0 and p_ok >= 0.95 and e_ok >= 0.95
    pass2 = t_ok >= 0.95
    prev_all = pd.read_parquet(root / "research" / "v4_owner" / "owner_replay.parquet")
    prev_all = prev_all[(prev_all["kind"] == "episode") & (prev_all["cfg"] == "taker")][
        "sim_pnl"
    ].sum()
    tk_sum, prev_sum = pe["taker_pnl"].sum(), pe["prev_taker"].sum()
    summ = pd.DataFrame(
        [
            {"检验": "(1) 委托成交张数一致", "结果": f"{int(ex['qty_ok'].sum())}/{len(ex)}"},
            {
                "检验": "(1) 成交均价差 ≤ 1 tick（张数一致的委托）",
                "结果": f"{int(ex.loc[ex['qty_ok'], 'px_ok'].sum())}/{int(ex['qty_ok'].sum())}",
            },
            {"检验": "(1) 段盈亏差 ≤ 容差", "结果": f"{int(pe['pnl_ok'].sum())}/{len(pe)}"},
            {
                "检验": "(1) 段盈亏合计（模拟 / 对账单）",
                "结果": f"{pe['sim_pnl'].sum():+,.2f} / {pe['act_pnl'].sum():+,.2f}",
            },
            {"检验": "(1) 是否通过", "结果": "通过" if pass1 else "不通过"},
            {
                "检验": "(2) 逐段与此前吃价差重放差 ≤ 0.01 美元",
                "结果": f"{int(pe['taker_ok'].sum())}/{len(pe)}",
            },
            {
                "检验": "(2) 吃价差合计（本引擎 MU 段 / 此前 MU 段 / 此前 51 段）",
                "结果": f"{tk_sum:+,.2f} / {prev_sum:+,.2f} / {prev_all:+,.2f}",
            },
            {"检验": "(2) 是否通过", "结果": "通过" if pass2 else "不通过"},
        ]
    )
    bad = ex[~(ex["qty_ok"] & ex["px_ok"])]
    n_buy, n_sell = int((ex["side"] == 1).sum()), int((ex["side"] == -1).sum())
    lines = [
        "# 盘中引擎已知答案检验",
        "",
        "`uv run python -m zr.research.intrabar_known_answer`；引擎 `zr/backtest/intrabar.py`。"
        f"样本：D-030 入选持仓段中的 MU 段 {len(pe)} 段、委托 {len(ex)} 张（买 {n_buy}"
        "、"
        f"卖 {n_sell}）。",
        "",
        df_to_md(summ),
        "",
        "## 不一致的委托",
        "",
        df_to_md(
            bad[
                [
                    "episode",
                    "order_id",
                    "symbol",
                    "type",
                    "side",
                    "qty",
                    "filled",
                    "act_px",
                    "px",
                    "px_diff_ticks",
                ]
            ].round(3)
        )
        if len(bad)
        else "无",
        "",
        "## 逐段",
        "",
        df_to_md(
            pe[
                [
                    "episode",
                    "act_pnl",
                    "sim_pnl",
                    "pnl_diff",
                    "tol",
                    "pnl_ok",
                    "taker_pnl",
                    "prev_taker",
                    "taker_diff",
                ]
            ].round(2)
        ),
        "",
        "## 边界",
        "",
        "- 行情：ThetaData 逐笔 NBBO + OPRA 成交（`diag/v4_ticks`、`diag/v4_trades`，引擎按需补"
        "拉）；"
        "盘前提交的委托从 09:30 起挂。改价历史为整秒，终结时刻加 1 秒。",
        "- 费用按 D-013 计算（对账单费用此前已核对一致，见 KNOWN_ANSWER_AUDIT.md）。",
        "",
    ]
    from zr.backtest.fill_limit import PriceImprovement

    pi = PriceImprovement.from_yaml(REPO / "config" / "price_improvement.yaml")
    acc, ar = acceptance(root, sel, exact, pi)
    passed = all(ar[k] for k in "abcd")
    acc47, ar47 = acceptance(root, sel, add_force(root, exact), pi)
    passed47 = all(ar47[k] for k in "abcd")
    lines[1:1] = [
        "",
        "## R-047 未卖出持仓按负责人真实卖出时刻补卖（预登记；标准同 R-046）",
        "",
        "与 R-046 相同，唯一改动：平仓卖单在负责人真实卖出时刻（该卖单最后一笔真实成交）仍未全部"
        "成交时，"
        "剩余张数于该时刻按买一卖出（价格改善按全部可立即成交样本抽样，买一为下限）；补卖算作该"
        "卖单成交。",
        "",
        df_to_md(acc47),
        "",
        f"**结论：{'通过' if passed47 else '不通过'}**（{sum(ar47[k] for k in 'abcd')}/4）。"
        f"发生补卖的卖单 {ar47['n_forced_orders']} 张；不计补卖的成交张数一致率 "
        f"{ar47['a_qty_rate_no_forced']:.1%}（参考）。改善量与重放在同一批订单上，属样本内检验。",
        "",
        "差额来源（事后描述）：种子 0 的模拟合计与对账单差 "
        f"{ar47['gap_total']:+,.2f} USD，其中成交张数不一致的 {ar47['n_qty_eps']} 段 "
        f"{ar47['gap_qty_eps']:+,.2f}；已成交张数上的成交价差合计 {ar47['gap_px_usd']:+,.2f} USD"
        "（正 = 模拟对负责人更好）。",
        "",
        "## R-046 价格改善模型验收（预登记标准，四项全过才算可用；结果保留）",
        "",
        "重放口径同下文 (1)：真实提交时刻、真实挂单与改价、真实卖出；可立即成交的订单成交价 = 对"
        "手价 − "
        "抽样改善量（`config/price_improvement.yaml`，`PRICE_IMPROVEMENT.md`）。",
        "",
        df_to_md(acc),
        "",
        f"**结论：{'通过' if passed else '不通过'}**（{sum(ar[k] for k in 'abcd')}/4）。"
        "改善量在同一批订单上校准，属样本内检验。下文为加入改善模型之前（吃价差 / 按挂单价）的检"
        "验，作对照。",
        "",
        "差额来源（事后描述，不改变上面的判定）：种子 0 的模拟合计比对账单少 "
        f"{-ar['gap_total']:,.2f} USD，其中成交张数不一致的 {ar['n_qty_eps']} 段贡献 "
        f"{-ar['gap_qty_eps']:,.2f}（最差一段 {ar['worst_ep_diff']:+,.2f}：模拟中平仓卖单没有成交，"
        "重放把未卖出的持仓按 0 计）；已成交张数上的成交价差合计 "
        f"{ar['gap_px_usd']:+,.2f} USD（正 = 模拟对负责人更好）。"
        "未卖出持仓如何估值没有在预登记里规定，这里不据此重新判定。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    store = root / "research" / "intrabar"
    pd.Series(ar["totals"]).to_frame("total").to_parquet(store / "r046_totals.parquet")
    pd.Series(ar47["totals"]).to_frame("total").to_parquet(store / "r047_totals.parquet")
    ar47["orders"].to_parquet(store / "r047_orders_seed0.parquet", index=False)
    print(acc47.to_string())
    ar["orders"].to_parquet(store / "r046_orders_seed0.parquet", index=False)
    print(acc.to_string())
    store.mkdir(parents=True, exist_ok=True)
    ex.to_parquet(store / "ka_exact.parquet", index=False)
    pe.to_parquet(store / "ka_episodes.parquet", index=False)
    print(summ.to_string())
    return 0 if (pass1 and pass2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
