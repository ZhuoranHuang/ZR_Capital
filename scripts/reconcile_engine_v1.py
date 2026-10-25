# ruff: noqa: E501  （对账报告的中文输出行较长）
"""新旧引擎对账（引擎 v1.1 收尾任务书第 4 项）：MU 48 段，吃价差口径。

基准 = −37,386.94 USD（ka_episodes.parquet 的 taker_pnl 合计，docs/research/INTRABAR_KNOWN_ANSWER.md
第 59 行）。先做基准自检，不等 → 停止。
输入：zr.research.intrabar_known_answer.legs(root) 的第 4 个返回值 taker（48 买 + 51 卖），逐条断言
order["mode"] == "taker" 且 order["lag_s"] == 2.0（不是则按实际值设置并报告）。
每个 episode 单独调用一次 zr.sim.engine.simulate_day：signal = 买单 t_ns，exit_rule = {"kind": "at",
"ts_ns": 第一笔卖单 t_ns}；张数 = 该 episode 买入张数之和（对账专用：monkeypatch
zr.sim.engine.size_position 返回该张数，不改引擎）。报价只读已缓存的逐笔文件
（owner_rules_oos.CachedOptionData，不拉新数据），转成 DataFrame[ts_ns, symbol, bid, ask]。
参数：max_entry_wait_s=3600、max_spread_pct=1.0、min_coverage=0.0、cooldown_s=0、日内风控关闭、
session 09:30 / 15:59 / 16:05。
基准 base_d013（负责人 2026-10-25 改）：与 taker 同样的成交价与张数，双边费用按 D-013
（option_order_fees）重算，复刻旧 pnl() 的持仓上限（每笔卖出 min(持仓, filled)，最后一笔卖完剩余）。
验收：A 组（单次卖出）逐笔 |pnl_new − base_d013| ≤ 0.01、组合计差 ≤ 1.00、符号一致；B 组（多次卖出）对反事实
pnl_expected = (第一笔卖出价 − 买入价) × 100 × 张数 − fees(张数, Buy, 买入价) − fees(张数, Sell,
第一笔卖出价)（旧吃价差重放的成交价，option_order_fees）逐笔 ≤ 0.01；差额闭合：Σtaker_pnl −
gap = Σ(taker_pnl − base_d013)，fees_gap = Σ[option_order_fees(qty_sell_used, "Sell", sell_px) −
sim_v3.sell_fee(qty_sell_used, sell_px)]（D-013 减旧，负责人 2026-10-25 批准的符号约定），
|gap − fees_gap| ≤ 0.01；并打印 48 笔合计相对旧基准的分解（卖出费用修正 + B 组一次全平），±0.01 自洽。
运行：PYTHONPATH=. .venv/bin/python scripts/reconcile_engine_v1.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import zr.sim.engine as eng
from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.research import intrabar_known_answer as ka
from zr.research import owner_rules_oos as o

BASELINE = -37386.94


def sell_fee(n: float, prem: float) -> float:
    """旧卖出费（2026-10-25 之前的 sim_v3.sell_fee，冻结副本）：基础费在 n >= 2 时固定为 2 张的值。
    sim_v3.sell_fee 已按 D-035 改为 D-013，对账基准 ka_episodes 仍是旧费率，所以这里保留旧公式。"""
    base = {k: float(option_order_fees(k, "Sell", 0.0)["total"]) - 0.01 for k in (1, 2)}[
        2 if n >= 2 else 1
    ]
    v = 0.00206 * prem * n
    return base + max(np.floor(v * 100 + 0.5 + 1e-9) / 100, 0.01)


def _fees(n: int, side: str, px: float) -> float:
    return float(option_order_fees(int(n), side, px)["total"])


def main() -> int:
    root = get_settings().zr_data_root
    ep = pd.read_parquet(root / "research" / "intrabar" / "ka_episodes.parquet")
    self_check = float(ep["taker_pnl"].sum())
    print(
        f"基准自检：ka_episodes taker_pnl 合计 = {self_check:,.2f}（要求 {BASELINE:,.2f} ± 0.01）"
    )
    if abs(self_check - BASELINE) >= 0.01:
        print("基准自检不通过 → 停止")
        return 2
    taker = ka.legs(root)[3]
    lags = {(lg["order"]["mode"], float(lg["order"]["lag_s"])) for lg in taker}
    lag_ok = lags == {("taker", 2.0)}
    print(
        f"legs() taker 订单 {len(taker)} 条；(mode, lag_s) 取值 = {sorted(lags)}；"
        f"断言 {'通过' if lag_ok else '不通过'}"
    )
    lag_s = 2.0 if lag_ok else float(next(iter(lags))[1])
    # 旧吃价差重放的逐条成交价（只读缓存：把 intrabar.OptionData 换成离线版，缺数据即报错，不拉取）
    ib.OptionData = o.CachedOptionData
    tk = pd.DataFrame(ib.replay(root, "MU", taker, q=1.0))
    od = o.CachedOptionData(root, "MU")
    pol = eng.SimPolicy(
        lag_s=lag_s,
        max_entry_wait_s=3600,
        max_spread_pct=1.0,
        min_coverage=0.0,
        cooldown_s=0,
        max_daily_loss_pct=None,
        max_entries_per_day=None,
    )
    session = {"first_entry": "09:30", "last_entry": "15:59", "flat": "16:05"}
    base = ep.set_index("episode")["taker_pnl"]
    rows = []
    gap_fees = 0.0
    for e, g in tk.groupby("episode"):
        b = g[g["side"] == 1].sort_values("t_ns")
        s = g[g["side"] == -1].sort_values("t_ns")
        sym, day = str(b["symbol"].iloc[0]), b["day"].iloc[0]
        qty_buy = int(round(b["filled"].sum()))
        buy_px = float(b["px"].iloc[0])
        # base_d013：与旧 pnl() 同样的价格、张数与持仓上限（每笔卖出 min(持仓, filled)，最后一笔卖完剩余），
        # 费用全部用 D-013（option_order_fees）
        held, cash = float(qty_buy), 0.0
        cash -= sum(
            float(r.px) * 100 * float(r.filled) + _fees(int(r.filled), "Buy", float(r.px))
            for r in b.itertuples()
        )
        qty_sell_used = 0
        for k, r in enumerate(s.itertuples()):
            n = min(held, float(r.filled)) if k < len(s) - 1 else held
            if n > 0:
                cash += float(r.px) * 100 * n - _fees(int(n), "Sell", float(r.px))
                gap_fees += _fees(int(n), "Sell", float(r.px)) - float(sell_fee(n, float(r.px)))
                held -= n
                qty_sell_used += int(n)
        base_d013 = cash
        bk = od.book(
            sym,
            day,
            pd.Timestamp(int(b["t_ns"].iloc[0]), tz="UTC") - pd.Timedelta(minutes=1),
            pd.Timestamp(int(s["t_ns"].iloc[-1]), tz="UTC") + pd.Timedelta(minutes=2),
        )
        q = pd.DataFrame({"ts_ns": bk.q_ns, "symbol": sym, "bid": bk.bid, "ask": bk.ask})
        eng.size_position = lambda *a, _q=qty_buy, **k: (
            _q
        )  # 对账专用：张数 = 该 episode 买入 filled 之和
        rule = {"kind": "at", "ts_ns": int(s["t_ns"].iloc[0]), "sl": None}
        r = eng.simulate_day(
            q,
            [(int(b["t_ns"].iloc[0]), 1, sym, rule)],
            eng.Account(equity_usd=1e9),
            session,
            pol,
        )
        t = r.trades[0]
        grp = "A" if len(s) == 1 else "B"
        sell1 = float(s["px"].iloc[0])
        expected = (
            (sell1 - buy_px) * 100 * qty_buy
            - _fees(qty_buy, "Buy", buy_px)
            - _fees(qty_buy, "Sell", sell1)
        )
        ref = base_d013 if grp == "A" else expected
        rows.append(
            {
                "episode": int(e),
                "组别": grp,
                "qty_buy": qty_buy,
                "qty_sell_used": qty_sell_used,
                "buy_px": buy_px,
                "sell_px": sell1,
                "taker_pnl(旧费率)": float(base.loc[e]),
                "base_d013": base_d013,
                "expected(B)": expected if grp == "B" else np.nan,
                "pnl_new": t.pnl_usd,
                "diff(base)": t.pnl_usd - base_d013,
                "diff(旧)": t.pnl_usd - float(base.loc[e]),
                "diff(对照)": t.pnl_usd - ref,
                "类别": "multi_sell" if grp == "B" else "",
                "_px_ok": abs(t.entry_px - buy_px) < 1e-9 and abs(t.exit_px - sell1) < 1e-9,
            }
        )
    d = pd.DataFrame(rows)
    a, bb = d[d["组别"] == "A"], d[d["组别"] == "B"]
    a_each = bool((a["diff(base)"].abs() <= 0.01).all())
    a_tot = abs(a["pnl_new"].sum() - a["base_d013"].sum())
    a_sign = bool((np.sign(a["pnl_new"]) == np.sign(a["base_d013"])).all())
    b_each = bool((bb["diff(对照)"].abs() <= 0.01).all())
    gap = float((d["taker_pnl(旧费率)"] - d["base_d013"]).sum())
    close_ok = abs(gap - gap_fees) <= 0.01
    # 分解：新引擎 48 笔合计 − 旧基准 = 卖出费用修正（Σbase_d013 − Σtaker_pnl）+ B 组一次全平
    # （ΣB pnl_new − ΣB base_d013）
    total_new = float(d["pnl_new"].sum())
    fee_effect = float(d["base_d013"].sum() - d["taker_pnl(旧费率)"].sum())
    b_effect = float(bb["pnl_new"].sum() - bb["base_d013"].sum())
    decomp_ok = abs((fee_effect + b_effect) - (total_new - BASELINE)) <= 0.01
    pd.set_option("display.width", 300)
    print(d.drop(columns=["diff(对照)", "_px_ok"]).round(2).to_string(index=False))
    print(
        f"成交价逐笔一致（入场 ask、出场 bid 与旧吃价差重放相同）：{int(d['_px_ok'].sum())}/{len(d)}"
    )
    print(
        f"A 组 {len(a)} 笔：Σbase_d013 {a['base_d013'].sum():,.2f}，Σpnl_new {a['pnl_new'].sum():,.2f}"
        "，"
        f"差 {a['pnl_new'].sum() - a['base_d013'].sum():,.2f}；逐笔 ≤ 0.01 "
        f"{int((a['diff(base)'].abs() <= 0.01).sum())}/{len(a)}；符号一致 {a_sign}"
    )
    print(
        f"B 组 {len(bb)} 笔：Σexpected {bb['expected(B)'].sum():,.2f}，Σpnl_new {bb['pnl_new'].sum():,.2f}"
        "，"
        f"差 {bb['pnl_new'].sum() - bb['expected(B)'].sum():,.2f}"
    )
    print(f"gap = Σ(taker_pnl − base_d013) = {gap:,.4f}")
    print(f"fees_gap = Σ[D-013 卖出费 − 旧 sell_fee] = {gap_fees:,.4f}")
    print(
        f"gap − fees_gap = {gap - gap_fees:,.4f}（|·| ≤ 0.01：{'通过' if close_ok else '不通过'}）"
    )
    print(
        f"分解：新引擎 48 笔合计 ΣA {a['pnl_new'].sum():,.2f} + ΣB {bb['pnl_new'].sum():,.2f} = "
        f"{total_new:,.2f}；相对旧基准 {BASELINE:,.2f} 差 {total_new - BASELINE:+,.2f}"
    )
    print(f"  卖出费用修正（旧 → D-013）{fee_effect:+,.2f}（fees_gap 闭合检查）")
    print(f"  B 组一次全平 vs 多次卖出 {b_effect:+,.2f}（B 组反事实逐笔）")
    print(
        f"  合计 {fee_effect + b_effect:+,.2f}；= Σpnl_new − 旧基准 {total_new - BASELINE:+,.2f}"
        f"（±0.01 自洽：{'通过' if decomp_ok else '不通过'}）"
    )
    ok = a_each and a_tot <= 1.00 and a_sign and b_each and close_ok and decomp_ok
    print(
        f"验收：A 逐笔 {'通过' if a_each else '不通过'}；A 合计差 {a_tot:.2f} "
        f"{'通过' if a_tot <= 1.00 else '不通过'}；A 符号 {'通过' if a_sign else '不通过'}；"
        f"B 反事实 {'通过' if b_each else '不通过'}；差额闭合 {'通过' if close_ok else '不通过'} → "
        f"{'全部通过' if ok else '未通过'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
