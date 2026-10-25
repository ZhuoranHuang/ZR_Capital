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
验收：A 组（单次卖出）逐笔 |pnl_new − taker_pnl| ≤ 0.01、组合计差 ≤ 1.00；B 组（多次卖出）对反事实
pnl_expected = (第一笔卖出价 − 买入价) × 100 × 张数 − fees(张数, Buy, 买入价) − fees(张数, Sell,
第一笔卖出价)（旧吃价差重放的成交价，option_order_fees）逐笔 ≤ 0.01；48 笔 sign(pnl_new) 与
sign(taker_pnl) 全部一致。
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
    for e, g in tk.groupby("episode"):
        b = g[g["side"] == 1].sort_values("t_ns")
        s = g[g["side"] == -1].sort_values("t_ns")
        sym, day = str(b["symbol"].iloc[0]), b["day"].iloc[0]
        qty = int(round(b["qty"].sum()))
        bk = od.book(
            sym,
            day,
            pd.Timestamp(int(b["t_ns"].iloc[0]), tz="UTC") - pd.Timedelta(minutes=1),
            pd.Timestamp(int(s["t_ns"].iloc[-1]), tz="UTC") + pd.Timedelta(minutes=2),
        )
        q = pd.DataFrame({"ts_ns": bk.q_ns, "symbol": sym, "bid": bk.bid, "ask": bk.ask})
        eng.size_position = lambda *a, _q=qty, **k: _q  # 对账专用：张数 = 该 episode 买入张数之和
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
        buy_px, sell1 = float(b["px"].iloc[0]), float(s["px"].iloc[0])
        if grp == "A":
            ref = float(base.loc[e])
        else:
            ref = (
                (sell1 - buy_px) * 100 * qty - _fees(qty, "Buy", buy_px) - _fees(qty, "Sell", sell1)
            )
        rows.append(
            {
                "episode": int(e),
                "组别": grp,
                "张数": qty,
                "买入价旧": buy_px,
                "买入价新": t.entry_px,
                "卖出价旧1": sell1,
                "卖出价新": t.exit_px,
                "taker_pnl": float(base.loc[e]),
                "对照值": ref,
                "pnl_new": t.pnl_usd,
                "差": t.pnl_usd - ref,
                "类别": "multi_sell" if grp == "B" else "",
            }
        )
    d = pd.DataFrame(rows)
    a, bb = d[d["组别"] == "A"], d[d["组别"] == "B"]
    a_each = bool((a["差"].abs() <= 0.01).all())
    a_tot = abs(a["pnl_new"].sum() - a["taker_pnl"].sum())
    b_each = bool((bb["差"].abs() <= 0.01).all())
    sign_ok = bool((np.sign(d["pnl_new"]) == np.sign(d["taker_pnl"])).all())
    pd.set_option("display.width", 250)
    print(d.round(4).to_string(index=False))
    print(
        f"A 组 {len(a)} 笔：Σtaker_pnl {a['taker_pnl'].sum():,.2f}，"
        f"Σpnl_new {a['pnl_new'].sum():,.2f}，"
        f"差 {a['pnl_new'].sum() - a['taker_pnl'].sum():,.2f}；"
        f"逐笔 ≤ 0.01 {int((a['差'].abs() <= 0.01).sum())}/{len(a)}"
    )
    print(
        f"B 组 {len(bb)} 笔：Σ反事实 {bb['对照值'].sum():,.2f}，"
        f"Σpnl_new {bb['pnl_new'].sum():,.2f}，"
        f"差 {bb['差'].sum():,.2f}；Σtaker_pnl {bb['taker_pnl'].sum():,.2f}"
    )
    print(f"48 笔 pnl_new 合计 {d['pnl_new'].sum():,.2f}（基准 {BASELINE:,.2f}）")
    ok = a_each and a_tot <= 1.00 and b_each and sign_ok
    print(
        f"验收：A 逐笔 {'通过' if a_each else '不通过'}；A 合计差 {a_tot:.2f} "
        f"{'通过' if a_tot <= 1.00 else '不通过'}；B 反事实 {'通过' if b_each else '不通过'}；"
        f"符号一致 {'通过' if sign_ok else '不通过'} → {'全部通过' if ok else '未通过'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
