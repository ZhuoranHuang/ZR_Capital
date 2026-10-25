"""新旧引擎对账（回测引擎 v1.1 任务书第 4 项）：MU 48 段已知答案。

基准 = −37,386.94 USD（吃价差回放，docs/research/INTRABAR_KNOWN_ANSWER.md 第 59 行）。
输入 $ZR_DATA_ROOT/research/intrabar/r048_orders_seed0.parquet（MU 段）。先做基准自检：
zr.research.intrabar_known_answer.pnl 逐段 sim_pnl 合计必须 = 基准 ± 0.01，不等 → 停止，不对账。
对账：每段一次 zr.sim.engine.simulate_day，signal = 买入单 t_ns，exit_rule = {"kind": "at",
"ts_ns": 卖出单 t_ns}（有多笔卖出时用最后一笔，类别记 multi_sell）；报价只读已缓存的逐笔文件
（owner_rules_oos.CachedOptionData，不拉新数据）；参数 lag_s=2.0、max_entry_wait_s=3600、
max_spread_pct=1.0、min_coverage=0.0、cooldown_s=0、日内风控关闭、session 09:30 / 15:59 / 16:00。
张数：对账专用 monkeypatch zr.sim.engine.size_position，返回该段买入的 filled 张数（不改引擎）。
运行：.venv/bin/python scripts/reconcile_engine_v1.py
"""

from __future__ import annotations

import sys

import pandas as pd

import zr.sim.engine as eng
from zr.config import get_settings
from zr.research import intrabar_known_answer as ka
from zr.research import owner_rules_oos as o

BASELINE = -37386.94
TOL_SELF = 0.01


def main() -> int:
    root = get_settings().zr_data_root
    d = pd.read_parquet(root / "research" / "intrabar" / "r048_orders_seed0.parquet")
    d = d[d["symbol"].str.startswith("MU")].copy()
    base = ka.pnl(d).set_index("episode")["sim_pnl"]
    self_check = float(base.sum())
    print(f"基准自检：ka.pnl 合计 = {self_check:,.2f}；要求 {BASELINE:,.2f} ± {TOL_SELF}")
    if abs(self_check - BASELINE) > TOL_SELF:
        print("基准自检不通过 → 停止，不对账")
        return 2
    od = o.CachedOptionData(root, "MU")
    pol = eng.SimPolicy(
        lag_s=2.0,
        max_entry_wait_s=3600,
        max_spread_pct=1.0,
        min_coverage=0.0,
        cooldown_s=0,
        max_daily_loss_pct=None,
        max_entries_per_day=None,
    )
    session = {"first_entry": "09:30", "last_entry": "15:59", "flat": "16:00"}
    rows = []
    for ep, g in d.groupby("episode"):
        b, s = g[g["side"] == 1].iloc[0], g[g["side"] == -1].sort_values("t_ns")
        day = pd.Timestamp(str(b["day"])).date()
        sym = str(b["symbol"])
        bk = od.book(
            sym,
            day,
            pd.Timestamp(int(b["t_ns"]), tz="UTC") - pd.Timedelta(minutes=1),
            pd.Timestamp(int(s["t_ns"].max()), tz="UTC") + pd.Timedelta(minutes=5),
        )
        q = pd.DataFrame({"ts_ns": bk.q_ns, "symbol": sym, "bid": bk.bid, "ask": bk.ask})
        qty = int(b["filled"])
        eng.size_position = lambda *a, _q=qty, **k: _q  # 对账专用：张数 = 该段买入 filled
        r = eng.simulate_day(
            q,
            [
                (
                    int(b["t_ns"]),
                    1,
                    sym,
                    {"kind": "at", "ts_ns": int(s["t_ns"].iloc[-1]), "sl": None},
                )
            ],
            eng.Account(equity_usd=1e9),
            session,
            pol,
        )
        new = r.trades[0].pnl_usd if r.trades else float("nan")
        diff = new - float(base.loc[ep])
        cat = ""
        if abs(diff) > 0.01:
            t = r.trades[0] if r.trades else None
            if len(s) > 1:
                cat = "multi_sell"
            elif t is None or t.qty != int(b["filled"]):
                cat = "qty_differs"
            elif abs(t.entry_px - float(b["px"])) > 1e-9:
                cat = "entry_quote_differs"
            else:
                cat = "exit_quote_differs"
        rows.append(
            {"episode": ep, "基准": float(base.loc[ep]), "新引擎": new, "差": diff, "类别": cat}
        )
    t = pd.DataFrame(rows)
    tot_b, tot_n = t["基准"].sum(), t["新引擎"].sum()
    sign_ok = ((t["基准"] > 0) == (t["新引擎"] > 0)).all()
    exact = int((t["差"].abs() <= 0.01).sum())
    total_ok = abs(tot_n - tot_b) <= 0.01 * abs(tot_b)
    print(t.round(2).to_string(index=False))
    print(f"合计：基准 {tot_b:,.2f}，新引擎 {tot_n:,.2f}，差 {tot_n - tot_b:,.2f}")
    print(
        f"验收：总额差 ≤ 1% × |基准| {'通过' if total_ok else '不通过'}；"
        f"符号一致 {'通过' if sign_ok else '不通过'}；逐笔差 ≤ $0.01 {exact}/48"
        f"（{'通过' if exact >= 45 else '不通过'}）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
