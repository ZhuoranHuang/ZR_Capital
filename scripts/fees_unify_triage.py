# ruff: noqa: E501  （中文报告输出行较长）
"""费用口径统一（fix/fees-unify，D-035）的分诊重算。

旧卖出费 = 修正前的 sim_v3.sell_fee（基础费在 n >= 2 时固定为 2 张的值）；新 = D-013
（option_order_fees）。两者只在与张数有关的基础费上不同（按权利金计的 SEC 公式相同），所以每张卖单的
修正量只取决于张数：d(n) = D-013(n) − 旧(n)，n = 1、2 时为 0。
分诊规则：修正量 = Σ(D-013 − 旧)；|修正量| < 0.05 × |结果与判定线的距离| → 只记录；否则重跑。
只读已缓存的逐笔数据（owner_rules_oos.CachedOptionData），不拉新数据。
运行：PYTHONPATH=. .venv/bin/python scripts/fees_unify_triage.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.backtest import intrabar as ib
from zr.backtest.fill_limit import PriceImprovement
from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.research import intrabar_known_answer as ka
from zr.research import owner_limit as ol
from zr.research import owner_rules_oos as o

REPO = ka.REPO
STATEMENT = 9734.90  # 对账单合计（R-046～R-048 判定 (c)(d) 的对照）


def old_sell_fee(n: int, prem: float) -> float:
    """修正前 sim_v3.sell_fee 的标量版（只用于计算修正量）。"""
    base = {k: float(option_order_fees(k, "Sell", 0.0)["total"]) - 0.01 for k in (1, 2)}[
        2 if n >= 2 else 1
    ]
    v = 0.00206 * prem * n
    return base + max(np.floor(v * 100 + 0.5 + 1e-9) / 100, 0.01)


def d(n: int, prem: float = 10.0) -> float:
    """卖出费修正量 D-013 − 旧（> 0 = 新费用更高、盈亏更低）。"""
    return (
        0.0
        if n <= 0
        else float(option_order_fees(int(n), "Sell", prem)["total"]) - old_sell_fee(int(n), prem)
    )


def main() -> int:
    root = get_settings().zr_data_root
    st = root / "research" / "intrabar"
    print("d(n)（与权利金无关）：", {n: round(d(n), 2) for n in (1, 2, 3, 4, 5, 10, 20)})
    sel, orders = ol.episode_orders(root)
    mu = sel[sel["und"] == "MU"]

    # R-046 / R-047（种子 0，存档订单按 D-013 重算；两项原本不通过，修正只会更远离对账单）
    for tag in ("r046", "r047"):
        od = pd.read_parquet(st / f"{tag}_orders_seed0.parquet")
        od = od[od["symbol"].str.startswith("MU")]
        old = float(pd.read_parquet(st / f"{tag}_totals.parquet")["total"].iloc[0])
        new = float(ka.pnl(od)["sim_pnl"].sum())
        print(
            f"{tag.upper()} 种子 0 合计：旧 {old:,.2f} → 新 {new:,.2f}（修正 {new - old:+,.2f}）；"
            f"(c) 差 {abs(old - STATEMENT) / STATEMENT:.1%} → {abs(new - STATEMENT) / STATEMENT:.1%}（判定线 10%）"
        )

    # R-048：(d) 余量 P95 − 对账单 = 89.33，修正量约 −114 → 可能跨线 → 100 个种子全部重跑（只读缓存）
    ib.OptionData = o.CachedOptionData
    _, _, exact, _ = ka.legs(root)
    pi = PriceImprovement.from_yaml(REPO / "config" / "price_improvement.yaml")
    old_tot = pd.read_parquet(st / "r048_totals.parquet")["total"].to_numpy()
    acc, ar = ka.acceptance(root, mu, ka.add_force(root, exact, 999_000_000), pi)
    new_tot = np.asarray(ar["totals"])
    for lab, x in (("旧", old_tot), ("新", new_tot)):
        p5, p95 = np.quantile(x, [0.05, 0.95])
        pct = float((x < STATEMENT).mean())
        print(
            f"R-048 {lab}：种子 0 {x[0]:,.2f}；P5 {p5:,.2f} / P95 {p95:,.2f}；对账单在第 {pct:.0%} 分位；"
            f"(c) 差 {abs(x[0] - STATEMENT) / STATEMENT:.1%}；(d) {'通过' if p5 <= STATEMENT <= p95 else '不通过'}"
        )
    print(
        f"R-048 每个种子的修正量：均值 {np.mean(new_tot - old_tot):+,.2f}，范围 {np.min(new_tot - old_tot):+,.2f}～{np.max(new_tot - old_tot):+,.2f}"
    )
    print("R-048 验收（新）：", {k: ar[k] for k in "abcd"})

    # R-052 第 1 步网格与出场挖掘替换表：每段一次全部卖出（张数 = 该段买入张数），修正对每个组合 / 规则相同
    qty = mu["buy_q"].astype(int)
    shift = -sum(d(int(q)) for q in qty)
    print(
        f"R-052 第 1 步 / 出场挖掘替换：每个组合与规则统一平移 {shift:+,.2f}（48 段，张数 = 买入张数）"
    )
    g = pd.read_parquet(root / "research" / "exit_finalize" / "grid_table.parquet")
    g["new_total"] = g["total"] + shift
    ok = g[g["sl"] > 0]
    pick_old = ok.loc[(ok["total"] - 9734.90).abs().idxmin(), ["sl", "cap"]].tolist()
    pick_new = ok.loc[(ok["new_total"] - 9734.90).abs().idxmin(), ["sl", "cap"]].tolist()
    print(g[["sl", "cap", "total", "new_total"]].round(2).to_string(index=False))
    print(f"R-052 选定组：旧 {pick_old} → 新 {pick_new}（统一平移不改变排序）")
    rep = pd.read_parquet(root / "research" / "exit_mining" / "replace.parquet")
    t = rep.groupby(["rule", "exec"])["sim"].sum()
    print((pd.DataFrame({"旧合计": t, "新合计": t + shift})).round(2).to_string())

    # R-040 / R-043（负责人出场部分）：模拟卖出与真实卖出两侧用同一费用函数，修正 = Σ[d(真实卖出张数) − d(模拟卖出张数)]
    canc = sum(d(int(s)) - d(int(b)) for b, s in zip(mu["buy_q"], mu["sell_q"], strict=True))
    print(
        f"R-040 / R-043：Σ[d(卖出张数) − d(买入张数)] = {canc:+,.2f}（买卖张数相同的段两侧修正相互抵消）"
    )

    # R-041 负责人重放（owner_replay）：模拟卖单按真实卖单张数依次卖出
    sells = orders[(orders["side"] == "Sell") & orders["symbol"].str.startswith("MU")]
    corr41 = -sum(d(int(q)) for q in sells["qty"])
    print(
        f"R-041 负责人重放 / 吃价差对照：修正量 ≈ {corr41:+,.2f}（51 张卖单按真实张数；部分成交时为近似）"
    )

    # R-042：同引擎对照的负责人模拟合计（确定性）重跑；随机分布只会因修正而下移
    from zr.research import skill_vs_luck as sv

    eps, holds = sv.setup(root)
    days = sv.prepare(root, eps)
    new_os = sv.owner_same_engine(root, eps, days)
    old_os = pd.read_pickle(root / "research" / "skill_vs_luck" / "totals.pkl")["owner_sim"]
    print(f"R-042 同引擎负责人模拟：旧 {old_os} → 新 {new_os}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
