"""OWNER_RULES_OOS.md（R-049）的生成；数字全部来自 owner_rules_oos 的落盘结果。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zr.ops.markdown import df_to_md
from zr.research import owner_rules_oos as o


def _conds(r: dict) -> str:
    return "；".join(
        f"{c.get('name', c['feature'])} {c['op']} {c.get('value', 0):.4g}"
        for c in r["entry"]["all"]
    )


def write(root: Path, rules: dict, days: list, stock: pd.DataFrame, opt: pd.DataFrame) -> dict:
    n_days = len(days)
    ss = o.stock_stats(stock, n_days)
    os_ = o.option_stats(opt) if len(opt) else pd.DataFrame()
    if len(os_):
        ss, os_ = o.fdr(ss, os_)
    else:  # 期权层未运行：只对正股格做 BH
        m = (ss["dir"] == "合并").to_numpy()
        ss["bh"] = None
        ss.loc[m, "bh"] = o.v4d.bh(ss.loc[m, "p"].fillna(1).to_numpy(), o.Q)
    st = o.store(root)
    ss.to_parquet(st / "stock_stats.parquet", index=False)
    if len(os_):
        os_.to_parquet(st / "option_stats.parquet", index=False)
    L = []
    a = L.append
    a("# R-049：负责人规则 / 调研规则 / 融合规则的样本外检验")
    a("")
    a(
        "`uv run python -m zr.research.owner_rules_oos`。预登记见 RESEARCH_LOG R-049（运行前写入，"
        "未在样本内调参）。"
        f"样本外：MU {len(days)} 个交易日（{days[0]}～2026-05-31、2026-09-01～{days[-1]}），"
        "2026-06～08 为成交参数校准窗口（D-033），引擎拒绝。"
    )
    a("")
    a("## 1. 规则")
    a("")
    a(
        df_to_md(
            pd.DataFrame(
                [
                    {"规则": k, "条件（方向化，d = ±1 分别评估）": _conds(r)}
                    for k, r in rules.items()
                ]
            )
        )
    )
    a("")
    n_cells = int(ss["bh"].notna().sum()) + (int(os_["bh"].notna().sum()) if len(os_) else 0)
    a(
        f"多重比较：主格 {n_cells} 个（正股合并格 + 期权 D-001 格），BH q = 0.10。"
        "通过 = BH 拒绝且均值 > 0。"
    )
    a("")
    a("## 2. 第一层：正股（触发 vs 同日同时段随机 50 秒，不计费用）")
    a("")
    main = ss[ss["dir"] == "合并"].copy()
    piv = main.pivot_table(index="rule", columns="h_min", values="excess_bp").round(2)
    piv.columns = [f"超额 {c} 分钟（bp）" for c in piv.columns]
    pp = main.pivot_table(index="rule", columns="h_min", values="p").round(3)
    pp.columns = [f"p {c} 分钟" for c in pp.columns]
    freq = main.groupby("rule")["per_day"].first().round(2).rename("触发 / 日")
    bhp = (
        main.groupby("rule")["bh"]
        .apply(lambda x: int(x.fillna(False).astype(bool).sum()))
        .rename("BH 通过格数")
    )
    a(df_to_md(pd.concat([freq, piv, pp, bhp], axis=1).reset_index()))
    a("")
    a(
        "超额 = 触发的方向化收益 − 其 50 个随机时刻同方向收益的均值；p = 按交易日聚类的单侧 t 检验"
        "（H0：超额 ≤ 0）。"
    )
    a("")
    a("### 多 / 空分开（15 分钟持有期，描述）")
    a("")
    d15 = ss[ss["h_min"] == 15][
        ["rule", "dir", "n", "ret_mean_bp", "rand_mean_bp", "excess_bp", "pct_mean", "hit", "p"]
    ]
    a(df_to_md(d15.round(3)))
    a("")
    a("### 全部格（含 MFE / MAE 中位数）")
    a("")
    a(df_to_md(ss.round(3)))
    a("")
    a("## 3. 第二层：期权（盘中引擎，1 张，费用 D-013）")
    a("")
    a(option_coverage(root, days))
    a("")
    if len(os_):
        a(
            "exec：owner = 负责人挂单策略（R-041）入场、止盈与时间上限按 R-041 卖出、止损吃买一；"
            "taker = 入场离场 +2 秒吃对手价。zdte：d001 = 10:15 后不开 0DTE、0DTE 10:45 前清仓（"
            "主口径）；"
            "free = 不限（敏感性）。net_pct_mean = 每笔净收益占权利金的百分比均值；p = 按交易日"
            "聚类单侧 t。"
        )
        a("")
        a(df_to_md(os_[os_["slice"] == "全部"].round(3)))
        a("")
        a("### 按方向与年份（描述）")
        a("")
        a(df_to_md(os_[os_["slice"] != "全部"].round(3)))
        a("")
    else:
        a("（期权层未运行）")
        a("")
    a("## 4. 边界")
    a("")
    a("- 标的只有 MU；正股价为看跌看涨平价反推的秒级价（误差中位数约 2–3 bp），第一层不计费用。")
    a(
        "- 触发去重约定（5 分钟、ORB 每日 1 次）、随机对照窗口（±30 分钟）为预登记约定；第二层同"
        "一时间"
        "只持一个仓位，后续触发被占用时跳过。"
    )
    a(
        "- 期权成交按 R-048 验收的引擎（盘口 / OPRA 成交越价、恰好等于以 0.9 概率），负责人挂单"
        "策略参数"
        "来自 2026-06～08 校准；未覆盖：多张数下的冲击、0DTE 减半（D-014，1 张无法减半）、负责人"
        "的加减仓。"
    )
    a(
        "- 规则来自 15 天样本内挖掘或公开策略，本检验只说明这些具体参数在 MU 上的样本外表现，不"
        "外推到其他标的或参数。"
    )
    a("")
    o.OUT.write_text("\n".join(L), encoding="utf-8")
    return {"stock": ss, "option": os_}


def option_coverage(root: Path, days: list) -> str:
    """期权层用到 / 跳过的交易日（预取中止后只用已缓存数据）。"""
    ck = o.store(root) / "option_days"
    used = sorted(f.stem for f in ck.glob("*.parquet"))
    sk = {f.stem: f.read_text(encoding="utf-8") for f in ck.glob("*.skipped")}
    late = sorted(d for d, r in sk.items() if r.startswith("预取中止"))
    bad = sorted(d for d in sk if d not in late)

    def rng(x):
        return f"{x[0]}～{x[-1]}，{len(x)} 天" if x else "无"

    s = (
        "**结果基于部分样本。** 期权逐笔数据预取于 2026-10-24 17:09 SGT 按负责人指令中止（Theta"
        "Data "
        "拉取速度下降），之后不再拉取任何期权数据；期权层只用已完整缓存的交易日，其余跳过。\n\n"
        f"- 第一层（正股）样本外交易日：{rng([str(d) for d in days])}。\n"
        f"- 期权层用到：{rng(used)}。\n"
        f"- 跳过（未预取，{o.PREFETCH_DONE_THROUGH} 之后）：{rng(late)}。\n"
        f"- 跳过（{o.PREFETCH_DONE_THROUGH} 及之前但所需时间窗缓存不完整或文件损坏）：{rng(bad)}"
    )
    if bad:
        s += "（" + "、".join(bad) + "）"
    s += "。\n- 期权层结果不覆盖 2026-04-29 之后（含 2026-09～10 全部样本外月份）。"
    return s
