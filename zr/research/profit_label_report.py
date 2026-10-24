"""docs/research/PROFIT_LABEL_MINING.md：四步队列的报告（第 1 节 R-049 正股补报、第 2 节出场挖掘、
第 3 节 R-050）。数字全部读落盘结果。运行：uv run python -m zr.research.profit_label_report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.ops.markdown import df_to_md
from zr.research import exit_mining as xm
from zr.research import owner_rules_oos as o
from zr.research import owner_rules_paths as rp
from zr.research import profit_label_mining as pl

OUT = Path(__file__).resolve().parents[2] / "docs" / "research" / "PROFIT_LABEL_MINING.md"


def _cond(c) -> str:
    return " 且 ".join(f"{n} {op} {v:.4g}" for n, op, v in c)


def _cov(g: dict) -> str:
    return f"{g['sell_cover']:.0%} / {g['hold_share']:.1%}"


def section1(root: Path) -> list[str]:
    p = o.store(root) / "paths.parquet"
    if not p.exists():
        stock = pd.read_parquet(o.store(root) / "stock.parquet")
        rp.run(root, stock).to_parquet(p, index=False)
    s = rp.summarize(pd.read_parquet(p))
    L = ["## 1. R-049 正股层补报（触发后路径，不计费用）", ""]
    L += [
        "触发 = R-049 第一层（时段内条件由假变真、同规则同方向 5 分钟去重、ORB 每日 1 次）；随机"
        " = 每个触发"
        "同日 ±30 分钟内同方向 5 个随机秒。收益为方向化（看涨按上涨、看跌按下跌为正），单位 %。",
        "",
        "### 平均方向化收益（%）",
        "",
        df_to_md(s["ret"].reset_index()),
        "",
        "### 最大有利偏移（MFE）中位数（%）",
        "",
        df_to_md(s["mfe_med"].reset_index()),
        "",
        "### 30 分钟内 MFE 分布（%）",
        "",
        df_to_md(s["mfe30_q"].reset_index()),
        "",
        "### 先后概率（30 分钟内；「先到 +x% / 先到 −y%」，其余为都没到）",
        "",
        df_to_md(s["first"].reset_index()),
        "",
    ]
    return L


def section2(root: Path) -> list[str]:
    st = xm.store(root)
    s = pd.read_parquet(st / "sample.parquet")
    L = ["## 2. 出场规律挖掘（D-030 样本内，不是检验）", ""]
    L.append(
        f"出场事件 = 卖单成交时刻 {int(s['label'].sum())} 个"
        f"（盈利段 {int(s.loc[s['win'], 'label'].sum())}、"
        f"亏损段 {int(s.loc[~s['win'], 'label'].sum())}），对照 = 同持仓期间每 10 秒仍持有的时刻 "
        f"{int((s['label'] == 0).sum())} 个，48 段。"
    )
    L.append("")
    for t, title in (("W", "盈利段"), ("L", "亏损段"), ("ALL", "合并")):
        u = pd.read_parquet(st / f"uni_{t}.parquet")
        top = u.reindex(u["z_perm"].abs().sort_values(ascending=False).index).head(12)
        L += [
            f"### {title}：卖出时刻 vs 持有时刻（单特征，置换 500 次）",
            "",
            f"经验 p ≤ 0.05：{int((u['p_perm'] <= 0.05).sum())} / {len(u)}；"
            f"BH 通过 {int(u['bh'].sum())}"
            "（持有期内特征强自相关，通过数会高估独立信息量）。",
            "",
            df_to_md(
                top[["feature", "auc", "med_sell", "med_hold", "pct_in_episode", "p_perm"]].round(3)
            ),
            "",
        ]
    mp = st / "models.parquet"
    if mp.exists():
        L += [
            "### 模型（按交易日留一；置换 500 次）",
            "",
            df_to_md(pd.read_parquet(mp).round(3)),
            "",
        ]
    rules = pd.read_pickle(st / "rules.pkl")
    L += ["### 出场规则（止盈组来自盈利段、止损组来自亏损段；任一组成立即全部卖出）", ""]
    L.append(
        df_to_md(
            pd.DataFrame(
                [
                    {
                        "规则": r["name"],
                        "止盈组": _cond(r["tp"]["conds"]),
                        "止盈组：卖出覆盖 / 持有占比": _cov(r["tp"]),
                        "止损组": _cond(r["sl"]["conds"]),
                        "止损组：卖出覆盖 / 持有占比": _cov(r["sl"]),
                    }
                    for r in rules
                ]
            )
        )
    )
    L.append("")
    rep = pd.read_parquet(st / "replace.parquet")
    act = rep.drop_duplicates("episode")
    t = rep.groupby(["rule", "exec"]).agg(
        合计_USD=("sim", "sum"),
        每笔_USD=("sim", "mean"),
        胜率=("sim", lambda x: (x > 0).mean()),
        出场分钟中位数=("exit_s", lambda x: (x / 60).median()),
    )
    by = rep[rep["exec"] == "owner"].groupby(["rule", "win"])["sim"].sum().unstack()
    by.columns = ["亏损段合计（R-041 卖出）", "盈利段合计（R-041 卖出）"]
    L += [
        "### 在负责人 48 笔真实入场上替换出场（扣费后，USD）",
        "",
        f"实际：合计 {act['actual'].sum():.2f}（盈利段 {act.loc[act['win'], 'actual'].sum():.2f}"
        "、亏损段 "
        f"{act.loc[~act['win'], 'actual'].sum():.2f}），胜率 {(act['actual'] > 0).mean():.0%}。"
        "替换：真实买入成交价与张数不变，规则首次成立时按 R-041 卖出全部张数（owner）或吃买一（"
        "taker），"
        "15:50 兜底；habit = +10% / −15% / 30 分钟。",
        "",
        df_to_md(t.round(2).reset_index()),
        "",
        df_to_md(by.round(2).reset_index()),
        "",
        f"R-050 出场 (b) 按预登记标准选 {pd.read_pickle(st / 'chosen_rule.pkl')['name']}（替换后"
        " R-041 卖出合计最高）。",
        "",
    ]
    return L


def section3(root: Path) -> list[str]:
    st = pl.store(root)
    L = ["## 3. R-050：以扣费后盈亏为标签的机会挖掘", ""]
    if not (st / "eval.parquet").exists():
        return L + ["（未运行）", ""]
    df = pl.load(root)
    cnt = df.groupby("split").agg(
        候选=("t_ns", "size"),
        交易日=("day", "nunique"),
        成交率=("filled_habit", "mean"),
        习惯出场每笔_pct=("net_pct_habit", "mean"),
        规则出场每笔_pct=("net_pct_rule", "mean"),
    )
    L += [
        "### 候选与基准（全部候选都做，成交笔的扣费后 % 权利金）",
        "",
        df_to_md(cnt.round(3).reset_index()),
        "",
    ]
    for name, title in (
        ("eval", "测试期（2026-09-01～10-21，只跑一次）"),
        ("val", "验证期（2026-01～05，只报告）"),
    ):
        e = pd.read_parquet(st / f"{name}.parquet")
        L += [f"### {title}", "", df_to_md(e.round(3)), ""]
    L += [
        "### 重要特征（验证期置换重要性前 10）",
        "",
        df_to_md(pd.read_parquet(st / "importance.parquet").round(4)),
        "",
    ]
    L += [
        "### 深度 ≤ 3 决策树的叶规则（训练期拟合；测试期扣费后）",
        "",
        df_to_md(pd.read_parquet(st / "tree.parquet").round(3)),
        "",
    ]
    return L


def write(root: Path) -> None:
    L = [
        "# 以盈亏为标准的出场与机会挖掘（四步队列）",
        "",
        "全部以扣费后盈亏为主评判；价格变动一律用 %。R-050 预登记见 RESEARCH_LOG。",
        "",
    ]
    L += section1(root) + section2(root) + section3(root)
    L += [
        "## 4. 边界",
        "",
        "- 第 1 节：MU 2025-01～2026-05 与 2026-09～10，正股秒级平价反推价，不计费用。",
        "- 第 2 节：负责人 D-030 样本 48 段（2026-07～08，D-033 校准窗口），样本内挖掘与替换，不"
        "是检验；"
        "替换只换出场、入场与张数用真实成交。",
        "- 第 3 节：每个候选独立模拟（不受持仓占用、资金约束），1 张；成交模型为 R-048 验收引擎；"
        "本机无 LightGBM，用 sklearn HistGradientBoosting；结论以测试期为准，测试期只有约 36 个"
        "交易日。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    write(get_settings().zr_data_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
