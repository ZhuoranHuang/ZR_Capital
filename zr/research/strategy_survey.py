"""公开策略调研（第二条，docs/strategy-survey）。

来源表校验、分类计数、候选规则 YAML 校验，生成报告中的自动表格。
来源表 `docs/research/strategy_survey_sources.yaml` 手工整理（每条规则都来自所列链接）；
本模块只做计数与渲染，保证 STRATEGY_SURVEY.md 里的数字由代码算出（CLAUDE.md 规则 2）。
「最常见的趋势突破 / 顺势回调规则」= breakout / pullback 类里条目数最多的 family；
候选 YAML 中 oos_pick 的突破 / 回调规则必须正好是这两个 family（另加盘中 RSI(2) 改写），否则报错。
报告里 `<!-- AUTO:名称 -->` 与 `<!-- /AUTO:名称 -->` 之间的内容由本模块重写，其余是手写正文。
运行：uv run python -m zr.research.strategy_survey
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from zr.ops.markdown import df_to_md

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "docs" / "research" / "strategy_survey_sources.yaml"
REPORT = ROOT / "docs" / "research" / "STRATEGY_SURVEY.md"
RULES = ROOT / "config" / "intrabar"

CATEGORIES = {
    "breakout": "趋势突破",
    "pullback": "顺势回调",
    "mean_reversion": "均值回归",
    "confluence": "多指标共振",
    "vol_filter": "波动率过滤",
}
TREND_DEFS = {
    "price_vs_ma": "价在长均线一侧",
    "ma_alignment": "均线排列",
    "ma_slope": "均线斜率",
    "adx": "ADX / DI",
    "swing_structure": "高低点结构",
    "channel": "通道（Donchian / Keltner / BOLL / VWAP 带）",
    "atr_trail": "ATR 跟踪线状态",
    "htf": "高一级周期确认",
    "opening_range": "开盘区间方向",
    "momentum": "动量 / MACD 符号",
    "none": "不判断趋势",
}
SITUATIONS = {
    "ignition": "点火",
    "channel_pullback": "通道回调",
    "range_avoid": "回避震荡",
    "none": "不对应",
}
POP_UNITS = {"tv_likes": "TV 点赞", "gh_stars_repo": "GitHub star（仓库）", "none": "未取得"}
READS = {"pine", "python", "sql", "desc", "secondary"}
REQUIRED = [
    "id",
    "platform",
    "name",
    "url",
    "pop_unit",
    "read",
    "bar",
    "indicators",
    "trend_def",
    "entry",
    "exit",
    "claimed",
    "risks",
    "category",
    "family",
    "symmetric",
    "situation",
]
RULE_REQUIRED = [
    "name",
    "family",
    "category",
    "direction",
    "entry",
    "exit",
    "oos_pick",
    "features_needed",
]
ENGINE_FEATURES = {"ret", "dist_extreme", "boll", "pctb", "macd_hist", "macd_hist_chg", "ef"}
CAL_START, CAL_END = "2026-06-01", "2026-08-31"


def _sym(v: object) -> str:
    return "yes" if v is True else str(v)


def load_sources(path: Path = SOURCES) -> list[dict]:
    rows = yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]
    ids = [r.get("id") for r in rows]
    dup = [k for k, n in Counter(ids).items() if n > 1]
    if dup:
        raise ValueError(f"来源 id 重复：{dup}")
    for r in rows:
        miss = [k for k in REQUIRED if k not in r]
        if miss:
            raise ValueError(f"{r.get('id')} 缺字段 {miss}")
        if r["category"] not in CATEGORIES:
            raise ValueError(f"{r['id']} category 非法：{r['category']}")
        if r["read"] not in READS:
            raise ValueError(f"{r['id']} read 非法：{r['read']}")
        if r["pop_unit"] not in POP_UNITS:
            raise ValueError(f"{r['id']} pop_unit 非法：{r['pop_unit']}")
        bad = [t for t in r["trend_def"] if t not in TREND_DEFS]
        bad += [s for s in r["situation"] if s not in SITUATIONS]
        if bad:
            raise ValueError(f"{r['id']} 枚举值非法：{bad}")
        sym = _sym(r["symmetric"])
        if sym not in ("yes", "mirror"):
            raise ValueError(f"{r['id']} symmetric 非法：{sym}")
        if sym == "mirror" and not r.get("mirror"):
            raise ValueError(f"{r['id']} 只做一边但没给镜像")
    return rows


def win_rate_lower(n: int, alpha: float = 0.05) -> float:
    """n 笔全部盈利时真实胜率的单侧 1 − alpha 下界（Clopper–Pearson，k = n 时为 alpha^(1/n)）。"""
    return alpha ** (1.0 / n)


def most_common_family(rows: list[dict], category: str) -> tuple[str, Counter]:
    c = Counter(r["family"] for r in rows if r["category"] == category)
    top = max(c.values())
    tied = sorted(f for f, n in c.items() if n == top)
    if len(tied) > 1:
        raise ValueError(f"{category} 最常见 family 并列：{tied}")
    return tied[0], c


def load_rules(rule_dir: Path = RULES) -> dict[str, dict]:
    out = {}
    for p in sorted(rule_dir.glob("survey_*.yaml")):
        out[p.stem] = yaml.safe_load(p.read_text(encoding="utf-8"))
    return out


def check_rules(rules: dict[str, dict], rows: list[dict]) -> None:
    """字段齐全、来源 id 存在、只用样本外窗口、oos_pick 正好是 RSI(2) + 最常见突破 + 最常见回调。"""
    ids = {r["id"] for r in rows}
    for name, r in rules.items():
        miss = [k for k in RULE_REQUIRED if k not in r]
        if miss:
            raise ValueError(f"{name} 缺字段 {miss}")
        if not r["entry"].get("all"):
            raise ValueError(f"{name} entry.all 为空")
        if r["direction"] not in ("long", "short", "both", "trend"):
            raise ValueError(f"{name} direction 非法")
        unknown = [s for s in r.get("sources", []) if s not in ids]
        if unknown:
            raise ValueError(f"{name} 来源 id 不存在：{unknown}")
        declared = {f["name"] for f in r["features_needed"]}
        used = {c["feature"] for c in r["entry"]["all"]}
        undeclared = used - declared - ENGINE_FEATURES
        if undeclared:
            raise ValueError(f"{name} 用了未声明的特征：{undeclared}")
        if bool(declared & used) == bool(r.get("engine_ready", False)):
            raise ValueError(f"{name} engine_ready 与是否需要新特征不一致")
        for w in r.get("oos_windows", []) + [r.get("days", {})]:
            if w and not (str(w["to"]) < CAL_START or str(w["from"]) > CAL_END):
                raise ValueError(f"{name} 窗口 {w} 与校准窗口重叠（D-033）")
    picks = {n: r for n, r in rules.items() if r["oos_pick"]}
    bo, _ = most_common_family(rows, "breakout")
    pb, _ = most_common_family(rows, "pullback")
    want = {("breakout", bo), ("pullback", pb)}
    got = {(r["category"], r["family"]) for n, r in picks.items() if n != "survey_rsi2_pullback"}
    if "survey_rsi2_pullback" not in picks or got != want or len(picks) != 3:
        raise ValueError(f"oos_pick 应为 RSI(2) + {sorted(want)}，实际 {sorted(picks)}")


def tables(rows: list[dict], rules: dict[str, dict]) -> dict[str, str]:
    df = pd.DataFrame(rows)
    df["sym"] = df["symmetric"].map(_sym)
    out: dict[str, str] = {}

    cat = (
        df.groupby("category")
        .agg(
            条目=("id", "size"),
            多空对称=("sym", lambda s: int((s == "yes").sum())),
            只做一边_已给镜像=("sym", lambda s: int((s == "mirror").sum())),
        )
        .reindex(list(CATEGORIES))
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    cat.insert(1, "类别", cat["category"].map(CATEGORIES))
    cat = cat.rename(
        columns={"category": "category（代码）", "只做一边_已给镜像": "只做一边（已给镜像）"}
    )
    total = pd.DataFrame(
        [
            {
                "category（代码）": "合计",
                "类别": "",
                "条目": len(df),
                "多空对称": int((df["sym"] == "yes").sum()),
                "只做一边（已给镜像）": int((df["sym"] == "mirror").sum()),
            }
        ]
    )
    out["category_counts"] = df_to_md(pd.concat([cat, total], ignore_index=True))

    fam = df.groupby(["category", "family"]).size().rename("条目").reset_index()
    fam["类别"] = fam["category"].map(CATEGORIES)
    fam = fam.sort_values(["category", "条目"], ascending=[True, False])
    out["family_counts"] = df_to_md(fam[["类别", "family", "条目"]])
    bo, _ = most_common_family(rows, "breakout")
    pb, _ = most_common_family(rows, "pullback")
    out["most_common"] = (
        f"最常见的趋势突破 family：`{bo}`；最常见的顺势回调 family：`{pb}`"
        "（按上表条目数，代码判定）。"
    )

    plat = (
        df.groupby("platform")
        .size()
        .rename("条目")
        .reset_index()
        .sort_values("条目", ascending=False)
    )
    rd = df.groupby("read").size().rename("条目").reset_index()
    out["platform_counts"] = (
        df_to_md(plat) + "\n\n读到的内容（pine/python/sql = 读了源码；desc = 作者描述；"
        "secondary = 二手摘要）：\n\n" + df_to_md(rd)
    )

    td = Counter(t for r in rows for t in r["trend_def"])
    td_by_cat = {
        c: Counter(t for r in rows if r["category"] == c for t in r["trend_def"])
        for c in CATEGORIES
    }
    trow = []
    for t, n in td.most_common():
        row = {"趋势判断方式": TREND_DEFS[t], "trend_def": t, "使用条目": n}
        row.update({CATEGORIES[c]: td_by_cat[c].get(t, 0) for c in CATEGORIES})
        trow.append(row)
    out["trend_def_counts"] = df_to_md(pd.DataFrame(trow))
    n_multi = sum(1 for r in rows if len([t for t in r["trend_def"] if t != "none"]) >= 2)
    out["trend_def_counts"] += f"\n\n同时用两种及以上趋势判断方式的条目：{n_multi} / {len(rows)}。"

    sit = Counter(s for r in rows for s in r["situation"])
    srow = []
    for s in SITUATIONS:
        cats = Counter(r["category"] for r in rows if s in r["situation"])
        srow.append(
            {
                "负责人情形": SITUATIONS[s],
                "条目": sit.get(s, 0),
                "按类别": "，".join(f"{CATEGORIES[c]} {n}" for c, n in cats.most_common()),
            }
        )
    out["situation_counts"] = df_to_md(pd.DataFrame(srow))

    pop = df[df["pop_unit"] == "tv_likes"].sort_values("popularity", ascending=False)
    out["tv_top"] = df_to_md(
        pop[["name", "author", "kind", "popularity", "category", "family"]].rename(
            columns={"popularity": "点赞（2026-10-23）"}
        )
    )
    k = pop["kind"].value_counts()
    out["tv_top"] += (
        f"\n\nTradingView {len(pop)} 条中指标脚本 {int(k.get('indicator', 0))} 条"
        f"（只画信号、无离场与成本）、策略脚本 {int(k.get('strategy', 0))} 条；"
        f"点赞最高的 10 条里指标 {int((pop.head(10)['kind'] == 'indicator').sum())} 条。"
    )

    srow = [
        {
            "截图声称笔数 n": n,
            "全部盈利时胜率 95% 单侧下界（Clopper–Pearson）": round(win_rate_lower(n), 3),
        }
        for n in (5, 6, 7)
    ]
    out["screenshot_ci"] = df_to_md(pd.DataFrame(srow))

    src = df.copy()
    src["热度"] = [
        f"{int(p)} {POP_UNITS[u]}" if u != "none" and pd.notna(p) else POP_UNITS["none"]
        for p, u in zip(src["popularity"], src["pop_unit"], strict=True)
    ]
    src["类别"] = src["category"].map(CATEGORIES)
    src["名称"] = [f"[{n}]({u})" for n, u in zip(src["name"], src["url"], strict=True)]
    src["趋势定义"] = src["trend_text"]
    src["镜像"] = [
        m if s == "mirror" else "原策略对称" for s, m in zip(src["sym"], src["mirror"], strict=True)
    ]
    src = src.rename(
        columns={
            "platform": "平台",
            "read": "读取",
            "bar": "周期",
            "indicators": "指标与参数",
            "entry": "入场",
            "exit": "离场 / 止损",
            "filters": "过滤",
            "claimed": "宣称表现与检验方式",
            "risks": "过拟合 / 前视风险",
            "family": "子类",
        }
    )
    cols = [
        "id",
        "平台",
        "名称",
        "热度",
        "读取",
        "周期",
        "指标与参数",
        "趋势定义",
        "入场",
        "离场 / 止损",
        "过滤",
        "宣称表现与检验方式",
        "过拟合 / 前视风险",
        "类别",
        "子类",
        "镜像",
    ]
    out["source_table"] = df_to_md(src[cols])

    rr = []
    skip = ("feature", "op", "value", "directional")
    for n, r in rules.items():
        conds = "；".join(
            f"{c['feature']}({', '.join(f'{k}={v}' for k, v in c.items() if k not in skip)})"
            f" {c['op']} {c['value']}"
            for c in r["entry"]["all"]
        )
        rr.append(
            {
                "文件": f"config/intrabar/{n}.yaml",
                "类别": CATEGORIES[r["category"]],
                "family": r["family"],
                "oos_pick": r["oos_pick"],
                "负责人情形": SITUATIONS.get(r.get("owner_situation", "none"), ""),
                "入场条件（全部方向化）": conds,
                "需新增特征": "、".join(f["name"] for f in r["features_needed"]),
            }
        )
    out["rule_table"] = df_to_md(pd.DataFrame(rr))
    return out


def render(report: Path, blocks: dict[str, str]) -> str:
    text = report.read_text(encoding="utf-8")
    for k, v in blocks.items():
        a, b = f"<!-- AUTO:{k} -->", f"<!-- /AUTO:{k} -->"
        if a not in text:
            continue
        i, j = text.index(a) + len(a), text.index(b)
        text = text[:i] + "\n" + v + "\n" + text[j:]
    return text


def main() -> None:
    rows = load_sources()
    rules = load_rules()
    check_rules(rules, rows)
    REPORT.write_text(render(REPORT, tables(rows, rules)), encoding="utf-8")
    print(f"{len(rows)} 条来源，{len(rules)} 条候选规则 → {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
