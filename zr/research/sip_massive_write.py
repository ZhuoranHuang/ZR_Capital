# ruff: noqa: E501  （中文报告行较长）
"""R-057 Massive 版正式报告：docs/research/R057_SIP_ORB.md（第一句给结论）与案例图 docs/research/r057_cases/。
表中数字全部由 zr.research.sip_massive_report 算出。运行：python -m zr.research.sip_massive_write
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from zr.ops.markdown import df_to_md
from zr.research import sip_massive_report as mr
from zr.research.sip_orb_report import PAPER_TAB, REPO, r_multiples
from zr.research.sip_orb_write import _pct, case_charts
from zr.research.timing.stats import fmt_ci, pooled

PAPER_RELVOL = "原文图 4：RelVol < 100% −0.02R、> 100% +0.08R、> 30 倍（3,000%）+0.38R"


def _u(x, nd: int = 1) -> str:
    """不带正负号的百分比（波动、胜率）。"""
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def fmt_be(B: pd.DataFrame) -> pd.DataFrame:
    def nan(v) -> bool:
        return v is None or (isinstance(v, float) and np.isnan(v))

    out = B.copy()
    out["零滑点每笔盈亏（美元）"] = out["零滑点每笔盈亏（美元）"].map(
        lambda v: "—" if nan(v) else f"{v:+,.2f}"
    )
    out["盈亏平衡滑点（美元 / 边）"] = out["盈亏平衡滑点（美元 / 边）"].map(
        lambda v: "—" if nan(v) else ("> 0.10" if np.isinf(v) else f"{v:.4f}")
    )
    out["占 R 中位数"] = out["占 R 中位数"].map(
        lambda v: "—" if nan(v) or np.isinf(v) else f"{v:.3f}"
    )
    if "R 中位数（美元 / 股）" in out:
        out["R 中位数（美元 / 股）"] = out["R 中位数（美元 / 股）"].map(
            lambda v: "—" if nan(v) else f"{v:.4f}"
        )
    return out


def fmt_decomp(D: pd.DataFrame) -> pd.DataFrame:
    out = D.copy()
    for c in ("每笔毛收益 R（含滑点）", "每笔净 R"):
        out[c] = out[c].map(lambda v: f"{v:+.3f}")
    out["每笔佣金 R"] = out["每笔佣金 R"].map(lambda v: f"{v:.3f}")
    out["夏普（固定权益）"] = out["夏普（固定权益）"].map(lambda v: f"{v:+.2f}")
    return out


def relvol_summary(tr: pd.DataFrame) -> dict:
    bt = r_multiples(tr)
    cut = pd.cut(bt["relvol"], [1, 2, 5, 10, 30, 1e12], right=False)
    ladder = bt.groupby(cut, observed=True)["r_mult"].mean().to_numpy()
    return {
        "lt1": float(bt.loc[bt["relvol"] < 1, "r_mult"].mean()),
        "ge1": float(bt.loc[bt["relvol"] >= 1, "r_mult"].mean()),
        "ge30": float(bt.loc[bt["relvol"] >= 30, "r_mult"].mean()),
        "n30": int((bt["relvol"] >= 30).sum()),
        "monotone": bool(len(ladder) > 1 and np.all(np.diff(ladder) > 0)),
        "all": float(bt["r_mult"].mean()),
    }


def placebo_table(PP: dict) -> pd.DataFrame:
    rows = []
    for sg in mr.SEGMENTS:
        for k, nm in (
            ("i", "(i) 合格 ∩ RelVol ≥ 1 中随机取 20 只"),
            ("ii", "(ii) 选股不变、随机方向"),
        ):
            act, sims = PP[sg][k]
            rows.append(
                {
                    "段": sg,
                    "安慰剂": nm,
                    "实际日收益均值": f"{act * 100:+.4f}%",
                    "安慰剂均值": f"{sims.mean() * 100:+.4f}%",
                    "95% 分位": f"{np.percentile(sims, 95) * 100:+.4f}%",
                    "实际所在分位": f"{(sims < act).mean():.1%}",
                    "超过 95% 分位": "是" if act > np.percentile(sims, 95) else "否",
                }
            )
    return pd.DataFrame(rows)


def criteria(X: dict, P: dict) -> dict:
    o = X["样本外"]
    m = o[mr.MAIN_NAME]
    D = m["p"]["daily"].set_index("date")
    yr = {}
    for y in (2024, 2025, 2026):
        r = D[[d.year == y for d in D.index]]["ret"]
        yr[y] = float(np.prod(1 + r) - 1) if len(r) else np.nan
    mm = m["m"]
    return {
        "日收益均值 > 0 且按日 t ≥ 2.6（样本外全部交易日）": bool(
            mm["mean"] > 0 and mm["t"] >= 2.6
        ),
        "夏普 > 1": bool(mm["sharpe"] > 1),
        "2024、2025、2026 三个自然年各自为正": bool(
            all(np.isfinite(v) and v > 0 for v in yr.values())
        ),
        "每边滑点 0.02 美元时日收益均值仍 > 0": bool(o["敏感性：滑点 0.02"]["m"]["mean"] > 0),
        "超过安慰剂 (i) 95% 分位": bool(
            P["样本外"]["i"][0] > np.percentile(P["样本外"]["i"][1], 95)
        ),
        "超过安慰剂 (ii) 95% 分位": bool(
            P["样本外"]["ii"][0] > np.percentile(P["样本外"]["ii"][1], 95)
        ),
        "_years": yr,
    }


def mrow(name: str, segname: str, days: list, m: dict) -> dict:
    return {
        "组合": name,
        "样本": f"{segname} {days[0]}～{days[-1]}",
        "总收益": _pct(m["total"], 0),
        "年化": _pct(m["irr"]),
        "波动": _u(m["vol"]),
        "夏普": f"{m['sharpe']:.2f}",
        "日胜率": _u(m["hit_day"]),
        "最大回撤": _pct(m["mdd"]),
        "最差单日": _pct(m["worst"], 2),
        "alpha（年化）": _pct(m["alpha"]),
        "beta": f"{m['beta']:.2f}",
        "笔数": f"{m['trades']:,}",
    }


def paper_rows() -> list[dict]:
    rows = []
    for name, v in PAPER_TAB.items():
        rows.append(
            {
                "组合": name,
                "样本": "原文 2016-01-01～2023-12-31",
                "总收益": _pct(v["total"], 0),
                "年化": _pct(v["irr"]),
                "波动": _u(v["vol"]),
                "夏普": f"{v['sharpe']:.2f}",
                "日胜率": _u(v["hit"]),
                "最大回撤": _pct(-v["mdd"]),
                "最差单日": _pct(v["worst"], 2),
                "alpha（年化）": _pct(v["alpha"]),
                "beta": f"{v['beta']:.2f}",
                "笔数": "—",
            }
        )
    return rows


def relvol_table(tr: pd.DataFrame) -> pd.DataFrame:
    bt = r_multiples(tr)
    bt["rv_b"] = pd.cut(
        bt["relvol"],
        [0, 1, 2, 5, 10, 30, 1e12],
        right=False,
        labels=["< 1", "1–2", "2–5", "5–10", "10–30", "≥ 30"],
    )
    rows = []
    groups = [(str(lv), g) for lv, g in bt.groupby("rv_b", observed=True)]
    groups += [
        ("< 1（合计）", bt[bt["relvol"] < 1]),
        ("≥ 1（合计）", bt[bt["relvol"] >= 1]),
        ("≥ 30（合计）", bt[bt["relvol"] >= 30]),
        ("RelVol 不足 14 日（空）", bt[bt["relvol"].isna()]),
    ]
    for lab, g in groups:
        if not len(g):
            continue
        s = pooled(g["r_mult"].to_numpy(), g["date"].to_numpy(), n_boot=500)
        rows.append(
            {
                "RelVol": lab,
                "笔数": s["n"],
                "每笔平均 R（按日聚类 95%）": fmt_ci(s, 3).replace("%", ""),
                "每笔占股价": f"{g['pct'].mean():+.3f}%",
            }
        )
    return pd.DataFrame(rows)


def composition(R: dict) -> dict:
    r = R["r"]
    C = R["C"]
    cs = pd.read_parquet(r / "ref" / "cs_tickers.parquet")
    active = set(cs.loc[cs["active"] == True, "ticker"])  # noqa: E712
    inactive = set(cs.loc[cs["active"] == False, "ticker"]) - active  # noqa: E712
    known = active | inactive

    def split3(syms: set) -> tuple[int, int, int, int]:
        return len(syms), len(syms & active), len(syms & inactive), len(syms - known)

    t50 = C[C["rank"] > 0]
    t20 = t50[t50["rank"] <= 20]
    per_day = C.groupby("date").agg(
        pool=("symbol", "size"),
        eligible=("eligible", "sum"),
        base=("base", "sum"),
        top=("rank", lambda x: int((x > 0).sum())),
    )
    chk = (
        json.loads((r / "derived" / "pool_check.json").read_text(encoding="utf-8"))
        if (r / "derived" / "pool_check.json").exists()
        else {}
    )
    # 周快照里「只在该周出现、前后两周都不在」的代码（reference 的 type 字段噪声）
    snaps = sorted((r / "ref" / "pit").glob("*.parquet"))
    sets = [set(pd.read_parquet(p)["ticker"].dropna()) for p in snaps]
    dates = [pd.Timestamp(p.stem).date() for p in snaps]
    oneoff = []
    for k in range(1, len(snaps) - 1):
        ex = sets[k] - (sets[k - 1] | sets[k + 1])
        if len(ex) >= 50:
            x = C[(C["date"] >= dates[k]) & (C["date"] < dates[k + 1]) & C["symbol"].isin(ex)]
            oneoff.append(
                {
                    "week": str(dates[k]),
                    "n": len(ex),
                    "eligible_rows": int(x["eligible"].sum()),
                    "top50_rows": int((x["rank"] > 0).sum()),
                    "top20_rows": int(((x["rank"] > 0) & (x["rank"] <= 20)).sum()),
                }
            )
    return {
        "pool": split3(set(C["symbol"])),
        "top50": split3(set(t50["symbol"])),
        "top20": split3(set(t20["symbol"])),
        "top20_rows": len(t20),
        "top20_rows_unknown": int((~t20["symbol"].isin(known)).sum()),
        "per_day_median": per_day.median().round(0).to_dict(),
        "days_top20_full": int((per_day["top"] >= 20).sum()),
        "days_short_after_start": int(
            ((per_day["top"] < 20) & (per_day.index >= R["start"])).sum()
        ),
        "days": len(per_day),
        "pool_check": chk,
        "oneoff": oneoff,
        "splits": int(pd.read_parquet(r / "ref" / "splits.parquet").shape[0]),
    }


def write(R: dict, X: dict, P: dict, B: dict, G: dict | None = None) -> str:
    crit = criteria(X, P)
    npass = sum(1 for k, v in crit.items() if not k.startswith("_") and v)
    passed = npass == 6
    mo = X["样本外"][mr.MAIN_NAME]["m"]
    sd = R["seg_days"]
    ip = X["样本内"][mr.PAPER_SIP]["m"]
    comp = composition(R)
    op = X["样本外"][mr.PAPER_SIP]["m"]
    first = (
        f"**{'复现成立' if passed else '复现不成立'}**（发表后样本外 {sd['样本外'][0]}～{sd['样本外'][-1]}，{len(sd['样本外'])} 个交易日）：主口径（保守算法、D-013、每边 0.01 美元、10 万美元）日收益均值 {mo['mean'] * 100:+.3f}%（t {mo['t']:+.2f}），"
        f"夏普 {mo['sharpe']:.2f}，{npass} / 6 项通过标准满足。同一段按原文口径（乐观算法、每股 0.0035 美元、零滑点、2.5 万美元）日收益均值 {op['mean'] * 100:+.3f}%（t {op['t']:+.2f}），夏普 {op['sharpe']:.2f}。"
        f"论文样本内（{sd['样本内'][0]}～{sd['样本内'][-1]}，只覆盖原文 8 年中的约 {len(sd['样本内']) / 252:.1f} 年）原文口径 SIP 前 20 夏普 {ip['sharpe']:.2f}（原文全期 2.81，不可直接比）。"
    )
    why = ""
    if G:
        d = G["样本外"]["decomp"].set_index(["算法", "每边滑点", "佣金"])["每笔净 R"]
        e = G["样本外"]["entry"]
        a0 = d[("乐观", "0.00", "每股 0.0035")]
        a1 = d[("保守", "0.01", "D-013")]
        if a0 > 0 and a1 < 0:
            lead = "原文自己的假设下每笔为正，换成长桥佣金、加上滑点、改用保守算法后转负"
        elif a0 > 0:
            lead = "原文自己的假设下每笔为正，换成主口径后仍不为负"
        else:
            lead = "原文自己的假设下每笔已不为正"
        why = (
            f"{lead}（样本外、SIP 前 20、固定权益，每笔净 R）：原文口径（乐观、零滑点、每股 0.0035）{a0:+.3f}R → 换成长桥 D-013 佣金 {d[('乐观', '0.00', 'D-013')]:+.3f}R → 再加每边 0.01 美元滑点 {d[('乐观', '0.01', 'D-013')]:+.3f}R → 再换保守算法（主口径）{d[('保守', '0.01', 'D-013')]:+.3f}R。"
            f"保守算法下 {e['entry_bar_stop']:.0%} 的交易在入场当根就被判止损（其中 {e['entry_bar_stop_certain']:.0%} 可以确定是入场后止损——开盘已在区间之外或当根收盘已越过止损价；其余 {e['entry_bar_stop_ambiguous']:.0%} 同根先后无法用 1 分钟 K 分辨）。"
        )
    L = [
        "# R-057：Zarattini、Barbon、Aziz（2024）SIP 开盘区间突破的发表后样本外复现（Massive 全市场数据）",
        "",
        first,
        "",
        *([why, ""] if why else []),
        "`python -m zr.research.sip_massive_write`（数据：`zr.data.massive_archive`、`zr.data.sip_stream`、`zr.research.sip_select`）。预登记 RESEARCH_LOG R-057（运行前提交，含 2026-11-14 的 Massive 修订与补充）。规则与参数全部取自原文，不调参。",
        "",
        "## 数据与标的池",
        "",
        "- 数据源：Massive（原 Polygon.io）flat files 美股 SIP 1 分钟聚合与日聚合（不复权），订阅可下载起点 2021-11-15（更早返回 403）。核对见 `docs/research/R057_MASSIVE_DATA.md`。",
        f"- 样本：首个能选股的交易日 {R['start']}（前 14 个交易日为合格与 RelVol 预热）～{R['last']}，{len(R['days'])} 个交易日；样本内 {len(sd['样本内'])} 天，样本外 {len(sd['样本外'])} 天。原文起点 2016-01-01，本样本起点晚 {(R['start'] - mr.PAPER_START).days / 365.25:.1f} 年。",
        f"- 标的池（每周首个交易日的 reference 时点快照里的 XNYS / XNAS 普通股 ∩ 当日有成交）：样本期内出现过 {comp['pool'][0]:,} 只——当前仍在市 {comp['pool'][1]:,} 只、已退市 {comp['pool'][2]:,} 只、不在当前普通股代码表 {comp['pool'][3]:,} 只（改过代码，或 reference 当时与现在给的类型不同）。"
        f"每日中位数：池内且有开盘区间 {comp['per_day_median'].get('pool', 0):,.0f} 只、合格 {comp['per_day_median'].get('eligible', 0):,.0f} 只、基础版候选 {comp['per_day_median'].get('base', 0):,.0f} 只、RelVol 前 50 实际 {comp['per_day_median'].get('top', 0):,.0f} 只；前 20 满 20 只的天数 {comp['days_top20_full']} / {comp['days']}（首个可选股日之后不足 20 只的有 {comp['days_short_after_start']} 天，其余是预热期）。",
        f"- 进入过前 50 的 {comp['top50'][0]:,} 只（在市 {comp['top50'][1]:,}、已退市 {comp['top50'][2]:,}、不在代码表 {comp['top50'][3]:,}）；进入过前 20 的 {comp['top20'][0]:,} 只（在市 {comp['top20'][1]:,}、已退市 {comp['top20'][2]:,}、不在代码表 {comp['top20'][3]:,}）。前 20 的 {comp['top20_rows']:,} 个标的日中，{comp['top20_rows_unknown']:,} 个（{comp['top20_rows_unknown'] / comp['top20_rows']:.2%}）来自不在当前普通股代码表的代码（其中既有后来改了代码或被并购的普通股，如 SATS、SIX、FB，也有快照当时被标成普通股的 ADR 或基金，如 YMM、HSAI、DXYZ——按代码判断，未逐只核实）。",
        "- reference 的类型字段有噪声："
        + (
            "；".join(
                f"{o['week']} 那周的快照有 {o['n']} 只「只在该周出现」的代码，它们在该周合格 {o['eligible_rows']} 个标的日、进入前 50 的 {o['top50_rows']} 个、进入前 20 的 {o['top20_rows']} 个"
                for o in comp["oneoff"]
            )
            + "（2022-08-22 那周多为 NYSE 封闭式基金，2025-06-30 那周多为优先股与票据——按代码与名称判断）。合格条件把它们几乎全部挡在外面，对结果的影响见上面的个数。"
            if comp["oneoff"]
            else "没有发现「只在某一周出现」的代码数 ≥ 50 的周快照。"
        ),
        f"- 拆股：拆股接口 2021 年以来 {comp['splits']:,} 条，比例超出 ×1.2 / ÷1.2 的按执行日剔除 15 个交易日。",
        "",
    ]
    if comp["pool_check"]:
        rows = [
            {
                "日期": d,
                "逐日快照只数": v["daily_snapshot"],
                "按周快照只数": v["weekly_snapshot"],
                "当日有成交、逐日有按周无": len(v["traded_in_daily_not_weekly"]),
                "当日有成交、按周有逐日无": len(v["traded_in_weekly_not_daily"]),
            }
            for d, v in comp["pool_check"].items()
        ]
        L += ["按周快照与逐日快照的抽查（12 个随机交易日）：", "", df_to_md(pd.DataFrame(rows)), ""]
    # 通过标准
    L += [
        "## 通过标准（样本外，主口径、10 万美元，全部满足才算「复现成立」）",
        "",
        "| 标准 | 结果 |",
        "|---|---|",
    ]
    L += [
        f"| {k} | {'满足' if v else '**不满足**'} |"
        for k, v in crit.items()
        if not k.startswith("_")
    ]
    L += [
        "",
        "自然年收益（主口径，样本外段内复利）："
        + "，".join(f"{y} {_pct(v)}" for y, v in crit["_years"].items())
        + "。",
        "",
    ]
    # 复现检查
    rows = paper_rows()
    for name in (mr.PAPER_BASE, mr.PAPER_SIP):
        rows.append(mrow(name, "样本内", sd["样本内"], X["样本内"][name]["m"]))
    L += [
        "## 复现检查（论文样本内，原文口径：乐观算法、每股 0.0035 美元、零滑点、2.5 万美元、4 倍上限）",
        "",
        f"原文表 1、表 2（SSRN 4729284；2016-01-01～2023-12-31，原文「Hit Ratio」按日）。**本样本内只有 {sd['样本内'][0]}～{sd['样本内'][-1]}（{len(sd['样本内'])} 个交易日，约为原文 8 年中的 {len(sd['样本内']) / 252:.1f} 年），起点比原文晚 {(R['start'] - mr.PAPER_START).days / 365.25:.1f} 年；原文没有分年数字，两者只能并列、不能直接比。**",
        "基础版 N = 当日合格且非十字星只数（09:35 已知）；SIP 前 20 的 N = 20。alpha / beta = 日收益对 SPY 日收益（Massive 日聚合官方收盘）回归，alpha × 252。",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
        f"按 RelVol 分桶的每笔平均 R（基础版，样本内，原文口径；{PAPER_RELVOL}，原文为扣佣金后）：",
        "",
        df_to_md(relvol_table(X["样本内"][mr.PAPER_BASE]["p"]["trades"])),
        "",
    ]
    pt = r_multiples(X["样本内"][mr.PAPER_SIP]["p"]["trades"])
    ps = pooled(pt["r_mult"].to_numpy(), pt["date"].to_numpy())
    pb = X["样本内"][mr.PAPER_BASE]["m"]
    rb = relvol_summary(X["样本内"][mr.PAPER_BASE]["p"]["trades"])
    L += [
        "**对得上与对不上的**：",
        "",
        f"- 对得上：SIP 前 20 的年化波动 {_u(ip['vol'])}（原文 14.8%）、beta {ip['beta']:.2f}（原文 0.00）；每笔平均 {fmt_ci(ps, 3).replace('%', '')} R（原文没有给前 20 的每笔 R）。",
        f"- RelVol 分桶（基础版、原文口径，每笔平均 R）：< 1 为 {rb['lt1']:+.3f}（原文 −0.02）、≥ 1 为 {rb['ge1']:+.3f}（原文 +0.08）、≥ 30 为 {rb['ge30']:+.3f}（原文 +0.38；本样本只有 {rb['n30']} 笔，区间见上表）。"
        f"{'< 1 为负、≥ 1 为正，符号与原文一致' if rb['lt1'] < 0 < rb['ge1'] else '< 1 与 ≥ 1 两档的符号与原文不一致'}；"
        f"{'≥ 30 一档远低于原文' if rb['ge30'] < 0.19 else '≥ 30 一档与原文同量级'}；"
        f"{'RelVol ≥ 1 的各档随 RelVol 升高而升高' if rb['monotone'] else 'RelVol ≥ 1 的各档没有随 RelVol 单调升高'}。",
        f"- 对不上：SIP 前 20 夏普 {ip['sharpe']:.2f}（原文 2.81）、年化 {_pct(ip['irr'])}（原文 +41.6%）、日胜率 {_u(ip['hit_day'])}（原文 48.4%）；基础版夏普 {pb['sharpe']:.2f}（原文 0.48）。任务书的排查门槛是「SIP 版夏普 < 1」，{'本样本内低于门槛' if ip['sharpe'] < 1 else '本样本内没有低于门槛，但差距大，仍做了下面的核查'}。",
        "",
        "**已做的核查（数据与实现）**：",
        "",
    ]
    r_ = R["r"] / "derived"
    if (r_ / "independent_check.json").exists():
        ck = json.loads((r_ / "independent_check.json").read_text(encoding="utf-8"))
        ok = sum(
            1
            for v in ck.values()
            if v["top20_same_order"]
            and v["eligible_agree"] == 1.0
            and v["dir_agree"] == 1.0
            and (v["relvol_max_abs_diff"] or 0) < 1e-9
        )
        L.append(
            f"- 选股层：随机 {len(ck)} 个交易日（{min(ck)}～{max(ck)}），直接从归档的原始日文件 / 分钟文件用另写的逐行实现重算合格、开盘区间、RelVol、方向与前 20 名，{ok} / {len(ck)} 天与流水线完全一致（`zr.research.sip_massive_check`）。"
        )
    if (r_ / "independent_trades_check.json").exists():
        ck = json.loads((r_ / "independent_trades_check.json").read_text(encoding="utf-8"))
        L.append(
            "- 执行层：随机抽样的成交用另写的逐根循环重算入场价与出场价，"
            + "；".join(
                f"{k} 设定 {v['checked']} 笔中不一致 {v['mismatch']} 笔" for k, v in ck.items()
            )
            + "。"
        )
    L += [
        "- 数据层：日聚合与长桥日线同口径、分钟 K 时间戳对齐、不复权、退市股在（`docs/research/R057_MASSIVE_DATA.md`）；选股与合格特征通过截断测试（`tests/test_sip_select.py`）。",
        "- 规则层：逐条对照原文 2.1 节与第 4 节（开盘 > 5 美元、前 14 日均量 ≥ 100 万股、ATR > 0.50、首根方向、区间高 / 低突破入场、止损距成交价 10% ATR、16:00 平仓、每仓止损亏 1%、4 倍上限、每股 0.0035 美元），未发现不一致。原文没有写日内数据粒度、同根处理与基础版的资金分配方式。",
        "- 没有发现数据或实现错误。**未能解释夏普差距**；可能的原因（均未验证）：(1) 时段不同——原文 8 年合并，没有分年数字，无法知道原文在 2022–2023 的表现；(2) 数据源不同（原文 IQFeed，本文 SIP 聚合）；(3) 原文未写明的实现细节（ATR 是否 Wilder 平滑、同根先后、代码变更与拆股处理）。",
        "",
    ]
    # 全部组合两段并列
    rows = []
    for name in X["样本外"]:
        for sg in mr.SEGMENTS:
            rows.append(mrow(name, sg, sd[sg], X[sg][name]["m"]))
    L += [
        "## 全部组合：样本外与同口径样本内并列（敏感性只报告，不用于挑选）",
        "",
        "各段从初始资金起复利；日胜率 = 有交易日中收益 > 0 的占比。",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
    ]
    # 主口径统计
    L += ["## 主口径统计", ""]
    for sg in mr.SEGMENTS:
        m = X[sg][mr.MAIN_NAME]["m"]
        tr = r_multiples(X[sg][mr.MAIN_NAME]["p"]["trades"])
        sR = pooled(tr["r_mult"].to_numpy(), tr["date"].to_numpy())
        sP = pooled(tr["pct"].to_numpy(), tr["date"].to_numpy())
        L.append(
            f"- {sg}：日收益均值 {m['mean'] * 100:+.4f}%（按日整块重抽样 95% 区间 {m['ci'][0] * 100:+.4f}%～{m['ci'][1] * 100:+.4f}%），t {m['t']:+.2f}，{m['days']} 个交易日（有交易 {m['traded_days']} 天），{m['trades']:,} 笔；"
            f"每笔平均 {fmt_ci(sR, 3).replace('%', '')} R（扣费后，R = 0.10 × ATR，按日聚类），每笔占股价 {fmt_ci(sP, 3)}。"
        )
    L.append("")
    if G:
        fx = {
            sg: G[sg]["decomp"]
            .set_index(["算法", "每边滑点", "佣金"])
            .loc[("保守", "0.01", "D-013")]
            for sg in mr.SEGMENTS
        }
        ee = {sg: X[sg][mr.MAIN_NAME]["m"]["end_equity"] for sg in mr.SEGMENTS}
        L += [
            f"**读这些数字要注意复利机制**：主口径按日复利，权益从 10 万美元跌到样本内期末 {ee['样本内']:,.0f} 美元、样本外期末 {ee['样本外']:,.0f} 美元；权益很低时每仓只有几股，D-013 的每单最低收费占比越来越大，"
            f"上面的每笔 R、日收益均值与下表靠后的年份都被这一机制放大（样本外 2026 年尤其明显）。不受它影响的固定权益口径（每天都按 10 万美元定仓，不复利）："
            f"样本内每笔 {fx['样本内']['每笔净 R']:+.3f}R、日收益均值 {fx['样本内']['日收益均值（固定权益）']}；样本外每笔 {fx['样本外']['每笔净 R']:+.3f}R、日收益均值 {fx['样本外']['日收益均值（固定权益）']}（见补充诊断）。通过标准按预登记的复利口径判定"
            + (
                "；固定权益口径下日收益均值同样为负。"
                if str(fx["样本外"]["日收益均值（固定权益）"]).startswith("-")
                else "；固定权益口径下样本外日收益均值不为负，与复利口径不同。"
            ),
            "",
        ]
    rows = []
    allX = {sg: X[sg][mr.MAIN_NAME] for sg in mr.SEGMENTS}
    trs = pd.concat([r_multiples(allX[sg]["p"]["trades"]) for sg in mr.SEGMENTS], ignore_index=True)
    dly = (
        pd.concat([allX[sg]["p"]["daily"] for sg in mr.SEGMENTS], ignore_index=True)
        .set_index("date")["ret"]
        .reindex(R["days"])
        .fillna(0)
    )
    for name, a, b in mr.year_periods(R["start"], R["last"]):
        x = trs[(trs["date"] >= a) & (trs["date"] <= b)]
        dd = dly[[a <= d <= b for d in dly.index]]
        if not len(dd):
            continue
        s = pooled(x["r_mult"].to_numpy(), x["date"].to_numpy(), n_boot=1000)
        rows.append(
            {
                "时段": name,
                "交易日": len(dd),
                "累计收益": _pct(float(np.prod(1 + dd) - 1)),
                "日均": f"{dd.mean() * 100:+.4f}%",
                "笔数": s["n"],
                "每笔 R（聚类 95%）": fmt_ci(s, 3).replace("%", ""),
            }
        )
    for sg in mr.SEGMENTS:
        t = r_multiples(allX[sg]["p"]["trades"])
        for nm, sgn in (("做多", 1), ("做空", -1)):
            x = t[t["dir"] == sgn]
            s = pooled(x["r_mult"].to_numpy(), x["date"].to_numpy(), n_boot=1000)
            rows.append(
                {
                    "时段": f"{sg} {nm}",
                    "交易日": "",
                    "累计收益": "",
                    "日均": "",
                    "笔数": s["n"],
                    "每笔 R（聚类 95%）": fmt_ci(s, 3).replace("%", ""),
                }
            )
    L += [
        "分时段与多空（主口径；2021、2022、2023 属样本内，各年累计收益按主口径在各自段内复利的日收益连乘）：",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
    ]
    # RelVol 分桶：样本外用原文口径的基础版（样本内的表在「复现检查」）
    nb = pd.Series(R["nbase"])
    nb_med = float(nb.median())
    mb = {sg: relvol_summary(X[sg][mr.MAIN_BASE]["p"]["trades"])["all"] for sg in mr.SEGMENTS}
    L += [
        f"按 RelVol 分桶的每笔平均 R（基础版，样本外，原文口径；{PAPER_RELVOL}，原文为扣佣金后）：",
        "",
        df_to_md(relvol_table(X["样本外"][mr.PAPER_BASE]["p"]["trades"])),
        "",
        f"基础版在主口径设定下没有可比性，分桶不列：每日候选中位数 {nb_med:,.0f} 只，10 万美元时每仓风险约 {100_000 / nb_med * 0.01:.2f} 美元，D-013 每单有最低收费，"
        f"有成交的交易每笔平均 样本内 {mb['样本内']:+.2f}R、样本外 {mb['样本外']:+.2f}R，几乎全是佣金（组合层面的数字见「全部组合」表）。",
        "",
    ]
    # 盈亏平衡滑点
    L += [
        "## 盈亏平衡滑点（本报告最重要的数）",
        "",
        "保守算法、D-013、股数固定为主口径实际股数；每边滑点 0～0.10 美元（步长 0.005）重算入场、止损与平仓成交价，线性插值求每笔平均盈亏归零的滑点。0 = 零滑点已不赚钱；「> 0.10」= 0.10 美元仍为正。",
        "",
    ]
    for sg in mr.SEGMENTS:
        L += [f"**{sg}**", "", df_to_md(fmt_be(B[sg])), ""]
    # 安慰剂
    L += [
        "## 安慰剂（各 1,000 次，1 分钟执行，保守、D-013、每边 0.01，固定权益 10 万美元的日收益，不复利；判定看样本外）",
        "",
        df_to_md(placebo_table(P)),
        "",
        "(ii) 的说明：反方向要先反向穿过整个开盘区间才触发，触发笔数少（"
        + "；".join(
            f"{sg} 原方向 {P[sg]['n0']:,} 笔、反方向 {P[sg]['n1']:,} 笔" for sg in mr.SEGMENTS
        )
        + "），成本为负的口径下交易少本身就亏得少，这个对照并不对称。",
        "",
    ]
    try:
        cases = case_charts(
            X["样本外"],
            REPO / "docs" / "research" / "r057_cases",
            load1=mr.bars_loader(R["r"]),
            key=mr.MAIN_NAME,
            title="R-057 主口径（样本外）：最近 10 笔盈利、10 笔亏损（1 分钟 K）",
        )
        L += [
            f"案例图：`{__import__('pathlib').Path(cases).relative_to(REPO)}`（主口径样本外最近 10 笔盈利、10 笔亏损）。",
            "",
        ]
    except Exception as e:  # noqa: BLE001
        L += [f"案例图生成失败：{type(e).__name__}: {e}", ""]
    if G:
        L += [
            "## 补充诊断（非预登记，只为解释主口径与原文口径的差距，不用于判定、不用于挑选）",
            "",
            "SIP 前 20，固定权益 10 万美元（不复利，避免权益变化改变股数与最低佣金的占比）。每笔毛收益 = 方向 ×（出场价 − 入场价）÷（0.10 × ATR），已含滑点；佣金按同一 R 折算。",
            "",
        ]
        for sg in mr.SEGMENTS:
            e = G[sg]["entry"]
            L += [
                f"**{sg}**：",
                "",
                df_to_md(fmt_decomp(G[sg]["decomp"])),
                "",
                f"保守算法（每边 0.01）下 {e['n']:,} 笔中 {e['stopped']:.1%} 止损出场（乐观算法 {e['stopped_optimistic']:.1%}）；**入场当根即判止损 {e['entry_bar_stop']:.1%}**，拆成三类：(a) 入场那根开盘已在区间之外（成交 = 开盘价，当根极值必然在入场之后）{e['entry_bar_stop_gap']:.1%}；(b) 入场那根收盘已越过止损价（入场后必然触及止损）{e['entry_bar_stop_close_beyond']:.1%}；(c) 当根极值触及止损但收盘仍在止损之内，同根先后无法分辨 {e['entry_bar_stop_ambiguous']:.1%}。(a)(b) 是确定的止损，保守与乐观的差距来自 (c)。入场时刻中位数为美东第 {e['entry_min_median']:.0f} 分钟（09:35 = 575）。",
                "",
                "盈亏平衡滑点（股数固定为固定权益下的股数，全部交易）：",
                "",
                df_to_md(fmt_be(G[sg]["be"])),
                "",
            ]
        L += [
            "同根先后只能用逐笔成交数据分辨；真实结果在保守与乐观之间。当前订阅是否提供逐笔成交的 flat files 未核实。",
            "",
            "原文口径下的同样两种安慰剂（乐观、零滑点、每股 0.0035，固定权益 10 万美元；只看在原文假设下选股与方向本身有没有贡献）：",
            "",
            df_to_md(placebo_table({sg: G[sg]["placebo_paper"] for sg in mr.SEGMENTS})),
            "",
        ]
    L += [
        "## 边界",
        "",
        f"- 标的池：Massive reference 时点快照（按周）里的 XNYS / XNAS 普通股，包含后来退市的股票（样本期内出现过的 {comp['pool'][0]:,} 只中 {comp['pool'][2]:,} 只现已退市）；周内上市的新股下周才进池、周内类型变化忽略；代码变更视为新标的（需重新累计 14 日）。不含 NYSE American、NYSE Arca、Cboe 主上市的股票（原文为「NYSE 与 Nasdaq」）。",
        f"- 样本：2021-11-15 起（订阅限制）；论文样本内只有约 {len(sd['样本内']) / 252:.1f} 年，复现检查不能与原文全期数字直接比。",
        "- 合格条件用 Massive 日聚合（与原预登记所用长桥日线同口径）；开盘区间、RelVol、执行用 SIP 1 分钟 K（只含有成交的分钟；聚合包含哪些成交条件按 Massive 的规则，未逐条核对）。",
        "- 1 分钟 K 的同根歧义：入场那根内先入场后止损无法分辨，保守 / 乐观两种算法都报；判定以保守为准。",
        "- 做空未计借券费，也未核实能否借到；开盘时段的真实买卖价差未知（只用固定每边滑点）；长桥账户能否做 4 倍日内杠杆未核实。",
        "- 基础版 N = 当日合格且非十字星只数（09:35 已知）；原文基础版的 N 未写明。",
        "- 本检验不计入 MU / QQQ / SPY 留出期额度；之后若在这套数据上调参，2026-06 以后的数据须重新封存。",
        "",
    ]
    out = REPO / "docs" / "research" / "R057_SIP_ORB.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)


def summary(R: dict, X: dict, P: dict, B: dict, G: dict) -> dict:
    """关键数字（写入 derived/r057_summary.json，供 RESEARCH_LOG / STATUS 引用）。"""
    crit = criteria(X, P)

    def mm(sg: str, name: str) -> dict:
        m = X[sg][name]["m"]
        return {
            k: (float(m[k]) if k != "trades" and k != "days" else int(m[k]))
            for k in (
                "days",
                "trades",
                "mean",
                "t",
                "sharpe",
                "total",
                "irr",
                "vol",
                "mdd",
                "hit_day",
            )
        }

    out = {
        "start": str(R["start"]),
        "last": str(R["last"]),
        "days": {sg: len(R["seg_days"][sg]) for sg in mr.SEGMENTS},
        "criteria": {k: v for k, v in crit.items() if not k.startswith("_")},
        "years": {str(k): float(v) for k, v in crit["_years"].items()},
        "passed": int(sum(1 for k, v in crit.items() if not k.startswith("_") and v)),
    }
    for sg in mr.SEGMENTS:
        d = G[sg]["decomp"].set_index(["算法", "每边滑点", "佣金"])["每笔净 R"]
        out[sg] = {
            "main": mm(sg, mr.MAIN_NAME),
            "paper_sip": mm(sg, mr.PAPER_SIP),
            "paper_base": mm(sg, mr.PAPER_BASE),
            "breakeven_main": {
                k: (None if not np.isfinite(v) else float(v))
                for k, v in B[sg]
                .iloc[0][["零滑点每笔盈亏（美元）", "盈亏平衡滑点（美元 / 边）"]]
                .items()
            },
            "breakeven_diag": [
                {
                    k: (v if isinstance(v, str) else (None if not np.isfinite(v) else float(v)))
                    for k, v in r.items()
                }
                for r in G[sg]["be"].to_dict("records")
            ],
            "net_R": {"|".join(k): float(v) for k, v in d.items()},
            "entry": {k: float(v) for k, v in G[sg]["entry"].items()},
            "placebo": {
                k: {
                    "actual": float(P[sg][k][0]),
                    "p95": float(np.percentile(P[sg][k][1], 95)),
                    "pct": float((P[sg][k][1] < P[sg][k][0]).mean()),
                }
                for k in ("i", "ii")
            },
        }
    return out


def main(argv=None) -> int:
    R = mr.run()
    X = mr.experiments(R)
    P = {sg: mr.placebos(R, sg) for sg in mr.SEGMENTS}
    B = {sg: mr.breakeven(R, X[sg][mr.MAIN_NAME]["p"]["trades"]) for sg in mr.SEGMENTS}
    G = {
        sg: {
            "decomp": mr.decomposition(R, sg),
            "entry": mr.entry_bar_stats(R, sg),
            "be": mr.breakeven_diag(R, sg),
            "placebo_paper": mr.placebos(R, sg, algo="opt", slip=0.0, fee="per_share"),
        }
        for sg in mr.SEGMENTS
    }
    print(write(R, X, P, B, G))
    sp = R["r"] / "derived" / "r057_summary.json"
    sp.write_text(
        json.dumps(summary(R, X, P, B, G), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
