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
        "波动": _pct(m["vol"]),
        "夏普": f"{m['sharpe']:.2f}",
        "日胜率": _pct(m["hit_day"]),
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
                "波动": _pct(v["vol"]),
                "夏普": f"{v['sharpe']:.2f}",
                "日胜率": _pct(v["hit"]),
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
    inactive = set(cs.loc[cs["active"] == False, "ticker"])  # noqa: E712
    pool_syms = set(C["symbol"])
    top_syms = set(C.loc[C["rank"] > 0, "symbol"])
    top20_syms = set(C.loc[(C["rank"] > 0) & (C["rank"] <= 20), "symbol"])
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
    return {
        "pool_symbols": len(pool_syms),
        "pool_symbols_now_inactive": len(pool_syms & inactive),
        "top50_symbols": len(top_syms),
        "top50_symbols_now_inactive": len(top_syms & inactive),
        "top20_symbols": len(top20_syms),
        "top20_symbols_now_inactive": len(top20_syms & inactive),
        "per_day_median": per_day.median().round(0).to_dict(),
        "days_top20_full": int((per_day["top"] >= 20).sum()),
        "days": len(per_day),
        "pool_check": chk,
        "splits": int(pd.read_parquet(r / "ref" / "splits.parquet").shape[0]),
    }


def write(R: dict, X: dict, P: dict, B: dict) -> str:
    crit = criteria(X, P)
    npass = sum(1 for k, v in crit.items() if not k.startswith("_") and v)
    passed = npass == 6
    mo = X["样本外"][mr.MAIN_NAME]["m"]
    sd = R["seg_days"]
    ip = X["样本内"][mr.PAPER_SIP]["m"]
    comp = composition(R)
    first = (
        f"**{'复现成立' if passed else '复现不成立'}**（发表后样本外 {sd['样本外'][0]}～{sd['样本外'][-1]}，{len(sd['样本外'])} 个交易日）：主口径（保守算法、D-013、每边 0.01 美元、10 万美元）日收益均值 {mo['mean'] * 100:+.3f}%（t {mo['t']:+.2f}），"
        f"夏普 {mo['sharpe']:.2f}，{npass} / 6 项通过标准满足。论文样本内（{sd['样本内'][0]}～{sd['样本内'][-1]}，只覆盖原文 8 年中的约 2 年）原文口径 SIP 前 20 夏普 {ip['sharpe']:.2f}（原文全期 2.81，不可直接比）。"
    )
    L = [
        "# R-057：Zarattini、Barbon、Aziz（2024）SIP 开盘区间突破的发表后样本外复现（Massive 全市场数据）",
        "",
        first,
        "",
        "`python -m zr.research.sip_massive_write`（数据：`zr.data.massive_archive`、`zr.data.sip_stream`、`zr.research.sip_select`）。预登记 RESEARCH_LOG R-057（运行前提交，含 2026-11-14 的 Massive 修订与补充）。规则与参数全部取自原文，不调参。",
        "",
        "## 数据与标的池",
        "",
        "- 数据源：Massive（原 Polygon.io）flat files 美股 SIP 1 分钟聚合与日聚合（不复权），订阅可下载起点 2021-11-15（更早返回 403）。核对见 `docs/research/R057_MASSIVE_DATA.md`。",
        f"- 样本：首个能选股的交易日 {R['start']}（前 14 个交易日为合格与 RelVol 预热）～{R['last']}，{len(R['days'])} 个交易日；样本内 {len(sd['样本内'])} 天，样本外 {len(sd['样本外'])} 天。原文起点 2016-01-01，本样本起点晚 {(R['start'] - mr.PAPER_START).days / 365.25:.1f} 年。",
        f"- 标的池（每周首个交易日的 reference 时点快照里的 XNYS / XNAS 普通股 ∩ 当日有成交）：样本期内出现过 {comp['pool_symbols']:,} 只，其中 {comp['pool_symbols_now_inactive']:,} 只现已退市（reference active=false）；"
        f"每日中位数：池内且有开盘区间 {comp['per_day_median'].get('pool', 0):,.0f} 只、合格 {comp['per_day_median'].get('eligible', 0):,.0f} 只、基础版候选 {comp['per_day_median'].get('base', 0):,.0f} 只、RelVol 前 50 实际 {comp['per_day_median'].get('top', 0):,.0f} 只。"
        f"进入过前 50 的 {comp['top50_symbols']:,} 只（现已退市 {comp['top50_symbols_now_inactive']:,}）；进入过前 20 的 {comp['top20_symbols']:,} 只（现已退市 {comp['top20_symbols_now_inactive']:,}）。前 20 满 20 只的天数 {comp['days_top20_full']} / {comp['days']}。",
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
    # RelVol 分桶（主口径设定的基础版）
    for sg in mr.SEGMENTS:
        L += [
            f"按 RelVol 分桶的每笔平均 R（基础版，{sg}，主口径设定；{PAPER_RELVOL}）：",
            "",
            df_to_md(relvol_table(X[sg][mr.MAIN_BASE]["p"]["trades"])),
            "",
        ]
    # 盈亏平衡滑点
    L += [
        "## 盈亏平衡滑点（本报告最重要的数）",
        "",
        "保守算法、D-013、股数固定为主口径实际股数；每边滑点 0～0.10 美元（步长 0.005）重算入场、止损与平仓成交价，线性插值求每笔平均盈亏归零的滑点。0 = 零滑点已不赚钱；inf = 0.10 美元仍为正。",
        "",
    ]
    for sg in mr.SEGMENTS:
        Bt = B[sg].copy()
        for c in (
            "零滑点每笔盈亏（美元）",
            "盈亏平衡滑点（美元 / 边）",
            "占 R 中位数",
            "R 中位数（美元 / 股）",
        ):
            Bt[c] = Bt[c].map(
                lambda v: (
                    "—"
                    if v is None or not np.isfinite(v)
                    else (f"{v:.4f}" if abs(v) < 10 else f"{v:,.0f}")
                )
            )
        L += [f"**{sg}**", "", df_to_md(Bt), ""]
    # 安慰剂
    rows = []
    for sg in mr.SEGMENTS:
        for k, nm in (
            ("i", "(i) 合格 ∩ RelVol ≥ 1 中随机取 20 只"),
            ("ii", "(ii) 选股不变、随机方向"),
        ):
            act, sims = P[sg][k]
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
    L += [
        "## 安慰剂（各 1,000 次，1 分钟执行，保守、D-013、每边 0.01，固定权益 10 万美元的日收益，不复利；判定看样本外）",
        "",
        df_to_md(pd.DataFrame(rows)),
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
    L += [
        "## 边界",
        "",
        f"- 标的池：Massive reference 时点快照（按周）里的 XNYS / XNAS 普通股，包含后来退市的股票（样本期内出现过的 {comp['pool_symbols']:,} 只中 {comp['pool_symbols_now_inactive']:,} 只现已退市）；周内上市的新股下周才进池、周内类型变化忽略；代码变更视为新标的（需重新累计 14 日）。不含 NYSE American、NYSE Arca、Cboe 主上市的股票（原文为「NYSE 与 Nasdaq」）。",
        f"- 样本：2021-11-15 起（订阅限制）；论文样本内只有约 {len(sd['样本内']) / 252:.1f} 年，复现检查不能与原文全期数字直接比。",
        "- 合格条件用 Massive 日聚合（与原预登记所用长桥日线同口径）；开盘区间、RelVol、执行用 SIP 1 分钟 K（只含有成交的分钟；零股成交按 SIP 惯例不更新开高低收）。",
        "- 1 分钟 K 的同根歧义：入场那根内先入场后止损无法分辨，保守 / 乐观两种算法都报；判定以保守为准。",
        "- 做空未计借券费，也未核实能否借到；开盘时段的真实买卖价差未知（只用固定每边滑点）；长桥账户能否做 4 倍日内杠杆未核实。",
        "- 基础版 N = 当日合格且非十字星只数（09:35 已知）；原文基础版的 N 未写明。",
        "- 本检验不计入 MU / QQQ / SPY 留出期额度；之后若在这套数据上调参，2026-06 以后的数据须重新封存。",
        "",
    ]
    out = REPO / "docs" / "research" / "R057_SIP_ORB.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return str(out)


def main(argv=None) -> int:
    R = mr.run()
    X = mr.experiments(R)
    P = {sg: mr.placebos(R, sg) for sg in mr.SEGMENTS}
    B = {sg: mr.breakeven(R, X[sg][mr.MAIN_NAME]) for sg in mr.SEGMENTS}
    print(write(R, X, P, B))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
