"""V4 决定性检验报告（R-033）→ docs/research/V4_DECISIVE.md。

读 zr/research/v4_decisive.py 写出的 $ZR_DATA_ROOT/research/v4/{cells,trades,run}_{C,A}.*，
另做 H4（负责人 157 笔按新口径重做状态分类，对比 OWN_POLICY_EXTRACTED.md 旧表）。
运行：uv run python -m zr.research.v4_report
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.ops.markdown import df_to_md
from zr.research import v4_decisive as v4d
from zr.research import v4_rules_test as rt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "V4_DECISIVE.md"
PERIODS = ("2023", "2024", "2025", "2026H1", "2026-07+")
HYP_DESC = {"H1": "E10 顺点火", "H2": "E1 通道回踩", "H3": "E11 突破延迟一根"}
EXIT_DESC = {
    "a": "连续 2 根收盘越过中轨",
    "b": "MACD 柱连续 2 根缩短",
    "c": "H1 %b 回到带内 / H2·H3 跌破更高低点",
}


def _load(root: Path, conv: str, sfx: str = ""):
    base = root / "research" / "v4"
    cells = pd.read_parquet(base / f"cells_{conv}{sfx}.parquet")
    tp = base / f"trades_{conv}{sfx}.parquet"
    trades = pd.read_parquet(tp) if tp.exists() else pd.DataFrame()
    run = json.loads((base / f"run_{conv}{sfx}.json").read_text(encoding="utf-8"))
    return cells, trades, run


def main_table(cells: pd.DataFrame, fdr: bool = True) -> pd.DataFrame:
    c = cells.copy()
    c["格子"] = c["hyp"] + "·" + c["exit"] + "·" + c["dir"] + "·" + c["und"]
    c["p"] = c["p"].map(lambda v: f"{v:.4f}")
    c["FDR"] = np.where(c["fdr_pass"], "通过", "—") if fdr else "待统一"
    c["随机分位"] = [
        f"{p}（{n}）" if n else "—" for p, n in zip(c["random_pctile"], c["random_n"], strict=True)
    ]
    cols = {
        "格子": "格子",
        "n": "笔数",
        "win_rate": "胜率",
        "payoff": "盈亏比",
        "mean_pct": "均值%",
        "median_pct": "中位%",
        "sum_usd": "合计 USD",
        "max_dd_usd": "最大回撤 USD",
        "mean_pct_ex_top5": "去前5盈利均值%",
        "随机分位": "随机分位（笔）",
        "t": "t",
        "p": "p",
        "FDR": "FDR",
    }
    return c[list(cols)].rename(columns=cols)


def period_table(trades: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = trades.copy()
    if len(t):
        t["period"] = [v4d.period_of(d) for d in t["day"]]
    for _, c in cells.iterrows():
        r = {"格子": f"{c['hyp']}·{c['exit']}·{c['dir']}·{c['und']}"}
        sub = (
            t[
                (t["hyp"] == c["hyp"])
                & (t["exit"] == c["exit"])
                & (t["dir"] == c["dir"])
                & (t["und"] == c["und"])
            ]
            if len(t)
            else t
        )
        for p in PERIODS:
            s = sub[sub["period"] == p] if len(sub) else sub
            r[p] = (
                f"{len(s)} 笔 · {s['net_pct'].mean():+.1f}% · {s['net_usd'].sum():+.0f}"
                if len(s)
                else "—"
            )
        rows.append(r)
    return pd.DataFrame(rows)


def exit_mix(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    g = trades.groupby(["hyp", "exit"])
    out = g["reason"].value_counts(normalize=True).unstack(fill_value=0).round(2)
    out["止盈减半比例"] = g["tp_used"].mean().round(2)
    out["0DTE 比例"] = g["dte"].apply(lambda s: round(float((s == 0).mean()), 2))
    out["中位持仓 分钟"] = g.apply(
        lambda s: round(float(np.median((s["exit_ns"] - s["entry_ns"]) / 6e10)), 1)
    )
    return out.reset_index()


# ---------------- H4（D-030：只用日内单腿清单的入选样本） ----------------


def old_table() -> pd.DataFrame:
    md = (REPO / "docs" / "research" / "OWN_POLICY_EXTRACTED.md").read_text(encoding="utf-8")
    sec = md.split("## 157 笔按状态分类")[1].split("\n## ")[0]
    rows = re.findall(r"^\| (.+?) \| (\d+) \| ([\d.]+) \| ([-\d.]+) \| ([-\d.]+) \|$", sec, re.M)
    return pd.DataFrame(
        [
            {"state": s, "n": int(n), "win_rate": float(w), "sum_pnl": float(p)}
            for s, n, w, p, _ in rows
        ]
    )


def h4_table(root: Path) -> pd.DataFrame:
    t = pd.read_parquet(root / "research" / "v4" / "owner_intraday_review.parquet")
    pos = t  # D-030：每行即一个持仓段
    parts = {"旧 157 行（D-026）": old_table().set_index("state")}
    for conv in ("A", "C"):
        g = pos.groupby(f"{conv}_state")
        parts[f"D-030 口径 {conv}"] = pd.DataFrame(
            {
                "持仓段": g.size(),
                "FIFO 行": g["lots"].sum(),
                "胜率": g["pnl_net"].apply(lambda s: round(float((s > 0).mean()), 2)),
                "合计": g["pnl_net"].sum().round(0),
            }
        )
    out = pd.concat(parts, axis=1)
    out.columns = [f"{a} {b}" for a, b in out.columns]
    return out.reset_index().rename(columns={"index": "state"}).fillna("—")


# ---------------- R-035 ----------------


REPLAYS = {  # 标签 → (输出前缀, 规格文件, 状态说明)
    "R-037": ("rules37", "v4_rules37_test.yaml", "现行：规则来自 D-030 入选样本"),
    "R-035": ("rules", "v4_rules_test.yaml", "已被取代：规则来自 D-029 样本，按其登记原样报告"),
}


def rules_tables(
    root: Path, v4: dict, sfx: str = "", tag: str = "R-037"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prefix, spec_file, _ = REPLAYS[tag]
    base = root / "research" / "v4"
    spec, rules = rt.load_spec(REPO / "config" / spec_file)
    A, C = spec["base"], spec["contrast"]
    keys = ["hyp", "exit", "dir", "und"]
    cols = [
        "n",
        "win_rate",
        "payoff",
        "mean_pct",
        "median_pct",
        "sum_usd",
        "max_dd_usd",
        "mean_pct_ex_top5",
        "random_pctile",
        "p",
    ]
    out = {}
    for part in ("main", "extract"):
        a = pd.read_parquet(base / f"{prefix}_cells_{A}_{part}.parquet").set_index(keys)[cols]
        c = pd.read_parquet(base / f"{prefix}_cells_{C}_{part}.parquet").set_index(keys)[cols]
        m = a.add_suffix(f"_{A}").join(c.add_suffix(f"_{C}")).reset_index()
        m["p"] = m[[f"p_{A}", f"p_{C}"]].max(axis=1)
        out[part] = m
    w0, w1 = (dt.date.fromisoformat(spec["window"][k]) for k in ("from", "to"))
    e0, e1 = (dt.date.fromisoformat(spec["extraction_window"][k]) for k in ("from", "to"))
    rows = []
    for conv in (A, C):
        p = base / f"trades_{conv}{sfx}.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        t = t[(t["und"] == "MU") & t["day"].map(lambda d: w0 <= d <= w1 and not e0 <= d <= e1)]
        for (h, x, d), g in t.groupby(["hyp", "exit", "dir"]):
            rows.append(
                {
                    "口径": conv,
                    "格子": f"{h}·{x}·{d}",
                    "笔数": len(g),
                    "均值%": round(float(g["net_pct"].mean()), 2),
                    "合计 USD": round(float(g["net_usd"].sum()), 0),
                }
            )
    info = {"spec": spec, "rules": rules}
    return out["main"], out["extract"], pd.DataFrame(rows), info


def _rule_text(r: dict) -> str:
    return " 且 ".join(f"{c} {op} {v:.4g}" for c, op, v in r["conditions"])


def rules_and_fdr(root: Path, v4: dict, cp: pd.DataFrame, partial: bool = False) -> list[str]:
    base = root / "research" / "v4"
    sfx = "_MU" if partial else ""
    h5p = base / ("h5_cells_MU.parquet" if partial else "h5_cells.parquet")
    h5 = pd.read_parquet(h5p) if h5p.exists() else pd.DataFrame(columns=["p"])
    L, mains = [], {}
    for tag, (_, _, status) in REPLAYS.items():
        try:
            main_t, ext_t, h13, info = rules_tables(root, v4, sfx, tag)
        except FileNotFoundError:
            L += [f"## 6b. {tag}：日内单腿规则回放（{status}）", "", "尚未运行。", ""]
            continue
        mains[tag] = main_t
        spec = info["spec"]
        ew = spec["extraction_window"]
        L += [
            f"## 6b. {tag}：日内单腿规则在两年 MU 上回放（{status}）",
            "",
            "规则（冻结在 `" + spec["rules_file"] + "`）：",
            "",
            df_to_md(
                pd.DataFrame(
                    [
                        {
                            "规则": r["id"],
                            "条件": _rule_text(r),
                            "样本内入场": r["n"],
                            "样本内胜率": r["win_rate"],
                        }
                        for r in info["rules"]
                    ]
                )
            ),
            "",
            (
                "**FDR 待 QQQ 完成后统一计算；下表 p 为未校正值**（p = 两口径较大者）。"
                if partial
                else "p = 两口径较大者（单侧、按日聚类）。"
            )
            + f"规则特征方向化，**规则本身不含方向**，多空分开回放；主检验剔除规则来源期"
            f"（{ew['from']}～{ew['to']}）。",
            "",
            df_to_md(main_t.round(3)),
            "",
            "规则来源期（样本内，描述性，不进 FDR）：",
            "",
            df_to_md(ext_t.round(3)),
            "",
        ]
        if tag == "R-037":
            L += [
                "同一窗口（MU 主检验日期）上 H1–H3 的表现（来自 R-033 成交明细，并列对照）：",
                "",
                df_to_md(h13) if len(h13) else "无",
                "",
            ]
    if partial:
        L += [f"H5（MU）结果见 `docs/research/V4_H5.md`（{len(h5)} 格，FDR 同样待统一）。", ""]
        return L
    n33, n5 = len(cp), len(h5)
    L += [
        "## 6c. 多重比较汇总（BH q=" + str(v4["stats"]["fdr_q"]) + "，各族按其预登记原样计算）",
        "",
    ]
    L += [f"- R-033 预登记族（口径 C {n33} 格）：通过 {int(cp['fdr_pass'].sum())} 格。"]
    fam34 = np.r_[cp["p"].to_numpy(), h5["p"].to_numpy()]
    r34 = v4d.bh(fam34, v4["stats"]["fdr_q"])
    L += [
        f"- R-034 预登记族（R-033 {n33} + H5 {n5} = {len(fam34)} 格）：通过 {int(r34.sum())} 格"
        f"（R-033 {int(r34[:n33].sum())}、H5 {int(r34[n33:].sum())}）。"
    ]
    for tag in ("R-035", "R-037"):
        if tag not in mains:
            continue
        mt = mains[tag]
        fam = np.r_[cp["p"].to_numpy(), h5["p"].to_numpy(), mt["p"].to_numpy()]
        rej = v4d.bh(fam, v4["stats"]["fdr_q"])
        mt[f"FDR（{tag} 族）"] = np.where(rej[n33 + n5 :], "通过", "—")
        L += [
            f"- {tag} 预登记族（R-033 {n33} + H5 {n5} + {tag} {len(mt)} = {len(fam)} 格）："
            f"通过 {int(rej.sum())} 格（R-033 {int(rej[:n33].sum())}、"
            f"H5 {int(rej[n33 : n33 + n5].sum())}、{tag} {int(rej[n33 + n5 :].sum())}）。"
        ]
    L += [""]
    return L


# ---------------- 主流程 ----------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--unds", default=None, help="部分报告（如 MU），FDR 待统一")
    args = ap.parse_args(argv)
    partial = args.unds is not None
    sfx = f"_{args.unds.replace(',', '_')}" if partial else ""
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    P, S = v4["primary"], v4["sensitivity"]
    cp, tp, rp = _load(root, P, sfx)
    cs, ts, rs = _load(root, S, sfx)
    passed = cp[cp["fdr_pass"]] if not partial else cp.iloc[0:0]
    L = [
        "# V4 决定性检验（R-033）" + (f"——{args.unds} 部分（QQQ 待补）" if partial else ""),
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.research.v4_decisive` → `uv run python -m zr.research.v4_report`",
        "",
        "预登记：`docs/RESEARCH_LOG.md` R-033 与 `config/v4_decisive.yaml`，先于任何运行提交"
        "（commit f2db68c）。指标口径：主检验 C（D-028：含盘前盘后连续、不重置、长桥原始 K），"
        "敏感性 A（D-027：RTH-only、按日重置）。口径复核与三个校验点见 "
        "`docs/research/V4_CHECKPOINTS.md`。",
        "",
        f"## 1. FDR（BH q={v4['stats']['fdr_q']}，口径 {P} 的 {len(cp)} 格一族）",
        "",
    ]
    if partial:
        top = cp.sort_values("p").head(5)
        L += [
            f"**部分报告（{args.unds}）。FDR 待 QQQ 完成后对两个标的统一计算；以下所有 p 均为"
            "未校正值，不能单独据此判定通过。**",
            "",
            f"MU 口径 {P} 共 {len(cp)} 格，未校正 p 最小的 5 格：",
            "",
            df_to_md(main_table(top, fdr=False)),
            "",
        ]
    elif passed.empty:
        L += [
            f"**没有格子通过。** 最小 p = {cp['p'].min():.4f}"
            f"（{cp.loc[cp['p'].idxmin(), ['hyp', 'exit', 'dir', 'und']].str.cat(sep='·')}）。",
            "",
        ]
    else:
        L += ["通过的格子：", "", df_to_md(main_table(passed)), ""]
    L += [
        f"## 2. 全部格子（口径 {P}）",
        "",
        "均值% / 中位% = 扣费后每笔盈亏占所付权利金（入场 ask × 100 × 2 张）的百分比；盈亏比 = "
        "平均盈利 ÷ 平均亏损（USD）；随机分位 = 实际均值在 1000 次同时段随机入场均值中的分位"
        "（括号为有随机候选的笔数）；p 为单侧、按日聚类。",
        "",
        df_to_md(main_table(cp, fdr=not partial)),
        "",
        "## 3. 按年份（每格：笔数 · 扣费后均值% · 合计 USD；2026-07 后单列）",
        "",
        df_to_md(period_table(tp, cp)),
        "",
        "## 4. 离场方式、0DTE 比例、持仓时长（口径 C，全部已成交笔）",
        "",
        df_to_md(exit_mix(tp)),
        "",
        f"## 5. 敏感性：口径 {S}（D-027）",
        "",
        "FDR 待统一。" if partial else f"FDR（{S} 另算一族）通过 {int(cs['fdr_pass'].sum())} 格。",
        "",
        df_to_md(main_table(cs, fdr=not partial)),
        "",
        "## 6. H4：负责人交易按新口径重做状态分类（D-030：只用日内单腿清单的入选样本）",
        "",
        "按负责人指令（D-030），H4 只用 `docs/research/INTRADAY_SINGLE_LEG_LIST.md` 的入选样本，"
        "以持仓段为单位（分批建仓 / 卖出已合并）。决策 K = 入场前最后一根已完成的 2 分钟 K；"
        "「同向 / 反向」= 状态方向与所买 call / put 是否一致；盈亏为实际成交扣费后 USD。"
        "旧表是 157 个 FIFO 行全样本、D-026 口径（含分箱 bug），只作对照。",
        "",
        df_to_md(h4_table(root)),
        "",
        *rules_and_fdr(root, v4, cp, partial),
        "## 7. 数据与运行",
        "",
        f"- 口径 {P}：{json.dumps(rp['info'], ensure_ascii=False)}",
        f"- tick 拉取 {P}：{json.dumps(rp['pull'], ensure_ascii=False)}",
        f"- 口径 {S}：{json.dumps(rs['info'], ensure_ascii=False)}",
        f"- tick 拉取 {S}：{json.dumps(rs['pull'], ensure_ascii=False)}",
        "",
        "## 8. 边界",
        "",
        "- 标的 MU、QQQ；日期 2026-07-01～2026-10-21 全部交易日 + 2023-10-02～2026-06-30 的周四"
        "周五（各 352 个有期权数据的交易日，QQQ 用同一组日期）；2026-10-21 本地正股数据止于 "
        "15:37。",
        "- 成交：ThetaData 逐笔 NBBO（报价，不是逐笔成交记录）；入场 +2 s 后第一个有效 ask、"
        "离场 +2 s 后第一个有效 bid，市价单、2 张、不考虑盘口深度与部分成交；D-013 费用。",
        "- 合约只用日历日 DTE 0/1；MU 2026-07 前只有周四（DTE 1）、周五（DTE 0），QQQ 周五 "
        "10:15 后无 DTE ≤1 合约 → 不交易。0DTE 未按 D-014 减半（与「+20% 减半」冲突）。",
        "- 口径 C 开盘前 40 分钟的指标值未经 App 截图直接核实（STATUS 待确认）；若核实为 A，"
        "以第 5 节为准。",
        "- 每格按入场时间贪心，单格内不重叠；不同格之间不做资金/仓位约束，不是组合回测。",
        "- 结论只适用于上述两只标的、上述日期和 R-033 固定的规则参数；未覆盖其他标的、DTE ≥ 2、"
        "限价单执行、其他离场参数。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
