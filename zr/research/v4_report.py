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
from zr.features.bars import load_bars, load_extended_bars, resample_bars
from zr.features.regime import STATE_NAMES, indicators, regimes
from zr.ops.markdown import df_to_md
from zr.research import v4_decisive as v4d
from zr.research.own_policy import decision_bars, owner_trades

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "V4_DECISIVE.md"
PERIODS = ("2023", "2024", "2025", "2026H1", "2026-07+")
HYP_DESC = {"H1": "E10 顺点火", "H2": "E1 通道回踩", "H3": "E11 突破延迟一根"}
EXIT_DESC = {"a": "连续 2 根收盘越过中轨", "b": "MACD 柱连续 2 根缩短",
             "c": "H1 %b 回到带内 / H2·H3 跌破更高低点"}


def _load(root: Path, conv: str):
    base = root / "research" / "v4"
    cells = pd.read_parquet(base / f"cells_{conv}.parquet")
    tp = base / f"trades_{conv}.parquet"
    trades = pd.read_parquet(tp) if tp.exists() else pd.DataFrame()
    run = json.loads((base / f"run_{conv}.json").read_text(encoding="utf-8"))
    return cells, trades, run


def main_table(cells: pd.DataFrame) -> pd.DataFrame:
    c = cells.copy()
    c["格子"] = c["hyp"] + "·" + c["exit"] + "·" + c["dir"] + "·" + c["und"]
    c["p"] = c["p"].map(lambda v: f"{v:.4f}")
    c["FDR"] = np.where(c["fdr_pass"], "通过", "—")
    c["随机分位"] = [f"{p}（{n}）" if n else "—" for p, n in
                  zip(c["random_pctile"], c["random_n"], strict=True)]
    cols = {"格子": "格子", "n": "笔数", "win_rate": "胜率", "payoff": "盈亏比",
            "mean_pct": "均值%", "median_pct": "中位%", "sum_usd": "合计 USD",
            "max_dd_usd": "最大回撤 USD", "mean_pct_ex_top5": "去前5盈利均值%",
            "随机分位": "随机分位（笔）", "t": "t", "p": "p", "FDR": "FDR"}
    return c[list(cols)].rename(columns=cols)


def period_table(trades: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = trades.copy()
    if len(t):
        t["period"] = [v4d.period_of(d) for d in t["day"]]
    for _, c in cells.iterrows():
        r = {"格子": f"{c['hyp']}·{c['exit']}·{c['dir']}·{c['und']}"}
        sub = t[(t["hyp"] == c["hyp"]) & (t["exit"] == c["exit"]) & (t["dir"] == c["dir"])
                & (t["und"] == c["und"])] if len(t) else t
        for p in PERIODS:
            s = sub[sub["period"] == p] if len(sub) else sub
            r[p] = (f"{len(s)} 笔 · {s['net_pct'].mean():+.1f}% · {s['net_usd'].sum():+.0f}"
                    if len(s) else "—")
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
        lambda s: round(float(np.median((s["exit_ns"] - s["entry_ns"]) / 6e10)), 1))
    return out.reset_index()


# ---------------- H4 ----------------


def classify_157(root: Path, conv_key: str, v4: dict) -> pd.DataFrame:
    rt = owner_trades(root)
    conv = v4["conventions"][conv_key]
    cfg = v4d.state_cfg(v4, conv_key)
    rows = []
    for u in sorted(rt["und"].unique()):
        sub = rt[rt["und"] == u]
        try:
            s = (load_extended_bars(root, u, clean=conv.get("clean", True)) if conv["ext"]
                 else load_bars(root, u))
        except (FileNotFoundError, ValueError):
            s = pd.DataFrame()
        if s.empty:
            rows += [{"und": u, "state": "未覆盖（无正股 1 分钟 K）", "pnl": p}
                     for p in sub["pnl_net"]]
            continue
        b = resample_bars(s, "2min")
        x = indicators(b, cfg, daily_reset=conv["reset"], warmup=conv["warmup"])
        reg = regimes(b, x, cfg, 2)
        pos = decision_bars(sub, b)
        for (_, r), p in zip(sub.iterrows(), pos, strict=True):
            if p < 0:
                st = "未覆盖（入场在首根 K 收盘前）"
            elif np.isnan(x["pctb"].iat[p]):
                st = "无有效 BOLL（开盘前 20 根）"
            else:
                k, d = int(reg["state"].iat[p]), int(reg["dir"].iat[p])
                st = STATE_NAMES[k] + ("" if d == 0 else (" 同向" if d == r["dir"] else " 反向"))
            rows.append({"und": u, "state": st, "pnl": r["pnl_net"]})
    return pd.DataFrame(rows)


def old_table() -> pd.DataFrame:
    md = (REPO / "docs" / "research" / "OWN_POLICY_EXTRACTED.md").read_text(encoding="utf-8")
    sec = md.split("## 157 笔按状态分类")[1].split("\n## ")[0]
    rows = re.findall(r"^\| (.+?) \| (\d+) \| ([\d.]+) \| ([-\d.]+) \| ([-\d.]+) \|$", sec, re.M)
    return pd.DataFrame([{"state": s, "n": int(n), "win_rate": float(w), "sum_pnl": float(p)}
                         for s, n, w, p, _ in rows])


def h4_table(root: Path, v4: dict) -> pd.DataFrame:
    def agg(cls: pd.DataFrame) -> pd.DataFrame:
        g = cls.groupby("state")["pnl"]
        return pd.DataFrame({"n": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()).round(2),
                             "sum_pnl": g.sum().round(0)})

    parts = {"旧（D-026）": old_table().set_index("state")}
    for k in (v4["primary"], v4["sensitivity"]):
        parts[f"口径 {k}"] = agg(classify_157(root, k, v4))
    out = pd.concat(parts, axis=1)
    out.columns = [f"{a} {b}" for a, b in out.columns]
    return out.reset_index().rename(columns={"index": "state"}).fillna("—")


# ---------------- 主流程 ----------------


def main() -> int:
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    P, S = v4["primary"], v4["sensitivity"]
    cp, tp, rp = _load(root, P)
    cs, ts, rs = _load(root, S)
    passed = cp[cp["fdr_pass"]]
    L = [
        "# V4 决定性检验（R-033）",
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
    if passed.empty:
        L += [f"**没有格子通过。** 最小 p = {cp['p'].min():.4f}"
              f"（{cp.loc[cp['p'].idxmin(), ['hyp', 'exit', 'dir', 'und']].str.cat(sep='·')}）。",
              ""]
    else:
        L += ["通过的格子：", "", df_to_md(main_table(passed)), ""]
    L += [
        f"## 2. 全部格子（口径 {P}）",
        "",
        "均值% / 中位% = 扣费后每笔盈亏占所付权利金（入场 ask × 100 × 2 张）的百分比；盈亏比 = "
        "平均盈利 ÷ 平均亏损（USD）；随机分位 = 实际均值在 1000 次同时段随机入场均值中的分位"
        "（括号为有随机候选的笔数）；p 为单侧、按日聚类。",
        "",
        df_to_md(main_table(cp)),
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
        f"FDR（{S} 另算一族）通过 {int(cs['fdr_pass'].sum())} 格。",
        "",
        df_to_md(main_table(cs)),
        "",
        "## 6. H4：负责人 157 笔按新口径重做状态分类",
        "",
        "决策 K = 入场前最后一根已收盘的 2 分钟 K；「同向 / 反向」= 状态方向与所买 call / put "
        "是否一致；盈亏为实际成交扣费后（USD）。旧表取自 OWN_POLICY_EXTRACTED.md（D-026 口径，"
        "含分箱 bug）。",
        "",
        df_to_md(h4_table(root, v4)),
        "",
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
