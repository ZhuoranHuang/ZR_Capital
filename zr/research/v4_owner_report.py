"""R-041 汇总 → docs/research/FILL_MODEL_RESULTS.md（负责人挂单策略成交；吃价差对照）。

输入：research/v4_owner/{sims,cells}_{base|0dte}_{C|A}_{MU|QQQ}.parquet、owner_replay.parquet、
owner_canceled.parquet；research/v4/h5_oos_*.parquet（R-034 样本外概率）；
research/audit/contract_choice.parquet（10:15 后 0DTE 的 14 段）；research/fill_model/*（校准）。
运行：uv run python -m zr.research.v4_owner_report
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.ops.markdown import df_to_md
from zr.research import v4_decisive as v4d
from zr.research import v4_h5
from zr.research import v4_limit as vl

log = logging.getLogger("zr.research.v4_owner_report")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "FILL_MODEL_RESULTS.md"
UNDS = ("MU", "QQQ")
PRIMARY = "owner_s0"


def _load(store: Path, kind: str, variant: str, conv: str) -> pd.DataFrame:
    parts = [
        pd.read_parquet(p)
        for u in UNDS
        if (p := store / f"{kind}_{variant}_{conv}_{u}.parquet").exists()
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def h5_cells(root: Path, v4: dict, store: Path) -> pd.DataFrame:
    h5 = v4_h5.load_h5()
    rows = []
    for und in UNDS:
        for x in v4_h5.EXITS:
            per = {}
            for conv in (h5["base"], h5["contrast"]):
                f = root / "research" / "v4" / f"h5_oos_{conv}_{und}_{x}.parquet"
                sp = store / f"sims_base_{conv}_{und}.parquet"
                if not f.exists() or not sp.exists():
                    continue
                o = pd.read_parquet(f)[["decision_ns", "direction", "hyp", "p"]]
                o["rule"] = [v4["exits"][x][h] for h in o["hyp"]]
                per[conv] = o.merge(pd.read_parquet(sp), on=["decision_ns", "direction", "rule"])
            if len(per) < 2:
                continue
            for cfg in sorted(set.intersection(*[set(d["cfg"]) for d in per.values()])):
                for thr in h5["eval"]["thresholds"]:
                    st = {}
                    for conv, d in per.items():
                        g = d[(d["cfg"] == cfg) & (d["p"] >= thr)].drop_duplicates(
                            ["decision_ns", "direction"]
                        )
                        t = vl.greedy_attempts(g) if len(g) else g
                        fl = t[t["filled"] > 0]
                        s = (
                            v4d.cell_stats(fl, v4)
                            if len(fl)
                            else {"n": 0, "mean_pct": np.nan, "p": 1.0, "mean_pct_ex_top5": np.nan}
                        )
                        s["fill_rate"] = len(fl) / len(t) if len(t) else np.nan
                        st[conv] = s
                    rows.append(
                        {
                            "und": und,
                            "exit": x,
                            "thr": thr,
                            "cfg": cfg,
                            **{f"n_{c}": st[c]["n"] for c in st},
                            **{f"mean%_{c}": st[c]["mean_pct"] for c in st},
                            **{f"fill_{c}": round(st[c]["fill_rate"], 3) for c in st},
                            "p": max(st[c]["p"] for c in st),
                            "条件满足": all(
                                np.isfinite(st[c]["mean_pct"])
                                and st[c]["mean_pct"] > 0
                                and np.isfinite(st[c]["mean_pct_ex_top5"])
                                and st[c]["mean_pct_ex_top5"] > 0
                                for c in st
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summary(c: pd.DataFrame) -> pd.DataFrame:
    return (
        c.groupby(["cfg", "und"])
        .agg(
            格子=("n", "size"),
            笔数=("n", "sum"),
            格子均值_平均=("mean_pct", "mean"),
            最好格=("mean_pct", "max"),
            最差格=("mean_pct", "min"),
            正均值格=("mean_pct", lambda x: int((x > 0).sum())),
            成交率=("fill_rate", "mean"),
            错过率=("miss_rate", "mean"),
            错过事件吃价差结果=("missed_taker_mean_pct", "mean"),
            成交事件吃价差结果=("filled_taker_mean_pct", "mean"),
            最小p=("p", "min"),
        )
        .round(3)
        .reset_index()
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    store = root / "research" / "v4_owner"
    cC, cA = _load(store, "cells", "base", "C"), _load(store, "cells", "base", "A")
    cZ = _load(store, "cells", "0dte", "C")
    h5 = h5_cells(root, v4, store)
    h5.to_parquet(store / "h5_cells.parquet", index=False)
    prim = cC[cC["cfg"] == PRIMARY].reset_index(drop=True)
    h5p = h5[h5["cfg"] == PRIMARY].reset_index(drop=True)
    rej = v4d.bh(np.r_[prim["p"].to_numpy(float), h5p["p"].to_numpy(float)], 0.10)
    prim["fdr_54"], h5p["fdr_54"] = rej[: len(prim)], rej[len(prim) :]
    prim.to_parquet(store / "family_v4_C.parquet", index=False)
    h5p.to_parquet(store / "family_h5.parquet", index=False)
    rep = pd.read_parquet(store / "owner_replay.parquet")
    can = pd.read_parquet(store / "owner_canceled.parquet")
    ep, od = rep[rep["kind"] == "episode"], rep[rep["kind"] == "order"]
    own = (
        ep.groupby("cfg")
        .agg(
            持仓段=("episode", "size"),
            模拟错过=("missed", "sum"),
            模拟盈亏=("sim_pnl", "sum"),
            对账单盈亏=("act_pnl", "sum"),
        )
        .round(2)
        .reset_index()
    )
    own["占对账单"] = (own["模拟盈亏"] / own["对账单盈亏"]).round(3)
    bias = (
        od.groupby(["cfg", "side"])
        .agg(
            委托=("bias_pct", "size"),
            偏差中位数_pct=("bias_pct", "median"),
            偏差中位数_tick=("bias_ticks", "median"),
        )
        .round(3)
        .reset_index()
    )
    cc = pd.read_parquet(root / "research" / "audit" / "contract_choice.parquet")
    late = cc[cc["规则无合约原因"] == "10:15 后排除 0DTE，且无 1DTE"]["开仓（ET）"]
    e14 = ep[ep["open_et"].isin(late)]
    t14 = (
        e14.groupby("cfg")
        .agg(
            持仓段=("episode", "size"),
            模拟错过=("missed", "sum"),
            模拟盈亏=("sim_pnl", "sum"),
            对账单盈亏=("act_pnl", "sum"),
        )
        .round(2)
        .reset_index()
    )
    ctab = (
        can.groupby("cfg")
        .agg(
            撤单机会=("order_id", "size"),
            模拟成交=("filled", lambda x: int((x > 0).sum())),
            净值=("net_usd", "sum"),
        )
        .round(2)
        .reset_index()
        if len(can)
        else pd.DataFrame()
    )
    sC, sA, sZ = (
        summary(cC),
        summary(cA) if len(cA) else pd.DataFrame(),
        summary(cZ) if len(cZ) else pd.DataFrame(),
    )
    h5s = (
        h5.groupby("cfg")
        .agg(
            格子=("p", "size"),
            条件满足=("条件满足", "sum"),
            最小p=("p", "min"),
            均值A=("mean%_A", "mean"),
            均值C=("mean%_C", "mean"),
        )
        .round(3)
        .reset_index()
    )
    fmd = root / "research" / "fill_model"
    val = pd.read_parquet(fmd / "owner_strategy_validation.parquet")
    feat = pd.read_parquet(fmd / "owner_order_strategy.parquet")
    sim_fr, act_fr = float(val["sim_fill_prob"].mean()), float(val["actual_filled"].mean())
    late_share = float((feat.loc[feat["filled"], "fill_s"] > 120).mean())
    tk = own[own["cfg"] == "taker"].iloc[0]
    op = own[own["cfg"] == PRIMARY].iloc[0]
    orng = own[own["cfg"].str.startswith("owner")]["模拟盈亏"]
    n_pos = int((prim["mean_pct"] > 0).sum())
    tk14 = t14[t14["cfg"] == "taker"]
    pr14 = t14[t14["cfg"] == PRIMARY]
    v4t = sC[sC["cfg"] == "taker"]
    v4o = sC[sC["cfg"] == PRIMARY]
    lines = [
        "# 负责人挂单策略下的重跑（R-041）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} · 复现：`zr.research.fil"
        "l_segments` → "
        "`zr.research.owner_strategy` → `zr.research.v4_owner --conv both` → `--conv C --varian"
        "t 0dte` → "
        "`zr.research.owner_replay` → `zr.research.v4_owner_report`。预登记 RESEARCH_LOG R-041、"
        "`config/v4_owner.yaml`；策略与验证见 `FILL_MODEL_CALIBRATION.md`。"
        "R-039（固定挂单价 × 固定 T）已作废，不在本报告的任何结论里。",
        "",
        "## 摘要",
        "",
        "1. **能复现负责人多少真实成交与执行优势**：成交判定本身在负责人的实际挂单价与时长下"
        "与真实记录一致（成交价、耗时偏差中位数 0，撤单全部一致）；按经验分布抽样的策略在验证中"
        f"成交 {sim_fr:.0%} 的单子（负责人 {act_fr:.0%}）。负责人 51 段在委托提交时刻按策略重放："
        f"种子 0 {op['模拟盈亏']:+,.2f} USD（{int(op['模拟错过'])} 段没成交），三个种子 "
        f"{orng.min():+,.2f}～{orng.max():+,.2f}；吃价差 {tk['模拟盈亏']:+,.2f}；对账单 "
        f"{tk['对账单盈亏']:+,.2f}。即模拟复现不了负责人的盈利——差别不在撮合，而在负责人挂哪里、"
        "挂多久与行情相关（挂得远的单子往往等到价格过来），独立抽样拿不到这部分。",
        f"2. **V4 在该策略下是否翻正**：口径 C 36 格（种子 0）均值 > 0 的 {n_pos} 格；与 H5 18 格"
        "合为 54 格族，"
        f"BH q=0.10 通过 {int(prim['fdr_54'].sum() + h5p['fdr_54'].sum())} 格（最终族待 R-040 补"
        "登后合并）。"
        "按标的的格子均值（吃价差 → 策略）："
        + "；".join(
            f"{u} {float(v4t[v4t['und'] == u]['格子均值_平均'].iloc[0]):+.2f}% → "
            f"{float(v4o[v4o['und'] == u]['格子均值_平均'].iloc[0]):+.2f}%"
            for u in UNDS
            if len(v4t[v4t["und"] == u]) and len(v4o[v4o["und"] == u])
        )
        + "（每笔占权利金）。",
        f"3. **10:15 后开仓的 0DTE（14 段）**：对账单 "
        f"{float(tk14['对账单盈亏'].iloc[0]) if len(tk14) else float('nan'):+,.2f} USD；吃价差 "
        f"{float(tk14['模拟盈亏'].iloc[0]) if len(tk14) else float('nan'):+,.2f}；策略（种子 0）"
        f"{float(pr14['模拟盈亏'].iloc[0]) if len(pr14) else float('nan'):+,.2f}"
        f"（{int(pr14['模拟错过'].iloc[0]) if len(pr14) else 0} 段没成交）。",
        "",
        "## 1. 负责人日内单腿（D-030 入选 51 段，决策 = 每张委托提交时刻）",
        "",
        df_to_md(own),
        "",
        "逐张委托的模拟成交价偏差（正 = 模拟比实际差；只含模拟成交的委托）：",
        "",
        df_to_md(bias),
        "",
        "### 1.1 10:15 后开仓、V4 规则无合约的 0DTE（14 段）",
        "",
        df_to_md(t14),
        "",
        "### 1.2 撤单 / 过期的开仓限价单（校准样本内，MU；模拟成交后按离场 (c) 管理）",
        "",
        df_to_md(ctab) if len(ctab) else "无",
        "",
        "## 2. V4 H1–H3（口径 C 主检验）",
        "",
        "owner_s0 为预登记主口径，owner_s1 / s2 只看范围；taker = R-033 吃价差。",
        "",
        df_to_md(sC),
        "",
        "主口径 36 格明细：",
        "",
        df_to_md(
            prim[
                [
                    "und",
                    "hyp",
                    "exit",
                    "dir",
                    "attempts",
                    "fill_rate",
                    "n",
                    "mean_pct",
                    "mean_pct_ex_top5",
                    "missed_taker_mean_pct",
                    "p",
                    "fdr_54",
                ]
            ]
        ),
        "",
        "## 3. 口径 A（敏感性）",
        "",
        df_to_md(sA) if len(sA) else "缺",
        "",
        "## 4. H5 阈值 0.55 / 0.60 / 0.65（样本外概率同 R-034，只换成交口径）",
        "",
        df_to_md(h5s),
        "",
        "## 5. 敏感性：0DTE 不设 10:15 限制、15:30 前清仓（口径 C，不入族）",
        "",
        df_to_md(sZ) if len(sZ) else "缺",
        "",
        "## 边界",
        "",
        "- V4：MU、QQQ，日期同 R-033（2026-07-01～2026-10-21 全部交易日 + 2023-10-02～2026-06-3"
        "0 周四周五）；"
        "负责人：2026-06-09～2026-08-19 D-030 入选 51 段（MU 48、DRAM 2、META 1）。",
        f"- 策略参数来自负责人同一时期的订单（{len(feat)} 单），对负责人样本是样本内；用于 V4 时"
        "假设负责"
        "人的挂单习惯"
        "与行情无关（独立抽样），这正是验证里复现不了的部分。",
        f"- 等待时长截到 5–120 秒（负责人指令），负责人已成交订单中 {late_share:.0%} 在 120 秒之"
        "后成交；",
        "- 成交判定：q = 0.9；V4 tick 缓存无挂单量（盘口越过视为全量）；OPRA 含负责人自己的成交。",
        "- 随机入场对照没有在策略口径下重算。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
