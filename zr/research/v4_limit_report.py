"""R-039 汇总：H5 限价格子 + docs/research/FILL_MODEL_RESULTS.md。

输入（均由 zr.research.v4_limit / owner_limit / fill_segments 落盘）：
research/v4_limit/{sims,cells}_{base|0dte}_{C|A}_{MU|QQQ}.parquet、owner_replay.parquet、
owner_canceled.parquet、owner_orders.parquet；research/v4/h5_oos_*.parquet（R-034 样本外概率）、
research/v4/cells_{C,A}.parquet（R-033 吃价差结果）；research/audit/contract_choice.parquet。
运行：uv run python -m zr.research.v4_limit_report
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

log = logging.getLogger("zr.research.v4_limit_report")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "FILL_MODEL_RESULTS.md"
UNDS = ("MU", "QQQ")


def _load(store: Path, kind: str, variant: str, conv: str) -> pd.DataFrame:
    parts = [
        pd.read_parquet(p)
        for u in UNDS
        if (p := store / f"{kind}_{variant}_{conv}_{u}.parquet").exists()
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def h5_limit(root: Path, v4: dict) -> pd.DataFrame:
    h5 = v4_h5.load_h5()
    store = root / "research" / "v4_limit"
    rows = []
    for und in UNDS:
        for x in v4_h5.EXITS:
            per_conv = {}
            for conv in (h5["base"], h5["contrast"]):
                f = root / "research" / "v4" / f"h5_oos_{conv}_{und}_{x}.parquet"
                sp = store / f"sims_base_{conv}_{und}.parquet"
                if not f.exists() or not sp.exists():
                    continue
                o = pd.read_parquet(f)[["decision_ns", "direction", "hyp", "p"]]
                o["rule"] = [v4["exits"][x][h] for h in o["hyp"]]
                sims = pd.read_parquet(sp)
                per_conv[conv] = o.merge(sims, on=["decision_ns", "direction", "rule"])
            for cfg in sorted({c for d in per_conv.values() for c in d["cfg"].unique()}):
                for thr in h5["eval"]["thresholds"]:
                    st = {}
                    for conv, d in per_conv.items():
                        g = d[(d["cfg"] == cfg) & (d["p"] >= thr)]
                        g = g.drop_duplicates(["decision_ns", "direction"])
                        t = vl.greedy_attempts(g) if len(g) else g
                        fl = t[t["filled"] > 0]
                        s = (
                            v4d.cell_stats(fl, v4)
                            if len(fl)
                            else {"n": 0, "mean_pct": np.nan, "p": 1.0, "mean_pct_ex_top5": np.nan}
                        )
                        s["fill_rate"] = len(fl) / len(t) if len(t) else np.nan
                        st[conv] = s
                    if len(st) < 2:
                        continue
                    rows.append(
                        {
                            "und": und,
                            "exit": x,
                            "thr": thr,
                            "cfg": cfg,
                            **{f"n_{c}": st[c]["n"] for c in st},
                            **{f"mean%_{c}": st[c]["mean_pct"] for c in st},
                            **{f"ex5%_{c}": st[c]["mean_pct_ex_top5"] for c in st},
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


def cfg_summary(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby(["cfg", "und"])
    return (
        g.agg(
            格子=("n", "size"),
            笔数=("n", "sum"),
            均值_mean_pct=("mean_pct", "mean"),
            最好格=("mean_pct", "max"),
            最差格=("mean_pct", "min"),
            正均值格数=("mean_pct", lambda x: int((x > 0).sum())),
            成交率=("fill_rate", "mean"),
            错过率=("miss_rate", "mean"),
            错过事件的吃价差结果=("missed_taker_mean_pct", "mean"),
            成交事件的吃价差结果=("filled_taker_mean_pct", "mean"),
            最小p=("p", "min"),
        )
        .round(3)
        .reset_index()
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    lc = vl.load_limit()
    store = root / "research" / "v4_limit"
    cC = _load(store, "cells", "base", "C")
    cA = _load(store, "cells", "base", "A")
    h5c = h5_limit(root, v4)
    h5c.to_parquet(store / "h5_limit_cells.parquet", index=False)
    lim = cC[cC["cfg"] != "taker"].reset_index(drop=True)
    h5l = h5c[h5c["cfg"] != "taker"].reset_index(drop=True)
    fam_p = np.r_[lim["p"].to_numpy(float), h5l["p"].to_numpy(float)]
    rej = v4d.bh(fam_p, lc["stats"]["fdr_q"]) if len(fam_p) else np.array([], bool)
    lim["fdr_486"] = rej[: len(lim)]
    h5l["fdr_486"] = rej[len(lim) :]
    lim.to_parquet(store / "family_v4_limit_C.parquet", index=False)
    h5l.to_parquet(store / "family_h5_limit.parquet", index=False)
    taker33 = pd.read_parquet(root / "research" / "v4" / "cells_C.parquet")
    # 自检：本模块重算的吃价差格子 vs R-033 落盘结果
    tk = cC[cC["cfg"] == "taker"].merge(
        taker33[["und", "hyp", "exit", "dir", "n", "mean_pct"]],
        on=["und", "hyp", "exit", "dir"],
        suffixes=("", "_r033"),
    )
    tk_same = int(
        ((tk["n"] == tk["n_r033"]) & np.isclose(tk["mean_pct"], tk["mean_pct_r033"])).sum()
    )

    rep = pd.read_parquet(store / "owner_replay.parquet")
    can = pd.read_parquet(store / "owner_canceled.parquet")
    ep = rep[rep["kind"] == "episode"]
    od = rep[rep["kind"] == "order"]
    owner_tab = (
        ep.groupby(["ref", "cfg"])
        .agg(
            持仓段=("episode", "size"),
            模拟错过=("missed", "sum"),
            模拟盈亏=("sim_pnl", "sum"),
            对账单盈亏=("act_pnl", "sum"),
        )
        .round(2)
        .reset_index()
    )
    owner_tab["复现比例"] = (owner_tab["模拟盈亏"] / owner_tab["对账单盈亏"]).round(3)
    bias = (
        od.groupby(["ref", "cfg", "side"])
        .agg(
            委托数=("order_id", "size"),
            偏差中位数_pct=("bias_pct", "median"),
            偏差均值_pct=("bias_pct", "mean"),
            偏差中位数_tick=("bias_ticks", "median"),
            成交比例=("fill_ratio", "mean"),
        )
        .round(3)
        .reset_index()
    )
    cc = pd.read_parquet(root / "research" / "audit" / "contract_choice.parquet")
    late = cc[cc["规则无合约原因"] == "10:15 后排除 0DTE，且无 1DTE"]
    ep14 = ep[ep["open_et"].isin(late["开仓（ET）"])]
    t14 = (
        ep14.groupby(["ref", "cfg"])
        .agg(
            持仓段=("episode", "size"),
            模拟错过=("missed", "sum"),
            模拟盈亏=("sim_pnl", "sum"),
            对账单盈亏=("act_pnl", "sum"),
        )
        .round(2)
        .reset_index()
    )
    can_tab = (
        can.groupby("cfg")
        .agg(
            撤单机会=("order_id", "size"),
            模拟成交=("filled", lambda x: int((x > 0).sum())),
            净值=("net_usd", "sum"),
            每笔均值_pct=("net_pct", "mean"),
        )
        .round(2)
        .reset_index()
        if len(can)
        else pd.DataFrame()
    )
    sC, sA = cfg_summary(cC), cfg_summary(cA)
    z = {v: _load(store, "cells", "0dte", v) for v in ("C", "A")}
    sZ = cfg_summary(z["C"]) if len(z["C"]) else pd.DataFrame()
    best = lim.sort_values("mean_pct", ascending=False).head(12)[
        [
            "und",
            "cfg",
            "hyp",
            "exit",
            "dir",
            "n",
            "fill_rate",
            "mean_pct",
            "mean_pct_ex_top5",
            "p",
            "fdr_486",
        ]
    ]
    h5_sum = (
        h5l.groupby(["cfg"])
        .agg(
            格子=("p", "size"),
            条件满足=("条件满足", "sum"),
            最小p=("p", "min"),
            均值C=("mean%_C", "mean"),
            均值A=("mean%_A", "mean"),
        )
        .round(3)
        .reset_index()
    )
    fl = ep[(ep["ref"] == "fill")]
    best_owner = owner_tab[(owner_tab["ref"] == "fill") & (owner_tab["cfg"] != "taker")]
    bo = best_owner.sort_values("模拟盈亏", ascending=False).iloc[0]
    tk_fill = owner_tab[(owner_tab["ref"] == "fill") & (owner_tab["cfg"] == "taker")].iloc[0]
    sub_best = owner_tab[(owner_tab["ref"] == "submit")].sort_values("模拟盈亏").iloc[-1]
    n_pos = int((lim["mean_pct"] > 0).sum())
    n_fdr = int(lim["fdr_486"].sum() + h5l["fdr_486"].sum())
    sg = pd.read_parquet(root / "research" / "fill_model" / "owner_segments.parquet")
    sg = sg[(sg["skip"] == "") & (sg["group"] == "d030") & (sg["side"] == "Buy") & (sg["seg"] == 0)]
    seg_med = float(sg["offset_spread"].median())
    lines = [
        "# 限价成交模型下的重跑（R-039）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} · 复现：`uv run python -m "
        "zr.research.fill_segments` → `zr.research.v4_limit --conv both` → `--variant 0dte` → "
        "`zr.research.owner_limit` → `zr.research.v4_limit_report`。预登记 RESEARCH_LOG R-039、"
        "`config/v4_limit.yaml`；校准见 `FILL_MODEL_CALIBRATION.md`。",
        "",
        "## 摘要",
        "",
        f"1. **能复现多少负责人的执行优势**：对账单 51 段合计 {tk_fill['对账单盈亏']:+,.2f} USD。"
        "按预登记口径（在负责人每张委托的提交时刻执行规则）最好的一组是 "
        f"{sub_best['cfg']}：{sub_best['模拟盈亏']:+,.2f} USD。"
        "负责人常把限价挂在远离市价处等价格过来"
        f"（开仓买单首段挂单位置中位数 = 中间价 {seg_med:+.2f} 个 spread，−0.5 即买一），"
        "在提交时刻执行规则复现不了这种「挂单等价」。补充口径（未预登记）：在每张委托实际首笔成交的"
        f"时刻执行规则——吃价差 {tk_fill['模拟盈亏']:+,.2f} USD，最好的 {bo['cfg']} "
        f"{bo['模拟盈亏']:+,.2f} USD（为对账单的 {bo['复现比例']:.0%}，其中 "
        f"{int(bo['模拟错过'])} 段模拟没成交）。即：给定负责人选的成交时刻，执行规则能拿回的部分见"
        "此比例；负责人「在哪个价位挂单、等多久」本身不是执行规则能复现的。",
        f"2. **V4 是否翻正**：口径 C 324 个限价格子（9 组挂单 × 36）均值 > 0 的 {n_pos} 格；"
        f"本族 486 格（+ H5 162 格）BH q=0.10 通过 {n_fdr} 格（最终族与 R-040 合并后计算）。"
        "逐组汇总见第 2 节。",
        f"3. **10:15 后开仓的 0DTE（规则无合约的 14 段）**：对账单 "
        f"{t14[t14['cfg'] == 'taker']['对账单盈亏'].iloc[0] if len(t14) else float('nan'):+,.2f}"
        " USD；"
        "各口径模拟见第 2.3 节；V4 去掉 10:15 限制的敏感性见第 5 节。",
        "",
        "## 1. 模拟器在负责人真实成交上的检验",
        "",
        "### 1.1 持仓段盈亏（51 段，模拟 vs 对账单）",
        "",
        "ref = submit：在每张委托的提交时刻执行（预登记）；ref = fill：在该委托实际首笔成交时刻执行"
        "（补充）。开仓没成交的张数放弃（错过）；平仓按模拟持有张数依次卖出。",
        "",
        df_to_md(owner_tab),
        "",
        "### 1.2 逐张委托的模拟成交价偏差（正 = 模拟比实际差）",
        "",
        df_to_md(bias),
        "",
        "### 1.3 10:15 后开仓、V4 规则无合约的 0DTE（14 段）",
        "",
        df_to_md(t14),
        "",
        "### 1.4 撤单 / 过期的开仓限价单（MU，模拟成交后按离场 (c) 管理）",
        "",
        df_to_md(can_tab) if len(can_tab) else "无",
        "",
        "## 2. V4 H1–H3（口径 C 主检验）",
        "",
        f"自检：本模块重算的吃价差格子与 R-033 落盘结果笔数、均值都相同的 {tk_same}/{len(tk)} 格。",
        "",
        df_to_md(sC),
        "",
        "均值最高的 12 个限价格子：",
        "",
        df_to_md(best),
        "",
        "## 3. 口径 A（敏感性）",
        "",
        df_to_md(sA) if len(sA) else "缺",
        "",
        "## 4. H5 阈值 0.55 / 0.60 / 0.65（样本外概率同 R-034，只换成交口径）",
        "",
        df_to_md(h5_sum),
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
        "- 成交模型：OPRA 成交越过挂单价即成交、恰好等于时 q = 0.9；V4 的 tick 缓存没有挂单量，"
        "盘口越过时"
        "视为全量可成交；排队位置无法观测；负责人自己的成交在 OPRA 数据里（校准报告给了上下界）。",
        "- +20% 减仓：入场只成交 1 张时不减仓（2 张时卖 1 张）。",
        "- 随机入场对照没有在限价口径下重算。",
        "- 负责人「ref = fill」口径用了负责人的实际成交时刻，含未来信息（只用于衡量执行环节，不"
        "是可交易规则）。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", OUT)
    _ = fl
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
