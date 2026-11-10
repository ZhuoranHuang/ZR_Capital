# ruff: noqa: E501  （中文报告行较长）
"""只读已有结果（不新增检验）：夜间研究阶段 5 等权合成（QQQ、SPY，2% 主档）与排名最前的 1–2 分钟反转因子
（p1_rsi14_chg1，按阶段 5 单因子对照的同一交易定义：方向 = 开发期 IC 符号 × 过去 20 日标准化值，|z| 前 2%、每日最多 6 笔、
间隔 ≥ 2 分钟，主出场 (a) 兜底 15 分钟、反转周期 2 分钟），在 t 收盘价 / t+1 / t+2 / t+3 / t+6 开盘入场下的验证期账户 B 扣费后每笔。
交易与各入场点的成交价全部取自夜间研究落盘的标签（labels_*.parquet 的 L4a15_* 列），选点用夜间研究同一代码（zr.research.timing.composite）。
运行：python -m zr.research.timing.entry_timing_readout → docs/research/night_20261111/ENTRY_TIMING_VAL.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.research.timing.composite import pick, score, select, zscore_past
from zr.research.timing.data import net_pct
from zr.research.timing.stats import fmt_ci, pooled

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "research" / "night_20261111" / "ENTRY_TIMING_VAL.md"
KEYS = (("c0", "t 收盘价（做不到）"), ("e1", "t+1 开盘（主）"), ("e2", "t+2 开盘"), ("e3", "t+3 开盘"), ("e6", "t+6 开盘"))
FACTOR = "p1_rsi14_chg1"
NIGHT_N = {("QQQ", "合成 2%"): 521, ("SPY", "合成 2%"): 444}  # 夜间 COMPOSITE.md 验证期主档笔数（核对用）


def curve_net(lab: pd.DataFrame, idx: np.ndarray, d: np.ndarray, mk: str) -> dict:
    day = lab["day"].to_numpy()[idx]
    out = {}
    for key, _ in KEYS:
        if key == "e1":
            ent = np.where(d > 0, lab["entry_long"].to_numpy()[idx], lab["entry_short"].to_numpy()[idx])
            ex = np.where(d > 0, lab[f"L4a15_p2_{mk}_long_x"].to_numpy()[idx], lab[f"L4a15_p2_{mk}_short_x"].to_numpy()[idx])
        else:
            ent = np.where(d > 0, lab[f"L4a15_{key}_{mk}_long_entry"].to_numpy()[idx], lab[f"L4a15_{key}_{mk}_short_entry"].to_numpy()[idx])
            ex = np.where(d > 0, lab[f"L4a15_{key}_{mk}_long_x"].to_numpy()[idx], lab[f"L4a15_{key}_{mk}_short_x"].to_numpy()[idx])
        nb, _ = net_pct(ent, ex, d, "B")
        out[key] = pooled(nb, day)
    return out


def main() -> int:
    root = get_settings().zr_data_root
    data = Path(root) / "research" / "night_20261111" / "data"
    T = pd.read_parquet(data / "s3_all.parquet")
    mc = json.loads((data / "m_choice.json").read_text())
    rows, checks = [], []
    for und in ("QQQ", "SPY"):
        lab = pd.read_parquet(data / f"labels_{und}.parquet")
        X = pd.read_parquet(data / f"feat_{und}.parquet")
        split = lab["split"].to_numpy()
        day, mod = lab["day"].to_numpy(), lab["mod"].to_numpy()
        dev, val = split == "dev", split == "val"
        mk = f"{mc[und] * 100:.2f}"
        fac = select(T, X, dev, und)
        sc = score(X, day, fac)
        ic1 = float(T[(T["und"] == und) & (T["feature"] == FACTOR) & (T["h"] == 15)]["ic"].iloc[0])
        sc1 = np.sign(ic1) * zscore_past(X[FACTOR], day)
        for name, s in (("合成 2%", sc), (f"单因子 {FACTOR}（2%）", sc1)):
            cut = float(np.nanquantile(np.abs(s[dev]), 0.98))
            idx = pick(s, cut, day, mod, val)
            d = np.sign(s[idx]).astype(int)
            cv = curve_net(lab, idx, d, mk)
            if (und, name) in NIGHT_N:
                checks.append(f"{und} {name}：重选 {len(idx)} 笔，夜间 {NIGHT_N[(und, name)]} 笔（{'一致' if len(idx) == NIGHT_N[(und, name)] else '不一致'}）")
            row = {"标的": und, "规则": name + ("（因子：" + "、".join(f for f, _, _ in fac) + "）" if name.startswith("合成") else ""), "验证期笔数 / 天数": f"{cv['e1']['n']} / {cv['e1']['days']}"}
            for key, lab_ in KEYS:
                row[lab_] = fmt_ci(cv[key], 4)
            row["t+1 → t+2 差"] = f"{cv['e2']['mean'] - cv['e1']['mean']:+.4f}%"
            row["_e2"] = cv["e2"]
            rows.append(row)
    from zr.ops.markdown import df_to_md

    alive = [r for r in rows if r["_e2"]["ci_lo"] > 0]
    one = ("**一句话：优势在 t+2 已不在**——四条规则在 t+2 开盘入场的验证期账户 B 扣费后每笔均值的 95% 区间都包含 0 或整体为负（见表），连 t+1 主口径本身也没有正的优势。"
           if not alive else "**一句话：** " + "、".join(f"{r['标的']} {r['规则']}" for r in alive) + " 在 t+2 入场的 95% 区间仍在 0 以上。")
    L = ["# 入场时点曲线：验证期、账户 B（只读夜间研究已有结果，不新增检验）", "",
         "表头：QQQ / SPY；验证期 2026-01-01～2026-05-31；1 分钟 K（口径 C）；账户 B（10 万美元，满仓整数股、固定初始权益）扣费后每笔（D-013 + 每边 1 tick）；"
         "主出场 (a) 兜底 15 分钟（反转周期 2 分钟），各入场点持有长度相同；区间 = 按交易日聚类 95%（括号内按日整块重抽样）。"
         f"排名最前的 1–2 分钟反转因子 = {FACTOR}（阶段 3：SPY h = 2 IC −0.0189、t −8.37，也是 QQQ 在 1–2 分钟的第一名），按阶段 5 单因子对照的同一交易定义（方向取开发期 15 分钟 IC 符号）。", "",
         one, "", df_to_md(pd.DataFrame([{k: v for k, v in r.items() if k != "_e2"} for r in rows])), "",
         "选点核对：" + "；".join(checks) + "。", "",
         "边界：只用夜间研究已落盘的标签与特征；单因子按 15 分钟出场规则交易，而它的信息集中在 1–2 分钟（阶段 3），这里只是把夜间研究已有的交易定义按入场点读出来，不是对该因子的新检验。", ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(one)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
