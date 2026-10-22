"""方法论 v3 第一轮报告 → docs/research/METHOD_V3_ROUND1.md。

章节：覆盖矩阵；状态层人工核对（MU 2026-10-21）；QQQ 修复摘要；规则卡（MU 主、QQQ 对照）；
157 笔按状态分类与提取规则回放；模型与规则提取；边界声明。
前置（各自写出中间结果）：
  uv run python -m zr.research.qqq_artifact
  uv run python -m zr.research.grid_v3 --symbol MU.US ；--symbol QQQ.US
  uv run python -m zr.research.own_policy
  uv run python -m zr.research.model_v3 --symbol MU.US ；--symbol QQQ.US
运行：uv run python -m zr.research.method_v3_round1
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features.bars import ET, load_extended_bars, resample_bars
from zr.features.regime import STATE_NAMES, indicators, load_config, regimes
from zr.ops.markdown import df_to_md
from zr.research import grid_v3
from zr.research import patterns_v3 as pt
from zr.research.method_v2_round1 import coverage

REPO = Path(__file__).resolve().parents[2]
CHECK_DAY = dt.date(2026, 10, 21)
CHECKS = [  # (起, 止, 期望状态, 期望方向, 说明)
    ("09:30", "09:44", "IGNITION", -1, "09:30 后急跌应为 IGNITION 下行"),
    ("10:00", "12:30", "RANGE", 0, "10:00–12:30 应为 RANGE"),
    ("12:18", "13:36", "CHANNEL", 1, "12:18–13:36 应为 CHANNEL 上行"),
]


def regime_check(root: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    s = load_extended_bars(root, "MU", CHECK_DAY - dt.timedelta(days=20), CHECK_DAY)
    b = resample_bars(s, "2min")
    x = indicators(b, cfg)
    reg = regimes(b, x, cfg, 2)
    et = b.index.tz_convert(ET)
    on = np.asarray(et.date) == CHECK_DAY
    hhmm = np.asarray(et.strftime("%H:%M"))
    rows = []
    for a, z, st, d, note in CHECKS:
        m = on & (hhmm >= a) & (hhmm <= z)
        names = [STATE_NAMES[v] for v in reg["state"].to_numpy()[m]]
        dirs = reg["dir"].to_numpy()[m]
        hit = np.mean(
            [(n == st and (d == 0 or di == d)) for n, di in zip(names, dirs, strict=True)]
        )
        dist = (
            pd.Series(
                [
                    f"{n}{'' if di == 0 else ('↑' if di > 0 else '↓')}"
                    for n, di in zip(names, dirs, strict=True)
                ]
            )
            .value_counts()
            .to_dict()
        )
        rows.append(
            {
                "核对点": note,
                "K 线数": int(m.sum()),
                "符合比例": round(float(hit), 2),
                "实际分布": ", ".join(f"{k} {v}" for k, v in dist.items()),
            }
        )
    # 不符原因的量化；跳空口径用常规定义（RTH 开盘 vs 前一 RTH 收盘），不用盘前/盘后价
    rth = s[s["session"] == "Intraday"] if "session" in s else s
    rth_day = np.asarray(rth.index.tz_convert(ET).date)
    prev_close = float(rth["close"][rth_day < CHECK_DAY].iloc[-1])
    day_open = float(rth["open"][rth_day == CHECK_DAY].iloc[0])
    m1 = on & (hhmm >= "09:30") & (hhmm <= "09:44")
    m2 = on & (hhmm >= "10:00") & (hhmm <= "12:30")
    score = reg["range_score"].to_numpy()
    why = {
        "gap_pct": round((day_open / prev_close - 1) * 100, 2),
        "pctb_min_open": round(float(np.nanmin(x["pctb"].to_numpy()[m1])), 3),
        "range_score_median": round(float(np.nanmedian(score[m2])), 3),
        "range_score_threshold": cfg["regime"]["range"]["score_threshold"],
        "range_share_le_threshold": round(
            float((score[m2] <= cfg["regime"]["range"]["score_threshold"]).mean()), 3
        ),
    }
    # E1 回踩核对：2 分钟 K 最低 913.37、中轨 912.69
    lows = b["low"].to_numpy()
    k = np.flatnonzero(on & (np.abs(lows - 913.37) < 0.006))
    txt = "未找到最低 913.37 的 2 分钟 K。"
    if len(k):
        i = int(k[0])
        fired = []
        for c in pt.entry_configs(cfg):
            if c["entry"] != "E1":
                continue
            sig = pt.entry_signal(b, x, reg, c, cfg)
            if sig[i] == 1:
                fired.append(pt.entry_name(c))
        pctb = (lows[i] - x["lower"].iat[i]) / (2 * x["hw"].iat[i])
        dist_hw = (lows[i] - x["mid"].iat[i]) / x["hw"].iat[i]
        txt = (
            f"{hhmm[i]} 的 2 分钟 K：最低 {lows[i]:.2f}、中轨 {x['mid'].iat[i]:.2f}、带半宽 "
            f"{x['hw'].iat[i]:.2f}，回踩低点 %b {pctb:.3f}、距中轨 {dist_hw:+.3f} 倍半宽；状态 "
            f"{STATE_NAMES[int(reg['state'].iat[i])]}（方向 {int(reg['dir'].iat[i]):+d}）。E1 "
            "识别为回踩的定义 "
            f"{len(fired)} / 36 个：{', '.join(fired) if fired else '无'}。"
        )
    return pd.DataFrame(rows), txt, why


def artifact_summary() -> pd.DataFrame:
    md = (REPO / "docs" / "research" / "QQQ_ARTIFACT.md").read_text(encoding="utf-8")
    rows = []
    for und in ("QQQ", "MU"):
        sec = md.split(f"## {und} 明细")[1].split("## ")[0] if f"## {und} 明细" in md else ""
        for v in ("V0 原口径", "stop_lag", "ALL"):
            m = re.search(
                rf"^\| {re.escape(v)} \| 30 \| ([-\d.]+) \| ([-\d.]+) \| (\d+) \|", sec, re.M
            )
            if m:
                rows.append(
                    {
                        "标的": und,
                        "口径": v,
                        "put（dir −1）": float(m[1]),
                        "call（dir +1）": float(m[2]),
                        "样本": int(m[3]),
                    }
                )
    return pd.DataFrame(rows)


def cards_section(root: Path, und: str, k: int = 10) -> list[str]:
    p = root / "research" / "v3" / f"cards_{und}.parquet"
    g = root / "research" / "v3" / f"grid_{und}.parquet"
    if not g.exists():
        return [f"待 {und} 网格运行。", ""]
    lt = pd.read_parquet(g)
    cards = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    tc = grid_v3.trial_counts(load_config())
    from scipy.stats import norm

    tb = float(norm.ppf(1 - 0.05 / (2 * tc["rows_total"])))
    L = [
        f"网格 {len(lt)} 行（试验计数 {tc['rows_total']}，Bonferroni |t| > {tb:.2f}）；"
        "扣费后均值 > 0 的行 "
        f"{int((lt['mean_net'] > 0).sum())}，其中 n ≥ 100 的 "
        f"{int(((lt['mean_net'] > 0) & (lt['n'] >= 100)).sum())}；"
        f"规则卡 {len(cards)} 张"
        + (
            f"，涉及入场定义 {cards['entry'].nunique()} 个；"
            f"过 Bonferroni {int((cards['p_bonf'] < 0.05).sum())} 张；"
            f"同时段随机分位 ≥ 95 的 {int((cards['rand_pctile'] >= 95).sum())} 张。"
            ""
            if len(cards)
            else "。"
        ),
        "",
    ]
    if len(cards):
        L += [
            f"前 {min(k, len(cards))} 张（按每张均值）：",
            "",
            df_to_md(
                cards.sort_values("mean_per_contract", ascending=False)
                .head(k)[grid_v3.CARD_COLS]
                .round(3)
            ),
            "",
        ]
    L += [
        "全部网格按 周期 × 形态 × 方向 汇总（含为负的）：",
        "",
        df_to_md(grid_v3.negatives_summary(lt)),
        "",
    ]
    return L


def append_model_rules(root: Path) -> None:
    """模型提取规则并入 RULE_CARDS_<UND>.md 附录（可重复运行，覆盖旧附录）。"""
    tag = "## 附录：模型样本外最高十分位提取的规则（第 8 节）"
    for und in ("MU", "QQQ"):
        card_md = REPO / "docs" / "research" / f"RULE_CARDS_{und}.md"
        rp = root / "research" / "v3" / f"model_rules_{und}.parquet"
        if not card_md.exists():
            continue
        text = card_md.read_text(encoding="utf-8").split(tag)[0].rstrip() + "\n\n"
        body = (
            df_to_md(pd.read_parquet(rp))
            if rp.exists() and len(pd.read_parquet(rp))
            else "待模型运行或无满足条件的叶子。"
        )
        text += (
            f"{tag}\n\n{body}\n\n规则中的缺失值按 −999 参与比较。只作描述："
            "规则在样本外月份上提取并评估，"
            "不满足规则卡的入选条件时不计为卡。\n"
        )
        card_md.write_text(text, encoding="utf-8")


def main() -> int:
    cfg = load_config()
    root = get_settings().zr_data_root
    append_model_rules(root)
    cov = coverage(root)
    chk, e1txt, why = regime_check(root, cfg)
    art = artifact_summary()
    own = (REPO / "docs" / "research" / "OWN_POLICY_EXTRACTED.md").read_text(encoding="utf-8")
    own = own.split("\n", 3)[3] if own.count("\n") > 3 else own
    L = [
        "# 方法论 v3 第一轮（MU 主、QQQ 对照）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 规格：负责人 "
        "2026-10-22 指令"
        "（METHOD_V3 第 0–9 节）· 决策 D-023、D-024 · 预登记 R-030～R-032",
        "",
        "## 0. 覆盖矩阵",
        "",
        df_to_md(cov),
        "",
        "边界：SPY 期权拉取暂停于 2024-07-30，本轮未覆盖 SPY；MU、QQQ 期权 2023-10-02～2026-10-21 "
        "全量。",
        "",
        "## 1. 状态层人工核对（MU 2026-10-21，2 分钟 K，MACD(6,13,6)、BOLL(20,2)，"
        "fix/lookahead-audit 修复后：含盘前盘后连续 K 线 + RANGE 打分制 D-026）",
        "",
        df_to_md(chk),
        "",
        e1txt,
        "",
        f"第 1 点（09:30 后急跌）符合比例 {chk['符合比例'].iloc[0]:.2f}：MU 当日开盘相对前收 "
        f"{why['gap_pct']:+.2f}%，09:30–09:44 %b 最低 {why['pctb_min_open']}（含盘前盘后连续带，"
        f"不再是「前一交易日 RTH 收盘沿用」的带）；IGNITION 下行阈值 "
        f"pctb_lo={cfg['regime']['ignition']['pctb_lo']}。",
        "",
        f"第 2 点（10:00–12:30 RANGE）符合比例 {chk['符合比例'].iloc[1]:.2f}：该时段 RANGE 打分"
        f"（D-026，带宽/|斜率|/|MACD 柱| 百分位秩加权和）中位数 {why['range_score_median']}，"
        f"阈值 {why['range_score_threshold']}（MU 2024 年标定，覆盖率目标 [20%,40%]），"
        f"≤ 阈值的比例 {why['range_share_le_threshold']}。",
        "",
        f"第 3 点（12:18–13:36 CHANNEL 上行）符合比例 {chk['符合比例'].iloc[2]:.2f}。",
        "",
        "边界：以上三项数字均由本次重算的 `reg`/`x` 直接读出；若符合比例 < 1，说明修复后仍有残余"
        "差异，不代表「定义错误」，只代表这一组参数（slope_n/side_m/score_threshold 等）下的"
        "分类结果；已写入 STATUS。",
        "",
        "## 2. QQQ 修复摘要（第 0 节，docs/research/QQQ_ARTIFACT.md）",
        "",
        "根因：v2 结构止损按正股 1 分钟 bar j 的收盘判定，却用时间戳 j（bar 开始时）的期权 bid "
        "成交，"
        "提前一分钟离场。修复后一切由 K 线收盘触发的出场用触发后第一个快照成交（D-023）。"
        "随机分钟入场（持仓上限 30，每 5 个期权日抽 1 天）扣费后 USD / 笔：",
        "",
        df_to_md(art),
        "",
        "QQQ 两方向由正转为约 −9 ～ −11；MU 亏损约翻倍。方法论 v2 "
        "第一轮的期权盈亏数字全部作废为绝对值结论。",
        "",
        "## 3. 规则卡（第 6 节）",
        "",
        "### MU",
        "",
    ]
    L += cards_section(root, "MU")
    L += ["### QQQ（对照）", ""]
    L += cards_section(root, "QQQ")
    L += [
        "边界：真实期权 1 分钟 NBBO；D-023 成交口径；2 张；D-013 费用；D-001；15:50 清仓；"
        "同标的同时一笔；"
        "「样本外 2026」是同一网格里按年份拆分的子样本，不是独立前瞻验证。完整卡片见 "
        "docs/research/RULE_CARDS_MU.md、RULE_CARDS_QQQ.md。",
        "",
        "## 4. 负责人 157 笔：按状态分类与提取规则回放（第 7 节）",
        "",
        own,
        "",
    ]
    for und in ("MU", "QQQ"):
        p = root / "research" / "v3" / f"model_{und}.md"
        L += [
            p.read_text(encoding="utf-8") if p.exists() else f"## 模型（{und}）\n\n待运行。\n",
            "",
        ]
    L += [
        "## 边界声明（汇总）",
        "",
        "- 标的：MU（主）、QQQ（对照）；SPY 未覆盖。期权 2023-10-02～2026-10-21，767 个交易日。",
        "- 口径：D-023（入场 +2 秒快照、离场触发后快照、bid=0 为 0、X6 按 mid 判定下一快照 bid "
        "成交）；"
        "2 张；D-013 费用；D-001；15:50 清仓。",
        "- 状态层与形态的阈值见 config/method_v3.yaml；实现方确定的口径见 D-024。",
        "- 多重比较：试验计数 127,872 行（去重后），Bonferroni 门槛见第 3 节。",
        "- 负责人订单只覆盖 2026-06～08 的 20 多个交易日；提取规则的学习期包含在回放期内。",
        "- 未覆盖：盘前盘后数据、DTE > 5 合约、SPY、秒级入场时机、神经网络（登记为备选）。",
        "",
    ]
    (REPO / "docs" / "research" / "METHOD_V3_ROUND1.md").write_text("\n".join(L), encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
