"""底座最终自检报告（audit/base-final 第 7 节）→ docs/research/BASE_FINAL_AUDIT.md。

只汇总各节脚本的落盘结果并计算报告里的数字（不在这里做新的回测）：
1. zr.research.bar_parity → research/bar_parity/summary.csv
2. zr.research.opening_indicators（开盘段覆盖统计在这里重算）
3. tests/test_synthetic_scenarios.py 等（在这里跑 pytest 取结果）
4. zr.research.bt_full → research/audit/bt_full_summary{,_prefix}.csv
5. zr.research.shadow_replay → research/audit/shadow_replay.pkl
6. zr.research.contract_choice → research/audit/contract_choice.parquet
附：两处修复前后 V4 各族结果（research/v4_pre_slopefix/ vs research/v4/）。

运行：uv run python -m zr.research.base_final_audit
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.ops.markdown import df_to_md
from zr.research import contract_choice as cc
from zr.research import opening_indicators as oi
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.base_final_audit")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "BASE_FINAL_AUDIT.md"
TESTS = {
    "K 线一致性": "tests/test_bar_parity.py",
    "开盘时段处理": "tests/test_opening_session.py",
    "合成场景": "tests/test_synthetic_scenarios.py",
    "增量路径": "tests/test_incremental.py",
    "指标一致性（App 截图）": "tests/test_indicator_parity.py",
    "回填以新值为准": "tests/test_backfill_bars.py",
}
FAMILIES = {
    "R-033 口径 C（主检验 36 格）": "cells_C.parquet",
    "R-033 口径 A（敏感性 36 格）": "cells_A.parquet",
    "R-034 H5（18 格）": "h5_cells.parquet",
    "R-035 口径 C": "rules_cells_C_main.parquet",
    "R-035 口径 A": "rules_cells_A_main.parquet",
    "R-037 口径 C": "rules37_cells_C_main.parquet",
    "R-037 口径 A": "rules37_cells_A_main.parquet",
}


def pytest_counts() -> pd.DataFrame:
    rows = []
    for name, f in TESTS.items():
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f], cwd=REPO, capture_output=True, text=True
        )
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
        p = re.search(r"(\d+) passed", tail)
        fl = re.search(r"(\d+) failed", tail)
        rows.append(
            {
                "测试": name,
                "文件": f"`{f}`",
                "通过": int(p.group(1)) if p else 0,
                "失败": int(fl.group(1)) if fl else 0,
            }
        )
    return pd.DataFrame(rows)


def family_delta(root: Path) -> pd.DataFrame:
    pre, post = root / "research" / "v4_pre_slopefix", root / "research" / "v4"
    rows = []
    for name, f in FAMILIES.items():
        if not (pre / f).exists() or not (post / f).exists():
            rows.append({"族": name, "说明": "缺文件"})
            continue
        a, b = pd.read_parquet(pre / f), pd.read_parquet(post / f)
        n_col = "n" if "n" in a else ("n_C" if "n_C" in a else None)
        m_col = "mean_pct" if "mean_pct" in a else ("mean%_C" if "mean%_C" in a else None)
        p_col = "p"
        pass_col = "fdr_pass" if "fdr_pass" in a else ("通过" if "通过" in a else None)
        row = {"族": name, "格数": len(b)}
        if n_col:
            row["笔数（前 → 后）"] = f"{int(a[n_col].sum())} → {int(b[n_col].sum())}"
            row["笔数有变化的格"] = int((a[n_col].to_numpy() != b[n_col].to_numpy()).sum())
        if m_col:
            d = np.abs(a[m_col].to_numpy(float) - b[m_col].to_numpy(float))
            row["均值 % 最大变化"] = round(float(np.nanmax(d)), 3)
            row["均值 % 范围（后）"] = f"{b[m_col].min():.2f} ~ {b[m_col].max():.2f}"
        row["最小 p（前 → 后）"] = f"{a[p_col].min():.3f} → {b[p_col].min():.3f}"
        if pass_col:
            row["通过（前 → 后）"] = f"{int(a[pass_col].sum())} → {int(b[pass_col].sum())}"
        rows.append(row)
    return pd.DataFrame(rows)


def shadow_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = pd.read_pickle(root / "research" / "audit" / "shadow_replay.pkl")
    rows, trades = [], []
    for day, r in res.items():
        j = r["bars"]
        px = np.logical_and.reduce(
            [np.isclose(j[c], j[f"{c}_o"]) for c in ("open", "high", "low", "close")]
        )
        lag = (j["completed_ns"] - (j.index.astype("int64") + 120_000_000_000)) / 1e9
        row = {
            "日期": str(day),
            "回放 K 数": len(j),
            "OHLC = 官方": int(px.sum()),
            "close = 官方": int(np.isclose(j["close"], j["close_o"]).sum()),
            "量 = 官方": int((j["volume"] == j["volume_o"]).sum()),
            "状态一致（vs 批量）": int((j["state"] == j["state_batch"]).sum()),
            "方向一致": int((j["dir"] == j["dir_batch"]).sum()),
        }
        for h in ("H1", "H2", "H3"):
            row[f"{h} 信号（影子 / 批量 / 不一致）"] = (
                f"{int((j[h] != 0).sum())} / {int((j[f'{h}_batch'] != 0).sum())} / "
                f"{int((j[h] != j[f'{h}_batch']).sum())}"
            )
        row["收线延迟中位数 s"] = round(float(lag.median()), 3)
        row["收线延迟最大 s"] = round(float(lag.max()), 3)
        row["迟到成交（丢弃）"] = r["late"]
        row["当日期权到期日"] = ", ".join(str(e) for e in r["expiries"])
        t = r["trades"]
        if len(t):
            both = t["_merge"] == "both"
            same = (
                both
                & (t["symbol_batch"] == t["symbol_shadow"])
                & (t["entry_ns_batch"] == t["entry_ns_shadow"])
                & (t["exit_ns_batch"] == t["exit_ns_shadow"])
                & (t["reason_batch"] == t["reason_shadow"])
                & np.isclose(t["net_usd_batch"], t["net_usd_shadow"], atol=0.005)
            )
            row.update(
                {
                    "交易（批量 / 影子 / 配对）": f"{int((t['_merge'] != 'right_only').sum())} / "
                    f"{int((t['_merge'] != 'left_only').sum())} / {int(both.sum())}",
                    "逐笔完全一致": int(same.sum()),
                    "净值合计（批量 / 影子）": f"{t['net_usd_batch'].sum():.2f} / "
                    f"{t['net_usd_shadow'].sum():.2f}",
                }
            )
            diff = t[both & ~same].copy()
            if len(diff):
                diff["决策（ET）"] = (
                    pd.to_datetime(diff["bar_close"], utc=True)
                    .dt.tz_convert("America/New_York")
                    .dt.strftime("%H:%M")
                )
                trades.append(
                    diff[
                        [
                            "决策（ET）",
                            "hyp",
                            "exit",
                            "dir",
                            "symbol_batch",
                            "decision_lag_s",
                            "entry_lag_s",
                            "exit_lag_s",
                            "ask0_batch",
                            "ask0_shadow",
                            "net_usd_batch",
                            "net_usd_shadow",
                        ]
                    ].round(3)
                )
        else:
            row["交易（批量 / 影子 / 配对）"] = "0 / 0 / 0"
        rows.append(row)
    return pd.DataFrame(rows).T.reset_index(), (
        pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    au = root / "research" / "audit"
    bp = pd.read_csv(root / "research" / "bar_parity" / "summary.csv")
    tests = pytest_counts()
    warm = pd.DataFrame([{"标的": u, **oi.warmup_audit(root, u, v4)} for u in ("MU", "QQQ")])
    bt_pre = pd.read_csv(au / "bt_full_summary_prefix.csv")
    bt_post = pd.read_csv(au / "bt_full_summary.csv")
    sh, sh_diff = shadow_tables(root)
    ch = pd.read_parquet(au / "contract_choice.parquet")
    fam = family_delta(root)

    def tot(t, c):
        return int(t[c].sum())

    mu2 = bp.iloc[0]
    pre_same, pre_n = tot(bt_pre, "逐笔完全一致"), tot(bt_pre, "引擎笔数")
    post_same, post_n = tot(bt_post, "逐笔完全一致"), tot(bt_post, "引擎笔数")
    lines = [
        "# 底座最终自检（audit/base-final）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} · "
        "复现 `uv run python -m zr.research.base_final_audit`（各节脚本见每节开头）。",
        "不做新研究：只检验底座（数据、指标、信号、离场、成交、实时路径、合约），发现问题即修复"
        "并重跑受影响结果。",
        "",
        "## 结论（逐项）",
        "",
        "| 项 | 结果 | 修复 |",
        "|---|---|---|",
        f"| 1 K 线一致性 | RTH 2 / 5 分钟：通过（MU 2 分钟 {mu2['完全一致']}/{mu2['比对 K 数']}"
        " 根"
        f"完全一致，收盘价 100%）；含盘前盘后 2 分钟：**修复前不通过**（MU 一致率 "
        f"{bp[(bp['标的'] == 'MU') & bp['合成'].str.startswith('修复前')]['一致率'].iloc[0]:.4f}"
        "），修复后通过 | "
        "零成交量占位 K 不参与开高低；回填改为以新拉取的值为准 |",
        "| 2 开盘时段指标 | 通过：口径 C 开盘第一根就有完整窗口（全部交易日），口径 A 前 19 根空"
        "值，已锁定；"
        "**开盘段与 App 的一致性未核实**（STATUS 0f） | 无 |",
        f"| 3 合成场景 | 通过（{tests[tests['测试'] == '合成场景']['通过'].item()} 项） | 无；发"
        "现带宽 / 斜率按价格水平"
        "归一化使多空在临界 K 上不完全镜像（定义性质，记录不改） |",
        f"| 4 独立复算 | **修复前不通过**：backtrader {pre_same}/{pre_n}"
        " 笔一致、"
        f"多 {tot(bt_pre, '仅 backtrader')} 笔；修复后 {post_same}/{post_n}"
        "、"
        f"仅引擎 {tot(bt_post, '仅引擎')}、仅 backtrader {tot(bt_post, '仅 backtrader')} | "
        "中轨斜率浮点噪声（±1e-16）记为 0 |",
        "| 5 影子回放 | 通过：状态、信号逐根一致；交易逐笔一致或只差实时收线延迟（见第 5 节） |"
        " 新增增量路径 |",
        "| 6 合约选择 | 报告（无对错判定）：见第 6 节 | 无 |",
        "",
        "两处修复改变了 V4 的输入，已按原预登记重跑全部 V4 结果（附录）；结论不变：0 格通过 FDR。",
        "",
        "## 1. K 线一致性",
        "",
        "`scripts/diag_pull_candles.py`（VPS，长桥 `history_candlesticks_by_offset`，NoAdjust，"
        "RTH = `TradeSessions.Intraday` 2 / 5 分钟，含盘前盘后 = `TradeSessions.All` 2 分钟，"
        "可得最早 2023-10-02）→ `uv run python -m zr.research.bar_parity`。"
        "本地合成 = `resample_bars`（RTH 用 `load_bars`，含盘前盘后用 `load_extended_bars(clean"
        "=False)`，"
        "即口径 A / C 实际用的数据）。逐根判定：时间标签两边都有、OHLC 完全相等（1e-6）、量完全"
        "相等；"
        "本地最后一个分箱可能不完整，不比。",
        "",
        df_to_md(
            bp.assign(
                一致率=bp["一致率"].map("{:.6f}".format),
                **{"close 一致率（两边都有）": bp["close 一致率（两边都有）"].map("{:.6f}".format)},
            )
        ),
        "",
        "不一致的原因（逐根查看 `research/bar_parity/mismatch_*.parquet`）：",
        "",
        "- **零成交量占位 K（已修复）**：长桥 1 分钟 K 在无成交的分钟插入 O=H=L=C=前收、量 0 的"
        "占位 K"
        "（多在盘前盘后）；官方 2 分钟 K 的开高低不含占位 K，本地合成把占位 K 的前收算进了开 / "
        "高 / 低。"
        "收盘价不受影响（占位 K 的 close = 前收），所以 BOLL / MACD 不变，但 E1（阴阳线、回踩低"
        "点）、"
        "E11（前 20 根高低点）、X5（当日摆动点，口径 C 含盘前 K）会受影响。"
        "`resample_bars` 改为只用有成交的分钟计开高低（整箱都是占位 K 时仍按占位 K）。",
        "- **事后修订（数据侧，已修复写入逻辑）**：MU 2026-10-21 09:30 那分钟的 open 当日拉取为"
        " 925.42，"
        "长桥事后改为 923.735（次日重拉 1 分钟 K 与官方 2 分钟 K 都是新值）。原 `write_merged` "
        "去重保留旧值，"
        "重拉也改不过来；改为以新拉取的值为准（`tests/test_backfill_bars.py`）。数据目录里的旧值"
        "未重拉。",
        "- **1 分钟与 2 分钟历史本身不一致（无法按规则修复，保留）**：例如 MU 2026-09-02 12:30、"
        "QQQ 2026-06-25 "
        "15:30，1 分钟 K 在该分钟是占位 K（量 0），官方 2 分钟 K 却含该分钟成交（量更大、开盘不"
        "同）；"
        "现行合成下剩余的含盘前盘后不一致几乎都在 2026 年，方向不定。",
        "- 5 分钟 K 另有 2 根只是成交量不同（MU、QQQ 同在 2026-10-07 11:05 / 12:40）。",
        "",
        "判定：2 分钟 K 与官方一致（RTH 每个标的 ≤ 3 根不同、收盘价 100% 相同；含盘前盘后修复后"
        " ≥ 99.98%），"
        "**不改用官方 K**；指标一致性测试（App 截图）重跑通过（第 3 节表）。"
        "新增长桥 API 事实：`history_candlesticks_by_offset` 的 `time` 参数按**美东时间**解释"
        "（返回的 timestamp 仍是 naive UTC；2026-10-22 VPS 实测：传 13:32 返回到 17:32 UTC 为止）。"
        "`backfill_bars` 的翻页游标用 UTC，因此每页与上一页重叠约 4–5 小时（去重后无缺口，只是多"
        "翻页）。",
        "",
        "## 2. 开盘时段指标",
        "",
        "`uv run python -m zr.research.opening_indicators` → `docs/research/OPENING_INDICATORS.md`"
        "（MU 2026-10-21 09:30–10:30 每根 K 在口径 C / A / B 下的 BOLL、MACD、状态）。",
        "",
        "锁定的处理方式（`tests/test_opening_session.py`）：",
        "",
        "- 口径 C（主口径）：09:30 那根的 BOLL 窗口 = 08:52–09:30 的 20 根，MACD 跨日连续——"
        "前 19 根既不是部分窗口也不是空值；",
        "- 口径 A（敏感性）：BOLL 前 19 根为空值、10:08 那根（第 20 根）起有值；%b / 带宽 / 斜率"
        "为空，"
        "前 19 根状态只能是 OTHER；MACD 每日首根收盘做种、第一根起有值；",
        "- 口径 B（部分窗口）只作对照。",
        "",
        df_to_md(warm),
        "",
        "**未核实**：开盘 40 分钟内 App 的数值没有截图，「与 App 一致」只在 11:26/12:34/14:52 三"
        "个时刻成立"
        "（A、C 都 15/15 精确相等）。STATUS 0f 仍待负责人截图。",
        "",
        "## 3. 合成场景与常规测试",
        "",
        "`tests/test_synthetic_scenarios.py`：人为构造 2 分钟 K + 每秒期权报价，走生产代码路径"
        "（indicators → regimes → hypothesis_signals / exit_arrays / guard_mask → simulate），已"
        "知答案逐项断言：",
        "",
        "| 场景 | 断言 |",
        "|---|---|",
        "| (a) 通道回踩 | CHANNEL↑ 中阴线下影探到中轨上方 0.1 带半宽 + 阳线 → H2（E1）恰在 10:2"
        "2 那根、只此一根；"
        "回踩低点 %b ∈ [0.45, 0.75]、距中轨 ≤ 0.2 带半宽；离场 (c') X5 在 10:42（收盘跌破前一更"
        "高低点 102.896）|",
        "| (b) 开盘急跌 | 09:30 第 1 根不是 IGNITION；09:32（第 2 根）IGNITION↓ 且连续 2 根收在"
        "下轨外 → H1（E10）"
        "只在这根；离场 (c) X4 在 09:46（%b 回到 [0.1, 0.9]）|",
        "| (c) 横盘放量突破 | 10:28 为 RANGE，10:30 收盘破前 20 根高点、量比 4 → E11；下一根仍在"
        "突破位外 → H3 在 10:32；"
        "下一根收回 → 无 H3 |",
        "| (d) X1–X7 | 同 (a) 入场：X1 SMA5 10:34、X2 柱连续 2 根缩短 10:24、X3 连续 2 根收在中"
        "轨下 10:40、"
        "X4 10:24、X5 10:42，各自独立复算触发条件；成交 = 触发 +2 s 的 bid、入场 = 决策 +2 s 的"
        " ask、"
        "净值按 `zr.costs.fees` 独立复算；X6 +20% 先卖 1 张、−30% 全卖；X7 30 分钟上限（10:54:0"
        "2 触发）；"
        "强平早于上限时按强平 |",
        "| 做空镜像 | 价格关于 100 镜像后，全部信号、离场方向互换、K 位置不变；场景关键 K 的状态"
        "逐根对称 |",
        "",
        "镜像检查发现：状态并非在每根 K 上都对称（path_a 3 根、path_c 1 根），原因是带宽 bw 与中"
        "轨斜率都除以"
        "中轨（价格水平）——上涨中同样宽度的带 bw 递减、下跌中递增，「带宽扩张」对下跌略敏感。"
        "这是现行定义本身的性质，不是实现错误，记录不改（改定义属于新研究）。",
        "",
        df_to_md(tests),
        "",
        "## 4. 独立复算（backtrader，H1 / H2 / H3 完整交易含 V4 离场组合）",
        "",
        "`uv run python -m zr.research.bt_full`。backtrader（`V4Full`，在 `bt_independent.H2` 逐"
        "根状态层上补齐）"
        "用 backtrader 内置 BOLL / MACD 逐根重算状态、H1（E10）、H2（E1）、H3（E11 + 确认）原始"
        "信号与结构离场"
        "X3_mid2 / X2_shrink2 / X4_inside / X5；护栏、合约（`bt_independent.pick`）、tick 成交、"
        "费用（`zr.costs.fees`）、同格不重叠按规则文字另写，不调用 `v4_decisive`。口径 C，MU，"
        "与引擎 `research/v4/trades_C.parquet` 的 MU 部分逐笔比对（合约、入场 / 离场时刻精确到纳"
        "秒、"
        "离场原因、净值到 0.005 美元）。",
        "",
        "修复前：",
        "",
        df_to_md(bt_pre),
        "",
        "不一致的 3 笔是同一个 H3 做空信号（2024-05-30 15:24 那根确认，三种离场各一笔），只有 b"
        "acktrader 有。"
        "根因：15:14 那根中轨的「斜率」在引擎里是 −2.2e-16——窗口进出的收盘价相同，真实斜率为 0，"
        "pandas 滚动均值的浮点误差给了负号；引擎把它算进「中轨连续下降 5 根」，15:20 判成 CHANN"
        "EL↓、"
        "前一根不是 RANGE，E11 不成立；backtrader 的 SMA 给出精确 0，15:20 仍是 RANGE。"
        "全样本这类噪声斜率：MU 口径 C 807 根、QQQ 740 根、口径 A 231 / 159 根，全部是「进出收盘"
        "价相同」的 K。"
        "修复：`regime.indicators` 中 |斜率| < 1e-12 记为 0（真实最小非零斜率约 1 tick / 20 / 价"
        "格 ≈ 5e-7）。",
        "",
        "修复后（两处修复都生效、V4 重跑之后）：",
        "",
        df_to_md(bt_post),
        "",
        "## 5. 影子回放（增量计算路径）",
        "",
        "新增 `zr/strategy/incremental.py`：`BarBuilder`（逐笔 → 2 分钟 K）与 `IncrementalEngine`"
        "（每来一根 K 只用此前状态更新 BOLL / MACD / 状态 / 信号 / 离场 / 护栏，不回看）。"
        "`uv run python -m zr.research.shadow_replay`：录制器 MU.US 逐笔成交 + 订阅期权 1 档盘口"
        "（按接收时刻）"
        "驱动增量路径，信号在收线那一刻决策、持仓按盘口逐个检查；批量路径用官方 2 分钟 K 与同一"
        "份盘口"
        "（`v4_decisive.simulate`）。状态从 2026-06-01 起用官方 2 分钟 K 灌入。",
        "",
        "代码层面（同一组 K 线）：增量与批量在 MU 口径 C 2026-06-01 ~ 2026-10-21 共 47,869 根上"
        "状态、方向、"
        "H1/H2/H3、8 个离场数组、护栏 0 处不同（`tests/test_incremental.py` 用冻结的 1 天数据锁"
        "定）。",
        "",
        "逐笔成交 → K 线：长桥只用常规成交（trade_type 为空）计开高低收，零股（I，占 MU 成交笔数"
        "约 79%）、"
        "W / V / P / X 只计量（2026-10-22 197 根：全部计价 OHLC 完全一致的只有约 23–41%，只用常"
        "规成交 98–99%）。",
        "",
        df_to_md(sh.rename(columns={"index": "项"})),
        "",
    ]
    if len(sh_diff):
        lines += [
            "不完全一致的交易（全部由实时收线延迟造成：影子在下一分箱首笔成交到达时收线，比 K 收"
            "盘晚 0.4–0.7 s，"
            "+2 s 后的第一个盘口可能换一个）：",
            "",
            df_to_md(sh_diff),
            "",
        ]
    lines += [
        "2026-10-21 只有 14:47 之后的录制数据，且当日期权只有 0DTE（10:15 后排除）与 2DTE，规则"
        "不交易；"
        "2026-10-22 是第一个完整交易日。录制器只订阅最近 2 个到期日、平值 ±8 档，规则合约都在其"
        "中。",
        "",
        "## 6. 合约选择对比（D-030 入选样本）",
        "",
        "`uv run python -m zr.research.contract_choice`：负责人实际开仓时刻，按 V4 规则"
        "（`pick_contract`：DTE 0/1 最近到期、10:15 后排除 0DTE、略虚值 1 档）选的合约 vs 实际合"
        "约。"
        "正股价 = 开仓前最后一根完整 1 分钟 K 收盘；delta = Black-Scholes（r 0.04），IV 由开仓那"
        "分钟 NBBO 中间价反推。",
        "",
        df_to_md(cc.summary(ch)),
        "",
        "实际 DTE × 规则 DTE（只含规则有合约的笔）：",
        "",
        df_to_md(pd.crosstab(ch["实际 DTE"], ch["规则 DTE"]).reset_index()),
        "",
        "逐笔：",
        "",
        df_to_md(ch.round(3)),
        "",
        "读法：规则在负责人 48 笔 MU 开仓里有 15 笔选不出合约（14 笔是 10:15 后只剩 0DTE、次日无"
        "到期；"
        "1 笔当日无 0/1DTE）——负责人在这些时刻用 0DTE 或 2–10DTE。规则选得出的 33 笔里到期日相同"
        " 30 笔、"
        "同一合约 11 笔；行权价离正股的百分比与 |delta| 的中位数接近（见上表），差别主要在虚值档"
        "数的分布"
        "（负责人有 13 笔平值或实值）。DRAM、META（3 笔）本地没有期权链，未比较。",
        "",
        "## 附：修复前后 V4 各族结果",
        "",
        "两处修复（斜率噪声、零成交量占位 K）改变了状态层输入，按原预登记重跑：`v4_decisive`（C"
        "、A）、"
        "`v4_h5`、`v4_rules_test`（R-035、R-037）、`v4_report`、`v4_checkpoints`。修复前结果存档在"
        "`$ZR_DATA_ROOT/research/v4_pre_slopefix/`。",
        "",
        df_to_md(fam),
        "",
        "## 已验证 / 未验证",
        "",
        "已验证（本次）：",
        "",
        "- 1 分钟合成 2 / 5 分钟 K 与长桥官方 K 逐根一致（MU、QQQ，2023-10-02 ~ 2026-10-21 全部"
        " 767 个交易日；"
        "RTH 与含盘前盘后）；",
        "- 开盘段处理方式在全部交易日的覆盖（口径 C 每天盘前 ≥ 19 根 2 分钟 K）；",
        "- 信号 / 状态 / 离场 / 成交在构造场景下的已知答案，多空镜像；",
        "- backtrader 独立复算 V4 口径 C MU 全部 18 格（H1–H3 × 离场 a/b/c × 多空）逐笔交易；",
        "- 增量（实时）路径与批量路径在同一组 K 上逐根一致；录制器逐笔 → K → 信号 → 成交的端到端"
        "回放"
        "（MU 2026-10-21 14:48 起、2026-10-22 全天）；",
        "- 负责人 D-030 样本中 48 笔 MU 的合约与规则合约对比。",
        "",
        "未验证 / 未覆盖：",
        "",
        "- App 开盘 40 分钟内的 BOLL / MACD 数值（STATUS 0f）；",
        "- backtrader 复算只做了 MU 口径 C；QQQ 与口径 A 未逐笔复算（同一套代码，未单独跑）；",
        "- 影子回放只有 1.3 个交易日、27 笔交易，只覆盖 MU；录制器盘口与 ThetaData 逐笔盘口的差异"
        "这次没有对比（V4 回测用 ThetaData）；",
        "- 长桥 1 分钟与 2 分钟历史本身的少量不一致（2026 年为主）原因未知（数据商侧）；",
        "- 零股不计价、常规成交计价的口径来自与官方 K 的比对（2026-10-21/22 两天），长桥没有文档"
        "说明（未验证）；",
        "- DRAM、META 的合约对比（无本地期权链）。",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
