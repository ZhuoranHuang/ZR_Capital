# ruff: noqa: E501  （中文报告行较长）
"""range_score 前视修复（D-037）的新旧对比 → docs/research/RANGE_SCORE_FIX.md。只报告，不重跑旧研究。

旧实现冻结在 _range_score_old（窗口 = window_days × 整段输入每日根数中位数，含 t 之后的行）。
在 MU、QQQ 全量历史上，口径 C（含盘前盘后原始 K、不按日重置）与 D-027（RTH-only、按日重置）各一遍：
2 分钟 K、v4 状态配置（config/v4_decisive.yaml 各口径的 range_threshold），比较 range_score
（|差| > 1e-12 或一边为 NaN 即不一致）与 regimes 的状态标签，按月列出不一致的行数与占比。
运行：python -m zr.research.range_score_fix
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.features import regime as rg
from zr.features.bars import ET
from zr.research import v4_decisive as v4d

REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "docs" / "research" / "RANGE_SCORE_FIX.md"
CONVS = {"C": "C（含盘前盘后、不按日重置）", "A": "D-027（RTH-only、按日重置）"}


def _range_score_old(x: pd.DataFrame, df: pd.DataFrame, rgc: dict, period: int) -> pd.Series:
    """修复前的实现（冻结副本，只供对比）：窗口取整段输入每日根数的中位数。"""
    w = max(1, round(rgc["window_days"] * rg.bars_per_day(df)))
    pr = lambda s: s.rolling(w, min_periods=max(20, w // 4)).rank(pct=True)  # noqa: E731
    wt = rgc["weights"]
    return (
        wt["bw"] * pr(x["bw"])
        + wt["slope"] * pr(x["slope"].abs())
        + wt["hist"] * pr((x["hist"] / df["close"]).abs())
    )


def compare(root: Path, und: str, conv_key: str) -> tuple[pd.DataFrame, dict]:
    v4 = v4d.load_v4()
    conv = v4["conventions"][conv_key]
    cfg = v4d.state_cfg(v4, conv_key)
    b = v4d.build_bars(root, und, conv, dt.date(2100, 1, 1))
    x = rg.indicators(b, cfg, daily_reset=conv["reset"], warmup=conv["warmup"])
    new = rg.regimes(b, x, cfg, 2)
    orig = rg.range_score
    try:
        rg.range_score = lambda x_, df_, rgc, p, bars_per_day=None: _range_score_old(x_, df_, rgc, p)
        old = rg.regimes(b, x, cfg, 2)
    finally:
        rg.range_score = orig
    a, o = new["range_score"].to_numpy(), old["range_score"].to_numpy()
    with np.errstate(invalid="ignore"):
        diff_s = ~((np.isnan(a) & np.isnan(o)) | (np.abs(a - o) <= 1e-12))
    diff_st = new["state"].to_numpy() != old["state"].to_numpy()
    et = b.index.tz_convert(ET)
    rth = np.asarray((et.hour * 60 + et.minute >= 570) & (et.hour * 60 + et.minute < 960))
    month = et.strftime("%Y-%m")
    t = pd.DataFrame({"month": month, "n": 1, "score": diff_s, "state": diff_st, "rth": rth, "state_rth": diff_st & rth})
    g = t.groupby("month").agg(rows=("n", "sum"), score_diff=("score", "sum"), state_diff=("state", "sum"), rth_rows=("rth", "sum"), state_diff_rth=("state_rth", "sum"))
    day = np.asarray(et.date)
    w_new = rg.range_windows(b, 2, cfg["regime"]["range"])
    meta = {
        "rows": len(b),
        "from": str(day.min()),
        "to": str(day.max()),
        "days": len(set(day)),
        "w_old": max(1, round(cfg["regime"]["range"]["window_days"] * rg.bars_per_day(b))),
        "w_new": sorted({int(v) for v in w_new}),
        "diff_days": sorted({str(d) for d in day[diff_s | diff_st]}),
        "score_diff": int(diff_s.sum()),
        "state_diff": int(diff_st.sum()),
    }
    return g, meta


CALLERS = [
    ("zr/live/core.py（实盘 / 纸面助手）", "不调用 range_score；增量引擎 IncrementalEngine 的窗口固定 195（口径 A）/ 480（口径 C）", "否（固定名义值，与修复后首日名义值一致）"),
    ("zr/strategy/incremental.py（增量引擎）", "窗口由调用方给定，独立实现，不调用 range_score", "否"),
    ("zr/research/shadow_replay.py:206（影子回放增量侧）", "w = bars_per_day(seed)，seed = SEED_FROM 至回放窗口开始前的历史 K（多日、只含过去）", "否（只用过去数据）"),
    ("zr/research/shadow_replay.py:325（影子回放批量对照侧）", "regimes(b, …)，b = 回放日及之前的 2 分钟 K（多日）", "仅当 b 只有 1–2 天时；按代码 b 含种子期多日，中位数与修复后一致的可能性高（未逐日核实）"),
    ("tests/test_incremental.py、tests/test_live_assistant.py", "单日夹具（MU 2026-10-21，口径 C 349 根至 15:36 / 口径 A RTH），引擎窗口曾取 bars_per_day(b) = 当日全天根数", "是：旧测试用整日根数（349）作窗口，含未来信息；已改为首日名义窗口，与 zr/live/core.py 一致"),
    ("tests/test_synthetic_scenarios.py", "单日合成场景（约百根），按「窗口 = 场景长度」设计", "是：已改为显式 bars_per_day=len(df)"),
    ("zr/research/v4_decisive.py:247、v4_h5.py:177（含 state_features）、entry_timing_diag.py:144、intraday_single_leg.py:312、guardrail_evidence.py:48、owner_entry_mining.py:132、sim_v3.py:226、own_policy.py:145/178（含 state_features）、method_v3_round1.py:46、lookahead_audit.py:98（含 state_features）、v4_checkpoints.py:89", "批量研究：全量或整年 / 多日 2 分钟 K（build_bars 读到 end 的全部历史，或研究窗口内全部日期）", "仅样本开头 1–2 天（见上表首日差异）；按日切片调用的（intraday_single_leg 等若逐日传入单日 K）会受影响——未逐个核实输入切片方式"),
    ("zr/research/model_v3.py:136（state_features）", "reg 来自上游批量 regimes", "同上游"),
    ("zr/features/range_calibration.py:37（RANGE 阈值标定）", "MU 2024 全年 2 分钟 K（RTH）", "仅 2024 年首日附近；阈值 0.3487 / 0.507 / 0.3499 未重标"),
]


def report(root: Path) -> str:
    from zr.ops.markdown import df_to_md

    L = ["# range_score 前视修复：新旧对比（D-037）", "",
         "`python -m zr.research.range_score_fix` 生成。只报告，不重跑旧研究。旧实现窗口 = window_days × 整段输入每日根数中位数（含 t 之后的行）；"
         "新实现 = 当日之前已完成交易日根数中位数（按日递推），首日用名义值（RTH 390 ÷ P、含盘前盘后 960 ÷ P）。2 分钟 K，v4 状态配置；"
         "不一致 = range_score |差| > 1e-12 或一边为 NaN；状态 = regimes 的 state。", ""]
    for und in ("MU", "QQQ"):
        for ck, cn in CONVS.items():
            g, m = compare(root, und, ck)
            bad = g[(g.score_diff > 0) | (g.state_diff > 0)].reset_index()
            bad["range_score 不一致占比"] = (bad.score_diff / bad.rows * 100).map(lambda v: f"{v:.2f}%")
            bad["状态不一致占比"] = (bad.state_diff / bad.rows * 100).map(lambda v: f"{v:.2f}%")
            L += [f"## {und} · 口径 {cn}", "",
                  f"数据：{m['from']}～{m['to']}，{m['days']} 个交易日，{m['rows']:,} 根 2 分钟 K。旧窗口 {m['w_old']} 根；新窗口取值 {m['w_new']}。"
                  f"不一致：range_score {m['score_diff']:,} 行，状态 {m['state_diff']:,} 行；涉及日期 {len(m['diff_days'])} 天"
                  + (f"（{', '.join(m['diff_days'][:10])}{' …' if len(m['diff_days']) > 10 else ''}）" if m["diff_days"] else "") + "。", ""]
            if len(bad):
                cols = {"month": "月", "rows": "行数", "score_diff": "range_score 不一致行", "range_score 不一致占比": "占比",
                        "state_diff": "状态不一致行", "状态不一致占比": "状态占比", "state_diff_rth": "其中 RTH 内状态不一致"}
                L += ["只列有不一致的月份（其余月份全部一致）：", "", df_to_md(bad[list(cols)].rename(columns=cols)), ""]
            else:
                L += ["全部月份一致。", ""]
    L += ["## 调用位置与输入长度（只列清单）", "",
          df_to_md(pd.DataFrame(CALLERS, columns=["调用位置", "输入", "输入只有 1–2 天时是否受过影响"])), "",
          "## 边界", "",
          "- 只比较 range_score 与状态标签；下游信号、交易、报告未重跑。",
          "- 全量历史上首日名义值（480 / 195）恰好等于全样本中位数、之后各日的递推中位数也不变，所以上表四组全部一致；修复只改变输入很短（1–2 天、或只有部分交易日）时的结果，见下方调用清单。",
          "- 调用清单按源码静态阅读整理，未逐个运行核实各研究脚本的实际输入切片。", ""]
    return "\n".join(L)


def main() -> int:
    root = get_settings().zr_data_root
    OUT_MD.write_text(report(root), encoding="utf-8")
    print(f"写入 {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
