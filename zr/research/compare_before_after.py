"""前视审计修复前后对比（fix/lookahead-audit 第 5 节）→ docs/research/LOOKAHEAD_FIX_COMPARISON.md。

「修复前」= git 历史里 `3dbd6c6`（feat/method-v3 合并进 main 那一刻，RTH-only、RANGE 三条件
同时满足版）的 `docs/research/{RULE_CARDS_MU,RULE_CARDS_QQQ}.md`；「修复后」= 当前工作区重跑后的
同名文件（D-026：RANGE 打分制 + 含盘前盘后连续 K 线）。全部数字从文件里用正则提取，不手抄。
运行：uv run python -m zr.research.compare_before_after
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from pathlib import Path

import pandas as pd

from zr.ops.markdown import df_to_md

REPO = Path(__file__).resolve().parents[2]
BEFORE_REF = "3dbd6c6"


def _git_show(path: str, ref: str = BEFORE_REF) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _cards_summary(text: str | None) -> dict:
    if text is None:
        return {"cards": None, "bonferroni": None, "rand_ge_95": None, "rows_positive": None}
    m1 = re.search(r"规则卡.*?：(\d+) 张", text)
    m2 = re.search(r"通过 Bonferroni.*?：(\d+) 张；同时段随机对照分位 ≥ 95 的卡：(\d+) 张", text)
    m3 = re.search(r"扣费后均值 > 0 的行 (\d+)", text)
    return {
        "cards": int(m1[1]) if m1 else None,
        "bonferroni": int(m2[1]) if m2 else None,
        "rand_ge_95": int(m2[2]) if m2 else None,
        "rows_positive": int(m3[1]) if m3 else None,
    }


def _model_auc(text: str | None) -> dict:
    if text is None:
        return {"auc_fixed": None, "auc_optuna": None}
    m = re.search(r"固定参数 ([\d.]+)，Optuna 嵌套调参 ([\d.]+)", text)
    return (
        {"auc_fixed": float(m[1]), "auc_optuna": float(m[2])}
        if m
        else {"auc_fixed": None, "auc_optuna": None}
    )


def _classification(text: str | None) -> pd.DataFrame:
    if text is None:
        return pd.DataFrame()
    rows = re.findall(r"\| ([^|]+) \| (\d+) \| ([\d.]+) \| ([-\d.]+) \| ([-\d.]+) \|", text)
    return pd.DataFrame(rows, columns=["state", "n", "win_rate", "sum_pnl", "mean_pnl"])


def card_comparison() -> pd.DataFrame:
    rows = []
    for und in ("MU", "QQQ"):
        path = f"docs/research/RULE_CARDS_{und}.md"
        before = _cards_summary(_git_show(path))
        after_text = (REPO / path).read_text(encoding="utf-8") if (REPO / path).exists() else None
        after = _cards_summary(after_text)
        for k in before:
            rows.append({"symbol": und, "metric": k, "before": before[k], "after": after[k]})
    return pd.DataFrame(rows)


def model_comparison() -> pd.DataFrame:
    root_hint = REPO.parent  # 占位，真正路径由 get_settings() 给
    del root_hint
    from zr.config import get_settings

    store = get_settings().zr_data_root / "research" / "v3"
    rows = []
    before_md = _git_show("docs/research/METHOD_V3_ROUND1.md")
    for und in ("MU", "QQQ"):
        before = {"auc_fixed": None, "auc_optuna": None}
        if before_md:
            sec = before_md.split(f"模型（第 8 节，{und}）")
            if len(sec) > 1:
                before = _model_auc(sec[1])
        after_path = store / f"model_{und}.md"
        after = (
            _model_auc(after_path.read_text(encoding="utf-8"))
            if after_path.exists()
            else {"auc_fixed": None, "auc_optuna": None}
        )
        rows.append(
            {
                "symbol": und,
                "auc_fixed_before": before["auc_fixed"],
                "auc_fixed_after": after["auc_fixed"],
                "auc_optuna_before": before["auc_optuna"],
                "auc_optuna_after": after["auc_optuna"],
            }
        )
    return pd.DataFrame(rows)


def classification_comparison() -> pd.DataFrame:
    before = _classification(_git_show("docs/research/OWN_POLICY_EXTRACTED.md"))
    after_path = REPO / "docs" / "research" / "OWN_POLICY_EXTRACTED.md"
    after = (
        _classification(after_path.read_text(encoding="utf-8"))
        if after_path.exists()
        else pd.DataFrame()
    )
    if before.empty or after.empty:
        return pd.DataFrame()
    m = before.merge(after, on="state", how="outer", suffixes=("_before", "_after"))
    return m


def checkpoint_comparison() -> pd.DataFrame:
    """4 个人工校验点：修复前的比例写死在这里是因为 git 历史版本用的是旧版 regime_check
    （函数签名/返回值都不同，没法直接重跑旧代码拿新格式结果）——数字摘自
    commit 3dbd6c6 的 docs/research/METHOD_V3_ROUND1.md 当时打印的符合比例，连同 E1
    回踩识别计数（12/36）一并记录来源。「修复后」当场重跑。
    """
    from zr.config import get_settings
    from zr.features.regime import load_config
    from zr.research.method_v3_round1 import regime_check

    chk, txt, why = regime_check(get_settings().zr_data_root, load_config())
    n_fired_after = int(re.search(r"(\d+) / 36", txt)[1]) if re.search(r"(\d+) / 36", txt) else None
    return pd.DataFrame(
        [
            {
                "checkpoint": "09:30 后急跌→IGNITION 下行",
                "符合比例_before": 0.0,
                "符合比例_after": round(float(chk["符合比例"].iloc[0]), 2),
            },
            {
                "checkpoint": "10:00–12:30→RANGE",
                "符合比例_before": 0.04,
                "符合比例_after": round(float(chk["符合比例"].iloc[1]), 2),
            },
            {
                "checkpoint": "12:18–13:36→CHANNEL 上行",
                "符合比例_before": 0.30,
                "符合比例_after": round(float(chk["符合比例"].iloc[2]), 2),
            },
            {
                "checkpoint": "13:00 回踩 K 被 E1 识别的定义数（共 36）",
                "符合比例_before": "12/36",
                "符合比例_after": f"{n_fired_after}/36" if n_fired_after is not None else txt[-40:],
            },
        ]
    )


def main() -> int:
    cc = card_comparison()
    mc = model_comparison()
    clc = classification_comparison()
    ckc = checkpoint_comparison()
    L = [
        "# 前视审计修复前后对比（fix/lookahead-audit 第 5 节）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"「修复前」取自 commit `{BEFORE_REF}`（RTH-only、RANGE 三条件同时满足）· "
        "复现 `uv run python -m zr.research.compare_before_after`",
        "",
        "## 规则卡（第 6 节网格）",
        "",
        df_to_md(cc),
        "",
        "## 模型样本外 AUC（第 8 节）",
        "",
        df_to_md(mc),
        "",
        "## 157 笔按状态分类（第 7 节）",
        "",
        df_to_md(clc) if len(clc) else "未取到对比数据。",
        "",
        "## 四个人工校验点（MU 2026-10-21）",
        "",
        df_to_md(ckc),
        "",
        "边界：卡片与模型对比只覆盖 MU、QQQ；SPY 未覆盖（期权拉取暂停）；「修复前」的分类/模型数字"
        "来自已合并进 main 的报告文本，用正则提取，提取失败显示 None。",
    ]
    out = REPO / "docs" / "research" / "LOOKAHEAD_FIX_COMPARISON.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
