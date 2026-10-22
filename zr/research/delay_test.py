"""延迟测试（fix/lookahead-audit 第 1 节）。

把全部入场信号在同一交易日内延迟 1 根 K 线（`sim_v3.shift_signal_within_day`）再跑一遍 2 分钟、
MACD(6,13,6) 的网格，与不延迟的原版比较：
- 全网格扣费后每张均值的中位数
- 已有规则卡（`cards_<UND>.parquet`）里前 20 张（按 mean_per_contract）各自在延迟版下的表现

预期：延迟通常让结果变差或持平（少了一根 K 线的"反应速度"）。如果延迟后**中位数或多数卡反而
显著变好**，说明原版的入场对齐可能有前视或执行时点问题，需要定位原因（而不是接受这个"更好"的
延迟版）。
运行：uv run python -m zr.research.delay_test --symbol MU.US
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd

from zr.config import get_settings
from zr.features.bars import load_bars
from zr.features.option_features import option_days
from zr.features.regime import load_config
from zr.ops.markdown import df_to_md
from zr.research import grid_v3
from zr.research import sim_v3 as sv

log = logging.getLogger("zr.research.delay_test")
PERIOD = 2
MACD_KEY = "macd"
MACD_TAG = "6,13,6"


def run(und: str, top_n: int = 20) -> tuple[str, pd.DataFrame]:
    t0 = time.monotonic()
    cfg = load_config()
    root = get_settings().zr_data_root
    stock = load_bars(root, und)
    days = option_days(root, und)

    base = sv.run_grid(root, und, stock, cfg, [PERIOD], days, macd_key=MACD_KEY)
    delayed = sv.run_grid(root, und, stock, cfg, [PERIOD], days, macd_key=MACD_KEY, delay_bars=1)
    lt_base = grid_v3.long_table({PERIOD: base[PERIOD]}, cfg, MACD_TAG)
    lt_delay = grid_v3.long_table({PERIOD: delayed[PERIOD]}, cfg, MACD_TAG)
    qty = cfg["execution"]["qty"]
    lt_base["mean_per_contract"] = lt_base["mean_net"] / qty
    lt_delay["mean_per_contract"] = lt_delay["mean_net"] / qty
    log.info("%s 两版网格完成，%.0fs", und, time.monotonic() - t0)

    join_cols = ["period", "entry", "side", "exit_rule", "tp", "sl", "cap"]
    m = lt_base.merge(lt_delay, on=join_cols, suffixes=("_base", "_delay"))
    m = m[(m["n_base"] > 0) & (m["n_delay"] > 0)]
    med_base = float(m["mean_per_contract_base"].median())
    med_delay = float(m["mean_per_contract_delay"].median())

    store = root / "research" / "v3"
    cards_path = store / f"cards_{und}.parquet"
    top_cmp = pd.DataFrame()
    if cards_path.exists():
        cards = pd.read_parquet(cards_path)
        cards = cards[(cards["period"] == PERIOD) & (cards["macd"] == MACD_TAG)]
        top = cards.sort_values("mean_per_contract", ascending=False).head(top_n)
        tc = top.merge(
            lt_delay[join_cols + ["mean_per_contract", "n", "mean_oos"]],
            on=join_cols,
            suffixes=("_base", "_delay"),
            how="left",
        )
        tc["delta"] = tc["mean_per_contract_delay"] - tc["mean_per_contract_base"]
        tc["improved"] = tc["delta"] > 0
        top_cmp = tc[
            join_cols
            + ["mean_per_contract_base", "mean_per_contract_delay", "delta", "n_base", "n_delay"]
        ]

    n_improved_median = int((m["mean_per_contract_delay"] > m["mean_per_contract_base"]).sum())
    n_total = len(m)
    suspicious = med_delay > med_base or (len(top_cmp) and (top_cmp["delta"] > 0).mean() > 0.5)
    verdict = (
        "延迟后反而显著变好，需要定位原因（见下）"
        if suspicious
        else "延迟后持平或变差，符合预期，未发现前视迹象"
    )
    L = [
        f"# 延迟测试（{und}，入场延迟 1 根 2 分钟 K）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 耗时 "
        f"{time.monotonic() - t0:.0f}s · 复现 `uv run python -m zr.research.delay_test "
        f"--symbol {und}.US`",
        "",
        f"全网格（2 分钟、MACD({MACD_TAG})，{n_total} 行同时有数据）扣费后每张均值中位数：",
        "",
        f"- 原版：{med_base:.4f}",
        f"- 延迟 1 根：{med_delay:.4f}",
        f"- 延迟后数值更高的行数：{n_improved_median} / {n_total}"
        f"（{n_improved_median / max(n_total, 1):.1%}）",
        "",
        f"**结论：{verdict}**",
        "",
    ]
    if len(top_cmp):
        L += [
            f"## 前 {len(top_cmp)} 张规则卡：原版 vs 延迟 1 根",
            "",
            df_to_md(top_cmp.round(3)),
            "",
            f"延迟后反而更好的卡：{int((top_cmp['delta'] > 0).sum())} / {len(top_cmp)}",
            "",
        ]
    else:
        L += [
            f"未找到 `{cards_path.name}`，跳过前 {top_n} 张卡的对照（先跑 "
            f"`uv run python -m zr.research.grid_v3 --symbol {und}.US`）。",
            "",
        ]
    L += [
        "边界：只测 2 分钟、MACD(6,13,6)（负责人实际参数）这一组；延迟实现为"
        "「同日内整体后移 1 根」，跨日边界的延迟后事件直接丢弃（不回绕到下一天），"
        "不是额外的前视来源。",
        "",
    ]
    return "\n".join(L), top_cmp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MU.US")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    und = a.symbol.split(".")[0]
    text, _ = run(und)
    out = Path(__file__).resolve().parents[2] / "docs" / "research" / f"DELAY_TEST_{und}.md"
    out.write_text(text, encoding="utf-8")
    log.info("写入 %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
