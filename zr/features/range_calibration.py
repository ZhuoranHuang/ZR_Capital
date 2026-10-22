"""RANGE 打分阈值标定（D-026，fix/lookahead-audit 第 3 节）。

在 MU 2024 年 2 分钟 K 上，取 `range_score` 的 30% 分位（[20%, 40%] 区间中点）作为
`config/method_v3.yaml` 的 `regime.range.score_threshold`；只在 2024 年数据上标定一次，
不逐年重新标定（否则是用未来数据调参）。另在 2025、2026 年用这个固定阈值检验覆盖率是否稳定。

运行：uv run python -m zr.features.range_calibration → 打印标定结果，不写文件（阈值人工抄入 yaml）。
"""

from __future__ import annotations

import datetime as dt
import logging

from zr.config import get_settings
from zr.features.bars import load_bars, resample_bars
from zr.features.regime import indicators, load_config, range_score

log = logging.getLogger("zr.features.range_calibration")
CALIBRATION_YEAR = 2024
TARGET_QUANTILE = 0.30  # [20%, 40%] 区间中点


def year_scores(root, und: str, year: int, cfg: dict):
    s = load_bars(root, und)
    et = s.index.tz_convert("America/New_York")
    import numpy as np

    d = s[
        (np.asarray(et.date) >= dt.date(year, 1, 1))
        & (np.asarray(et.date) <= dt.date(year, 12, 31))
    ]
    if d.empty:
        return None
    b2 = resample_bars(d, "2min")
    x = indicators(b2, cfg)
    return range_score(x, b2, cfg["regime"]["range"], 2).dropna()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    root = get_settings().zr_data_root
    cal = year_scores(root, "MU", CALIBRATION_YEAR, cfg)
    thr = float(cal.quantile(TARGET_QUANTILE))
    log.info("MU %d 年：score 分位 → 阈值候选", CALIBRATION_YEAR)
    for q in (0.20, 0.25, 0.30, 0.35, 0.40):
        log.info("  q=%.2f -> %.4f", q, float(cal.quantile(q)))
    log.info("采用阈值（q=%.2f）：%.4f", TARGET_QUANTILE, thr)
    for year in (2024, 2025, 2026):
        s = year_scores(root, "MU", year, cfg)
        if s is None:
            continue
        cov = float((s <= thr).mean())
        log.info("%d 年覆盖率（固定阈值 %.4f）：%.4f（n=%d）", year, thr, cov, len(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
