"""跨资产特征（METHOD_V2 规格 4）：参照标的（SPY / QQQ / SMH，去掉自身）同一分钟的 1 分钟收益
与 5 分钟实现波动，按时间戳对齐（只用参照标的 ≤t 的 bar；参照缺该分钟则 NaN）。
列：xa_<REF>_ret1、xa_<REF>_rv5，以及与本标的同向的参照个数 xa_n_same_sign。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zr.features.bars import ET, load_bars

REFERENCES = ("SPY", "QQQ", "SMH")


def _ref_frame(ref: pd.DataFrame) -> pd.DataFrame:
    day = pd.Series(ref.index.tz_convert(ET).date, index=ref.index)
    lr = np.log(ref["close"]).diff().where(day.eq(day.shift(1)))
    return pd.DataFrame(
        {
            "ret1": ref["close"].pct_change().where(day.eq(day.shift(1))),
            "rv5": lr.rolling(5, min_periods=5).std(ddof=1),
        },
        index=ref.index,
    )


def compute_cross_asset(
    df: pd.DataFrame,
    und: str,
    root: Path | None = None,
    refs: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    own_day = pd.Series(df.index.tz_convert(ET).date, index=df.index)
    own = df["close"].pct_change().where(own_day.eq(own_day.shift(1)))
    same = pd.Series(0.0, index=df.index)
    n_ok = pd.Series(0.0, index=df.index)
    for r in REFERENCES:
        if r == und:
            continue
        if refs is not None and r in refs:
            rb = refs[r]
        else:
            try:
                rb = load_bars(root, r)
            except FileNotFoundError:
                out[f"xa_{r}_ret1"] = np.nan
                out[f"xa_{r}_rv5"] = np.nan
                continue
        f = _ref_frame(rb).reindex(df.index)
        out[f"xa_{r}_ret1"] = f["ret1"]
        out[f"xa_{r}_rv5"] = f["rv5"]
        ok = f["ret1"].notna() & own.notna()
        same += (np.sign(f["ret1"]) == np.sign(own)) & ok
        n_ok += ok
    out["xa_n_same_sign"] = same.where(n_ok > 0)
    return out


def cross_asset_columns(und: str) -> list[str]:
    cols = []
    for r in REFERENCES:
        if r != und:
            cols += [f"xa_{r}_ret1", f"xa_{r}_rv5"]
    return cols + ["xa_n_same_sign"]
