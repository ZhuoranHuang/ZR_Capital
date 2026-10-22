"""标签层 v2（METHOD_V2 规格 5）。决策价 = 触发 bar t 的收盘；窗口 = t+1 … t+H（同一交易日内）。

- (a) MFE / MAE：H 内正股方向化最大有利 / 不利偏移（多头 MFE = max(high)/c−1，MAE = min(low)/c−1；
  空头对称）；
      期权 bid 最大涨幅 bid_max_ret_H 来自 zr/research/outcomes.py（有期权数据的日期）
- (b) ever_up_x：H 内 MFE 是否曾 ≥ x·σ_H（σ_H = 60 分钟滚动 1 分钟对数收益 std × √H），x ∈ {0.3,
  0.5, 1}
- (c) 真实期权盈亏：outcomes 表 net（持仓上限 10 / 30）
- (d) 对称障碍对照：zr/research/labels.triple_barrier（k=1）
H（分钟）∈ {1, 2, 3, 5, 10, 15, 30}；10 秒 / 30 秒两档只在有录制器逐笔的日期计算，见
zr/research/seconds_v2.py。
事件响应曲线延伸到 60 分钟仅作参考，不做交易模拟。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.research.labels import rolling_sigma, triple_barrier

H_MIN = (1, 2, 3, 5, 10, 15, 30)
X_SIGMA = (0.3, 0.5, 1.0)


def _windows(arr: np.ndarray, pos: np.ndarray, day: np.ndarray, h: int) -> np.ndarray:
    """(n_events, h) 的前向窗口矩阵，跨日或越界为 NaN。"""
    n = len(arr)
    idx = pos[:, None] + np.arange(1, h + 1)[None, :]
    ok = idx < n
    idx_c = np.minimum(idx, n - 1)
    w = arr[idx_c].astype(float)
    same = day[idx_c] == day[pos][:, None]
    w[~(ok & same)] = np.nan
    return w


def mfe_mae(
    df: pd.DataFrame, ts: pd.Series, direction: np.ndarray, h: int
) -> tuple[np.ndarray, np.ndarray]:
    pos = df.index.get_indexer(ts)
    ok = pos >= 0
    day = np.asarray(df.index.tz_convert(ET).date)
    c0 = np.full(len(ts), np.nan)
    c0[ok] = df["close"].to_numpy()[pos[ok]]
    p = np.where(ok, pos, 0)
    hi = _windows(df["high"].to_numpy(), p, day, h)
    lo = _windows(df["low"].to_numpy(), p, day, h)
    with np.errstate(all="ignore"):
        up_fav = np.nanmax(hi, axis=1) / c0 - 1
        up_adv = np.nanmin(lo, axis=1) / c0 - 1
        dn_fav = 1 - np.nanmin(lo, axis=1) / c0
        dn_adv = 1 - np.nanmax(hi, axis=1) / c0
    long = direction == 1
    mfe = np.where(long, up_fav, dn_fav)
    mae = np.where(long, up_adv, dn_adv)
    mfe[~ok] = np.nan
    mae[~ok] = np.nan
    return mfe, mae


def stock_labels(df: pd.DataFrame, events: pd.DataFrame, sigma_window: int = 60) -> pd.DataFrame:
    """events：ts、direction。返回与 events 对齐的标签列。"""
    out = pd.DataFrame(index=events.index)
    sigma = rolling_sigma(df["close"], sigma_window)
    sig_ev = sigma.reindex(events["ts"]).to_numpy()
    d = events["direction"].to_numpy()
    for h in H_MIN:
        mfe, mae = mfe_mae(df, events["ts"], d, h)
        out[f"mfe_{h}"], out[f"mae_{h}"] = mfe, mae
        s_h = sig_ev * np.sqrt(h)
        for x in X_SIGMA:
            v = np.where(
                np.isfinite(mfe) & np.isfinite(s_h), (mfe >= x * s_h).astype(float), np.nan
            )
            out[f"ever_up_{x}_{h}"] = v
        tb = triple_barrier(df["close"], events[["ts", "direction"]], h, 1.0, sigma)
        out[f"tb_label_{h}"] = tb["label"].to_numpy()
        out[f"tb_ret_{h}"] = tb["ret_signed"].to_numpy()
    out["sigma_1m"] = sig_ev
    return out


def option_labels(
    events: pd.DataFrame, outcomes: pd.DataFrame, bar_minute: np.ndarray
) -> pd.DataFrame:
    """按 (日期, 决策分钟 = 触发分钟+1, 方向) 关联 outcomes：net_cap10 / net_cap30 / bid_max_ret_H
    / dte。
    """
    if outcomes.empty:
        return pd.DataFrame(index=events.index)
    key = pd.DataFrame(
        {
            "day": events["ts"].dt.tz_convert(ET).dt.date.to_numpy(),
            "minute": bar_minute + 1,
            "direction": events["direction"].to_numpy(),
        },
        index=events.index,
    )
    bm = [c for c in outcomes.columns if c.startswith("bid_max_ret_")]
    o10 = outcomes[outcomes["cap"] == 10][
        ["day", "minute", "direction", "net", "dte", "d001_violation", *bm]
    ]
    o30 = outcomes[outcomes["cap"] == 30][["day", "minute", "direction", "net"]].rename(
        columns={"net": "net_cap30"}
    )
    m = key.reset_index().merge(
        o10.rename(columns={"net": "net_cap10"}), on=["day", "minute", "direction"], how="left"
    )
    m = m.merge(o30, on=["day", "minute", "direction"], how="left").set_index("index")
    return m.drop(columns=["day", "minute", "direction"]).reindex(events.index)


def response_curve_60(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """参考用：1–60 分钟方向化收益均值与分位（不做交易模拟）。"""
    from zr.research.labels import response_curve

    return response_curve(
        df["close"], events[["ts", "direction"]], (1, 2, 3, 5, 10, 15, 30, 45, 60)
    )
