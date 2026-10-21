"""标签（STRATEGY_V1 第 2 层）：triple-barrier 与事件响应曲线。

- triple_barrier：上障 +k·σ_H、下障 −k·σ_H、时间上限 H 分钟；
  σ_H = 滚动 1 分钟对数收益 std × √H（事件时刻）。
  路径只取同一交易日内的后续 bar；提前收盘按时间上限处理。
- response_curve：触发后 1/2/3/5/10/15/30 分钟的方向化收益均值与分位数。
- clustered_t：按日聚合的 t 值（处理同日样本重叠）。
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
HORIZONS = (1, 2, 3, 5, 10, 15, 30)


def rolling_sigma(close: pd.Series, window: int = 60) -> pd.Series:
    day = pd.Series(close.index.tz_convert(ET).date, index=close.index)
    r = np.log(close).diff().where(day.eq(day.shift(1)))
    return r.rolling(window, min_periods=window).std(ddof=1)


def _forward_paths(close: pd.Series, pos: np.ndarray, horizon: int) -> np.ndarray:
    """返回形状 (n_events, horizon) 的前向收益矩阵（close[t+j]/close[t] − 1），跨日处为 NaN。"""
    c = close.to_numpy()
    day = np.asarray(close.index.tz_convert(ET).date)
    n = len(c)
    out = np.full((len(pos), horizon), np.nan)
    for k, i in enumerate(pos):
        j_end = min(i + horizon, n - 1)
        js = np.arange(i + 1, j_end + 1)
        if len(js) == 0:
            continue
        same_day = day[js] == day[i]
        js = js[same_day]
        out[k, : len(js)] = c[js] / c[i] - 1
    return out


def triple_barrier(
    close: pd.Series, events: pd.DataFrame, horizon: int, k: float, sigma: pd.Series
) -> pd.DataFrame:
    """events 需含 ts、direction。返回每个事件的 label(+1/−1/0)、ret_signed、exit_min、barrier。"""
    if events.empty:
        return events.assign(H=horizon, label=[], ret_signed=[], exit_min=[], barrier=[])
    idx = close.index.get_indexer(events["ts"])
    ok = idx >= 0
    ev = events[ok].reset_index(drop=True)
    pos = idx[ok]
    paths = _forward_paths(close, pos, horizon) * ev["direction"].to_numpy()[:, None]
    sig = sigma.to_numpy()[pos] * np.sqrt(horizon)
    up, dn = k * sig, -k * sig
    labels = np.zeros(len(ev), dtype=int)
    rets = np.full(len(ev), np.nan)
    exits = np.full(len(ev), np.nan)
    for i in range(len(ev)):
        p = paths[i]
        valid = np.isfinite(p)
        if not valid.any() or not np.isfinite(sig[i]):
            continue
        hit_up = np.where(valid & (p >= up[i]))[0]
        hit_dn = np.where(valid & (p <= dn[i]))[0]
        first_up = hit_up[0] if len(hit_up) else np.inf
        first_dn = hit_dn[0] if len(hit_dn) else np.inf
        last = np.where(valid)[0][-1]
        if first_up < first_dn:
            labels[i], rets[i], exits[i] = 1, p[int(first_up)], first_up + 1
        elif first_dn < first_up:
            labels[i], rets[i], exits[i] = -1, p[int(first_dn)], first_dn + 1
        else:
            labels[i], rets[i], exits[i] = 0, p[last], last + 1
    out = ev.copy()
    out["H"] = horizon
    out["label"] = labels
    out["ret_signed"] = rets
    out["exit_min"] = exits
    out["barrier"] = up
    return out


def response_curve(close: pd.Series, events: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """每个 horizon 的方向化收益统计。"""
    if events.empty:
        return pd.DataFrame(
            columns=["horizon", "n", "mean", "median", "q10", "q25", "q75", "q90", "hit_rate"]
        )
    idx = close.index.get_indexer(events["ts"])
    ok = idx >= 0
    pos, d = idx[ok], events["direction"].to_numpy()[ok]
    paths = _forward_paths(close, pos, max(horizons)) * d[:, None]
    rows = []
    for h in horizons:
        col = paths[:, h - 1]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        rows.append(
            {
                "horizon": h,
                "n": len(col),
                "mean": col.mean(),
                "median": np.median(col),
                "q10": np.quantile(col, 0.1),
                "q25": np.quantile(col, 0.25),
                "q75": np.quantile(col, 0.75),
                "q90": np.quantile(col, 0.9),
                "hit_rate": float((col > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def clustered_t(values: pd.Series, days: pd.Series) -> tuple[float, int]:
    """按日聚合的 t 值：先对每日取均值，再对日均值做单样本 t；返回 (t, 天数)。"""
    daily = values.groupby(days.to_numpy()).mean().dropna()
    n = len(daily)
    if n < 2 or daily.std(ddof=1) == 0:
        return float("nan"), n
    return float(daily.mean() / (daily.std(ddof=1) / np.sqrt(n))), n
