"""事件检测器 E1–E8（STRATEGY_V1 第 1 层）。输出 (ts, symbol, direction, detector)。

每个检测器是纯函数 f(df, feats, params) -> pd.Series[int]（+1 多 / −1 空 / 0 无），同索引。
参数来自 config/events.yaml；所有阈值是待测参数（CLAUDE.md 规则 11）。
无前视：只用 ≤ t 的 bar 与特征。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from zr.features import indicators as ind

ET = ZoneInfo("America/New_York")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "events.yaml"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def _minute_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    e = index.tz_convert(ET)
    return (e.hour * 60 + e.minute).to_numpy()


def _day(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.tz_convert(ET).date, index=index)


def _first_per_day(mask: pd.Series) -> pd.Series:
    """每日只保留第一次为 True 的位置。"""
    day = _day(mask.index)
    return mask & (mask.groupby(day).cumsum() == 1)


def _sign(x: pd.Series, lo: float = 0.0) -> pd.Series:
    return pd.Series(np.where(x > lo, 1, np.where(x < -lo, -1, 0)), index=x.index)


# ---- E1 通道回踩 ----
def e1_channel_pullback(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    eps = p.get("eps", 0.001)
    mid = f["bb_mid"]
    up_trend = (
        (f["macd_dif_atr"] > f["macd_dea_atr"]) & (f["macd_dif_atr"] > 0) & (f["bb_mid_slope"] > 0)
    )
    touch_lo = (df["low"] - mid).abs() / mid <= eps
    long = up_trend & touch_lo & (df["close"] > mid)
    dn_trend = (
        (f["macd_dif_atr"] < f["macd_dea_atr"]) & (f["macd_dif_atr"] < 0) & (f["bb_mid_slope"] < 0)
    )
    touch_hi = (df["high"] - mid).abs() / mid <= eps
    short = dn_trend & touch_hi & (df["close"] < mid)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


# ---- E2 开盘区间突破 ----
def e2_opening_range_breakout(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    minutes = p.get("range_min", 30)
    mod = _minute_of_day(df.index)
    in_range = mod < 9 * 60 + 30 + minutes
    day = _day(df.index)
    hi = df["high"].where(in_range).groupby(day).cummax().groupby(day).ffill()
    lo = df["low"].where(in_range).groupby(day).cummin().groupby(day).ffill()
    after = pd.Series(~in_range, index=df.index)
    long = _first_per_day(after & (df["close"] > hi))
    short = _first_per_day(after & (df["close"] < lo))
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


# ---- E3 VWAP 失而复得 / 得而复失 ----
def e3_vwap_reclaim(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    n = p.get("min_bars", 5)
    below = (df["close"] < f["vwap"]).astype(int)
    above = 1 - below
    day = _day(df.index)
    below_run = (
        below.groupby(day)
        .apply(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
        .droplevel(0)
    )
    above_run = (
        above.groupby(day)
        .apply(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum())
        .droplevel(0)
    )
    long = (above == 1) & (below_run.shift(1) >= n) & (day == day.shift(1))
    short = (below == 1) & (above_run.shift(1) >= n) & (day == day.shift(1))
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


# ---- E4 长均线定向 + 震荡指标顺势交叉 ----
def e4_trend_oscillator_cross(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    n_ma, n_rsi = p.get("ma_n", 120), p.get("rsi_n", 14)
    lo, hi = p.get("rsi_low", 30), p.get("rsi_high", 70)
    ma = ind.sma(df["close"], n_ma)
    slope = ind.bar_slope(ma, p.get("slope_n", 10))
    rsi = ind.wilder_rsi(df["close"], n_rsi)
    up = (df["close"] > ma) & (slope > 0) & (rsi.shift(1) < lo) & (rsi >= lo)
    dn = (df["close"] < ma) & (slope < 0) & (rsi.shift(1) > hi) & (rsi <= hi)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


# ---- E5 CUSUM ----
def e5_cusum(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    """对称 CUSUM（López de Prado 2018 §2.5.2.1）。

    阈值 h = k × 滚动 σ（1 分钟对数收益）；日内重置。
    """
    k, win = p.get("k", 3.0), p.get("sigma_window", 60)
    r = np.log(df["close"]).diff()
    sigma = r.rolling(win, min_periods=win).std(ddof=1)
    day = _day(df.index).to_numpy()
    out = np.zeros(len(df), dtype=int)
    s_pos = s_neg = 0.0
    rv, sv = r.to_numpy(), sigma.to_numpy()
    for i in range(1, len(df)):
        if day[i] != day[i - 1] or not np.isfinite(rv[i]) or not np.isfinite(sv[i]):
            s_pos = s_neg = 0.0
            continue
        h = k * sv[i]
        s_pos = max(0.0, s_pos + rv[i])
        s_neg = min(0.0, s_neg + rv[i])
        if s_pos > h:
            out[i], s_pos = 1, 0.0
        elif s_neg < -h:
            out[i], s_neg = -1, 0.0
    return pd.Series(out, index=df.index)


# ---- E6 布林带挤压后扩张 ----
def e6_bb_squeeze_expansion(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    days, pct, exp = p.get("lookback_days", 20), p.get("low_pct", 0.1), p.get("expand", 0.05)
    width = f["bb_width"]
    day = _day(df.index)
    daily_q = width.groupby(day).quantile(pct)  # 每日带宽低分位
    thr = pd.Series(
        day.map(daily_q.shift(1).rolling(days, min_periods=5).min()).to_numpy(), index=df.index
    )
    squeezed = width.shift(1) <= thr
    expanding = width > width.shift(1) * (1 + exp)
    trig = _first_per_day(squeezed & expanding)
    direction = _sign(df["close"] - f["bb_mid"])
    return pd.Series(np.where(trig, direction, 0), index=df.index)


# ---- E7 日内动量（收盘前 30 分钟）----
def e7_intraday_momentum(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    at, min_ret = p.get("entry_minute", 15 * 60 + 30), p.get("min_abs_ret", 0.0)
    mod = _minute_of_day(df.index)
    trig = pd.Series(mod == at, index=df.index)
    direction = _sign(f["ret_since_open"], min_ret)
    return pd.Series(np.where(trig, direction, 0), index=df.index)


# ---- E8 VWAP 均值回归 ----
def e8_vwap_mean_reversion(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    """|close−VWAP| > k×ATR 时反向。gex_sign 列存在且 require_gex 时只在 gex_sign>0 启用；
    无期权数据时不门控（结果标注 ungated）。"""
    k = p.get("k", 2.0)
    d = f["vwap_dist_atr"]
    far = d.abs() > k
    direction = -_sign(d)
    if p.get("require_gex", False) and "gex_sign" in f:
        far = far & (f["gex_sign"] > 0)
    # 只在首次越过阈值时触发（避免连续分钟重复）
    trig = far & ~far.shift(1, fill_value=False)
    return pd.Series(np.where(trig, direction, 0), index=df.index)


DETECTORS: dict[str, Callable[[pd.DataFrame, pd.DataFrame, dict], pd.Series]] = {
    "E1": e1_channel_pullback,
    "E2": e2_opening_range_breakout,
    "E3": e3_vwap_reclaim,
    "E4": e4_trend_oscillator_cross,
    "E5": e5_cusum,
    "E6": e6_bb_squeeze_expansion,
    "E7": e7_intraday_momentum,
    "E8": e8_vwap_mean_reversion,
}


def e9_dip_buy(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    """E9 BinHV45 式抄底（定义在 zr/research/scan.py，延迟导入避免循环依赖）。"""
    from zr.research.scan import e9_dip_buy as _e9

    return _e9(df, f, p)


def e1_tight(df: pd.DataFrame, f: pd.DataFrame, p: dict) -> pd.Series:
    """E1 收紧版（METHOD_V2 规格 2）：通道 A 内回踩，ε 与确认根数来自参数。"""
    from zr.features.channel import channel_states
    from zr.research.replay_rule import pullback_events

    chans = p.get("_chans")
    chans = channel_states(df, f) if chans is None else chans
    return pullback_events(
        df, f, chans[f"chan_{p.get('channel', 'A')}"], p.get("eps", 0.0005), p.get("confirm", 1)
    )


DETECTORS["E9"] = e9_dip_buy
DETECTORS["E1T"] = e1_tight

# 共振家族（METHOD_V2 规格 3）：趋势 / 反转 / 波动
FAMILIES = {
    "trend": ("E1", "E1T", "E2", "E4", "E7"),
    "reversal": ("E3", "E8", "E9"),
    "vol": ("E5", "E6"),
}


def signal_matrix(
    df: pd.DataFrame, feats: pd.DataFrame, cfg: dict, extra: dict | None = None
) -> pd.DataFrame:
    """每个启用检测器一列（+1/−1/0），同索引。extra 会并入每个检测器的参数（例如预先算好的 _chans）
    。
    """
    cols = {}
    for name, fn in DETECTORS.items():
        params = cfg.get("detectors", {}).get(name)
        if params is None or not params.get("enabled", True):
            continue
        cols[name] = fn(df, feats, {**params, **(extra or {})}).astype(int)
    return pd.DataFrame(cols, index=df.index)


def confluence(sig: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """过去 window 根（含当根，日内）同向触发的检测器数 / 家族数，按方向分列。"""
    day = _day(sig.index).to_numpy()
    out = pd.DataFrame(index=sig.index)
    for d, tag in ((1, "up"), (-1, "dn")):
        hit = (sig == d).astype(int)
        rolled = hit.groupby(day).transform(lambda x: x.rolling(window, min_periods=1).max())
        out[f"confl_n_{tag}"] = rolled.sum(axis=1)
        fam = pd.DataFrame(
            {
                k: rolled[[c for c in v if c in rolled.columns]].max(axis=1)
                if any(c in rolled.columns for c in v)
                else 0
                for k, v in FAMILIES.items()
            },
            index=sig.index,
        )
        out[f"confl_fam_{tag}"] = fam.sum(axis=1)
    return out


def events_from_matrix(
    sig: pd.DataFrame, symbol: str, confl: pd.DataFrame | None = None
) -> pd.DataFrame:
    """矩阵 → 事件行；附上与事件同向的共振特征 confl_n_same_dir_5m /
    confl_families_5m（不含事件自身）。
    """
    rows = []
    for name in sig.columns:
        s = sig[name]
        hit = s[s != 0]
        rows.append(
            pd.DataFrame(
                {"ts": hit.index, "symbol": symbol, "direction": hit.to_numpy(), "detector": name}
            )
        )
    ev = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["ts", "symbol", "direction", "detector"])
    )
    if confl is not None and len(ev):
        c = confl.reindex(ev["ts"])
        up = ev["direction"].to_numpy() == 1
        n = np.where(up, c["confl_n_up"].to_numpy(), c["confl_n_dn"].to_numpy())
        fam = np.where(up, c["confl_fam_up"].to_numpy(), c["confl_fam_dn"].to_numpy())
        ev["confl_n_same_dir_5m"] = n - 1  # 不含自身
        ev["confl_families_5m"] = fam
    return ev.sort_values(["ts", "detector"]).reset_index(drop=True)


def detect_events(df: pd.DataFrame, feats: pd.DataFrame, symbol: str, cfg: dict) -> pd.DataFrame:
    rows = []
    for name, fn in DETECTORS.items():
        params = cfg.get("detectors", {}).get(name)
        if params is None or not params.get("enabled", True):
            continue
        sig = fn(df, feats, params)
        hit = sig[sig != 0]
        rows.append(
            pd.DataFrame(
                {"ts": hit.index, "symbol": symbol, "direction": hit.to_numpy(), "detector": name}
            )
        )
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "direction", "detector"])
    return pd.concat(rows, ignore_index=True).sort_values(["ts", "detector"]).reset_index(drop=True)
