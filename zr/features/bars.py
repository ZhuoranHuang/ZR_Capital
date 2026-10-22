"""1 分钟 K 的加载、聚合（2/5 分钟）与成交量 bar。

全部按美东交易日分组，不跨日。

输入统一为 DataFrame，索引 `ts`（UTC tz-aware、升序），
列 open/high/low/close/volume（可含 turnover/session）。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def load_bars(
    root: Path,
    und: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    intraday_only: bool = True,
    rth_only: bool = True,
) -> pd.DataFrame:
    """读取 bars_1m/underlying=XXX/year=*/bars.parquet。

    返回以 ts(UTC) 为索引的升序 DataFrame。
    rth_only（默认）：只保留交易日历上的交易日、[09:30, 收盘) 内的 bar（提前收盘日到 13:00）；
    剔除长桥在休市日生成的零成交 bar（zr/data/calendar.py）。rth_only=True 隐含 intraday_only。
    """
    files = sorted((root / "bars_1m" / f"underlying={und}").glob("year=*/bars.parquet"))
    if not files:
        raise FileNotFoundError(f"没有 {und} 的 1 分钟 K：{root / 'bars_1m'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if intraday_only and "session" in df:
        df = df[df["session"] == "Intraday"]
    df = df.set_index("ts").sort_index()
    df.index = df.index.tz_convert("UTC")
    if rth_only:
        from zr.data.calendar import rth_mask

        df = df[rth_mask(df.index)]
    if start is not None or end is not None:
        d = et_date(df.index)
        if start is not None:
            df = df[d >= start]
        if end is not None:
            df = df[et_date(df.index) <= end]
    return df[~df.index.duplicated(keep="last")]


def et_date(index: pd.DatetimeIndex) -> np.ndarray:
    return index.tz_convert(ET).date


def load_extended_bars(
    root: Path,
    und: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """读取含盘前盘后的连续 1 分钟 K（Pre + Intraday + Post），按交易日历过滤交易日，不做
    RTH 裁剪（供 v3 状态层的 BOLL/MACD 等指标用，与长桥 App 的连续图一致；信号/交易仍限
    09:30–16:00，由下游按分钟索引过滤，见 zr/research/sim_v3.py）。已用
    `zr.data.bar_quality.flag_extended_bars` 剔除坏 bar。

    clean=False：不剔除，保留长桥原始 K（App 画的就是原始 K）。注意 clean=True 会误删真实的
    RTH 大振幅 bar：ATR 跨 session 连续计算，安静盘前之后的开盘急跌 bar 振幅 > 5×ATR 被当成坏 bar
    （MU 2026-10-21 的 09:32、12:39 两根被删，13:00 中轨与 App 差 0.24，见 V4_CHECKPOINTS.md）。
    """
    from zr.data.bar_quality import clean_extended_bars
    from zr.data.calendar import is_trading_day

    files = sorted((root / "bars_1m" / f"underlying={und}").glob("year=*/bars.parquet"))
    if not files:
        raise FileNotFoundError(f"没有 {und} 的 1 分钟 K：{root / 'bars_1m'}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "session" in df:
        df = df[df["session"].isin(["Pre", "Intraday", "Post"])]
    df = df.set_index("ts").sort_index()
    df.index = df.index.tz_convert("UTC")
    days = et_date(df.index)
    uniq = {d: is_trading_day(d) for d in set(days)}
    df = df[np.fromiter((uniq[d] for d in days), dtype=bool, count=len(days))]
    df = df[~df.index.duplicated(keep="last")]
    if clean:
        df = clean_extended_bars(df)
    if start is not None or end is not None:
        d = et_date(df.index)
        if start is not None:
            df = df[d >= start]
        if end is not None:
            df = df[et_date(df.index) <= end]
    return df


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """按美东日分组后 resample（label=left，bar 时间戳 = 开盘时刻），不跨日。

    rule 如 "2min" / "5min"。分箱锚定在 UTC 零点（origin="start_day"），即整分钟网格
    （2 分钟 K 落在偶数分钟，与长桥原生 2 分钟 K 一致）。不用 origin="start"：含盘前盘后数据
    经 clean_extended_bars 清洗后当日 04:00 那根常被删、首根变成 04:01，会把全天分箱错开一分钟
    （MU 767 天里 286 天，见 V4_CHECKPOINTS.md）。
    """
    parts = []
    for _, g in df.groupby(et_date(df.index)):
        r = g.resample(rule, label="left", closed="left", origin="start_day").agg(OHLCV_AGG)
        parts.append(r.dropna(subset=["open"]))
    out = pd.concat(parts)
    out.index.name = "ts"
    return out


def daily_volume(df: pd.DataFrame) -> pd.Series:
    s = df["volume"].groupby(et_date(df.index)).sum()
    s.index = pd.Index(s.index, name="date")
    return s


def volume_threshold(
    df: pd.DataFrame, lookback_days: int = 20, divisor: float = 100.0
) -> pd.Series:
    """每个交易日的成交量 bar 阈值 = 之前 lookback_days 个交易日日均量 / divisor。

    不含当日，无前视。
    """
    dv = daily_volume(df)
    thr = dv.shift(1).rolling(lookback_days, min_periods=lookback_days).mean() / divisor
    thr.name = "threshold"
    return thr


def volume_bars(df: pd.DataFrame, threshold: float | pd.Series) -> pd.DataFrame:
    """成交量 bar：累计成交量达到阈值即收一根；日内重置。

    threshold 可为标量或按日 Series（index=date）。
    """
    rows = []
    for day, g in df.groupby(et_date(df.index)):
        thr = float(threshold) if np.isscalar(threshold) else threshold.get(day, np.nan)
        if not np.isfinite(thr) or thr <= 0:
            continue
        cum, o, h, lo, n, t_open = 0.0, None, -np.inf, np.inf, 0, None
        for ts, r in zip(
            g.index,
            g[["open", "high", "low", "close", "volume"]].itertuples(index=False),
            strict=True,
        ):
            if o is None:
                o, t_open = r.open, ts
            h, lo = max(h, r.high), min(lo, r.low)
            cum += r.volume
            n += 1
            if cum >= thr:
                rows.append(
                    {
                        "ts_open": t_open,
                        "ts_close": ts,
                        "open": o,
                        "high": h,
                        "low": lo,
                        "close": r.close,
                        "volume": cum,
                        "n_bars": n,
                        "date": day,
                    }
                )
                cum, o, h, lo, n, t_open = 0.0, None, -np.inf, np.inf, 0, None
    return pd.DataFrame(rows)
