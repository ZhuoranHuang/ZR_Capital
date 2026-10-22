"""技术指标：纯函数、无前视。第 i 个输出只依赖第 0..i 根 bar。

输入为按时间升序的 pd.Series；输出与输入同索引。逻辑参考 legacy/utils/indicators.py。
"""

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """递推 EMA：e_0 = x_0，e_t = a*x_t + (1-a)*e_{t-1}，a = 2/(n+1)。"""
    return s.ewm(span=n, adjust=False).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI（alpha=1/n）。avg_loss=0 时取 100。"""
    delta = close.diff()
    avg_gain = _wilder(delta.clip(lower=0), n)
    avg_loss = _wilder((-delta).clip(lower=0), n)
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi.mask(avg_loss == 0, 100.0)


def cross_signal(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """金叉 +1，死叉 -1，其余 0。仅当本根和上一根两条线都有值时才判定事件。"""
    valid = fast.notna() & slow.notna()
    both_valid = valid & valid.shift(1, fill_value=False)
    diff = (fast > slow).astype(int).diff()
    return diff.where(both_valid, 0).fillna(0).astype(int)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """hist = dif - dea（TradingView 口径；部分国内软件用 2*(dif-dea)）。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": dif - dea})


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0, min_periods: int | None = None
) -> pd.DataFrame:
    """min_periods=None 即 n（不足 n 根为 NaN）；给 1 则用部分窗口（不足 n 根时按已有根数算）。"""
    mp = n if min_periods is None else min_periods
    mid = close.rolling(n, min_periods=mp).mean()
    std = close.rolling(n, min_periods=mp).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    band = upper - lower
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "width": band / mid,
            "pct_b": (close - lower) / band.replace(0, np.nan),
        }
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """第一根无前收，取 high - low。"""
    prev_close = close.shift(1)
    parts = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return parts.max(axis=1, skipna=True)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), n)


def session_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """按交易日重置的 VWAP，价格用典型价 (H+L+C)/3。

    以索引的日历日期分组，索引须为 DatetimeIndex 且已转为交易所本地时区。
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("session_vwap 需要 DatetimeIndex")
    day = close.index.normalize()
    tp = (high + low + close) / 3
    cum_pv = (tp * volume).groupby(day).cumsum()
    cum_v = volume.groupby(day).cumsum()
    return cum_pv / cum_v.replace(0, np.nan)


def realized_vol(close: pd.Series, n: int, periods_per_year: float | None = None) -> pd.Series:
    """对数收益的滚动标准差（ddof=1）。给定 periods_per_year 时年化；年化因子由调用方提供。"""
    rv = np.log(close).diff().rolling(n, min_periods=n).std(ddof=1)
    if periods_per_year is not None:
        rv = rv * np.sqrt(periods_per_year)
    return rv


def bar_slope(s: pd.Series, n: int) -> pd.Series:
    """最近 n 根 bar 的 OLS 斜率（每根 bar 的变化量）。窗口内有 NaN 则为 NaN。"""
    if n < 2:
        raise ValueError("n 至少为 2")
    out = np.full(len(s), np.nan)
    if len(s) >= n:
        x = np.arange(n) - (n - 1) / 2
        w = x / (x @ x)
        windows = np.lib.stride_tricks.sliding_window_view(s.to_numpy(dtype=float), n)
        out[n - 1 :] = windows @ w
    return pd.Series(out, index=s.index)


def kdj(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, m1: int = 3, m2: int = 3
) -> pd.DataFrame:
    """KDJ：RSV = (C − Ln)/(Hn − Ln)×100；K = EMA(RSV, 1/m1)；D = EMA(K, 1/m2)；J = 3K − 2D。
    国内常用口径（SMA 递推 α=1/m）；无前视。"""
    ln = low.rolling(n, min_periods=n).min()
    hn = high.rolling(n, min_periods=n).max()
    rsv = ((close - ln) / (hn - ln).replace(0, np.nan) * 100).fillna(50.0)
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    return pd.DataFrame({"k": k, "d": d, "j": 3 * k - 2 * d})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Wilder ADX 与 +DI / −DI。无前视。"""
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(high, low, close)
    atr_n = _wilder(tr, n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / atr_n
    minus_di = 100 * _wilder(minus_dm, n) / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})
