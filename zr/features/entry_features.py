"""方向化入场特征（逐秒网格），供负责人入场挖掘（zr/research/owner_entry_mining.py）与样本外
规则检验（R-049）共用——同一套代码、同一口径。

时间：当日 RTH 1 秒网格（09:30 起，intrabar.second_grid）。时刻 t 的每个特征只用 t 时已经可得的
数据：秒级正股价（看跌看涨平价反推，ts ≤ t）；K 线只用收盘时刻（标签 + 周期）≤ t 的已完成 K；
期权 1 分钟成交量 bar 同样按标签 + 60 秒可得；20 日同时段均值只用此前交易日。截断检验
（truncation_check）：把 t 之后的数据全部删掉重算，t 时刻的值必须不变。

方向化（d = +1 买看涨 / −1 买看跌）。每个特征登记为一种 kind：
- signed：值 × d（如涨跌幅：看涨按上涨为正，看跌按下跌为正）；
- pair：(a, b)，d = +1 取 a，d = −1 取 b（如「顺向极值距离」：看涨看高点、看跌看低点）；
- pct1 / pct100：d = −1 时取 1 − x / 100 − x（%b、RSI、KDJ）；
- none：无方向（波动率、带宽、时刻等）。

K 线口径 C（D-028）：长桥原始 1 分钟 K 含盘前盘后连续（load_extended_bars(clean=False)）、不按日
重置；1 / 2 / 5 / 15 分钟 K 由 1 分钟 K 按整分钟网格聚合（resample_bars）。每天只取此前 14 个日历
日的 K 计算指标（EMA 预热足够；挖掘与样本外同一做法）。

特征族（名前缀）：
- b{P}_*：P 分钟已完成 K 的趋势特征（BOLL 中轨 / 上下轨斜率与加速度、带宽、%b、MA5/10/20/60
  斜率与排列、距均线 ATR 倍数、近 N 根收在中轨同侧的根数、连续同向收盘、效率比、高低点结构、距
  20 根极值的根数、回归斜率与 R²、ADX / DI、Aroon、MACD(6,13,6) 与 (12,26,9)、RSI(2) / RSI(14)、
  KDJ(9,3,3)、中轨同侧持续根数）；live{P}_*：当前秒级价相对已完成 K 的布林（%b、距中轨）与正在
  形成的 K（已过秒数、实体、上下影）；
- s_*：秒级（5–600 秒涨跌、实现波动、最大单秒跳动、上涨秒数占比、距近 1–60 分钟与当日极值、
  是否处在新极值、距 VWAP / 昨收 / 开盘、跳空）；
- v_*：成交量（最近 1 分钟、5 分钟相对前 20 日同时段均值，1 / 5 分钟量比，近 20 根涨跌量比）；
- o_*：期权（平价反推所用最近到期、平值附近行权价的看涨 / 看跌：价差 % 与 tick 数、报价更新
  次数与加速、中间价 30 / 60 / 120 秒变化、跨式近似隐含波动率及 5 / 15 分钟变化、DTE、是否到期
  日；期权 1 分钟 bar：近 5 / 15 分钟看涨 / 看跌成交量比、近平值成交量相对前 20 日同时段）；
- c_*：日历（开盘后 / 收盘前分钟数、星期、距定时宏观发布分钟数）；
- x_*：跨标的（SPY / QQQ / SMH 近 1 / 5 / 15 分钟涨跌，相对 SMH 强弱，SPY 30 分钟实现波动）；
- f_*：暴力因子（算子库 × 原始序列 × 窗口 30 秒–30 分钟，见 brute()）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest.intrabar import MAX_AGE, NS
from zr.data.calendar import session_close
from zr.features.bars import ET, et_date, load_bars, load_extended_bars, resample_bars
from zr.features.calendar_features import release_times
from zr.features.indicators import adx, atr, kdj, wilder_rsi

PERIODS = (1, 2, 5, 15)
LIVE_PERIODS = (1, 2, 5)
SEC_RET = (5, 10, 30, 60, 120, 300, 600)
EXT_MIN = (1, 5, 15, 30, 60)
BRUTE_W = (30, 60, 120, 300, 900, 1800)
CROSS = ("SPY", "QQQ", "SMH")
TOL_BP = 2.0  # 「处在新极值」：距极值 ≤ 2 bp（平价反推价的噪声量级）
LOOKBACK_DAYS = 14
FAMILIES = ("bar", "live", "sec", "vol", "opt", "optvol", "cal", "cross", "brute")


@dataclass
class F:
    kind: str  # signed / pair / pct1 / pct100 / none
    a: np.ndarray
    b: np.ndarray | None = None

    def directional(self, d: np.ndarray) -> np.ndarray:
        if self.kind == "signed":
            return self.a * d
        if self.kind == "pair":
            return np.where(d == 1, self.a, self.b)
        if self.kind == "pct1":
            return np.where(d == 1, self.a, 1 - self.a)
        if self.kind == "pct100":
            return np.where(d == 1, self.a, 100 - self.a)
        return self.a


# ---------------- 数据 ----------------


@dataclass
class Ctx:
    """跨日共享的数据（一次加载）。"""

    root: Path
    und: str = "MU"
    ext: pd.DataFrame = field(default=None, repr=False)
    rth: dict = field(default_factory=dict, repr=False)
    releases: pd.DatetimeIndex = field(default=None, repr=False)
    _optvol: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, root: Path, und: str = "MU") -> Ctx:
        c = cls(root=root, und=und)
        c.ext = load_extended_bars(root, und, clean=False)
        for u in (und, *CROSS):
            c.rth[u] = load_bars(root, u)
        c.releases = release_times()
        return c

    def optvol_minutes(self, day: dt.date) -> pd.DataFrame | None:
        """期权 1 分钟 bar 按分钟汇总：看涨 / 看跌量（最近到期、行权价在开盘价 ±5% 内）、近平值量
        （任意到期、±2%）。行 = 分钟标签（UTC ns）。"""
        if day in self._optvol:
            return self._optvol[day]
        p = self.root / "options_1m" / f"underlying={self.und}" / f"date={day}" / "ohlc.parquet"
        out = None
        if p.exists():
            o = pd.read_parquet(p, columns=["ts", "expiry", "strike", "right", "volume"])
            r = self.rth[self.und]
            r = r[et_date(r.index) == day]
            if len(o) and len(r):
                ref = float(r["open"].iloc[0])
                o["expiry"] = pd.to_datetime(o["expiry"]).dt.date
                fut = [e for e in o["expiry"].unique() if e >= day]
                if not fut:
                    self._optvol[day] = None
                    return None
                front = min(fut)
                o["t"] = pd.DatetimeIndex(o["ts"]).as_unit("ns").asi8
                m5 = (o["expiry"] == front) & ((o["strike"] / ref - 1).abs() <= 0.05)
                m2 = (o["strike"] / ref - 1).abs() <= 0.02
                g = o[m5].groupby(["t", "right"])["volume"].sum().unstack(fill_value=0)
                out = pd.DataFrame(
                    {
                        "call": g.get("call", pd.Series(dtype=float)),
                        "put": g.get("put", pd.Series(dtype=float)),
                    }
                )
                out["near"] = o[m2].groupby("t")["volume"].sum()
                out = out.fillna(0.0).sort_index()
                out["mod"] = [
                    (pd.Timestamp(x, tz="UTC").tz_convert(ET).hour * 60)
                    + pd.Timestamp(x, tz="UTC").tz_convert(ET).minute
                    for x in out.index
                ]
        self._optvol[day] = out
        return out


@dataclass
class Day:
    """单日输入（可截断到某时刻，用于截断检验）。"""

    day: dt.date
    t: np.ndarray  # 1 秒网格（ns）
    S: np.ndarray
    opt: pd.DataFrame  # 网格上的 c_mid / p_mid / c_spread / p_spread / strike / expiry / n_upd
    ext: pd.DataFrame  # 当日及此前 LOOKBACK_DAYS 的 1 分钟 K（口径 C）
    rth: dict  # 标的 → 当日及此前 30 个日历日的 RTH 1 分钟 K
    optvol: pd.DataFrame | None
    optvol_hist: list  # 此前 20 个交易日的 optvol（按分钟）
    releases: np.ndarray


def _asof_idx(ts: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.searchsorted(ts, t, side="right") - 1


def load_day(ctx: Ctx, day: dt.date, t_cut: int | None = None) -> Day | None:
    p = ctx.root / "intrabar" / f"underlying={ctx.und}" / f"date={day}" / "S.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p)
    if t_cut is not None:
        s = s[s["ts"] <= t_cut]
    open_ns = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).value
    close_ns = pd.Timestamp(dt.datetime.combine(day, session_close(day)), tz=ET).value
    grid = np.arange(open_ns, close_ns, NS)
    ts = s["ts"].to_numpy()
    i = _asof_idx(ts, grid)
    ok = (i >= 0) & ((grid - ts[np.clip(i, 0, None)]) <= MAX_AGE)
    ic = np.clip(i, 0, None)

    def col(c):
        v = s[c].to_numpy()
        return np.where(ok, v[ic], np.nan) if len(v) else np.full(len(grid), np.nan)

    opt = pd.DataFrame({c: col(c) for c in ("c_mid", "p_mid", "c_spread", "p_spread", "strike")})
    opt["n_upd"] = np.where(i >= 0, i + 1, 0).astype(float)  # 截至该秒的累计报价更新次数
    exp = pd.to_datetime(s["expiry"]).dt.date.to_numpy() if len(s) else np.array([])
    opt["dte"] = np.where(ok, [(exp[j] - day).days if len(exp) else np.nan for j in ic], np.nan)
    opt["exp_close"] = np.where(
        ok,
        [
            pd.Timestamp(dt.datetime.combine(exp[j], dt.time(16, 0)), tz=ET).value
            if len(exp)
            else np.nan
            for j in ic
        ],
        np.nan,
    )
    lo = day - dt.timedelta(days=LOOKBACK_DAYS)
    ed = et_date(ctx.ext.index)
    ext = ctx.ext[(ed >= lo) & (ed <= day)]
    rth = {}
    for u, df in ctx.rth.items():
        d0 = et_date(df.index)
        rth[u] = df[(d0 >= day - dt.timedelta(days=45)) & (d0 <= day)]
    if t_cut is not None:  # 1 分钟 K 在标签 + 60 秒收盘后才可得
        ext = ext[_ns(ext.index) + 60 * NS <= t_cut]
        rth = {u: df[_ns(df.index) + 60 * NS <= t_cut] for u, df in rth.items()}
    ov = ctx.optvol_minutes(day)
    if ov is not None and t_cut is not None:
        ov = ov[ov.index + 60 * NS <= t_cut]
    hist_days = sorted({x for x in et_date(ctx.rth[ctx.und].index) if x < day})[-20:]
    hist = [h for h in (ctx.optvol_minutes(x) for x in hist_days) if h is not None]
    return Day(day, grid, col("S"), opt, ext, rth, ov, hist, _ns(ctx.releases))


def _ns(idx) -> np.ndarray:
    return pd.DatetimeIndex(idx).as_unit("ns").asi8


# ---------------- 工具 ----------------


def _roll(x: np.ndarray, w: int, fn: str, minp: float = 0.5) -> np.ndarray:
    r = pd.Series(x).rolling(w, min_periods=max(int(w * minp), 2))
    return getattr(r, fn)().to_numpy()


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x.copy()
    if k >= len(x):
        return np.full(len(x), np.nan)
    return np.r_[np.full(k, np.nan), x[:-k]]


def _roll_slope(x: np.ndarray, w: int) -> np.ndarray:
    """窗口内 x 对序号的 OLS 斜率（每步）；窗口内有缺失按 pandas 协方差处理。"""
    s = pd.Series(x)
    idx = pd.Series(np.arange(len(x), dtype=float))
    mp = max(int(w * 0.5), 3)
    cov = s.rolling(w, min_periods=mp).cov(idx)
    var = idx.where(s.notna()).rolling(w, min_periods=mp).var()
    return (cov / var).to_numpy()


def _bars_since(flag: np.ndarray) -> np.ndarray:
    out = np.full(len(flag), np.nan)
    last = -1
    for k, f in enumerate(flag):
        if f:
            last = k
        out[k] = k - last if last >= 0 else np.nan
    return out


def _run_signed(x: np.ndarray) -> np.ndarray:
    """连续同号的长度，带符号（+3 = 连续 3 根上涨收盘）。"""
    out = np.zeros(len(x))
    for k in range(len(x)):
        s = np.sign(x[k]) if np.isfinite(x[k]) else 0
        if s == 0:
            out[k] = 0
        elif k and np.sign(out[k - 1]) == s:
            out[k] = out[k - 1] + s
        else:
            out[k] = s
    return out


# ---------------- K 线特征 ----------------


def bar_block(b: pd.DataFrame) -> dict[str, F]:
    """K 线级（每根已完成 K 一个值）。"""
    c, h, lo, v = b["close"], b["high"], b["low"], b["volume"]
    out: dict[str, F] = {}
    a = atr(h, lo, c, 14).replace(0, np.nan)
    an = a.to_numpy()
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    up, dn = mid + 2 * sd, mid - 2 * sd
    for k in (1, 3, 5, 10):
        out[f"boll_mid_slope{k}"] = F("signed", ((mid - mid.shift(k)) / k / a).to_numpy())
    s3 = (mid - mid.shift(3)) / 3 / a
    out["boll_mid_acc3"] = F("signed", (s3 - s3.shift(3)).to_numpy())
    us, ls = ((up - up.shift(5)) / 5 / a).to_numpy(), ((dn - dn.shift(5)) / 5 / a).to_numpy()
    out["boll_band_with_slope5"] = F("pair", us, -ls)
    out["boll_band_against_slope5"] = F("pair", ls, -us)
    bw = ((up - dn) / mid * 1e4).to_numpy()
    out["boll_bw_bp"] = F("none", bw)
    out["boll_bw_chg5"] = F("none", bw - _shift(bw, 5))
    out["pctb_close"] = F("pct1", ((c - dn) / (up - dn).replace(0, np.nan)).to_numpy())
    mas = {n: c.rolling(n, min_periods=n).mean() for n in (5, 10, 20, 60)}
    for n, m in mas.items():
        out[f"ma{n}_slope5"] = F("signed", ((m - m.shift(5)) / 5 / a).to_numpy())
        out[f"dist_ma{n}_atr"] = F("signed", ((c - m) / a).to_numpy())
    al = (
        np.sign(mas[5] - mas[10]) + np.sign(mas[10] - mas[20]) + np.sign(mas[20] - mas[60])
    ).to_numpy()
    out["ma_align"] = F("signed", al)
    side = np.sign(c - mid).to_numpy()
    for n in (3, 5, 10, 20):
        out[f"side_count{n}"] = F("signed", _roll(side, n, "sum", minp=1.0))
    dc = c.diff().to_numpy()
    out["consec_closes"] = F("signed", _run_signed(dc))
    for n in (10, 20):
        er = (c - c.shift(n)).to_numpy() / _roll(np.abs(dc), n, "sum", minp=1.0)
        out[f"eff_ratio{n}"] = F("signed", er)
    hh = ((h > h.shift(1)) & (lo > lo.shift(1))).astype(float).to_numpy()
    ll = ((h < h.shift(1)) & (lo < lo.shift(1))).astype(float).to_numpy()
    out["struct10"] = F("signed", _roll(hh - ll, 10, "sum", minp=1.0))
    hi20 = h.rolling(20, min_periods=20).max().to_numpy()
    lo20 = lo.rolling(20, min_periods=20).min().to_numpy()
    s_hi = _bars_since(h.to_numpy() >= hi20)
    s_lo = _bars_since(lo.to_numpy() <= lo20)
    out["since_ext20_with"] = F("pair", s_hi, s_lo)
    out["since_ext20_against"] = F("pair", s_lo, s_hi)
    cn = c.to_numpy(float)
    for n in (10, 20):
        sl = _roll_slope(cn, n)
        out[f"reg_slope{n}_atr"] = F("signed", sl / an)
        r = pd.Series(cn).rolling(n, min_periods=n).corr(pd.Series(np.arange(len(cn), dtype=float)))
        out[f"reg_r2_{n}"] = F("none", (r**2).to_numpy())
    ax = adx(h, lo, c, 14)
    out["adx14"] = F("none", ax["adx"].to_numpy())
    out["di_diff"] = F("signed", (ax["plus_di"] - ax["minus_di"]).to_numpy())
    ar_up = 100 * (14 - _bars_since(h.to_numpy() >= h.rolling(15, min_periods=15).max().to_numpy()))
    ar_dn = 100 * (
        14 - _bars_since(lo.to_numpy() <= lo.rolling(15, min_periods=15).min().to_numpy())
    )
    ar_up, ar_dn = np.clip(ar_up / 14, 0, 100), np.clip(ar_dn / 14, 0, 100)
    out["aroon_osc"] = F("signed", ar_up - ar_dn)
    for tag, (f_, s_, g_) in {"m6": (6, 13, 6), "m12": (12, 26, 9)}.items():
        dif = c.ewm(span=f_, adjust=False).mean() - c.ewm(span=s_, adjust=False).mean()
        dea = dif.ewm(span=g_, adjust=False).mean()
        hist = (dif - dea).to_numpy()
        out[f"{tag}_dif_atr"] = F("signed", dif.to_numpy() / an)
        out[f"{tag}_dea_atr"] = F("signed", dea.to_numpy() / an)
        out[f"{tag}_hist_atr"] = F("signed", hist / an)
        for k in (1, 3, 5):
            out[f"{tag}_hist_chg{k}"] = F("signed", (hist - _shift(hist, k)) / an)
        out[f"{tag}_dif_side"] = F("signed", np.sign(dif.to_numpy()))
        cs = np.sign(hist)
        out[f"{tag}_cross_side"] = F("signed", cs)
        out[f"{tag}_since_cross"] = F("none", _bars_since(cs != _shift(cs, 1)))
    out["rsi2"] = F("pct100", wilder_rsi(c, 2).to_numpy())
    out["rsi14"] = F("pct100", wilder_rsi(c, 14).to_numpy())
    kd = kdj(h, lo, c, 9, 3, 3)
    for k in ("k", "d", "j"):
        out[f"kdj_{k}"] = F("pct100", kd[k].to_numpy())
    out["kdj_j_minus_d"] = F("signed", (kd["j"] - kd["d"]).to_numpy())
    out["side_duration"] = F("none", _bars_since(side != _shift(side, 1)))
    if v.sum() > 0:
        vn = v.to_numpy(float)
        out["vol_ratio_1_5"] = F("none", vn / _roll(vn, 5, "mean", minp=1.0))
        upv = _roll(np.where(dc > 0, vn, 0.0), 20, "sum", minp=1.0)
        dnv = _roll(np.where(dc < 0, vn, 0.0), 20, "sum", minp=1.0)
        out["updown_vol20"] = F("signed", np.log((upv + 1) / (dnv + 1)))
    return out


def _gather(fs: dict[str, F], i: np.ndarray, prefix: str) -> dict[str, F]:
    ok = i >= 0
    ic = np.clip(i, 0, None)

    def g(x):
        return np.where(ok, x[ic], np.nan) if x is not None else None

    return {f"{prefix}_{k}": F(f.kind, g(f.a), g(f.b)) for k, f in fs.items()}


def bar_features(day: Day, live: bool = True) -> dict[str, F]:
    out: dict[str, F] = {}
    for p in PERIODS:
        b = day.ext if p == 1 else resample_bars(day.ext, f"{p}min")
        b = b[b["close"].notna()]
        close_ns = _ns(b.index) + p * 60 * NS
        i = _asof_idx(close_ns, day.t)
        blk = bar_block(b)
        out.update(_gather(blk, i, f"b{p}"))
        if not live or p not in LIVE_PERIODS:
            continue
        c = b["close"]
        mid = c.rolling(20, min_periods=20).mean().to_numpy()
        sd = c.rolling(20, min_periods=20).std(ddof=0).to_numpy()
        a = atr(b["high"], b["low"], c, 14).replace(0, np.nan).to_numpy()
        ok, ic = i >= 0, np.clip(i, 0, None)
        m, s_, at = (np.where(ok, x[ic], np.nan) for x in (mid, sd, a))
        up, dn = m + 2 * s_, m - 2 * s_
        S = day.S
        out[f"live{p}_pctb"] = F("pct1", (S - dn) / np.where(up > dn, up - dn, np.nan))
        out[f"live{p}_dist_mid_atr"] = F("signed", (S - m) / at)
        # 正在形成的 K：起点 = 上一根已完成 K 的收盘时刻
        start = np.where(ok, close_ns[ic], day.t[0])
        out[f"live{p}_elapsed_s"] = F("none", (day.t - start) / NS)
        j0 = np.clip(np.searchsorted(day.t, start, side="left"), 0, len(day.t) - 1)
        o = S[j0]
        key = pd.Series(start)
        hi = pd.Series(S).groupby(key.values).cummax().to_numpy()
        lw = pd.Series(S).groupby(key.values).cummin().to_numpy()
        out[f"live{p}_body_atr"] = F("signed", (S - o) / at)
        uw, lwk = (hi - np.fmax(o, S)) / at, (np.fmin(o, S) - lw) / at
        out[f"live{p}_wick_with_atr"] = F("pair", uw, lwk)
        out[f"live{p}_wick_against_atr"] = F("pair", lwk, uw)
    return out


# ---------------- 秒级 ----------------


def sec_features(day: Day) -> dict[str, F]:
    S, t = day.S, day.t
    out: dict[str, F] = {}
    for w in SEC_RET:
        out[f"s_ret{w}_bp"] = F("signed", (S / _shift(S, w) - 1) * 1e4)
    r1 = (S / _shift(S, 1) - 1) * 1e4
    for w in (60, 300, 600):
        out[f"s_rv{w}_bp"] = F("none", _roll(r1, w, "std"))
    for w in (30, 120):
        out[f"s_jump{w}_bp"] = F("none", _roll(np.abs(r1), w, "max"))
        mx, mn = _roll(r1, w, "max"), -_roll(r1, w, "min")
        out[f"s_jump_with{w}_bp"] = F("pair", mx, mn)
        out[f"s_jump_against{w}_bp"] = F("pair", mn, mx)
    sg = np.sign(r1)
    for w in (60, 300):
        nz = _roll(np.abs(sg), w, "sum")
        out[f"s_upshare{w}"] = F("signed", _roll(sg, w, "sum") / np.where(nz > 0, nz, np.nan))
    for n in EXT_MIN:
        w = n * 60
        hi = pd.Series(S).rolling(w, min_periods=1).max().to_numpy()
        lo = pd.Series(S).rolling(w, min_periods=1).min().to_numpy()
        th, tl = (hi / S - 1) * 1e4, (1 - lo / S) * 1e4
        out[f"s_ext_with{n}m_bp"] = F("pair", th, tl)
        out[f"s_ext_against{n}m_bp"] = F("pair", tl, th)
        out[f"s_at_ext_with{n}m"] = F("pair", (th <= TOL_BP) * 1.0, (tl <= TOL_BP) * 1.0)
    r = day.rth[_und(day)]
    td = r[et_date(r.index) == day.day]
    prev = r[et_date(r.index) < day.day]
    cl = _ns(td.index) + 60 * NS
    i = _asof_idx(cl, t)
    ok, ic = i >= 0, np.clip(i, 0, None)
    bh = np.maximum.accumulate(td["high"].to_numpy()) if len(td) else np.array([np.nan])
    bl = np.minimum.accumulate(td["low"].to_numpy()) if len(td) else np.array([np.nan])
    run_hi = np.fmax.accumulate(np.where(np.isfinite(S), S, -np.inf))
    run_lo = np.fmin.accumulate(np.where(np.isfinite(S), S, np.inf))
    dh = np.fmax(np.where(ok, bh[ic], -np.inf), run_hi)
    dl = np.fmin(np.where(ok, bl[ic], np.inf), run_lo)
    th, tl = (dh / S - 1) * 1e4, (1 - dl / S) * 1e4
    out["s_ext_with_day_bp"] = F("pair", th, tl)
    out["s_ext_against_day_bp"] = F("pair", tl, th)
    out["s_at_ext_with_day"] = F("pair", (th <= TOL_BP) * 1.0, (tl <= TOL_BP) * 1.0)
    if len(td):
        tp = (td["high"] + td["low"] + td["close"]).to_numpy() / 3
        vv = td["volume"].to_numpy(float)
        vwap = np.cumsum(tp * vv) / np.where(np.cumsum(vv) > 0, np.cumsum(vv), np.nan)
        out["s_vs_vwap_bp"] = F("signed", np.where(ok, (S / vwap[ic] - 1) * 1e4, np.nan))
        op = float(td["open"].iloc[0])
        op_ok = ok  # 首根 1 分钟 K 收盘后才用开盘价
        out["s_vs_open_bp"] = F("signed", np.where(op_ok, (S / op - 1) * 1e4, np.nan))
    else:
        op = np.nan
        out["s_vs_vwap_bp"] = F("signed", np.full(len(t), np.nan))
        out["s_vs_open_bp"] = F("signed", np.full(len(t), np.nan))
    pc = float(prev["close"].iloc[-1]) if len(prev) else np.nan
    out["s_vs_prevclose_bp"] = F("signed", (S / pc - 1) * 1e4)
    out["s_gap_bp"] = F("signed", np.where(ok, (op / pc - 1) * 1e4, np.nan) * np.ones(len(t)))
    return out


def _und(day: Day) -> str:
    return next(u for u in day.rth if u not in CROSS)


# ---------------- 成交量 ----------------


def _same_time_mean(r: pd.DataFrame, day: dt.date, n_days: int, win: int) -> pd.Series:
    """前 n_days 个交易日、同一分钟（含此前 win−1 分钟）的成交量和的均值；索引 = 分钟数（ET）。"""
    et = r.index.tz_convert(ET)
    df = pd.DataFrame({"d": et.date, "m": et.hour * 60 + et.minute, "v": r["volume"].to_numpy()})
    df = df[df["d"] < day]
    days = sorted(df["d"].unique())[-n_days:]
    df = df[df["d"].isin(days)]
    pv = df.pivot_table(index="m", columns="d", values="v", aggfunc="sum").sort_index()
    pv = pv.rolling(win, min_periods=1).sum()
    return pv.mean(axis=1)


def vol_features(day: Day) -> dict[str, F]:
    r = day.rth[_und(day)]
    td = r[et_date(r.index) == day.day]
    out: dict[str, F] = {}
    t = day.t
    if not len(td):
        return {k: F("none", np.full(len(t), np.nan)) for k in ("v_rel1_20d", "v_rel5_20d")}
    cl = _ns(td.index) + 60 * NS
    i = _asof_idx(cl, t)
    ok, ic = i >= 0, np.clip(i, 0, None)
    et = td.index.tz_convert(ET)
    mins = np.asarray(et.hour * 60 + et.minute)
    vv = td["volume"].to_numpy(float)
    for w in (1, 5):
        base = _same_time_mean(r, day.day, 20, w).reindex(mins).to_numpy()
        cur = pd.Series(vv).rolling(w, min_periods=1).sum().to_numpy()
        out[f"v_rel{w}_20d"] = F("none", np.where(ok, (cur / base)[ic], np.nan))
    return out


# ---------------- 期权 ----------------


def opt_features(day: Day) -> dict[str, F]:
    o, S, t = day.opt, day.S, day.t
    out: dict[str, F] = {}
    cm, pm = o["c_mid"].to_numpy(), o["p_mid"].to_numpy()
    cs, ps = o["c_spread"].to_numpy(), o["p_spread"].to_numpy()
    out["o_spread_pct"] = F("pair", cs / cm * 100, ps / pm * 100)
    tick_c = np.where(cm >= 3, 0.05, 0.01)
    tick_p = np.where(pm >= 3, 0.05, 0.01)
    out["o_spread_ticks"] = F("pair", cs / tick_c, ps / tick_p)
    n = o["n_upd"].to_numpy()
    for w in (10, 30, 60):
        out[f"o_updates{w}s"] = F("none", n - _shift(n, w))
    base = (_shift(n, 60) - _shift(n, 120)) / 60
    out["o_update_accel10"] = F("none", (n - _shift(n, 10)) / 10 / np.where(base > 0, base, np.nan))
    k = o["strike"].to_numpy()
    for w in (30, 60, 120):
        same = k == _shift(k, w)  # 换档前后不是同一合约，不算
        cc = np.where(same, (cm / _shift(cm, w) - 1) * 100, np.nan)
        pc = np.where(same, (pm / _shift(pm, w) - 1) * 100, np.nan)
        out[f"o_mid_chg{w}s_pct"] = F("pair", cc, pc)
    T = np.maximum((o["exp_close"].to_numpy() - t) / (365 * 24 * 3600e9), 1 / (365 * 24 * 60))
    iv = (cm + pm) / (0.8 * S * np.sqrt(T)) * 100  # Brenner–Subrahmanyam 跨式近似（%）
    out["o_iv_atm"] = F("none", iv)
    for m in (5, 15):
        out[f"o_iv_chg{m}m"] = F("none", iv - _shift(iv, m * 60))
    out["o_dte"] = F("none", o["dte"].to_numpy())
    out["o_expiry_day"] = F("none", (o["dte"].to_numpy() == 0) * 1.0)
    return out


def optvol_features(day: Day) -> dict[str, F]:
    t = day.t
    ov = day.optvol
    names = ("o_cp_vol_ratio5m", "o_cp_vol_ratio15m", "o_near_vol5m_20d")
    if ov is None or not len(ov):
        return {k: F("signed" if "cp" in k else "none", np.full(len(t), np.nan)) for k in names}
    i = _asof_idx(ov.index.to_numpy() + 60 * NS, t)
    ok, ic = i >= 0, np.clip(i, 0, None)
    out: dict[str, F] = {}
    for m in (5, 15):
        c = ov["call"].rolling(m, min_periods=1).sum().to_numpy()
        p = ov["put"].rolling(m, min_periods=1).sum().to_numpy()
        out[f"o_cp_vol_ratio{m}m"] = F(
            "signed", np.where(ok, np.log((c + 1) / (p + 1))[ic], np.nan)
        )
    near5 = ov["near"].rolling(5, min_periods=1).sum()
    hist = []
    for h in day.optvol_hist:
        hist.append(pd.Series(h["near"].rolling(5, min_periods=1).sum().to_numpy(), index=h["mod"]))
    if hist:
        base = pd.concat(hist, axis=1).groupby(level=0).mean().mean(axis=1)
        b = base.reindex(ov["mod"].to_numpy()).to_numpy()
        out["o_near_vol5m_20d"] = F("none", np.where(ok, (near5.to_numpy() / b)[ic], np.nan))
    else:
        out["o_near_vol5m_20d"] = F("none", np.full(len(t), np.nan))
    return out


# ---------------- 日历 / 跨标的 ----------------


def cal_features(day: Day) -> dict[str, F]:
    t = day.t
    open_ns = t[0]
    close_ns = pd.Timestamp(dt.datetime.combine(day.day, session_close(day.day)), tz=ET).value
    rel = day.releases if len(day.releases) else np.array([np.iinfo(np.int64).max])
    j = np.searchsorted(rel, t, side="left")
    nxt = np.where(j < len(rel), rel[np.clip(j, 0, len(rel) - 1)], np.nan)
    prv = np.where(j > 0, rel[np.clip(j - 1, 0, None)], np.nan)
    to_next = (nxt - t) / 60e9
    since = (t - prv) / 60e9
    return {
        "c_min_from_open": F("none", (t - open_ns) / 60e9),
        "c_min_to_close": F("none", (close_ns - t) / 60e9),
        "c_weekday": F("none", np.full(len(t), float(day.day.weekday()))),
        "c_min_to_release": F("none", np.where(to_next <= 390, to_next, np.nan)),
        "c_min_since_release": F("none", np.where(since <= 390, since, np.nan)),
    }


def cross_features(day: Day) -> dict[str, F]:
    t = day.t
    out: dict[str, F] = {}
    rets = {}
    for u in (_und(day), *CROSS):
        r = day.rth.get(u)
        if r is None or not len(r):
            continue
        td = r[et_date(r.index) == day.day]
        if not len(td):
            continue
        cl = _ns(td.index) + 60 * NS
        i = _asof_idx(cl, t)
        ok, ic = i >= 0, np.clip(i, 0, None)
        c = td["close"].to_numpy(float)
        for m in (1, 5, 15):
            rr = (c / _shift(c, m) - 1) * 1e4
            rets[(u, m)] = np.where(ok, rr[ic], np.nan)
        if u == "SPY":
            lr = np.log(c / _shift(c, 1)) * 1e4
            out["x_spy_rv30"] = F("none", np.where(ok, _roll(lr, 30, "std")[ic], np.nan))
    for u in CROSS:
        for m in (1, 5, 15):
            if (u, m) in rets:
                out[f"x_{u.lower()}_ret{m}m_bp"] = F("signed", rets[(u, m)])
    for m in (5, 15):
        if (_und(day), m) in rets and ("SMH", m) in rets:
            out[f"x_rel_smh{m}m_bp"] = F("signed", rets[(_und(day), m)] - rets[("SMH", m)])
    return out


# ---------------- 暴力因子 ----------------


def brute(day: Day) -> dict[str, F]:
    """算子库 × 原始序列 × 窗口（秒）BRUTE_W。
    价格（对数价，方向化）：差分、z 分数、窗口内秩、OLS 斜率；
    非方向序列（|1 秒收益|、两腿价差 %、每秒报价更新数、最近 1 分钟成交量、最近 1 分钟振幅）：
    均值 / 4 倍窗口均值、z 分数、窗口内秩、变异系数；以及「z 分数 × 同窗口价格方向」（方向化）。"""
    S, t = day.S, day.t
    lp = np.log(S)
    out: dict[str, F] = {}
    o = day.opt
    spr = (o["c_spread"] / o["c_mid"] + o["p_spread"] / o["p_mid"]).to_numpy() * 50
    upd = np.r_[np.nan, np.diff(o["n_upd"].to_numpy())]
    r = day.rth[_und(day)]
    td = r[et_date(r.index) == day.day]
    if len(td):
        i = _asof_idx(_ns(td.index) + 60 * NS, t)
        ok, ic = i >= 0, np.clip(i, 0, None)
        vol1 = np.where(ok, td["volume"].to_numpy(float)[ic], np.nan)
        rng1 = np.where(ok, ((td["high"] - td["low"]) / td["close"]).to_numpy()[ic] * 1e4, np.nan)
    else:
        vol1 = rng1 = np.full(len(t), np.nan)
    series = {
        "absret": np.abs(np.r_[np.nan, np.diff(lp)]) * 1e4,
        "spread": spr,
        "updates": upd,
        "vol1m": vol1,
        "range1m": rng1,
    }
    for w in BRUTE_W:
        m, sd = _roll(lp, w, "mean"), _roll(lp, w, "std")
        out[f"f_px_diff{w}"] = F("signed", (lp - _shift(lp, w)) * 1e4)
        out[f"f_px_z{w}"] = F("signed", (lp - m) / np.where(sd > 0, sd, np.nan))
        rk = pd.Series(lp).rolling(w, min_periods=max(w // 2, 2)).rank(pct=True).to_numpy()
        out[f"f_px_rank{w}"] = F("pct1", rk)
        out[f"f_px_slope{w}"] = F("signed", _roll_slope(lp, w) * 1e4 * w)
        pdir = np.sign(lp - _shift(lp, w))
        for nm, x in series.items():
            mw, sw = _roll(x, w, "mean"), _roll(x, w, "std")
            m4 = _roll(x, 4 * w, "mean")
            z = (x - mw) / np.where(sw > 0, sw, np.nan)
            out[f"f_{nm}_m{w}_m{4 * w}"] = F("none", mw / np.where(m4 > 0, m4, np.nan))
            out[f"f_{nm}_z{w}"] = F("none", z)
            rk = pd.Series(x).rolling(w, min_periods=max(w // 2, 2)).rank(pct=True).to_numpy()
            out[f"f_{nm}_rank{w}"] = F("none", rk)
            out[f"f_{nm}_cv{w}"] = F("none", sw / np.where(mw > 0, mw, np.nan))
            out[f"f_{nm}_zdir{w}"] = F("signed", z * pdir)
    return out


# ---------------- 汇总 ----------------

FAMILY_FN = {
    "sec": sec_features,
    "vol": vol_features,
    "opt": opt_features,
    "optvol": optvol_features,
    "cal": cal_features,
    "cross": cross_features,
    "brute": brute,
}


def day_features(day: Day, families: tuple[str, ...] = FAMILIES) -> dict[str, F]:
    out: dict[str, F] = {}
    if "bar" in families or "live" in families:
        bf = bar_features(day, live="live" in families)
        out.update(
            {
                k: v
                for k, v in bf.items()
                if ("bar" in families and k.startswith("b")) or k.startswith("live")
            }
        )
    for fam in families:
        if fam in ("bar", "live"):
            continue
        out.update(FAMILY_FN[fam](day))
    return out


def family_of(name: str) -> str:
    if name.startswith("live"):
        return "live"
    if name.startswith("b") and name[1].isdigit():
        return "bar"
    return {
        "s": "sec",
        "v": "vol",
        "c": "cal",
        "x": "cross",
        "f": "brute",
    }.get(name[0], "optvol" if name.startswith(("o_cp", "o_near")) else "opt")


def sample(fs: dict[str, F], rows: np.ndarray, d: np.ndarray) -> pd.DataFrame:
    """取网格行 rows 的方向化值（d 与 rows 等长）。"""
    return pd.DataFrame({k: _dir_rows(f, rows, d) for k, f in fs.items()})


def _dir_rows(f: F, rows: np.ndarray, d: np.ndarray) -> np.ndarray:
    a = f.a[rows]
    b = f.b[rows] if f.b is not None else None
    return F(f.kind, a, b).directional(d)


def grid_rows(day: Day, t_ns: np.ndarray) -> np.ndarray:
    """时刻 → 网格行（取 ≤ t 的最后一个整秒）。"""
    return np.clip(np.searchsorted(day.t, t_ns, side="right") - 1, 0, len(day.t) - 1)


def truncation_check(
    ctx: Ctx, day: dt.date, t_list: np.ndarray, families: tuple[str, ...] = FAMILIES
) -> pd.DataFrame:
    """全日数据 vs 截断到 t 的数据，在 t 处逐特征比较（两个方向）。返回 特征 × 不一致次数。"""
    full = load_day(ctx, day)
    ff = day_features(full, families)
    bad = {k: 0 for k in ff}
    for t in t_list:
        tr = load_day(ctx, day, t_cut=int(t))
        tf = day_features(tr, families)
        r_full = grid_rows(full, np.array([t]))
        r_tr = grid_rows(tr, np.array([t]))
        for k, f in ff.items():
            for d in (1, -1):
                a = _dir_rows(f, r_full, np.array([d]))[0]
                g = tf.get(k)
                b = _dir_rows(g, r_tr, np.array([d]))[0] if g is not None else np.nan
                same = (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, rtol=1e-7, atol=1e-9)
                if not same:
                    bad[k] += 1
    return pd.DataFrame({"feature": list(bad), "mismatch": list(bad.values())})
