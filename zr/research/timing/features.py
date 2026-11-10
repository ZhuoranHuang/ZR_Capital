# ruff: noqa: E501  （中文文档字符串与注释较长）
"""特征库（夜间择时研究 R-055 归档）。build(ext1, ref1) → 与 ext1 逐行对齐（每根 1 分钟 K 收盘时刻的值）。

口径 C：ext1 = 含盘前盘后的长桥原始 1 分钟 K，指标跨日连续；周期 P ∈ {1, 2, 5} 的已完成 K 由
resample_bars 聚合，按「K 收盘时刻 ≤ 当前 1 分钟 K 收盘时刻」对齐；另算正在形成的 2 / 5 分钟 K
（用其内部已完成的 1 分钟 K 合成）的同名特征（前缀 f2_ / f5_）。RTH 相关特征按交易日历收盘时刻
判定 RTH；日级汇总一律先得到日序列、按日 shift(1)、再按日期映射回分钟行。
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from zr.features.bars import ET, resample_bars
from zr.features.indicators import adx

REG: dict[str, tuple[str, str, str]] = {}  # 名 → (组, 定义, 可用时点)


def reg(name: str, group: str, desc: str, avail: str = "该 1 分钟 K 收盘"):
    REG[name] = (group, desc, avail)


def run_len(f) -> np.ndarray:
    f = np.asarray(f, bool)
    idx = np.arange(len(f))
    lf = np.maximum.accumulate(np.where(~f, idx, -1))
    return np.where(f, idx - lf, 0)


def signed_run(pos, neg) -> np.ndarray:
    return run_len(pos) - run_len(neg)


def bars_since(f) -> np.ndarray:
    f = np.asarray(f, bool)
    idx = np.arange(len(f))
    lt = np.maximum.accumulate(np.where(f, idx, -1))
    return np.where(lt >= 0, idx - lt, np.nan).astype(float)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(c: pd.Series, n: int) -> pd.Series:
    dl = c.diff()
    g, l_ = _wilder(dl.clip(lower=0), n), _wilder((-dl).clip(lower=0), n)
    return (100 - 100 / (1 + g / l_)).mask(l_ == 0, 100.0)


def kdj50(h, l_, c, n=9):
    """KDJ(9,3,3)，K、D 初值 50。"""
    ln, hn = l_.rolling(n, min_periods=n).min(), h.rolling(n, min_periods=n).max()
    rsv = ((c - ln) / (hn - ln).replace(0, np.nan) * 100).fillna(50.0)
    k = (
        pd.concat([pd.Series([50.0]), rsv.reset_index(drop=True)])
        .ewm(alpha=1 / 3, adjust=False)
        .mean()
        .iloc[1:]
        .to_numpy()
    )
    k = pd.Series(k, index=c.index)
    d = (
        pd.concat([pd.Series([50.0]), k.reset_index(drop=True)])
        .ewm(alpha=1 / 3, adjust=False)
        .mean()
        .iloc[1:]
        .to_numpy()
    )
    d = pd.Series(d, index=c.index)
    return k, d, 3 * k - 2 * d, ln, hn


def atr14(h, l_, c):
    tr = pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
    return _wilder(tr, 14)


def waves(hist: np.ndarray):
    """MACD 柱波段：当前柱 ÷ 上一同向波段最大 |柱|；本波段至今最大 ÷ 上一同向波段最大。"""
    s = np.where(hist >= 0, 1, -1)
    s = np.where(np.isfinite(hist), s, 0)
    wid = np.cumsum(np.r_[1, (s[1:] != s[:-1]).astype(int)])
    a = pd.Series(np.abs(hist))
    cur = a.groupby(wid).cummax().to_numpy()
    fin = a.groupby(wid).max()
    prev_same = pd.Series(wid - 2).map(fin).to_numpy()  # 上一同向波段（已结束）
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(hist) / prev_same, cur / prev_same


def supertrend_dir(h, l_, c, n=10, k=3.0) -> np.ndarray:
    a = (
        atr14(h, l_, c)
        if n == 14
        else _wilder(
            pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1),
            n,
        )
    )
    hl2 = ((h + l_) / 2).to_numpy()
    av, cv = a.to_numpy(), c.to_numpy()
    ub, lb = hl2 + k * av, hl2 - k * av
    out = np.zeros(len(cv))
    fu, fl, d = np.nan, np.nan, 1
    for i in range(len(cv)):
        if not np.isfinite(av[i]):
            continue
        fu = ub[i] if not np.isfinite(fu) or ub[i] < fu or cv[i - 1] > fu else fu
        fl = lb[i] if not np.isfinite(fl) or lb[i] > fl or cv[i - 1] < fl else fl
        if d == 1 and cv[i] < fl:
            d = -1
        elif d == -1 and cv[i] > fu:
            d = 1
        out[i] = d
    return out


def linreg(c: pd.Series, n: int = 20):
    idx = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    cov = c.rolling(n, min_periods=n).cov(idx)
    var = idx.rolling(n, min_periods=n).var()
    corr = c.rolling(n, min_periods=n).corr(idx)
    return (cov / var) / c * 1e4, corr**2


def aroon(h, l_, n=25):
    from numpy.lib.stride_tricks import sliding_window_view

    hv, lv = h.to_numpy(), l_.to_numpy()
    out = np.full(len(hv), np.nan)
    if len(hv) > n:
        wh, wl = sliding_window_view(hv, n + 1), sliding_window_view(lv, n + 1)
        up = 100 * wh.argmax(1) / n
        dn = 100 * wl.argmin(1) / n
        out[n:] = up - dn
    return out


# ---------------- 周期块（已完成 K） ----------------


def block(b: pd.DataFrame, P: int) -> pd.DataFrame:
    c, h, l_, v = b["close"], b["high"], b["low"], b["volume"]
    p = f"p{P}_"
    out = {}
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    up, lo = mid + 2 * sd, mid - 2 * sd
    a = atr14(h, l_, c).replace(0, np.nan)
    bw = (up - lo) / mid
    out["pctb"] = (c - lo) / (up - lo).replace(0, np.nan)
    out["bw"] = bw * 100
    out["bw_rank"] = bw.rolling(5 * 960 // P, min_periods=200 // P).rank(pct=True)
    for k in (1, 3, 5, 10):
        out[f"mid_slope{k}"] = (mid - mid.shift(k)) / k / c * 1e4
    out["up_slope5"] = (up - up.shift(5)) / 5 / c * 1e4
    out["lo_slope5"] = (lo - lo.shift(5)) / 5 / c * 1e4
    out["bw_chg5"] = bw / bw.shift(5) - 1
    out["dist_mid_hw"] = (c - mid) / (2 * sd).replace(0, np.nan)
    out["dist_mid_atr"] = (c - mid) / a
    out["side_run"] = signed_run(c > mid, c < mid)
    out["since_touch_mid"] = bars_since((l_ <= mid) & (h >= mid))
    out["ride_run"] = signed_run(h >= up, l_ <= lo)
    hh, ll = h.rolling(20, min_periods=20).max(), l_.rolling(20, min_periods=20).min()
    rngv = (hh - ll).replace(0, np.nan)
    out["pb_depth_up"] = (hh - c) / rngv
    out["pb_depth_dn"] = (c - ll) / rngv
    kc_mid = _ema(c, 20)
    a20 = _wilder(
        pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1), 20
    )
    sq = ((up < kc_mid + 1.5 * a20) & (lo > kc_mid - 1.5 * a20)).to_numpy()
    out["squeeze"] = sq.astype(float)
    out["squeeze_run"] = run_len(sq)
    rel = np.r_[False, (~sq[1:]) & sq[:-1]]
    out["since_release"] = np.where(sq, np.nan, bars_since(rel))
    for tag, (f_, s_, g_) in {"m6": (6, 13, 6), "m12": (12, 26, 9)}.items():
        dif = _ema(c, f_) - _ema(c, s_)
        dea = _ema(dif, g_)
        hist = dif - dea
        hv = hist.to_numpy()
        out[f"{tag}_dif_atr"] = dif / a
        out[f"{tag}_dea_atr"] = dea / a
        out[f"{tag}_hist_atr"] = hist / a
        out[f"{tag}_hist_chg1"] = (hist - hist.shift(1)) / a
        out[f"{tag}_hist_chg3"] = (hist - hist.shift(3)) / a
        out[f"{tag}_zero"] = np.sign(dif)
        sg = np.sign(hv)
        out[f"{tag}_since_cross"] = bars_since(np.r_[False, sg[1:] != sg[:-1]])
        cur_vs, wave_vs = waves(hv)
        out[f"{tag}_hist_vs_prevwave"] = cur_vs
        out[f"{tag}_wave_vs_prevwave"] = wave_vs
        hi20 = c >= c.rolling(20, min_periods=20).max()
        lo20 = c <= c.rolling(20, min_periods=20).min()
        out[f"{tag}_div"] = np.where(
            hi20 & (hist < hist.rolling(20, min_periods=20).max()),
            -1,
            np.where(lo20 & (hist > hist.rolling(20, min_periods=20).min()), 1, 0),
        )
        ah = np.abs(hv)
        out[f"{tag}_shrink_run"] = run_len(np.r_[False, ah[1:] < ah[:-1]])
    k, d, j, _, _ = kdj50(h, l_, c)
    out["k"], out["d"], out["j"] = k, d, j
    ks = np.sign((k - d).to_numpy())
    out["kd_side"] = ks
    out["since_kdcross"] = bars_since(np.r_[False, ks[1:] != ks[:-1]])
    out["j_ext_run"] = signed_run(j > 100, j < 0)
    for n in (2, 6, 14):
        out[f"rsi{n}"] = rsi(c, n)
    r14 = out["rsi14"]
    out["rsi14_chg1"], out["rsi14_chg3"] = r14 - r14.shift(1), r14 - r14.shift(3)
    hi20 = c >= c.rolling(20, min_periods=20).max()
    lo20 = c <= c.rolling(20, min_periods=20).min()
    out["rsi14_div"] = np.where(
        hi20 & (r14 < r14.rolling(20, min_periods=20).max()),
        -1,
        np.where(lo20 & (r14 > r14.rolling(20, min_periods=20).min()), 1, 0),
    )
    # 量
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    vs10 = v.rolling(10, min_periods=10).sum().replace(0, np.nan)
    out["obv_slope10"] = (obv - obv.shift(10)) / vs10
    out["obv_div"] = np.where(
        hi20 & (obv < obv.rolling(20, min_periods=20).max()),
        -1,
        np.where(lo20 & (obv > obv.rolling(20, min_periods=20).min()), 1, 0),
    )
    tp = (h + l_ + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=14).sum()
    out["mfi14"] = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    dc = c.diff()
    upv = v.where(dc > 0, 0.0).rolling(20, min_periods=20).sum()
    dnv = v.where(dc < 0, 0.0).rolling(20, min_periods=20).sum()
    out["updown_vol20"] = np.log((upv + 1) / (dnv + 1))
    # 波动与趋势质量
    out["atr_pct"] = a / c * 100
    out["range_exp"] = (h - l_) / a
    out["er10"] = (c - c.shift(10)).abs() / c.diff().abs().rolling(
        10, min_periods=10
    ).sum().replace(0, np.nan)
    ax = adx(h, l_, c, 14)
    out["adx14"], out["di_diff"] = ax["adx"], ax["plus_di"] - ax["minus_di"]
    sl, r2 = linreg(c, 20)
    out["linreg_slope20"], out["linreg_r2_20"] = sl, r2
    out["aroon25"] = aroon(h, l_, 25)
    out["supertrend"] = supertrend_dir(h, l_, c, 10, 3.0)
    df = pd.DataFrame(out, index=b.index)
    return df.add_prefix(p)


def forming(e1: pd.DataFrame, b: pd.DataFrame, P: int) -> pd.DataFrame:
    """正在形成的 P 分钟 K（P ∈ {2, 5}）的同名特征：用此前已完成的 P 分钟 K 状态 + 当前 K 内已完成的 1 分钟 K。"""
    c, h, l_ = b["close"], b["high"], b["low"]
    lab = b.index.as_unit("ns").asi8
    t1 = e1.index.as_unit("ns").asi8
    Pn = P * 60 * 10**9
    # 当前 1 分钟 K 所在 P 分钟 K 的标签（整分钟网格，与 resample_bars origin=start_day 一致）
    start = (e1.index.tz_convert(ET).floor(f"{P}min")).tz_convert("UTC").as_unit("ns").asi8
    i = np.searchsorted(lab, start, side="left") - 1  # 上一根已完成 P 分钟 K
    ok = i >= 0
    ic = np.clip(i, 0, None)
    grp = pd.Series(start)
    cf = e1["close"].to_numpy()
    hf = pd.Series(e1["high"].to_numpy()).groupby(grp.values).cummax().to_numpy()
    lf = pd.Series(e1["low"].to_numpy()).groupby(grp.values).cummin().to_numpy()
    s19, q19 = (
        c.rolling(19, min_periods=19).sum().to_numpy(),
        (c**2).rolling(19, min_periods=19).sum().to_numpy(),
    )
    mid = (s19[ic] + cf) / 20
    var = (q19[ic] + cf**2) / 20 - mid**2
    sd = np.sqrt(np.clip(var, 0, None))
    up, lo = mid + 2 * sd, mid - 2 * sd
    a = atr14(h, l_, c).to_numpy()[ic]
    e6, e13 = _ema(c, 6).to_numpy()[ic], _ema(c, 13).to_numpy()[ic]
    dif_c = _ema(c, 6) - _ema(c, 13)
    dea_c = _ema(dif_c, 6).to_numpy()[ic]
    e6f = e6 + (2 / 7) * (cf - e6)
    e13f = e13 + (2 / 14) * (cf - e13)
    dif = e6f - e13f
    dea = dea_c + (2 / 7) * (dif - dea_c)
    dl = c.diff()
    ag, al = (
        _wilder(dl.clip(lower=0), 14).to_numpy()[ic],
        _wilder((-dl).clip(lower=0), 14).to_numpy()[ic],
    )
    ch = cf - c.to_numpy()[ic]
    agf, alf = (
        ag + (max(0, 0) + np.clip(ch, 0, None) - ag) / 14,
        al + (np.clip(-ch, 0, None) - al) / 14,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        rsif = np.where(alf == 0, 100.0, 100 - 100 / (1 + agf / alf))
    k, d, _, _, _ = kdj50(h, l_, c)
    ll8, hh8 = (
        l_.rolling(8, min_periods=8).min().to_numpy()[ic],
        h.rolling(8, min_periods=8).max().to_numpy()[ic],
    )
    lnf, hnf = np.minimum(ll8, lf), np.maximum(hh8, hf)
    with np.errstate(invalid="ignore", divide="ignore"):
        rsv = np.where(hnf > lnf, (cf - lnf) / (hnf - lnf) * 100, 50.0)
        rsv = np.where(np.isfinite(rsv), rsv, 50.0)
        kf = k.to_numpy()[ic] + (rsv - k.to_numpy()[ic]) / 3
        df_ = d.to_numpy()[ic] + (kf - d.to_numpy()[ic]) / 3
        out = {
            "pctb": (cf - lo) / np.where(up > lo, up - lo, np.nan),
            "dist_mid_hw": (cf - mid) / np.where(sd > 0, 2 * sd, np.nan),
            "bw": (up - lo) / mid * 100,
            "m6_dif_atr": dif / a,
            "m6_dea_atr": dea / a,
            "m6_hist_atr": (dif - dea) / a,
            "rsi14": rsif,
            "k": kf,
            "d": df_,
            "j": 3 * kf - 2 * df_,
        }
    f = pd.DataFrame(out, index=e1.index)
    f[~ok] = np.nan
    _ = (t1, Pn)
    return f.add_prefix(f"f{P}_")


def _align(blk: pd.DataFrame, P: int, idx: pd.DatetimeIndex) -> pd.DataFrame:
    close_t = (blk.index + pd.Timedelta(minutes=P)).as_unit("ns").asi8
    dec = (idx + pd.Timedelta(minutes=1)).as_unit("ns").asi8
    i = np.searchsorted(close_t, dec, side="right") - 1
    v = blk.to_numpy(np.float64)
    out = np.where((i >= 0)[:, None], v[np.clip(i, 0, None)], np.nan)
    return pd.DataFrame(out, index=idx, columns=blk.columns)


@functools.lru_cache(maxsize=4096)
def _close_min(day) -> int:
    from zr.data.calendar import session_close

    t = session_close(day)
    return t.hour * 60 + t.minute


# ---------------- 主函数 ----------------


def build(e1: pd.DataFrame, ref1: pd.DataFrame | None) -> pd.DataFrame:
    e1 = e1[["open", "high", "low", "close", "volume"]].astype(float)
    idx = e1.index
    parts = []
    for P in (1, 2, 5):
        b = e1 if P == 1 else resample_bars(e1, f"{P}min")
        b = b[b["close"].notna()]
        parts.append(_align(block(b, P), P, idx))
        if P > 1:
            parts.append(forming(e1, b, P))
    F = pd.concat(parts, axis=1)
    # 多周期一致性（1、2、5 分钟中轨斜率与 MACD 柱同号的个数，带符号）
    cons = np.zeros(len(F))
    for P in (1, 2, 5):
        s, hh = (
            np.sign(F[f"p{P}_mid_slope3"].to_numpy()),
            np.sign(F[f"p{P}_m6_hist_atr"].to_numpy()),
        )
        cons += np.where((s == hh) & np.isfinite(s), s, 0)
    F["mtf_consistency"] = cons
    F = pd.concat([F, session_features(e1, ref1)], axis=1)
    return F.astype(np.float32)


def session_features(e1: pd.DataFrame, ref1: pd.DataFrame | None) -> pd.DataFrame:
    idx = e1.index
    et = idx.tz_convert(ET)
    day = np.asarray(et.date)
    mod = np.asarray(et.hour * 60 + et.minute)
    cmin = np.array([_close_min(d) for d in day]) if len(day) else np.array([])
    rth = (mod >= 570) & (mod < cmin)
    o, h, l_, c, v = (e1[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    out: dict[str, np.ndarray] = {}
    R = pd.DataFrame({"day": day, "mod": mod, "o": o, "h": h, "l": l_, "c": c, "v": v})[rth]
    # 日级（RTH）：先日序列，再按日 shift(1)，再映射
    g = R.groupby("day")
    D = pd.DataFrame(
        {"o": g["o"].first(), "h": g["h"].max(), "l": g["l"].min(), "c": g["c"].last()}
    )
    prev = D.shift(1)
    pmap = lambda s: pd.Series(day).map(s).to_numpy(float)  # noqa: E731
    pc, ph, pl = (np.where(rth, pmap(prev[k]), np.nan) for k in ("c", "h", "l"))  # 只在 RTH 行使用
    # 当日开盘（首根 RTH 开盘价，开盘即可用）：用「日序列的 first」映射，只映射到 RTH 及之后的行
    dopen = pmap(D["o"])
    dopen = np.where(mod >= 570, dopen, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["prev_close_dist"] = (c / pc - 1) * 100
        out["prev_high_dist"] = (c / ph - 1) * 100
        out["prev_low_dist"] = (c / pl - 1) * 100
        out["gap"] = np.where(rth, (dopen / pc - 1) * 100, np.nan)
        P_ = (ph + pl + pc) / 3
        for nm, lv in (
            ("P", P_),
            ("R1", 2 * P_ - pl),
            ("S1", 2 * P_ - ph),
            ("R2", P_ + (ph - pl)),
            ("S2", P_ - (ph - pl)),
        ):
            out[f"pivot_{nm}_dist"] = (c / lv - 1) * 100
        out["ret_since_open"] = np.where(rth, (c / dopen - 1) * 100, np.nan)
        out["ret_since_prev_close"] = np.where(rth, (c / pc - 1) * 100, np.nan)
    # 当日累计（RTH，逐行 cum，只用 ≤ t）
    r = pd.DataFrame(
        {
            "day": day,
            "rth": rth,
            "h": np.where(rth, h, np.nan),
            "l": np.where(rth, l_, np.nan),
            "c": c,
            "v": np.where(rth, v, 0.0),
            "tp": np.where(rth, (h + l_ + c) / 3, np.nan),
        }
    )
    gd = r.groupby("day")
    hod, lod = gd["h"].cummax().to_numpy(), gd["l"].cummin().to_numpy()
    cv = gd["v"].cumsum().to_numpy()
    tpv = (r["tp"].fillna(0) * r["v"]).groupby(r["day"]).cumsum().to_numpy()
    tp2v = (r["tp"].fillna(0) ** 2 * r["v"]).groupby(r["day"]).cumsum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = tpv / cv
        sig = np.sqrt(np.clip(tp2v / cv - vwap**2, 0, None))
        out["vwap_dist"] = np.where(rth, (c / vwap - 1) * 100, np.nan)
        out["vwap_z"] = np.where(rth, (c - vwap) / np.where(sig > 0, sig, np.nan), np.nan)
        out["hod_dist"] = np.where(rth, (hod / c - 1) * 100, np.nan)
        out["lod_dist"] = np.where(rth, (c / lod - 1) * 100, np.nan)
    ph_, pl_ = np.r_[np.nan, hod[:-1]], np.r_[np.nan, lod[:-1]]
    with np.errstate(invalid="ignore"):
        newh = rth & ~(ph_ >= hod)  # 当日首根 RTH（前一行无当日高点）或创新高
        newl = rth & ~(pl_ <= lod)
    out["since_new_high"] = np.where(rth, bars_since(newh & rth), np.nan)
    out["since_new_low"] = np.where(rth, bars_since(newl & rth), np.nan)
    sgn = np.sign(c - o) * np.where(rth, v, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["signed_vol_cum"] = np.where(
            rth, pd.Series(sgn).groupby(day).cumsum().to_numpy() / cv, np.nan
        )
    # 开盘区间（区间结束后才可用）、首 30 分钟收益（10:00 后可用）
    for N in (5, 15, 30):
        inwin = rth & (mod < 570 + N)
        orh = pd.Series(np.where(inwin, h, np.nan)).groupby(day).cummax().to_numpy()
        orl = pd.Series(np.where(inwin, l_, np.nan)).groupby(day).cummin().to_numpy()
        orh = pd.Series(orh).groupby(day).ffill().to_numpy()
        orl = pd.Series(orl).groupby(day).ffill().to_numpy()
        avail = rth & (mod >= 570 + N - 1)  # 第 N 根（09:30+N−1）收盘起
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"or{N}_hi_dist"] = np.where(avail, (c / orh - 1) * 100, np.nan)
            out[f"or{N}_lo_dist"] = np.where(avail, (c / orl - 1) * 100, np.nan)
    c30 = pd.Series(np.where(rth & (mod == 599), c, np.nan)).groupby(day).ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["first30_ret"] = np.where(rth & (mod >= 599), (c30 / dopen - 1) * 100, np.nan)
    # 相对成交量与 R-045 噪声区（按日 × 分钟的日序列 → 前 20 / 14 日均值 → shift(1)）
    piv_v = R.pivot_table(index="day", columns="mod", values="v", aggfunc="sum")
    avg_v = piv_v.rolling(20, min_periods=5).mean().shift(1)
    R2 = R.assign(mv=np.abs(R["c"] / R["day"].map(D["o"]) - 1))
    piv_m = R2.pivot_table(index="day", columns="mod", values="mv", aggfunc="last")
    sig14 = piv_m.rolling(14, min_periods=14).mean().shift(1)
    key = pd.MultiIndex.from_arrays([day, mod])
    av = avg_v.stack().reindex(key).to_numpy(float)
    s14 = sig14.stack().reindex(key).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["relvol"] = np.where(rth, v / av, np.nan)
        v5 = (
            pd.Series(np.where(rth, v, np.nan))
            .groupby(day)
            .rolling(5, min_periods=5)
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
            .to_numpy()
        )
        av5 = (
            pd.Series(av)
            .groupby(day)
            .rolling(5, min_periods=5)
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
            .to_numpy()
        )
        out["relvol5"] = np.where(rth, v5 / av5, np.nan)
        out["vol_spike"] = np.where(rth, (v / av > 3).astype(float), np.nan)
        ub = np.maximum(dopen, pc) * (1 + s14)
        lb = np.minimum(dopen, pc) * (1 - s14)
        out["noise_ub_dist"] = np.where(rth, (c / ub - 1) * 100, np.nan)
        out["noise_lb_dist"] = np.where(rth, (c / lb - 1) * 100, np.nan)
    # 已实现波动及分位（1 分钟对数收益，连续序列）
    lr = pd.Series(np.log(c)).diff()
    for w in (5, 15, 30):
        rv = lr.rolling(w, min_periods=w).std() * 1e4
        out[f"rv{w}"] = rv.to_numpy()
        out[f"rv{w}_rank"] = rv.rolling(20 * 960, min_periods=960).rank(pct=True).to_numpy()
    # 摆动高低点与支撑阻力（1 分钟左右 5 根；5 分钟左右 3 根），确认后才可用
    for tag, P, L in (("sw1", 1, 5), ("sw5", 5, 3)):
        b = e1 if P == 1 else resample_bars(e1, f"{P}min")
        b = b[b["close"].notna()]
        bh, bl = b["high"], b["low"]
        mh, ml = (
            bh.rolling(2 * L + 1, min_periods=2 * L + 1).max(),
            bl.rolling(2 * L + 1, min_periods=2 * L + 1).min(),
        )
        ph_ = bh.shift(L) == mh
        pl_ = bl.shift(L) == ml
        sh = bh.shift(L).where(ph_).ffill()
        slv = bl.shift(L).where(pl_).ffill()
        bc = b["close"]
        brk = np.where(bc > sh, 1, np.where(bc < slv, -1, 0))
        brk_up_recent = (
            pd.Series(np.r_[False, (brk[1:] == 1) & (brk[:-1] != 1)])
            .rolling(10, min_periods=1)
            .max()
            .to_numpy()
            > 0
        )
        brk_dn_recent = (
            pd.Series(np.r_[False, (brk[1:] == -1) & (brk[:-1] != -1)])
            .rolling(10, min_periods=1)
            .max()
            .to_numpy()
            > 0
        )
        sh_prev, sl_prev = sh.shift(1), slv.shift(1)
        retest = np.where(
            brk_up_recent & (bl <= sh_prev * 1.0005) & (bc > sh_prev),
            1,
            np.where(brk_dn_recent & (bh >= sl_prev * 0.9995) & (bc < sl_prev), -1, 0),
        )
        blk = pd.DataFrame(
            {
                f"{tag}_res_dist": (sh / bc - 1) * 100,
                f"{tag}_sup_dist": (bc / slv - 1) * 100,
                f"{tag}_break": brk,
                f"{tag}_retest": retest,
            },
            index=b.index,
        )
        al = _align(blk, P, idx)
        for k_ in al.columns:
            out[k_] = al[k_].to_numpy()
    # 整数关口
    for k_ in (1.0, 5.0, 10.0):
        out[f"round{int(k_)}_dist"] = (c - np.round(c / k_) * k_) / c * 100
    # 时间
    out["min_since_open"] = np.where(rth, mod - 570, np.nan).astype(float)
    out["min_to_close"] = np.where(rth, cmin - mod, np.nan).astype(float)
    out["weekday"] = np.asarray(et.weekday, float)
    out["month"] = np.asarray(et.month, float)
    # 跨资产（参考标的同样截到 t）
    if ref1 is not None and len(ref1):
        rc = ref1["close"].reindex(idx).to_numpy(float)
        for k_ in (1, 3, 5):
            own = c / pd.Series(c).shift(k_).to_numpy() - 1
            ref = rc / pd.Series(rc).shift(k_).to_numpy() - 1
            out[f"xret{k_}_diff"] = (own - ref) * 100
    return pd.DataFrame(out, index=idx)


def describe() -> list[tuple[str, str, str, str]]:
    """特征清单（名称、组、定义、可用时点），供 FEATURES.md。"""
    rows = []
    blk_desc = {
        "pctb": ("A 布林", "%b（BOLL 20,2，ddof=0）"),
        "bw": ("A 布林", "带宽 (上−下)/中 ×100"),
        "bw_rank": ("A 布林", "带宽在过去 5 日 K 中的分位"),
        "mid_slope1": ("A 布林", "中轨 1 根斜率（bp/根）"),
        "mid_slope3": ("A 布林", "中轨 3 根斜率"),
        "mid_slope5": ("A 布林", "中轨 5 根斜率"),
        "mid_slope10": ("A 布林", "中轨 10 根斜率"),
        "up_slope5": ("A 布林", "上轨 5 根斜率"),
        "lo_slope5": ("A 布林", "下轨 5 根斜率"),
        "bw_chg5": ("A 布林", "带宽 5 根变化率"),
        "dist_mid_hw": ("A 布林", "离中轨距离 ÷ 带半宽"),
        "dist_mid_atr": ("A 布林", "离中轨距离 ÷ ATR14"),
        "side_run": ("A 布林", "连续在中轨同侧根数（带符号）"),
        "since_touch_mid": ("A 布林", "距上次碰中轨根数"),
        "ride_run": ("A 布林", "沿轨运行根数（带符号）"),
        "pb_depth_up": ("A 布林", "回踩深度：(20 根高 − 收) ÷ 20 根区间"),
        "pb_depth_dn": ("A 布林", "反弹深度：(收 − 20 根低) ÷ 20 根区间"),
        "squeeze": ("A 布林", "挤压：布林在 Keltner(EMA20 ± 1.5 ATR20) 内"),
        "squeeze_run": ("A 布林", "挤压持续根数"),
        "since_release": ("A 布林", "挤压释放后根数"),
        "k": ("C KDJ/RSI", "KDJ K（9,3,3，初值 50）"),
        "d": ("C KDJ/RSI", "KDJ D"),
        "j": ("C KDJ/RSI", "KDJ J"),
        "kd_side": ("C KDJ/RSI", "K−D 符号"),
        "since_kdcross": ("C KDJ/RSI", "距 KD 交叉根数"),
        "j_ext_run": ("C KDJ/RSI", "J > 100 / < 0 持续根数（带符号）"),
        "rsi2": ("C KDJ/RSI", "RSI(2) Wilder"),
        "rsi6": ("C KDJ/RSI", "RSI(6)"),
        "rsi14": ("C KDJ/RSI", "RSI(14)"),
        "rsi14_chg1": ("C KDJ/RSI", "RSI14 1 根变化"),
        "rsi14_chg3": ("C KDJ/RSI", "RSI14 3 根变化"),
        "rsi14_div": ("C KDJ/RSI", "RSI14 背离（价创 20 根新高而 RSI 未创 → −1，镜像 +1）"),
        "obv_slope10": ("E 量", "OBV 10 根变化 ÷ 10 根量"),
        "obv_div": ("E 量", "OBV 背离"),
        "mfi14": ("E 量", "MFI(14)"),
        "updown_vol20": ("E 量", "20 根涨跌量比（对数）"),
        "atr_pct": ("F 波动趋势", "ATR14 ÷ 价"),
        "range_exp": ("F 波动趋势", "当根振幅 ÷ ATR14"),
        "er10": ("F 波动趋势", "Kaufman 效率比(10)"),
        "adx14": ("F 波动趋势", "ADX(14)"),
        "di_diff": ("F 波动趋势", "+DI − −DI"),
        "linreg_slope20": ("F 波动趋势", "20 根回归斜率（bp/根）"),
        "linreg_r2_20": ("F 波动趋势", "20 根回归 R²"),
        "aroon25": ("F 波动趋势", "Aroon(25) 振荡"),
        "supertrend": ("F 波动趋势", "SuperTrend(10,3) 方向"),
    }
    for tag, nm in (("m6", "MACD(6,13,6)"), ("m12", "MACD(12,26,9)")):
        for k_, dsc in (
            ("dif_atr", "DIF ÷ ATR"),
            ("dea_atr", "DEA ÷ ATR"),
            ("hist_atr", "柱 ÷ ATR"),
            ("hist_chg1", "柱 1 根变化"),
            ("hist_chg3", "柱 3 根变化"),
            ("zero", "零轴位置（DIF 符号）"),
            ("since_cross", "距上次金叉死叉根数"),
            ("hist_vs_prevwave", "当前 |柱| ÷ 上一同向波段最大 |柱|"),
            ("wave_vs_prevwave", "本波段至今最大 ÷ 上一同向波段最大"),
            ("div", "价创 20 根新高（低）而柱未创 → −1（+1）"),
            ("shrink_run", "柱连续缩短根数"),
        ):
            blk_desc[f"{tag}_{k_}"] = ("B MACD", f"{nm} {dsc}")
    for P in (1, 2, 5):
        for k_, (g_, dsc) in blk_desc.items():
            rows.append(
                (
                    f"p{P}_{k_}",
                    g_,
                    f"{P} 分钟已完成 K：{dsc}",
                    f"{P} 分钟 K 收盘（对齐到 ≤ 当前 1 分钟 K 收盘）",
                )
            )
    for P in (2, 5):
        for k_ in (
            "pctb",
            "dist_mid_hw",
            "bw",
            "m6_dif_atr",
            "m6_dea_atr",
            "m6_hist_atr",
            "rsi14",
            "k",
            "d",
            "j",
        ):
            rows.append(
                (
                    f"f{P}_{k_}",
                    "形成中 K",
                    f"正在形成的 {P} 分钟 K（内部已完成 1 分钟 K 合成）的 {k_}",
                    "当前 1 分钟 K 收盘",
                )
            )
    rows.append(
        (
            "mtf_consistency",
            "F 波动趋势",
            "1/2/5 分钟中轨 3 根斜率与 MACD 柱同号的个数（带符号）",
            "当前 1 分钟 K 收盘",
        )
    )
    sess = [
        ("prev_close_dist", "D 结构", "距前一日 RTH 收盘（%）", "前一日（按日 shift）"),
        ("prev_high_dist", "D 结构", "距前一日高", "前一日"),
        ("prev_low_dist", "D 结构", "距前一日低", "前一日"),
        ("gap", "D 结构", "隔夜跳空 = 当日首根开盘 ÷ 前收 − 1", "09:30 开盘"),
        *[
            (f"pivot_{n}_dist", "D 结构", f"距经典 Pivot {n}（前一日 HLC）", "前一日")
            for n in ("P", "R1", "S1", "R2", "S2")
        ],
        ("vwap_dist", "D 结构", "距 RTH VWAP（%）", "当前 1 分钟 K 收盘"),
        ("vwap_z", "D 结构", "(价 − VWAP) ÷ 成交量加权标准差（±1σ/±2σ 带）", "当前 1 分钟 K 收盘"),
        ("hod_dist", "D 结构", "距当日最高（%）", "当前 1 分钟 K 收盘"),
        ("lod_dist", "D 结构", "距当日最低", "当前 1 分钟 K 收盘"),
        ("since_new_high", "D 结构", "创新高后根数", "当前 1 分钟 K 收盘"),
        ("since_new_low", "D 结构", "创新低后根数", "当前 1 分钟 K 收盘"),
        *[
            (
                f"or{n}_{s}_dist",
                "D 结构",
                f"开盘区间 {n} 分钟{'上沿' if s == 'hi' else '下沿'}距离",
                f"09:30+{n} 分钟（区间完成）",
            )
            for n in (5, 15, 30)
            for s in ("hi", "lo")
        ],
        *[
            (
                f"{t}_{k}",
                "D 结构",
                f"摆动点（{'1 分钟左右 5 根' if t == 'sw1' else '5 分钟左右 3 根'}确认）：{d}",
                "确认 K 收盘后",
            )
            for t in ("sw1", "sw5")
            for k, d in (
                ("res_dist", "距最近阻力"),
                ("sup_dist", "距最近支撑"),
                ("break", "突破标记"),
                ("retest", "回踩标记"),
            )
        ],
        *[
            (f"round{k}_dist", "D 结构", f"距整数关口（{k} 美元倍数）", "当前 1 分钟 K 收盘")
            for k in (1, 5, 10)
        ],
        (
            "relvol",
            "E 量",
            "相对成交量 = 当分钟量 ÷ 前 20 日同一分钟均量",
            "当前 1 分钟 K 收盘（均量按日 shift）",
        ),
        ("relvol5", "E 量", "近 5 分钟量 ÷ 前 20 日同时段均量", "同上"),
        ("vol_spike", "E 量", "量突增（相对成交量 > 3）", "同上"),
        (
            "signed_vol_cum",
            "E 量",
            "当日按 K 线方向累计的带符号成交量 ÷ 当日累计量",
            "当前 1 分钟 K 收盘",
        ),
        *[
            (
                f"rv{w}",
                "F 波动趋势",
                f"{w} 分钟已实现波动（1 分钟对数收益标准差）",
                "当前 1 分钟 K 收盘",
            )
            for w in (5, 15, 30)
        ],
        *[
            (f"rv{w}_rank", "F 波动趋势", f"rv{w} 在过去 20 日中的分位", "当前 1 分钟 K 收盘")
            for w in (5, 15, 30)
        ],
        ("ret_since_open", "G 动量", "开盘以来收益", "当前 1 分钟 K 收盘"),
        ("ret_since_prev_close", "G 动量", "前收以来收益", "当前 1 分钟 K 收盘"),
        ("first30_ret", "G 动量", "首 30 分钟收益", "10:00（09:59 K 收盘）"),
        ("noise_ub_dist", "G 动量", "距 R-045 噪声区上沿", "当前 1 分钟 K 收盘（σ 按日 shift）"),
        ("noise_lb_dist", "G 动量", "距 R-045 噪声区下沿", "同上"),
        ("min_since_open", "H 时间", "距开盘分钟数", "—"),
        ("min_to_close", "H 时间", "距收盘分钟数", "—"),
        ("weekday", "H 时间", "星期", "—"),
        ("month", "H 时间", "月份", "—"),
        *[
            (
                f"xret{k}_diff",
                "I 跨资产",
                f"过去 {k} 分钟收益差（本标的 − 参考：QQQ↔SPY，MU−SMH）",
                "当前 1 分钟 K 收盘",
            )
            for k in (1, 3, 5)
        ],
    ]
    return rows + sess
