"""期权维度特征（PLAN_W3 线 a），每分钟计算并对齐到正股 1 分钟 K。

输入：ZR_DATA_ROOT/options_1m/underlying=XXX/date=YYYY-MM-DD/{quotes,ohlc,oi}.parquet
      （zr/data/thetadata.py）
      + 正股 1 分钟 K（`zr.features.bars.load_bars`，索引 ts = bar 开盘时刻，UTC）。

对齐（无前视）：正股 bar t 覆盖 [t, t+1)，在 t+1 收盘时可用的期权信息是
- 期权 OHLC bar t（同样覆盖 [t, t+1)）；
- Theta quote 时间戳 t+1 的 NBBO（Theta 1m 时间戳 = 该时刻的 NBBO，见 CLAUDE.md），
  即 quote.ts − 1 分钟对齐到 bar t；
- OI：OPRA 约 06:30 ET 发布前一交易日收盘 OI，开盘前已知，全天常数。
ATM 用正股 bar t 的收盘价；「近端」= 当日仍未到期的最近到期日（0DTE 优先）。

特征（列名均以 opt_ 开头）：
- opt_near_vol：近端到期 ATM±2 档（5 个行权价）call+put 1 分钟成交量
- opt_near_vol_ratio20：opt_near_vol ÷ 前 20 个交易日同一分钟均值（至少 5 天）
- opt_cp_lr_near / opt_cp_lr_all：log((call 量+1)/(put 量+1))，
  近端 ATM±2 / 开盘价 ±b 内全部已拉取合约
- opt_atm_iv：近端 ATM call/put 中间价 BS 反推 IV 均值；opt_atm_iv_chg5 / chg15：日内 5、15 分钟变化
- OI 水平（D-021，取代 D-019 的日变化）：oi.parquet 为当日 06:30 ET 发布的
  前一交易日收盘 OI，开盘前已知。
  合约集合 = 当日已拉取到期日中行权价在 开盘价 × (1 ± b) 内的合约
  （b = 拉取带宽，SPY/QQQ 3%、MU 8%）；
  开盘价 09:30 已知，且拉取带宽 [日低×(1−b), 日高×(1+b)] 必然包含该区间，
  所以集合不依赖当日后续走势。
  opt_cp_oi：call OI ÷ put OI；opt_maxoi_dist：(现价 − 最大 OI 行权价) ÷ 行权价间距
  （按行权价汇总 call+put；间距 = 离开盘价最近 11 个行权价的相邻差中位数，从数据取）；
  opt_atm3_oi_share：离现价最近行权价 ±3 档（共 7 个）的 OI ÷ 集合内全部 OI。
  未用全链：拉取时 OI 已按带宽过滤，全链 OI 需另行拉取
- opt_front_dte：近端到期日距今天数
未包含：净 gamma 近似（GEX）——等 OI 全量到位后另加（PLAN_W3 线 a）。
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
R = 0.04  # 无风险利率假设（对 0–5DTE 影响可忽略）
N_EACH_SIDE = 2
OI_ATM_EACH_SIDE = 3
SPACING_N = 11
VOL_LOOKBACK_DAYS = 20
VOL_MIN_DAYS = 5

OPTION_FEATURE_COLUMNS = [
    "opt_near_vol",
    "opt_near_vol_ratio20",
    "opt_cp_lr_near",
    "opt_cp_lr_all",
    "opt_atm_iv",
    "opt_atm_iv_chg5",
    "opt_atm_iv_chg15",
    "opt_cp_oi",
    "opt_maxoi_dist",
    "opt_atm3_oi_share",
    "opt_front_dte",
]


# ---------- Black-Scholes ----------


def bs_price(s: float, k: float, t: float, sigma: float, right: str) -> float:
    if t <= 0 or sigma <= 0:
        return max(0.0, (s - k) if right == "call" else (k - s))
    d1 = (math.log(s / k) + (R + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if right == "call":
        return s * norm.cdf(d1) - k * math.exp(-R * t) * norm.cdf(d2)
    return k * math.exp(-R * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def implied_vol(price: float, s: float, k: float, t: float, right: str) -> float:
    if not (np.isfinite(price) and price > 0 and t > 0):
        return float("nan")
    intrinsic = max(0.0, (s - k) if right == "call" else (k - s))
    if price <= intrinsic + 1e-6:
        return float("nan")
    try:
        return brentq(lambda v: bs_price(s, k, t, v, right) - price, 1e-4, 5.0, maxiter=100)
    except ValueError:
        return float("nan")


def years_to_expiry(ts_utc: pd.Series, expiry: pd.Series) -> np.ndarray:
    close = pd.to_datetime(expiry.astype(str) + " 16:00").dt.tz_localize(ET).dt.tz_convert("UTC")
    return ((close - ts_utc).dt.total_seconds() / (365 * 24 * 3600)).to_numpy()


# ---------- 读取 ----------


def load_option_day(root: Path, und: str, day: dt.date) -> dict[str, pd.DataFrame] | None:
    d = root / "options_1m" / f"underlying={und}" / f"date={day.isoformat()}"
    if not all((d / f"{n}.parquet").exists() for n in ("quotes", "ohlc", "oi")):
        return None
    out = {n: pd.read_parquet(d / f"{n}.parquet") for n in ("quotes", "ohlc", "oi")}
    if out["quotes"].empty and out["ohlc"].empty:
        return None
    return out


def option_days(root: Path, und: str) -> list[dt.date]:
    base = root / "options_1m" / f"underlying={und}"
    if not base.is_dir():
        return []
    return sorted(dt.date.fromisoformat(p.name[5:]) for p in base.glob("date=*"))


# ---------- 单日计算 ----------


def _front_expiry(expiries, day: dt.date) -> dt.date | None:
    fut = sorted(e for e in set(expiries) if e >= day)
    return fut[0] if fut else None


def _near_strikes(strikes: np.ndarray, spot: np.ndarray, n_each: int) -> np.ndarray:
    """每行（分钟）选离 spot 最近的 2n+1 个行权价，返回布尔矩阵 (n_minutes, n_strikes)。"""
    dist = np.abs(strikes[None, :] - spot[:, None])
    k = min(2 * n_each + 1, len(strikes))
    idx = np.argpartition(dist, k - 1, axis=1)[:, :k]
    mask = np.zeros_like(dist, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


def open_window(strikes: pd.Series, open_px: float, band: float) -> pd.Series:
    """行权价是否在 开盘价 × (1 ± band) 内（09:30 已知，不依赖当日后续走势）。"""
    return strikes.between(open_px * (1 - band), open_px * (1 + band))


def strike_spacing(strikes: np.ndarray, ref: float, n: int = SPACING_N) -> float:
    """离 ref 最近 n 个行权价的相邻差中位数（行权价间距从数据取，CLAUDE.md 规则 7）。"""
    ks = np.unique(strikes)
    if len(ks) < 2:
        return float("nan")
    near = np.sort(ks[np.argsort(np.abs(ks - ref))[:n]])
    d = np.diff(near)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else float("nan")


def oi_level_features(
    oi: pd.DataFrame, spot: pd.Series, open_px: float, band: float, day: dt.date
) -> pd.DataFrame:
    """D-021 OI 水平特征（每分钟；OI 全天常数，随现价变化的是距离与 ATM 占比）。"""
    out = pd.DataFrame(
        index=spot.index, columns=["opt_cp_oi", "opt_maxoi_dist", "opt_atm3_oi_share"], dtype=float
    )
    if oi is None or oi.empty or not np.isfinite(open_px):
        return out
    exp = pd.to_datetime(oi["expiry"]).dt.date
    x = oi[(exp >= day).to_numpy() & open_window(oi["strike"], open_px, band).to_numpy()]
    if x.empty:
        return out
    calls = float(x.loc[x["right"] == "call", "open_interest"].sum())
    puts = float(x.loc[x["right"] == "put", "open_interest"].sum())
    out["opt_cp_oi"] = calls / puts if puts > 0 else np.nan
    per_k = x.groupby("strike")["open_interest"].sum().sort_index()
    total = float(per_k.sum())
    if total <= 0:
        return out
    ks, v = per_k.index.to_numpy(float), per_k.to_numpy(float)
    spacing = strike_spacing(ks, open_px)
    k_max = ks[int(np.argmax(v))]
    s = spot.to_numpy(float)
    out["opt_maxoi_dist"] = (s - k_max) / spacing if spacing > 0 else np.nan
    i = np.abs(ks[None, :] - s[:, None]).argmin(axis=1)
    csum = np.r_[0.0, np.cumsum(v)]
    lo = np.maximum(i - OI_ATM_EACH_SIDE, 0)
    hi = np.minimum(i + OI_ATM_EACH_SIDE + 1, len(ks))
    out["opt_atm3_oi_share"] = (csum[hi] - csum[lo]) / total
    return out


def day_features(
    stock: pd.DataFrame, opt: dict[str, pd.DataFrame], day: dt.date, band: float = 0.05
) -> pd.DataFrame:
    """stock：当日正股 1 分钟 K（索引 ts UTC，含 open/close）。返回以 stock.index 为索引的期权特征。

    band：该标的的拉取带宽（zr/data/thetadata.py 的 STRIKE_BAND），决定「开盘价 ±b」合约集合。
    """
    out = pd.DataFrame(index=stock.index)
    spot = stock["close"]
    open_px = float(stock["open"].iloc[0]) if "open" in stock and len(stock) else float("nan")
    quotes, ohlc = opt["quotes"], opt["ohlc"]
    expiries = list(ohlc["expiry"].unique()) + list(quotes["expiry"].unique())
    front = _front_expiry(expiries, day)
    out["opt_front_dte"] = float((front - day).days) if front else np.nan

    # --- 成交量：期权 OHLC bar t 与正股 bar t 同标签 ---
    if not ohlc.empty and front is not None:
        v = ohlc[["ts", "strike", "right", "expiry", "volume"]].copy()
        v["volume"] = v["volume"].fillna(0).astype(float)
        va = v[open_window(v["strike"], open_px, band)] if np.isfinite(open_px) else v
        call_all = va[va["right"] == "call"].groupby("ts")["volume"].sum()
        put_all = va[va["right"] == "put"].groupby("ts")["volume"].sum()
        out["opt_cp_lr_all"] = np.log(
            (call_all.reindex(out.index).fillna(0) + 1) / (put_all.reindex(out.index).fillna(0) + 1)
        )
        f = v[v["expiry"] == front]
        strikes = np.sort(f["strike"].unique())
        if len(strikes):
            mask = _near_strikes(strikes, spot.to_numpy(), N_EACH_SIDE)
            piv = f.pivot_table(
                index="ts", columns=["right", "strike"], values="volume", aggfunc="sum"
            )
            piv = piv.reindex(out.index).fillna(0.0)
            calls = (
                piv["call"].reindex(columns=strikes, fill_value=0.0).to_numpy()
                if "call" in piv
                else 0
            )
            puts = (
                piv["put"].reindex(columns=strikes, fill_value=0.0).to_numpy()
                if "put" in piv
                else 0
            )
            c_near = (
                (np.asarray(calls) * mask).sum(axis=1) if np.ndim(calls) else np.zeros(len(out))
            )
            p_near = (np.asarray(puts) * mask).sum(axis=1) if np.ndim(puts) else np.zeros(len(out))
            out["opt_near_vol"] = c_near + p_near
            out["opt_cp_lr_near"] = np.log((c_near + 1) / (p_near + 1))

    # --- ATM IV：quote.ts − 1 分钟 对齐到 bar t ---
    if not quotes.empty and front is not None:
        q = quotes[quotes["expiry"] == front][["ts", "strike", "right", "bid", "ask"]].copy()
        q["bar"] = q["ts"] - pd.Timedelta(minutes=1)
        q = q[(q["bid"] > 0) & (q["ask"] > 0)]
        q["mid"] = (q["bid"] + q["ask"]) / 2
        strikes = np.sort(q["strike"].unique())
        if len(strikes):
            atm_k = strikes[np.abs(strikes[None, :] - spot.to_numpy()[:, None]).argmin(axis=1)]
            atm = pd.DataFrame({"bar": out.index, "strike": atm_k, "spot": spot.to_numpy()})
            m = atm.merge(q, on=["bar", "strike"], how="left")
            m["T"] = years_to_expiry(
                m["bar"] + pd.Timedelta(minutes=1), pd.Series([front] * len(m))
            )
            m["iv"] = [
                implied_vol(p, s, k, t, r) if isinstance(r, str) else np.nan
                for p, s, k, t, r in zip(
                    m["mid"], m["spot"], m["strike"], m["T"], m["right"], strict=True
                )
            ]
            iv = m.groupby("bar")["iv"].mean().reindex(out.index)
            out["opt_atm_iv"] = iv
            out["opt_atm_iv_chg5"] = iv - iv.shift(5)
            out["opt_atm_iv_chg15"] = iv - iv.shift(15)
    lv = oi_level_features(opt.get("oi"), spot, open_px, band, day)
    out[lv.columns] = lv
    return out


def compute_option_features(root: Path, und: str, stock: pd.DataFrame) -> pd.DataFrame:
    """对 stock（多日，索引 ts UTC）中有期权数据的交易日计算特征；无数据的日期为 NaN。"""
    from zr.data.thetadata import STRIKE_BAND

    band = STRIKE_BAND.get(und, 0.05)
    out = pd.DataFrame(index=stock.index, columns=OPTION_FEATURE_COLUMNS, dtype=float)
    sday = pd.Series(stock.index.tz_convert(ET).date, index=stock.index)
    days = [d for d in option_days(root, und) if d in set(sday.unique())]
    loaded: list[dt.date] = []
    for day in days:
        opt = load_option_day(root, und, day)
        if opt is None:
            continue
        loaded.append(day)
        idx = sday.index[sday == day]
        f = day_features(stock.loc[idx], opt, day, band)
        out.loc[idx, f.columns] = f.to_numpy(dtype=float)
    # 同时段 20 日均量（只用之前的交易日）
    if loaded:
        et = stock.index.tz_convert(ET)
        key = pd.MultiIndex.from_arrays([sday.to_numpy(), et.hour * 60 + et.minute])
        piv = (
            pd.Series(out["opt_near_vol"].to_numpy(dtype=float), index=key)
            .unstack(level=1)
            .sort_index()
        )
        base = piv.shift(1).rolling(VOL_LOOKBACK_DAYS, min_periods=VOL_MIN_DAYS).mean()
        out["opt_near_vol_ratio20"] = (
            out["opt_near_vol"].to_numpy(dtype=float)
            / base.stack(future_stack=True).reindex(key).to_numpy()
        )
    return out
