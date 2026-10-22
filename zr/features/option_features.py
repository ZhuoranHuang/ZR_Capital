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
- opt_cp_lr_near / opt_cp_lr_all：log((call 量+1)/(put 量+1))，近端 ATM±2 / 全部已拉取合约
- opt_atm_iv：近端 ATM call/put 中间价 BS 反推 IV 均值；opt_atm_iv_chg5 / chg15：日内 5、15 分钟变化
- opt_oi_chg（D-019 同期限对比）：今日 DTE=d 的合约 OI 合计 ÷ 昨日 DTE=d 的合约 OI 合计 − 1，
  d 为两日都出现的日历日 DTE，只计两日已拉取行权价区间的交集；无共同 DTE 时 NaN
  （周度到期只有周五的标的，相邻交易日 DTE 集合不重叠，此列基本为 NaN）；
  opt_oi_n_dte：参与对比的 DTE 个数；opt_pc_oi：put/call OI 比
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
    "opt_oi_chg",
    "opt_oi_n_dte",
    "opt_pc_oi",
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


def day_features(stock: pd.DataFrame, opt: dict[str, pd.DataFrame], day: dt.date) -> pd.DataFrame:
    """stock：当日正股 1 分钟 K（索引 ts UTC，含 close）。返回以 stock.index 为索引的期权特征。"""
    out = pd.DataFrame(index=stock.index)
    spot = stock["close"]
    quotes, ohlc = opt["quotes"], opt["ohlc"]
    expiries = list(ohlc["expiry"].unique()) + list(quotes["expiry"].unique())
    front = _front_expiry(expiries, day)
    out["opt_front_dte"] = float((front - day).days) if front else np.nan

    # --- 成交量：期权 OHLC bar t 与正股 bar t 同标签 ---
    if not ohlc.empty and front is not None:
        v = ohlc[["ts", "strike", "right", "expiry", "volume"]].copy()
        v["volume"] = v["volume"].fillna(0).astype(float)
        call_all = v[v["right"] == "call"].groupby("ts")["volume"].sum()
        put_all = v[v["right"] == "put"].groupby("ts")["volume"].sum()
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
    return out


def oi_daily(root: Path, und: str, days: list[dt.date]) -> pd.DataFrame:
    """每日 OI 特征（D-019）：同 DTE（日历日）桶的 OI 合计日变化率、put/call OI 比。"""
    rows, prev = [], None
    for day in days:
        d = root / "options_1m" / f"underlying={und}" / f"date={day.isoformat()}" / "oi.parquet"
        if not d.exists():
            prev = None
            continue
        oi = pd.read_parquet(d)
        if oi.empty:
            prev = None
            continue
        dte = (pd.to_datetime(oi["expiry"]) - pd.Timestamp(day)).dt.days
        cur = pd.DataFrame(
            {"dte": dte.to_numpy(), "strike": oi["strike"], "oi": oi["open_interest"]}
        )
        cur = cur[cur["dte"] >= 0]
        calls = oi.loc[oi["right"] == "call", "open_interest"].sum()
        puts = oi.loc[oi["right"] == "put", "open_interest"].sum()
        chg, n_dte = np.nan, 0
        if prev is not None and len(cur) and len(prev):
            lo = max(cur["strike"].min(), prev["strike"].min())
            hi = min(cur["strike"].max(), prev["strike"].max())
            a = cur[cur["strike"].between(lo, hi)].groupby("dte")["oi"].sum()
            b = prev[prev["strike"].between(lo, hi)].groupby("dte")["oi"].sum()
            common = a.index.intersection(b.index)
            n_dte = len(common)
            if n_dte and b[common].sum() > 0:
                chg = a[common].sum() / b[common].sum() - 1
        rows.append(
            {
                "date": day,
                "opt_oi_chg": chg,
                "opt_oi_n_dte": n_dte,
                "opt_pc_oi": puts / calls if calls > 0 else np.nan,
            }
        )
        prev = cur
    return (
        pd.DataFrame(rows).set_index("date")
        if rows
        else pd.DataFrame(columns=["opt_oi_chg", "opt_oi_n_dte", "opt_pc_oi"])
    )


def compute_option_features(root: Path, und: str, stock: pd.DataFrame) -> pd.DataFrame:
    """对 stock（多日，索引 ts UTC）中有期权数据的交易日计算特征；无数据的日期为 NaN。"""
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
        f = day_features(stock.loc[idx], opt, day)
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
        daily = oi_daily(root, und, loaded)
        for c in ("opt_oi_chg", "opt_oi_n_dte", "opt_pc_oi"):
            out[c] = sday.map(daily[c]).to_numpy(dtype=float) if len(daily) else np.nan
    return out
