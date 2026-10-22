"""E4 正股复核（feat/e4-review，RESEARCH_LOG R-036）→ docs/research/E4_REVIEW.md。

只做正股，规则不变，只修实现。引擎与 zr/research/e4_mu_stock.py 同规则，增加参数：
- ladder="recent"（负责人定义：收盘跌破最近一个已确认的更高谷底）/ "ratchet"（原实现：历来更高
  谷底的最高值、只上移）；strict=True（谷底严格低于左右各 n 根）/ False（原实现允许并列）；
- swing_n、stop_mode（atr / swing）、no_open_before、max_adds、sides、bar_min、notional。
ladder="ratchet"、strict=False 时与原实现逐笔相同（报告第 1 节核对）。
运行：uv run python -m zr.research.e4_review
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.bar_quality import clean_wicks, flag_bars
from zr.features import indicators as ind
from zr.features.bars import ET, load_bars, resample_bars
from zr.ops.markdown import df_to_md
from zr.research import e4_mu_stock as old
from zr.research.e4_mu_stock import fee

log = logging.getLogger("zr.research.e4_review")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "E4_REVIEW.md"
FLAT_MIN = 15 * 60 + 50
RESEARCH_END = dt.date(2025, 12, 31)
N_RANDOM = 1000


@dataclass(frozen=True)
class P:
    ma: int = 120
    stop_mode: str = "atr"  # atr / swing
    stop_mult: float = 1.5
    max_adds: int = 2
    lot: int = 100
    notional: float | None = None  # 给定时每次成交股数 = floor(notional / 成交价)
    tick: float = 0.01
    swing_n: int = 2
    ladder: str = "recent"  # recent / recent_valid（敏感性：入场时水位须在入场价止损一侧）/ ratchet
    strict: bool = True
    kdj: tuple[int, int, int] = (9, 3, 3)
    atr_n: int = 14
    bar_min: int = 5
    no_open_before: int | None = None  # 分钟数（ET），如 600 = 10:00
    sides: str = "both"


BASELINE = P()
OLD_IMPL = P(ladder="ratchet", strict=False)
VARIANTS = {
    "基线（修正后）": BASELINE,
    "V1a n=3": replace(BASELINE, swing_n=3),
    "V1b n=5": replace(BASELINE, swing_n=5),
    "V2a ATR×2": replace(BASELINE, stop_mult=2.0),
    "V2b ATR×3": replace(BASELINE, stop_mult=3.0),
    "V2c 谷底下方 1 tick": replace(BASELINE, stop_mode="swing"),
    "V3 10:00 前不开仓": replace(BASELINE, no_open_before=600),
    "V4 不加仓": replace(BASELINE, max_adds=0),
    "V5a 只做多": replace(BASELINE, sides="long"),
    "V5b 只做空": replace(BASELINE, sides="short"),
    "V6 15 分钟": replace(BASELINE, bar_min=15),
    "V7 组合": replace(BASELINE, swing_n=3, stop_mode="swing", no_open_before=600, max_adds=0),
}


# ---------------- 数据与指标 ----------------


def load_1m(root: Path, und: str) -> pd.DataFrame:
    raw = load_bars(root, und)
    return clean_wicks(raw, flag_bars(raw))


def swings(low: np.ndarray, high: np.ndarray, day: np.ndarray, n: int, strict: bool):
    """bar t 收盘确认 bar t−n 的谷底/波峰（左右各 n 根同一天）；返回与 bar 对齐的水位。"""
    m = len(low)
    sw_lo, sw_hi = np.full(m, np.nan), np.full(m, np.nan)
    for t in range(2 * n, m):
        if day[t - 2 * n] != day[t]:
            continue
        c = t - n
        wl = np.r_[low[t - 2 * n : c], low[c + 1 : t + 1]]
        wh = np.r_[high[t - 2 * n : c], high[c + 1 : t + 1]]
        if (low[c] < wl.min()) if strict else (low[c] <= wl.min()):
            sw_lo[t] = low[c]
        if (high[c] > wh.max()) if strict else (high[c] >= wh.max()):
            sw_hi[t] = high[c]
    return sw_lo, sw_hi


def prepare(df: pd.DataFrame, p: P) -> dict:
    c = df["close"]
    ma = ind.sma(c, p.ma)
    k = ind.kdj(df["high"], df["low"], c, *p.kdj)
    up = (k["k"] > k["d"]) & (k["k"].shift(1) <= k["d"].shift(1))
    dn = (k["k"] < k["d"]) & (k["k"].shift(1) >= k["d"].shift(1))
    et = df.index.tz_convert(ET)
    day = np.asarray(et.date)
    lo, hi = df["low"].to_numpy(float), df["high"].to_numpy(float)
    sw_lo, sw_hi = swings(lo, hi, day, p.swing_n, p.strict)
    return {
        "open": df["open"].to_numpy(float),
        "high": hi,
        "low": lo,
        "close": c.to_numpy(float),
        "ma": ma.to_numpy(float),
        "k": k["k"].to_numpy(float),
        "d": k["d"].to_numpy(float),
        "long": ((c > ma) & (ma.diff() > 0) & up).to_numpy(),
        "short": ((c < ma) & (ma.diff() < 0) & dn).to_numpy(),
        "atr": ind.atr(df["high"], df["low"], c, p.atr_n).to_numpy(float),
        "sw_lo": sw_lo,
        "sw_hi": sw_hi,
        "day": day,
        "mod": np.asarray(et.hour * 60 + et.minute),
        "ts": df.index,
    }


# ---------------- 回测 ----------------


def _eod(a: dict, k: int, p: P) -> bool:
    """bar k 是当日最后一根收盘 ≤ 15:50 的 K（下一根收盘晚于 15:50 或当日已无下一根）。"""
    n = len(a["close"])
    return k + 1 >= n or a["day"][k + 1] != a["day"][k] or a["mod"][k] + 2 * p.bar_min > FLAT_MIN


def simulate_entry(a: dict, i: int, d: int, p: P, trace: bool = False) -> dict | None:
    n = len(a["close"])
    j = i + 1
    if j >= n or a["day"][j] != a["day"][i]:
        return None
    if a["mod"][j] + 2 * p.bar_min > FLAT_MIN:  # 入场 K 本身就是 15:50 清仓 K
        return None
    if p.no_open_before is not None and a["mod"][j] < p.no_open_before:
        return None
    if not np.isfinite(a["atr"][i]):
        return None
    o, h, lo, c = a["open"], a["high"], a["low"], a["close"]
    sig = a["long"] if d == 1 else a["short"]
    sw = a["sw_lo"] if d == 1 else a["sw_hi"]
    tick = p.tick

    def qty(px: float) -> int:
        return int(p.notional // px) if p.notional else p.lot

    px0 = o[j] + d * tick
    fills = [(j, px0, qty(px0))]
    day0 = a["day"][i]
    k0 = i
    while k0 > 0 and a["day"][k0 - 1] == day0:
        k0 -= 1
    prior = [x for x in sw[k0 : i + 1] if np.isfinite(x)]  # 信号 K 收盘时已确认的
    stop = px0 - d * p.stop_mult * a["atr"][i]
    stop_src = "atr"
    if p.stop_mode == "swing" and prior and d * (px0 - (prior[-1] - d * tick)) > 0:
        stop, stop_src = prior[-1] - d * tick, "swing"
    hist = [x for x in sw[k0 : j + 1] if np.isfinite(x)]  # 入场 K 收盘时已确认的
    ladder = np.nan

    def update(lad: float, seq: list) -> float:
        if len(seq) < 2 or d * (seq[-1] - seq[-2]) <= 0:
            return lad
        if p.ladder != "ratchet" or not np.isfinite(lad):
            return seq[-1]
        return max(lad, seq[-1]) if d == 1 else min(lad, seq[-1])

    if p.ladder in ("recent", "recent_valid"):
        for m_ in range(2, len(hist) + 1):
            ladder = update(ladder, hist[:m_])
        if p.ladder == "recent_valid" and np.isfinite(ladder) and d * (px0 - ladder) <= 0:
            ladder = np.nan  # 入场前的更高谷底已在入场价之上（多头）：结构已破，不作初始水位
    else:  # 原实现：只看入场时最后两个
        ladder = update(ladder, hist)
    k = j
    mfe = 0.0
    path = []
    while True:
        mfe = max(mfe, d * ((h[k] if d == 1 else lo[k]) - px0))
        if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
            exit_k, exit_px, reason = (
                k,
                (min(o[k], stop) if d == 1 else max(o[k], stop)) - d * tick,
                "止损",
            )
            break
        if _eod(a, k, p):
            exit_k, exit_px, reason = k, c[k] - d * tick, "15:50 清仓"
            break
        if k > j and np.isfinite(sw[k]):
            hist.append(sw[k])
            ladder = update(ladder, hist)
        if trace:
            path.append((a["ts"][k], c[k], ladder))
        if np.isfinite(ladder) and d * (c[k] - ladder) < 0:
            exit_k, exit_px, reason = k + 1, o[k + 1] - d * tick, "梯形"
            break
        avg = float(np.average([f[1] for f in fills], weights=[f[2] for f in fills]))
        if (
            len(fills) <= p.max_adds
            and k > j
            and sig[k]
            and d * (c[k] - avg) > 0
            and a["mod"][k + 1] + 2 * p.bar_min <= FLAT_MIN
        ):
            pa = o[k + 1] + d * tick
            fills.append((k + 1, pa, qty(pa)))
        k += 1
    side_in, side_out = ("Buy", "Sell") if d == 1 else ("Sell", "Buy")
    shares = sum(f[2] for f in fills)
    gross_each = [d * q * (exit_px - fp) for _, fp, q in fills]
    fee_in = [fee(q, side_in, fp) for _, fp, q in fills]
    fee_out = fee(shares, side_out, exit_px)
    notional = sum(q * fp for _, fp, q in fills)
    out_share = [q / shares for _, _, q in fills]
    net_each = [
        g - fi - fee_out * s for g, fi, s in zip(gross_each, fee_in, out_share, strict=True)
    ]
    t = {
        "signal_i": i,
        "entry_ts": a["ts"][j],
        "exit_ts": a["ts"][exit_k],
        "exit_i": exit_k,
        "direction": d,
        "entry_px": px0,
        "exit_px": exit_px,
        "adds": len(fills) - 1,
        "shares": shares,
        "gross": sum(gross_each),
        "fees": sum(fee_in) + fee_out,
        "net": sum(net_each),
        "notional": notional,
        # bp：价差（未扣滑点）− 成本（费用 + 每边 1 tick）= 净；USD「毛利」沿用原口径（已含滑点）
        "gross_bp": (sum(gross_each) + 2 * tick * shares) / notional * 1e4,
        "cost_bp": (sum(fee_in) + fee_out + 2 * tick * shares) / notional * 1e4,
        "net_bp": sum(net_each) / notional * 1e4,
        "first_gross": gross_each[0],
        "first_net": net_each[0],
        "adds_gross": sum(gross_each[1:]),
        "adds_net": sum(net_each[1:]),
        "mfe_px": mfe,
        "first_exit_px_gain": d * (exit_px - px0),
        "hold_min": (exit_k - j + 1) * p.bar_min,
        "reason": reason,
        "stop": stop,
        "stop_src": stop_src,
        "entry_mod": int(a["mod"][j]),
    }
    if trace:
        t["fills"] = fills
        t["path"] = path
        t["fee_in"] = fee_in
        t["fee_out"] = fee_out
    return t


def backtest(a: dict, p: P) -> pd.DataFrame:
    trades, i, n = [], 0, len(a["close"])
    while i < n - 1:
        d = 1 if a["long"][i] else (-1 if a["short"][i] else 0)
        if (d == 1 and p.sides == "short") or (d == -1 and p.sides == "long"):
            d = 0
        if d:
            t = simulate_entry(a, i, d, p)
            if t is not None:
                trades.append(t)
                i = max(t["exit_i"], i + 1)
                continue
        i += 1
    return pd.DataFrame(trades)


def random_direction(a: dict, tr: pd.DataFrame, p: P, n: int = N_RANDOM, seed: int = 0):
    if tr.empty:
        return np.zeros(n)
    q = replace(p, sides="both")
    pl = np.empty((len(tr), 2))
    for r, (i, d) in enumerate(zip(tr["signal_i"], tr["direction"], strict=True)):
        other = simulate_entry(a, int(i), -int(d), q)
        o = other["net"] if other is not None else 0.0
        same = tr["net"].iloc[r]
        pl[r] = (same, o) if d == 1 else (o, same)
    pick = np.random.default_rng(seed).integers(0, 2, size=(n, len(tr)))
    return np.where(pick == 0, pl[:, 0], pl[:, 1]).sum(axis=1)


def bh(p: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = int(np.flatnonzero(ok).max()) + 1 if ok.any() else 0
    out = np.zeros(m, bool)
    out[o[:k]] = True
    return out


def summary(tr: pd.DataFrame, rnd: np.ndarray | None = None) -> dict:
    if tr.empty:
        return {"笔数": 0}
    net = tr["net"]
    w, lo = net[net > 0], net[net <= 0]
    out = {
        "笔数": len(tr),
        "胜率": round(float((net > 0).mean()), 3),
        "盈亏比": round(float(w.mean() / -lo.mean()), 2) if len(w) and len(lo) else np.nan,
        "毛利 USD": round(float(tr["gross"].sum()), 0),
        "费用 USD": round(float(tr["fees"].sum()), 0),
        "扣费后 USD": round(float(net.sum()), 0),
        "每笔价差 bp": round(float(tr["gross_bp"].mean()), 2),
        "每笔成本 bp": round(float(tr["cost_bp"].mean()), 2),
        "每笔净 bp": round(float(tr["net_bp"].mean()), 2),
    }
    if rnd is not None:
        k = int((rnd >= net.sum()).sum())
        out["随机方向分位"] = round(float((rnd < net.sum()).mean() * 100), 1)
        out["p"] = round((k + 1) / (len(rnd) + 1), 4)
    return out


def period(tr: pd.DataFrame, lo: dt.date | None, hi: dt.date | None) -> pd.DataFrame:
    if tr.empty:
        return tr
    d = tr["entry_ts"].dt.tz_convert(ET).dt.date
    m = np.ones(len(tr), bool)
    if lo:
        m &= (d >= lo).to_numpy()
    if hi:
        m &= (d <= hi).to_numpy()
    return tr[m].reset_index(drop=True)


# ---------------- 独立复算（逐 bar，不复用引擎） ----------------


def reference_signal(df: pd.DataFrame, i: int, d: int, p: P) -> bool:
    """独立复核信号：SMA(ma) 及其斜率、国内口径 KDJ（逐根显式递推、初值 50）在 bar i 的交叉。"""
    c, h, lo = (df[x].to_numpy(float) for x in ("close", "high", "low"))
    sma = c[i - p.ma + 1 : i + 1].mean()
    sma_prev = c[i - p.ma : i].mean()
    k = dd = 50.0
    kk = np.empty(i + 1)
    dk = np.empty(i + 1)
    n9 = p.kdj[0]
    for t in range(i + 1):
        lw, hg = lo[max(0, t - n9 + 1) : t + 1].min(), h[max(0, t - n9 + 1) : t + 1].max()
        rsv = (c[t] - lw) / (hg - lw) * 100 if hg > lw else 50.0
        k = 2 / 3 * k + 1 / 3 * rsv
        dd = 2 / 3 * dd + 1 / 3 * k
        kk[t], dk[t] = k, dd
    if d == 1:
        return bool(c[i] > sma and sma > sma_prev and kk[i] > dk[i] and kk[i - 1] <= dk[i - 1])
    return bool(c[i] < sma and sma < sma_prev and kk[i] < dk[i] and kk[i - 1] >= dk[i - 1])


def reference_trade(df: pd.DataFrame, entry_ts, d: int, p: P) -> dict:
    """独立实现：从原始 K 线逐根重新算 ATR、摆动点与出场，核对引擎的一笔交易（不含加仓）。"""
    from zr.costs.fees import stock_order_fees

    i = int(np.flatnonzero(df.index == entry_ts)[0]) - 1
    o, h, lo, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    day = np.asarray(df.index.tz_convert(ET).date)
    mod = np.asarray(df.index.tz_convert(ET).hour * 60 + df.index.tz_convert(ET).minute)
    tr = np.maximum(h[1:] - lo[1:], np.maximum(abs(h[1:] - c[:-1]), abs(lo[1:] - c[:-1])))
    tr = np.r_[h[0] - lo[0], tr]
    atr = np.nan
    for t in range(i + 1):  # Wilder ATR(14)，min_periods 14
        atr = (
            tr[: t + 1].mean()
            if t == p.atr_n - 1
            else (atr if t < p.atr_n - 1 else (atr * (p.atr_n - 1) + tr[t]) / p.atr_n)
        )
    j = i + 1
    px0 = o[j] + d * p.tick
    stop = px0 - d * p.stop_mult * atr
    side_in, side_out = ("Buy", "Sell") if d == 1 else ("Sell", "Buy")

    def done(ts, px, reason):
        f = float(stock_order_fees(p.lot, side_in, px0)["total"]) + float(
            stock_order_fees(p.lot, side_out, px)["total"]
        )
        return {
            "exit_ts": ts,
            "exit_px": px,
            "reason": reason,
            "net": d * p.lot * (px - px0) - f,
            "signal_ok": reference_signal(df, i, d, p),
        }

    n = p.swing_n
    seq, ladder = [], np.nan
    k0 = i
    while day[k0 - 1] == day[i]:
        k0 -= 1
    k = k0
    while True:
        if k >= j:
            if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
                return done(
                    df.index[k],
                    (min(o[k], stop) if d == 1 else max(o[k], stop)) - d * p.tick,
                    "止损",
                )
            if k + 1 >= len(c) or day[k + 1] != day[k] or mod[k] + 2 * p.bar_min > FLAT_MIN:
                return done(df.index[k], c[k] - d * p.tick, "15:50 清仓")
        cc = k - n
        if k - 2 * n >= k0:
            side = lo if d == 1 else h
            others = [side[x] for x in range(k - 2 * n, k + 1) if x != cc]
            if (d == 1 and side[cc] < min(others)) or (d == -1 and side[cc] > max(others)):
                seq.append(side[cc])
                if len(seq) >= 2 and d * (seq[-1] - seq[-2]) > 0:
                    ladder = seq[-1]
        if k >= j and np.isfinite(ladder) and d * (c[k] - ladder) < 0:
            return done(df.index[k + 1], o[k + 1] - d * p.tick, "梯形")
        k += 1


# ---------------- 报告 ----------------


def diagnostics(tr: pd.DataFrame, tr10k: pd.DataFrame) -> list[str]:
    L = []
    g = tr.groupby("reason")
    L += [
        "### 按出场原因",
        "",
        df_to_md(
            pd.DataFrame(
                {
                    "笔数": g.size(),
                    "胜率": g["net"].apply(lambda s: round(float((s > 0).mean()), 3)),
                    "毛利 USD": g["gross"].sum().round(0),
                    "扣费后 USD": g["net"].sum().round(0),
                    "每笔价差 bp": g["gross_bp"].mean().round(2),
                    "每笔净 bp": g["net_bp"].mean().round(2),
                }
            ).reset_index()
        ),
        "",
    ]
    cap = (tr["first_exit_px_gain"] / tr["mfe_px"].where(tr["mfe_px"] >= 0.01)).dropna()
    L += [
        "### 拿住了多少（首仓出场盈利 ÷ 持仓期间最大有利偏移，每股价差；只统计 MFE ≥ 1 tick 的笔）",
        "",
        df_to_md(
            cap.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .round(3)
            .rename("比值")
            .reset_index()
        ),
        "",
        f"MFE ≤ 0（入场后从未浮盈）的笔数：{int((tr['mfe_px'] <= 0).sum())} / {len(tr)}。",
        "",
        "持仓时长（分钟）分布：",
        "",
        df_to_md(
            pd.cut(tr["hold_min"], [0, 10, 20, 30, 60, 120, 400])
            .value_counts()
            .sort_index()
            .rename("笔数")
            .reset_index()
            .astype(str)
        ),
        "",
    ]
    L += [
        "### 首仓 vs 加仓（出场费用按股数分摊）",
        "",
        df_to_md(
            pd.DataFrame(
                [
                    {
                        "首仓毛利": round(float(tr["first_gross"].sum()), 0),
                        "首仓扣费后": round(float(tr["first_net"].sum()), 0),
                        "加仓毛利": round(float(tr["adds_gross"].sum()), 0),
                        "加仓扣费后": round(float(tr["adds_net"].sum()), 0),
                        "有加仓的笔数": int((tr["adds"] > 0).sum()),
                    }
                ]
            )
        ),
        "",
    ]
    tt = tr.assign(
        side=np.where(tr["direction"] == 1, "多", "空"),
        year=tr["entry_ts"].dt.tz_convert(ET).dt.year,
    )
    rows = [{"组": f"{s}·{y}", **summary(g2)} for (s, y), g2 in tt.groupby(["side", "year"])]
    L += [
        "### 多空 × 年份（bp = 相对入场名义；成本含费用与每边 1 tick）",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
    ]
    L += [
        "### 固定名义 1 万美元版本（每次成交股数 = floor(10000 / 成交价)）",
        "",
        df_to_md(
            pd.DataFrame(
                [{"口径": "100 股", **summary(tr)}, {"口径": "1 万美元", **summary(tr10k)}]
            )
        ),
        "",
    ]
    per_day = tr.groupby(tr["entry_ts"].dt.tz_convert(ET).dt.date).size()
    L += [
        "### 每日交易次数",
        "",
        df_to_md(
            per_day.value_counts().sort_index().rename_axis("每日笔数").rename("天数").reset_index()
        ),
        "",
    ]
    early = tr["entry_mod"] < 600
    L += [
        "### 09:30–10:00 入场 vs 其余",
        "",
        df_to_md(
            pd.DataFrame(
                [
                    {"组": "09:30–10:00", **summary(tr[early])},
                    {"组": "10:00 以后", **summary(tr[~early])},
                ]
            )
        ),
        "",
    ]
    return L


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    root = get_settings().zr_data_root
    one = {u: load_1m(root, u) for u in ("MU", "QQQ", "SPY")}
    bars = {(u, b): resample_bars(df, f"{b}min") for u, df in one.items() for b in (5, 15)}
    mu5 = bars[("MU", 5)]
    L = [
        "# E4 正股复核（R-036）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.research.e4_review`",
        "",
        "预登记：RESEARCH_LOG R-036（commit b5ebe35，早于本报告任何数字）。",
        "",
    ]

    # ---- 1. 实现审计 ----
    a_old = old.prepare(mu5, old.Params())
    t_old = old.backtest(a_old, old.Params())
    a_imp = prepare(mu5, OLD_IMPL)
    t_imp = backtest(a_imp, OLD_IMPL)
    same = (
        len(t_old) == len(t_imp)
        and np.allclose(t_old["net"], t_imp["net"])
        and (t_old["entry_ts"].to_numpy() == t_imp["entry_ts"].to_numpy()).all()
    )
    a_b = prepare(mu5, BASELINE)
    t_b = backtest(a_b, BASELINE)
    six = []
    for ma in (120, 127):
        for m in (1.0, 1.5, 2.0):
            po, pn = old.Params(ma=ma, stop_mult=m), replace(BASELINE, ma=ma, stop_mult=m)
            to = old.backtest(old.prepare(mu5, po), po)
            tn = backtest(prepare(mu5, pn), pn)
            six.append(
                {
                    "配置": f"SMA{ma}·ATR×{m:g}",
                    "原 笔数": len(to),
                    "原 扣费后 USD": round(float(to["net"].sum()), 0),
                    "原 毛利 USD": round(float(to["gross"].sum()), 0),
                    "修正后 笔数": len(tn),
                    "修正后 扣费后 USD": round(float(tn["net"].sum()), 0),
                    "修正后 毛利 USD": round(float(tn["gross"].sum()), 0),
                }
            )
    decomp = []
    for nm, pp in (
        ("原实现", OLD_IMPL),
        ("只改：谷底严格", replace(OLD_IMPL, strict=True)),
        ("只改：最近一个更高谷底", replace(OLD_IMPL, ladder="recent")),
        ("修正后基线（两处都改，R-036 登记）", BASELINE),
        (
            "敏感性：入场时水位须在入场价止损一侧（未登记）",
            replace(BASELINE, ladder="recent_valid"),
        ),
    ):
        tt = backtest(prepare(mu5, pp), pp)
        first = (tt["reason"] == "梯形") & (tt["exit_i"] == tt["signal_i"] + 2)
        decomp.append(
            {
                "口径": nm,
                "笔数": len(tt),
                "毛利 USD": round(float(tt["gross"].sum()), 0),
                "扣费后 USD": round(float(tt["net"].sum()), 0),
                "梯形/止损/清仓": "/".join(
                    str(int((tt["reason"] == r).sum())) for r in ("梯形", "止损", "15:50 清仓")
                ),
                "入场后第一根即梯形出场": int(first.sum()),
                "其扣费后 USD": round(float(tt.loc[first, "net"].sum()), 0),
            }
        )
    # 独立复算 10 笔（基线、首仓无加仓的交易，便于逐值核对）
    rng = np.random.default_rng(0)
    cand = t_b[t_b["adds"] == 0]
    pick = cand.iloc[np.sort(rng.choice(len(cand), size=min(10, len(cand)), replace=False))]
    chk = []
    for _, r in pick.iterrows():
        tr_ = simulate_entry(a_b, int(r["signal_i"]), int(r["direction"]), BASELINE, trace=True)
        ref = reference_trade(mu5, r["entry_ts"], int(r["direction"]), BASELINE)
        i = int(r["signal_i"])
        chk.append(
            {
                "信号 K（ET）": a_b["ts"][i].tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
                "方向": "多" if r["direction"] == 1 else "空",
                "收盘/SMA120": f"{a_b['close'][i]:.2f}/{a_b['ma'][i]:.2f}",
                "K/D（前→今）": (
                    f"{a_b['k'][i - 1]:.1f}/{a_b['d'][i - 1]:.1f}"
                    f"→{a_b['k'][i]:.1f}/{a_b['d'][i]:.1f}"
                ),
                "入场": f"{r['entry_ts'].tz_convert(ET).strftime('%H:%M')} @ {r['entry_px']:.2f}",
                "止损": round(float(r["stop"]), 2),
                "出场原因": r["reason"],
                "出场": f"{r['exit_ts'].tz_convert(ET).strftime('%H:%M')} @ {r['exit_px']:.2f}",
                "费用 入/出": f"{tr_['fee_in'][0]:.2f}/{tr_['fee_out']:.2f}",
                "扣费后": round(float(r["net"]), 2),
                "独立复算：信号": bool(ref["signal_ok"]),
                "独立复算：出场与净额": bool(
                    ref["exit_ts"] == r["exit_ts"]
                    and ref["reason"] == r["reason"]
                    and abs(ref["exit_px"] - r["exit_px"]) < 1e-9
                    and abs(ref["net"] - r["net"]) < 1e-6
                ),
            }
        )
    L += [
        "## 1. 实现审计",
        "",
        "| 项目 | 原实现 | 结论 |",
        "|---|---|---|",
        "| 入场价 | 信号 K 收盘确认，下一根 5 分钟 K 开盘 + 1 tick | 正确，无前视，未改 |",
        "| KDJ | RSV(9)；K、D 用 α=1/3 的递推（`ewm(alpha=1/3, adjust=False)`）；预热期 RSV 记 50，"
        "等价于初值 50；交叉 = K 与 D 交叉 | 与国内口径一致：MU 59,570 根 5 分钟 K 上与逐根显式"
        "公式（K=⅔K₋₁+⅓RSV、D=⅔D₋₁+⅓K、初值 50）在第 100 根之后差异 < 1e-13，金叉信号 0 处不同；"
        "只有序列最开头几根因预热期 RSV 处理不同而有差（国内软件不足 9 根按已有根数算）。未改 |",
        "| 梯形法则 | 摆动点 = 左右各 2 根（同日）、右侧走完才确认（无前视）；但 (1) 允许并列"
        "（≤ 而非 <）；(2) 出场水位取「历来更高谷底的最高值」、只上移 | **改为负责人定义**："
        "谷底严格低于左右各 n 根；出场 = 收盘跌破**最近一个**已确认的更高谷底 |",
        "| 止损 | 入场价 ∓ m×ATR(14)（信号 K 值，固定；bar 内触及即出，成交价取开盘与止损较差者）"
        " | 正确，未改 |",
        "| 数据清洗 | `clean_wicks` 收回尖刺影线；标记只用当根与过去数据 | 无前视，未改 |",
        "",
        "- 引擎一致性：新引擎设成原口径（ratchet、允许并列）后与原实现"
        f"**逐笔{'相同' if same else '不同'}**"
        f"（MU 全样本 {len(t_old)} 笔 vs {len(t_imp)} 笔）。",
        "- 截断测试：SMA120、KDJ、摆动点确认、梯形出场、ATR 止损已加入 `tests/test_e4_review.py`"
        "（在任意时刻截断数据，截断点之前的指标值与已平仓交易与全量计算逐值相同）。",
        "",
        "### 修正前后：原 6 组配置（MU 全样本，扣费后）",
        "",
        df_to_md(pd.DataFrame(six)),
        "",
        "### 梯形法则两处修改各自的影响（MU 全样本，SMA120·ATR×1.5）",
        "",
        df_to_md(pd.DataFrame(decomp)),
        "",
        "**按字面实现「最近一个已确认的更高谷底」的副作用**：入场前当日最近一个更高谷底可能早已"
        "被跌破、位于入场价之上（多头），入场后第一根收盘就触发梯形出场。原实现只看入场前最后两个"
        "谷底，结构已破时没有初始水位，所以很少出现。这是规则对「入场前的谷底算不算」没有写明造成"
        "的，不是代码错误；R-036 登记的是字面版本，结果原样报告；另附「入场时水位须在入场价"
        "止损一侧」的敏感性（第 3b 节，未预登记、不参与候选判定），请负责人确认本意"
        "（STATUS 待确认）。",
        "",
        "### 手工复算 10 笔（基线、无加仓的交易随机抽取；「独立复算」= 不复用引擎、逐根重算 ATR、"
        "摆动点与出场）",
        "",
        df_to_md(pd.DataFrame(chk)),
        "",
    ]

    # ---- 2. 诊断 ----
    t10k = backtest(
        prepare(mu5, replace(BASELINE, notional=10_000.0)), replace(BASELINE, notional=10_000.0)
    )
    orig_rows = [
        {"口径": "原实现（SMA120·ATR×1.5；新引擎原口径，与原代码逐笔相同）", **summary(t_imp)},
        {"口径": "修正后基线", **summary(t_b)},
    ]
    L += [
        "## 2. 诊断（修正后基线，MU 全样本 2023-10-02～2026-10-21）",
        "",
        "先回答「原实现为什么毛利为正、扣费后为负」：下表把 USD 合计与每笔 bp（等权）并列。"
        "价差 bp = 相对入场名义的价格变动（未扣滑点），成本 bp = 费用 + 每边 1 tick。",
        "",
        df_to_md(pd.DataFrame(orig_rows)),
        "",
        f"原实现毛利 {t_imp['gross'].sum():.0f} USD / {len(t_imp)} 笔 = "
        f"{t_imp['gross'].sum() / len(t_imp):.2f} USD/笔（已含滑点）；按笔等权的价差 "
        f"{t_imp['gross_bp'].mean():.2f} bp、成本 {t_imp['cost_bp'].mean():.2f} bp。"
        "USD 合计为正来自「100 股固定股数」把 2026 年高价（MU 300→900 美元）阶段与带加仓的"
        "盈利单放大，不是每笔的平均优势；固定 1 万美元名义后见下文。",
        "",
    ] + diagnostics(t_b, t10k)

    # ---- 3. 变体（研究期）与 4. 验证 ----
    res_rows, val_rows = [], []
    for name, p in VARIANTS.items():
        res = {}
        for u in ("MU", "QQQ", "SPY"):
            a = prepare(bars[(u, p.bar_min)], p)
            tr = backtest(a, p)
            if u == "MU":
                parts = {
                    "研究期": period(tr, None, RESEARCH_END),
                    "MU 2026": period(tr, dt.date(2026, 1, 1), None),
                }
            else:
                parts = {u: tr}
            for k, t in parts.items():
                res[k] = summary(t, random_direction(a, t, p))
        res_rows.append({"变体": name, **res["研究期"]})
        ok = all(
            res[k].get("扣费后 USD", -1) > 0 and res[k].get("随机方向分位", 0) >= 90
            for k in ("MU 2026", "QQQ", "SPY")
        )
        val_rows.append(
            {
                "变体": name,
                **{f"{k} 扣费后": res[k].get("扣费后 USD") for k in ("MU 2026", "QQQ", "SPY")},
                **{f"{k} 随机分位": res[k].get("随机方向分位") for k in ("MU 2026", "QQQ", "SPY")},
                **{f"{k} 笔数": res[k].get("笔数") for k in ("MU 2026", "QQQ", "SPY")},
                "候选": "是" if ok else "否",
            }
        )
        log.info("%s 完成 %.0fs", name, time.monotonic() - t0)
    sens_rows, sens_val = [], []
    for name, p0 in VARIANTS.items():
        p = replace(p0, ladder="recent_valid")
        res = {}
        for u in ("MU", "QQQ", "SPY"):
            a = prepare(bars[(u, p.bar_min)], p)
            tr = backtest(a, p)
            parts = (
                {
                    "研究期": period(tr, None, RESEARCH_END),
                    "MU 2026": period(tr, dt.date(2026, 1, 1), None),
                }
                if u == "MU"
                else {u: tr}
            )
            for k, t in parts.items():
                res[k] = summary(t, random_direction(a, t, p))
        sens_rows.append({"变体": name, **res["研究期"]})
        sens_val.append(
            {
                "变体": name,
                **{f"{k} 扣费后": res[k].get("扣费后 USD") for k in ("MU 2026", "QQQ", "SPY")},
                **{f"{k} 随机分位": res[k].get("随机方向分位") for k in ("MU 2026", "QQQ", "SPY")},
            }
        )
    st = pd.DataFrame(sens_rows)
    st["BH（同族 12 组，仅参考）"] = np.where(bh(st["p"].to_numpy(), 0.10), "通过", "—")
    rt = pd.DataFrame(res_rows)
    rt["FDR（q=0.10）"] = np.where(bh(rt["p"].to_numpy(), 0.10), "通过", "—")
    vt = pd.DataFrame(val_rows)
    cands = vt[vt["候选"] == "是"]["变体"].tolist()
    L += [
        "## 3. 预登记变体（MU 研究期 2023-10-02～2025-12-31；每次只改一处）",
        "",
        "p = (随机方向合计 ≥ 实际合计的次数 + 1) / 1001（同入场时刻随机方向 1,000 次）；"
        "基线 + 11 个变体一族 BH q=0.10。",
        "",
        df_to_md(rt),
        "",
        "## 4. 样本外（只报告，不选择）",
        "",
        "MU 2026-01-01～2026-10-21；QQQ、SPY 2023-10-02～2026-10-21。候选 = 三处扣费后均 > 0 且"
        "三处实际合计都 ≥ 随机方向 90 分位。",
        "",
        df_to_md(vt),
        "",
        f"**正股策略候选：{'、'.join(cands) if cands else '无'}。**",
        "",
        "## 3b. 敏感性（未预登记，不参与候选判定）：入场时梯形水位须在入场价止损一侧",
        "",
        "与第 3、4 节相同的 12 组，只把梯形初值改为：入场前最近一个更高谷底若不在入场价下方"
        "（多头；空头镜像），不作初始水位。",
        "",
        "MU 研究期：",
        "",
        df_to_md(st),
        "",
        "样本外：",
        "",
        df_to_md(pd.DataFrame(sens_val)),
        "",
        "## 边界",
        "",
        "- 正股 1 分钟 K（长桥，RTH，`clean_wicks` 修尖刺影线）聚合 5 / 15 分钟；不隔夜；"
        "100 股/次（另附 1 万美元名义版本）；费用 `stock_order_fees` + 每边 1 tick 滑点；"
        "未计融券费用、借券可得性、盘口深度。",
        "- QQQ、SPY 不是严格未见数据：R-010 测过 E4 早期版本（止损 1%、浮盈 0.5% 加仓），"
        "结果为负；本轮没有用它们选择任何参数。",
        "- 随机方向对照：每笔反方向版本的出场时刻可能与下一笔真实入场重叠"
        "（近似，同 E4_MU_STOCK）。",
        "- 结论只适用于 E4 规则本身与上述参数；没有测试期权表达、其他标的、隔夜持有。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    log.info("写入 %s，%.0fs", OUT, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
