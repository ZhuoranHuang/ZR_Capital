"""网上流行的六条单标的共振策略的正股回测（负责人插队任务）→
docs/research/POPULAR_CONFLUENCE_STOCK.md。

标的 MU、QQQ；周期：日线（长桥前复权日 K，VPS 一次性只读拉取，落 $ZR_DATA_ROOT/diag/daily_fwd/）、
1 小时与 15 分钟（本地长桥 RTH 1 分钟 K 按 09:30 起的网格合成，2023-10 起）。
策略全部用原作者默认参数、不调参（MACD 复合除外：按负责人指令只在样本内用 NSGA-II 优化后冻结），
多空对称，另报「只做多」。
成交：信号 K 收盘后、下一根 K 开盘成交，不利滑点 1 tick（0.01 美元）；费用 D-013 正股
（zr/costs/fees.stock_order_fees 的逐单公式）；做空不计借券费与保证金（未覆盖）。
资金：满仓复利，每次用全部权益开整数股，同一时间一个仓位；两档资金：1 万人民币（按测试起始日
USD/CNY 折美元，FRED DEXCHUS）与 10 万美元。日线可过夜；日内 15:50 ET（提前收盘日为收盘前 10 分钟）
按该分钟开盘价清仓，不过夜。持仓中出现反向入场信号：先平仓再反手（与 Pine strategy.entry 一致）。
对照：买入持有（测试期首根开盘买入整数股、末根收盘卖出，含费用）；方向随机对照（同样笔数与持仓
根数、随机时点与方向，1,000 次）的分位。
运行：uv run python -m zr.research.popular_confluence
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from zr.config import get_settings
from zr.costs.fees import stock_order_fees
from zr.data.calendar import session_close
from zr.features.bars import ET, load_bars
from zr.features.indicators import adx, atr, ema, kdj, sma, true_range, wilder_rsi

log = logging.getLogger("zr.research.popular_confluence")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "POPULAR_CONFLUENCE_STOCK.md"
UNDS = ("MU", "QQQ")
TICK = 0.01
N_RAND = 1000
SEED = 0
# USD/CNY（FRED DEXCHUS，美联储 H.10 纽约中午买入价）：https://fred.stlouisfed.org/series/DEXCHUS
USDCNY = {
    dt.date(2021, 10, 1): 6.4434,
    dt.date(2023, 10, 2): 7.2960,
    dt.date(2024, 1, 2): 7.1426,
    dt.date(2024, 7, 1): 7.2683,
}
DAILY_FROM, END = dt.date(2021, 10, 1), dt.date(2026, 10, 21)
INTRA_FROM = dt.date(2023, 10, 2)
SPLIT = {  # MACD 复合：样本内（优化）/ 样本外（冻结测试）
    "1d": ((DAILY_FROM, dt.date(2023, 12, 29)), (dt.date(2024, 1, 2), END)),
    "intra": ((INTRA_FROM, dt.date(2024, 6, 28)), (dt.date(2024, 7, 1), END)),
}


# ---- 费用（D-013 正股，按订单；与 zr/costs/fees.stock_order_fees 一致，见测试） ----


def stock_fee(shares: float, side: str, px: float) -> float:
    """已统一（D-035）：转调 zr.costs.fees.stock_order_fees（D-013）。"""
    if shares <= 0:
        return 0.0
    return float(stock_order_fees(int(shares), side, px)["total"])


# ---------------- 数据 ----------------


def daily(root: Path, und: str) -> pd.DataFrame:
    d = pd.read_parquet(root / "diag" / "daily_fwd" / f"{und}.parquet")
    d["day"] = pd.to_datetime(d["ts"]).dt.date  # naive UTC 04:00/05:00 = 美东当日 00:00
    d = d.drop_duplicates("day").set_index("day").sort_index()
    d = d[d.index <= END]
    d["start"] = [pd.Timestamp(dt.datetime.combine(x, dt.time(9, 30)), tz=ET) for x in d.index]
    return d[["open", "high", "low", "close", "volume", "start"]]


def intraday(root: Path, und: str, minutes: int) -> tuple[pd.DataFrame, pd.Series]:
    """RTH 1 分钟 K → 09:30 起的 minutes 分钟 K；另返回每日清仓价（15:50 那一分钟的开盘价）。"""
    m = load_bars(root, und, INTRA_FROM, END)
    et = m.index.tz_convert(ET)
    day = np.asarray(et.date)
    mins = (et.hour * 60 + et.minute - 570).to_numpy()
    df = pd.DataFrame(
        {
            "day": day,
            "bin": mins // minutes,
            "open": m["open"].to_numpy(),
            "high": m["high"].to_numpy(),
            "low": m["low"].to_numpy(),
            "close": m["close"].to_numpy(),
            "volume": m["volume"].to_numpy(float),
        }
    )
    g = df.groupby(["day", "bin"], sort=True)
    b = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    b["start"] = [
        pd.Timestamp(dt.datetime.combine(d, dt.time(9, 30)), tz=ET)
        + pd.Timedelta(minutes=int(k) * minutes)
        for d, k in zip(b["day"], b["bin"], strict=True)
    ]
    flat = {}
    for d, ix in df.groupby("day").indices.items():
        cut = (dt.datetime.combine(d, session_close(d)) - dt.timedelta(minutes=10)).time()
        k = cut.hour * 60 + cut.minute - 570
        sel = ix[mins[ix] >= k]
        flat[d] = float(df["open"].iloc[sel[0]]) if len(sel) else float(df["close"].iloc[ix[-1]])
    return b.drop(columns="bin"), pd.Series(flat)


# ---------------- 指标（Pine 口径） ----------------


def linreg(x: pd.Series, n: int) -> pd.Series:
    """Pine ta.linreg(x, n, 0)：最近 n 根的最小二乘直线在最后一根处的值。"""
    t = np.arange(n, dtype=float)
    tm = t.mean()
    w_slope = (t - tm) / ((t - tm) ** 2).sum()

    def f(v):
        b = (v * w_slope).sum()
        return v.mean() + b * (n - 1 - tm)

    return x.rolling(n, min_periods=n).apply(f, raw=True)


def supertrend(
    h: pd.Series, lo: pd.Series, c: pd.Series, n: int = 10, k: float = 3.0
) -> np.ndarray:
    """Pine ta.supertrend(k, n)：+1 上升 / −1 下降（ATR = RMA(TR, n)）。"""
    a = atr(h, lo, c, n).to_numpy()
    hl2 = ((h + lo) / 2).to_numpy()
    cc = c.to_numpy()
    up, dn = hl2 - k * a, hl2 + k * a
    lb, ub = up.copy(), dn.copy()
    d = np.full(len(cc), np.nan)
    for i in range(len(cc)):
        if not np.isfinite(a[i]):
            continue
        if i > 0 and np.isfinite(lb[i - 1]):
            lb[i] = up[i] if (up[i] > lb[i - 1] or cc[i - 1] < lb[i - 1]) else lb[i - 1]
            ub[i] = dn[i] if (dn[i] < ub[i - 1] or cc[i - 1] > ub[i - 1]) else ub[i - 1]
        if i == 0 or not np.isfinite(d[i - 1]):
            d[i] = -1.0  # Pine：首个有效值方向为 1（下降）
        elif d[i - 1] == -1.0:
            d[i] = 1.0 if cc[i] > ub[i] else -1.0
        else:
            d[i] = -1.0 if cc[i] < lb[i] else 1.0
    return d


def mfi(h: pd.Series, lo: pd.Series, c: pd.Series, v: pd.Series, n: int = 14) -> pd.Series:
    tp = (h + lo + c) / 3
    raw = tp * v
    ch = tp.diff()
    pos = raw.where(ch > 0, 0.0).rolling(n, min_periods=n).sum()
    neg = raw.where(ch < 0, 0.0).rolling(n, min_periods=n).sum()
    out = 100 - 100 / (1 + pos / neg.replace(0, np.nan))
    return out.mask((neg == 0) & (pos > 0), 100.0)  # 无负资金流时为 100（Pine ta.mfi）


def wavetrend(h, lo, c, n1: int = 10, n2: int = 21) -> tuple[pd.Series, pd.Series]:
    ap = (h + lo + c) / 3
    esa = ema(ap, n1)
    d = ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = ema(ci, n2)
    return wt1, sma(wt1, 4)


def _xup(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return (a > b) & (np.r_[np.nan, a[:-1]] <= np.r_[np.nan, b[:-1]])


def _xdn(a, b):
    return _xup(b, a)


# ---------------- 策略信号（收盘时刻；返回 {long, short, exit_long, exit_short}） ----------------


def s_squeeze(b: pd.DataFrame) -> dict:
    """LazyBear Squeeze Momentum（BB 20/2、KC 20/1.5、动能 = linreg）+ EMA200 过滤。"""
    c, h, lo = b["close"], b["high"], b["low"]
    basis, dev = sma(c, 20), 2.0 * c.rolling(20, min_periods=20).std(ddof=0)
    ubb, lbb = basis + dev, basis - dev
    ma, rng = sma(c, 20), sma(true_range(h, lo, c), 20)
    ukc, lkc = ma + 1.5 * rng, ma - 1.5 * rng
    on = ((lbb > lkc) & (ubb < ukc)).to_numpy()
    off = ((lbb < lkc) & (ubb > ukc)).to_numpy()
    rel = off & np.r_[False, on[:-1]]
    hh, ll = h.rolling(20, min_periods=20).max(), lo.rolling(20, min_periods=20).min()
    val = linreg(c - ((hh + ll) / 2 + sma(c, 20)) / 2, 20).to_numpy()
    e200 = ema(c, 200).to_numpy()
    cc = c.to_numpy()
    v1, v2 = np.r_[np.nan, val[:-1]], np.r_[np.nan, np.nan, val[:-2]]
    return {
        "long": rel & (val > 0) & (cc > e200),
        "short": rel & (val < 0) & (cc < e200),
        "exit_long": ((val < v1) & (v1 < v2)) | (val < 0),
        "exit_short": ((val > v1) & (v1 > v2)) | (val > 0),
    }


def s_supertrend(b: pd.DataFrame) -> dict:
    c, h, lo = b["close"], b["high"], b["low"]
    st = supertrend(h, lo, c, 10, 3.0)
    prev = np.r_[np.nan, st[:-1]]
    up, dn = (st == 1) & (prev == -1), (st == -1) & (prev == 1)
    ax = adx(h, lo, c, 14)["adx"].to_numpy()
    e200, cc = ema(c, 200).to_numpy(), c.to_numpy()
    return {
        "long": up & (ax > 25) & (cc > e200),
        "short": dn & (ax > 25) & (cc < e200),
        "exit_long": dn,
        "exit_short": up,
    }


def s_bb_rsi(b: pd.DataFrame) -> dict:
    """ChartArt「Bollinger + RSI, Double Strategy」：BB(200, 2)、RSI(14) 30/70。"""
    c = b["close"]
    basis, dev = sma(c, 200), 2.0 * c.rolling(200, min_periods=200).std(ddof=0)
    r = wilder_rsi(c, 14)
    lg = _xup(r, 30) & _xup(c, basis - dev)
    sh = _xdn(r, 70) & _xdn(c, basis + dev)
    return {"long": lg, "short": sh, "exit_long": sh, "exit_short": lg}


def s_bb_rsi_src(b: pd.DataFrame) -> dict:
    """同一脚本的源码默认参数（Pine v1.1：RSI(6) 以 50 为界、BB(200, 2)），与负责人给定的
    RSI(14) 30/70 不同，作附加对照。"""
    c = b["close"]
    basis, dev = sma(c, 200), 2.0 * c.rolling(200, min_periods=200).std(ddof=0)
    r = wilder_rsi(c, 6)
    lg = _xup(r, 50) & _xup(c, basis - dev)
    sh = _xdn(r, 50) & _xdn(c, basis + dev)
    return {"long": lg, "short": sh, "exit_long": sh, "exit_short": lg}


def s_wavetrend(b: pd.DataFrame) -> dict:
    c, h, lo, v = b["close"], b["high"], b["low"], b["volume"]
    wt1, wt2 = wavetrend(h, lo, c)
    m = mfi(h, lo, c, v, 14).to_numpy()
    mp = np.r_[np.nan, m[:-1]]
    xu, xd = _xup(wt1, wt2), _xdn(wt1, wt2)
    w = wt1.to_numpy()
    return {
        "long": xu & (w < -60) & (m > mp),
        "short": xd & (w > 60) & (m < mp),
        "exit_long": xd,
        "exit_short": xu,
    }


def s_four(b: pd.DataFrame) -> dict:
    """MACD(12,26,9) 柱、RSI(14) 相对 50、KDJ(9,3,3) K 相对 D、价格相对布林(20,2)中轨。"""
    c, h, lo = b["close"], b["high"], b["low"]
    dif = ema(c, 12) - ema(c, 26)
    hist = (dif - ema(dif, 9)).to_numpy()
    r = wilder_rsi(c, 14).to_numpy()
    kd = kdj(h, lo, c, 9, 3, 3)
    kk = (kd["k"] - kd["d"]).to_numpy()
    mid = (c - sma(c, 20)).to_numpy()
    votes = np.sign(hist) + np.sign(r - 50) + np.sign(kk) + np.sign(mid)
    valid = np.isfinite(hist) & np.isfinite(r) & np.isfinite(kk) & np.isfinite(mid)
    bull = np.stack([hist > 0, r > 50, kk > 0, mid > 0]).sum(0)
    bear = np.stack([hist < 0, r < 50, kk < 0, mid < 0]).sum(0)
    return {
        "long": valid & (votes == 4),
        "short": valid & (votes == -4),
        "exit_long": valid & (bear >= 2),
        "exit_short": valid & (bull >= 2),
    }


MACD_SPACE = (  # (名, 下限, 上限, 整数)
    ("fast", 3, 20, True),
    ("slow", 15, 60, True),
    ("signal", 3, 15, True),
    ("use_ema", 0, 1, False),
    ("ema_len", 20, 250, True),
    ("use_adx", 0, 1, False),
    ("adx_thr", 10, 40, False),
    ("use_rsi", 0, 1, False),
    ("rsi_lo", 20, 50, False),
    ("rsi_hi", 50, 80, False),
    ("use_vol", 0, 1, False),
    ("vol_mult", 1.0, 3.0, False),
)


def decode(x) -> dict:
    p = {}
    for (n, a, b_, isint), v in zip(MACD_SPACE, x, strict=True):
        v = float(np.clip(v, a, b_))
        p[n] = int(round(v)) if isint else v
    p["slow"] = max(p["slow"], p["fast"] + 2)
    for k in ("use_ema", "use_adx", "use_rsi", "use_vol"):
        p[k] = p[k] > 0.5
    return p


def s_macd(b: pd.DataFrame, p: dict) -> dict:
    c, h, lo, v = b["close"], b["high"], b["low"], b["volume"]
    dif = ema(c, p["fast"]) - ema(c, p["slow"])
    sig = ema(dif, p["signal"])
    xu, xd = _xup(dif, sig), _xdn(dif, sig)
    okl, oks = np.ones(len(c), bool), np.ones(len(c), bool)
    cc = c.to_numpy()
    if p["use_ema"]:
        e = ema(c, p["ema_len"]).to_numpy()
        okl &= cc > e
        oks &= cc < e
    if p["use_adx"]:
        a = adx(h, lo, c, 14)["adx"].to_numpy() > p["adx_thr"]
        okl &= a
        oks &= a
    if p["use_rsi"]:
        r = wilder_rsi(c, 14).to_numpy()
        okl &= (r > p["rsi_lo"]) & (r < p["rsi_hi"])
        oks &= (r > 100 - p["rsi_hi"]) & (r < 100 - p["rsi_lo"])
    if p["use_vol"]:
        vv = (v > p["vol_mult"] * sma(v, 20)).to_numpy()
        okl &= vv
        oks &= vv
    return {"long": xu & okl, "short": xd & oks, "exit_long": xd, "exit_short": xu}


STRATS = {
    "squeeze_ema200": s_squeeze,
    "supertrend_adx_ema200": s_supertrend,
    "bb_rsi_chartart": s_bb_rsi,
    "bb_rsi_chartart_src": s_bb_rsi_src,
    "wavetrend_mfi": s_wavetrend,
    "four_confluence": s_four,
}


# ---------------- 回测 ----------------


@dataclass
class Frame:
    o: np.ndarray
    c: np.ndarray
    day: np.ndarray
    start: np.ndarray  # 每根 K 开始时刻（ns）
    flat_px: np.ndarray  # 日内：该根 K 内须清仓时的价格（否则 NaN）
    enter_next_ok: np.ndarray  # 下一根 K 开盘可入场
    intraday: bool


def make_frame(b: pd.DataFrame, flat: pd.Series | None, minutes: int | None) -> Frame:
    day = np.asarray([x.date() for x in b["start"]])
    st = np.asarray([x.value for x in b["start"]], dtype=np.int64)
    n = len(b)
    fp = np.full(n, np.nan)
    ok = np.ones(n, bool)
    if minutes is not None:
        nxt_same = np.r_[day[1:] == day[:-1], False]
        for i in range(n):
            cut = pd.Timestamp(
                dt.datetime.combine(day[i], session_close(day[i])) - dt.timedelta(minutes=10), tz=ET
            ).value
            end = st[i] + minutes * 60 * 10**9
            last = not nxt_same[i]
            if (st[i] <= cut < end) or last:
                fp[i] = flat[day[i]]
            ok[i] = nxt_same[i] and (i + 1 < n and st[i + 1] < cut)
        ok &= np.isnan(fp)
    return Frame(
        b["open"].to_numpy(float), b["close"].to_numpy(float), day, st, fp, ok, minutes is not None
    )


def simulate(
    f: Frame,
    sig: dict,
    lo_i: int,
    hi_i: int,
    capital: float,
    fees: bool = True,
    long_only: bool = False,
) -> dict:
    """在 [lo_i, hi_i] 根 K 内回测。返回成交记录与逐根权益。"""
    eq = capital
    pos, sh, px0, f_in, i0 = 0, 0, 0.0, 0.0, -1
    trades, curve = [], np.full(hi_i - lo_i + 1, np.nan)
    pend = None  # ("exit"/"enter", 方向)

    def fee(n, side, px):
        return stock_fee(n, side, px) if fees else 0.0

    def close(i, px, why):
        nonlocal eq, pos, sh
        px = px - pos * TICK
        fo = fee(sh, "Sell" if pos == 1 else "Buy", px)
        gross = pos * sh * (px - px0)
        eq += gross - fo
        trades.append(
            {"i0": i0, "i1": i, "dir": pos, "ret": pos * (px / px0 - 1), "net": gross - fo - f_in}
        )
        pos, sh = 0, 0

    for i in range(lo_i, hi_i + 1):
        if pend is not None:
            act, d = pend
            pend = None
            if act in ("exit", "flip") and pos != 0:
                close(i, f.o[i], "exit")
            if act in ("enter", "flip") and pos == 0:
                px = f.o[i] + d * TICK
                n = int(eq // px)
                if n >= 1:
                    pos, sh, px0, i0 = d, n, px, i
                    f_in = fee(n, "Buy" if d == 1 else "Sell", px)
                    eq -= f_in
        if pos != 0 and np.isfinite(f.flat_px[i]):
            close(i, f.flat_px[i], "flat")
        curve[i - lo_i] = eq + (pos * sh * (f.c[i] - px0) if pos else 0.0)
        if curve[i - lo_i] <= 0:  # 权益归零（多为满仓做空遇大涨）：按收盘强平，此后不再交易
            if pos != 0:
                close(i, f.c[i], "wipe")
            eq = 0.0
            curve[i - lo_i :] = 0.0
            break
        if i == hi_i:
            if pos != 0:
                close(i, f.c[i], "end")  # 测试期末根收盘平仓（同样 1 tick 不利滑点）
                curve[-1] = eq
            break
        el, es = bool(sig["exit_long"][i]), bool(sig["exit_short"][i])
        nl = bool(sig["long"][i])
        ns = bool(sig["short"][i]) and not long_only
        can = f.enter_next_ok[i] if f.intraday else True
        if pos == 1 and (el or ns):
            pend = ("flip", -1) if (ns and can) else ("exit", 0)
        elif pos == -1 and (es or nl):
            pend = ("flip", 1) if (nl and can) else ("exit", 0)
        elif pos == 0 and can and (nl or ns):
            pend = ("enter", 1 if nl else -1)
    return {"trades": trades, "curve": curve, "final": eq, "wiped": bool(eq <= 0)}


def metrics(res: dict, f: Frame, lo_i: int, hi_i: int, capital: float) -> dict:
    tr = pd.DataFrame(res["trades"])
    curve = res["curve"]
    days = (f.day[hi_i] - f.day[lo_i]).days or 1
    R = res["final"] / capital - 1
    peak = np.maximum.accumulate(curve)
    dd = float(np.nanmax((peak - curve) / peak)) if len(curve) else 0.0
    dser = pd.Series(curve, index=f.day[lo_i : hi_i + 1]).groupby(level=0).last()
    r = dser.pct_change().dropna()
    win = tr[tr["net"] > 0]["net"] if len(tr) else pd.Series(dtype=float)
    loss = tr[tr["net"] <= 0]["net"] if len(tr) else pd.Series(dtype=float)
    return {
        "trades": len(tr),
        "win_rate": float((tr["net"] > 0).mean()) if len(tr) else math.nan,
        "payoff": float(win.mean() / -loss.mean())
        if len(win) and len(loss) and loss.mean() < 0
        else math.nan,
        "ret": R,
        "ann": (1 + R) ** (365.25 / days) - 1 if R > -1 else -1.0,
        "mdd": dd,
        "sharpe": float(r.mean() / r.std() * np.sqrt(252))
        if len(r) > 2 and r.std() > 0
        else math.nan,
        "wiped": res.get("wiped", False),
    }


def buy_hold(f: Frame, lo_i: int, hi_i: int, capital: float) -> float:
    px = f.o[lo_i] + TICK
    n = int(capital // px)
    if n < 1:
        return 0.0
    end = f.c[hi_i] - TICK
    pnl = n * (end - px) - stock_fee(n, "Buy", px) - stock_fee(n, "Sell", end)
    return pnl / capital


def random_control(
    res: dict, f: Frame, lo_i: int, hi_i: int, capital: float, long_only: bool, seed: int
) -> np.ndarray:
    """同样笔数与持仓根数、随机时点与方向（只做多版本只做多）的复利收益，N_RAND 次。"""
    tr = res["trades"]
    if not tr:
        return np.array([])
    dur = np.array([max(t["i1"] - t["i0"], 1) for t in tr])
    n_bars = hi_i - lo_i
    idle = max(n_bars - dur.sum(), 0)
    rng = np.random.default_rng(seed)
    out = np.empty(N_RAND)
    for k in range(N_RAND):
        dd = rng.permutation(dur)
        gaps = rng.multinomial(idle, np.ones(len(dd) + 1) / (len(dd) + 1))[:-1]
        dirs = np.ones(len(dd), int) if long_only else rng.choice([-1, 1], len(dd))
        eq, i = capital, lo_i
        for g, du, d in zip(gaps, dd, dirs, strict=True):
            i0 = i + int(g)
            if i0 >= hi_i:
                break
            i1 = min(i0 + int(du), hi_i)
            if f.intraday:  # 不过夜：跨日则在入场当日清仓价平
                same = np.flatnonzero(f.day[i0 : i1 + 1] != f.day[i0])
                fp = np.flatnonzero(np.isfinite(f.flat_px[i0 : i1 + 1]))
                if len(fp) and (not len(same) or fp[0] < same[0]):
                    x_px, i1 = f.flat_px[i0 + fp[0]], i0 + fp[0]
                else:
                    x_px = f.o[i1]
            else:
                x_px = f.o[i1] if i1 < hi_i else f.c[hi_i]
            p0 = f.o[i0] + d * TICK
            n = int(eq // p0)
            if n >= 1:
                p1 = x_px - d * TICK
                eq += (
                    d * n * (p1 - p0)
                    - stock_fee(n, "Buy" if d == 1 else "Sell", p0)
                    - stock_fee(n, "Sell" if d == 1 else "Buy", p1)
                )
            i = i1 + 1
            if eq <= 0:
                eq = 0.0
                break
        out[k] = eq / capital - 1
    return out


# ---------------- NSGA-II（MACD 复合） ----------------


def optimize_macd(
    b: pd.DataFrame, f: Frame, lo_i: int, hi_i: int, long_only: bool, seed: int = 1
) -> dict:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    xl = np.array([a for _, a, _, _ in MACD_SPACE], float)
    xu = np.array([u for _, _, u, _ in MACD_SPACE], float)
    cap = 100_000.0

    class P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=len(xl), n_obj=2, xl=xl, xu=xu)

        def _evaluate(self, x, out, *a, **k):
            res = simulate(f, s_macd(b, decode(x)), lo_i, hi_i, cap, True, long_only)
            m = metrics(res, f, lo_i, hi_i, cap)
            out["F"] = [-m["ret"], m["mdd"]]

    r = minimize(P(), NSGA2(pop_size=40), ("n_gen", 25), seed=seed, verbose=False)
    best, bs = None, -np.inf
    for x in np.atleast_2d(r.X):
        p = decode(x)
        m = metrics(simulate(f, s_macd(b, p), lo_i, hi_i, cap, True, long_only), f, lo_i, hi_i, cap)
        s = m["sharpe"] if np.isfinite(m["sharpe"]) else -np.inf
        if s > bs:
            best, bs = p, s
    return {"params": best, "pareto_n": len(np.atleast_2d(r.X)), "evals": 40 * 25}


# ---------------- 主流程 ----------------


def _idx(f: Frame, a: dt.date, b: dt.date) -> tuple[int, int]:
    m = np.flatnonzero((f.day >= a) & (f.day <= b))
    return int(m[0]), int(m[-1])


def run_cell(und, tf, b, f, strat, sig, a, z, long_only, seed, extra=None) -> list[dict]:
    lo_i, hi_i = _idx(f, a, z)
    rows = []
    rate = USDCNY[min(USDCNY, key=lambda d: abs((d - a).days))]
    for cap_name, cap in (("1 万人民币", 10_000 / rate), ("10 万美元", 100_000.0)):
        res = simulate(f, sig, lo_i, hi_i, cap, True, long_only)
        gross = simulate(f, sig, lo_i, hi_i, cap, False, long_only)
        m = metrics(res, f, lo_i, hi_i, cap)
        bh = buy_hold(f, lo_i, hi_i, cap)
        rc = random_control(res, f, lo_i, hi_i, cap, long_only, seed)
        rows.append(
            {
                "und": und,
                "tf": tf,
                "strategy": strat,
                "side": "只做多" if long_only else "多空",
                "capital": cap_name,
                "window": f"{f.day[lo_i]}～{f.day[hi_i]}",
                **m,
                "ret_gross": gross["final"] / cap - 1,
                "bh": bh,
                "vs_bh": m["ret"] - bh,
                "rand_pct": float((rc < m["ret"]).mean()) if len(rc) else math.nan,
                "usdcny": rate if cap_name.startswith("1 万") else math.nan,
                **(extra or {}),
            }
        )
    return rows


def run_und_tf(root: Path, und: str, tf: str) -> list[dict]:
    if tf == "1d":
        b = daily(root, und)
        f = make_frame(b, None, None)
        a, z, sp = DAILY_FROM, END, SPLIT["1d"]
    else:
        mins = {"1h": 60, "15m": 15}[tf]
        b, flat = intraday(root, und, mins)
        f = make_frame(b, flat, mins)
        a, z, sp = INTRA_FROM, END, SPLIT["intra"]
    rows = []
    for k, (name, fn) in enumerate(STRATS.items()):
        sig = fn(b)
        for lo in (False, True):
            rows += run_cell(und, tf, b, f, name, sig, a, z, lo, SEED + k)
    for lo in (False, True):
        lo_i, hi_i = _idx(f, *sp[0])
        opt = optimize_macd(b, f, lo_i, hi_i, lo)
        sig = s_macd(b, opt["params"])
        ps = ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in opt["params"].items()
        )
        for part, (pa, pz) in (("样本内", sp[0]), ("样本外", sp[1])):
            rows += run_cell(
                und,
                tf,
                b,
                f,
                "macd_nsga2",
                sig,
                pa,
                pz,
                lo,
                SEED + 99,
                {"macd_part": part, "macd_params": ps},
            )
    log.info("%s %s 完成", und, tf)
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    jobs = [(u, tf) for u in UNDS for tf in ("1d", "1h", "15m")]
    parts = Parallel(n_jobs=4)(delayed(run_und_tf)(root, u, tf) for u, tf in jobs)
    df = pd.DataFrame([r for p in parts for r in p])
    st = root / "research" / "popular_confluence"
    st.mkdir(parents=True, exist_ok=True)
    df.to_parquet(st / "results.parquet", index=False)
    from zr.research import popular_confluence_report

    popular_confluence_report.write(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
