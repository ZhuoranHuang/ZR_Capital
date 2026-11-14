# ruff: noqa: E501  （中文文档字符串较长）
"""R-057：Zarattini、Barbon、Aziz（2024）5 分钟开盘区间突破 + Stocks in Play 的样本外复现（预登记见 RESEARCH_LOG R-057）。

本模块第一部分：标的名单过滤、每日合格集合（只用前一日及更早 + 当日开盘价）、拆股剔除、开盘区间与 RelVol、选股。
VPS 上的拉取脚本（zr/data/sip_pull.py）与 Mac 上的回测共用这些函数。
"""

from __future__ import annotations

import functools
import io
import re

import numpy as np
import pandas as pd

ET = "America/New_York"
MIN_PRICE = 5.0
MIN_VOL = 1_000_000
MIN_ATR = 0.50
LOOKBACK = 14
SPLIT_BAND = 1.2  # 前复权因子单日变化超过 ×1.2 或 ÷1.2 视为拆股 / 合股
SPLIT_BLOCK = 14  # 拆股日 s 起，交易日序号 s … s+14 的股票日剔除（窗口里含拆股前的数据）

# ---------------- 标的名单 ----------------

NEG = [
    ("warrant", r"\bwarrants?\b"),
    ("unit", r"\bunits?\b"),
    ("right", r"\brights?\b"),
    ("preferred", r"\bpreferred\b|\bpfd\b|%"),
    ("depositary", r"depositary|\bADS\b|\bADR\b"),
    ("note", r"\bnotes?\b|\bdebentures?\b|\bbonds?\b"),
]


def parse_symbol_directory(nasdaq_txt: str, other_txt: str) -> tuple[pd.DataFrame, str]:
    """纳斯达克官方 Symbol Directory → (名单, 文件日期)。include = NYSE（otherlisted 交易所 N）或 Nasdaq 上市、
    非 ETF、非测试代码、名称含 Common Stock / Ordinary Share(s)，且不是权证、单位、权利、优先股、存托凭证、债券。"""

    def read(txt):
        lines = [x for x in txt.strip().splitlines() if not x.startswith("File Creation Time")]
        return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False)

    m = re.search(r"File Creation Time: (\d{2})(\d{2})(\d{4})", nasdaq_txt)
    listed = f"{m.group(3)}-{m.group(1)}-{m.group(2)}" if m else ""
    a = read(nasdaq_txt)
    a = pd.DataFrame(
        {
            "symbol": a["Symbol"],
            "name": a["Security Name"],
            "exchange": "Q",
            "etf": a["ETF"],
            "test": a["Test Issue"],
        }
    )
    b = read(other_txt)
    b = pd.DataFrame(
        {
            "symbol": b["ACT Symbol"],
            "name": b["Security Name"],
            "exchange": b["Exchange"],
            "etf": b["ETF"],
            "test": b["Test Issue"],
        }
    )
    u = pd.concat([a, b], ignore_index=True)
    reason = []
    for r in u.itertuples():
        if r.exchange not in ("Q", "N"):
            reason.append("exchange")
        elif r.etf == "Y":
            reason.append("etf")
        elif r.test == "Y":
            reason.append("test")
        elif "$" in r.symbol:
            reason.append("preferred")
        else:
            neg = next((k for k, pat in NEG if re.search(pat, r.name, re.I)), None)
            if neg:
                reason.append(neg)
            elif re.search(r"common stock|ordinary share", r.name, re.I):
                reason.append("")
            else:
                reason.append("not_common")
    u["reason"] = reason
    u["include"] = u["reason"] == ""
    return u, listed


def lb_symbol(symbol: str) -> str:
    """纳斯达克目录代码 → 长桥代码（BRK.B → BRK.B.US）。"""
    return f"{symbol}.US"


# ---------------- 每日合格集合 ----------------


def wilder_atr(h: pd.Series, l_: pd.Series, c: pd.Series, n: int = LOOKBACK) -> pd.Series:
    tr = pd.concat([h - l_, (h - c.shift()).abs(), (l_ - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def daily_eligibility(d: pd.DataFrame, split_dates=None) -> pd.DataFrame:
    """单个标的的日线（不复权 open/high/low/close/volume + 前复权 fwd_open，按日期升序）→ 每日合格标记。
    当日合格 ⇔ 当日开盘 > 5 美元 且 前 14 个交易日日均量 ≥ 100 万股 且 前一日 ATR(14)（Wilder，不复权）> 0.50 美元
    且不在拆股剔除期内。除当日开盘价外只用前一日及更早的数据；拆股识别用前复权开盘 ÷ 不复权开盘的因子
    （开盘即可知），因子单日变化超过 ×1.2 / ÷1.2 记为拆股日 s，交易日序号 s … s+14 剔除。
    split_dates 不为 None 时（R-057 Massive 版）：拆股日取自拆股接口的执行日（提前公告，开盘即可知，调用方已按
    ×1.2 / ÷1.2 过滤），s = 该标的第一个日期 ≥ 执行日的交易日序号，不需要 fwd_open。"""
    d = d.sort_values("date").reset_index(drop=True)
    atr = wilder_atr(d["high"], d["low"], d["close"])
    out = pd.DataFrame({"date": d["date"], "open": d["open"]})
    out["atr_prev"] = atr.shift(1)
    out["vol14_prev"] = d["volume"].rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1)
    idx = np.arange(len(d))
    if split_dates is None:
        with np.errstate(invalid="ignore", divide="ignore"):
            f = (d["fwd_open"] / d["open"]).to_numpy()
            ch = np.r_[1.0, f[:-1] / f[1:]]
        split = (ch > SPLIT_BAND) | (ch < 1 / SPLIT_BAND)
    else:
        dates = np.asarray(d["date"], dtype=object)
        split = np.zeros(len(d), bool)
        for x in split_dates:
            k = int(np.searchsorted(dates, x, side="left"))
            if k < len(d):
                split[k] = True
    last_split = np.maximum.accumulate(np.where(split, idx, -(10**9)))
    out["split_block"] = (idx - last_split) <= SPLIT_BLOCK
    out["eligible"] = (
        (out["open"] > MIN_PRICE)
        & (out["vol14_prev"] >= MIN_VOL)
        & (out["atr_prev"] > MIN_ATR)
        & ~out["split_block"]
    ).fillna(False)
    return out


# ---------------- 开盘区间与 RelVol ----------------


def opening_range(b5: pd.DataFrame) -> pd.DataFrame:
    """单个标的的常规时段 5 分钟 K（ts = UTC 开盘时刻）→ 每日首根（09:30–09:35）K 的高低开收量、方向、RelVol。
    RelVol = 当日 ORVol ÷ 前 14 个有首根 K 的交易日 ORVol 均值（不足 14 个 → 空）。方向：收 > 开 = +1，收 < 开 = −1，相等 = 0。"""
    t = pd.DatetimeIndex(b5["ts"]).tz_convert(ET)
    first = b5[(t.hour == 9) & (t.minute == 30)].copy()
    first["date"] = pd.DatetimeIndex(first["ts"]).tz_convert(ET).date
    first = first.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    o = pd.DataFrame(
        {
            "date": first["date"],
            "or_open": first["open"],
            "or_high": first["high"],
            "or_low": first["low"],
            "or_close": first["close"],
            "or_vol": first["volume"],
        }
    )
    return or_stats(o)


def or_stats(o: pd.DataFrame) -> pd.DataFrame:
    """每日开盘区间（date、or_open、or_high、or_low、or_close、or_vol，按日期升序、只含有开盘区间的日子）→ 加 RelVol 与方向。
    RelVol = 当日 ORVol ÷ 前 14 个有开盘区间的交易日 ORVol 均值（不足 14 个 → 空）。方向：收 > 开 = +1，收 < 开 = −1，相等 = 0。"""
    o = o.reset_index(drop=True).copy()
    base = o["or_vol"].rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        o["relvol"] = np.where(base > 0, o["or_vol"] / base, np.nan)
    o["dir"] = np.sign(np.round(o["or_close"] - o["or_open"], 6)).astype(int)
    return o


def select(c: pd.DataFrame, n: int) -> pd.DataFrame:
    """每日：合格 ∩ RelVol ≥ 1 ∩ 非十字星，按 RelVol 降序（同值按代码）取前 n。"""
    x = c[c["eligible"].astype(bool) & (c["relvol"] >= 1) & (c["dir"] != 0)]
    x = x.sort_values(["date", "relvol", "symbol"], ascending=[True, False, True], kind="mergesort")
    return x.groupby("date", sort=True).head(n).reset_index(drop=True)


# ---------------- 执行（1 分钟或 5 分钟 K） ----------------

STOP_ATR = 0.10
RISK = 0.01
PER_SHARE = 0.0035


def sim_core(
    m,
    o,
    h,
    lo,
    c,
    d: int,
    or_high: float,
    or_low: float,
    atr: float,
    slip: float,
    conservative: bool,
):
    """numpy 版执行核心。m = 美东分钟数（09:30 = 570），o/h/lo/c 同长数组（当日常规时段、升序）。
    返回 (入场价, 止损价, 出场价, 入场下标, 出场下标, 是否止损) 或 None（未触发）。规则见 simulate_trade。"""
    idx = np.flatnonzero(m >= 575)
    if not len(idx):
        return None
    hit = idx[(h[idx] >= or_high) if d > 0 else (lo[idx] <= or_low)]
    if not len(hit):
        return None
    j = int(hit[0])
    entry = (max(or_high, o[j]) + slip) if d > 0 else (min(or_low, o[j]) - slip)
    stop = entry - d * STOP_ATR * atr
    if conservative and ((d > 0 and lo[j] <= stop) or (d < 0 and h[j] >= stop)):
        return entry, stop, stop - d * slip, j, j, True
    rest = (lo[j + 1 :] <= stop) if d > 0 else (h[j + 1 :] >= stop)
    if rest.any():
        k = j + 1 + int(rest.argmax())
        px = min(stop, o[k]) if d > 0 else max(stop, o[k])
        return entry, stop, px - d * slip, j, k, True
    return entry, stop, c[-1] - d * slip, j, len(c) - 1, False


def simulate_trade(
    bars: pd.DataFrame,
    d: int,
    or_high: float,
    or_low: float,
    atr: float,
    slip: float,
    conservative: bool,
):
    """单只单日：从 09:35 那根起，多头最高价 ≥ 区间高即触发，成交 max(区间高, 开盘) + 滑点；止损 = 成交价 − 0.10 × ATR，
    触发成交 min(止损, 开盘) − 滑点（入场那根按止损价）；保守 = 入场那根最低价 ≤ 止损即当根止损，乐观 = 从下一根起判；
    未止损按当日最后一根收盘 − 滑点。空头镜像。返回 dict 或 None（未触发）。"""
    t = pd.DatetimeIndex(bars["ts"]).tz_convert(ET)
    m = np.asarray(t.hour * 60 + t.minute)
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    r = sim_core(m, o, h, lo, c, d, or_high, or_low, atr, slip, conservative)
    if r is None:
        return None
    entry, stop, xpx, j, k, stopped = r
    return {
        "entry": float(entry),
        "stop": float(stop),
        "exit": float(xpx),
        "reason": "stop" if stopped else "close",
        "entry_ts": bars["ts"].iloc[j],
        "exit_ts": bars["ts"].iloc[k],
    }


def shares(equity: float, n: int, atr: float, entry: float, lev: float) -> int:
    """股数 = min(floor(权益 ÷ N × 1% ÷ (0.10 × ATR)), floor(lev × 权益 ÷ N ÷ 入场价))。"""
    if not (atr > 0 and entry > 0 and equity > 0):
        return 0
    by_risk = np.floor(equity / n * RISK / (STOP_ATR * atr) + 1e-9)
    by_cap = np.floor(lev * equity / n / entry + 1e-9)
    return int(max(0, min(by_risk, by_cap)))


def trade_fees(sh: int, entry: float, exit_: float, d: int, fee: str) -> float:
    """两侧佣金：per_share = 每股 0.0035 美元（不设最低）；d013 = zr.costs.fees.stock_order_fees（做空开仓 Sell、平仓 Buy）。"""
    if sh <= 0:
        return 0.0
    if fee == "per_share":
        return 2 * PER_SHARE * sh
    from zr.costs.fees import stock_order_fees

    o_side, c_side = ("Buy", "Sell") if d > 0 else ("Sell", "Buy")
    return float(
        stock_order_fees(sh, o_side, entry)["total"] + stock_order_fees(sh, c_side, exit_)["total"]
    )


@functools.lru_cache(maxsize=4_000_000)
def _d013(sh: int, side: str, px: float) -> float:
    from zr.costs.fees import stock_order_fees

    return float(stock_order_fees(sh, side, px)["total"])


def fees_vec(sh, entry, exit_, d, fee: str) -> np.ndarray:
    """向量版 trade_fees（D-013 经缓存逐笔调用 stock_order_fees，价格按 6 位小数作缓存键）。"""
    sh = np.asarray(sh, np.int64)
    if fee == "per_share":
        return 2 * PER_SHARE * sh.astype(float)
    out = np.zeros(len(sh))
    for k, (s_, e_, x_, d_) in enumerate(
        zip(sh, np.round(entry, 6), np.round(exit_, 6), d, strict=True)
    ):
        if s_ > 0:
            o_side, c_side = ("Buy", "Sell") if d_ > 0 else ("Sell", "Buy")
            out[k] = _d013(int(s_), o_side, float(e_)) + _d013(int(s_), c_side, float(x_))
    return out


def shares_vec(equity, n, atr, entry, lev) -> np.ndarray:
    atr, entry = np.asarray(atr, float), np.asarray(entry, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        by_risk = np.floor(equity / n * RISK / (STOP_ATR * atr) + 1e-9)
        by_cap = np.floor(lev * equity / n / entry + 1e-9)
    s = np.minimum(by_risk, by_cap)
    return np.where((atr > 0) & (entry > 0) & np.isfinite(s), np.maximum(s, 0), 0).astype(np.int64)


def portfolio(
    T: pd.DataFrame, equity0: float, n, lev: float, fee: str, compound: bool = True
) -> dict:
    """逐日：当日所有仓位按当日期初权益定股数（N = n；n 为 None 时 N = 当日笔数；n 为 dict 时 N = n[日期]，
    R-057 Massive 版基础版用「当日合格且非十字星只数」，09:35 已知），日末权益 = 期初 + Σ 盈亏。
    T：date / dir / atr / entry / exit（成交价，已含滑点）。compound=False 时每日都用 equity0（安慰剂比较用）。"""
    T = T.sort_values(["date", "rank"] if "rank" in T else ["date"], kind="mergesort").reset_index(
        drop=True
    )
    eq = equity0
    sh_all, pnl_all, rows = np.zeros(len(T), np.int64), np.zeros(len(T)), []
    d_all, a_all, e_all, x_all = (T[k].to_numpy() for k in ("dir", "atr", "entry", "exit"))
    for day, ix in T.groupby("date", sort=True).indices.items():
        base = eq if compound else equity0
        N = n[day] if isinstance(n, dict) else (n if n is not None else len(ix))
        sh = shares_vec(base, N, a_all[ix], e_all[ix], lev)
        p = sh * d_all[ix] * (x_all[ix] - e_all[ix]) - fees_vec(
            sh, e_all[ix], x_all[ix], d_all[ix], fee
        )
        sh_all[ix], pnl_all[ix] = sh, p
        pnl_d = float(p.sum())
        rows.append(
            {
                "date": day,
                "equity_start": base,
                "pnl": pnl_d,
                "ret": pnl_d / base,
                "n": int((sh > 0).sum()),
            }
        )
        if compound:
            eq += pnl_d
    T = T.assign(shares=sh_all, pnl=pnl_all)
    D = pd.DataFrame(rows)
    eqs = equity0 + np.cumsum(D["pnl"].to_numpy()) if compound else np.full(len(D), equity0)
    return {"trades": T, "daily": D, "equity": pd.Series(eqs, index=D["date"])}
