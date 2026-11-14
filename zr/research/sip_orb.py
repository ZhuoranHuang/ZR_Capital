# ruff: noqa: E501  （中文文档字符串较长）
"""R-057：Zarattini、Barbon、Aziz（2024）5 分钟开盘区间突破 + Stocks in Play 的样本外复现（预登记见 RESEARCH_LOG R-057）。

本模块第一部分：标的名单过滤、每日合格集合（只用前一日及更早 + 当日开盘价）、拆股剔除、开盘区间与 RelVol、选股。
VPS 上的拉取脚本（zr/data/sip_pull.py）与 Mac 上的回测共用这些函数。
"""

from __future__ import annotations

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


def daily_eligibility(d: pd.DataFrame) -> pd.DataFrame:
    """单个标的的日线（不复权 open/high/low/close/volume + 前复权 fwd_open，按日期升序）→ 每日合格标记。
    当日合格 ⇔ 当日开盘 > 5 美元 且 前 14 个交易日日均量 ≥ 100 万股 且 前一日 ATR(14)（Wilder，不复权）> 0.50 美元
    且不在拆股剔除期内。除当日开盘价外只用前一日及更早的数据；拆股识别用前复权开盘 ÷ 不复权开盘的因子
    （开盘即可知），因子单日变化超过 ×1.2 / ÷1.2 记为拆股日 s，交易日序号 s … s+14 剔除。"""
    d = d.sort_values("date").reset_index(drop=True)
    atr = wilder_atr(d["high"], d["low"], d["close"])
    out = pd.DataFrame({"date": d["date"], "open": d["open"]})
    out["atr_prev"] = atr.shift(1)
    out["vol14_prev"] = d["volume"].rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (d["fwd_open"] / d["open"]).to_numpy()
        ch = np.r_[1.0, f[:-1] / f[1:]]
    split = (ch > SPLIT_BAND) | (ch < 1 / SPLIT_BAND)
    idx = np.arange(len(d))
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
