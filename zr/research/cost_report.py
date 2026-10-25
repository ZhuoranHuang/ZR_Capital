# ruff: noqa: E501  （中文报告行较长）
"""成本报告（只描述成本，不测策略）→ docs/research/COST_REPORT.md。

数据：本地 options_1m（ThetaData 1 分钟 NBBO 快照，zr.features.option_features.load_option_day）与
bars_1m（长桥 1 分钟 K，RTH）。启动时禁止 ThetaData 联网（ThetaClient.get_json 抛错）。
合约：每个交易分钟（09:30–15:49 起点）、DTE ∈ {0, 1}（到期日 − 交易日的日历日），平值 = 行权价最接近
该分钟正股价（上一根 1 分钟 K 的收盘；09:30 用 09:30 那根的开盘），看涨取平值、向上 1 / 2 档，看跌取平值、
向下 1 / 2 档。该分钟该合约无报价行 → 丢弃计数（不用邻近分钟填充）。有效报价：bid > 0、ask > 0、
ask >= bid、mid >= 0.05。
指标（按日落盘为聚合统计：每格 n、sum 与固定宽度直方图，不含原始报价）：
2.1 报价质量（价差 tick / %、> 15% 占比、bid == 0 占比、权利金）；2.2 往返成本
cost_rt = 价差 × 100 × 张数 + 买入费 + 卖出费（option_order_fees，买 ask、卖 bid，同一分钟），每张；
2.3 持有成本：同一行权价跨式 mid 的 h 分钟变化 / 2（单腿），对照 BS 理论 theta（中间价反推 IV，
r = 0.04、q = 0、T = (到期日 16:00 ET − 当前) / (365 × 24 × 3600) 年）；2.4 回本门槛
threshold% = (cost_rt − 单腿持有成本) / (|delta| × 100 × S) × 100（单腿持有成本 = change / 2，衰减时为负，
所以减去它即加上损耗）；3.1 用持有期末那一分钟的价差重算门槛；3.2 0DTE 平值 IV − 随后 30 分钟实现波动。
运行：python -m zr.research.cost_report [--only-und QQQ] [--from D] [--to D] [--force] [--dry-run]；
python -m zr.research.cost_report --report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.data import thetadata as _td
from zr.features.bars import ET, load_bars
from zr.features.option_features import load_option_day

log = logging.getLogger("zr.research.cost_report")
REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "docs" / "research" / "COST_REPORT.md"
UNDS = ("QQQ", "MU", "SPY")
SESSIONS = (("09:30", "10:00"), ("10:00", "11:30"), ("11:30", "14:00"), ("14:00", "15:30"), ("15:30", "15:50"))
SESS_NAMES = [f"{a}–{b}" for a, b in SESSIONS]
HS = (5, 15, 30)
QS = (1, 2, 5)
R = 0.04
MIN_MID = 0.05
MIN_N = 30
YEAR_S = 365 * 24 * 3600
MIN_PER_YEAR = 365 * 24 * 60
LOCK_STALE_H = 6
# 每个指标的直方图：(下界, 宽度, 箱数)；超出范围的值落在两端的箱里
BINS = {
    "spread_ticks": (-0.5, 1.0, 500),  # 先四舍五入到整数 tick，箱中心 = 整数
    "spread_pct": (0.0, 0.1, 1000),
    "premium_usd": (0.0, 1.0, 10000),
    "cost_usd": (0.0, 0.05, 10000),
    "cost_pct": (0.0, 0.05, 4000),
    "hold_usd": (-500.0, 0.05, 20000),
    "hold_pct": (-100.0, 0.05, 4000),
    "leg_chg_usd": (-500.0, 0.05, 20000),
    "theta_usd": (-500.0, 0.05, 20000),
    "thr_pct": (0.0, 0.001, 20000),
    "thr_end_pct": (0.0, 0.001, 20000),
    "thr_diff_pct": (-5.0, 0.001, 10000),
    "iv": (0.0, 0.005, 2000),
    "rv": (0.0, 0.005, 2000),
    "iv_rv": (-5.0, 0.005, 2000),
}
KEYS = ["metric", "dte", "m", "right", "sess", "h", "q"]


# ---------------- 纯函数（有测试） ----------------


@lru_cache(maxsize=500_000)
def _fee(q: int, side: str, cents: int) -> float:
    return float(option_order_fees(q, side, cents / 100)["total"])


def cost_rt(bid: float, ask: float, q: int) -> float:
    """q 张一次往返（买 ask、卖 bid，同一分钟）的总美元成本：价差 + 双边 D-013 费用。"""
    return (ask - bid) * 100 * q + _fee(q, "Buy", round(ask * 100)) + _fee(q, "Sell", round(bid * 100))


def hold_leg(change: np.ndarray | float) -> np.ndarray | float:
    """单腿持有成本 = 跨式中间价变化 / 2（跨式是两条腿之和）。"""
    return np.asarray(change, float) / 2 if np.ndim(change) else change / 2


def threshold_pct(cost_usd_contract: np.ndarray, delta: np.ndarray, S: np.ndarray, mult: int = 100):
    """回本门槛（%）：每张美元成本 / (|delta| × 合约乘数 × S) × 100。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(cost_usd_contract, float) / (np.abs(delta) * mult * np.asarray(S, float)) * 100


def valid_quotes(bid: np.ndarray, ask: np.ndarray) -> tuple[np.ndarray, dict]:
    """有效报价掩码与剔除计数：present（有报价行）、bid0、invalid（ask <= 0 或 ask < bid）、
    lowmid（mid < 0.05）、valid。NaN = 该分钟无报价行。"""
    bid, ask = np.asarray(bid, float), np.asarray(ask, float)
    present = np.isfinite(bid) & np.isfinite(ask)
    with np.errstate(invalid="ignore"):
        ok_px = present & (bid > 0) & (ask > 0) & (ask >= bid)
        mid = (bid + ask) / 2
        low = ok_px & (mid < MIN_MID)
    valid = ok_px & ~low
    c = {
        "missing": int((~present).sum()),
        "present": int(present.sum()),
        "bid0": int((present & (bid == 0)).sum()),
        "invalid": int((present & ~ok_px).sum()),
        "lowmid": int(low.sum()),
        "valid": int(valid.sum()),
    }
    return valid, c


def changes(mid: np.ndarray, valid: np.ndarray, h: int) -> tuple[np.ndarray, int]:
    """mid(t+h) − mid(t)；t 或 t+h 无效 → NaN（剔除并计数，不用邻近分钟填充）。"""
    n = len(mid)
    out = np.full(n, np.nan)
    if h < n:
        ok = valid[:-h] & valid[h:]
        out[:-h] = np.where(ok, mid[h:] - mid[:-h], np.nan)
    dropped = int((valid & ~np.isfinite(out)).sum())
    return out, dropped


# ---------------- Black–Scholes ----------------


def _bs(S, K, T, sig, call: bool):
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (R + 0.5 * sig**2) * T) / (sig * sq)
    d2 = d1 - sig * sq
    disc = K * np.exp(-R * T)
    if call:
        return S * ndtr(d1) - disc * ndtr(d2), d1, d2
    return disc * ndtr(-d2) - S * ndtr(-d1), d1, d2


def implied_vol(price, S, K, T, call: bool, iters: int = 60):
    """向量化二分法反推 IV；超出无套利区间或不收敛 → NaN。"""
    price, S, K, T = (np.asarray(x, float) for x in (price, S, K, T))
    disc = K * np.exp(-R * T)
    lo_b = np.maximum(S - disc, 0) if call else np.maximum(disc - S, 0)
    hi_b = S if call else disc
    ok = np.isfinite(price) & (T > 0) & (price > lo_b + 1e-9) & (price < hi_b)
    lo, hi = np.full(price.shape, 1e-4), np.full(price.shape, 5.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        p, _, _ = _bs(S, K, T, mid, call)
        up = p > price
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    iv = (lo + hi) / 2
    ok &= (iv > 1.1e-4) & (iv < 4.99)
    return np.where(ok, iv, np.nan)


def greeks(S, K, T, sig, call: bool):
    """(delta, theta 年化，每股)。"""
    _, d1, d2 = _bs(S, K, T, sig, call)
    phi = np.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
    disc = K * np.exp(-R * T)
    if call:
        delta = ndtr(d1)
        theta = -S * phi * sig / (2 * np.sqrt(T)) - R * disc * ndtr(d2)
    else:
        delta = ndtr(d1) - 1
        theta = -S * phi * sig / (2 * np.sqrt(T)) + R * disc * ndtr(-d2)
    return delta, theta


# ---------------- 单日 ----------------


class _Acc:
    """聚合：每格 n / sum（bin = −1 行）与直方图计数（bin >= 0 行）。"""

    def __init__(self):
        self.rows: list[pd.DataFrame] = []

    def add(self, metric, vals, dte, m, right, sess, h=0, q=0, bins_key=None):
        vals = np.asarray(vals, float)
        keep = np.isfinite(vals)
        if not keep.any():
            return
        v = vals[keep]
        n = len(v)
        df = pd.DataFrame(
            {
                "dte": np.broadcast_to(dte, len(vals))[keep],
                "m": np.broadcast_to(m, len(vals))[keep],
                "right": np.broadcast_to(right, len(vals))[keep],
                "sess": np.broadcast_to(sess, len(vals))[keep],
            }
        )
        df["metric"], df["h"], df["q"], df["v"] = metric, h, q, v
        lo, w, nb = BINS[bins_key or metric]
        df["bin"] = np.clip(np.floor((v - lo) / w).astype(np.int64), 0, nb - 1)
        g = df.groupby(KEYS + ["bin"], sort=False)["v"].agg(["count", "sum"]).reset_index()
        s = df.groupby(KEYS, sort=False)["v"].agg(["count", "sum"]).reset_index()
        s["bin"] = -1
        self.rows += [g, s]
        _ = n

    def count(self, metric, value, dte, m, right, sess, h=0, q=0):
        self.rows.append(
            pd.DataFrame(
                [{"metric": metric, "dte": dte, "m": m, "right": right, "sess": sess, "h": h, "q": q,
                  "bin": -1, "count": int(value), "sum": float(value)}]
            )
        )

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=KEYS + ["bin", "count", "sum"])
        d = pd.concat(self.rows, ignore_index=True)
        return d.groupby(KEYS + ["bin"], sort=False)[["count", "sum"]].sum().reset_index()


def _tick(und: str, mid: np.ndarray) -> np.ndarray:
    from zr.execution.ticks import tick_size

    sym = f"{und}260101C00100000.US"
    small, big = float(tick_size(sym, 0.5)), float(tick_size(sym, 10.0))
    return np.where(mid < 3.0, small, big)


def build_day(root: Path, und: str, day: dt.date, bars: pd.DataFrame) -> pd.DataFrame:
    od = load_option_day(root, und, day)
    if od is None or od["quotes"].empty:
        raise ValueError("无期权报价")
    q = od["quotes"]
    q = q.assign(expiry=pd.to_datetime(q["expiry"]).dt.date)
    q["dte"] = [(e - day).days for e in q["expiry"]]
    q = q[q["dte"].isin((0, 1))]
    b = bars[bars.index.tz_convert(ET).date == day]
    if b.empty:
        raise ValueError("无正股 1 分钟 K")
    open_ns = pd.Timestamp(f"{day} 09:30", tz=ET).value
    tgrid = open_ns + np.arange(0, 390 + 31) * 60 * 10**9  # 09:30 起每分钟（含 30 分钟 horizon 余量）
    blab = b.index.as_unit("ns").asi8
    # S(t) = 上一根 1 分钟 K 的收盘；09:30 用 09:30 那根的开盘
    j = np.searchsorted(blab + 60 * 10**9, tgrid, side="right") - 1
    S = np.where(j >= 0, b["close"].to_numpy(float)[np.clip(j, 0, None)], np.nan)
    S[0] = float(b["open"].iloc[0]) if blab[0] == open_ns else S[0]
    S = np.where(tgrid <= blab[-1] + 60 * 10**9, S, np.nan)
    hm = pd.DatetimeIndex(tgrid).tz_localize("UTC").tz_convert(ET).strftime("%H:%M")
    sess = np.full(len(tgrid), -1)
    for k, (a, z) in enumerate(SESSIONS):
        sess[(hm >= a) & (hm < z)] = k
    starts = np.flatnonzero(sess >= 0)
    acc = _Acc()
    for dte in (0, 1):
        x = q[q["dte"] == dte]
        if x.empty:
            continue
        exp = x["expiry"].iloc[0]
        ks = np.sort(x["strike"].unique())
        x = x.assign(t=pd.DatetimeIndex(x["ts"]).as_unit("ns").asi8)
        ti = ((x["t"].to_numpy() - open_ns) // (60 * 10**9)).astype(int)
        x = x[(ti >= 0) & (ti < len(tgrid))]
        ti = ti[(ti >= 0) & (ti < len(tgrid))]
        ki = np.searchsorted(ks, x["strike"].to_numpy())
        grid = {}
        for rt in ("call", "put"):
            mk = (x["right"] == rt).to_numpy()
            bid = np.full((len(tgrid), len(ks)), np.nan)
            ask = np.full((len(tgrid), len(ks)), np.nan)
            bid[ti[mk], ki[mk]] = x["bid"].to_numpy(float)[mk]
            ask[ti[mk], ki[mk]] = x["ask"].to_numpy(float)[mk]
            grid[rt] = (bid, ask)
        T = np.maximum((pd.Timestamp(f"{exp} 16:00", tz=ET).value - tgrid) / 1e9 / YEAR_S, 1e-8)
        atm = np.array([int(np.argmin(np.abs(ks - s))) if np.isfinite(s) else -1 for s in S])
        # 每个档位、方向的行权价下标（逐分钟）
        for right_i, rt in enumerate(("call", "put")):
            bidm, askm = grid[rt]
            for m in (0, 1, 2):
                kk = atm + (m if rt == "call" else -m)
                okk = (atm >= 0) & (kk >= 0) & (kk < len(ks))
                kk = np.clip(kk, 0, len(ks) - 1)
                rows = np.arange(len(tgrid))
                bid = np.where(okk, bidm[rows, kk], np.nan)
                ask = np.where(okk, askm[rows, kk], np.nan)
                # 跨式另一条腿：同一行权价的另一方向
                ob, oa = grid["put" if rt == "call" else "call"]
                obid = np.where(okk, ob[rows, kk], np.nan)
                oask = np.where(okk, oa[rows, kk], np.nan)
                K = np.where(okk, ks[kk], np.nan)
                v, _ = valid_quotes(bid[starts], ask[starts])
                vo, _ = valid_quotes(obid, oask)
                vall, _ = valid_quotes(bid, ask)
                for s_k in range(len(SESSIONS)):
                    sel = sess[starts] == s_k
                    _, cs = valid_quotes(bid[starts][sel], ask[starts][sel])
                    for name, val in cs.items():
                        acc.count(f"n_{name}", val, dte, m, right_i, s_k)
                mid = (bid + ask) / 2
                st = starts[v]
                ss = sess[st]
                sp = ask[st] - bid[st]
                md = mid[st]
                acc.add("spread_ticks", np.round(sp / _tick(und, md)), dte, m, right_i, ss)
                acc.add("spread_pct", sp / md * 100, dte, m, right_i, ss)
                acc.add("premium_usd", md * 100, dte, m, right_i, ss)
                acc.add("flag_wide", (sp / md * 100 > 15).astype(float), dte, m, right_i, ss, bins_key="iv")
                pres = np.isfinite(bid[starts]) & np.isfinite(ask[starts])
                acc.add("flag_bid0", (bid[starts][pres] == 0).astype(float), dte, m, right_i, sess[starts][pres], bins_key="iv")
                # IV / delta / theta（本腿，与另一腿）
                iv = implied_vol(mid, S, K, T, rt == "call")
                omid = (obid + oask) / 2
                oiv = implied_vol(omid, S, K, T, rt != "call")
                delta, theta = greeks(S, K, T, iv, rt == "call")
                _, otheta = greeks(S, K, T, oiv, rt != "call")
                otheta = np.where(vo, otheta, np.nan)
                acc.count("n_iv_fail", int((v & ~np.isfinite(iv[starts])).sum()), dte, m, right_i, -1)
                # 往返成本（每张）
                costs = {}
                for qn in QS:
                    cost = np.array([cost_rt(b_, a_, qn) / qn for b_, a_ in zip(bid[st], ask[st], strict=True)])
                    costs[qn] = cost
                    acc.add("cost_usd", cost, dte, m, right_i, ss, q=qn)
                    acc.add("cost_pct", cost / (md * 100) * 100, dte, m, right_i, ss, q=qn, bins_key="cost_pct")
                # 持有成本：同一行权价跨式
                strad_ok = vall & vo
                strad = np.where(strad_ok, mid + omid, np.nan)
                for h in HS:
                    chg, dropped = changes(strad, strad_ok, h)
                    acc.count("n_hold_dropped", dropped, dte, m, right_i, -1, h=h)
                    hl = hold_leg(chg[st]) * 100  # 美元 / 张（单腿）
                    acc.add("hold_usd", hl, dte, m, right_i, ss, h=h)
                    acc.add("hold_pct", chg[st] / strad[st] * 100, dte, m, right_i, ss, h=h)
                    lc, _ = changes(mid, vall, h)
                    acc.add("leg_chg_usd", lc[st] * 100, dte, m, right_i, ss, h=h)
                    th = (theta[st] + otheta[st]) / 2 * h / MIN_PER_YEAR * 100  # 单腿理论 theta，美元 / 张
                    acc.add("theta_usd", th, dte, m, right_i, ss, h=h)
                    # 门槛与期末价差敏感性
                    dl = delta[st]
                    acc.count("n_delta_nan", int((~np.isfinite(dl)).sum()), dte, m, right_i, -1, h=h)
                    end = st + h
                    end_ok = vall[np.clip(end, 0, len(tgrid) - 1)] & (end < len(tgrid))
                    for qn in QS:
                        thr = threshold_pct(costs[qn] - hl, dl, S[st])
                        acc.add("thr_pct", thr, dte, m, right_i, ss, h=h, q=qn)
                        if qn == 1:
                            eb, ea = bid[np.clip(end, 0, len(tgrid) - 1)], ask[np.clip(end, 0, len(tgrid) - 1)]
                            cend = np.array(
                                [cost_rt(b_, a_, 1) if ok_ else np.nan for b_, a_, ok_ in zip(eb, ea, end_ok, strict=True)]
                            )
                            thr_e = threshold_pct(cend - hl, dl, S[st])
                            acc.add("thr_end_pct", thr_e, dte, m, right_i, ss, h=h, q=1)
                            acc.add("thr_diff_pct", thr_e - thr, dte, m, right_i, ss, h=h, q=1)
        # IV − 随后 30 分钟实现波动（0DTE 平值）
        if dte == 0:
            kk = atm
            okk = kk >= 0
            rows = np.arange(len(tgrid))
            kk = np.clip(kk, 0, len(ks) - 1)
            ivs = []
            for rt in ("call", "put"):
                bidm, askm = grid[rt]
                bb, aa = np.where(okk, bidm[rows, kk], np.nan), np.where(okk, askm[rows, kk], np.nan)
                vv, _ = valid_quotes(bb, aa)
                ivs.append(np.where(vv, implied_vol((bb + aa) / 2, S, ks[kk], T, rt == "call"), np.nan))
            iv0 = (ivs[0] + ivs[1]) / 2
            lr = np.diff(np.log(S))
            rv = np.full(len(tgrid), np.nan)
            for t in starts:
                seg = lr[t : t + 30]
                if len(seg) == 30 and np.isfinite(seg).all():
                    rv[t] = np.std(seg, ddof=1) * math.sqrt(MIN_PER_YEAR)
            acc.add("iv", iv0[starts], 0, 0, -1, sess[starts])
            acc.add("rv", rv[starts], 0, 0, -1, sess[starts])
            acc.add("iv_rv", (iv0 - rv)[starts], 0, 0, -1, sess[starts])
    out = acc.frame()
    out.insert(0, "date", str(day))
    out.insert(0, "und", und)
    return out


# ---------------- 调度 ----------------


def _no_network(*a, **k):
    raise RuntimeError("cost_report 禁止联网")


def out_dir(root: Path, und: str | None = None) -> Path:
    p = root / "reports" / "cost_report"
    return p / f"underlying={und}" if und else p


def _days(root: Path, und: str, lo: dt.date | None, hi: dt.date | None) -> list[dt.date]:
    base = root / "options_1m" / f"underlying={und}"
    ds = sorted(dt.date.fromisoformat(p.name[5:]) for p in base.glob("date=*"))
    return [d for d in ds if (lo is None or d >= lo) and (hi is None or d <= hi)]


def run(root: Path, unds, lo, hi, force: bool, dry: bool) -> dict:
    lock = out_dir(root) / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        age_h = (time.time() - lock.stat().st_mtime) / 3600
        msg = "疑似残留锁（超过 6 小时），请人工确认后删除" if age_h > LOCK_STALE_H else "已有实例在运行"
        print(f"{msg}：{lock}")
        sys.exit(3)
    stats = {}
    try:
        for und in unds:
            days = _days(root, und, lo, hi)
            if dry:
                days = days[:3]
            od = out_dir(root, und)
            od.mkdir(parents=True, exist_ok=True)
            bars = load_bars(root, und, days[0], days[-1]) if days else None
            t0, fails, done = time.time(), [], 0
            for k, day in enumerate(days, 1):
                p = od / f"date={day}.parquet"
                fp = od / f"date={day}.FAILED.json"
                if p.exists() and not force:
                    continue
                t1 = time.time()
                try:
                    df = build_day(root, und, day, bars)
                    tmp = p.with_suffix(".parquet.tmp")
                    df.to_parquet(tmp, index=False)
                    os.replace(tmp, p)
                    if fp.exists():
                        fp.unlink()
                    done += 1
                    rows = len(df)
                except Exception as e:  # noqa: BLE001 - 单日失败隔离
                    fails.append(str(day))
                    fp.write_text(json.dumps({"type": type(e).__name__, "msg": str(e)[:500]}, ensure_ascii=False))
                    rows = 0
                el = time.time() - t1
                if el > 2:
                    log.warning("%s %s 用时 %.1f 秒（> 2 秒）", und, day, el)
                if k % 25 == 0 or k == len(days):
                    tot = time.time() - t0
                    eta = tot / k * (len(days) - k)
                    print(f"[{k}/{len(days)}] {und} {day} rows={rows} elapsed={tot:.0f}s ETA={eta:.0f}s", flush=True)
                if len(fails) > max(3, 0.10 * len(days)):
                    raise RuntimeError(f"{und} 失败日 {len(fails)} > 10%，停止：{fails[:10]}")
            stats[und] = {"days": len(days), "computed": done, "failed": fails}
    finally:
        lock.unlink(missing_ok=True)
    return stats


# ---------------- 汇总报告 ----------------


def load_all(root: Path, und: str) -> tuple[pd.DataFrame, int, int]:
    """逐批合并日聚合（不一次载入全部日）→ (跨日合并的聚合, 有 DTE 0 / 1 行的日数, 有 0DTE 有效报价的日数)。"""
    fs = sorted(out_dir(root, und).glob("date=*.parquet"))
    acc, n_any, n_0 = None, 0, 0
    for i in range(0, len(fs), 50):
        part = [pd.read_parquet(f) for f in fs[i : i + 50]]
        for x in part:
            n_any += int(len(x) > 0)
            n_0 += int(((x["metric"] == "n_valid") & (x["dte"] == 0) & (x["sum"] > 0)).any()) if len(x) else 0
        d = pd.concat([acc, *part] if acc is not None else part, ignore_index=True).drop(columns=["und", "date"], errors="ignore")
        acc = d.groupby(KEYS + ["bin"], sort=False)[["count", "sum"]].sum().reset_index()
    return (acc if acc is not None else pd.DataFrame()), n_any, n_0


def _quant(h: pd.DataFrame, key: str, qs=(0.25, 0.5, 0.75, 0.9)) -> dict:
    lo, w, _ = BINS[key]
    h = h.groupby("bin")["count"].sum().sort_index()
    c = h.cumsum().to_numpy()
    tot = c[-1] if len(c) else 0
    out = {}
    for qq in qs:
        if tot == 0:
            out[qq] = math.nan
            continue
        i = int(np.searchsorted(c, qq * tot, side="left"))
        out[qq] = lo + (h.index[i] + 0.5) * w
    return out


def _agg(d: pd.DataFrame, metric: str, key: str | None = None, **flt) -> dict:
    x = d[d["metric"] == metric]
    for k, v in flt.items():
        x = x[x[k].isin(v if isinstance(v, (list, tuple)) else [v])]
    s = x[x["bin"] == -1]
    n, sm = float(s["count"].sum()), float(s["sum"].sum())
    hist = x[x["bin"] >= 0]
    qd = _quant(hist, key or metric) if len(hist) else {}
    return {"n": int(n), "mean": sm / n if n else math.nan, **{f"p{int(k * 100)}": v for k, v in qd.items()}}


def _f(x, nd=2, n=None):
    if n is not None and n < MIN_N:
        return "n<30"
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def report(root: Path) -> str:
    from zr.ops.markdown import df_to_md

    L = ["# 成本报告（只描述成本，不测策略）", ""]
    bound = (
        "**边界**：1 分钟报价是快照，0DTE 约 3% 误差，价差估计偏粗；本报告只用于量级判断，不用于结算级对账。"
        "分位数取自固定宽度直方图的箱中心（价差 % 箱宽 0.1 个百分点、门槛箱宽 0.001 个百分点、往返成本 / 持有成本 / theta 箱宽 0.05 美元、权利金箱宽 1 美元），均值为精确值。"
        "任何格子观测 < 30 标 n<30、不给结论。"
    )
    L += [
        "`python -m zr.research.cost_report`（日聚合）→ `--report`（本文）。数据：本地 ThetaData 1 分钟 NBBO（options_1m）"
        "与长桥 1 分钟 K（bars_1m），运行时禁止联网。合约：DTE 0 / 1（日历日），平值 = 行权价最接近当分钟正股价"
        "（上一根 1 分钟 K 收盘），向虚值 1 / 2 档；有效报价 bid > 0、ask > 0、ask >= bid、mid >= 0.05。",
        "",
        bound,
        "",
    ]
    data, cov = {}, []
    for und in UNDS:
        d, n_any, n_0 = load_all(root, und)
        if d.empty:
            continue
        data[und] = d
        od = out_dir(root, und)
        fails = sorted(p.name[5:15] for p in od.glob("date=*.FAILED.json"))
        ds = sorted(p.name[5:15] for p in od.glob("date=*.parquet"))
        nv = _agg(d, "n_valid")["mean"] * _agg(d, "n_valid")["n"]
        cov.append(
            {
                "标的": und,
                "起": ds[0],
                "止": ds[-1],
                "交易日（成功）": len(ds),
                "有 DTE 0 / 1 合约的日数": n_any,
                "失败日": len(fails),
                "有效报价观测（合约分钟）": int(nv),
                "0DTE 有有效报价的日数": n_0,
                "备注": "SPY 只有已拉取部分（2023-10～2024-07）" if und == "SPY" else "",
            }
        )
    L += ["## 1 覆盖", "", df_to_md(pd.DataFrame(cov)), ""]
    # 2.1
    rows = []
    for und, d in data.items():
        for dte in (0, 1):
            for m in (0, 1, 2):
                for s_k, sn in enumerate(SESS_NAMES):
                    f = {"dte": dte, "m": m, "sess": s_k}
                    t = _agg(d, "spread_ticks", **f)
                    p = _agg(d, "spread_pct", **f)
                    pr = _agg(d, "premium_usd", **f)
                    w = _agg(d, "flag_wide", "iv", **f)
                    b0 = _agg(d, "flag_bid0", "iv", **f)
                    n = p["n"]
                    rows.append(
                        {
                            "标的": und, "DTE": dte, "档位": m, "时段": sn, "n": n,
                            "价差 tick 中位（P25/P75/P90）": "n<30" if n < MIN_N else f"{t['p50']:.0f}（{t['p25']:.0f}/{t['p75']:.0f}/{t['p90']:.0f}）",
                            "价差 % 中位（P25/P75/P90）": "n<30" if n < MIN_N else f"{p['p50']:.2f}%（{p['p25']:.2f}/{p['p75']:.2f}/{p['p90']:.2f}）",
                            "价差 > 15% 占比": _f(w["mean"] * 100, 1, n) + ("%" if n >= MIN_N else ""),
                            "bid = 0 占比": _f(b0["mean"] * 100, 1, n) + ("%" if n >= MIN_N else ""),
                            "每张成本（mid × 100）中位，美元": _f(pr["p50"], 0, n),
                            "2000 美元可买张数": "n<30" if n < MIN_N or not np.isfinite(pr["p50"]) or pr["p50"] <= 0 else int(2000 // pr["p50"]),
                        }
                    )
    L += ["## 2.1 报价质量（看涨看跌合并；数据：options_1m；样本 n = 有效报价的合约分钟）", "", bound, "", df_to_md(pd.DataFrame(rows)), ""]
    # 2.2
    rows = []
    for und, d in data.items():
        for dte in (0, 1):
            for m in (0, 1, 2):
                for qn in QS:
                    f = {"dte": dte, "m": m, "q": qn}
                    cu, cp = _agg(d, "cost_usd", **f), _agg(d, "cost_pct", **f)
                    rows.append(
                        {"标的": und, "DTE": dte, "档位": m, "张数": qn, "n": cu["n"],
                         "每张往返成本 中位 / 均值，美元": "n<30" if cu["n"] < MIN_N else f"{cu['p50']:.2f} / {cu['mean']:.2f}",
                         "占权利金 中位 / 均值": "n<30" if cp["n"] < MIN_N else f"{cp['p50']:.2f}% / {cp['mean']:.2f}%"}
                    )
    L += ["## 2.2 单次往返成本（价差 + D-013 双边费用，同一分钟买 ask 卖 bid；全部时段合并）", "", bound, "", df_to_md(pd.DataFrame(rows)), ""]
    # 2.3
    rows = []
    for und, d in data.items():
        for dte in (0, 1):
            for s_k, sn in enumerate(SESS_NAMES):
                for h in HS:
                    f = {"dte": dte, "m": 0, "sess": s_k, "h": h}
                    hu, hp, th = _agg(d, "hold_usd", **f), _agg(d, "hold_pct", **f), _agg(d, "theta_usd", **f)
                    lc_c = _agg(d, "leg_chg_usd", **f, right=0)
                    lc_p = _agg(d, "leg_chg_usd", **f, right=1)
                    n = hu["n"]
                    rows.append(
                        {"标的": und, "DTE": dte, "时段": sn, "h（分钟）": h, "n": n,
                         "单腿持有成本 均值 / 中位，美元": "n<30" if n < MIN_N else f"{hu['mean']:.2f} / {hu['p50']:.2f}",
                         "占权利金 均值 / 中位": "n<30" if n < MIN_N else f"{hp['mean']:.3f}% / {hp['p50']:.3f}%",
                         "理论 theta（BS）均值，美元": _f(th["mean"], 2, th["n"]),
                         "看涨腿 / 看跌腿 平均变化，美元": "n<30" if min(lc_c["n"], lc_p["n"]) < MIN_N else f"{lc_c['mean']:.2f} / {lc_p['mean']:.2f}"}
                    )
    L += [
        "## 2.3 持有成本（平值；同一行权价跨式中间价变化 ÷ 2）",
        "",
        "> 跨式的平均变化 ≈ 时间损耗 − 已实现波动带来的收益，即「什么都不做、只是拿着」的平均成本；单腿取一半是期望意义上的近似。",
        "",
        bound, "", df_to_md(pd.DataFrame(rows)), "",
    ]
    # 2.4（附录式，全表）
    rows = []
    for und, d in data.items():
        for dte in (0, 1):
            for m in (0, 1, 2):
                for right_i, rn in ((0, "call"), (1, "put")):
                    for h in HS:
                        for qn in QS:
                            t = _agg(d, "thr_pct", dte=dte, m=m, right=right_i, h=h, q=qn)
                            rows.append({"标的": und, "DTE": dte, "档位": m, "方向": rn, "h": h, "张数": qn, "n": t["n"],
                                         "门槛 中位": _f(t["p50"], 3, t["n"]) + ("%" if t["n"] >= MIN_N else ""),
                                         "门槛 均值": _f(t["mean"], 3, t["n"]) + ("%" if t["n"] >= MIN_N else "")})
    tbl24 = pd.DataFrame(rows)
    # 2.5
    excess = {5: [("开盘区间突破", 0.094)], 30: [("RSI(2) 回调", 0.141), ("均线回调", 0.084)]}
    rows = []
    for und, d in data.items():
        for h in HS:
            tc = _agg(d, "thr_pct", dte=0, m=0, right=0, h=h, q=1)
            tp = _agg(d, "thr_pct", dte=0, m=0, right=1, h=h, q=1)
            worst = max(tc["p50"], tp["p50"]) if min(tc["n"], tp["n"]) >= MIN_N else math.nan
            base = {"标的": und, "h（分钟）": h, "n（看涨 / 看跌）": f"{tc['n']} / {tp['n']}",
                    "门槛 看涨 中位": _f(tc["p50"], 3, tc["n"]) + ("%" if tc["n"] >= MIN_N else ""),
                    "门槛 看跌 中位": _f(tp["p50"], 3, tp["n"]) + ("%" if tp["n"] >= MIN_N else "")}
            if und != "MU" or h not in excess:
                rows.append({**base, "已测超额（MU，扣费前，相对随机）": "—", "判定": "无数据"})
                continue
            for name, ex in excess[h]:
                verdict = "n<30" if not np.isfinite(worst) else ("覆盖" if worst < ex else "不覆盖")
                rows.append({**base, "已测超额（MU，扣费前，相对随机）": f"{name} +{ex:.3f}%", "判定": verdict})
    L += [
        "## 2.5 总表：0DTE 平值、1 张的回本门槛 vs 已测超额",
        "",
        "门槛% = (每张往返成本 − 单腿持有成本) / (|delta| × 100 × S) × 100，取每个起始分钟的门槛的中位数（全部时段合并）；"
        "判定用看涨 / 看跌两者中较高的门槛（保守）：门槛 < 已测超额 → 覆盖，否则不覆盖。已测超额来自 R-049 第一层"
        "（OWNER_RULES_OOS.md，MU、扣费前、相对随机时刻），只有 MU 的 5 分钟与 30 分钟有数。",
        "",
        "**QQQ / SPY 没有任何已测超额，判定一律为「无数据」，不可外推；MU 的 15 分钟同样无已测超额。**",
        "",
        bound, "", df_to_md(pd.DataFrame(rows)), "",
    ]
    # 3.1
    rows = []
    for und, d in data.items():
        for dte in (0, 1):
            for h in HS:
                dd = _agg(d, "thr_diff_pct", dte=dte, m=0, h=h, q=1)
                rows.append({"标的": und, "DTE": dte, "h": h, "n": dd["n"],
                             "门槛差（期末价差 − 入场价差）中位 / 均值": "n<30" if dd["n"] < MIN_N else f"{dd['p50']:+.3f} / {dd['mean']:+.4f} 个百分点"})
    L += ["## 3.1 价差时点敏感性（平值、1 张）", "", bound, "", df_to_md(pd.DataFrame(rows)), ""]
    # 3.2
    rows = []
    for und, d in data.items():
        for s_k, sn in enumerate(SESS_NAMES):
            iv, rv, dd = (_agg(d, k, dte=0, sess=s_k) for k in ("iv", "rv", "iv_rv"))
            rows.append({"标的": und, "时段": sn, "n": dd["n"], "入场 IV 均值": _f(iv["mean"] * 100, 1, iv["n"]) + "%",
                         "随后 30 分钟 RV 均值": _f(rv["mean"] * 100, 1, rv["n"]) + "%",
                         "IV − RV 均值 / 中位": "n<30" if dd["n"] < MIN_N else f"{dd['mean'] * 100:+.1f} / {dd['p50'] * 100:+.1f} 个百分点"})
    L += [
        "## 3.2 0DTE 平值：入场 IV − 随后 30 分钟实现波动",
        "",
        "IV = 平值看涨 / 看跌中间价反推 IV 的平均；RV = 随后 30 个 1 分钟对数收益的标准差 × √(365 × 24 × 60)（与 IV 的日历时间口径一致）。",
        "",
        bound, "", df_to_md(pd.DataFrame(rows)), "",
    ]
    # 3.3
    rows = []
    for und, d in data.items():
        for h in HS:
            f = {"dte": 0, "m": 0, "sess": 4, "h": h}
            hu, th = _agg(d, "hold_usd", **f), _agg(d, "theta_usd", **f)
            tc = _agg(d, "thr_pct", **f, right=0, q=1)
            tp = _agg(d, "thr_pct", **f, right=1, q=1)
            rows.append({"标的": und, "h": h, "n": hu["n"],
                         "单腿持有成本 均值，美元": _f(hu["mean"], 2, hu["n"]),
                         "理论 theta 均值，美元": _f(th["mean"], 2, th["n"]),
                         "门槛 看涨 / 看跌 中位": "n<30" if min(tc["n"], tp["n"]) < MIN_N else f"{tc['p50']:.3f}% / {tp['p50']:.3f}%"})
    L += ["## 3.3 15:30–15:50 单列（0DTE 平值、1 张）", "", bound, "", df_to_md(pd.DataFrame(rows)), ""]
    # 计数与失败日
    rows = []
    for und, d in data.items():
        c = {k: int(d[(d["metric"] == f"n_{k}") & (d["bin"] == -1)]["sum"].sum()) for k in
             ("missing", "present", "bid0", "invalid", "lowmid", "valid", "iv_fail", "delta_nan", "hold_dropped")}
        rows.append({"标的": und, **c})
    L += ["## 4 剔除与失败计数", "", "missing = 该分钟该合约无报价行（含档位超出已拉取行权价带、正股无价）；present = 有报价行；"
          "lowmid = mid < 0.05 剔除；iv_fail = IV 反解失败；delta_nan = 门槛不可算；hold_dropped = 持有期内任一端无效被剔除。", "",
          df_to_md(pd.DataFrame(rows)), ""]
    fl = []
    for und in UNDS:
        for p in sorted(out_dir(root, und).glob("date=*.FAILED.json")):
            info = json.loads(p.read_text())
            fl.append({"标的": und, "日期": p.name[5:15], "异常": info.get("type"), "信息": info.get("msg", "")[:120]})
    L += ["### 失败日", "", df_to_md(pd.DataFrame(fl)) if fl else "无。", ""]
    L += ["## 附录 2.4 回本门槛全表（全部时段合并）", "", bound, "", df_to_md(tbl24), ""]
    L += [
        "## 边界",
        "",
        "- 只描述成本，不测任何策略；已测超额只引用 R-049 第一层（MU）。QQQ / SPY 无已测超额，不可外推。",
        "- 1 分钟报价是快照，0DTE 约 3% 误差，价差估计偏粗；本报告只用于量级判断，不用于结算级对账。",
        "- 正股价 = 上一根长桥 1 分钟 K 收盘（与期权快照时刻有最多 1 分钟错位）；IV / theta / delta 用 BS（r = 0.04、q = 0），美式提前行权未建模。",
        "- 单腿持有成本 = 跨式变化 ÷ 2，是期望意义上的近似；门槛按「往返成本 − 单腿持有成本（负值即损耗）」计算。",
        "- MU 0DTE 只在有当日到期合约的日子存在（多为周五）；SPY 只有已拉取的 2023-10～2024-07。",
        "- 价差 tick 用 zr.execution.ticks 的默认 Penny 名单（MU 视为 Penny：< 3 美元 0.01、≥ 3 美元 0.05），名单未核对 OCC（D-015）。",
        "",
        "## 题设口径歧义（按下述理解执行，待负责人确认）",
        "",
        "1. 门槛公式符号：题设 threshold = (cost_rt + 单腿持有成本) / …，而单腿持有成本 = change / 2 在时间损耗下为负，"
        "直接相加会让持有「降低」门槛。本报告按「持有损耗增加门槛」处理：分子 = cost_rt − change / 2（损耗为负，减去即加上其绝对值）。",
        "2. 测试 2 的数字：delta 0.5、S 100、成本 1.00 → 2.00% 只在成本按每股（1.00 美元 / 股 = 100 美元 / 张）时成立；"
        "按公式里的 ×100（每张口径）且成本 1.00 美元 / 张则为 0.02%。代码按每张美元成本 ÷ (|delta| × 100 × S)，测试两种口径都验证。",
        "3. 虚值档位的持有成本：跨式取同一行权价（看涨虚值 1 档的跨式 = 该行权价的看涨 + 看跌）；2.3 主表只列平值。",
        "4. 门槛里的往返成本为每张（cost_rt ÷ 张数），张数只通过费用的最低收费影响每张成本。",
        "",
    ]
    txt = "\n".join(L)
    OUT_MD.write_text(txt, encoding="utf-8")
    (out_dir(root) / "COST_REPORT.md").write_text(txt, encoding="utf-8")
    return txt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-und", default=None)
    ap.add_argument("--from", dest="lo", default=None)
    ap.add_argument("--to", dest="hi", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _td.ThetaClient.get_json = _no_network  # 禁止联网
    root = get_settings().zr_data_root
    if args.report:
        report(root)
        print(f"写入 {OUT_MD}")
        return 0
    unds = [args.only_und] if args.only_und else list(UNDS)
    lo = dt.date.fromisoformat(args.lo) if args.lo else None
    hi = dt.date.fromisoformat(args.hi) if args.hi else None
    st = run(root, unds, lo, hi, args.force, args.dry_run)
    print(json.dumps(st, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
