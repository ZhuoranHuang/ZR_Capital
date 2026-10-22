"""v3 真实期权回放引擎（METHOD_V3 第 0、3、4、6 节）。

成交口径（修复 docs/research/QQQ_ARTIFACT.md 的前视）：
- 信号在周期 P 的 K 线收盘确认（收盘分钟 c）；入场 = 决策时刻 + 2 秒后的第一个 1 分钟快照 = c + 1，
  买 ask。
- 结构离场（X1–X5）在 bar 收盘 c' 触发 → 触发后第一个快照 c' + 1 卖 bid。
- X6：快照 s 的 mid ≥ 入场 ask × (1 + X) → s + 1 卖 1 张；mid ≤ 入场 ask × (1 − Y) → s + 1
  全部卖出。
- X7：入场快照 + cap 分钟卖出；15:50 清仓；D-001：10:15 后不开 0DTE，0DTE 10:45 前清仓。
- 报价质量：ask > 0、bid ≥ 0、ask ≥ bid；bid = 0 视为可成交价 0（不回看更早的 bid）；
  成交分钟无快照时
  用之后第一个有效快照；入场快照与前 3 个快照完全相同视为陈旧，不入场。
- 合约：最近可用到期日（已拉取 DTE ≤ 5 全部 + 最近两个）、略虚值一档（相对信号 bar 收盘价），2 张，
  D-013 费用。
- 同一标的同时一笔：按天贪心（全部当日清仓，所以每天独立），对每个出场组合分别执行。

出场组合 = 一条结构规则（或 none）× X6(止盈 X, 止损 Y) × X7(cap)，任一触发即出。
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zr.costs.fees import option_order_fees
from zr.features.bars import ET, resample_bars
from zr.features.option_features import load_option_day
from zr.features.regime import indicators, regimes
from zr.research import exits as ex
from zr.research import patterns_v3 as pt

log = logging.getLogger("zr.research.sim_v3")
MINUTES = 391
HOLD_BUCKETS = (1, 3, 5, 10, 30)  # 0–1 / 1–3 / 3–5 / 5–10 / 10–30 / >30
BUCKET_NAMES = ("0-1", "1-3", "3-5", "5-10", "10-30", ">30")


def hm(s: str) -> int:
    h, m = map(int, s.split(":"))
    return h * 60 + m - 570


# ---------------- 费用（D-013，向量化；tests 对照 option_order_fees 逐值相等） ----------------

_BUY = {n: float(option_order_fees(n, "Buy", 1.0)["total"]) for n in (1, 2)}
_SELL_BASE = {
    n: float(option_order_fees(n, "Sell", 0.0)["total"])
    - max(0.01, 0.0)  # 权利金 0 时 SEC 取最低 0.01，下面加回按价计算的 SEC
    for n in (1, 2)
}


def sell_fee(n: np.ndarray, prem: np.ndarray) -> np.ndarray:
    n = np.asarray(n, float)
    v = 0.00206 * np.asarray(prem, float) * n
    sec = np.maximum(np.floor(v * 100 + 0.5 + 1e-9) / 100, 0.01)
    base = np.where(n >= 2, _SELL_BASE[2], _SELL_BASE[1])
    return np.where(n > 0, base + sec, 0.0)


# ---------------- 期权日数据 ----------------


@dataclass
class OptDay:
    day: dt.date
    expiries: list
    chains: dict  # (expiry, right) -> (strikes ndarray, symbols list)
    bid: dict
    ask: dict
    mid: dict
    stale: dict


def minute_of(ts) -> np.ndarray:
    et = pd.DatetimeIndex(ts).tz_convert(ET)
    return (et.hour * 60 + et.minute - 570).to_numpy()


def load_day(root: Path, und: str, day: dt.date, stale_n: int = 3) -> OptDay | None:
    opt = load_option_day(root, und, day)
    if opt is None or opt["quotes"].empty:
        return None
    q = opt["quotes"]
    q = q[(q["ask"] > 0) & (q["bid"] >= 0) & (q["ask"] >= q["bid"])]
    m = minute_of(q["ts"])
    keep = (m >= 0) & (m < MINUTES)
    q, m = q[keep], m[keep]
    bid, ask, mid, stale = {}, {}, {}, {}
    sym = q["symbol"].to_numpy()
    order = np.argsort(sym, kind="stable")
    sym_s, m_s = sym[order], m[order]
    b_s = q["bid"].to_numpy(float)[order]
    a_s = q["ask"].to_numpy(float)[order]
    has_size = "bid_size" in q
    bz = q["bid_size"].to_numpy(float)[order] if has_size else np.zeros(len(q))
    az = q["ask_size"].to_numpy(float)[order] if has_size else np.zeros(len(q))
    cuts = np.flatnonzero(np.r_[True, sym_s[1:] != sym_s[:-1], True])
    for k in range(len(cuts) - 1):
        lo, hi = cuts[k], cuts[k + 1]
        s = sym_s[lo]
        bb, aa = np.full(MINUTES, np.nan), np.full(MINUTES, np.nan)
        mm = m_s[lo:hi]
        bb[mm], aa[mm] = b_s[lo:hi], a_s[lo:hi]
        key = np.stack([b_s[lo:hi], a_s[lo:hi], bz[lo:hi], az[lo:hi]], axis=1)
        same = np.r_[False, (key[1:] == key[:-1]).all(axis=1)]
        run = np.zeros(len(same), dtype=int)
        for i in range(1, len(same)):
            run[i] = run[i - 1] + 1 if same[i] else 0
        st = np.zeros(MINUTES, dtype=bool)
        st[mm] = run >= stale_n
        bid[s], ask[s], mid[s], stale[s] = bb, aa, (bb + aa) / 2, st
    con = q[["symbol", "expiry", "strike", "right"]].drop_duplicates("symbol")
    expiries = sorted(e for e in con["expiry"].unique() if e >= day)
    chains = {}
    for (e, r), g in con.groupby(["expiry", "right"]):
        g = g.sort_values("strike")
        chains[(e, r)] = (g["strike"].to_numpy(float), g["symbol"].tolist())
    return OptDay(day, expiries, chains, bid, ask, mid, stale)


def pick(od: OptDay, direction: int, spot: float, e: int, d001_entry: int):
    """最近可用到期日上的略虚值一档；D-001：入场晚于 10:15 跳过 0DTE。返回 (symbol, expiry) 或
    None。
    """
    right = "call" if direction == 1 else "put"
    for exp in od.expiries:
        if exp == od.day and e > d001_entry:
            continue
        ch = od.chains.get((exp, right))
        if ch is None:
            continue
        s, syms = ch
        j = (
            np.searchsorted(s, spot, side="right")
            if direction == 1
            else (np.searchsorted(s, spot, side="left") - 1)
        )
        if j < 0 or j >= len(s):
            continue
        sym = syms[j]
        a = od.ask[sym][e]
        if np.isfinite(a) and a > 0 and not od.stale[sym][e]:
            return sym, exp
    return None


def _fwd_index(valid: np.ndarray) -> np.ndarray:
    """nxt[i] = ≥ i 的第一个有效分钟；没有则 MINUTES（越界）。"""
    nxt = np.full(MINUTES + 2, MINUTES, dtype=int)
    for i in range(MINUTES - 1, -1, -1):
        nxt[i] = i if valid[i] else nxt[i + 1]
    return nxt


# ---------------- 周期 P 的信号准备 ----------------


@dataclass
class PeriodData:
    period: int
    bars: pd.DataFrame
    close_min: np.ndarray  # bar 收盘分钟
    day: np.ndarray
    entries: dict  # name -> int8 signal（bar 级）
    exits: dict  # rule -> (long bool, short bool)
    rules: list


def shift_signal_within_day(sig: np.ndarray, day: np.ndarray, k: int) -> np.ndarray:
    """把信号数组在同一交易日内后移 k 根（跨日边界的丢弃，不回绕到下一天/上一天）。

    用于前视审计的「延迟测试」（fix/lookahead-audit 第 1 节）：若延迟 k 根后结果反而显著变好，
    说明原版对齐可能有前视或执行时点问题。k=0 原样返回。
    """
    if k == 0:
        return sig
    out = np.zeros_like(sig)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        if len(idx) > k:
            out[idx[k:]] = sig[idx[:-k]]
    return out


def prepare_period(
    stock: pd.DataFrame,
    period: int,
    cfg: dict,
    macd_key: str = "macd",
    extra_entries: dict | None = None,
    delay_bars: int = 0,
) -> PeriodData:
    bars = stock if period == 1 else resample_bars(stock, f"{period}min")
    x = indicators(bars, cfg, macd_key)
    reg = regimes(bars, x, cfg, period)
    entries = {}
    for c in pt.entry_configs(cfg):
        entries[pt.entry_name(c)] = pt.entry_signal(bars, x, reg, c, cfg)
    if extra_entries:
        for k, fn in extra_entries.items():
            entries[k] = fn(bars, x, reg)
    day = np.asarray(bars.index.tz_convert(ET).date)
    if delay_bars:
        entries = {k: shift_signal_within_day(v, day, delay_bars) for k, v in entries.items()}
    exs = ex.exit_signals(bars, x, cfg)
    rules = ex.exit_rules(cfg, "E10")  # 全集（含 X4）；非 E10 入场在汇总时去掉 X4
    et = bars.index.tz_convert(ET)
    close_min = (et.hour * 60 + et.minute - 570).to_numpy() + period
    return PeriodData(period, bars, close_min, day, entries, exs, rules)


# ---------------- 单日事件评估 ----------------


def evaluate(od: OptDay, ev: pd.DataFrame, trig: dict, cfg: dict) -> dict | None:
    """ev：当日唯一事件（e 入场分钟、direction、spot）。trig[rule][dir] = 当日该规则触发的 bar
    收盘分钟（升序）。

    返回 net[n_ev, R, X, Y, C]、hold[...]、以及事件元数据。
    """
    exe, x6, x7 = cfg["execution"], cfg["exits"]["X6"], cfg["exits"]["X7"]
    tps, sls = np.array(x6["tp"]), np.array(x6["sl"])
    caps = np.array(x7["cap_min"])
    rules = list(trig.keys())
    flat, d001e, d001f = hm(exe["flat"]), hm(exe["d001_last_0dte_entry"]), hm(exe["d001_flat_0dte"])
    qty = exe["qty"]
    rows, meta = [], []
    for e, d, spot in zip(
        ev["e"].to_numpy(), ev["direction"].to_numpy(), ev["spot"].to_numpy(), strict=True
    ):
        pk = pick(od, int(d), float(spot), int(e), d001e)
        if pk is None:
            continue
        sym, exp = pk
        end = d001f if exp == od.day else flat
        if e >= end:
            continue
        ask0 = od.ask[sym][e]
        bid, mid = od.bid[sym], od.mid[sym]
        nxt = _fwd_index(np.isfinite(bid))
        path = mid[e + 1 : end + 1]
        with np.errstate(invalid="ignore"):
            hit_tp = path[None, :] >= ask0 * (1 + tps)[:, None]
            hit_sl = path[None, :] <= ask0 * (1 - sls)[:, None]
        big = 10_000
        tp_fill = np.where(hit_tp.any(1), e + 1 + hit_tp.argmax(1) + 1, big)
        sl_fill = np.where(hit_sl.any(1), e + 1 + hit_sl.argmax(1) + 1, big)
        st_fill = np.full(len(rules), big)
        for i, r in enumerate(rules):
            t = trig[r][int(d)]
            j = np.searchsorted(t, e, side="left")
            if j < len(t):
                st_fill[i] = t[j] + 1
        cap_fill = e + caps
        shape = (len(rules), len(tps), len(sls), len(caps))
        F = np.minimum(
            np.minimum(st_fill[:, None, None, None], sl_fill[None, None, :, None]),
            np.minimum(cap_fill[None, None, None, :], end),
        )
        F = np.broadcast_to(F, shape)
        tpf = np.broadcast_to(tp_fill[None, :, None, None], F.shape)
        used = tpf < F
        Fi = np.minimum(F, MINUTES - 1)
        bi = nxt[Fi]
        bF = np.where(bi < MINUTES, bid[np.minimum(bi, MINUTES - 1)], 0.0)
        ti = nxt[np.minimum(tpf, MINUTES - 1)]
        bT = np.where(ti < MINUTES, bid[np.minimum(ti, MINUTES - 1)], 0.0)
        rem = qty - used.astype(int)
        cash = (
            -ask0 * 100 * qty
            - _BUY[qty]
            + np.where(used, bT * 100 - np.where(bT > 0, sell_fee(1, bT), 0.0), 0.0)
            + rem * bF * 100
            - np.where(bF > 0, sell_fee(rem, bF), 0.0)
        )
        rows.append((cash.astype(np.float32), (F - e).astype(np.int16)))
        meta.append((int(e), int(d), sym, (exp - od.day).days, float(ask0)))
    if not rows:
        return None
    return {
        "net": np.stack([r[0] for r in rows]),
        "hold": np.stack([r[1] for r in rows]),
        "meta": pd.DataFrame(meta, columns=["e", "direction", "symbol", "dte", "ask0"]),
        "rules": rules,
    }


def day_triggers(pdat: PeriodData, mask: np.ndarray) -> dict:
    cm = pdat.close_min[mask]
    out = {}
    for r in pdat.rules:
        lg, sh = pdat.exits[r]
        out[r] = {1: cm[lg[mask]], -1: cm[sh[mask]]}
    return out


def day_events(pdat: PeriodData, mask: np.ndarray, last_entry: int) -> tuple[pd.DataFrame, dict]:
    """当日全部入场配置的事件；返回唯一事件表与 {配置: [(事件行号, 方向)]}。"""
    cm = pdat.close_min[mask]
    close = pdat.bars["close"].to_numpy()[mask]
    uniq: dict[tuple[int, int], int] = {}
    rows, by_cfg = [], {}
    for name, sig in pdat.entries.items():
        s = sig[mask]
        idx = np.flatnonzero(s != 0)
        lst = []
        for i in idx:
            e = int(cm[i]) + 1
            if e < 1 or e > last_entry:
                continue
            key = (e, int(s[i]))
            if key not in uniq:
                uniq[key] = len(rows)
                rows.append({"e": e, "direction": int(s[i]), "spot": float(close[i])})
            lst.append(uniq[key])
        if lst:
            by_cfg[name] = lst
    return pd.DataFrame(rows), by_cfg


def greedy(e: np.ndarray, hold: np.ndarray) -> np.ndarray:
    """按入场时间排序的事件 × 组合：返回布尔 take[n_ev, ...]（上一笔出场前的事件跳过）。"""
    n = len(e)
    shape = hold.shape[1:]
    take = np.zeros((n,) + shape, dtype=bool)
    free = np.full(shape, -1)
    for i in range(n):
        t = e[i] >= free
        take[i] = t
        free = np.where(t, e[i] + hold[i], free)
    return take


# ---------------- 网格累计 ----------------


class Accum:
    """按 (配置, 方向) 累计每个出场组合的统计量（逐日更新，不保存逐笔）。"""

    def __init__(self, shape: tuple, oos_year: int):
        z = lambda: np.zeros(shape)  # noqa: E731
        self.n, self.s, self.s2, self.win, self.wsum, self.lsum = z(), z(), z(), z(), z(), z()
        self.n_oos, self.s_oos = z(), z()
        self.d1, self.d2, self.dn = z(), z(), z()
        self.bucket = np.zeros(shape + (len(BUCKET_NAMES),))
        self.oos_year = oos_year

    def add_day(self, net: np.ndarray, hold: np.ndarray, take: np.ndarray, year: int) -> None:
        v = np.where(take, net, 0.0)
        cnt = take.sum(0)
        self.n += cnt
        self.s += v.sum(0)
        self.s2 += (v**2).sum(0)
        self.win += (take & (net > 0)).sum(0)
        self.wsum += np.where(take & (net > 0), net, 0.0).sum(0)
        self.lsum += np.where(take & (net <= 0), net, 0.0).sum(0)
        if year == self.oos_year:
            self.n_oos += cnt
            self.s_oos += v.sum(0)
        ds = v.sum(0)
        has = cnt > 0
        self.d1 += np.where(has, ds, 0.0)
        self.d2 += np.where(has, ds**2, 0.0)
        self.dn += has
        b = np.digitize(hold, HOLD_BUCKETS, right=True)
        for k in range(len(BUCKET_NAMES)):
            self.bucket[..., k] += (take & (b == k)).sum(0)


def run_grid(
    root: Path,
    und: str,
    stock: pd.DataFrame,
    cfg: dict,
    periods: list[int],
    days: list[dt.date],
    macd_key: str = "macd",
    extra_entries: dict | None = None,
    collect: set | None = None,
    delay_bars: int = 0,
) -> dict:
    """返回 {period: {"names": [...], "accum": {name: Accum}, "rules": [...], "trades": {...}}}。

    collect：需要保留逐笔的 (period, 配置名) 集合（用于规则卡随机对照、提取规则回放）。
    delay_bars：入场信号延迟 k 根再用（前视审计的延迟测试，见 shift_signal_within_day）。
    """
    exe = cfg["execution"]
    last_entry = hm(exe["last_entry"])
    t0 = time.monotonic()
    pdats = {p: prepare_period(stock, p, cfg, macd_key, extra_entries, delay_bars) for p in periods}
    log.info("%s 信号准备完成 %.0fs", und, time.monotonic() - t0)
    x6, x7 = cfg["exits"]["X6"], cfg["exits"]["X7"]
    out = {}
    for p, pdat in pdats.items():
        shape = (2, len(pdat.rules), len(x6["tp"]), len(x6["sl"]), len(x7["cap_min"]))
        out[p] = {
            "names": list(pdat.entries),
            "rules": pdat.rules,
            "accum": {n: Accum(shape, cfg["cards"]["oos_year"]) for n in pdat.entries},
            "trades": {},
            "events": [],
        }
    day_set = set(days)
    for k, day in enumerate(days):
        od = load_day(root, und, day, exe["stale_snapshots"])
        if od is None:
            continue
        for p, pdat in pdats.items():
            mask = pdat.day == day
            if not mask.any():
                continue
            ev, by_cfg = day_events(pdat, mask, last_entry)
            if ev.empty:
                continue
            res = evaluate(od, ev, day_triggers(pdat, mask), cfg)
            if res is None:
                continue
            meta = res["meta"]
            pos = {
                (e, d): i for i, (e, d) in enumerate(zip(meta["e"], meta["direction"], strict=True))
            }
            out[p]["events"].append(meta.assign(day=day))
            for name, lst in by_cfg.items():
                acc = out[p]["accum"][name]
                for di, d in enumerate((1, -1)):
                    rows = sorted(
                        {
                            pos[(ev["e"].iat[j], d)]
                            for j in lst
                            if ev["direction"].iat[j] == d and (ev["e"].iat[j], d) in pos
                        }
                    )
                    if not rows:
                        continue
                    rows = sorted(rows, key=lambda r: meta["e"].iat[r])
                    net, hold = res["net"][rows], res["hold"][rows]
                    take = greedy(meta["e"].to_numpy()[rows], hold)
                    sub = _AccumView(acc, di)
                    sub.add_day(net, hold, take, day.year)
                    if collect and (p, name) in collect:
                        out[p]["trades"].setdefault(name, []).append(
                            {
                                "day": day,
                                "dir": d,
                                "e": meta["e"].to_numpy()[rows],
                                "net": net,
                                "hold": hold,
                                "take": take,
                                "symbol": meta["symbol"].to_numpy()[rows],
                            }
                        )
        if (k + 1) % 50 == 0:
            log.info("%s %d/%d 天 %.0fs", und, k + 1, len(days), time.monotonic() - t0)
    del day_set
    return out


class _AccumView:
    """Accum 在方向维 di 上的视图（add_day 写回原数组）。"""

    def __init__(self, acc: Accum, di: int):
        self.acc, self.di = acc, di

    def add_day(self, net, hold, take, year):
        a, di = self.acc, self.di
        tmp = Accum(net.shape[1:], a.oos_year)
        tmp.add_day(net, hold, take, year)
        for f in (
            "n",
            "s",
            "s2",
            "win",
            "wsum",
            "lsum",
            "n_oos",
            "s_oos",
            "d1",
            "d2",
            "dn",
            "bucket",
        ):
            getattr(a, f)[di] += getattr(tmp, f)
