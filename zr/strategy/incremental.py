"""V4 规则的增量（实时）计算路径：逐笔成交 → K 线 → 指标 / 状态 / 信号 / 离场，逐根更新。

与批量回测（zr/features/regime.py + zr/research/v4_decisive.py，整列向量化）逻辑相同、实现独立：
每来一根 K 只用此前状态和这根 K 更新，不回看、不重算历史。供影子回放（zr/research/shadow_replay.py）
与后续实盘信号层使用。只做信号与模拟成交，不下单
（下单只能经 zr/execution 的 OMS，CLAUDE.md 规则 6）。

口径 = V4 主口径 C：含盘前盘后连续 2 分钟 K、不按日重置；BOLL(20,2) ddof=0；MACD(6,13,6) 标准 EMA；
状态（点火 / 通道 / RANGE）与 H1（E10）、H2（E1）、H3（E11 + 下一根确认）、结构离场
X3_mid2 / X2_shrink2 / X4_inside_0.1_0.9 / X5_higher_low、护栏全部按 config/v4_decisive.yaml。
RANGE 打分窗口 range_w 由调用方给定（批量路径按数据实际 bar/日折算，口径 C 2 分钟 K = 480 根）。
"""

from __future__ import annotations

import datetime as dt
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.features.regime import CHANNEL, IGNITION, OTHER, RANGE, SLOPE_EPS

NS_MIN = 60_000_000_000


@dataclass
class Bar:
    ts: int  # 分箱起点（UTC 纳秒）
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------- 逐笔 → K 线 ----------------


class BarBuilder:
    """按成交时间戳分箱（[t, t+P)，UTC 整分钟网格）。分箱在收到时间戳 ≥ 下一分箱起点的成交、或
    外部时钟越过分箱终点 + grace 时收线；之后再到的属于已收线分箱的成交计为迟到、丢弃。
    price_ok(trade_type)：该成交是否参与开高低收（量全部计入）。整箱无成交 → 占位 K（前收、量 0）。
    """

    def __init__(self, period_min: int, price_ok=lambda tt: True, grace_s: float = 1.0):
        self.p = period_min * NS_MIN
        self.price_ok = price_ok
        self.grace = int(grace_s * 1e9)
        self.cur: int | None = None
        self.o = self.h = self.lo = self.c = math.nan
        self.v = 0.0
        self.last_close = math.nan
        self.late = 0

    def _emit(self) -> Bar:
        if math.isnan(self.o):  # 有量但没有计价成交，或空箱
            px = self.last_close
            b = Bar(self.cur, px, px, px, px, self.v)
        else:
            b = Bar(self.cur, self.o, self.h, self.lo, self.c, self.v)
            self.last_close = self.c
        self.o = self.h = self.lo = self.c = math.nan
        self.v = 0.0
        return b

    def _advance(self, to_bin: int) -> list[Bar]:
        out = []
        while self.cur is not None and self.cur < to_bin:
            out.append(self._emit())
            self.cur += self.p
        return out

    def on_trade(self, t_ns: int, price: float, volume: float, trade_type: str) -> list[Bar]:
        b = t_ns - t_ns % self.p
        if self.cur is None:
            self.cur = b
        if b < self.cur:
            self.late += 1
            return []
        out = self._advance(b)
        self.v += volume
        if self.price_ok(trade_type):
            if math.isnan(self.o):
                self.o = self.h = self.lo = price
            self.h, self.lo, self.c = max(self.h, price), min(self.lo, price), price
        return out

    def close(self) -> list[Bar]:
        """数据结束：收当前分箱。"""
        if self.cur is None:
            return []
        out = [self._emit()]
        self.cur += self.p
        return out

    def on_clock(self, now_ns: int) -> list[Bar]:
        """外部时钟：当前分箱终点 + grace 已过 → 收线（之后空箱也逐个补占位 K）。"""
        if self.cur is None:
            return []
        to = now_ns - self.grace
        return self._advance(to - to % self.p) if to >= self.cur + self.p else []


# ---------------- 逐根指标 / 状态 / 信号 ----------------


def _pct_rank(win: deque) -> float:
    """窗口内最后一个值的百分位秩（平均名次 / 有效个数），
    与 pandas rolling.rank(pct=True) 同口径。"""
    v = np.fromiter((x for x in win if not math.isnan(x)), float)
    x = win[-1]
    if math.isnan(x):
        return math.nan
    return float((np.sum(v < x) + (np.sum(v == x) + 1) / 2) / len(v))


@dataclass
class _Run:
    n: int = 0

    def step(self, cond: bool) -> int:
        self.n = self.n + 1 if cond else 0
        return self.n


@dataclass
class EngineState:
    closes: deque = field(default_factory=lambda: deque(maxlen=20))
    ema_f: float = math.nan
    ema_s: float = math.nan
    dea: float = math.nan
    prev: dict | None = None  # 上一根的记录
    bw_hist: deque = field(default_factory=lambda: deque(maxlen=6))  # bw[t-5..t]
    above: deque = field(default_factory=lambda: deque(maxlen=10))
    below: deque = field(default_factory=lambda: deque(maxlen=10))
    highs: deque = field(default_factory=lambda: deque(maxlen=20))  # 前 M 根（不含当前）
    lows: deque = field(default_factory=lambda: deque(maxlen=20))
    vols: deque = field(default_factory=lambda: deque(maxlen=20))
    sw_low: deque = field(default_factory=lambda: deque(maxlen=5))
    sw_high: deque = field(default_factory=lambda: deque(maxlen=5))
    swing: dict = field(
        default_factory=lambda: {True: [math.nan] * 3, False: [math.nan] * 3}
    )  # [前一个摆动点, 最近一个, 当前有效水位]
    e11_pending: tuple | None = None  # (方向, 突破水位, 日期)


class IncrementalEngine:
    def __init__(
        self, cfg: dict, v4: dict, range_w: int, days: set, daily_reset: bool = False
    ):
        """daily_reset=False：口径 C（连续、不重置）；True：口径 A（调用方只喂 RTH K，
        BOLL / MACD 与中轨斜率每个交易日从头算，其余状态量跨日连续，
        与批量 `indicators(daily_reset=True)` 一致）。"""
        self.cfg, self.v4, self.days = cfg, v4, days
        self.daily_reset = daily_reset
        bo, m = cfg["boll"], cfg["macd"]
        self.n, self.k = bo["n"], bo["k"]
        self.af, self.as_, self.ag = (2 / (m[x] + 1) for x in ("fast", "slow", "signal"))
        rc = cfg["regime"]
        self.ig, self.ch, self.rg = rc["ignition"], rc["channel"], rc["range"]
        self.cn, self.cm = self.ch["default"]
        self.w = range_w
        self.mp = max(20, range_w // 4)
        self.rw = {k: deque(maxlen=range_w) for k in ("bw", "slope", "hist")}
        self.s = EngineState()
        self.s.above = deque(maxlen=self.cm)
        self.s.below = deque(maxlen=self.cm)
        self.s.bw_hist = deque(maxlen=self.cn + 1)
        h = v4["hypotheses"]
        self.e10_k = h["H1"]["k"]
        self.h2 = h["H2"]
        self.h3 = h["H3"]
        self.s.highs = deque(maxlen=self.h3["M"])
        self.s.lows = deque(maxlen=self.h3["M"])
        self.s.vols = deque(maxlen=self.h3["vol_window"])
        self.runs = {
            k: _Run()
            for k in (
                "exp",
                "hi",
                "lo",
                "hup",
                "hdn",
                "sup",
                "sdn",
                "out_up",
                "out_dn",
                "fall",
                "rise",
            )
        }
        self.ign = 0

    def on_bar(self, b: Bar) -> dict:
        s, r = self.s, self.runs
        et = pd.Timestamp(b.ts, tz="UTC").tz_convert(ET)
        day = et.date()
        prev = s.prev
        same_prev = prev is not None and prev["day"] == day
        if self.daily_reset and not same_prev:
            s.closes.clear()
            s.ema_f = s.ema_s = s.dea = math.nan
        # BOLL
        s.closes.append(b.close)
        if len(s.closes) == self.n:
            arr = np.fromiter(s.closes, float)
            mid = float(arr.mean())
            sd = float(arr.std(ddof=0))
            up, lo_ = mid + self.k * sd, mid - self.k * sd
            band = up - lo_
            bw = band / mid
            pctb = (b.close - lo_) / band if band != 0 else math.nan
        else:
            mid = up = lo_ = bw = pctb = math.nan
        # MACD（首值做种）
        if math.isnan(s.ema_f):
            s.ema_f = s.ema_s = b.close
        else:
            s.ema_f = self.af * b.close + (1 - self.af) * s.ema_f
            s.ema_s = self.as_ * b.close + (1 - self.as_) * s.ema_s
        dif = s.ema_f - s.ema_s
        s.dea = dif if math.isnan(s.dea) else self.ag * dif + (1 - self.ag) * s.dea
        hist = dif - s.dea
        pm = prev["mid"] if prev and (same_prev or not self.daily_reset) else math.nan
        slope = (mid - pm) / mid if not (math.isnan(mid) or math.isnan(pm)) else math.nan
        if not math.isnan(slope) and abs(slope) < SLOPE_EPS:
            slope = 0.0
        ph = prev["hist"] if prev else math.nan
        pbw = prev["bw"] if prev else math.nan
        # 点火
        ig = self.ig
        exp = r["exp"].step(bw > pbw)
        hi = r["hi"].step(pctb > ig["pctb_hi"])
        lo = r["lo"].step(pctb < ig["pctb_lo"])
        hup = r["hup"].step(hist > ph and hist > 0)
        hdn = r["hdn"].step(hist < ph and hist < 0)
        kk = ig["outer_bars"]
        start_up = exp >= ig["expand_bars"] and hi >= kk and hup >= kk
        start_dn = exp >= ig["expand_bars"] and lo >= kk and hdn >= kk
        if start_up or (self.ign == 1 and pctb > ig["pctb_hi"]):
            self.ign = 1
        elif start_dn or (self.ign == -1 and pctb < ig["pctb_lo"]):
            self.ign = -1
        else:
            self.ign = 0
        # 通道（默认 N, M）
        sup = r["sup"].step(slope > 0)
        sdn = r["sdn"].step(slope < 0)
        s.above.append(1.0 if b.close > mid else 0.0)
        s.below.append(1.0 if b.close < mid else 0.0)
        s.bw_hist.append(bw)
        fa = sum(s.above) / self.cm if len(s.above) == self.cm else math.nan
        fb = sum(s.below) / self.cm if len(s.below) == self.cm else math.nan
        bw_n = s.bw_hist[0] if len(s.bw_hist) == self.cn + 1 else math.nan
        keep = bw >= self.ch["bw_shrink_tol"] * bw_n
        chan = 0
        if sup >= self.cn and fa >= self.ch["side_frac"] and keep:
            chan = 1
        elif sdn >= self.cn and fb >= self.ch["side_frac"] and keep:
            chan = -1
        # RANGE 打分
        self.rw["bw"].append(bw)
        self.rw["slope"].append(abs(slope))
        self.rw["hist"].append(abs(hist / b.close))
        wt = self.rg["weights"]
        parts = []
        for key in ("bw", "slope", "hist"):
            win = self.rw[key]
            nv = sum(1 for x in win if not math.isnan(x))
            parts.append(_pct_rank(win) if nv >= self.mp else math.nan)
        score = wt["bw"] * parts[0] + wt["slope"] * parts[1] + wt["hist"] * parts[2]
        state, sdir = OTHER, 0
        if score <= self.rg["score_threshold"]:
            state = RANGE
        if chan:
            state, sdir = CHANNEL, chan
        if self.ign:
            state, sdir = IGNITION, self.ign
        # 信号
        sig = {"H1": 0, "H2": 0, "H3": 0}
        out_up = r["out_up"].step(pctb > 1)
        out_dn = r["out_dn"].step(pctb < 0)
        if state == IGNITION and sdir == 1 and out_up == self.e10_k:
            sig["H1"] = 1
        elif state == IGNITION and sdir == -1 and out_dn == self.e10_k:
            sig["H1"] = -1
        if prev is not None and same_prev and not math.isnan(lo_):
            hw = (up - lo_) / 2
            a, bb = self.h2["pctb"]
            c = self.h2["c"]
            low2, high2 = min(prev["low"], b.low), max(prev["high"], b.high)
            pl, ph_ = (low2 - lo_) / (2 * hw), (high2 - lo_) / (2 * hw)
            if (
                chan == 1
                and prev["close"] < prev["open"]
                and b.close > b.open
                and a <= pl <= bb
                and abs(low2 - mid) <= c * hw
            ):
                sig["H2"] = 1
            elif (
                chan == -1
                and prev["close"] > prev["open"]
                and b.close < b.open
                and a <= 1 - ph_ <= bb
                and abs(high2 - mid) <= c * hw
            ):
                sig["H2"] = -1
        # H3：上一根的 E11 突破在这根确认
        if s.e11_pending is not None:
            d0, lvl, dd = s.e11_pending
            if dd == day and d0 * (b.close - lvl) > 0:
                sig["H3"] = d0
            s.e11_pending = None
        full = len(s.highs) == self.h3["M"]
        hh = max(s.highs) if full else math.nan
        ll = min(s.lows) if full else math.nan
        vr = (
            b.volume / (sum(s.vols) / len(s.vols))
            if len(s.vols) == self.h3["vol_window"] and sum(s.vols) > 0
            else (math.inf if len(s.vols) == self.h3["vol_window"] and b.volume > 0 else math.nan)
        )
        prev_range = same_prev and prev["state"] == RANGE
        if prev_range and b.close > hh and vr >= self.h3["v"]:
            s.e11_pending = (1, hh, day)
        elif prev_range and b.close < ll and vr >= self.h3["v"]:
            s.e11_pending = (-1, ll, day)
        s.highs.append(b.high)
        s.lows.append(b.low)
        s.vols.append(b.volume)
        # 结构离场（多头, 空头）
        fall = r["fall"].step(hist < ph)
        rise = r["rise"].step(hist > ph)
        pcl = prev["close"] if prev else math.nan
        ex = {
            "X3_mid2": (
                b.close < mid and same_prev and pcl < pm,
                b.close > mid and same_prev and pcl > pm,
            ),
            "X2_shrink2": (fall >= 2, rise >= 2),
            "X4_inside_0.1_0.9": (pctb < 0.9, pctb > 0.1),
        }
        lv_l, lv_s = self._swing(b, day, same_prev)
        ex["X5_higher_low"] = (b.close < lv_l, b.close > lv_s)
        rec = {
            "ts": b.ts,
            "day": day,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "mid": mid,
            "upper": up,
            "lower": lo_,
            "bw": bw,
            "pctb": pctb,
            "slope": slope,
            "dif": dif,
            "dea": s.dea,
            "hist": hist,
            "state": state,
            "dir": sdir,
            "range_score": score,
            **sig,
            "ex": ex,
        }
        rec["ok_long"], rec["ok_short"] = self._guards(et, state, sdir)
        s.prev = rec
        return rec

    def _swing(self, b: Bar, day: dt.date, same_prev: bool) -> tuple[float, float]:
        s = self.s
        if not same_prev:
            s.sw_low.clear()
            s.sw_high.clear()
            s.swing = {True: [math.nan] * 3, False: [math.nan] * 3}
        s.sw_low.append(b.low)
        s.sw_high.append(b.high)
        for long, win in ((True, s.sw_low), (False, s.sw_high)):
            if len(win) == 5:
                x = win[2]
                st = s.swing[long]
                if (long and x == min(win)) or (not long and x == max(win)):
                    st[0], st[1] = st[1], x
                    if not math.isnan(st[0]) and (
                        (long and st[1] > st[0]) or (not long and st[1] < st[0])
                    ):
                        st[2] = st[1]
        return s.swing[True][2], s.swing[False][2]

    def _guards(self, et: pd.Timestamp, state: int, sdir: int) -> tuple[bool, bool]:
        from zr.research.v4_decisive import day_limits

        exe, g = self.v4["execution"], self.v4["guardrails"]
        d = et.date()
        if d not in self.days:
            return False, False
        last = day_limits(d, exe)[0]
        close_t = (et + pd.Timedelta(minutes=self.v4["period"])).time()
        ok = et.strftime("%H:%M") >= exe["first_entry_bar"] and close_t <= last
        if g["no_range_entry"] and state == RANGE:
            ok = False
        ok_l = ok and not (g["no_counter_ignition"] and state == IGNITION and sdir == -1)
        ok_s = ok and not (g["no_counter_ignition"] and state == IGNITION and sdir == 1)
        return ok_l, ok_s
