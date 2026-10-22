"""独立引擎复算（audit/known-answer 第 2 节）：用 backtrader 重新实现 E4 基线与 V4 H2 的信号/入场。

不复用 zr/research 里任何回测或状态层代码：均线、ATR、布林带、MACD 用 backtrader 自带指标，
KDJ、摆动点、梯形、状态层（IGNITION / CHANNEL / RANGE）、E1 回踩、护栏、合约选择都在这里重写。
只共享原始数据读取（1 分钟 K、期权链、tick 缓存）与费用公式（D-013）。

backtrader 成交时点（已用合成数据核实，见 tests/test_bt_independent.py）：
- cheat_on_open + broker.set_coo：next_open 里下的限价单（开盘价 ± 1 tick，必成交）在该根开盘成交；
  同时下的止损单在同一根内触发（开盘越过止损按开盘价）。
- broker.set_coc：next() 里下的市价单按该根收盘价成交（用于 15:50 清仓）。
- 固定滑点 0.01（slip_out=True）。
"""

from __future__ import annotations

import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=SyntaxWarning)
import backtrader as bt  # noqa: E402

from zr.costs.fees import stock_order_fees  # noqa: E402
from zr.features.bars import ET  # noqa: E402

TICK = 0.01
FLAT = 15 * 60 + 50


# ---------------- 公共 ----------------


class StockFees(bt.CommInfoBase):
    params = (("stocklike", True), ("commtype", bt.CommInfoBase.COMM_FIXED))

    def _getcommission(self, size, price, pseudoexec):
        if size == 0:
            return 0.0
        return float(
            stock_order_fees(abs(int(size)), "Buy" if size > 0 else "Sell", price)["total"]
        )


class KDJ(bt.Indicator):
    """国内口径：RSV(n)，K = ⅔K₋₁ + ⅓RSV，D = ⅔D₋₁ + ⅓K，初值 50。"""

    lines = ("k", "d")
    params = (("n", 9),)

    def __init__(self):
        self.addminperiod(self.p.n)
        self._k = self._d = 50.0

    def next(self):
        hi = max(self.data.high.get(size=self.p.n))
        lo = min(self.data.low.get(size=self.p.n))
        rsv = (self.data.close[0] - lo) / (hi - lo) * 100 if hi > lo else 50.0
        self._k = 2 / 3 * self._k + 1 / 3 * rsv
        self._d = 2 / 3 * self._d + 1 / 3 * self._k
        self.lines.k[0], self.lines.d[0] = self._k, self._d


def _feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    d = df[["open", "high", "low", "close", "volume"]].copy()
    d.index = d.index.tz_convert("UTC").tz_localize(None)
    return bt.feeds.PandasData(dataname=d)


def _cerebro() -> bt.Cerebro:
    c = bt.Cerebro(cheat_on_open=True, stdstats=False)
    c.broker.setcash(1e12)
    c.broker.set_coo(True)
    c.broker.set_coc(True)
    c.broker.set_slippage_fixed(
        TICK, slip_open=True, slip_limit=True, slip_match=True, slip_out=True
    )
    c.broker.addcommissioninfo(StockFees())
    return c


# ---------------- E4 基线 ----------------


class E4(bt.Strategy):
    """E4 基线（R-036）：SMA120 定向 + KDJ 交叉 → 下一根开盘入场；止损 入场价 ∓ 1.5×ATR(14)；
    梯形 = 收盘跌破最近一个已确认的更高谷底（谷底严格低于左右各 2 根、同日）；
    浮盈且再出同向信号加仓、最多 2 次；15:50 清仓（提前收盘日最后一根收盘）；100 股。"""

    params = (
        ("ma", 120),
        ("stop_mult", 1.5),
        ("n", 2),
        ("max_adds", 2),
        ("lot", 100),
        ("day", None),
        ("mod", None),
    )

    def __init__(self):
        self.sma = bt.ind.SMA(self.data.close, period=self.p.ma)
        self.atr = bt.ind.ATR(self.data, period=14)
        self.kdj = KDJ(self.data, n=9)
        self.trades, self.pend = [], None
        self.pos_dir, self.fills, self.stop_o, self.cur = 0, [], None, None
        self.swing_lo, self.swing_hi = [], []
        self.ladder = np.nan
        self.exiting = False

    # 时间表（只用 bar 时间戳，不用未来价格）
    def _eod(self, b: int) -> bool:
        day, mod = self.p.day, self.p.mod
        return b + 1 >= len(day) or day[b + 1] != day[b] or mod[b] + 10 > FLAT

    def _swings(self, b: int) -> None:
        day, n = self.p.day, self.p.n
        if b == 0 or day[b] != day[b - 1]:
            self.swing_lo, self.swing_hi = [], []
        if b < 2 * n or day[b - 2 * n] != day[b]:
            return
        lows = [self.data.low[-k] for k in range(2 * n, -1, -1)]  # bar b−2n … b
        highs = [self.data.high[-k] for k in range(2 * n, -1, -1)]
        c = n
        if all(lows[c] < lows[x] for x in range(2 * n + 1) if x != c):
            self.swing_lo.append(lows[c])
        if all(highs[c] > highs[x] for x in range(2 * n + 1) if x != c):
            self.swing_hi.append(highs[c])

    def _ladder_from(self, seq: list, d: int) -> float:
        lad = np.nan
        for a, b in zip(seq[:-1], seq[1:], strict=True):
            if d * (b - a) > 0:
                lad = b
        return lad

    def next_open(self):
        if self.pend is None:
            return
        kind, info = self.pend
        self.pend = None
        o = self.data.open[0]
        if kind == "entry":
            d, atr, sig_b = info
            px = o + d * TICK
            self.pos_dir, self.fills = d, []
            self.cur = {
                "signal_bar": sig_b,
                "entry_bar": len(self),
                "direction": d,
                "adds": 0,
                "reason": None,
                "fees": 0.0,
            }
            (self.buy if d == 1 else self.sell)(size=self.p.lot, exectype=bt.Order.Limit, price=px)
            stop = px - d * self.p.stop_mult * atr
            self.cur["stop"] = stop
            self.stop_o = (self.sell if d == 1 else self.buy)(
                size=self.p.lot, exectype=bt.Order.Stop, price=stop
            )
        elif kind == "exit":
            self.cancel(self.stop_o)
            self.exiting = True
            d = self.pos_dir
            (self.sell if d == 1 else self.buy)(
                size=abs(self.position.size), exectype=bt.Order.Limit, price=o - d * TICK
            )
        elif kind == "add":
            d = self.pos_dir
            (self.buy if d == 1 else self.sell)(
                size=self.p.lot, exectype=bt.Order.Limit, price=o + d * TICK
            )
            self.cancel(self.stop_o)
            self.stop_o = (self.sell if d == 1 else self.buy)(
                size=abs(self.position.size) + self.p.lot,
                exectype=bt.Order.Stop,
                price=self.cur["stop"],
            )
            self.cur["adds"] += 1

    def notify_order(self, order):
        if order.status != order.Completed or self.cur is None:
            return
        self.cur["fees"] += order.executed.comm
        opening = (order.isbuy() and self.pos_dir == 1) or (order.issell() and self.pos_dir == -1)
        when = pd.Timestamp(
            bt.num2date(order.executed.dt), tz="UTC"
        )  # 成交所在 bar（不是通知到达的 bar）
        if opening:
            if not self.fills:
                self.cur["entry_ts"] = when
            self.fills.append((len(self), order.executed.price, abs(order.executed.size)))
            return
        if order.exectype == bt.Order.Stop:
            self.cur["reason"] = "止损"
        elif self.cur["reason"] is None:
            self.cur["reason"] = "梯形"
        d = self.pos_dir
        px = order.executed.price
        self.cur.update(
            {
                "exit_ts": when,
                "exit_px": px,
                "entry_px": self.fills[0][1],
                "gross": sum(d * q * (px - f) for _, f, q in self.fills),
            }
        )
        self.cur["net"] = self.cur["gross"] - self.cur["fees"]
        self.trades.append(self.cur)
        if order.exectype != bt.Order.Stop and self.stop_o is not None:
            self.cancel(self.stop_o)
        self.cur, self.pos_dir, self.fills, self.stop_o = None, 0, [], None
        self.ladder, self.exiting = np.nan, False

    def _signal(self, d: int) -> bool:
        c, s, s1 = self.data.close[0], self.sma[0], self.sma[-1]
        k, dd, k1, d1 = self.kdj.k[0], self.kdj.d[0], self.kdj.k[-1], self.kdj.d[-1]
        if d == 1:
            return c > s and s > s1 and k > dd and k1 <= d1
        return c < s and s < s1 and k < dd and k1 >= d1

    def next(self):
        b = len(self) - 1
        self._swings(b)
        if len(self) < max(self.p.ma + 1, 15):
            return
        mod, day = self.p.mod, self.p.day
        if self.position.size != 0 and not self.exiting and self.cur is not None:
            d = self.pos_dir
            if self._eod(b):  # 15:50 / 当日最后一根：收盘清仓（coc）
                self.cancel(self.stop_o)
                self.cur["reason"] = "15:50 清仓"
                self.exiting = True
                (self.sell if d == 1 else self.buy)(size=abs(self.position.size))
                return
            if b == self.cur["entry_bar"]:
                self.ladder = self._ladder_from(self.swing_lo if d == 1 else self.swing_hi, d)
            else:
                seq = self.swing_lo if d == 1 else self.swing_hi
                if len(seq) >= 2 and self._new_swing and d * (seq[-1] - seq[-2]) > 0:
                    self.ladder = seq[-1]
            if np.isfinite(self.ladder) and d * (self.data.close[0] - self.ladder) < 0:
                self.pend = ("exit", None)
                return
            avg = np.average([f for _, f, _ in self.fills], weights=[q for _, _, q in self.fills])
            if (
                self.cur["adds"] < self.p.max_adds
                and b > self.cur["entry_bar"]
                and self._signal(d)
                and d * (self.data.close[0] - avg) > 0
                and b + 1 < len(mod)
                and mod[b + 1] + 10 <= FLAT
            ):
                self.pend = ("add", None)
            return
        if self.position.size == 0 and self.cur is None and self.pend is None:
            for d in (1, -1):
                if self._signal(d):
                    j = b + 1
                    if (
                        j < len(day)
                        and day[j] == day[b]
                        and mod[j] + 10 <= FLAT
                        and np.isfinite(self.atr[0])
                    ):
                        self.pend = ("entry", (d, self.atr[0], b))
                    break

    def prenext(self):
        self._swings(len(self) - 1)

    def _track_new_swing(self):
        pass


def _wrap_swing(strat_cls):
    """记录本根是否新确认了摆动点（梯形只在新确认时更新）。"""
    orig = strat_cls._swings

    def _swings(self, b):
        n_lo, n_hi = len(self.swing_lo), len(self.swing_hi)
        orig(self, b)
        d = self.pos_dir
        self._new_swing = (len(self.swing_lo) > n_lo) if d == 1 else (len(self.swing_hi) > n_hi)

    strat_cls._swings = _swings
    return strat_cls


_wrap_swing(E4)


def run_e4(df5: pd.DataFrame) -> pd.DataFrame:
    et = df5.index.tz_convert(ET)
    c = _cerebro()
    c.adddata(_feed(df5))
    c.addstrategy(E4, day=np.asarray(et.date), mod=np.asarray(et.hour * 60 + et.minute))
    s = c.run()[0]
    t = pd.DataFrame(s.trades)
    if t.empty:
        return t
    idx = df5.index
    t["signal_ts"] = idx[t["signal_bar"].to_numpy()]
    return t


# ---------------- V4 H2（信号与入场时刻） ----------------


class H2(bt.Strategy):
    """V4 H2 的信号层（口径 C：含盘前盘后连续 2 分钟 K、不重置）与护栏，逐根独立重写。"""

    params = (("p", None),)

    def __init__(self):
        p = self.p.p
        self.bb = bt.ind.BollingerBands(self.data.close, period=20, devfactor=2.0)
        self.macd = bt.ind.MACD(self.data.close, period_me1=6, period_me2=13, period_signal=6)
        self.rows = []
        n = p["n_bars"]
        self.bw = np.full(n, np.nan)
        self.slope = np.full(n, np.nan)
        self.hist = np.full(n, np.nan)
        self.pctb = np.full(n, np.nan)
        self.mid = np.full(n, np.nan)
        self.state = np.zeros(n, int)
        self.sdir = np.zeros(n, int)
        self.ign = np.zeros(n, int)
        self.run = {k: 0 for k in ("exp", "hi", "lo", "hup", "hdn", "sup", "sdn")}

    def next(self):
        p, b = self.p.p, len(self) - 1
        top, mid, bot = self.bb.top[0], self.bb.mid[0], self.bb.bot[0]
        c, o = self.data.close[0], self.data.open[0]
        self.mid[b] = mid
        self.bw[b] = (top - bot) / mid
        self.pctb[b] = (c - bot) / (top - bot) if top > bot else np.nan
        self.slope[b] = (mid - self.mid[b - 1]) / mid if b else np.nan
        self.hist[b] = self.macd.macd[0] - self.macd.signal[0]
        r = self.run
        bw, pb, hs, sl = self.bw, self.pctb, self.hist, self.slope

        def step(key, cond):
            r[key] = r[key] + 1 if cond else 0
            return r[key]

        with np.errstate(invalid="ignore"):
            exp = step("exp", b > 0 and bw[b] > bw[b - 1])
            hi = step("hi", pb[b] > 0.9)
            lo = step("lo", pb[b] < 0.1)
            hup = step("hup", b > 0 and hs[b] > hs[b - 1] and hs[b] > 0)
            hdn = step("hdn", b > 0 and hs[b] < hs[b - 1] and hs[b] < 0)
            sup = step("sup", sl[b] > 0)
            sdn = step("sdn", sl[b] < 0)
        if exp >= 2 and hi >= 2 and hup >= 2:
            self.ign[b] = 1
        elif exp >= 2 and lo >= 2 and hdn >= 2:
            self.ign[b] = -1
        elif b and self.ign[b - 1] == 1 and pb[b] > 0.9:
            self.ign[b] = 1
        elif b and self.ign[b - 1] == -1 and pb[b] < 0.1:
            self.ign[b] = -1
        # 通道 (5, 10)
        chan = 0
        if b >= 10:
            closes = np.array(self.data.close.get(size=10))
            mids = self.mid[b - 9 : b + 1]
            fa = np.mean(closes > mids) if np.isfinite(mids).all() else np.nan
            fb = np.mean(closes < mids) if np.isfinite(mids).all() else np.nan
            keep = b >= 5 and bw[b] >= 0.9 * bw[b - 5]
            if sup >= 5 and fa >= 0.7 and keep:
                chan = 1
            elif sdn >= 5 and fb >= 0.7 and keep:
                chan = -1
        # RANGE：三项在近 w 根中的百分位秩（平均名次 / 有效个数）的加权和
        w, mp = p["range_w"], max(20, p["range_w"] // 4)
        score = np.nan
        if b + 1 >= mp:
            lo_i = max(0, b - w + 1)
            close_arr = np.array(self.data.close.get(size=b - lo_i + 1))
            feats = (
                bw[lo_i : b + 1],
                np.abs(sl[lo_i : b + 1]),
                np.abs(hs[lo_i : b + 1] / close_arr),
            )
            parts = []
            for arr in feats:
                v = arr[np.isfinite(arr)]
                x = arr[-1]
                if len(v) < mp or not np.isfinite(x):
                    parts = None
                    break
                parts.append((np.sum(v < x) + (np.sum(v == x) + 1) / 2) / len(v))
            if parts:
                score = sum(wt * q for wt, q in zip(p["weights"], parts, strict=True))
        st, sd = 0, 0
        if np.isfinite(score) and score <= p["range_thr"]:
            st = 3
        if chan:
            st, sd = 2, chan
        if self.ign[b]:
            st, sd = 1, self.ign[b]
        self.state[b], self.sdir[b] = st, sd
        # E1 回踩（H2）
        day = p["day"]
        if b < 1 or day[b - 1] != day[b] or not np.isfinite(top - bot) or top <= bot:
            return
        hw = (top - bot) / 2
        o1, c1 = self.data.open[-1], self.data.close[-1]
        L = min(self.data.low[-1], self.data.low[0])
        H = max(self.data.high[-1], self.data.high[0])
        sig = 0
        pl, ph = (L - bot) / (2 * hw), (H - bot) / (2 * hw)
        if chan == 1 and c1 < o1 and c > o and 0.45 <= pl <= 0.75 and abs(L - mid) <= 0.2 * hw:
            sig = 1
        elif (
            chan == -1 and c1 > o1 and c < o and 0.45 <= 1 - ph <= 0.75 and abs(H - mid) <= 0.2 * hw
        ):
            sig = -1
        if not sig:
            return
        lab, cls = p["label"][b], p["close_hm"][b]
        if not (lab >= "09:30" and cls <= "15:40" and day[b] in p["days"]):
            return
        if st == 3 or (st == 1 and sd == -sig):
            return
        self.rows.append({"bar": b, "direction": sig, "spot": c})


def pick(chain: pd.DataFrame, day: dt.date, d: int, spot: float, t_et: dt.time):
    """独立重写的合约规则：DTE（日历日）0/1 中最近到期日（10:15 后排除 0DTE），略虚值一档。"""
    right = "call" if d == 1 else "put"
    g = chain[chain["right"] == right]
    exps = sorted(e for e in g["expiry"].unique() if 0 <= (e - day).days <= 1)
    if t_et > dt.time(10, 15):
        exps = [e for e in exps if e != day]
    if not exps:
        return None
    ks = np.sort(g.loc[g["expiry"] == exps[0], "strike"].unique())
    cand = ks[ks > spot] if d == 1 else ks[ks < spot][::-1]
    if not len(cand):
        return None
    k = float(cand[0])
    sym = g[(g["expiry"] == exps[0]) & (g["strike"] == k)]["symbol"].iloc[0]
    return sym, exps[0]


def run_h2(
    root: Path, bars2: pd.DataFrame, days: set, range_thr: float, weights, range_days: float = 1.0
) -> pd.DataFrame:
    """range_days=1 复现现有实现的实际窗口（配置写 2 天，代码多除了一次周期，见审计报告）。"""
    et = bars2.index.tz_convert(ET)
    day = np.asarray(et.date)
    bpd = float(pd.Series(day).value_counts().median())
    params = {
        "n_bars": len(bars2),
        "day": day,
        "days": days,
        "range_thr": range_thr,
        "weights": weights,
        "range_w": max(1, round(range_days * bpd)),
        "label": np.asarray(et.strftime("%H:%M")),
        "close_hm": np.asarray((et + pd.Timedelta(minutes=2)).strftime("%H:%M")),
    }
    c = bt.Cerebro(stdstats=False)
    c.adddata(_feed(bars2))
    c.addstrategy(H2, p=params)
    s = c.run()[0]
    rows = pd.DataFrame(s.rows)
    if rows.empty:
        return rows
    rows["decision_ts"] = bars2.index[rows["bar"].to_numpy()] + pd.Timedelta(minutes=2)
    out, chains = [], {}
    for _, r in rows.iterrows():
        d0 = r["decision_ts"].tz_convert(ET).date()
        if d0 not in chains:
            p_ = root / "options_1m" / "underlying=MU" / f"date={d0.isoformat()}" / "quotes.parquet"
            chains[d0] = (
                pd.read_parquet(
                    p_, columns=["symbol", "expiry", "strike", "right"]
                ).drop_duplicates("symbol")
                if p_.exists()
                else pd.DataFrame()
            )
        if chains[d0].empty:
            continue
        pk = pick(
            chains[d0],
            d0,
            int(r["direction"]),
            float(r["spot"]),
            r["decision_ts"].tz_convert(ET).time(),
        )
        if pk is None:
            continue
        sym, exp = pk
        flat = pd.Timestamp(
            dt.datetime.combine(d0, dt.time(10, 44) if exp == d0 else dt.time(15, 49)), tz=ET
        ).tz_convert("UTC")
        if r["decision_ts"] >= flat:
            continue
        entry = first_tick(root, sym, d0, r["decision_ts"] + pd.Timedelta(seconds=2), flat)
        out.append(
            {
                "decision_ts": r["decision_ts"],
                "direction": int(r["direction"]),
                "symbol": sym,
                **entry,
            }
        )
    return pd.DataFrame(out)


def first_tick(root: Path, sym: str, d0: dt.date, t0: pd.Timestamp, flat: pd.Timestamp) -> dict:
    base = root / "diag" / "v4_ticks" / "underlying=MU" / f"date={d0.isoformat()}"
    files = sorted(base.glob(f"{sym}_*.parquet"))
    if not files:
        return {"entry_ts": pd.NaT, "entry_ask": np.nan}
    tk = pd.concat([pd.read_parquet(f) for f in files])
    tk = tk[(tk["ask"] > 0) & (tk["bid"] >= 0) & (tk["ask"] >= tk["bid"])]
    tk["ts"] = pd.to_datetime(tk["ts"], utc=True)
    tk = tk[tk["ts"] >= t0].sort_values("ts", kind="stable")
    if tk.empty or tk["ts"].iloc[0] >= flat:
        return {"entry_ts": pd.NaT, "entry_ask": np.nan}
    return {"entry_ts": tk["ts"].iloc[0], "entry_ask": float(tk["ask"].iloc[0])}
