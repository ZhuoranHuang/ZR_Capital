"""V4 H1 / H2 / H3 完整交易的独立复算（audit/base-final 第 4 节）→ BASE_FINAL_AUDIT.md 第 4 节。

分工（全部独立于 zr/research/v4_decisive.py 的实现，只共用数据、样本日与合约链文件）：
- backtrader（V4Full，继承 bt_independent.H2 的逐根状态层）：BOLL / MACD（backtrader 内置指标）、
  点火 / 通道 / RANGE 状态、H1（E10 k=2）、H2（E1）、H3（E11 + 下一根确认）的原始信号，
  以及结构离场触发 X3_mid2、X2_shrink2、X4_inside_0.1_0.9、X5_higher_low。
- 本模块的纯 Python 部分（按规则文字重写）：护栏（时间窗含提前收盘日、RANGE、逆向点火）、合约
  （bt_independent.pick）、tick 成交（+2 s 后第一个有效报价；ask 买入、bid 卖出；+20% 卖 1 张、
  −30% 全卖、结构 / 30 分钟上限 / 强平先到者）、费用（zr.costs.fees.option_order_fees）、
  同格不重叠（上一笔离场前的信号跳过）。
口径 C（主口径），MU。与引擎 research/v4/trades_C_MU.parquet 逐笔比对。

运行：uv run python -m zr.research.bt_full
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.data.calendar import session_close
from zr.features.bars import ET
from zr.research import bt_independent as bi
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.bt_full")
EXIT_RULES = ("X3_mid2", "X2_shrink2", "X4_inside_0.1_0.9", "X5_higher_low")


class V4Full(bi.H2):
    """在 H2 的逐根状态层上补 E10 / E11 原始信号与结构离场触发（多头、空头各一组）。"""

    def __init__(self):
        super().__init__()
        n = self.p.p["n_bars"]
        self.e10 = np.zeros(n, np.int8)
        self.e1 = np.zeros(n, np.int8)
        self.e11 = np.zeros(n, np.int8)
        self.hh = np.full(n, np.nan)
        self.ll = np.full(n, np.nan)
        self.ex = {r: (np.zeros(n, bool), np.zeros(n, bool)) for r in EXIT_RULES}
        self.out_up = self.out_dn = 0
        self.fall = self.rise = 0
        self.sw = {True: [np.nan, np.nan, np.nan], False: [np.nan, np.nan, np.nan]}

    def _e1(self, b, top, mid, bot, c, o, chan, st, sd):
        """E1 原始信号（不含护栏，护栏在 Python 部分统一处理）。"""
        day = self.p.p["day"]
        if b < 1 or day[b - 1] != day[b] or not np.isfinite(top - bot) or top <= bot:
            return
        hw = (top - bot) / 2
        o1, c1 = self.data.open[-1], self.data.close[-1]
        lo_ = min(self.data.low[-1], self.data.low[0])
        hi_ = max(self.data.high[-1], self.data.high[0])
        pl, ph = (lo_ - bot) / (2 * hw), (hi_ - bot) / (2 * hw)
        if chan == 1 and c1 < o1 and c > o and 0.45 <= pl <= 0.75 and abs(lo_ - mid) <= 0.2 * hw:
            self.e1[b] = 1
        elif (
            chan == -1
            and c1 > o1
            and c < o
            and 0.45 <= 1 - ph <= 0.75
            and abs(hi_ - mid) <= 0.2 * hw
        ):
            self.e1[b] = -1

    def next(self):
        b, top, mid, bot, c, o, chan, st, sd = self._regime()
        self._e1(b, top, mid, bot, c, o, chan, st, sd)
        p, day = self.p.p, self.p.p["day"]
        pb, hs = self.pctb[b], self.hist[b]
        same_prev = b >= 1 and day[b - 1] == day[b]
        # E10：点火中第 k 根收在外轨之外（%b > 1 / < 0 的连续根数 == k）
        self.out_up = self.out_up + 1 if pb > 1 else 0
        self.out_dn = self.out_dn + 1 if pb < 0 else 0
        k = p["e10_k"]
        if st == 1 and sd == 1 and self.out_up == k:
            self.e10[b] = 1
        elif st == 1 and sd == -1 and self.out_dn == k:
            self.e10[b] = -1
        # E11：前一根 RANGE（同日）、收盘破前 M 根高 / 低点、量比 ≥ v（相对前 vw 根均量）
        m, v, vw = p["e11_M"], p["e11_v"], p["e11_vw"]
        if b >= m:
            self.hh[b] = max(self.data.high.get(ago=-1, size=m))
            self.ll[b] = min(self.data.low.get(ago=-1, size=m))
        if b >= vw and same_prev and self.state[b - 1] == 3:
            vr = self.data.volume[0] / np.mean(self.data.volume.get(ago=-1, size=vw))
            if c > self.hh[b] and vr >= v:
                self.e11[b] = 1
            elif c < self.ll[b] and vr >= v:
                self.e11[b] = -1
        # 结构离场
        with np.errstate(invalid="ignore"):
            below, above = c < mid, c > mid
            pb1, pc1 = self.mid[b - 1] if b else np.nan, self.data.close[-1] if b else np.nan
            self.ex["X3_mid2"][0][b] = bool(below and same_prev and pc1 < pb1)
            self.ex["X3_mid2"][1][b] = bool(above and same_prev and pc1 > pb1)
            self.fall = self.fall + 1 if b and hs < self.hist[b - 1] else 0
            self.rise = self.rise + 1 if b and hs > self.hist[b - 1] else 0
            self.ex["X2_shrink2"][0][b] = self.fall >= 2
            self.ex["X2_shrink2"][1][b] = self.rise >= 2
            self.ex["X4_inside_0.1_0.9"][0][b] = bool(pb < 0.9)
            self.ex["X4_inside_0.1_0.9"][1][b] = bool(pb > 0.1)
        for long in (True, False):
            lv = self._swing(b, long)
            self.ex["X5_higher_low"][0 if long else 1][b] = bool(
                np.isfinite(lv) and (c < lv if long else c > lv)
            )

    def _swing(self, b: int, long: bool) -> float:
        """当日最近一个更高低点（多头）/ 更低高点（空头）：摆动点左右各 2 根确认。"""
        day = self.p.p["day"]
        s = self.sw[long]  # [前一个摆动点, 最近一个摆动点, 当前有效水位]
        if b and day[b] != day[b - 1]:
            s[:] = [np.nan, np.nan, np.nan]
        if b >= 4 and day[b - 4] == day[b]:
            line = self.data.low if long else self.data.high
            win = np.array(line.get(size=5))
            x = win[2]
            if (long and x == win.min()) or (not long and x == win.max()):
                s[0], s[1] = s[1], x
                if np.isfinite(s[0]) and ((long and s[1] > s[0]) or (not long and s[1] < s[0])):
                    s[2] = s[1]
        return s[2]


def run_bars(bars2: pd.DataFrame, v4: dict, days: set) -> dict:
    et = bars2.index.tz_convert(ET)
    day = np.asarray(et.date)
    bpd = float(pd.Series(day).value_counts().median())
    h = v4["hypotheses"]
    params = {
        "n_bars": len(bars2),
        "day": day,
        "days": days,
        "range_thr": v4["conventions"]["C"]["range_threshold"],
        "weights": (0.3333, 0.3333, 0.3334),
        "range_w": max(1, round(1.0 * bpd)),
        "label": np.asarray(et.strftime("%H:%M")),
        "close_hm": np.asarray((et + pd.Timedelta(minutes=2)).strftime("%H:%M")),
        "e10_k": h["H1"]["k"],
        "e11_M": h["H3"]["M"],
        "e11_v": h["H3"]["v"],
        "e11_vw": h["H3"]["vol_window"],
    }
    c = bt.Cerebro(stdstats=False)
    c.adddata(bi._feed(bars2))
    c.addstrategy(V4Full, p=params)
    s = c.run()[0]
    # H3：E11 后下一根（同日）收盘仍在突破水位之外
    h3 = np.zeros(len(bars2), np.int8)
    cl = bars2["close"].to_numpy(float)
    for i in np.flatnonzero(s.e11):
        j = i + 1
        if j < len(bars2) and day[j] == day[i]:
            if s.e11[i] == 1 and cl[j] > s.hh[i]:
                h3[j] = 1
            elif s.e11[i] == -1 and cl[j] < s.ll[i]:
                h3[j] = -1
    return {
        "sig": {"H1": s.e10, "H2": s.e1, "H3": h3},
        "state": s.state,
        "sdir": s.sdir,
        "exits": s.ex,
    }


# ---------------- Python 部分：护栏、合约、tick 成交 ----------------


def guards(bars2: pd.DataFrame, state, sdir, days: set) -> tuple[np.ndarray, np.ndarray]:
    """(多头可入场, 空头可入场)。最后入场 K 收盘 ≤ 收盘前 20 分钟（正常日 15:40）。"""
    et = bars2.index.tz_convert(ET)
    d = np.asarray(et.date)
    lab = np.asarray(et.strftime("%H:%M"))
    cls = et + pd.Timedelta(minutes=2)
    last = {
        x: (dt.datetime.combine(x, session_close(x)) - dt.timedelta(minutes=20)).time()
        for x in set(d)
        if x in days
    }
    ok_t = np.array(
        [
            x in last and lb >= "09:30" and c.time() <= last[x]
            for x, lb, c in zip(d, lab, cls, strict=True)
        ]
    )
    ok = ok_t & (state != 3)
    return ok & ~((state == 1) & (sdir == -1)), ok & ~((state == 1) & (sdir == 1))


def load_ticks(root: Path, sym: str, d0: dt.date) -> pd.DataFrame | None:
    base = root / "diag" / "v4_ticks" / "underlying=MU" / f"date={d0.isoformat()}"
    files = sorted(base.glob(f"{sym}_*.parquet"))
    if not files:
        return None
    tk = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tk = tk[(tk["ask"] > 0) & (tk["bid"] >= 0) & (tk["ask"] >= tk["bid"])].drop_duplicates()
    tk["t"] = pd.to_datetime(tk["ts"], utc=True).astype("int64")
    if tk["t"].iloc[0] < 10**17:  # 微秒精度 → 纳秒
        tk["t"] = pd.DatetimeIndex(pd.to_datetime(tk["ts"], utc=True)).as_unit("ns").asi8
    return tk.sort_values("t", kind="stable").reset_index(drop=True)


def _at(tk: pd.DataFrame, t: int) -> int:
    k = int(np.searchsorted(tk["t"].to_numpy(), t, side="left"))
    return min(k, len(tk) - 1)


def fill(tk: pd.DataFrame, dec: int, struct: list[int], flat: int, v4: dict) -> dict | None:
    exe, prem = v4["execution"], v4["premium"]
    lag, qty = int(exe["lag_s"] * 1e9), exe["qty"]
    t = tk["t"].to_numpy()
    i0 = int(np.searchsorted(t, dec + lag, side="left"))
    if i0 >= len(tk) or t[i0] >= flat:
        return None
    ask0, te = float(tk["ask"].iloc[i0]), int(t[i0])
    mid = ((tk["bid"] + tk["ask"]) / 2).to_numpy()[i0 + 1 :]
    after = t[i0 + 1 :]
    inf = np.iinfo(np.int64).max
    hit_tp = after[mid >= ask0 * (1 + prem["tp"])]
    hit_sl = after[mid <= ask0 * (1 - prem["sl"])]
    t_tp = int(hit_tp[0]) if len(hit_tp) else inf
    nxt = [x for x in struct if x > dec]
    order = [
        ("structure", nxt[0] if nxt else inf),
        ("sl", int(hit_sl[0]) if len(hit_sl) else inf),
        ("cap", te + int(v4["cap_min"] * 60e9)),
        ("flat", int(flat)),
    ]
    reason, t_f = min(order, key=lambda r: r[1])  # 同时刻按上面顺序
    cash = -ask0 * 100 * qty - float(option_order_fees(qty, "Buy", ask0)["total"])
    left = qty
    if t_tp < t_f:
        b = float(tk["bid"].iloc[_at(tk, t_tp + lag)])
        cash += b * 100 - (float(option_order_fees(1, "Sell", b)["total"]) if b > 0 else 0.0)
        left -= 1
    k = _at(tk, t_f + lag)
    b = float(tk["bid"].iloc[k])
    cash += left * b * 100 - (float(option_order_fees(left, "Sell", b)["total"]) if b > 0 else 0)
    return {
        "entry_ns": te,
        "exit_ns": int(t[k]),
        "ask0": ask0,
        "net_usd": round(cash, 6),
        "tp_used": t_tp < t_f,
        "reason": reason,
    }


def chain_for(root: Path, d0: dt.date, cache: dict) -> pd.DataFrame:
    if d0 not in cache:
        p = root / "options_1m" / "underlying=MU" / f"date={d0.isoformat()}" / "quotes.parquet"
        cache[d0] = (
            pd.read_parquet(p, columns=["symbol", "expiry", "strike", "right"]).drop_duplicates(
                "symbol"
            )
            if p.exists()
            else pd.DataFrame()
        )
    return cache[d0]


def trades(root: Path, bars2: pd.DataFrame, res: dict, v4: dict, days: set) -> pd.DataFrame:
    ok_l, ok_s = guards(bars2, res["state"], res["sdir"], days)
    close_utc = bars2.index + pd.Timedelta(minutes=2)
    cns = pd.DatetimeIndex(close_utc).as_unit("ns").asi8
    et = close_utc.tz_convert(ET)
    spot = bars2["close"].to_numpy(float)
    chains, ticks, out = {}, {}, []
    for h in ("H1", "H2", "H3"):
        for xk in ("a", "b", "c"):
            rule = v4["exits"][xk][h]
            for d in (1, -1):
                trig = cns[res["exits"][rule][0 if d == 1 else 1]].tolist()
                cand = np.flatnonzero((res["sig"][h] == d) & (ok_l if d == 1 else ok_s))
                last_exit = -1
                for i in cand:
                    dec = int(cns[i])
                    if dec < last_exit:
                        continue
                    d0 = et[i].date()
                    ch = chain_for(root, d0, chains)
                    if ch.empty:
                        continue
                    pk = bi.pick(ch, d0, d, spot[i], et[i].time())
                    if pk is None:
                        continue
                    sym, exp = pk
                    close = dt.datetime.combine(d0, session_close(d0))
                    flat_t = (close - dt.timedelta(minutes=11)).time()  # 15:49；提前收盘日前移
                    if exp == d0:
                        flat_t = min(flat_t, dt.time(10, 44))
                    flat = pd.Timestamp(dt.datetime.combine(d0, flat_t), tz=ET).value
                    if dec >= flat:
                        continue
                    if (sym, d0) not in ticks:
                        ticks[(sym, d0)] = load_ticks(root, sym, d0)
                    tk = ticks[(sym, d0)]
                    if tk is None or tk.empty:
                        continue
                    r = fill(tk, dec, trig, flat, v4)
                    if r is None:
                        continue
                    last_exit = r["exit_ns"]
                    out.append(
                        {
                            "hyp": h,
                            "exit": xk,
                            "dir": "long" if d == 1 else "short",
                            "decision_ns": dec,
                            "symbol": sym,
                            **r,
                        }
                    )
    return pd.DataFrame(out)


def compare(eng: pd.DataFrame, btt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    k = ["hyp", "exit", "dir", "decision_ns"]
    j = eng.merge(btt, on=k, how="outer", suffixes=("_eng", "_bt"), indicator=True)
    both = j["_merge"] == "both"
    j["same_symbol"] = both & (j["symbol_eng"] == j["symbol_bt"])
    j["same_entry"] = both & (j["entry_ns_eng"] == j["entry_ns_bt"])
    j["same_exit"] = both & (j["exit_ns_eng"] == j["exit_ns_bt"])
    j["same_reason"] = both & (j["reason_eng"] == j["reason_bt"])
    j["same_net"] = both & np.isclose(j["net_usd_eng"], j["net_usd_bt"], atol=0.005)
    j["all_same"] = (
        j["same_symbol"] & j["same_entry"] & j["same_exit"] & j["same_reason"] & j["same_net"]
    )
    g = j.groupby(["hyp", "exit", "dir"])
    summ = pd.DataFrame(
        {
            "引擎笔数": g.apply(lambda x: int((x["_merge"] != "right_only").sum())),
            "backtrader 笔数": g.apply(lambda x: int((x["_merge"] != "left_only").sum())),
            "逐笔完全一致": g["all_same"].sum().astype(int),
            "仅引擎": g.apply(lambda x: int((x["_merge"] == "left_only").sum())),
            "仅 backtrader": g.apply(lambda x: int((x["_merge"] == "right_only").sum())),
            "引擎净值": g["net_usd_eng"].sum().round(2),
            "backtrader 净值": g["net_usd_bt"].sum().round(2),
        }
    ).reset_index()
    return summ, j


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    days = v4d.sample_days(root, "MU", v4)
    bars2 = v4d.build_bars(root, "MU", v4["conventions"]["C"], max(days))
    log.info("backtrader 逐根：%d 根", len(bars2))
    res = run_bars(bars2, v4, set(days))
    btt = trades(root, bars2, res, v4, set(days))
    eng = pd.read_parquet(root / "research" / "v4" / "trades_C_MU.parquet")
    summ, j = compare(eng, btt)
    out = root / "research" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    j.to_parquet(out / "bt_full_compare.parquet")
    summ.to_csv(out / "bt_full_summary.csv", index=False)
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
