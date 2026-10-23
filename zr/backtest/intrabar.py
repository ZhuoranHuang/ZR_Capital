"""盘中（逐秒）回测引擎：YAML 规则 → 逐秒评估入场条件 → tick 级挂单 / 成交 → 离场，费用 D-013。

数据：
- 正股秒级价：$ZR_DATA_ROOT/intrabar/underlying=<UND>/date=D/S.parquet（zr/data/intrabar_pull.py，
  看跌看涨平价反推），按 1 秒网格取最近值（超过 10 秒未更新为空）；
- 已完成 2 分钟 K（口径 C：含盘前盘后连续、长桥原始 K）的 BOLL(20,2) / MACD(6,13,6)，
  某秒只能看到收盘时刻 ≤ 该秒的 K；
- 期权：入场时按规则选合约（到期日与行权价来自 ThetaData list 端点），按需拉逐笔 NBBO
  （diag/v4_ticks）与 OPRA 成交（diag/v4_trades）。
成交：zr/backtest/fill_limit（盘口越过 / OPRA 成交越过挂单价 / 恰好等于时以 q 成交；到达即可成交的
挂单按对手价成交）。同一时间只持一个仓位。

规则 YAML 见 config/intrabar/example_owner_profile.yaml；条件特征：
- ret：近 window_s 秒正股涨跌（bp，directional=true 时按方向）；
- dist_extreme：距近 window_min 分钟高 / 低点（bp；side=with 顺向、against 逆向）；
- boll：当前价相对已完成 K 布林（ref = mid / upper / lower，比较当前价 − 轨道，bp，可按方向）；
- pctb：当前价在已完成 K 布林中的 %b（directional=true：看跌用 1 − %b）；
- macd_hist：已完成 K 的柱（directional=true 时乘方向）；macd_hist_chg：柱相对上一根的变化；
- trend：最近 bars 根已完成 K 中收在中轨同侧的根数 ≥ min_same 的那一侧
  （direction=trend 时用作方向）。
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.backtest.fill_limit import Book, PriceImprovement, _rest, _round_passive, _tick, cross_now
from zr.costs.fees import option_order_fees
from zr.data.calendar import session_close
from zr.data.thetadata import ET, ThetaClient, lb_symbol, parse_contract
from zr.features.regime import indicators
from zr.research import v4_decisive as v4d
from zr.research.sim_v3 import sell_fee

log = logging.getLogger("zr.backtest.intrabar")
NS = 1_000_000_000
MAX_AGE = 10 * NS
OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal}


def load_rule(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ---------------- 数据 ----------------


def second_grid(root: Path, und: str, day: dt.date) -> pd.DataFrame | None:
    p = root / "intrabar" / f"underlying={und}" / f"date={day}" / "S.parquet"
    if not p.exists():
        return None
    s = pd.read_parquet(p, columns=["ts", "S"])
    open_ns = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).value
    close_ns = pd.Timestamp(dt.datetime.combine(day, session_close(day)), tz=ET).value
    grid = np.arange(open_ns, close_ns, NS)
    ts = s["ts"].to_numpy()
    i = np.searchsorted(ts, grid, side="right") - 1
    ok = (i >= 0) & ((grid - ts[np.clip(i, 0, None)]) <= MAX_AGE)
    return pd.DataFrame(
        {"t": grid, "S": np.where(ok, s["S"].to_numpy()[np.clip(i, 0, None)], np.nan)}
    )


class BarState:
    """已完成 2 分钟 K 的指标，按秒查询（只看收盘时刻 ≤ t 的 K）。"""

    def __init__(self, root: Path, und: str, end: dt.date, conv_key: str = "C"):
        v4 = v4d.load_v4()
        conv = v4["conventions"][conv_key]
        cfg = v4d.state_cfg(v4, conv_key)
        self.bars = v4d.build_bars(root, und, conv, end)
        x = indicators(self.bars, cfg, daily_reset=conv["reset"], warmup=conv["warmup"])
        self.close_ns = v4d.ns(self.bars.index + pd.Timedelta(minutes=v4["period"]))
        self.mid, self.up, self.lo = (x[k].to_numpy() for k in ("mid", "upper", "lower"))
        self.hist = x["hist"].to_numpy()
        self.close = self.bars["close"].to_numpy(float)

    def idx(self, t: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.close_ns, t, side="right") - 1

    def trend(self, t: np.ndarray, bars: int, min_same: int) -> np.ndarray:
        i = self.idx(t)
        above = (self.close > self.mid).astype(float)
        below = (self.close < self.mid).astype(float)
        ca, cb = np.r_[0, np.cumsum(above)], np.r_[0, np.cumsum(below)]
        lo = np.clip(i - bars + 1, 0, None)
        na, nb = ca[i + 1] - ca[lo], cb[i + 1] - cb[lo]
        return np.where(na >= min_same, 1, np.where(nb >= min_same, -1, 0))


def features(g: pd.DataFrame, bs: BarState, specs: list[dict]) -> dict[str, np.ndarray]:
    """每个条件需要的（非方向化）原始量；方向化在 evaluate 里做。"""
    t, s = g["t"].to_numpy(), g["S"].to_numpy()
    out: dict[str, np.ndarray] = {}
    i = bs.idx(t)
    for c in specs:
        f = c["feature"]
        if f == "ret":
            w = int(c["window_s"])
            prev = np.r_[np.full(w, np.nan), s[:-w]]
            out[f"ret_{w}"] = (s / prev - 1) * 1e4
        elif f == "dist_extreme":
            n = int(c["window_min"]) * 60
            ser = pd.Series(s)
            hi = ser.rolling(n, min_periods=1).max().to_numpy()
            lo = ser.rolling(n, min_periods=1).min().to_numpy()
            out[f"hi_{n}"], out[f"lo_{n}"] = (hi / s - 1) * 1e4, (1 - lo / s) * 1e4
        elif f in ("boll", "pctb"):
            out["mid"], out["up"], out["lo"] = bs.mid[i], bs.up[i], bs.lo[i]
        elif f in ("macd_hist", "macd_hist_chg"):
            out["hist"] = bs.hist[i]
            out["hist_chg"] = bs.hist[i] - bs.hist[np.clip(i - 1, 0, None)]
    out["S"] = s
    return out


def evaluate(feat: dict, specs: list[dict], d: int) -> np.ndarray:
    """方向 d（+1 / −1）下全部条件同时成立的布尔数组。"""
    s = feat["S"]
    ok = np.isfinite(s)
    for c in specs:
        f, op, v = c["feature"], OPS[c["op"]], float(c.get("value", 0))
        sgn = d if c.get("directional", True) else 1
        if f == "ret":
            x = sgn * feat[f"ret_{int(c['window_s'])}"]
        elif f == "dist_extreme":
            n = int(c["window_min"]) * 60
            hi, lo = feat[f"hi_{n}"], feat[f"lo_{n}"]
            with_d, against_d = (hi, lo) if d == 1 else (lo, hi)
            x = with_d if c.get("side", "with") == "with" else against_d
        elif f == "boll":
            ref = feat[{"mid": "mid", "upper": "up", "lower": "lo"}[c.get("ref", "mid")]]
            x = sgn * (s / ref - 1) * 1e4
        elif f == "pctb":
            p = (s - feat["lo"]) / (feat["up"] - feat["lo"])
            x = p if sgn == 1 else 1 - p
        elif f == "macd_hist":
            x = sgn * feat["hist"]
        elif f == "macd_hist_chg":
            x = sgn * feat["hist_chg"]
        else:
            raise ValueError(f"未知特征 {f}")
        with np.errstate(invalid="ignore"):
            ok &= op(x, v)
    return ok


# ---------------- 期权数据 ----------------


class OptionData:
    def __init__(self, root: Path, und: str, concurrency: int = 4):
        self.root, self.und = root, und
        self.client = ThetaClient(concurrency=concurrency)
        self._exps: list[dt.date] = []
        self._strikes: dict[dt.date, np.ndarray] = {}

    def expirations(self) -> list[dt.date]:
        if not self._exps:
            self._exps = self.client.list_expirations(self.und)
        return self._exps

    def strikes(self, exp: dt.date) -> np.ndarray:
        if exp not in self._strikes:
            rows = self.client.get_json(
                "/option/list/strikes", symbol=self.und, expiration=exp.strftime("%Y%m%d")
            )
            self._strikes[exp] = np.sort(np.array([float(r["strike"]) for r in rows]))
        return self._strikes[exp]

    def pick(self, day: dt.date, t_et: dt.time, d: int, spot: float, rule: dict) -> str | None:
        c = rule.get("contract", {})
        exps = [e for e in self.expirations() if 0 <= (e - day).days <= c.get("dte_max", 1)]
        cut = c.get("no_0dte_after")
        if cut and t_et > dt.time.fromisoformat(cut):
            exps = [e for e in exps if e != day]
        if not exps:
            return None
        ks = self.strikes(exps[0])
        k = c.get("otm_steps", 1)
        if d == 1:
            cand = ks[ks > spot]
            j = k - 1 if k >= 1 else 0
        else:
            cand = ks[ks < spot][::-1]
            j = k - 1 if k >= 1 else 0
        if k == 0:  # 平值：最接近的行权价
            return lb_symbol(
                self.und,
                exps[0],
                "call" if d == 1 else "put",
                float(ks[np.argmin(np.abs(ks - spot))]),
            )
        if len(cand) <= j:
            return None
        return lb_symbol(self.und, exps[0], "call" if d == 1 else "put", float(cand[j]))

    def book(self, sym: str, day: dt.date, start: pd.Timestamp, end: pd.Timestamp) -> Book | None:
        c = parse_contract(sym)
        w = pd.DataFrame(
            [
                {
                    "symbol": sym,
                    "day": day,
                    "start": start,
                    "end": end,
                    "expiry": c["expiry"],
                    "strike": c["strike"],
                    "right": c["right"],
                }
            ]
        )
        v4d.pull_ticks(self.root, self.und, w)
        self._trades(sym, day)
        base = (
            self.root / "diag" / "v4_ticks" / f"underlying={self.und}" / f"date={day.isoformat()}"
        )
        qf = sorted(base.glob(f"{sym}_*.parquet"))
        if not qf:
            return None
        q = pd.concat([pd.read_parquet(f) for f in qf], ignore_index=True).drop_duplicates()
        q = q[(q["ask"] > 0) & (q["bid"] > 0) & (q["ask"] >= q["bid"])]
        if q.empty:
            return None
        tb = self.root / "diag" / "v4_trades" / f"underlying={self.und}" / f"date={day.isoformat()}"
        tp = [pd.read_parquet(tb / f.name) for f in qf if (tb / f.name).exists()]
        tp = [x for x in tp if len(x)]
        return Book.from_frames(
            q, pd.concat(tp, ignore_index=True).drop_duplicates() if tp else None
        )

    def _trades(self, sym: str, day: dt.date) -> None:
        base = (
            self.root / "diag" / "v4_ticks" / f"underlying={self.und}" / f"date={day.isoformat()}"
        )
        out = (
            self.root / "diag" / "v4_trades" / f"underlying={self.und}" / f"date={day.isoformat()}"
        )
        c = parse_contract(sym)
        for f in sorted(base.glob(f"{sym}_*.parquet")):
            dst = out / f.name
            if dst.exists():
                continue
            a, b = f.stem.rsplit("_", 1)[1].split("-")
            t = self.client.history_trade(
                c["und"],
                c["expiry"],
                day,
                f"{c['strike']:.3f}",
                c["right"],
                start_time=f"{a[:2]}:{a[2:4]}:{a[4:6]}.000",
                end_time=f"{b[:2]}:{b[2:4]}:{b[4:6]}.000",
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            cols = [x for x in ("ts", "price", "size") if x in t.columns]
            (t[cols] if cols else pd.DataFrame(columns=["ts", "price", "size"])).to_parquet(
                dst, index=False
            )


# ---------------- 下单与成交 ----------------


def place(
    book: Book,
    side: int,
    qty: float,
    t: int,
    order: dict,
    sym: str,
    q: float,
    end: int | None,
    rng: np.random.Generator,
    pi: PriceImprovement | None = None,
    info: dict | None = None,
) -> tuple[float, float, int | None, int]:
    """返回 (成交张数, 均价, 最后成交时刻, 结束时刻)。order：
    {mode: taker} → +lag 后第一个对手价；
    {mode: passive, k_spread, wait_s} → 中间价 − 方向 × k × 价差（按 tick 向己方取整，不越过
    对手价），等 wait_s 秒，未成交的放弃；
    {mode: path, path: [(t_ns, 价格), ...], end_ns} → 按给定价格路径挂单 / 改价（已知答案重放）。"""
    lag = int(order.get("lag_s", 2.0) * NS)
    mode = order.get("mode", "taker")
    if mode == "taker":
        c = cross_now(book, side, t, lag / NS, pi, sym, rng)
        if c is None:
            return 0.0, math.nan, None, t
        return float(qty), float(c[1]), int(c[0]), int(c[0])
    if mode == "passive":
        t0 = t + lag
        i = book.asof(t0)
        if i is None:
            return 0.0, math.nan, None, t0
        mid, spr = book.mid(i), book.ask[i] - book.bid[i]
        tick = _tick(sym, mid)
        px = _round_passive(mid - side * float(order.get("k_spread", 0.0)) * spr, tick, side)
        px = min(px, book.ask[i]) if side == 1 else max(px, book.bid[i])
        e = t0 + int(float(order.get("wait_s", 60)) * NS)
        if end is not None:
            e = min(e, end)
        fills, _ = _rest(book, side, px, t0, e, qty, rng, q, pi, sym)
    elif mode == "path":
        fills, left = [], float(qty)
        path = order["path"]
        force = order.get("force_ns")  # R-047：到该时刻仍未全成的张数按当时对手价补成交
        stop = int(force) if force is not None else None
        for k, (ts_k, px_k) in enumerate(path):
            if left <= 0:
                break
            e = path[k + 1][0] if k + 1 < len(path) else order["end_ns"]
            a0, b0 = int(ts_k) + lag, int(e) + lag
            if stop is not None:
                if a0 >= stop:
                    break
                b0 = min(b0, stop)
            f, left = _rest(book, side, float(px_k), a0, b0, left, rng, q, pi, sym)
            fills += f
        e = int(order["end_ns"]) + lag
        if stop is not None and left > 0:
            i = book.first_at(stop)
            i = book.asof(stop) if i is None else i
            if i is not None:
                bid, ask = float(book.bid[i]), float(book.ask[i])
                opp = ask if side == 1 else bid
                if pi is not None:  # 立即成交类别；以对手价为界（卖出不低于买一、买入不高于卖一）
                    opp = pi.price(side, bid, ask, "immediate", sym, rng, limit=opp)
                fills.append((max(stop, int(book.q_ns[i])), opp, left))
                if info is not None:
                    info["forced"] = left
                left = 0.0
                e = stop
    else:
        raise ValueError(mode)
    if not fills:
        return 0.0, math.nan, None, e
    n = sum(f[2] for f in fills)
    return float(n), sum(f[1] * f[2] for f in fills) / n, int(fills[-1][0]), int(fills[-1][0])


def fees(n: float, px: float, side: str) -> float:
    if n <= 0:
        return 0.0
    return (
        float(option_order_fees(int(n), "Buy", px)["total"])
        if side == "Buy"
        else float(sell_fee(int(n), px))
    )


# ---------------- 主循环 ----------------


def run_day(
    root: Path, rule: dict, day: dt.date, bs: BarState, od: OptionData, seed: int = 0
) -> list[dict]:
    g = second_grid(root, rule["underlying"], day)
    if g is None:
        return []
    ent = rule["entry"]["all"]
    feat = features(g, bs, ent + rule.get("exit", {}).get("signal", []))
    t = g["t"].to_numpy()
    et = pd.DatetimeIndex(t).tz_localize("UTC").tz_convert(ET)
    hm = np.asarray(et.strftime("%H:%M"))
    sess = rule.get("session", {})
    in_win = (hm >= sess.get("first_entry", "09:32")) & (hm <= sess.get("last_entry", "15:40"))
    dirs = {"long": [1], "short": [-1], "both": [1, -1]}.get(rule.get("direction", "both"))
    if rule.get("direction") == "trend":
        tr = rule["trend"]
        trend = bs.trend(t, int(tr["bars"]), int(tr["min_same"]))
        cond = {
            1: evaluate(feat, ent, 1) & (trend == 1),
            -1: evaluate(feat, ent, -1) & (trend == -1),
        }
    else:
        cond = {d: evaluate(feat, ent, d) for d in dirs}
    sig_exit = rule.get("exit", {}).get("signal")
    flat = pd.Timestamp(
        dt.datetime.combine(day, dt.time.fromisoformat(sess.get("flat", "15:50"))), tz=ET
    ).value
    ex = rule.get("exit", {})
    q = float(rule.get("fill", {}).get("queue_fill_prob", 0.9))
    rng = np.random.default_rng(seed)
    pi = None
    if rule.get("fill", {}).get("price_improvement"):  # R-046；默认关闭（吃价差 = 保守对照）
        pi = PriceImprovement.from_yaml(
            Path(__file__).resolve().parents[2] / "config" / "price_improvement.yaml"
        )
    trades, k = [], 0
    while k < len(t):
        hit = [d for d in cond if cond[d][k] and in_win[k]]
        if not hit:
            k += 1
            continue
        d, t0 = hit[0], int(t[k])
        sym = od.pick(day, et[k].time(), d, float(g["S"].iloc[k]), rule)
        if sym is None:
            k += 1
            continue
        hold = int(float(ex.get("max_hold_min", 30)) * 60 * NS)
        bk = od.book(
            sym,
            day,
            pd.Timestamp(t0, tz="UTC") - pd.Timedelta(minutes=1),
            pd.Timestamp(min(t0 + hold + 300 * NS, flat + 120 * NS), tz="UTC"),
        )
        if bk is None:
            k += 1
            continue
        qty = float(rule.get("qty", 1))
        n, px0, te, tend = place(
            bk, 1, qty, t0, rule.get("order", {"mode": "taker"}), sym, q, flat, rng
        )
        if n <= 0:
            trades.append({"day": day, "decision_ns": t0, "dir": d, "symbol": sym, "filled": 0})
            k = int(np.searchsorted(t, tend, side="right"))
            continue
        i0 = int(np.searchsorted(bk.q_ns, te, side="right"))
        qt, mid = bk.q_ns[i0:], (bk.bid[i0:] + bk.ask[i0:]) / 2
        big = np.iinfo(np.int64).max
        cand = {"flat": flat, "cap": te + hold}
        if "tp" in ex:
            h = qt[mid >= px0 * (1 + float(ex["tp"]))]
            cand["tp"] = int(h[0]) if len(h) else big
        if "sl" in ex:
            h = qt[mid <= px0 * (1 - float(ex["sl"]))]
            cand["sl"] = int(h[0]) if len(h) else big
        if sig_exit:
            se = evaluate(feat, sig_exit, d)
            j = np.flatnonzero(se & (t > te))
            cand["signal"] = int(t[j[0]]) if len(j) else big
        reason = min(cand, key=cand.get)
        x_order = ex.get("order", {"mode": "taker"})
        if reason in ("sl", "flat"):
            x_order = {"mode": "taker", "lag_s": x_order.get("lag_s", 2.0)}
        xn, xpx, xt, _ = place(bk, -1, n, cand[reason], x_order, sym, q, flat, rng, pi)
        if xn < n:  # 离场挂单没成交完：剩余吃买一
            c = cross_now(bk, -1, cand[reason], 2.0, pi, sym, rng)
            rem = n - xn
            xpx = ((xpx * xn if xn else 0.0) + c[1] * rem) / n if c else xpx
            xt = c[0] if c else xt
        cash = -px0 * 100 * n - fees(n, px0, "Buy") + xpx * 100 * n - fees(n, xpx, "Sell")
        trades.append(
            {
                "day": day,
                "decision_ns": t0,
                "dir": d,
                "symbol": sym,
                "filled": n,
                "entry_ns": te,
                "entry_px": px0,
                "exit_ns": xt,
                "exit_px": xpx,
                "reason": reason,
                "net_usd": cash,
                "net_pct": cash / (px0 * 100 * n) * 100,
            }
        )
        k = int(np.searchsorted(t, xt or cand[reason], side="right"))
    return trades


def calibration_window() -> tuple[dt.date, dt.date]:
    """D-033：成交参数的校准月份（config/calibration_window.yaml），规则测试不得使用。"""
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "calibration_window.yaml").read_text(
            encoding="utf-8"
        )
    )
    return dt.date.fromisoformat(cfg["from"]), dt.date.fromisoformat(cfg["to"])


def check_out_of_sample(days: list[dt.date]) -> None:
    lo, hi = calibration_window()
    bad = [d for d in days if lo <= d <= hi]
    if bad:
        raise ValueError(
            f"D-033：{len(bad)} 个测试日落在成交参数校准窗口 {lo}～{hi} 内（如 {bad[0]}），"
            "规则测试只能用其他月份"
        )


def run(root: Path, rule: dict, days: list[dt.date], seed: int = 0) -> pd.DataFrame:
    check_out_of_sample(days)
    bs = BarState(root, rule["underlying"], max(days), rule.get("convention", "C"))
    od = OptionData(root, rule["underlying"])
    rows = []
    for day in days:
        rows += run_day(root, rule, day, bs, od, seed)
    return pd.DataFrame(rows)


def replay(
    root: Path,
    und: str,
    legs: list[dict],
    q: float = 1.0,
    pi: PriceImprovement | None = None,
    seed: int = 0,
    cache: dict | None = None,
) -> list[dict]:
    """已知答案重放：legs = [{symbol, day, side(+1/−1), qty, order}]，按给定订单在 tick 上成交。
    cache：{(symbol, day): Book}，多次重放时复用。"""
    od = OptionData(root, und)
    cache = {} if cache is None else cache
    out = []
    rng = np.random.default_rng(seed)
    for lg in legs:
        o = lg["order"]
        t0 = int(o["path"][0][0]) if o["mode"] == "path" else int(lg["t_ns"])
        end = int(o.get("end_ns", t0)) + 120 * NS
        key = (lg["symbol"], lg["day"], t0, end)
        if key not in cache:
            cache[key] = od.book(
                lg["symbol"],
                lg["day"],
                pd.Timestamp(t0, tz="UTC") - pd.Timedelta(minutes=1),
                pd.Timestamp(end, tz="UTC"),
            )
        bk = cache[key]
        if bk is None:
            out.append(
                {**{k: v for k, v in lg.items() if k != "order"}, "filled": 0.0, "px": math.nan}
            )
            continue
        info: dict = {}
        n, px, tl, _ = place(bk, lg["side"], lg["qty"], t0, o, lg["symbol"], q, None, rng, pi, info)
        out.append(
            {
                **{k: v for k, v in lg.items() if k != "order"},
                "filled": n,
                "px": px,
                "fill_ns": tl,
                "forced": float(info.get("forced", 0.0)),
            }
        )
    return out
