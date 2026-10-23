"""影子回放（audit/base-final 第 5 节）：录制器逐笔数据驱动增量计算路径
（zr/strategy/incremental.py），MU 每根 K 的状态 / 信号与每笔入场 / 离场，与批量回测逐根、逐笔对比。

数据：录制器（VPS 实时推送，$ZR_DATA_ROOT/recorder/date=D/underlying=MU）——MU.US 逐笔成交、
订阅期权的 1 档盘口（depth）。时间一律用接收时刻 recv_ts（实盘能看到的时刻）。
回放窗口：录制器开始后第一个完整 2 分钟分箱起，到 16:00 ET；2026-10-21 从 14:48 起（录制器
14:47 才启动），2026-10-22 从 09:26 起。窗口之前的状态用长桥官方 2 分钟 K（TradeSessions.All）
从 2026-06-01 起逐根灌入（口径 C 连续、不重置）。

两条路径：
- 影子（实时）：逐笔成交 → BarBuilder（只有常规成交 trade_type="" 计价，全部成交计量；
  分箱在下一分箱的成交到达或时钟越过终点 + 1 s 时收线）→ IncrementalEngine；
  信号在收线那一刻（接收时刻）决策，+2 s 后第一个盘口 ask 买入；持仓中逐个盘口检查
  +20% / −30%，收线时检查结构离场，30 分钟上限、强平按时钟；离场 = 触发 +2 s 后第一个盘口 bid。
- 批量（回测）：官方 2 分钟 K → zr/features/regime + v4_decisive 向量化信号；决策 = K 收盘时刻；
  v4_decisive.simulate 用同一份录制器盘口（按接收时刻）成交。
合约：v4_decisive.pick_contract，期权链 = 当日录制器订阅的期权代码（最近 2 个到期日、平值 ±8 档）。

运行：uv run python -m zr.research.shadow_replay
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.thetadata import parse_contract
from zr.features.bars import ET
from zr.features.regime import bars_per_day, indicators, regimes
from zr.research import v4_decisive as v4d
from zr.strategy.incremental import Bar, BarBuilder, IncrementalEngine

log = logging.getLogger("zr.research.shadow_replay")
WINDOWS = {dt.date(2026, 10, 21): "14:48", dt.date(2026, 10, 22): "09:26"}
SEED_FROM = pd.Timestamp("2026-06-01", tz="UTC")
HYPS = ("H1", "H2", "H3")


def price_ok(trade_type: str) -> bool:
    return trade_type == ""


# ---------------- 数据 ----------------


def load_official(root: Path) -> pd.DataFrame:
    o = pd.read_parquet(root / "diag" / "official_bars" / "MU_2m_all.parquet")
    o["ts"] = pd.to_datetime(o["ts"], utc=True)
    return o.drop_duplicates("ts", keep="last").set_index("ts").sort_index()


def load_recorder(root: Path, day: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "recorder" / f"date={day}" / "underlying=MU"
    tr = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(base / "trade_*.parquet")))])
    tr = tr[tr["symbol"] == "MU.US"].sort_values("recv_ts_utc_ns", kind="stable")
    rows = []
    for rv, p in zip(tr["recv_ts_utc_ns"], tr["payload"], strict=True):
        for x in json.loads(p)["trades"]:
            t = int(pd.Timestamp(x["timestamp"], tz="UTC").value)  # SDK 时间为 naive UTC
            rows.append((int(rv), t, float(x["price"]), float(x["volume"]), x["trade_type"]))
    trades = pd.DataFrame(rows, columns=["recv", "t", "px", "vol", "tt"])
    dp = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(base / "depth_*.parquet")))])
    dp = dp[dp["symbol"] != "MU.US"].sort_values("recv_ts_utc_ns", kind="stable")
    q = []
    for rv, sym, p in zip(dp["recv_ts_utc_ns"], dp["symbol"], dp["payload"], strict=True):
        d = json.loads(p)
        b = d["bids"][0]["price"] if d.get("bids") else None
        a = d["asks"][0]["price"] if d.get("asks") else None
        q.append(
            (int(rv), sym, np.nan if b is None else float(b), np.nan if a is None else float(a))
        )
    quotes = pd.DataFrame(q, columns=["recv", "symbol", "bid", "ask"])
    return trades, quotes


def chain_from(quotes: pd.DataFrame) -> dict:
    """录制器订阅的期权代码 → v4_decisive.day_chain 同格式。"""
    rows = [{"symbol": s, **parse_contract(s)} for s in quotes["symbol"].unique()]
    c = pd.DataFrame(rows)
    out = {}
    for (e, r), g in c.groupby(["expiry", "right"]):
        g = g.sort_values("strike")
        out[(e, r)] = (g["strike"].to_numpy(float), g["symbol"].tolist())
    return out


def valid(bid: float, ask: float) -> bool:
    return np.isfinite(bid) and np.isfinite(ask) and ask > 0 and bid >= 0 and ask >= bid


# ---------------- 影子路径的持仓 ----------------


@dataclass
class Position:
    cell: tuple
    symbol: str
    decision: int
    flat: int
    lag: int
    tp: float
    sl: float
    cap: int
    qty: int
    entry: int | None = None
    ask0: float = np.nan
    t_tp: int | None = None
    tp_fill: tuple | None = None
    trig: tuple | None = None  # (原因, 触发时刻)
    exit_fill: tuple | None = None
    last_quote: tuple | None = field(default=None, repr=False)

    def on_quote(self, recv: int, bid: float, ask: float) -> None:
        if not valid(bid, ask):
            return
        self.last_quote = (recv, bid)
        if self.entry is None:
            if recv >= self.decision + self.lag:
                if recv >= self.flat:
                    self.trig = ("no_entry", recv)
                    self.exit_fill = (recv, np.nan)
                    return
                self.entry, self.ask0 = recv, ask
            return
        mid = (bid + ask) / 2
        if recv > self.entry and self.trig is None:
            if self.t_tp is None and mid >= self.ask0 * (1 + self.tp):
                self.t_tp = recv
            if mid <= self.ask0 * (1 - self.sl):
                self.trig = ("sl", recv)
        if self.t_tp is not None and self.tp_fill is None and recv >= self.t_tp + self.lag:
            self.tp_fill = (recv, bid)
        if self.trig is not None and self.exit_fill is None and recv >= self.trig[1] + self.lag:
            self.exit_fill = (recv, bid)

    def on_clock(self, now: int) -> None:
        if self.entry is None or self.trig is not None:
            return
        for reason, t in (("cap", self.entry + self.cap), ("flat", self.flat)):
            if now >= t:
                self.trig = (reason, t)
                return

    def on_structure(self, now: int) -> None:
        if self.entry is not None and self.trig is None:
            self.trig = ("structure", now)

    def done(self) -> bool:
        return self.exit_fill is not None

    def finish(self) -> None:
        """数据结束仍未离场：用最后一个有效盘口（与批量回放「之后再无报价用最后一个」一致）。"""
        if self.entry is not None and self.exit_fill is None and self.last_quote is not None:
            self.exit_fill = self.last_quote
            if self.trig is None:
                self.trig = ("end", self.last_quote[0])


def net_usd(p: Position) -> float:
    from zr.research.sim_v3 import _BUY, sell_fee

    cash = -p.ask0 * 100 * p.qty - _BUY[p.qty]
    left = p.qty
    used = p.t_tp is not None and p.trig is not None and p.t_tp < p.trig[1]
    if used and p.tp_fill is not None:
        b = p.tp_fill[1]
        cash += b * 100 - (float(sell_fee(1, b)) if b > 0 else 0.0)
        left -= 1
    b = p.exit_fill[1]
    cash += left * b * 100 - (float(sell_fee(left, b)) if b > 0 else 0.0)
    return cash


def window_start(root: Path, day: dt.date) -> pd.Timestamp:
    """回放起点：WINDOWS 里登记过的日子用登记值；否则 = 录制器当日第一笔 MU.US 成交的接收时刻
    向上取整到 2 分钟分箱（第一个完整分箱）。"""
    if day in WINDOWS:
        return pd.Timestamp(f"{day} {WINDOWS[day]}", tz=ET).tz_convert("UTC")
    base = root / "recorder" / f"date={day}" / "underlying=MU"
    first = min(
        int(
            pd.read_parquet(f, columns=["recv_ts_utc_ns", "symbol"])
            .query("symbol == 'MU.US'")["recv_ts_utc_ns"]
            .min()
        )
        for f in sorted(glob.glob(str(base / "trade_*.parquet")))[:5]
    )
    p = 120_000_000_000
    return pd.Timestamp(first - first % p + p, tz="UTC")


def shadow_day(root: Path, day: dt.date, v4: dict, cfg: dict, official: pd.DataFrame):
    from zr.data.calendar import session_close

    start = window_start(root, day)
    end = pd.Timestamp(dt.datetime.combine(day, session_close(day)), tz=ET).tz_convert("UTC")
    seed = official[(official.index >= SEED_FROM) & (official.index < start)]
    w = max(1, round(bars_per_day(seed)))
    eng = IncrementalEngine(cfg, v4, w, {day})
    for ts, o, h, lo, c, vol in zip(
        v4d.ns(seed.index),
        seed["open"],
        seed["high"],
        seed["low"],
        seed["close"],
        seed["volume"],
        strict=True,
    ):
        eng.on_bar(Bar(int(ts), o, h, lo, c, vol))
    trades, quotes = load_recorder(root, day)
    trades = trades[(trades["t"] >= start.value) & (trades["t"] < end.value)]
    chain = chain_from(quotes)
    exe, prem = v4["execution"], v4["premium"]
    lag = int(exe["lag_s"] * 1e9)
    bb = BarBuilder(v4["period"], price_ok)
    ev = pd.concat(
        [
            trades.assign(kind=0)[["recv", "kind", "t", "px", "vol", "tt"]],
            quotes.assign(kind=1)[["recv", "kind", "symbol", "bid", "ask"]],
        ],
        ignore_index=True,
    ).sort_values(["recv", "kind"], kind="stable")
    recs, positions, open_pos, busy = [], [], {}, {}
    exit_rules = {(h, xk): v4["exits"][xk][h] for h in HYPS for xk in ("a", "b", "c")}

    def on_bar(bar: Bar, now: int) -> None:
        if bar.ts >= end.value:
            return
        rec = eng.on_bar(bar)
        rec["completed_ns"] = now
        recs.append(rec)
        for key, pos in list(open_pos.items()):  # 结构离场：收线时刻触发
            h, xk, d = key
            if rec["ex"][exit_rules[(h, xk)]][0 if d == 1 else 1]:
                pos.on_structure(now)
        for h in HYPS:
            d = rec[h]
            if d == 0 or not (rec["ok_long"] if d == 1 else rec["ok_short"]):
                continue
            close_et = pd.Timestamp(bar.ts, tz="UTC").tz_convert(ET) + pd.Timedelta(minutes=2)
            pk = v4d.pick_contract(chain, day, d, bar.close, close_et.time(), v4)
            if pk is None:
                continue
            flat_t = v4d.day_limits(day, exe)[1]
            if pk["dte"] == 0:
                flat_t = min(flat_t, v4d._hm(exe["d001_flat_0dte"]))
            flat = (v4d.et_ts(day, flat_t) - pd.Timedelta(seconds=exe["flat_lead_s"])).value
            if now >= flat:
                continue
            for xk in ("a", "b", "c"):
                key = (h, xk, d)
                if key in open_pos or now < busy.get(key, -1):
                    continue
                pos = Position(
                    key,
                    pk["symbol"],
                    now,
                    flat,
                    lag,
                    prem["tp"],
                    prem["sl"],
                    int(v4["cap_min"] * 60e9),
                    exe["qty"],
                )
                open_pos[key] = pos
                positions.append(pos)

    for r in ev.itertuples(index=False):
        now = int(r.recv)
        for bar in bb.on_clock(now):
            on_bar(bar, now)
        for pos in open_pos.values():
            pos.on_clock(now)
        if r.kind == 0:
            for bar in bb.on_trade(int(r.t), float(r.px), float(r.vol), r.tt):
                on_bar(bar, now)
        else:
            for pos in open_pos.values():
                if pos.symbol == r.symbol:
                    pos.on_quote(now, float(r.bid), float(r.ask))
        for key in [k for k, p in open_pos.items() if p.done()]:
            busy[key] = open_pos.pop(key).exit_fill[0]
    for bar in bb.close():
        on_bar(bar, int(ev["recv"].max()))
    for p in open_pos.values():
        p.finish()
    inc = pd.DataFrame([{k: v for k, v in x.items() if k != "ex"} for x in recs])
    inc["ts"] = pd.to_datetime(inc["ts"], utc=True)
    rows = []
    for p in positions:
        if p.entry is None or p.trig is None or p.trig[0] == "no_entry":
            continue
        rows.append(
            {
                "hyp": p.cell[0],
                "exit": p.cell[1],
                "dir": "long" if p.cell[2] == 1 else "short",
                "decision_ns": p.decision,
                "symbol": p.symbol,
                "entry_ns": p.entry,
                "ask0": p.ask0,
                "exit_ns": p.exit_fill[0],
                "reason": p.trig[0],
                "tp_used": p.t_tp is not None and p.t_tp < p.trig[1],
                "net_usd": net_usd(p),
            }
        )
    return inc.set_index("ts"), pd.DataFrame(rows), bb.late, quotes, chain, start, end


# ---------------- 批量路径 ----------------


def batch_day(day, v4, cfg, official, quotes, chain, start, end):
    b = official[(official.index >= SEED_FROM) & (official.index < end)]
    x = indicators(b, cfg, daily_reset=False, warmup="nan")
    reg = regimes(b, x, cfg, v4["period"])
    sig = v4d.hypothesis_signals(b, x, reg, cfg, v4)
    exits = v4d.exit_arrays(b, x, cfg)
    time_ok, ok_l, ok_s = v4d.guard_mask(b, reg, v4, {day})
    cns = v4d.ns(b.index + pd.Timedelta(minutes=v4["period"]))
    win = np.asarray(b.index >= start)
    per_bar = pd.DataFrame(
        {"state": reg["state"].to_numpy(), "dir": reg["dir"].to_numpy(), **sig}, index=b.index
    )[win]
    exe = v4["execution"]
    rows = []
    ticks = {}
    for h in HYPS:
        for xk in ("a", "b", "c"):
            rule = v4["exits"][xk][h]
            for d in (1, -1):
                trig = cns[exits[rule][0 if d == 1 else 1]]
                cand = np.flatnonzero(win & (sig[h] == d) & time_ok & (ok_l if d == 1 else ok_s))
                last = -1
                for i in cand:
                    dec = int(cns[i])
                    if dec < last:
                        continue
                    close_et = (b.index[i] + pd.Timedelta(minutes=2)).tz_convert(ET)
                    pk = v4d.pick_contract(
                        chain, day, d, float(b["close"].iloc[i]), close_et.time(), v4
                    )
                    if pk is None:
                        continue
                    flat_t = v4d.day_limits(day, exe)[1]
                    if pk["dte"] == 0:
                        flat_t = min(flat_t, v4d._hm(exe["d001_flat_0dte"]))
                    flat = (v4d.et_ts(day, flat_t) - pd.Timedelta(seconds=exe["flat_lead_s"])).value
                    if dec >= flat:
                        continue
                    sym = pk["symbol"]
                    if sym not in ticks:
                        g = quotes[quotes["symbol"] == sym]
                        g = g[(g["ask"] > 0) & (g["bid"] >= 0) & (g["ask"] >= g["bid"])]
                        ticks[sym] = v4d.Ticks(
                            g["recv"].to_numpy(np.int64),
                            g["bid"].to_numpy(float),
                            g["ask"].to_numpy(float),
                        )
                    res = v4d.simulate(ticks[sym], dec, trig, flat, v4)
                    if res is None:
                        continue
                    last = res["exit_ns"]
                    rows.append(
                        {
                            "hyp": h,
                            "exit": xk,
                            "dir": "long" if d == 1 else "short",
                            "decision_ns": dec,
                            "symbol": sym,
                            **{
                                k: res[k]
                                for k in (
                                    "entry_ns",
                                    "ask0",
                                    "exit_ns",
                                    "reason",
                                    "tp_used",
                                    "net_usd",
                                )
                            },
                        }
                    )
    return per_bar, pd.DataFrame(rows)


def compare_trades(sh: pd.DataFrame, ba: pd.DataFrame) -> pd.DataFrame:
    """按格子与「信号 K」配对：影子的决策时刻 = 收线接收时刻（≥ K 收盘），批量 = K 收盘。"""
    if sh.empty and ba.empty:
        return pd.DataFrame()
    sh = sh.copy()
    ba = ba.copy()
    sh["bar_close"] = (sh["decision_ns"] // 120_000_000_000) * 120_000_000_000
    ba["bar_close"] = ba["decision_ns"]
    k = ["hyp", "exit", "dir", "bar_close"]
    j = ba.merge(sh, on=k, how="outer", suffixes=("_batch", "_shadow"), indicator=True)
    j["decision_lag_s"] = (j["decision_ns_shadow"] - j["decision_ns_batch"]) / 1e9
    j["entry_lag_s"] = (j["entry_ns_shadow"] - j["entry_ns_batch"]) / 1e9
    j["exit_lag_s"] = (j["exit_ns_shadow"] - j["exit_ns_batch"]) / 1e9
    j["net_diff"] = j["net_usd_shadow"] - j["net_usd_batch"]
    return j


def run(root: Path, days=None) -> dict:
    v4 = v4d.load_v4()
    cfg = v4d.state_cfg(v4, "C")
    official = load_official(root)
    out = {}
    for day in days or WINDOWS:
        inc, sh, late, quotes, chain, start, end = shadow_day(root, day, v4, cfg, official)
        per_bar, ba = batch_day(day, v4, cfg, official, quotes, chain, start, end)
        j = inc.join(
            official[["open", "high", "low", "close", "volume"]], rsuffix="_o", how="left"
        ).join(per_bar, rsuffix="_batch", how="left")
        out[day] = {
            "bars": j,
            "shadow": sh,
            "batch": ba,
            "trades": compare_trades(sh, ba),
            "late": late,
            "expiries": sorted({e for (e, _) in chain}),
        }
        log.info("%s：K %d，影子交易 %d，批量交易 %d", day, len(inc), len(sh), len(ba))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    res = run(root)
    out = root / "research" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(res, out / "shadow_replay.pkl")
    for day, r in res.items():
        j = r["bars"]
        px = np.logical_and.reduce(
            [np.isclose(j[c], j[f"{c}_o"]) for c in ("open", "high", "low", "close")]
        )
        print(
            day,
            "K",
            len(j),
            "OHLC = 官方",
            int(px.sum()),
            "量 = 官方",
            int((j["volume"] == j["volume_o"]).sum()),
            "状态一致",
            int((j["state"] == j["state_batch"]).sum()),
            {
                h: (
                    int((j[h] != 0).sum()),
                    int((j[f"{h}_batch"] != 0).sum()),
                    int((j[h] != j[f"{h}_batch"]).sum()),
                )
                for h in HYPS
            },
            "迟到成交",
            r["late"],
            "到期日",
            r["expiries"],
        )
        print(
            r["trades"]
            .drop(columns=[c for c in r["trades"].columns if c.startswith("_")], errors="ignore")
            .to_string()
            if len(r["trades"])
            else "无交易"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
