"""V4 决定性检验（RESEARCH_LOG R-033，预登记；参数全部来自 config/v4_decisive.yaml）。

H1 E10 顺点火、H2 E1 通道回踩、H3 E11 突破延迟一根；每个假设 3 种离场 × 2 方向 × 2 标的，
0/1 DTE 合约、tick 级 NBBO 成交（负责人采纳 STATUS 0e）。主检验用口径 C，敏感性用口径 A（D-028）。

流程：
1. 口径 → 2 分钟 K → 状态层 → 三个假设的入场 K（含护栏）。
2. 每笔事件按合约规则选合约；随机对照候选（同日、同方向、±15 分钟内其余满足护栏的 K）同样选合约。
3. 按「合约-日」合并所需时间窗，用 ThetaData interval=tick 拉逐笔 NBBO（缓存，重跑不再请求）。
4. tick 级回放：入场 = 信号 K 收盘 + 2 s 后第一个有效 ask；结构离场 = 触发 K 收盘 + 2 s 后第一个
   有效 bid；权利金规则用 tick mid 判定、+2 s 后第一个有效 bid 成交；30 分钟上限；强平。
5. 每格（假设 × 离场 × 方向 × 标的）按入场时间贪心（持仓中不再开仓），统计、随机对照、BH-FDR。

运行（需本机 Theta Terminal）：uv run python -m zr.research.v4_decisive [--conv C|A|both]
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.config import get_settings
from zr.data.calendar import is_trading_day, session_close
from zr.data.thetadata import ThetaClient
from zr.features.bars import ET, load_bars, load_extended_bars, resample_bars
from zr.features.option_features import option_days
from zr.features.regime import IGNITION, RANGE, indicators, load_config, regimes
from zr.research import exits as ex
from zr.research import patterns_v3 as pt
from zr.research.sim_v3 import _BUY, sell_fee

log = logging.getLogger("zr.research.v4_decisive")
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config" / "v4_decisive.yaml"
EXIT_KEYS = ("a", "b", "c")
HYPS = ("H1", "H2", "H3")


def load_v4(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _hm(s: str) -> dt.time:
    h, m = map(int, s.split(":"))
    return dt.time(h, m)


def et_ts(day: dt.date, t: dt.time) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, t), tz=ET).tz_convert("UTC")


# ---------------- 样本日 ----------------


def sample_days(root: Path, und: str, v4: dict) -> list[dt.date]:
    s = v4["sample"]
    a0, a1 = dt.date.fromisoformat(s["all_days_from"]), dt.date.fromisoformat(s["all_days_to"])
    t0, t1 = dt.date.fromisoformat(s["thu_fri_from"]), dt.date.fromisoformat(s["thu_fri_to"])
    out = []
    for d in option_days(root, und):
        if not is_trading_day(d):
            continue
        if a0 <= d <= a1 or (t0 <= d <= t1 and d.weekday() in (3, 4)):
            out.append(d)
    return out


# ---------------- 状态层与信号 ----------------


def state_cfg(v4: dict, conv: str) -> dict:
    c = copy.deepcopy(load_config())
    c["boll"] = dict(v4["boll"])
    c["macd"] = dict(v4["macd"])
    c["regime"]["range"]["score_threshold"] = v4["conventions"][conv]["range_threshold"]
    return c


def build_bars(root: Path, und: str, conv: dict, end: dt.date) -> pd.DataFrame:
    if conv["ext"]:
        s = load_extended_bars(root, und, end=end, clean=conv.get("clean", True))
    else:
        s = load_bars(root, und, end=end)
    return resample_bars(s, "2min")


def e11_delayed(df: pd.DataFrame, x: pd.DataFrame, reg: pd.DataFrame, h: dict) -> np.ndarray:
    """E11 突破（前一根 RANGE、收盘破前 M 根高/低点、量比 ≥ v）后，下一根（同日）收盘仍在突破
    水位之外才入场；信号落在确认那根 K 上。"""
    raw = pt.entry_signal(
        df,
        x,
        reg,
        {"entry": "E11", "M": h["M"], "v": h["v"]},
        {"entries": {"E11": {"vol_window": h["vol_window"]}}},
    )
    hi, lo, cl = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    hh = pd.Series(hi).shift(1).rolling(h["M"], min_periods=h["M"]).max().to_numpy()
    ll = pd.Series(lo).shift(1).rolling(h["M"], min_periods=h["M"]).min().to_numpy()
    day = np.asarray(df.index.tz_convert(ET).date)
    k = int(h.get("confirm_bars", 1))
    out = np.zeros(len(df), dtype=np.int8)
    for i in np.flatnonzero(raw != 0):
        j = i + k
        if j >= len(df) or day[j] != day[i]:
            continue
        if raw[i] == 1 and np.all(cl[i + 1 : j + 1] > hh[i]):
            out[j] = 1
        elif raw[i] == -1 and np.all(cl[i + 1 : j + 1] < ll[i]):
            out[j] = -1
    return out


def hypothesis_signals(df, x, reg, cfg, v4) -> dict[str, np.ndarray]:
    h = v4["hypotheses"]
    e1 = h["H2"]
    return {
        "H1": pt.entry_signal(df, x, reg, {"entry": "E10", "k": h["H1"]["k"]}, cfg),
        "H2": pt.entry_signal(
            df,
            x,
            reg,
            {"entry": "E1", "pctb": tuple(e1["pctb"]), "c": e1["c"], "N": e1["N"], "M": e1["M"]},
            cfg,
        ),
        "H3": e11_delayed(df, x, reg, h["H3"]),
    }


def exit_arrays(df, x, cfg) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = ex.exit_signals(df, x, cfg)
    pctb = x["pctb"].to_numpy()
    with np.errstate(invalid="ignore"):
        out["X4_inside_0.1_0.9"] = (pctb < 0.9, pctb > 0.1)  # 多头：%b 回到 0.9 以下；空头镜像
    return out


def day_limits(day: dt.date, exe: dict) -> tuple[dt.time, dt.time]:
    """(最后入场 K 收盘, 清仓) 时刻：提前收盘日按收盘前同样的间隔前移（正常日 15:40 / 15:50）。"""
    close = dt.datetime.combine(day, session_close(day))
    full = dt.datetime.combine(day, dt.time(16, 0))
    last = dt.datetime.combine(day, _hm(exe["last_entry"])) - (full - close)
    flat = dt.datetime.combine(day, _hm(exe["flat"])) - (full - close)
    return last.time(), flat.time()


def guard_mask(df: pd.DataFrame, reg: pd.DataFrame, v4: dict, days: set) -> tuple:
    """可入场 K（与方向无关的部分）与按方向的状态护栏。返回 (time_ok, ok_long, ok_short)。"""
    exe, g = v4["execution"], v4["guardrails"]
    et = df.index.tz_convert(ET)
    label = np.asarray(et.strftime("%H:%M"))
    close = et + pd.Timedelta(minutes=v4["period"])
    close_hm = np.asarray(close.strftime("%H:%M"))
    day = np.asarray(et.date)
    last = {d: day_limits(d, exe)[0].strftime("%H:%M") for d in set(day) if d in days}
    last_hm = np.array([last.get(d, "00:00") for d in day])
    time_ok = (label >= exe["first_entry_bar"]) & (close_hm <= last_hm) & np.isin(day, list(days))
    st, d = reg["state"].to_numpy(), reg["dir"].to_numpy()
    ok_l = np.ones(len(df), bool)
    ok_s = np.ones(len(df), bool)
    if g["no_range_entry"]:
        ok_l &= st != RANGE
        ok_s &= st != RANGE
    if g["no_counter_ignition"]:
        ok_l &= ~((st == IGNITION) & (d == -1))
        ok_s &= ~((st == IGNITION) & (d == 1))
    return time_ok, ok_l, ok_s


# ---------------- 合约 ----------------


def day_chain(root: Path, und: str, day: dt.date) -> dict:
    """{(expiry, right): (strikes 升序, symbols)}，来自当日期权链数据（规则 7）。"""
    p = root / "options_1m" / f"underlying={und}" / f"date={day.isoformat()}" / "quotes.parquet"
    if not p.exists():
        return {}
    q = pd.read_parquet(p, columns=["symbol", "expiry", "strike", "right"])
    q = q.drop_duplicates("symbol")
    out = {}
    for (e, r), g in q.groupby(["expiry", "right"]):
        g = g.sort_values("strike")
        out[(e, r)] = (g["strike"].to_numpy(float), g["symbol"].tolist())
    return out


def pick_contract(
    chain: dict, day: dt.date, direction: int, spot: float, decision_et: dt.time, v4: dict
) -> dict | None:
    """DTE（日历日）∈ [0, max_dte] 中最近的到期日（10:15 后排除 0DTE）、略虚值 otm_steps 档。"""
    c, exe = v4["contract"], v4["execution"]
    right = "call" if direction == 1 else "put"
    exps = sorted({e for (e, r) in chain if r == right and 0 <= (e - day).days <= c["max_dte"]})
    if decision_et > _hm(exe["d001_last_0dte_entry"]):
        exps = [e for e in exps if e != day]
    if not exps:
        return None
    exp = exps[0]
    s, syms = chain[(exp, right)]
    k = c["otm_steps"] - 1
    j = (
        int(np.searchsorted(s, spot, side="right")) + k
        if direction == 1
        else int(np.searchsorted(s, spot, side="left")) - 1 - k
    )
    if not 0 <= j < len(s):
        return None
    return {
        "symbol": syms[j],
        "expiry": exp,
        "strike": float(s[j]),
        "right": right,
        "dte": (exp - day).days,
    }


def ns(x) -> np.ndarray:
    """时间戳 → int64 纳秒（pandas 默认微秒精度，统一换成纳秒，避免单位混用）。"""
    return pd.DatetimeIndex(x).as_unit("ns").asi8


# ---------------- 事件 ----------------


def build_events(
    root: Path, und: str, conv_key: str, v4: dict, sig_fn=None, hyps=HYPS, days: list | None = None
) -> tuple[pd.DataFrame, dict]:
    """返回事件表（每行 = 一个可入场 K × 方向 × 所选合约，含实际信号与随机候选）与离场触发。

    列：bar_i、day、decision_ts（K 收盘，UTC）、direction、spot、symbol…、flat_ts、
    hyp_H1/H2/H3（该 K 是否为该假设的信号）。随机候选 = 全部满足护栏的 K × 两个方向。
    """
    conv = v4["conventions"][conv_key]
    cfg = state_cfg(v4, conv_key)
    days = sample_days(root, und, v4) if days is None else days
    df = build_bars(root, und, conv, max(days))
    x = indicators(df, cfg, daily_reset=conv["reset"], warmup=conv["warmup"])
    reg = regimes(df, x, cfg, v4["period"])
    if sig_fn is None:
        raw = hypothesis_signals(df, x, reg, cfg, v4)
        sig = {h: (raw[h] == 1, raw[h] == -1) for h in hyps}
    else:  # {名称: (多头布尔, 空头布尔)}；同一根 K 两个方向可以同时为真
        sig = sig_fn(df, x, reg, cfg, conv)
    exits = exit_arrays(df, x, cfg)
    time_ok, ok_l, ok_s = guard_mask(df, reg, v4, set(days))
    et = df.index.tz_convert(ET)
    close_utc = df.index + pd.Timedelta(minutes=v4["period"])
    close_et = close_utc.tz_convert(ET)
    day = np.asarray(et.date)
    spot = df["close"].to_numpy(float)
    exe = v4["execution"]
    lead = pd.Timedelta(seconds=exe["flat_lead_s"])
    w = pd.Timedelta(minutes=v4["random"]["window_min"]).value
    cns = ns(close_utc)
    rows = []
    chains: dict = {}
    for direction, ok in ((1, ok_l), (-1, ok_s)):
        elig = time_ok & ok
        is_sig = {h: elig & sig[h][0 if direction == 1 else 1] for h in hyps}
        any_sig = np.flatnonzero(np.logical_or.reduce(list(is_sig.values())))
        need = np.zeros(len(df), bool)
        for i in any_sig:  # 实际信号 K 及其 ±window 内的随机候选（同日、同方向、满足护栏）
            lo = np.searchsorted(cns, cns[i] - w, side="left")
            hi = np.searchsorted(cns, cns[i] + w, side="right")
            need[lo:hi] |= elig[lo:hi] & (day[lo:hi] == day[i])
        for i in np.flatnonzero(need):
            d = day[i]
            if d not in chains:
                chains[d] = day_chain(root, und, d)
            pk = pick_contract(chains[d], d, direction, spot[i], close_et[i].time(), v4)
            if pk is None:
                continue
            flat_t = day_limits(d, exe)[1]
            if pk["dte"] == 0:
                flat_t = min(flat_t, _hm(exe["d001_flat_0dte"]))
            flat = et_ts(d, flat_t) - lead
            if close_utc[i] >= flat:
                continue
            rows.append(
                {
                    "bar_i": int(i),
                    "day": d,
                    "decision_ts": close_utc[i],
                    "direction": direction,
                    "spot": spot[i],
                    **pk,
                    "flat_ts": flat,
                    **{f"hyp_{h}": bool(is_sig[h][i]) for h in hyps},
                }
            )
    ev = pd.DataFrame(rows)
    trig = {}
    for rule in exits:
        lg, sh = exits[rule]
        trig[rule] = {1: cns[lg], -1: cns[sh]}
    meta = {
        "n_bars": len(df),
        "days": days,
        "signals": {h: int(sig[h][0].sum() + sig[h][1].sum()) for h in hyps},
    }
    return ev, {"trig": trig, "meta": meta}


def needed_windows(ev: pd.DataFrame, v4: dict) -> pd.DataFrame:
    """每行事件需要 [K 收盘 − pre, + post]；按合约-日合并重叠时间段。"""
    t = v4["ticks"]
    pre, post = pd.Timedelta(minutes=t["pre_min"]), pd.Timedelta(minutes=t["post_min"])
    out = []
    for (sym, d), g in ev.groupby(["symbol", "day"]):
        iv = sorted(zip(g["decision_ts"] - pre, g["decision_ts"] + post, strict=True))
        cur_s, cur_e = iv[0]
        for s, e in iv[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                out.append((sym, d, cur_s, cur_e))
                cur_s, cur_e = s, e
        out.append((sym, d, cur_s, cur_e))
    w_df = pd.DataFrame(out, columns=["symbol", "day", "start", "end"])
    info = ev.drop_duplicates("symbol").set_index("symbol")[["expiry", "strike", "right"]]
    return w_df.join(info, on="symbol")


# ---------------- tick 拉取与缓存 ----------------


def tick_path(root: Path, und: str, sym: str, d: dt.date, s: pd.Timestamp, e: pd.Timestamp):
    base = root / "diag" / "v4_ticks" / f"underlying={und}" / f"date={d.isoformat()}"
    se = f"{s.tz_convert(ET).strftime('%H%M%S')}-{e.tz_convert(ET).strftime('%H%M%S')}"
    return base / f"{sym}_{se}.parquet"


def _cached_intervals(root: Path, und: str, sym: str, d: dt.date) -> list[tuple[int, int]]:
    base = root / "diag" / "v4_ticks" / f"underlying={und}" / f"date={d.isoformat()}"
    out = []
    for f in base.glob(f"{sym}_*.parquet"):
        a, b = f.stem.rsplit("_", 1)[1].split("-")
        to = [et_ts(d, dt.time(int(x[:2]), int(x[2:4]), int(x[4:6]))) for x in (a, b)]
        out.append((to[0].value, to[1].value))
    return sorted(out)


def uncovered(start: pd.Timestamp, end: pd.Timestamp, have: list[tuple[int, int]]):
    """[start, end] 去掉已缓存区间后剩下的段（纳秒，按秒取整）。"""
    segs, cur = [], start.value
    for a, b in have:
        if b <= cur or a >= end.value:
            continue
        if a > cur:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < end.value:
        segs.append((cur, end.value))
    return [
        (pd.Timestamp(a, tz="UTC").floor("s"), pd.Timestamp(b, tz="UTC").ceil("s"))
        for a, b in segs
        if b - a >= 1e9
    ]


def pull_ticks(root: Path, und: str, windows: pd.DataFrame, concurrency: int = 4) -> dict:
    """按合约-日拉逐笔 NBBO；已缓存覆盖的时间段不再请求。"""
    import threading

    client = ThetaClient(concurrency=concurrency)
    stats = {"segments": 0, "pulled": 0, "failed": 0, "empty": 0}
    lock = threading.Lock()

    def bump(k):
        with lock:
            stats[k] += 1

    jobs = []
    for r in windows.itertuples(index=False):
        have = _cached_intervals(root, und, r.symbol, r.day)
        for a, b in uncovered(r.start, r.end, have):
            jobs.append((r, a, b))
    stats["segments"] = len(jobs)

    def one(job):
        r, a, b = job
        p = tick_path(root, und, r.symbol, r.day, a, b)
        try:
            df = client.history_quote(
                und,
                r.expiry,
                r.day,
                interval="tick",
                strike=f"{r.strike:.3f}",
                right=r.right,
                start_time=a.tz_convert(ET).strftime("%H:%M:%S.000"),
                end_time=b.tz_convert(ET).strftime("%H:%M:%S.000"),
            )
        except Exception as e:  # noqa: BLE001 - 单个合约失败不中断，计数并报告
            log.warning("tick 拉取失败 %s %s：%s", r.symbol, r.day, e)
            bump("failed")
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ("ts", "bid", "ask") if c in df.columns]
        empty = pd.DataFrame(columns=["ts", "bid", "ask"])
        (df[cols] if cols else empty).to_parquet(p, index=False)
        bump("pulled" if cols else "empty")

    log.info("%s：%d 个时间窗，需请求 %d 段", und, len(windows), len(jobs))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, jobs))
    return stats


@dataclass
class Ticks:
    ts: np.ndarray  # int64 ns UTC，升序，只含有效报价
    bid: np.ndarray
    ask: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        return (self.bid + self.ask) / 2


def load_ticks(root: Path, und: str, sym: str, d: dt.date) -> Ticks | None:
    base = root / "diag" / "v4_ticks" / f"underlying={und}" / f"date={d.isoformat()}"
    files = sorted(base.glob(f"{sym}_*.parquet"))
    if not files:
        return None
    parts = [pd.read_parquet(f) for f in files]
    parts = [x for x in parts if len(x)]
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["ask"] > 0) & (df["bid"] >= 0) & (df["ask"] >= df["bid"])]
    df = df.drop_duplicates().sort_values("ts", kind="stable")
    ts = ns(pd.to_datetime(df["ts"], utc=True))
    return Ticks(ts, df["bid"].to_numpy(float), df["ask"].to_numpy(float))


# ---------------- tick 级回放 ----------------


def _first_at(tk: Ticks, t_ns: int) -> int | None:
    i = int(np.searchsorted(tk.ts, t_ns, side="left"))
    if i < len(tk.ts):
        return i
    return len(tk.ts) - 1 if len(tk.ts) else None  # 窗口内之后再无报价：用最后一个有效报价


def simulate(tk: Ticks, decision_ns: int, struct_ns: np.ndarray, flat_ns: int, v4: dict):
    """单笔 tick 级回放。struct_ns = 该方向结构离场触发的 K 收盘时刻（ns，升序）。

    返回 dict（entry_ns、exit_ns、ask0、net_usd、net_pct、tp_used、reason）或 None（无法入场）。
    """
    exe, prem = v4["execution"], v4["premium"]
    lag = int(exe["lag_s"] * 1e9)
    qty = exe["qty"]
    i0 = int(np.searchsorted(tk.ts, decision_ns + lag, side="left"))
    if i0 >= len(tk.ts) or tk.ts[i0] >= flat_ns:
        return None
    ask0, te = float(tk.ask[i0]), int(tk.ts[i0])
    mid = tk.mid[i0 + 1 :]
    big = np.iinfo(np.int64).max
    tp_hit = np.flatnonzero(mid >= ask0 * (1 + prem["tp"]))
    sl_hit = np.flatnonzero(mid <= ask0 * (1 - prem["sl"]))
    t_tp = int(tk.ts[i0 + 1 + tp_hit[0]]) if len(tp_hit) else big
    t_sl = int(tk.ts[i0 + 1 + sl_hit[0]]) if len(sl_hit) else big
    j = int(np.searchsorted(struct_ns, decision_ns, side="right"))
    t_st = int(struct_ns[j]) if j < len(struct_ns) else big
    t_cap = te + int(v4["cap_min"] * 60e9)
    cands = {"structure": t_st, "sl": t_sl, "cap": t_cap, "flat": int(flat_ns)}
    reason = min(cands, key=cands.get)
    t_f = cands[reason]
    cash = -ask0 * 100 * qty - _BUY[qty]
    used = t_tp < t_f
    if used:
        k = _first_at(tk, t_tp + lag)
        b = float(tk.bid[k])
        cash += b * 100 - (float(sell_fee(1, b)) if b > 0 else 0.0)
    rem = qty - int(used)
    k = _first_at(tk, t_f + lag)
    b = float(tk.bid[k])
    cash += rem * b * 100 - (float(sell_fee(rem, b)) if b > 0 else 0.0)
    return {
        "entry_ns": te,
        "exit_ns": int(tk.ts[k]),
        "ask0": ask0,
        "net_usd": cash,
        "net_pct": cash / (ask0 * 100 * qty) * 100,
        "tp_used": used,
        "reason": reason,
    }


def greedy(rows: pd.DataFrame) -> pd.DataFrame:
    """按入场决策时刻排序，上一笔离场前的事件跳过（每格独立执行）。"""
    rows = rows.sort_values("decision_ns")
    take, last = [], -1
    for dn, xn in zip(rows["decision_ns"], rows["exit_ns"], strict=True):
        if dn >= last:
            take.append(True)
            last = xn
        else:
            take.append(False)
    return rows[np.array(take, bool)]


# ---------------- 统计 ----------------


def cluster_p(x: np.ndarray, groups: np.ndarray, min_n: int) -> tuple[float, float]:
    """单侧 H0：均值 ≤ 0。按日聚类稳健标准误，t 分布 df = 聚类数 − 1。"""
    from scipy import stats

    n = len(x)
    g = pd.Series(x - x.mean()).groupby(groups).sum().to_numpy()
    G = len(g)
    if n < min_n or G < 2:
        return np.nan, 1.0
    se = np.sqrt((g**2).sum()) / n
    if se == 0:
        return np.nan, 1.0
    t = x.mean() / se
    return float(t), float(stats.t.sf(t, G - 1))


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini–Hochberg：返回是否拒绝 H0。"""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    thr = q * np.arange(1, m + 1) / m
    ok = p[order] <= thr
    k = int(np.flatnonzero(ok).max()) + 1 if ok.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out


def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max())


def period_of(d: dt.date) -> str:
    if d >= dt.date(2026, 7, 1):
        return "2026-07+"
    if d.year == 2026:
        return "2026H1"
    return str(d.year)


def cell_stats(t: pd.DataFrame, v4: dict) -> dict:
    x = t["net_pct"].to_numpy()
    usd = t["net_usd"].to_numpy()
    wins, losses = usd[usd > 0], usd[usd <= 0]
    tval, p = cluster_p(x, t["day"].to_numpy(), v4["stats"]["min_n"])
    k = v4["stats"]["top_k_remove"]
    rest = np.sort(usd)[:-k] if len(usd) > k else np.array([])
    rest_pct = t.sort_values("net_usd")["net_pct"].to_numpy()[:-k] if len(usd) > k else []
    return {
        "n": len(t),
        "win_rate": round(float((usd > 0).mean()), 3) if len(t) else np.nan,
        "payoff": round(float(wins.mean() / -losses.mean()), 2)
        if len(wins) and len(losses) and losses.mean() < 0
        else np.nan,
        "mean_pct": round(float(x.mean()), 2) if len(t) else np.nan,
        "median_pct": round(float(np.median(x)), 2) if len(t) else np.nan,
        "sum_usd": round(float(usd.sum()), 2),
        "max_dd_usd": round(max_drawdown(t.sort_values("entry_ns")["net_usd"].to_numpy()), 2),
        "mean_pct_ex_top5": round(float(np.mean(rest_pct)), 2) if len(rest) else np.nan,
        "t": round(tval, 2) if np.isfinite(tval) else np.nan,
        "p": p,
    }


def random_percentile(t: pd.DataFrame, cand: dict, v4: dict) -> tuple[float, int]:
    """每笔实际交易换成同日同方向 ±window 内随机一根合格 K（同合约规则、同离场）；n_rep 次的均值
    分布里，实际均值的分位（%）。返回 (分位, 有候选的笔数)。"""
    rng = np.random.default_rng(v4["random"]["seed"])
    keys = [k for k in t["key"] if len(cand.get(k, ())) > 0]
    if not keys:
        return np.nan, 0
    actual = t.set_index("key").loc[keys, "net_pct"].to_numpy().mean()
    arrs = [np.asarray(cand[k]) for k in keys]
    reps = np.empty(v4["random"]["n_rep"])
    for r in range(len(reps)):
        reps[r] = np.mean([a[rng.integers(len(a))] for a in arrs])
    return round(float((reps < actual).mean() * 100), 1), len(keys)


# ---------------- 主流程 ----------------


def run_conv(
    root: Path,
    conv_key: str,
    v4: dict,
    pull: bool = True,
    sig_fn=None,
    hyps=HYPS,
    exit_map: dict | None = None,
    days_fn=None,
    unds: list | None = None,
) -> dict:
    exit_map = v4["exits"] if exit_map is None else exit_map
    trades_all, cells, pull_stats, info = [], [], {}, {}
    for und in v4["sample"]["underlyings"] if unds is None else unds:
        days = days_fn(root, und) if days_fn else None
        ev, aux = build_events(root, und, conv_key, v4, sig_fn=sig_fn, hyps=hyps, days=days)
        trig = aux["trig"]
        windows = needed_windows(ev, v4)
        if pull:
            pull_stats[und] = pull_ticks(root, und, windows)
        info[und] = {
            "days": len(aux["meta"]["days"]),
            "bars": aux["meta"]["n_bars"],
            "signals_raw": aux["meta"]["signals"],
            "windows": len(windows),
            "event_rows": len(ev),
            "signal_rows": {h: int(ev[f"hyp_{h}"].sum()) for h in hyps},
        }
        ev = ev.reset_index(drop=True)
        ev["decision_ns"] = ns(ev["decision_ts"])
        ev["flat_ns"] = ns(ev["flat_ts"])
        tick_cache: dict = {}
        sims: dict = {}  # (row, rule) -> result
        rules = sorted({r for v in exit_map.values() for r in v.values()})
        for i, r in ev.iterrows():
            key = (r["symbol"], r["day"])
            if key not in tick_cache:
                tick_cache[key] = load_ticks(root, und, *key)
            tk = tick_cache[key]
            if tk is None:
                continue
            for rule in rules:
                res = simulate(
                    tk,
                    int(r["decision_ns"]),
                    trig[rule][int(r["direction"])],
                    int(r["flat_ns"]),
                    v4,
                )
                if res is not None:
                    sims[(i, rule)] = res
        w = pd.Timedelta(minutes=v4["random"]["window_min"]).value
        for h in hyps:
            for xk in EXIT_KEYS:
                rule = exit_map[xk][h]
                for direction in (1, -1):
                    m = ev[f"hyp_{h}"] & (ev["direction"] == direction)
                    rows = []
                    for i in np.flatnonzero(m.to_numpy()):
                        res = sims.get((i, rule))
                        if res is not None:
                            rows.append(
                                {
                                    "key": i,
                                    "day": ev.at[i, "day"],
                                    "decision_ns": int(ev.at[i, "decision_ns"]),
                                    "dte": int(ev.at[i, "dte"]),
                                    "symbol": ev.at[i, "symbol"],
                                    **res,
                                }
                            )
                    t = (
                        greedy(pd.DataFrame(rows))
                        if rows
                        else pd.DataFrame(
                            columns=["key", "day", "decision_ns", "net_usd", "net_pct", "entry_ns"]
                        )
                    )
                    cand = {}
                    same = ev[(ev["direction"] == direction)]
                    for k, d0, dn in zip(t["key"], t["day"], t["decision_ns"], strict=True):
                        c = same[
                            (same["day"] == d0)
                            & ((same["decision_ns"] - dn).abs() <= w)
                            & (same.index != k)
                        ]
                        cand[k] = [sims[(j, rule)]["net_pct"] for j in c.index if (j, rule) in sims]
                    s = cell_stats(t, v4)
                    s["random_pctile"], s["random_n"] = random_percentile(t, cand, v4)
                    s.update(
                        {
                            "conv": conv_key,
                            "und": und,
                            "hyp": h,
                            "exit": xk,
                            "rule": rule,
                            "dir": "long" if direction == 1 else "short",
                        }
                    )
                    cells.append(s)
                    if len(t):
                        tt = t.assign(
                            conv=conv_key, und=und, hyp=h, exit=xk, rule=rule, dir=s["dir"]
                        )
                        trades_all.append(tt)
    cells = pd.DataFrame(cells)
    cells["fdr_pass"] = bh(cells["p"].to_numpy(), v4["stats"]["fdr_q"])
    trades = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    return {"cells": cells, "trades": trades, "pull": pull_stats, "info": info}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="both", choices=["C", "A", "both"])
    ap.add_argument(
        "--unds", default=None, help="逗号分隔；给定时输出文件带 _<标的> 后缀，FDR 不在此计算"
    )
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = load_v4()
    out = root / "research" / "v4"
    out.mkdir(parents=True, exist_ok=True)
    convs = [v4["primary"], v4["sensitivity"]] if args.conv == "both" else [args.conv]
    unds = args.unds.split(",") if args.unds else None
    sfx = f"_{'_'.join(unds)}" if unds else ""
    for c in convs:
        res = run_conv(root, c, v4, pull=not args.no_pull, unds=unds)
        res["cells"].to_parquet(out / f"cells_{c}{sfx}.parquet", index=False)
        if len(res["trades"]):
            res["trades"].to_parquet(out / f"trades_{c}{sfx}.parquet", index=False)
        pd.Series({"pull": res["pull"], "info": res["info"]}).to_json(
            out / f"run_{c}{sfx}.json", default_handler=str
        )
        log.info("口径 %s%s 完成：%d 格", c, sfx, len(res["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
