# ruff: noqa: E501  （报告行较长）
"""规则 → 结果（任务书：规则回测最小流水线）。接口说明见 docs/RULE_INTERFACE.md。

    python scripts/run_rule.py --rule rules/example_boll_macd.py --symbol MU \
        --start 2025-01-02 --end 2025-12-31 --mode stock --out reports/example

- run_rule(bars_1m, rule, account, session, mode, quotes=None, contract_cfg=None) → RunResult（纯函数）；
- main()：解析参数、读本地数据（bars_1m、options_1m，不联网）、按 D-033 剔除成交参数校准窗口、
  调 run_rule、写 trades.csv 与 summary.md。

流程：RTH 1 分钟 K → resample_bars 到 PERIOD → rule.signal(bars) → 信号在 PERIOD K 收盘时刻 T 出现，入场 =
1 分钟 K 中开盘时刻等于 T 的那一根（没有 → no_bar）→ 逐日 zr.sim.stock（正股）或 zr.sim.engine（期权，
出场时刻由 zr.sim.stock.plan_exit 在正股 1 分钟 K 上给出，交给引擎的 at 出场）。
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.util
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from zr.sim import engine as eng
from zr.sim.engine import ET, NS, Account, SimPolicy
from zr.sim.stock import SKIP_KEYS as STOCK_SKIP_KEYS
from zr.sim.stock import StockPolicy, plan_exit, run_day

MIN_NS = 60 * NS
OPTION_SKIP_KEYS = (
    "no_bar",
    "outside_session",
    "daily_loss",
    "max_entries",
    "in_position",
    "cooldown",
    "no_contract",
    "no_quote",
    "wide_spread",
    "late_quote",
    "size_zero",
    "early_exit",
)
COLUMNS = [
    "date", "mode", "symbol", "contract", "direction", "signal_time", "stock_entry_time",
    "stock_entry_px", "stock_exit_time", "stock_exit_px", "stock_exit_reason", "option_entry_time",
    "option_entry_px", "option_exit_time", "option_exit_px", "qty", "fees", "pnl_usd", "pnl_pct",
    "skip_reason",
]  # fmt: skip


@dataclass
class RunResult:
    mode: str
    trades: pd.DataFrame  # 每个信号一行；skip_reason == "" 为成交
    skipped: dict
    equity_start: float
    equity_end: float
    notes: list = field(default_factory=list)


def load_rule(path: str | Path):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for k in ("PERIOD", "DIRECTION", "EXIT", "signal"):
        if not hasattr(mod, k):
            raise ValueError(f"规则文件 {p} 缺少 {k}")
    return mod


def resample_rule_bars(bars_1m: pd.DataFrame, period: int) -> pd.DataFrame:
    """RTH 1 分钟 K（索引 = UTC 开盘时刻）→ PERIOD 分钟 K（resample_bars，按美东日、整分钟网格）。"""
    from zr.features.bars import resample_bars

    if period not in (1, 2, 5):
        raise ValueError("PERIOD 只能是 1 / 2 / 5")
    b = bars_1m if period == 1 else resample_bars(bars_1m, f"{period}min")
    return b[b["close"].notna()]


def _hms(ts_ns) -> str:
    if ts_ns is None or (isinstance(ts_ns, float) and math.isnan(ts_ns)):
        return ""
    return pd.Timestamp(int(ts_ns), tz="UTC").tz_convert(ET).strftime("%H:%M:%S")


def _scaled_exit(exit_rule: dict, period: int) -> dict:
    """EXIT 里的 bars / max_bars 是 PERIOD 根数 → 换算成 1 分钟根数。"""
    r = dict(exit_rule)
    for k in ("bars", "max_bars"):
        if r.get(k) is not None:
            r[k] = int(r[k]) * period
    return r


def _signals(bars_1m: pd.DataFrame, rule) -> pd.DataFrame:
    """→ DataFrame[day, T（信号 K 收盘时刻 ns）, dir, spot（信号 K 收盘价）]。"""
    rb = resample_rule_bars(bars_1m, int(rule.PERIOD))
    s = rule.signal(rb)
    s = pd.Series(np.asarray(s), index=rb.index)
    if len(s) != len(rb):
        raise ValueError(f"signal 长度 {len(s)} ≠ bars 长度 {len(rb)}")
    v = s.to_numpy()
    if pd.isna(v).any() or not set(np.unique(v.astype(float))) <= {-1.0, 0.0, 1.0}:
        raise ValueError("signal 取值只能是 −1 / 0 / +1（不能有 NaN）")
    v = v.astype(int)
    allow = {"long": (1,), "short": (-1,), "both": (1, -1)}[rule.DIRECTION]
    keep = np.isin(v, allow)
    lab = rb.index.as_unit("ns").asi8[keep]
    return pd.DataFrame(
        {
            "day": rb.index[keep].tz_convert(ET).date,
            "T": lab + int(rule.PERIOD) * MIN_NS,
            "dir": v[keep],
            "spot": rb["close"].to_numpy(float)[keep],
        }
    )


def _day_bars(bars_1m: pd.DataFrame) -> dict:
    days = bars_1m.index.tz_convert(ET).date
    out = {}
    for d in sorted(set(days)):
        x = bars_1m[days == d]
        out[d] = pd.DataFrame(
            {
                "ts_ns": x.index.as_unit("ns").asi8,
                **{c: x[c].to_numpy(float) for c in ("open", "high", "low", "close")},
            }
        )
    return out


def _row(day, mode, symbol, d, T, **kw) -> dict:
    r = dict.fromkeys(COLUMNS, "")
    r.update(date=str(day), mode=mode, symbol=symbol, direction=int(d), signal_time=_hms(T))
    r.update(kw)
    return r


def _prep_quotes(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    if "ts_ns" not in q:
        q["ts_ns"] = pd.DatetimeIndex(q["ts"]).as_unit("ns").asi8
    q["expiry"] = pd.to_datetime(q["expiry"]).dt.date
    return q.sort_values(["symbol", "ts_ns"], kind="stable")


def pick_contract(q: pd.DataFrame, day: dt.date, spot: float, d: int, cfg: dict) -> str | None:
    """最近可交易到期日（0 ≤ DTE ≤ dte_max，日历日），平值（行权价最接近 spot）向虚值 otm_steps 档；
    不存在 → None（不回退到别的合约）。"""
    right = "call" if d == 1 else "put"
    x = q[q["right"] == right]
    exps = sorted(e for e in x["expiry"].unique() if 0 <= (e - day).days <= int(cfg["dte_max"]))
    if not exps:
        return None
    x = x[x["expiry"] == exps[0]]
    ks = np.sort(x["strike"].unique())
    if not len(ks):
        return None
    k = int(np.argmin(np.abs(ks - spot))) + (1 if d == 1 else -1) * int(cfg.get("otm_steps", 0))
    if not 0 <= k < len(ks):
        return None
    sym = x.loc[x["strike"] == ks[k], "symbol"].unique()
    return str(sym[0]) if len(sym) == 1 else None


def run_rule(
    bars_1m: pd.DataFrame,
    rule,
    account: Account,
    session: dict,
    mode: str,
    quotes: pd.DataFrame | Callable[[dt.date], pd.DataFrame] | None = None,
    contract_cfg: dict | None = None,
    *,
    stock_policy: StockPolicy = StockPolicy(),  # noqa: B008 - frozen
    max_daily_loss_pct: float | None = 0.10,
    max_entries_per_day: int | None = 20,
    max_spread_pct: float = 0.15,
    lag_s: float = 2.0,
    max_entry_wait_s: float = 60.0,
) -> RunResult:
    """bars_1m：RTH 1 分钟 K（索引 = UTC 开盘时刻，列 open/high/low/close/volume，与 load_bars 相同）；
    rule：含 PERIOD / DIRECTION / EXIT / CONTRACT / signal 的模块或对象；mode：stock / option；
    quotes（期权）：options_1m 报价表（ts 或 ts_ns、symbol、expiry、strike、right、bid、ask），或按日返回
    该表的函数。期权模式的当日开仓上限取 stock_policy.max_entries_per_day，未设则用 max_entries_per_day。"""
    if mode not in ("stock", "option"):
        raise ValueError("mode 只能是 stock / option")
    symbol = str(bars_1m["symbol"].iloc[0]).split(".")[0] if "symbol" in bars_1m and len(bars_1m) else ""
    sig = _signals(bars_1m, rule)
    exit_rule = _scaled_exit(rule.EXIT, int(rule.PERIOD))
    days = _day_bars(bars_1m)
    rows: list[dict] = []
    equity0 = equity = float(account.equity_usd)
    keys = STOCK_SKIP_KEYS if mode == "stock" else OPTION_SKIP_KEYS
    skipped = dict.fromkeys(keys, 0)
    notes = []
    cfg = dict(getattr(rule, "CONTRACT", {}) or {})
    cfg.update(contract_cfg or {})
    q_days = None
    if mode == "option" and isinstance(quotes, pd.DataFrame):
        t = quotes["ts"] if "ts" in quotes else pd.to_datetime(quotes["ts_ns"], utc=True)
        q_days = pd.DatetimeIndex(t).tz_convert(ET).date
    for day, sd in sig.groupby("day", sort=True):
        db = days.get(day, pd.DataFrame(columns=["ts_ns", "open", "high", "low", "close"]))
        pos = {int(t): i for i, t in enumerate(db["ts_ns"].to_numpy(np.int64))}
        if mode == "stock":
            mapped, no_bar = [], []
            for r in sd.itertuples():
                (mapped if int(r.T) in pos else no_bar).append(r)
            res, outs = run_day(
                db, [(pos[int(r.T)], int(r.dir)) for r in mapped],
                dataclasses.replace(account, equity_usd=equity), session, stock_policy,
                exit_rule=exit_rule, symbol=symbol,
            )  # fmt: skip
            for r, (kind, val) in zip(mapped, outs, strict=True):
                if kind == "skip":
                    rows.append(_row(day, mode, symbol, r.dir, r.T, skip_reason=val))
                    continue
                t = val
                rows.append(
                    _row(
                        day, mode, symbol, r.dir, r.T,
                        stock_entry_time=_hms(t.entry_ts_ns), stock_entry_px=t.entry_px,
                        stock_exit_time=_hms(t.exit_ts_ns), stock_exit_px=t.exit_px,
                        stock_exit_reason=t.reason, qty=t.qty, fees=t.fees_usd, pnl_usd=t.pnl_usd,
                        pnl_pct=t.pnl_usd / (t.entry_px * t.qty) * 100,
                    )
                )  # fmt: skip
            for k, v in res.skipped.items():
                skipped[k] += v
            for r in no_bar:
                skipped["no_bar"] += 1
                rows.append(_row(day, mode, symbol, r.dir, r.T, skip_reason="no_bar"))
            equity = res.equity_end
            continue
        # ---------------- 期权模式：账户层由 runner 承担 ----------------
        if quotes is None:
            raise ValueError("期权模式需要 quotes")
        dq = quotes(day) if callable(quotes) else quotes[q_days == day]
        dq = _prep_quotes(dq) if dq is not None and len(dq) else None
        fe, le, fl = (eng._ns_at(day, session[k]) for k in ("first_entry", "last_entry", "flat"))
        cap = stock_policy.max_entries_per_day if stock_policy.max_entries_per_day is not None else max_entries_per_day
        eq_day0, n_open, last_exit = equity, 0, None
        pol = SimPolicy(
            cooldown_s=0.0, min_coverage=0.0, max_spread_pct=max_spread_pct, lag_s=lag_s,
            max_entry_wait_s=max_entry_wait_s, max_daily_loss_pct=None, max_entries_per_day=None,
        )  # fmt: skip
        for r in sd.sort_values("T", kind="stable").itertuples():
            T, d = int(r.T), int(r.dir)

            def skip(key: str, _r=r, _day=day, **kw) -> None:
                skipped[key] += 1
                rows.append(_row(_day, mode, symbol, _r.dir, _r.T, skip_reason=key, **kw))

            if T not in pos or T >= fl:
                skip("no_bar")
                continue
            if T < fe or T > le:
                skip("outside_session")
                continue
            if max_daily_loss_pct is not None and eq_day0 - equity >= max_daily_loss_pct * eq_day0:
                skip("daily_loss")
                continue
            if cap is not None and n_open >= cap:
                skip("max_entries")
                continue
            if last_exit is not None and T < last_exit:
                skip("in_position")
                continue
            if last_exit is not None and T < last_exit + stock_policy.cooldown_bars * MIN_NS:
                skip("cooldown")
                continue
            i = pos[T]
            j, sx_px, sx_ts, sx_reason = plan_exit(db, i, exit_rule, session, d, stock_policy)
            stock = {
                "stock_entry_time": _hms(T),
                "stock_entry_px": float(db["open"].iloc[i]) + d * stock_policy.slippage_ticks * stock_policy.tick,
                "stock_exit_time": _hms(sx_ts), "stock_exit_px": sx_px, "stock_exit_reason": sx_reason,
            }  # fmt: skip
            sym = pick_contract(dq, day, float(r.spot), d, cfg) if dq is not None else None
            if sym is None:
                skip("no_contract", **stock)
                continue
            cq = dq[dq["symbol"] == sym]
            tq = cq["ts_ns"].to_numpy(np.int64)
            t0 = T + int(lag_s * NS)
            k0 = int(np.searchsorted(tq, t0, side="left"))
            if k0 >= len(tq) or tq[k0] > t0 + int(max_entry_wait_s * NS):
                skip("no_quote", contract=sym, **stock)
                continue
            b0, a0 = float(cq["bid"].iloc[k0]), float(cq["ask"].iloc[k0])
            if not (b0 > 0 and a0 > 0 and a0 >= b0):
                skip("no_quote", contract=sym, **stock)
                continue
            if (a0 - b0) / ((a0 + b0) / 2) > max_spread_pct:
                skip("wide_spread", contract=sym, **stock)
                continue
            # 区间 [入场分钟, 出场分钟] 每分钟都要有该合约的报价行（数据缺口 → no_quote；不按报价好坏剔除）
            m0, m1 = tq[k0] // MIN_NS, sx_ts // MIN_NS
            have = set((tq // MIN_NS).tolist())
            if m1 >= m0 and any(m not in have for m in range(int(m0), int(m1) + 1)):
                skip("no_quote", contract=sym, **stock)
                continue
            try:
                res = eng.simulate_day(
                    cq[["ts_ns", "symbol", "bid", "ask"]], [(T, d, sym, {"kind": "at", "ts_ns": int(sx_ts)})],
                    dataclasses.replace(account, equity_usd=equity), session, pol,
                )  # fmt: skip
            except ValueError:
                skip("no_quote", contract=sym, **stock)
                continue
            if not res.trades:
                key = next(k for k, v in res.skipped.items() if v)
                skip(key if key in skipped else "no_quote", contract=sym, **stock)
                continue
            t = res.trades[0]
            if t.exit_ts_ns < sx_ts:
                skip("early_exit", contract=sym, **stock)
                continue
            rows.append(
                _row(
                    day, mode, symbol, d, T, contract=sym, **stock,
                    option_entry_time=_hms(t.entry_ts_ns), option_entry_px=t.entry_px,
                    option_exit_time=_hms(t.exit_ts_ns), option_exit_px=t.exit_px, qty=t.qty,
                    fees=t.fees_usd, pnl_usd=t.pnl_usd, pnl_pct=t.pnl_usd / (t.entry_px * 100 * t.qty) * 100,
                )
            )  # fmt: skip
            equity += t.pnl_usd
            n_open += 1
            last_exit = t.exit_ts_ns
    if mode == "stock":
        notes.append("做空不建模借券费。")
    else:
        notes.append(
            "期权报价为 options_1m 的 1 分钟快照；引擎的 2 秒延迟在 1 分钟报价下约等于延后到下一分钟"
            "（入场 = 信号收盘后下一分钟报价的 ask，出场 = 正股出场时刻后下一分钟报价的 bid）。"
        )
    out = pd.DataFrame(rows, columns=COLUMNS).sort_values(["date", "signal_time"], kind="stable")
    return RunResult(mode, out.reset_index(drop=True), skipped, equity0, equity, notes)


def summary_md(res: RunResult, header: str) -> str:
    t = res.trades[res.trades["skip_reason"] == ""]
    pnl = t["pnl_usd"].astype(float).to_numpy()
    curve = res.equity_start + np.cumsum(pnl) if len(pnl) else np.array([res.equity_start])
    peak = np.maximum.accumulate(np.r_[res.equity_start, curve])
    dd = (np.r_[res.equity_start, curve] - peak).min()
    dd_pct = ((np.r_[res.equity_start, curve] - peak) / peak).min() * 100
    L = [
        f"# {header}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 信号数 | {len(res.trades)} |",
        f"| 成交笔数 n | {len(t)} |",
        f"| 胜率 | {(pnl > 0).mean() * 100:.1f}% |" if len(t) else "| 胜率 | — |",
        f"| 平均盈亏% | {t['pnl_pct'].astype(float).mean():+.3f}% |" if len(t) else "| 平均盈亏% | — |",
        f"| 总盈亏（美元，已扣费） | {pnl.sum():+,.2f} |",
        f"| 期初 / 期末权益 | {res.equity_start:,.2f} / {res.equity_end:,.2f} |",
        f"| 最大回撤 | {dd:,.2f} 美元（{dd_pct:.2f}%） |",
        f"| 费用合计 | {t['fees'].astype(float).sum():,.2f} |",
        "",
        "跳过明细：" + "，".join(f"{k} {v}" for k, v in res.skipped.items()),
        "",
        "数据分辨率：正股 1 分钟 K（长桥 bars_1m）"
        + ("；期权 1 分钟报价（ThetaData options_1m 快照）" if res.mode == "option" else "")
        + "。",
        "",
        *[f"- {n}" for n in res.notes],
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="规则回测：规则文件 → trades.csv + summary.md")
    ap.add_argument("--rule", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--mode", choices=("stock", "option"), default="stock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--first-entry", default="09:30")
    ap.add_argument("--last-entry", default="15:40")
    ap.add_argument("--flat", default="15:50")
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--cooldown-bars", type=int, default=0)
    a = ap.parse_args(argv)

    from zr.backtest.intrabar import calibration_window
    from zr.config import get_settings
    from zr.features.bars import load_bars
    from zr.features.option_features import load_option_day

    root = get_settings().zr_data_root
    rule = load_rule(a.rule)
    start, end = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    bars = load_bars(root, a.symbol, start, end)
    lo, hi = calibration_window()
    days = bars.index.tz_convert(ET).date
    cal = (days >= lo) & (days <= hi)
    n_cal = len(set(days[cal]))
    bars = bars[~cal]  # D-033：规则测试不得使用成交参数校准窗口
    if bars.empty:
        print("所选区间没有可用的 1 分钟 K", file=sys.stderr)
        return 1

    def quotes(day: dt.date) -> pd.DataFrame | None:
        od = load_option_day(root, a.symbol, day)
        return None if od is None else od["quotes"]

    session = {"first_entry": a.first_entry, "last_entry": a.last_entry, "flat": a.flat}
    res = run_rule(
        bars, rule, Account(equity_usd=a.equity), session, a.mode,
        quotes=quotes if a.mode == "option" else None,
        stock_policy=StockPolicy(slippage_ticks=a.slippage_ticks, cooldown_bars=a.cooldown_bars),
    )  # fmt: skip
    if n_cal:
        res.notes.append(f"按 D-033 剔除成交参数校准窗口 {lo}～{hi} 内的 {n_cal} 个交易日。")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = {c: 4 for c in ("stock_entry_px", "stock_exit_px", "option_entry_px", "option_exit_px", "pnl_pct")}
    rnd.update(fees=2, pnl_usd=2)
    tr = res.trades.copy()
    for c, nd in rnd.items():
        tr[c] = pd.to_numeric(tr[c], errors="coerce").round(nd)
    tr.to_csv(out / "trades.csv", index=False)
    d0, d1 = sorted(set(bars.index.tz_convert(ET).date))[0], sorted(set(bars.index.tz_convert(ET).date))[-1]
    head = f"{Path(a.rule).stem} · {a.symbol} · {a.mode} · {d0}～{d1}（{len(set(bars.index.tz_convert(ET).date))} 个交易日）"
    (out / "summary.md").write_text(summary_md(res, head), encoding="utf-8")
    print(f"写入 {out / 'trades.csv'}、{out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
