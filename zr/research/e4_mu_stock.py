"""E4 截图策略在 MU 正股上的专项回测 → docs/research/E4_MU_STOCK.md。

数据：MU 正股 1 分钟 K（`load_bars`，RTH、交易日历过滤），`bar_quality.clean_wicks` 修掉单根
尖刺影线（bar_range_gt_5atr 标记的大 bar 保留：多为真实大波动），再按日 resample 成 5 分钟 K。

规则（每条都是本模块的参数，见 Params）：
- 方向：收盘 > SMA(N) 且 SMA 上行（较上一根）→ 只做多；收盘 < SMA 且 SMA 下行 → 只做空。
  N ∈ {120, 127}（127 为对照）。SMA 跨日连续计算（只用 ≤t 的 bar）。
- 入场：KDJ(9,3,3) K 上穿 D（多）/ 下穿（空），信号 bar 收盘确认，下一根 5 分钟 bar 开盘成交。
- 止损：入场价 ∓ m × ATR(14)（5 分钟 bar，取信号 bar 的值），m ∈ {1, 1.5, 2}；固定不移动，
  加仓不改止损；bar 内触及即出，成交价 = min(开盘, 止损)（多）/ max(开盘, 止损)（空）。
- 梯形退出：当日已确认的摆动低点（左右各 2 根）出现「更高低点」后，梯形水位 = 该低点（只上移）；
  收盘跌破水位 → 下一根开盘出场。空头对称（更低高点）。
- 加仓：持仓浮盈（收盘相对持仓均价）> 0 且再次出现同向入场信号 → 下一根开盘加同等股数，最多 2 次。
- 日内平仓：15:45 那根 5 分钟 bar 收盘（= 15:50）全部平仓；提前收盘日在当日最后一根 bar 收盘平仓；
  入场 bar 开盘须早于 15:45。不隔夜（D-001）。
- 股数：每次入场 / 加仓 100 股。
- 成本：每单 `stock_order_fees`（D-013 口径的正股费用；做空的开仓为 Sell、平仓为 Buy），
  每边 1 tick（0.01 USD）滑点计入成交价。融券成本未计（日内）。

对照：MU 买入持有 100 股（首日开盘买、末日收盘卖，隔夜持有）；同入场时刻随机方向 1,000 次
（每个真实入场 bar 预先用同一套出场规则模拟多、空两个版本，随机方向即随机挑一个；
反方向版本的出场时刻可能与下一笔真实入场重叠，属于近似）。
运行：uv run python -m zr.research.e4_mu_stock
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.costs.fees import stock_order_fees
from zr.data.bar_quality import clean_wicks, flag_bars
from zr.features import indicators as ind
from zr.features.bars import ET, load_bars, resample_bars
from zr.ops.markdown import df_to_md

log = logging.getLogger("zr.research.e4_mu_stock")

MA_PERIODS = (120, 127)
STOP_MULTS = (1.0, 1.5, 2.0)
FLAT_MIN = 15 * 60 + 50  # 15:50 ET 前全部平仓
BAR_MIN = 5


@dataclass(frozen=True)
class Params:
    ma: int = 120
    stop_mult: float = 1.5
    max_adds: int = 2
    lot: int = 100
    tick: float = 0.01
    swing_w: int = 2
    kdj: tuple[int, int, int] = (9, 3, 3)
    atr_n: int = 14


@lru_cache(maxsize=200_000)
def _fee(shares: int, side: str, px_cents: int) -> float:
    return float(stock_order_fees(shares, side, px_cents / 100)["total"])


def fee(shares: int, side: str, px: float) -> float:
    return _fee(int(shares), side, int(round(px * 100)))


def prepare(df5: pd.DataFrame, p: Params) -> dict:
    """5 分钟 bar → 回测用数组（全部只用 ≤t 的数据）。"""
    c = df5["close"]
    ma = ind.sma(c, p.ma)
    ma_up = ma.diff() > 0
    ma_dn = ma.diff() < 0
    k = ind.kdj(df5["high"], df5["low"], c, *p.kdj)
    up = (k["k"] > k["d"]) & (k["k"].shift(1) <= k["d"].shift(1))
    dn = (k["k"] < k["d"]) & (k["k"].shift(1) >= k["d"].shift(1))
    et = df5.index.tz_convert(ET)
    day = np.asarray(et.date)
    w = p.swing_w
    # 摆动点：bar t−w 是 [t−2w, t] 的最低（高）→ 在 bar t 收盘确认；不跨日
    same_day = pd.Series(day, index=df5.index)
    lo_min = df5["low"].rolling(2 * w + 1).min()
    hi_max = df5["high"].rolling(2 * w + 1).max()
    win_same = (same_day.shift(2 * w) == same_day).to_numpy()
    lo_c = df5["low"].shift(w).to_numpy()
    hi_c = df5["high"].shift(w).to_numpy()
    sw_lo = np.where(win_same & (lo_c == lo_min.to_numpy()), lo_c, np.nan)
    sw_hi = np.where(win_same & (hi_c == hi_max.to_numpy()), hi_c, np.nan)
    return {
        "open": df5["open"].to_numpy(float),
        "high": df5["high"].to_numpy(float),
        "low": df5["low"].to_numpy(float),
        "close": c.to_numpy(float),
        "long": ((c > ma) & ma_up & up).to_numpy(),
        "short": ((c < ma) & ma_dn & dn).to_numpy(),
        "atr": ind.atr(df5["high"], df5["low"], c, p.atr_n).to_numpy(float),
        "sw_lo": sw_lo,
        "sw_hi": sw_hi,
        "day": day,
        "mod": np.asarray(et.hour * 60 + et.minute),
        "ts": df5.index,
    }


def simulate_entry(a: dict, i: int, d: int, p: Params) -> dict | None:
    """信号在 bar i 收盘确认，方向 d（+1 多 / −1 空）。

    返回一笔交易（含加仓）；不可入场时返回 None。
    """
    n = len(a["close"])
    j = i + 1
    if j >= n or a["day"][j] != a["day"][i] or a["mod"][j] >= FLAT_MIN - BAR_MIN:
        return None
    if not np.isfinite(a["atr"][i]):
        return None
    o, h, lo, c = a["open"], a["high"], a["low"], a["close"]
    sig = a["long"] if d == 1 else a["short"]
    sw = a["sw_lo"] if d == 1 else a["sw_hi"]
    tick = p.tick
    fills = [(j, o[j] + d * tick)]
    stop = fills[0][1] - d * p.stop_mult * a["atr"][i]
    # 当日已确认的摆动点（含入场前）
    day0 = a["day"][i]
    k0 = i
    while k0 > 0 and a["day"][k0 - 1] == day0:
        k0 -= 1
    swings = [x for x in sw[k0 : j + 1] if np.isfinite(x)]
    last_sw = swings[-1] if swings else np.nan
    prev_sw = swings[-2] if len(swings) > 1 else np.nan
    ladder = np.nan

    def ladder_update(last: float, prev: float, cur: float) -> float:
        if np.isfinite(last) and np.isfinite(prev) and d * (last - prev) > 0:
            return last if not np.isfinite(cur) else (max(cur, last) if d == 1 else min(cur, last))
        return cur

    ladder = ladder_update(last_sw, prev_sw, ladder)
    k = j
    exit_k, exit_px, reason = None, None, None
    while exit_k is None:
        if (d == 1 and lo[k] <= stop) or (d == -1 and h[k] >= stop):
            exit_k, exit_px, reason = (
                k,
                (min(o[k], stop) if d == 1 else max(o[k], stop)) - d * tick,
                "stop",
            )
            break
        last_bar = k + 1 >= n or a["day"][k + 1] != a["day"][k]
        if a["mod"][k] >= FLAT_MIN - BAR_MIN or last_bar:
            exit_k, exit_px, reason = k, c[k] - d * tick, "eod"
            break
        if k > j and np.isfinite(sw[k]):
            prev_sw, last_sw = last_sw, sw[k]
            ladder = ladder_update(last_sw, prev_sw, ladder)
        if np.isfinite(ladder) and d * (c[k] - ladder) < 0:
            exit_k, exit_px, reason = k + 1, o[k + 1] - d * tick, "ladder"
            break
        avg = float(np.mean([f[1] for f in fills]))
        if (
            len(fills) <= p.max_adds
            and k > j
            and sig[k]
            and d * (c[k] - avg) > 0
            and a["mod"][k + 1] < FLAT_MIN - BAR_MIN
        ):
            fills.append((k + 1, o[k + 1] + d * tick))
        k += 1
    side_in, side_out = ("Buy", "Sell") if d == 1 else ("Sell", "Buy")
    shares = p.lot * len(fills)
    gross = sum(d * p.lot * (exit_px - fp) for _, fp in fills)
    fees = sum(fee(p.lot, side_in, fp) for _, fp in fills) + fee(shares, side_out, exit_px)
    notional = sum(p.lot * fp for _, fp in fills)
    return {
        "signal_i": i,
        "entry_ts": a["ts"][j],
        "exit_ts": a["ts"][exit_k],
        "exit_i": exit_k,
        "direction": d,
        "entry_px": fills[0][1],
        "exit_px": exit_px,
        "adds": len(fills) - 1,
        "shares": shares,
        "gross": gross,
        "fees": fees,
        "net": gross - fees,
        "net_bp": (gross - fees) / notional * 1e4,
        "slippage": tick * 2 * shares,
        "reason": reason,
    }


def backtest(a: dict, p: Params) -> pd.DataFrame:
    """顺序回测：空仓时遇信号入场；出场 bar 起重新找信号（同一时刻只有一笔持仓）。"""
    trades, i, n = [], 0, len(a["close"])
    while i < n - 1:
        d = 1 if a["long"][i] else (-1 if a["short"][i] else 0)
        if d:
            t = simulate_entry(a, i, d, p)
            if t is not None:
                trades.append(t)
                i = max(t["exit_i"], i + 1)
                continue
        i += 1
    return pd.DataFrame(trades)


def random_direction(
    a: dict, trades: pd.DataFrame, p: Params, n: int = 1000, seed: int = 0
) -> np.ndarray:
    """同入场时刻随机方向：每笔预算多、空两版净盈亏，随机挑选，n 次合计。"""
    if trades.empty:
        return np.zeros(n)
    pl = np.empty((len(trades), 2))
    for r, (i, d) in enumerate(zip(trades["signal_i"], trades["direction"], strict=True)):
        same = trades["net"].iloc[r]
        other = simulate_entry(a, int(i), -int(d), p)
        o = other["net"] if other is not None else 0.0
        pl[r] = (same, o) if d == 1 else (o, same)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, 2, size=(n, len(trades)))
    return np.where(pick == 0, pl[:, 0], pl[:, 1]).sum(axis=1)


def daily_pnl(trades: pd.DataFrame, days: list[dt.date]) -> pd.Series:
    if trades.empty:
        return pd.Series(0.0, index=days)
    d = trades.groupby(trades["exit_ts"].dt.tz_convert(ET).dt.date)["net"].sum()
    return d.reindex(days, fill_value=0.0)


BASE: pd.Series | None = None  # 日 → 100 股 × 当日开盘价（日收益的分母），main() 里设置


def metrics(trades: pd.DataFrame, days: list[dt.date]) -> dict:
    """日收益 = 当日平仓净额 ÷（100 股 × 当日开盘价）；无交易日计 0。

    total_ret_pct 为日收益简单加总。
    """
    if trades.empty:
        return {"trades": 0}
    net = trades["net"]
    win, loss = net[net > 0], net[net <= 0]
    dp = daily_pnl(trades, days)
    cum = dp.cumsum()
    r = dp / BASE.reindex(dp.index) if BASE is not None else dp
    cr = r.cumsum()
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252) if r.std(ddof=1) > 0 else np.nan
    return {
        "trades": int(len(net)),
        "win_rate": round(float((net > 0).mean()), 3),
        "payoff": round(float(win.mean() / -loss.mean()), 2)
        if len(win) and len(loss) and loss.mean() < 0
        else np.nan,
        "mean_bp": round(float(trades["net_bp"].mean()), 2),
        "mean_usd": round(float(net.mean()), 2),
        "gross_usd": round(float(trades["gross"].sum()), 0),
        "fees_usd": round(float(trades["fees"].sum()), 0),
        "slippage_usd": round(float(trades["slippage"].sum()), 0),
        "total_usd": round(float(net.sum()), 0),
        "total_ret_pct": round(float(r.sum() * 100), 1),
        "max_dd_usd": round(float((cum - cum.cummax()).min()), 0),
        "max_dd_ret_pct": round(float((cr - cr.cummax()).min() * 100), 1),
        "sharpe_daily": round(float(sharpe), 2),
        "adds_share": round(float((trades["adds"] > 0).mean()), 3),
    }


def buy_and_hold(df1: pd.DataFrame, lot: int = 100, tick: float = 0.01) -> dict:
    px_in = float(df1["open"].iloc[0]) + tick
    px_out = float(df1["close"].iloc[-1]) - tick
    closes = df1["close"].groupby(np.asarray(df1.index.tz_convert(ET).date)).last()
    usd = (closes - px_in) * lot
    ret = closes.pct_change().dropna()
    fees = fee(lot, "Buy", px_in) + fee(lot, "Sell", px_out)
    return {
        "trades": 1,
        "total_usd": round(float((px_out - px_in) * lot - fees), 0),
        "total_ret_pct": round(float((px_out / px_in - 1) * 100), 1),
        "fees_usd": round(fees, 2),
        "max_dd_usd": round(float((usd - usd.cummax()).min()), 0),
        "max_dd_ret_pct": round(float(((closes / closes.cummax()) - 1).min() * 100), 1),
        "sharpe_daily": round(float(ret.mean() / ret.std(ddof=1) * np.sqrt(252)), 2),
        "px_in": round(px_in, 2),
        "px_out": round(px_out, 2),
    }


def coverage_table(raw: pd.DataFrame, flags: pd.DataFrame, df5: pd.DataFrame) -> pd.DataFrame:
    et = raw.index.tz_convert(ET)
    m = pd.Series(et.tz_localize(None).to_period("M").astype(str), index=raw.index)
    t = pd.DataFrame(
        {
            "days": pd.Series(np.asarray(et.date), index=raw.index).groupby(m).nunique(),
            "bars_1m": m.value_counts().sort_index(),
            "wick_spike_cleaned": flags["wick_spike"].groupby(m).sum(),
            "bar_range_gt_5atr_kept": flags["bar_range_gt_5atr"].groupby(m).sum(),
        }
    )
    m5 = pd.Series(
        df5.index.tz_convert(ET).tz_localize(None).to_period("M").astype(str), index=df5.index
    )
    t["bars_5m"] = m5.value_counts().reindex(t.index).fillna(0).astype(int)
    return t.rename_axis("month").reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MU.US")
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--out", default="docs/research/E4_MU_STOCK.md")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    und = args.symbol.split(".")[0]
    root = get_settings().zr_data_root
    raw = load_bars(root, und)
    flags = flag_bars(raw)
    df1 = clean_wicks(raw, flags)
    df5 = resample_bars(df1, f"{BAR_MIN}min")
    days = sorted(set(np.asarray(df5.index.tz_convert(ET).date)))
    global BASE
    BASE = df1["open"].groupby(np.asarray(df1.index.tz_convert(ET).date)).first() * Params().lot
    first, last = days[0], days[-1]
    bound = (
        f"标的 {und} 正股；{first}～{last} 共 {len(days)} 个交易日（RTH、交易日历过滤，"
        f"修正尖刺影线 {int(flags['wick_spike'].sum())} 根）；5 分钟 K；每次 100 股；"
        "费用 stock_order_fees + 每边 1 tick；不隔夜；未计融券成本"
    )
    cov = coverage_table(raw, flags, df5)

    rows, by_year, by_side, by_reason, rand_rows = [], [], [], [], []
    all_trades = {}
    for ma in MA_PERIODS:
        for m in STOP_MULTS:
            p = Params(ma=ma, stop_mult=m)
            a = prepare(df5, p)
            tr = backtest(a, p)
            cfg = f"SMA{ma}·ATR×{m:g}"
            all_trades[cfg] = tr
            met = metrics(tr, days)
            rnd = random_direction(a, tr, p, args.n_random)
            pct = float((rnd <= tr["net"].sum()).mean() * 100) if len(tr) else np.nan
            rows.append(
                {"config": cfg, "ma": ma, "stop_atr": m, **met, "rand_pctile": round(pct, 1)}
            )
            rand_rows.append(
                {
                    "config": cfg,
                    "real_total": met.get("total_usd"),
                    "rand_mean": round(float(rnd.mean()), 0),
                    "rand_p05": round(float(np.percentile(rnd, 5)), 0),
                    "rand_p50": round(float(np.percentile(rnd, 50)), 0),
                    "rand_p95": round(float(np.percentile(rnd, 95)), 0),
                    "real_pctile": round(pct, 1),
                }
            )
            if len(tr):
                yr = tr["entry_ts"].dt.tz_convert(ET).dt.year
                for y, g in tr.groupby(yr):
                    yd = [d for d in days if d.year == y]
                    by_year.append({"config": cfg, "year": y, **metrics(g, yd)})
                for s, g in tr.groupby("direction"):
                    by_side.append(
                        {"config": cfg, "side": "long" if s == 1 else "short", **metrics(g, days)}
                    )
                for r, g in tr.groupby("reason"):
                    by_reason.append(
                        {
                            "config": cfg,
                            "exit": r,
                            "n": len(g),
                            "share": round(len(g) / len(tr), 3),
                            "mean_usd": round(float(g["net"].mean()), 2),
                        }
                    )
            log.info("%s：%d 笔，%.0fs", cfg, len(tr), time.monotonic() - t0)
    main_t = pd.DataFrame(rows)
    bh = buy_and_hold(df1)

    def agg_by(col: str) -> pd.DataFrame:
        parts = []
        for v, cfgs in main_t.groupby(col)["config"]:
            tr = pd.concat([all_trades[c] for c in cfgs], ignore_index=True)
            parts.append({col: v, "configs": len(cfgs), **metrics(tr, days)})
        return pd.DataFrame(parts)

    best = main_t.sort_values("total_usd", ascending=False).iloc[0]
    L = [
        f"# E4 截图策略 · {und} 正股专项",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 耗时 "
        f"{time.monotonic() - t0:.0f}s · 规则与口径见 `zr/research/e4_mu_stock.py` docstring · "
        f"复现：`uv run python -m zr.research.e4_mu_stock`",
        "",
        "## 覆盖矩阵",
        "",
        df_to_md(cov),
        "",
        f"边界：{bound}。",
        "",
        "## 主表（6 组 = SMA 120/127 × ATR 止损 1/1.5/2 倍；全部为扣费后）",
        "",
        df_to_md(main_t),
        "",
        "列：win_rate / payoff（平均盈利 ÷ 平均亏损）/ mean_bp（每笔净额 ÷ 入场名义）/ "
        "total_ret_pct / max_dd_ret_pct / sharpe_daily 用日收益"
        " = 当日净额 ÷（100 股 × 当日开盘价），"
        "无交易日计 0，Sharpe = 均值 ÷ 标准差 × √252，total_ret_pct 为简单加总（买入持有为复利）/ "
        f"rand_pctile（真实合计在 {args.n_random} 次同入场时刻随机方向中的分位）。",
        "",
        f"边界：{bound}；6 组配置同时报告，未挑选（最好的一组是 {best['config']}，"
        "只作描述，不作为参数选择）。",
        "",
        "## 对照：买入持有与随机方向",
        "",
        "MU 买入持有 100 股（隔夜持有，与策略口径不同）：",
        "",
        df_to_md(pd.DataFrame([bh])),
        "",
        f"同入场时刻随机方向（{args.n_random} 次，合计净额 USD）：",
        "",
        df_to_md(pd.DataFrame(rand_rows)),
        "",
        f"边界：{bound}；随机方向的反向版本出场时刻可能与下一笔真实入场重叠（近似）。",
        "",
        "## 分组：按止损倍数（合并两个均线周期）",
        "",
        df_to_md(agg_by("stop_atr")),
        "",
        "## 分组：按均线周期（合并三个止损倍数）",
        "",
        df_to_md(agg_by("ma")),
        "",
        "## 分组：按年（逐配置）",
        "",
        df_to_md(pd.DataFrame(by_year)),
        "",
        f"边界：{bound}；2023 年只有 10–12 月，2026 年截至 {last}。",
        "",
        "## 分组：多 / 空（逐配置）",
        "",
        df_to_md(pd.DataFrame(by_side)),
        "",
        "## 出场原因（逐配置）",
        "",
        df_to_md(pd.DataFrame(by_reason)),
        "",
        f"边界：{bound}。",
        "",
    ]
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    log.info("写入 %s，%.0fs", args.out, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
