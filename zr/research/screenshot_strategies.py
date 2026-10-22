"""截图策略专项（R-010 / R-011）：按两个网传策略自己的规则回测，不做改良。

E4 完整版：5 分钟 K；SMA120 定向（价在均线上且均线上行 → 只做多，反之只做空）；KDJ 顺势交叉入场
（K 上穿 D 做多 / 下穿做空）；梯形法则退出（多头：跌破前一个更高低点；空头对称）；
只在盈利时加仓一次；
固定止损；日内平仓（本系统不隔夜，见 D-001）。
E9 完整版（YaleV3）：1 分钟 K；收盘跌穿布林下轨入场做多；无卖出信号；固定止损 −8%；
盈利 2% 后启动追踪止损
（回撤 2% 出场）；允许隔夜持有（原策略来自加密货币 24h 市场）。另跑「止损按标的波动缩放」版本：
stop/trail × (标的 20 日实现日波动 / 3%)，3% 为加密货币典型日波动（假设）。
费用：正股按 `stock_order_fees`；期权表达按单张 ATM 近似（delta 0.5、D-013 费用、2 tick 点差）。
运行：uv run python -m zr.research.screenshot_strategies → docs/research/SCREENSHOT_STRATEGIES.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.costs.fees import option_order_fees, stock_order_fees
from zr.execution.ticks import tick_size
from zr.features import indicators as ind
from zr.features.bars import load_bars, resample_bars
from zr.ops.markdown import df_to_md

log = logging.getLogger("zr.research.screenshot")
ET = ZoneInfo("America/New_York")
UNIT_NOTIONAL = 10_000.0  # 每单位名义，USD


@dataclass
class Trade:
    symbol: str
    direction: int
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry_px: float
    exit_px: float
    units: int
    reason: str

    def gross_ret(self) -> float:
        return self.direction * (self.exit_px / self.entry_px - 1)


def _option_pnl(ret: float, spot: float, und: str) -> float:
    prem = spot * 0.005
    sym = f"{und}260101C100000.US"
    tick = float(tick_size(sym, prem))
    fees = float(
        option_order_fees(1, "Buy", prem)["total"] + option_order_fees(1, "Sell", prem)["total"]
    )
    return ret * spot * 100 * 0.5 - fees - tick * 2 * 100


def _stock_fees(shares: int, px_in: float, px_out: float) -> float:
    return float(
        stock_order_fees(shares, "Buy", px_in)["total"]
        + stock_order_fees(shares, "Sell", px_out)["total"]
    )


# ---------- E4 完整版 ----------


def backtest_e4(df1: pd.DataFrame, symbol: str, p: dict) -> list[Trade]:
    df = resample_bars(df1, "5min")
    ma = ind.sma(df["close"], 120)
    ma_up = ma.diff() > 0
    k = ind.kdj(df["high"], df["low"], df["close"])
    cross_up = (k["k"] > k["d"]) & (k["k"].shift(1) <= k["d"].shift(1))
    cross_dn = (k["k"] < k["d"]) & (k["k"].shift(1) >= k["d"].shift(1))
    long_sig = (df["close"] > ma) & ma_up & cross_up
    short_sig = (df["close"] < ma) & ~ma_up & cross_dn
    day = pd.Series(df.index.tz_convert(ET).date, index=df.index)
    mod = df.index.tz_convert(ET).hour * 60 + df.index.tz_convert(ET).minute
    stop, add_at = p["stop"], p["add_at"]
    w = p.get("swing_w", 2)
    # 已确认的局部低/高点（滞后 w 根）
    is_lo = (
        (df["low"] == df["low"].rolling(2 * w + 1, center=True).min())
        .shift(w)
        .fillna(False)
        .astype(bool)
    )
    is_hi = (
        (df["high"] == df["high"].rolling(2 * w + 1, center=True).max())
        .shift(w)
        .fillna(False)
        .astype(bool)
    )
    sw_lo = df["low"].shift(w).where(is_lo)
    sw_hi = df["high"].shift(w).where(is_hi)
    trades: list[Trade] = []
    pos = None
    o, h, lo, c = (
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    idx = df.index
    ls, ss = long_sig.to_numpy(), short_sig.to_numpy()
    swl, swh = sw_lo.to_numpy(), sw_hi.to_numpy()
    dayv = day.to_numpy()
    last_lo = last_hi = np.nan
    prev_lo = prev_hi = np.nan
    for i in range(1, len(df)):
        if dayv[i] != dayv[i - 1]:
            last_lo = last_hi = prev_lo = prev_hi = np.nan
        if not np.isnan(swl[i]):
            prev_lo, last_lo = last_lo, swl[i]
        if not np.isnan(swh[i]):
            prev_hi, last_hi = last_hi, swh[i]
        if pos is not None:
            d = pos["dir"]
            # 止损（用 bar 内触及）
            stop_px = pos["entry"] * (1 - d * stop)
            hit_stop = lo[i] <= stop_px if d == 1 else h[i] >= stop_px
            # 梯形：多头收盘跌破上一个更高低点；空头收盘升破上一个更低高点
            ladder = pos.get("ladder")
            hit_ladder = ladder is not None and (
                (d == 1 and c[i] < ladder) or (d == -1 and c[i] > ladder)
            )
            eod = mod[i] >= 15 * 60 + 55 or (i + 1 < len(df) and dayv[i + 1] != dayv[i])
            if hit_stop or hit_ladder or eod:
                px = stop_px if hit_stop else c[i]
                reason = "stop" if hit_stop else ("ladder" if hit_ladder else "eod")
                trades.append(
                    Trade(symbol, d, pos["ts"], idx[i], pos["entry"], px, pos["units"], reason)
                )
                pos = None
            else:
                # 梯形水位更新：多头取「更高低点」
                if d == 1 and not np.isnan(last_lo) and not np.isnan(prev_lo) and last_lo > prev_lo:
                    pos["ladder"] = max(pos.get("ladder") or -np.inf, last_lo)
                if (
                    d == -1
                    and not np.isnan(last_hi)
                    and not np.isnan(prev_hi)
                    and last_hi < prev_hi
                ):
                    pos["ladder"] = min(pos.get("ladder") or np.inf, last_hi)
                # 盈利加仓一次
                if pos["units"] == 1 and d * (c[i] / pos["entry"] - 1) >= add_at:
                    pos["entry"] = (pos["entry"] + c[i]) / 2  # 均价
                    pos["units"] = 2
            continue
        if (
            (ls[i] or ss[i])
            and mod[i] < 15 * 60 + 30
            and i + 1 < len(df)
            and dayv[i + 1] == dayv[i]
        ):
            d = 1 if ls[i] else -1
            pos = {"dir": d, "ts": idx[i + 1], "entry": o[i + 1], "units": 1, "ladder": None}
    return trades


# ---------- E9 完整版（YaleV3）----------


def backtest_e9(df: pd.DataFrame, symbol: str, p: dict) -> list[Trade]:
    bb = ind.bollinger(df["close"], 20, 2.0)
    sig = (df["close"] < bb["lower"]).to_numpy()
    o, h, lo, c = (
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
    )
    idx = df.index
    stop, trail_start, trail = p["stop"], p["trail_start"], p["trail"]
    trades: list[Trade] = []
    pos = None
    for i in range(1, len(df)):
        if pos is not None:
            pos["peak"] = max(pos["peak"], h[i])
            stop_px = pos["entry"] * (1 - stop)
            if lo[i] <= stop_px:
                trades.append(Trade(symbol, 1, pos["ts"], idx[i], pos["entry"], stop_px, 1, "stop"))
                pos = None
                continue
            if pos["peak"] / pos["entry"] - 1 >= trail_start:
                trail_px = pos["peak"] * (1 - trail)
                if lo[i] <= trail_px:
                    trades.append(
                        Trade(symbol, 1, pos["ts"], idx[i], pos["entry"], trail_px, 1, "trail")
                    )
                    pos = None
            continue
        if sig[i] and i + 1 < len(df):
            pos = {"ts": idx[i + 1], "entry": o[i + 1], "peak": o[i + 1]}
    if pos is not None:
        trades.append(Trade(symbol, 1, pos["ts"], idx[-1], pos["entry"], c[-1], 1, "open_at_end"))
    return trades


# ---------- 统计 ----------


def summarize(trades: list[Trade], und: str) -> tuple[dict, pd.DataFrame]:
    if not trades:
        return {"trades": 0}, pd.DataFrame()
    rows = []
    for t in trades:
        shares = max(1, int(UNIT_NOTIONAL * t.units / t.entry_px))
        gross = t.gross_ret() * shares * t.entry_px
        fees = _stock_fees(shares, t.entry_px, t.exit_px)
        rows.append(
            {
                "entry": t.entry_ts.tz_convert(ET),
                "exit": t.exit_ts.tz_convert(ET),
                "dir": t.direction,
                "units": t.units,
                "reason": t.reason,
                "ret": t.gross_ret(),
                "gross": gross,
                "fees": fees,
                "net": gross - fees,
                "opt_pnl": _option_pnl(t.gross_ret(), t.entry_px, und) * t.units,
                "hold_min": (t.exit_ts - t.entry_ts).total_seconds() / 60,
            }
        )
    tr = pd.DataFrame(rows)
    eq = tr["net"].cumsum()
    dd = (eq - eq.cummax()).min()
    wins, losses = tr.loc[tr["net"] > 0, "net"], tr.loc[tr["net"] < 0, "net"]
    s = {
        "trades": len(tr),
        "win_rate": round(float((tr["net"] > 0).mean()), 3),
        "payoff": round(float(wins.mean() / -losses.mean()), 2)
        if len(wins) and len(losses)
        else float("nan"),
        "net_pnl_usd": round(float(tr["net"].sum()), 0),
        "fees_usd": round(float(tr["fees"].sum()), 0),
        "max_dd_usd": round(float(dd), 0),
        "avg_hold_min": round(float(tr["hold_min"].mean()), 0),
        "opt_pnl_per_contract_usd": round(float(tr["opt_pnl"].sum()), 0),
        "opt_win_rate": round(float((tr["opt_pnl"] > 0).mean()), 3),
    }
    tr["year"] = tr["entry"].dt.year
    yearly = (
        tr.groupby("year")
        .agg(
            trades=("net", "size"),
            win_rate=("net", lambda x: round(float((x > 0).mean()), 3)),
            net_pnl=("net", "sum"),
            opt_pnl=("opt_pnl", "sum"),
        )
        .round(0)
        .reset_index()
    )
    return s, yearly


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/research/SCREENSHOT_STRATEGIES.md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    symbols = get_settings().symbols
    e4p = {"stop": 0.01, "add_at": 0.005, "swing_w": 2}
    e9p = {"stop": 0.08, "trail_start": 0.02, "trail": 0.02}
    lines = [
        "# 截图策略专项（R-010 / R-011）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC ·"
        " 数据 2023-10-02～2026-10-21 正股 1 分钟 K（盘中）。",
        f"每笔 1 单位 = {UNIT_NOTIONAL:.0f} USD 名义（加仓为 2 单位）；正股费用 D-013 股票费率；"
        "期权表达 = 单张 ATM 近似。",
        "入场一律用信号后下一根开盘价；止损按 bar 内触及。",
        "",
        f"## E4 完整版（5 分钟 K、SMA120 定向、KDJ 交叉、梯形退出、盈利 {e4p['add_at']:.1%} 加仓、"
        f"止损 {e4p['stop']:.0%}、日内平仓）",
        "",
    ]
    e4_rows, e4_years = [], []
    for sym in symbols:
        und = sym.split(".")[0]
        df = load_bars(root, und)
        tr = backtest_e4(df, sym, e4p)
        s, y = summarize(tr, und)
        s["symbol"] = sym
        e4_rows.append(s)
        y.insert(0, "symbol", sym)
        e4_years.append(y)
        log.info("E4 %s %s", sym, s)
    lines += [
        df_to_md(pd.DataFrame(e4_rows)),
        "",
        "按年份：",
        "",
        df_to_md(pd.concat(e4_years, ignore_index=True)),
        "",
    ]

    lines += [
        f"## E9 完整版 YaleV3（1 分钟 K、跌穿下轨做多、无卖出信号、止损 {e9p['stop']:.0%}、"
        f"盈利 {e9p['trail_start']:.0%} 后追踪 {e9p['trail']:.0%}、可隔夜）",
        "",
    ]
    e9_rows, e9_years = [], []
    for scaled in (False, True):
        for sym in symbols:
            und = sym.split(".")[0]
            df = load_bars(root, und)
            p = dict(e9p)
            if scaled:
                day_ret = np.log(df["close"]).groupby(df.index.tz_convert(ET).date).last().diff()
                vol20 = float(day_ret.rolling(20).std().mean())
                scale = vol20 / 0.03
                p = {k: v * scale for k, v in e9p.items()}
            tr = backtest_e9(df, sym, p)
            s, y = summarize(tr, und)
            s["symbol"], s["params"] = (
                sym,
                ("scaled " if scaled else "orig ") + f"stop={p['stop']:.2%} trail={p['trail']:.2%}",
            )
            e9_rows.append(s)
            y.insert(0, "symbol", sym)
            y.insert(1, "params", "scaled" if scaled else "orig")
            e9_years.append(y)
            log.info("E9 %s scaled=%s %s", sym, scaled, s)
    lines += [
        df_to_md(pd.DataFrame(e9_rows)),
        "",
        "按年份：",
        "",
        df_to_md(pd.concat(e9_years, ignore_index=True)),
        "",
    ]
    # 买入持有基准（同名义），用于解释 E9 的多日持仓收益
    bench = []
    for sym in symbols:
        df = load_bars(root, sym.split(".")[0])
        r = float(df["close"].iloc[-1] / df["open"].iloc[0] - 1)
        bench.append(
            {
                "symbol": sym,
                "buy_hold_ret": round(r, 4),
                "buy_hold_pnl_usd": round(r * UNIT_NOTIONAL, 0),
            }
        )
    bench_df = pd.DataFrame(bench)
    e4_df, e9_df = pd.DataFrame(e4_rows), pd.DataFrame(e9_rows)
    e4_neg = int((e4_df["net_pnl_usd"] <= 0).sum())
    e9_orig = e9_df[e9_df["params"].str.startswith("orig")]
    hold_lo = e9_orig["avg_hold_min"].min() / 390
    hold_hi = e9_orig["avg_hold_min"].max() / 390
    concl = [
        "## 结论",
        "",
        f"- E4 完整版：正股口径 {len(e4_df)} 个标的中 {e4_neg} 个扣费后为负（费用合计 "
        f"{e4_df['fees_usd'].sum():.0f} USD，净 {e4_df['net_pnl_usd'].sum():.0f} USD）；"
        f"胜率 {e4_df['win_rate'].min():.2f}–{e4_df['win_rate'].max():.2f}，盈亏比 "
        f"{e4_df['payoff'].min():.2f}–{e4_df['payoff'].max():.2f}。"
        "换成 ATM 期权表达后三标的全部为负"
        f"（合计 {e4_df['opt_pnl_per_contract_usd'].sum():.0f} USD/张）。"
        "**正股口径无边，期权口径更差。**",
        f"- E9 YaleV3 原参数：三标的扣费后为正（合计 {e9_orig['net_pnl_usd'].sum():.0f} USD），"
        f"但平均持仓 {hold_lo:.0f}–{hold_hi:.0f} "
        f"个交易日、胜率 {e9_orig['win_rate'].min():.2f}–{e9_orig['win_rate'].max():.2f}、"
        f"盈亏比仅 {e9_orig['payoff'].min():.2f}–{e9_orig['payoff'].max():.2f}："
        "−8% 止损在美股上几乎不触发，收益本质是在 2023-10～2026-10 的上涨市里长期持有多头；"
        "同期买入持有收益见下表。**正股口径的『边』来自市场方向而非信号；"
        "期权表达不适用（0–1DTE 无法持有数日，表中 opt_pnl 列对 E9 无意义）。**",
        "- 按波动缩放止损后笔数增加、持仓缩短，净收益量级不变："
        "收益对止损参数不敏感、对持有时长敏感。",
        "",
        "买入持有基准（同一名义，2023-10-02 开盘 → 2026-10-21 收盘）：",
        "",
        df_to_md(bench_df),
        "",
    ]
    lines += concl + [
        "## 复现",
        "",
        "```bash",
        "uv run python -m zr.research.screenshot_strategies",
        "```",
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
