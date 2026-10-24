"""调研三条必测规则（survey_rsi2_pullback / survey_ema_pullback / survey_orb_breakout）的正股模拟
（负责人插队任务）→ 输出每条策略 × 两种出场的按周扣费后盈亏表（人民币）。

- 期间：MU 2025-01-02～2026-05-31、2026-09-01～2026-10-21（有秒级平价价的交易日；D-033 窗口不含）。
- 资金：初始 1 万人民币，按 USD/CNY = 6.6725 折美元（XE 中间价，2026-10-23 18:20 UTC，
  https://www.xe.com/en-us/currencyconverter/convert/?Amount=1&From=USD&To=CNY）；每次信号用全部可用
  资金买入（做多）或卖空（做空）整数股，同一时间一笔，买不起 1 股跳过；盈亏滚入资金。卖空不计借券费
  与保证金要求（未覆盖）。
- 信号：R-049 第一层算好的逐秒入场条件（owner_rules_oos 的 conds 缓存，与 R-049 同一套），条件由假变
  真的那一秒为一次信号；持仓期间的信号忽略；ORB 每日至多 1 次。
- 成交：信号（或出场触发）后 2 秒的秒级价（平价反推，前值填充）± 1 个 tick（0.01 美元）不利滑点；
  费用 D-013 正股（zr/costs/fees.stock_order_fees，按订单）。
- 出场原版：RSI(2) 回调 = 已完成 2 分钟 K 收盘按方向重新越过 SMA5；均线回调 = 30 分钟上限（原版无
  出场）；ORB = 区间另一端止损、10 倍风险止盈。统一版：正股朝交易方向 +0.3% 全平、反向 −0.2% 全平、
  30 分钟上限。全部 15:50 ET 清仓，不过夜。
- 按周（周一起）汇总；测试期内没有交易的周计 0。
运行：uv run python -m zr.research.survey_stock_sim
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from zr.backtest import intrabar as ib
from zr.config import get_settings
from zr.costs.fees import stock_order_fees
from zr.features import entry_features as ef
from zr.research import owner_rules_oos as o

USDCNY = 6.6725
CAPITAL_CNY = 10_000.0
TICK = 0.01
LAG = 2
HOLD = 1800
RULES = ("survey_rsi2_pullback", "survey_ema_pullback", "survey_orb_breakout")
EXITS = ("原版", "统一 +0.3% / −0.2% / 30 分钟")


def _fee(n: int, side: str, px: float) -> float:
    return float(stock_order_fees(n, side, px)["total"])


def day_trades(
    name: str, rule: dict, exit_kind: str, g: pd.DataFrame, cond: dict, svy: dict, cash: float
) -> tuple[list[dict], float]:
    S = pd.Series(g["S"].to_numpy()).ffill().to_numpy()
    t = g["t"].to_numpy()
    n = len(S)
    _, in_win = ib.entry_conditions(rule, g, None, None)
    hm = np.asarray(pd.DatetimeIndex(t).tz_localize("UTC").tz_convert(ib.ET).strftime("%H:%M:%S"))
    flat = int(np.searchsorted(hm, "15:50:00"))
    edges = {d: c & in_win & ~np.r_[False, (c & in_win)[:-1]] for d, c in cond.items()}
    sig = np.zeros(n, int)
    for d, e in edges.items():
        sig[e & (sig == 0)] = d
    sma5 = {d: svy["svy_sma5_dist_bp"].directional(np.full(n, d)) for d in (1, -1)}
    seg = g["S"].to_numpy()[:300]
    hi, lo = (np.nanmax(seg), np.nanmin(seg)) if np.isfinite(seg).any() else (math.nan, math.nan)
    out, k, cap = [], 0, int(rule.get("max_entries_per_day") or 10**6)
    while k < n and len(out) < cap:
        nz = np.flatnonzero(sig[k:flat])
        if not len(nz):
            break
        k = k + int(nz[0])
        d = int(sig[k])
        e = k + LAG
        if e >= flat or not np.isfinite(S[e]):
            k += 1
            continue
        px0 = S[e] + d * TICK
        q = int(cash // px0)
        if q < 1:
            k += 1
            continue
        path = d * (S[e + 1 : flat + 1] / px0 - 1)
        j_rel = len(path) - 1  # 默认 15:50
        if exit_kind == EXITS[1]:
            hit = np.flatnonzero((path >= 0.003) | (path <= -0.002))
            j_rel = min(j_rel, int(hit[0]) if len(hit) else j_rel, HOLD - 1)
        elif name == "survey_rsi2_pullback":
            hit = np.flatnonzero(sma5[d][e + 1 : flat + 1] > 0)
            j_rel = min(j_rel, int(hit[0]) if len(hit) else j_rel)
        elif name == "survey_ema_pullback":
            j_rel = min(j_rel, HOLD - 1)
        elif name == "survey_orb_breakout":
            stop = lo if d == 1 else hi
            risk = d * (px0 - stop)
            if risk > 0:
                tgt = px0 + d * 10 * risk
                pp = S[e + 1 : flat + 1]
                hit = np.flatnonzero((d * (pp - stop) <= 0) | (d * (pp - tgt) >= 0))
                j_rel = min(j_rel, int(hit[0]) if len(hit) else j_rel)
        x_sig = e + 1 + j_rel
        x = min(x_sig + LAG, n - 1) if x_sig < flat else flat
        px1 = S[x] - d * TICK
        f_in = _fee(q, "Buy" if d == 1 else "Sell", px0)
        f_out = _fee(q, "Sell" if d == 1 else "Buy", px1)
        pnl = d * q * (px1 - px0) - f_in - f_out
        cash += pnl
        out.append(
            {"entry_ns": int(t[e]), "dir": d, "qty": q, "px0": px0, "px1": px1, "pnl_usd": pnl}
        )
        k = x + 1
    return out, cash


def run(root: Path) -> pd.DataFrame:
    rules = {k: v for k, v in o.load_rules().items() if k in RULES}
    days = o.oos_days(root)
    ctx = ef.Ctx.load(root)
    rows = []
    cash = {(r, x): CAPITAL_CNY / USDCNY for r in RULES for x in EXITS}
    for day in days:
        g = ib.second_grid(root, "MU", day)
        if g is None:
            continue
        dd = ef.load_day(ctx, day)
        svy = ef.svy_features(dd)
        for name, rule in rules.items():
            cond = o.load_conds(root, day, name, len(g))
            if cond is None:
                continue
            for x in EXITS:
                tr, cash[(name, x)] = day_trades(name, rule, x, g, cond, svy, cash[(name, x)])
                rows += [{"rule": name, "exit": x, "day": day, **r} for r in tr]
    return pd.DataFrame(rows)


def _max_dd(sub: pd.DataFrame) -> str:
    """逐笔权益曲线（含初始资金）的最大回撤：金额（元）与占峰值比例。"""
    eq = (
        CAPITAL_CNY
        + np.r_[0.0, np.cumsum(sub.sort_values("entry_ns")["pnl_usd"].to_numpy())] * USDCNY
    )
    peak = np.maximum.accumulate(eq)
    i = int(np.argmax(peak - eq))
    return f"{peak[i] - eq[i]:.0f} 元（{(peak[i] - eq[i]) / peak[i]:.1%}）"


def weekly_table(tr: pd.DataFrame, days: list[dt.date]) -> pd.DataFrame:
    weeks = sorted({d - dt.timedelta(days=d.weekday()) for d in days})
    out = []
    for (r, x), sub in tr.groupby(["rule", "exit"]):
        wk = sub.assign(w=[d - dt.timedelta(days=d.weekday()) for d in sub["day"]])
        s = wk.groupby("w")["pnl_usd"].sum().reindex(weeks, fill_value=0.0) * USDCNY
        out.append(
            {
                "策略": r.replace("survey_", ""),
                "出场": x,
                "交易笔数": len(sub),
                "周数": len(s),
                "每周中位数（元）": round(s.median(), 2),
                "每周平均（元）": round(s.mean(), 2),
                "最好一周（元）": round(s.max(), 2),
                "最差一周（元）": round(s.min(), 2),
                "亏损周占比": f"{(s < 0).mean():.0%}",
                "期末权益（元）": round(CAPITAL_CNY + s.sum(), 2),
                "总收益率": f"{s.sum() / CAPITAL_CNY:+.1%}",
                "最大回撤": _max_dd(sub),
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    root = get_settings().zr_data_root
    p = o.store(root) / "survey_stock_sim.parquet"
    if p.exists() and "--rerun" not in __import__("sys").argv:
        tr = pd.read_parquet(p)
    else:
        tr = run(root)
        tr.to_parquet(p, index=False)
    print(weekly_table(tr, o.oos_days(root)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
