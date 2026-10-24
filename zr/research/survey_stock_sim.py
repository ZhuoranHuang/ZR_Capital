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
    name: str,
    rule: dict,
    exit_kind: str,
    g: pd.DataFrame,
    cond: dict,
    svy: dict,
    cash: float,
    fees: bool = True,
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
        f_in = _fee(q, "Buy" if d == 1 else "Sell", px0) if fees else 0.0
        f_out = _fee(q, "Sell" if d == 1 else "Buy", px1) if fees else 0.0
        pnl = d * q * (px1 - px0) - f_in - f_out
        cash += pnl
        out.append(
            {
                "entry_ns": int(t[e]),
                "dir": d,
                "qty": q,
                "px0": px0,
                "px1": px1,
                "pnl_usd": pnl,
                "fees_usd": f_in + f_out,
            }
        )
        k = x + 1
    return out, cash


def run(root: Path, capital_usd: float | None = None, fees: bool = True) -> pd.DataFrame:
    """capital_usd=None：1 万人民币按 USDCNY 折美元（原版）；否则用给定美元初始资金。"""
    rules = {k: v for k, v in o.load_rules().items() if k in RULES}
    days = o.oos_days(root)
    ctx = ef.Ctx.load(root)
    rows = []
    c0 = CAPITAL_CNY / USDCNY if capital_usd is None else capital_usd
    cash = {(r, x): c0 for r in RULES for x in EXITS}
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
                tr, cash[(name, x)] = day_trades(name, rule, x, g, cond, svy, cash[(name, x)], fees)
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


def buy_hold(root: Path, capital: float, days: list[dt.date]) -> pd.DataFrame:
    """买入持有 MU（长桥前复权日线）：每个测试区间首日开盘全仓买入整数股（+1 tick、D-013 费用），
    区间末日收盘卖出（−1 tick），下一区间用新权益重新买入；按日收盘计权益。"""
    d = pd.read_parquet(root / "diag" / "daily_fwd" / "MU.parquet")
    d["day"] = pd.to_datetime(d["ts"]).dt.date  # 时间戳为 naive UTC 04:00 / 05:00，日期即美东交易日
    d = d.set_index("day").sort_index()
    eq, cash, fees_tot = [], capital, 0.0
    for a, b in o.RANGES:
        seg = d[(d.index >= a) & (d.index <= b) & d.index.isin(days)]
        if seg.empty:
            continue
        px0 = float(seg["open"].iloc[0]) + TICK
        q = int(cash // px0)
        f0 = _fee(q, "Buy", px0)
        left = cash - q * px0 - f0
        for day, r in seg.iterrows():
            eq.append((day, left + q * float(r["close"])))
        px1 = float(seg["close"].iloc[-1]) - TICK
        f1 = _fee(q, "Sell", px1)
        cash = left + q * px1 - f1
        fees_tot += f0 + f1
        eq[-1] = (eq[-1][0], cash)
    s = pd.Series(dict(eq))
    return s, fees_tot


def usd_table(
    root: Path, tr: pd.DataFrame, tr0: pd.DataFrame, days: list[dt.date], capital: float
) -> pd.DataFrame:
    """10 万美元版：每条策略 × 出场按周（美元）+ 扣费前总收益率、费用占比；另加买入持有 MU。"""
    weeks = sorted({x - dt.timedelta(days=x.weekday()) for x in days})
    out = []

    def dd(eq: np.ndarray) -> str:
        peak = np.maximum.accumulate(eq)
        i = int(np.argmax(peak - eq))
        return f"{peak[i] - eq[i]:,.0f}（{(peak[i] - eq[i]) / peak[i]:.1%}）"

    for (r, x), sub in tr.groupby(["rule", "exit"]):
        wk = sub.assign(w=[v - dt.timedelta(days=v.weekday()) for v in sub["day"]])
        s = wk.groupby("w")["pnl_usd"].sum().reindex(weeks, fill_value=0.0)
        g0 = tr0[(tr0["rule"] == r) & (tr0["exit"] == x)]["pnl_usd"].sum()
        eq = capital + np.r_[0.0, np.cumsum(sub.sort_values("entry_ns")["pnl_usd"].to_numpy())]
        out.append(
            {
                "策略": r.replace("survey_", ""),
                "出场": x,
                "交易笔数": len(sub),
                "每周中位数": round(s.median(), 2),
                "每周平均": round(s.mean(), 2),
                "最好一周": round(s.max(), 2),
                "最差一周": round(s.min(), 2),
                "亏损周占比": f"{(s < 0).mean():.0%}",
                "期末权益": round(capital + s.sum(), 2),
                "总收益率（扣费后）": f"{s.sum() / capital:+.1%}",
                "扣费前总收益率": f"{g0 / capital:+.1%}",
                "费用占比（总费用 / 初始资金）": f"{sub['fees_usd'].sum() / capital:.1%}",
                "最大回撤": dd(eq),
            }
        )
    bh, fees_bh = buy_hold(root, capital, days)
    wk = bh.groupby([v - dt.timedelta(days=v.weekday()) for v in bh.index]).last()
    wchg = wk.diff().fillna(wk.iloc[0] - capital).reindex(weeks, fill_value=0.0)
    out.append(
        {
            "策略": "买入持有 MU",
            "出场": "区间末收盘",
            "交易笔数": 2 * len(o.RANGES),
            "每周中位数": round(wchg.median(), 2),
            "每周平均": round(wchg.mean(), 2),
            "最好一周": round(wchg.max(), 2),
            "最差一周": round(wchg.min(), 2),
            "亏损周占比": f"{(wchg < 0).mean():.0%}",
            "期末权益": round(float(bh.iloc[-1]), 2),
            "总收益率（扣费后）": f"{bh.iloc[-1] / capital - 1:+.1%}",
            "扣费前总收益率": f"{(bh.iloc[-1] + fees_bh) / capital - 1:+.1%}",
            "费用占比（总费用 / 初始资金）": f"{fees_bh / capital:.2%}",
            "最大回撤": dd(np.r_[capital, bh.to_numpy()]),
        }
    )
    return pd.DataFrame(out)


def main_usd(capital: float = 100_000.0) -> pd.DataFrame:
    root = get_settings().zr_data_root
    st = o.store(root)
    p1, p0 = st / "survey_stock_sim_usd.parquet", st / "survey_stock_sim_usd_nofee.parquet"
    if not p1.exists():
        run(root, capital, True).to_parquet(p1, index=False)
    if not p0.exists():
        run(root, capital, False).to_parquet(p0, index=False)
    tr, tr0 = pd.read_parquet(p1), pd.read_parquet(p0)
    for x in (tr, tr0):
        x["day"] = pd.to_datetime(x["day"]).dt.date
    return usd_table(root, tr, tr0, o.oos_days(root), capital)


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


def write_report() -> None:
    """docs/research/SURVEY_STOCK_SIM.md：1 万人民币与 10 万美元两档的表（读落盘结果）。"""
    from zr.ops.markdown import df_to_md

    root = get_settings().zr_data_root
    st = o.store(root)
    days = o.oos_days(root)
    tr = pd.read_parquet(st / "survey_stock_sim.parquet")
    tr["day"] = pd.to_datetime(tr["day"]).dt.date
    L = [
        "# 调研三条规则的正股模拟（满仓复利）",
        "",
        "`zr/research/survey_stock_sim.py`。信号 = R-049 第一层的逐秒入场条件（条件由假变真的秒"
        "），信号后 2 秒"
        "秒级价 ± 1 tick 成交，D-013 正股费用，做空不计借券费与保证金（未覆盖），同一时间一笔、"
        "整数股、"
        "15:50 清仓。测试期 MU 2025-01-02～2026-05-31、2026-09-01～2026-10-21（388 个交易日），"
        "按周汇总、"
        "无交易的周计 0。价格变动与收益率为 %。",
        "",
        f"## 初始 1 万人民币（USD/CNY {USDCNY}，XE 2026-10-23 中间价），金额为人民币",
        "",
        df_to_md(weekly_table(tr, days)),
        "",
        "## 初始 10 万美元，金额为美元",
        "",
        df_to_md(pd.read_parquet(st / "survey_stock_sim_usd_table.parquet")),
        "",
        "扣费前总收益率：同样规则、费用置 0 的独立复利模拟（交易序列可能因权益不同而略有差异）。"
        "费用占比 = 扣费版全部订单费用之和 / 初始资金。买入持有 MU：长桥前复权日线，每个测试区间"
        "首日开盘"
        "全仓买入、末日收盘卖出，再进入下一区间。",
        "",
    ]
    (Path(__file__).resolve().parents[2] / "docs" / "research" / "SURVEY_STOCK_SIM.md").write_text(
        "\n".join(L), encoding="utf-8"
    )
