"""正股日内动量复现（R-045）→ docs/research/SPY_INTRADAY_MOMENTUM.md。

Zarattini、Aziz、Barbon《Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF
(SPY)》（SSRN 4824172，2025-02-03 版）原文规则，参数不改：
- 噪声区：move[d, m] = |Close(m) / Open(9:30) − 1|，σ[d, m] = 前 14 个同一分钟观测的均值
  （不含当日）；UB = max(开盘, 前收) × (1 + σ)，LB = min(开盘, 前收) × (1 − σ)。
- 只在 HH:00 / HH:30 判定（10:00 起，最后一个判定时刻早于收盘）：收盘 > UB 且 > VWAP → 多；
  收盘 < LB 且 < VWAP → 空；否则平仓。持仓到下一判定时刻，收盘全部平仓。
- 股数 = round(前日权益 / 开盘 × min(4, 2% / σ_日))，σ_日 = 前 14 日收盘收益样本标准差。
原文未写明、按作者公开 Python 代码（concretumgroup.com）取舍的：判定用 bar 标签 HH:MM 那根 1 分钟 K
的收盘并按该收盘成交；σ 按同一分钟前 14 个观测滚动。
另跑原文两个基础版本（反向带止损 100% 仓位；当前带 + VWAP 100% 仓位）作描述性对照。

数据：长桥 1 分钟 RTH K（NoAdjust，`load_bars`），VWAP = 累计成交额 / 累计成交量（RTH）。
成本：主口径 D-013 正股费用 + 每边 1 tick；
对照 = 原文 0.0035 USD/股（每单最低 0.35）+ 0.001 USD/股。

运行：uv run python -m zr.research.spy_intraday_momentum
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.costs.fees import stock_order_fees
from zr.data.calendar import session_close
from zr.features.bars import ET, load_bars
from zr.ops.markdown import df_to_md

log = logging.getLogger("zr.research.spy_intraday_momentum")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "SPY_INTRADAY_MOMENTUM.md"
LOOKBACK = 14
VM = 1.0
TARGET_VOL = 0.02
MAX_LEV = 4.0
TRADE_FREQ = 30
FIRST_CHECK = 30  # 10:00 = 开盘后 30 分钟（原文：第一笔在 10:00）
AUM0 = 100_000.0
TICK = 0.01
N_RANDOM = 1000
SEED = 0
UNDS = ("SPY", "QQQ")


# ---------------- 数据整理 ----------------


@dataclass
class Panel:
    days: list  # 交易日（升序）
    close: np.ndarray  # [日, 开盘后分钟]
    vwap: np.ndarray
    open_: np.ndarray  # 每日 09:30 开盘
    last_m: np.ndarray  # 每日最后一根 bar 的分钟序号（收盘平仓用）
    day_close: np.ndarray  # 每日收盘（最后一根 bar 的收盘）


def panel(bars: pd.DataFrame) -> Panel:
    """1 分钟 RTH K → 日 × 分钟矩阵。只保留完整交易日（最后一根 bar = 收盘前 1 分钟）。"""
    et = bars.index.tz_convert(ET)
    day = np.asarray(et.date)
    m = np.asarray((et.hour * 60 + et.minute) - (9 * 60 + 30))
    days = sorted(set(day))
    full = []
    for d in days:
        c = session_close(d)
        need = c.hour * 60 + c.minute - (9 * 60 + 30) - 1
        if m[day == d].max() >= need and m[day == d].min() == 0:
            full.append(d)
    idx = {d: i for i, d in enumerate(full)}
    keep = np.isin(day, full)
    n, width = len(full), 390
    close = np.full((n, width), np.nan)
    tov = np.zeros((n, width))
    vol = np.zeros((n, width))
    r = np.fromiter((idx[d] for d in day[keep]), int)
    cm = m[keep]
    close[r, cm] = bars["close"].to_numpy(float)[keep]
    tov[r, cm] = bars["turnover"].to_numpy(float)[keep] if "turnover" in bars else 0.0
    vol[r, cm] = bars["volume"].to_numpy(float)[keep]
    open_ = np.full(n, np.nan)
    first = bars[keep].groupby(day[keep])["open"].first()
    open_[[idx[d] for d in first.index]] = first.to_numpy(float)
    # 缺分钟：收盘向前填（VWAP 用累计量，缺分钟量为 0）
    close = pd.DataFrame(close).ffill(axis=1).to_numpy(copy=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.cumsum(tov, axis=1) / np.cumsum(vol, axis=1)
    last_m = np.array(
        [session_close(d).hour * 60 + session_close(d).minute - (9 * 60 + 30) - 1 for d in full]
    )
    for i, lm in enumerate(last_m):  # 收盘后的列置空
        close[i, lm + 1 :] = np.nan
        vwap[i, lm + 1 :] = np.nan
    day_close = close[np.arange(n), last_m]
    return Panel(full, close, vwap, open_, last_m, day_close)


def sigma_open(close: np.ndarray, open_: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    """σ[d, m] = 同一分钟 m 前 lookback 个观测的 |close/open − 1| 均值（不含当日）。"""
    move = np.abs(close / open_[:, None] - 1)
    out = np.full_like(move, np.nan)
    for j in range(move.shape[1]):
        col = move[:, j]
        ok = np.isfinite(col)
        s = pd.Series(col[ok]).rolling(lookback, min_periods=lookback).mean().shift(1)
        out[ok, j] = s.to_numpy()
    return out


def bands(p: Panel, sig: np.ndarray, vm: float = VM) -> tuple[np.ndarray, np.ndarray]:
    prev = np.r_[np.nan, p.day_close[:-1]]
    hi = np.fmax(p.open_, prev)  # 第一天无前收：只用开盘
    lo = np.fmin(p.open_, prev)
    return hi[:, None] * (1 + vm * sig), lo[:, None] * (1 - vm * sig)


def daily_vol(p: Panel, lookback: int = LOOKBACK) -> np.ndarray:
    """σ_日[d] = 前 lookback 日收盘收益的样本标准差（ddof=1，原文公式 1/13）。"""
    ret = pd.Series(p.day_close).pct_change()
    return ret.rolling(lookback, min_periods=lookback).std(ddof=1).shift(1).to_numpy()


def signal(c: float, ub: float, lb: float, vw: float, prev: int, variant: str) -> int:
    """判定时刻的目标方向。final / band_vwap：当前带 + VWAP；opp_band：反向带止损（原文基础版）。"""
    if variant == "opp_band":
        if c > ub:
            return 1
        if c < lb:
            return -1
        return prev
    if c > ub and c > vw:
        return 1
    if c < lb and c < vw:
        return -1
    return 0


# ---------------- 回测 ----------------


def order_cost(shares: int, side: str, px: float, cost: str) -> float:
    if shares <= 0:
        return 0.0
    if cost == "paper":
        return max(0.35, 0.0035 * shares) + 0.001 * shares
    return float(stock_order_fees(shares, side, px)["total"]) + TICK * shares


def backtest(
    p: Panel, variant: str = "final", cost: str = "main", first_check: int = FIRST_CHECK
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (逐日, 逐笔)。variant：final（动态仓位）/ band_vwap（100%）/ opp_band（100%）。"""
    sig = sigma_open(p.close, p.open_)
    ub, lb = bands(p, sig)
    dvol = daily_vol(p)
    aum = AUM0
    days, trades = [], []
    for i, d in enumerate(p.days):
        valid = np.isfinite(sig[i]).any() and (variant != "final" or np.isfinite(dvol[i]))
        if not valid:
            continue
        lev = min(MAX_LEV, TARGET_VOL / dvol[i]) if variant == "final" else 1.0
        shares = int(round(aum / p.open_[i] * lev))
        pos, entry_px, entry_m = 0, np.nan, -1
        pnl = 0.0
        checks = list(range(first_check, p.last_m[i], TRADE_FREQ))
        for m in checks + [p.last_m[i]]:
            c = p.close[i, m]
            if not np.isfinite(c):
                continue
            want = (
                0 if m == p.last_m[i] else signal(c, ub[i, m], lb[i, m], p.vwap[i, m], pos, variant)
            )
            if not np.isfinite(ub[i, m]) and m != p.last_m[i]:
                want = pos
            if want == pos:
                continue
            if pos != 0:  # 平仓
                gross = pos * shares * (c - entry_px)
                cst = order_cost(
                    shares, "Buy" if pos == 1 else "Sell", entry_px, cost
                ) + order_cost(shares, "Sell" if pos == 1 else "Buy", c, cost)
                trades.append(
                    {
                        "day": d,
                        "dir": pos,
                        "entry_m": entry_m,
                        "exit_m": m,
                        "shares": shares,
                        "entry_px": entry_px,
                        "exit_px": c,
                        "gross": gross,
                        "cost": cst,
                        "net": gross - cst,
                    }
                )
                pnl += gross - cst
            pos = want
            if pos != 0:
                entry_px, entry_m = c, m
        days.append({"day": d, "aum_prev": aum, "pnl": pnl, "ret": pnl / aum, "lev": lev})
        aum += pnl
    return pd.DataFrame(days), pd.DataFrame(trades)


# ---------------- 统计与对照 ----------------


def stats(ret: pd.Series, bench: pd.Series | None = None) -> dict:
    r = ret.to_numpy(float)
    eq = np.cumprod(1 + r)
    n = len(r)
    out = {
        "交易日": n,
        "总收益 %": round((eq[-1] - 1) * 100, 2) if n else np.nan,
        "年化 %": round((eq[-1] ** (252 / n) - 1) * 100, 2) if n else np.nan,
        "年化波动 %": round(float(np.std(r, ddof=1) * np.sqrt(252) * 100), 2) if n > 1 else np.nan,
        "Sharpe": round(float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)), 2)
        if n > 1 and np.std(r) > 0
        else np.nan,
        "最大回撤 %": round(float(np.max(1 - eq / np.maximum.accumulate(eq)) * 100), 2)
        if n
        else np.nan,
    }
    if bench is not None and n > 2:
        b = bench.reindex(ret.index).to_numpy(float)
        ok = np.isfinite(b)
        beta, alpha = np.polyfit(b[ok], r[ok], 1)
        out["alpha 年化 %"] = round(alpha * 252 * 100, 2)
        out["beta"] = round(float(beta), 3)
    return out


def buy_hold(p: Panel, start_i: int) -> pd.Series:
    c = pd.Series(p.day_close, index=p.days)
    return c.pct_change().iloc[start_i:].fillna(0.0)


def random_direction(trades: pd.DataFrame, n: int = N_RANDOM, seed: int = SEED) -> np.ndarray:
    """保持每笔入场 / 离场时刻、股数、成本，只把方向换成随机 ±1：每次重复的净盈亏合计（USD）。"""
    rng = np.random.default_rng(seed)
    move = (trades["shares"] * (trades["exit_px"] - trades["entry_px"])).to_numpy(float)
    cost = trades["cost"].to_numpy(float)
    d = rng.choice([-1.0, 1.0], size=(n, len(trades)))
    return (d * move - cost).sum(axis=1)


def by_year(days: pd.DataFrame, trades: pd.DataFrame, bh: pd.Series) -> pd.DataFrame:
    rows = []
    days = days.set_index(pd.to_datetime(days["day"]))
    tr = trades.assign(year=pd.to_datetime(trades["day"]).dt.year)
    for y, g in days.groupby(days.index.year):
        b = bh.reindex(g["day"]).to_numpy(float)
        t = tr[tr["year"] == y]
        rows.append(
            {
                "年份": int(y),
                "交易日": len(g),
                "策略收益 %": round((np.prod(1 + g["ret"]) - 1) * 100, 2),
                "策略 Sharpe": round(float(g["ret"].mean() / g["ret"].std() * np.sqrt(252)), 2)
                if g["ret"].std() > 0
                else np.nan,
                "买入持有 %": round((np.prod(1 + np.nan_to_num(b)) - 1) * 100, 2),
                "笔数": len(t),
                "做多净值 USD": round(float(t.loc[t["dir"] == 1, "net"].sum()), 2),
                "做空净值 USD": round(float(t.loc[t["dir"] == -1, "net"].sum()), 2),
                "成本 USD": round(float(t["cost"].sum()), 2),
            }
        )
    return pd.DataFrame(rows)


def side_table(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, name in ((1, "做多"), (-1, "做空")):
        t = trades[trades["dir"] == d]
        rows.append(
            {
                "方向": name,
                "笔数": len(t),
                "胜率": round(float((t["net"] > 0).mean()), 3) if len(t) else np.nan,
                "毛利 USD": round(float(t["gross"].sum()), 2),
                "成本 USD": round(float(t["cost"].sum()), 2),
                "净值 USD": round(float(t["net"].sum()), 2),
                "每笔净值 bp（占名义）": round(
                    float((t["net"] / (t["shares"] * t["entry_px"])).mean() * 1e4), 2
                )
                if len(t)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# 原文 FAQ Q24 逐年逐月表（最终版，SPY，% / 月），与本样本重叠的月份
PAPER_MONTHLY = {
    "2023-11": 0.5, "2023-12": 3.8, "2024-01": 8.8, "2024-02": -1.5, "2024-03": -0.4,
    "2024-04": 5.8, "2024-05": -4.3, "2024-06": 1.6, "2024-07": 8.2, "2024-08": -2.8,
    "2024-09": 4.1, "2024-10": 6.7, "2024-11": -2.6, "2024-12": 5.7,
}  # fmt: skip


def monthly(days: pd.DataFrame) -> pd.Series:
    s = days.set_index(pd.to_datetime(days["day"]))["ret"]
    return s.groupby(s.index.to_period("M")).apply(lambda x: (np.prod(1 + x) - 1) * 100)


def paper_overlap(p: Panel) -> tuple[pd.DataFrame, dict]:
    """诊断（不是检验结果）：原文成本下与原文月度表对照；另跑「09:30 也判定」看笔数差异来源。"""
    d, tr = backtest(p, "final", "paper")
    d0, t0 = backtest(p, "final", "paper", first_check=0)
    a, b = monthly(d), monthly(d0)
    rows = [
        {
            "月份": k,
            "原文 %": v,
            "本复现 %（10:00 起）": round(float(a.get(pd.Period(k), np.nan)), 2),
            "诊断 %（09:30 也判定）": round(float(b.get(pd.Period(k), np.nan)), 2),
        }
        for k, v in PAPER_MONTHLY.items()
    ]
    t = pd.DataFrame(rows)
    x = t["原文 %"].to_numpy(float)
    info = {
        "corr": round(float(np.corrcoef(x, t["本复现 %（10:00 起）"])[0, 1]), 3),
        "corr_930": round(float(np.corrcoef(x, t["诊断 %（09:30 也判定）"])[0, 1]), 3),
        "trades_per_day": round(len(tr) / len(d), 2),
        "trades_per_day_930": round(len(t0) / len(d0), 2),
        "sum_paper": round(float(np.prod(1 + x / 100) - 1) * 100, 1),
        "sum_ours": round(float(np.prod(1 + t["本复现 %（10:00 起）"] / 100) - 1) * 100, 1),
        "sum_930": round(float(np.prod(1 + t["诊断 %（09:30 也判定）"] / 100) - 1) * 100, 1),
    }
    return t, info


def run(root: Path) -> dict:
    res = {}
    for u in UNDS:
        p = panel(load_bars(root, u))
        out = {"panel_days": (p.days[0], p.days[-1], len(p.days))}
        for variant in ("final", "band_vwap", "opp_band"):
            for cost in ("main", "paper"):
                days, trades = backtest(p, variant, cost)
                out[(variant, cost)] = (days, trades)
        d0, t0 = out[("final", "main")]
        start = p.days.index(d0["day"].iloc[0])
        bh = buy_hold(p, start)
        out["bh"] = bh
        out["random"] = random_direction(t0)
        if u == "SPY":
            out["overlap"] = paper_overlap(p)
        res[u] = out
        log.info("%s：%d 天，最终版主口径 %d 笔", u, len(d0), len(t0))
    return res


# ---------------- 报告 ----------------

PAPER = {  # 原文 Table 3 / Table 4（2007-05～2024-04，SPY，0.0035 + 0.001 USD/股）
    "final": "总收益 1,985%、年化 19.6%、波动 14.3%、Sharpe 1.33、日胜率 43%、最大回撤 25%、"
    "alpha 19.6%、beta −0.07；7,668 笔、1.8 笔/日",
    "band_vwap": "总收益 380%、年化 9.7%、波动 7.7%、Sharpe 1.24、最大回撤 12%",
    "opp_band": "总收益 178%、年化 6.2%、波动 10.9%、Sharpe 0.61、最大回撤 21%",
}
VARIANT_NAME = {
    "final": "当前带 + VWAP、动态仓位（原文最终版，主检验）",
    "band_vwap": "当前带 + VWAP、100% 仓位",
    "opp_band": "反向带止损、100% 仓位（原文基础版）",
}


def overlap_lines(res: dict) -> list[str]:
    if "overlap" not in res.get("SPY", {}):
        return []
    t, info = res["SPY"]["overlap"]
    return [
        "与原文 FAQ Q24 月度表重叠的 14 个月（SPY，最终版，**原文成本**；诊断，不是检验结果）：",
        "",
        df_to_md(t),
        "",
        f"14 个月复利：原文 {info['sum_paper']}%，本复现 {info['sum_ours']}%；月度相关系数 "
        f"{info['corr']}。笔数：本复现 {info['trades_per_day']} 笔（开平一次）/ 日，原文 Table 4 "
        "为 1.8——作者代码按 |仓位变化| 计单数（开、平各算一次），折成单数本复现约为其 2 倍，"
        "与 1.8 接近（未验证：原文未定义 Trades 的口径）。"
        f"诊断：若 09:30 那根也判定（作者代码 `min_from_open % 30 == 0` 含 0；原文正文写第一笔在 "
        f"10:00），{info['trades_per_day_930']} 笔 / 日、14 个月复利 {info['sum_930']}%、相关系数 "
        f"{info['corr_930']}。主检验按原文正文（10:00 起），不因诊断改动。",
        "",
    ]


def conclusion_lines(res: dict) -> list[str]:
    rows = []
    for u, r in res.items():
        d0, t0 = r[("final", "main")]
        s = stats(d0.set_index("day")["ret"], r["bh"])
        bh = stats(r["bh"])
        actual = float(t0["net"].sum())
        pct = float((r["random"] < actual).mean() * 100)
        sides = side_table(t0).set_index("方向")["净值 USD"]
        ok = (
            s["总收益 %"] > 0
            and s["Sharpe"] > 0.5
            and pct >= 95
            and sides["做多"] > 0
            and sides["做空"] > 0
        )
        rows.append(
            {
                "标的": u,
                "总收益 %": s["总收益 %"],
                "Sharpe": s["Sharpe"],
                "买入持有 %": bh["总收益 %"],
                "买入持有 Sharpe": bh["Sharpe"],
                "随机方向分位": round(pct, 1),
                "做多净值 USD": sides["做多"],
                "做空净值 USD": sides["做空"],
                "预测成立": "是" if ok else "否",
            }
        )
    return [
        "## 结论（最终版、主口径成本 D-013 + 1 tick）",
        "",
        df_to_md(pd.DataFrame(rows)),
        "",
        "预登记预测：总收益 > 0、Sharpe > 0.5、实际方向优于随机方向 95 分位、多空都为正，"
        "两个标的都满足才算成立。",
        "",
    ]


def report(res: dict) -> str:
    lines = [
        "# 正股日内动量复现（R-045）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} · "
        "复现 `uv run python -m zr.research.spy_intraday_momentum`。",
        "",
        "原文：Zarattini, C., Aziz, A., Barbon, A.,《Beat the Market: An Effective Intraday Mom"
        "entum "
        "Strategy for S&P500 ETF (SPY)》，SSRN 4824172，2025-02-03 版"
        "（<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172>；全文取自 "
        "<https://www.alexandria.unisg.ch/handle/20.500.14171/122124>）。"
        "原文未写明的实现细节按作者公开代码页面"
        "（<https://www.concretumgroup.com/python-backtesting-beat-the-market-an-effective-intr"
        "aday-"
        "momentum-strategy-for-the-sp500-etf-spy/>）。规则与预测见 RESEARCH_LOG R-045（运行前登"
        "记）。",
        "",
        *conclusion_lines(res),
        "## 规则（原文，参数不改）",
        "",
        "1. 噪声区：对每个分钟 HH:MM，move = |Close(HH:MM) / Open(9:30) − 1|；σ = 前 14 个交易日"
        "同一分钟 move 的均值（不含当日）。UB = max(当日开盘, 前收) × (1 + σ)，"
        "LB = min(当日开盘, 前收) × (1 − σ)，波动乘数 1。",
        "2. 只在 HH:00 / HH:30 判定，10:00 起到收盘前最后一个半点：收盘 > UB 且 > VWAP → 多；"
        "收盘 < LB 且 < VWAP → 空；否则平仓（= 原文「当前带或 VWAP 作追踪止损」）。"
        "持仓到下一判定时刻，收盘全部平仓。VWAP 只用 RTH。",
        "3. 股数 = round(前日权益 / 当日开盘 × min(4, 2% / σ_日))，σ_日 = 前 14 日收盘收益样本标"
        "准差；"
        "初始 100,000 USD。",
        "4. 成本：主口径 D-013 正股费用（`stock_order_fees`，做空开仓为 Sell）+ 每边 1 tick（0."
        "01）；"
        "对照 = 原文 0.0035 USD/股（每单最低 0.35）+ 0.001 USD/股。",
        "",
        "实现取舍（原文未写明）：判定与成交都用 bar 标签 HH:MM 那根 1 分钟 K 的收盘（作者代码写法；"
        "原文滑点实验按 HH:MM 那分钟的开盘成交，两者差 1 分钟）；σ 按同一分钟前 14 个观测滚动"
        "（提前收盘日没有的分钟不占窗口）；VWAP = 累计成交额 / 累计成交量；前收不做股息调整"
        "（作者代码用股息调整后的前收，长桥 NoAdjust 数据没有调整，未验证影响，每年约 4 个除息日"
        "）；"
        "收盘平仓用最后一根 1 分钟 K（15:59，提前收盘日 12:59）的收盘。",
        "",
    ]
    for u, r in res.items():
        d0, t0 = r[("final", "main")]
        bh = r["bh"]
        ret = d0.set_index("day")["ret"]
        a, b, n = r["panel_days"]
        lines += [
            f"## {u}",
            "",
            f"数据：长桥 1 分钟 RTH K，完整交易日 {n} 天（{a} ~ {b}），前 14 天只用于预热；"
            f"策略交易日 {len(d0)} 天（{d0['day'].iloc[0]} ~ {d0['day'].iloc[-1]}）。",
            "",
        ]
        rows = []
        for v in ("final", "band_vwap", "opp_band"):
            for c in ("main", "paper"):
                dd, tt = r[(v, c)]
                s = stats(dd.set_index("day")["ret"], bh)
                tr_days = dd[dd["pnl"] != 0]
                rows.append(
                    {
                        "版本": VARIANT_NAME[v],
                        "成本": "D-013 + 1 tick" if c == "main" else "原文 0.0035 + 0.001",
                        **s,
                        "日胜率（有交易日）": round(float((tr_days["ret"] > 0).mean()), 3),
                        "笔数": len(tt),
                        "笔 / 日": round(len(tt) / len(dd), 2),
                        "成本合计 USD": round(float(tt["cost"].sum()), 2),
                    }
                )
        rows.append({"版本": "买入持有（无股息）", "成本": "—", **stats(bh)})
        lines += [df_to_md(pd.DataFrame(rows)), ""]
        rnd = r["random"]
        actual = float(t0["net"].sum())
        pct = float((rnd < actual).mean() * 100)
        lines += [
            "随机方向对照（最终版、主口径成本；保持每笔时刻、股数与成本，只随机方向，"
            f"种子 {SEED}、{N_RANDOM} 次；按每笔净盈亏 USD 合计、不复利）：实际 {actual:,.2f} USD；"
            f"随机中位数 {np.median(rnd):,.2f}、95 分位 {np.percentile(rnd, 95):,.2f}；"
            f"实际位于随机的 {pct:.1f} 分位。",
            "",
            "多空拆分（最终版、主口径）：",
            "",
            df_to_md(side_table(t0)),
            "",
            "按年份（最终版、主口径）：",
            "",
            df_to_md(by_year(d0, t0, bh)),
            "",
        ]
        _ = ret
    lines += [
        "## 与原文对照",
        "",
        *overlap_lines(res),
        "原文（SPY，IQFeed 1 分钟 K，2007-05～2024-04，成本 0.0035 + 0.001 USD/股）：",
        "",
        *[f"- {VARIANT_NAME[k]}：{v}" for k, v in PAPER.items()],
        "- 原文 FAQ Q4 / Q24 的逐年表：2023 年 +37.2%、2024 年 +32.2%（最终版，SPY）。",
        "",
        "本复现与原文的样本期只在 2023-10～2024-04 重叠；原文 17 年里的大部分收益来自高波动年份"
        "（2008 +63.4%、2018 +61.1%），本样本期不含这类年份。",
        "",
        "## 边界",
        "",
        "- 标的：SPY、QQQ；数据：长桥 1 分钟 RTH K（NoAdjust），2023-10-02 起到本地最后一个完整"
        "交易日；不含股息（策略与买入持有都不含）。",
        "- 成交：判定那根 1 分钟 K 的收盘价成交，不模拟盘口与市场冲击；成本只有费用 + 每边 1 ti"
        "ck。",
        "- 未覆盖：原文的 VIX 分层、日线形态、星期效应、gamma 代理等延伸分析；参数敏感性（回看期、"
        "波动乘数）——预登记要求不改参数，未做。",
        "- 随机方向对照只换方向，保留了策略自己的入场时刻与持仓时长。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    res = run(root)
    OUT.write_text(report(res), encoding="utf-8")
    out = root / "research" / "spy_momentum"
    out.mkdir(parents=True, exist_ok=True)
    for u, r in res.items():
        r[("final", "main")][1].to_parquet(out / f"trades_{u}.parquet")
    log.info("写入 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
