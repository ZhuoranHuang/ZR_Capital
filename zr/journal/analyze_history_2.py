"""交易史第二轮（docs/TRADE_HISTORY_REPORT_2.md）。

输入：journal parquet（trade_history.py）→ analyze_history.fifo_round_trips 的
157 笔往返（全部多头期权）。
内容：
1. 累计净盈亏曲线，标出最大 5 笔亏损（docs/research/trade_history_cum_pnl.png）。
2. 最长连续盈利区间，及其交易特征 vs 其余交易
   （时段、DTE、持仓、张数、价外程度、当日标的涨跌与日内波动）。
3. 三种口径重算：
   a. 剔除最大 5 笔亏损；
   b. 30 分钟时间止损：持仓 >30 分钟的，按入场后第 30 分钟该合约 NBBO 的 bid 卖出；
   c. 30% 权利金止损：入场后逐分钟看 NBBO 中间价，首次 ≤ 入场价 ×0.7 时按该分钟 bid 卖出
      （跨日持仓逐日拉取）。
   b/c 用 ThetaData 历史 1 分钟 NBBO（真实报价，不是近似）；费用沿用实际费用（张数不变，
   卖出价变化只影响 SEC/TAF，<0.05 USD/张，忽略）。
4. 随机入场对照 1,000 次：同日期范围、同笔数、同标的、同 call/put 与张数；入场日与分钟随机，
   持仓时长从真实分布抽样。期权盈亏用 delta 模型近似：
   张数 ×100×0.5×(S_exit − S_entry)×方向 − D-013 费用，不含 theta；
   真实交易用同一模型计算后求分位，保证可比。只含有正股 1 分钟 K 的标的（MU、QQQ）。
5. 盈亏分布：偏度、峰度、大亏笔数占比。
运行：uv run python -m zr.journal.analyze_history_2
      （需 Theta Terminal 运行；报价缓存在 ZR_DATA_ROOT/journal/option_paths/）
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.features.bars import load_bars
from zr.journal.analyze_history import fifo_round_trips, load
from zr.ops.markdown import df_to_md

log = logging.getLogger("zr.journal.analyze_history_2")
ET = ZoneInfo("America/New_York")
SESSIONS = [
    ("09:30–10:00", 570, 600),
    ("10:00–11:30", 600, 690),
    ("11:30–14:00", 690, 840),
    ("14:00–15:30", 840, 930),
    ("15:30–16:00", 930, 961),
]
DELTA = 0.5
N_RANDOM = 1000


# ---------- 期权报价路径（ThetaData，带缓存）----------


def _right_word(r: str) -> str:
    return "call" if r.upper().startswith("C") else "put"


def option_path(root: Path, trip: pd.Series, client) -> pd.DataFrame:
    """入场日到出场日之间该合约的 1 分钟 NBBO（ts UTC，bid/ask/mid），按 (合约, 日期) 缓存。"""
    cache = root / "journal" / "option_paths"
    cache.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(trip["entry_et"].date(), trip["exit_et"].date())
    parts = []
    for d in days:
        f = cache / f"{trip['symbol']}_{d.date().isoformat()}.parquet"
        if f.exists():
            q = pd.read_parquet(f)
        else:
            q = client.history_quote(
                trip["und"],
                trip["expiry"],
                d.date(),
                strike=f"{trip['strike']:.3f}",
                right=_right_word(trip["right"]),
            )
            q = (
                q[["ts", "bid", "ask"]]
                if not q.empty
                else pd.DataFrame(columns=["ts", "bid", "ask"])
            )
            q.to_parquet(f, index=False)
        parts.append(q)
    q = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "bid", "ask"])
    q = q[(q["bid"] > 0) & (q["ask"] > 0)].copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    return q.sort_values("ts").reset_index(drop=True)


def time_stop_exit(trip: pd.Series, path: pd.DataFrame, minutes: int = 30) -> tuple[float, str]:
    """持仓 > minutes：入场后第 minutes 分钟的 bid；否则原出场。返回 (出场价, 说明)。"""
    if trip["hold_min"] <= minutes:
        return trip["exit_px"], "unchanged"
    t = trip["entry_et"].tz_convert("UTC") + pd.Timedelta(minutes=minutes)
    after = path[path["ts"] >= t.ceil("min")]
    if after.empty:
        return trip["exit_px"], "no_quote"
    return float(after["bid"].iloc[0]), "time_stop"


def premium_stop_exit(trip: pd.Series, path: pd.DataFrame, stop: float = 0.30) -> tuple[float, str]:
    """入场后（不含入场分钟）到原出场前，mid 首次 ≤ 入场价×(1−stop) 时按该分钟 bid 出场。"""
    start = trip["entry_et"].tz_convert("UTC").floor("min") + pd.Timedelta(minutes=2)
    end = trip["exit_et"].tz_convert("UTC")
    w = path[(path["ts"] >= start) & (path["ts"] <= end)]
    hit = w[w["mid"] <= trip["entry_px"] * (1 - stop)]
    if hit.empty:
        return trip["exit_px"], "unchanged" if len(w) else "no_quote"
    return float(hit["bid"].iloc[0]), "premium_stop"


def recompute(trips: pd.DataFrame, exits: pd.Series) -> pd.Series:
    """出场价替换后的净盈亏（费用沿用实际）。"""
    return (exits - trips["entry_px"]) * 100 * trips["qty"] - trips["fees_actual"]


# ---------- 统计 ----------


def stats(pnl: pd.Series) -> dict:
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "trips": len(pnl),
        "net_usd": round(float(pnl.sum()), 2),
        "win_rate": round(float((pnl > 0).mean()), 3) if len(pnl) else np.nan,
        "payoff": round(float(wins.mean() / -losses.mean()), 2)
        if len(wins) and len(losses)
        else np.nan,
        "avg_usd": round(float(pnl.mean()), 2) if len(pnl) else np.nan,
    }


def longest_win_streak(win: pd.Series) -> tuple[int, int]:
    """返回最长连续 True 的 [start, end] 位置（含）。"""
    best, cur_start, best_span = 0, None, (0, -1)
    for i, w in enumerate(win.to_numpy()):
        if w:
            cur_start = i if cur_start is None else cur_start
            if i - cur_start + 1 > best:
                best, best_span = i - cur_start + 1, (cur_start, i)
        else:
            cur_start = None
    return best_span


def session_of(ts: pd.Series) -> pd.Series:
    mod = ts.dt.hour * 60 + ts.dt.minute
    out = pd.Series("盘外", index=ts.index)
    for name, lo, hi in SESSIONS:
        out[(mod >= lo) & (mod < hi)] = name
    return out


def attach_context(trips: pd.DataFrame, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """加：入场时段、价外程度（%）、入场日标的涨跌、日内波动（年化）。无正股 K 的标的为 NaN。"""
    t = trips.copy()
    t["session"] = session_of(t["entry_et"])
    t["spot_entry"] = np.nan
    t["day_ret"] = np.nan
    t["day_rv"] = np.nan
    for und, b in bars.items():
        m = t["und"] == und
        if not m.any():
            continue
        t.loc[m, "spot_entry"] = _asof(b["close"], t.loc[m, "entry_et"].dt.tz_convert("UTC"))
        day = pd.Series(b.index.tz_convert(ET).date, index=b.index)
        g = b.groupby(day.to_numpy())
        dret = g["close"].last() / g["open"].first() - 1
        drv = np.log(b["close"]).diff().groupby(day.to_numpy()).std() * np.sqrt(252 * 390)
        ed = t.loc[m, "entry_et"].dt.date
        t.loc[m, "day_ret"] = ed.map(dret).to_numpy()
        t.loc[m, "day_rv"] = ed.map(drv).to_numpy()
    sign = np.where(t["right"].str.upper().str.startswith("C"), 1, -1)
    t["otm_pct"] = sign * (t["strike"] / t["spot_entry"] - 1) * 100  # >0 价外
    return t


def profile(t: pd.DataFrame) -> dict:
    return {
        "trips": len(t),
        "dte_mean": round(float(t["dte"].mean()), 2),
        "dte_median": float(t["dte"].median()) if len(t) else np.nan,
        "dte0_share": round(float((t["dte"] == 0).mean()), 2),
        "hold_min_median": round(float(t["hold_min"].median()), 1),
        "qty_mean": round(float(t["qty"].mean()), 2),
        "otm_pct_median": round(float(t["otm_pct"].median()), 2),
        "day_ret_median_pct": round(float(t["day_ret"].median() * 100), 2),
        "day_rv_median": round(float(t["day_rv"].median()), 3),
        "call_share": round(float(t["right"].str.upper().str.startswith("C").mean()), 2),
        "top_session": t["session"].value_counts().idxmax() if len(t) else "",
    }


# ---------- 随机对照 ----------


def _asof(close: pd.Series, ts_utc) -> np.ndarray:
    # 正股 K 线索引是 us 精度（parquet），随机出场时间带 ns 小数；统一转 ns 再比较
    idx = pd.DatetimeIndex(close.index).as_unit("ns")
    ts = pd.DatetimeIndex(pd.to_datetime(ts_utc, utc=True)).as_unit("ns")
    i = idx.searchsorted(ts, side="right") - 1
    return close.to_numpy()[np.clip(i, 0, len(close) - 1)]


def model_pnl(und_close: pd.Series, entry_utc, exit_utc, sign, qty, prem) -> np.ndarray:
    ds = _asof(und_close, exit_utc) - _asof(und_close, entry_utc)
    fees = np.array(
        [
            float(
                option_order_fees(int(q), "Buy", p)["total"]
                + option_order_fees(int(q), "Sell", p)["total"]
            )
            for q, p in zip(qty, prem, strict=True)
        ]
    )
    return qty * 100 * DELTA * ds * sign - fees


def random_control(
    trips: pd.DataFrame, bars: dict[str, pd.DataFrame], n: int, seed: int = 0
) -> dict:
    sub = trips[trips["und"].isin(bars)].reset_index(drop=True)
    if sub.empty:
        return {}
    sign = np.where(sub["right"].str.upper().str.startswith("C"), 1.0, -1.0)
    qty, prem = sub["qty"].to_numpy(), sub["entry_px"].to_numpy()
    real_model = 0.0
    for und, g in sub.groupby("und"):
        i = g.index.to_numpy()
        real_model += model_pnl(
            bars[und]["close"],
            g["entry_et"].dt.tz_convert("UTC").to_numpy(),
            g["exit_et"].dt.tz_convert("UTC").to_numpy(),
            sign[i],
            qty[i],
            prem[i],
        ).sum()
    holds = trips["hold_min"].to_numpy()  # 全部 157 笔的真实持仓分布
    lo = trips["entry_et"].min().date()
    hi = trips["exit_et"].max().date()
    rng = np.random.default_rng(seed)
    # 每个标的的可选入场分钟（盘中、日期范围内）
    cand = {}
    for und, b in bars.items():
        d = b.index.tz_convert(ET)
        m = (d.date >= lo) & (d.date <= hi)
        cand[und] = b.index[m]
    totals = np.empty(n)
    # 费用与随机无关（张数、权利金固定），预先算
    fee_cache = {}
    for und, g in sub.groupby("und"):
        i = g.index.to_numpy()
        fee_cache[und] = np.array(
            [
                float(
                    option_order_fees(int(q), "Buy", p)["total"]
                    + option_order_fees(int(q), "Sell", p)["total"]
                )
                for q, p in zip(qty[i], prem[i], strict=True)
            ]
        )
    for k in range(n):
        tot = 0.0
        for und, g in sub.groupby("und"):
            i = g.index.to_numpy()
            c = cand[und]
            entry = c[rng.integers(0, len(c), len(i))]
            exit_ = entry + pd.to_timedelta(rng.choice(holds, len(i)), unit="m")
            close = bars[und]["close"]
            ds = _asof(close, exit_.to_numpy()) - _asof(close, entry.to_numpy())
            tot += float((qty[i] * 100 * DELTA * ds * sign[i] - fee_cache[und]).sum())
        totals[k] = tot
    return {
        "covered_trips": len(sub),
        "covered_unds": sorted(sub["und"].unique()),
        "real_actual_net": round(float(sub["pnl_net"].sum()), 2),
        "real_model_net": round(float(real_model), 2),
        "random_mean": round(float(totals.mean()), 2),
        "random_p05": round(float(np.percentile(totals, 5)), 2),
        "random_p50": round(float(np.percentile(totals, 50)), 2),
        "random_p95": round(float(np.percentile(totals, 95)), 2),
        "real_model_percentile": round(float((totals <= real_model).mean() * 100), 1),
        "n": n,
    }


# ---------- 主流程 ----------


def plot_cum(t: pd.DataFrame, top5: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = t.sort_values("exit_et").reset_index(drop=True)
    cum = s["pnl_net"].cumsum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s["exit_et"], cum, lw=1.2)
    for _, r in top5.iterrows():
        j = s.index[(s["exit_et"] == r["exit_et"]) & (s["symbol"] == r["symbol"])][0]
        ax.scatter(s.loc[j, "exit_et"], cum[j], color="red", zorder=3)
        ax.annotate(
            f"{r['pnl_net']:.0f}",
            (s.loc[j, "exit_et"], cum[j]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
            color="red",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Cumulative net P&L (USD), red = 5 largest losses")
    ax.set_ylabel("USD")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def build_report(root: Path, out_md: Path, n_random: int) -> str:
    from zr.data.thetadata import ThetaClient

    t0 = time.monotonic()
    _, execs, charges = load(root)
    trips, unmatched = fifo_round_trips(execs, charges)
    trips = trips.sort_values("exit_et").reset_index(drop=True)
    bars = {}
    for und in sorted(trips["und"].unique()):
        try:
            bars[und] = load_bars(root, und)
        except FileNotFoundError:
            continue
    t = attach_context(trips, bars)

    # 1. 累计曲线
    top5 = t.nsmallest(5, "pnl_net")
    png = out_md.parent / "research" / "trade_history_cum_pnl.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    plot_cum(t, top5, png)
    cum = t["pnl_net"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    # 2. 最长连胜
    s0, s1 = longest_win_streak(t["pnl_net"] > 0)
    streak = t.iloc[s0 : s1 + 1]
    rest = t.drop(index=streak.index)
    prof = pd.DataFrame(
        [
            {"group": "最长连胜", **profile(streak)},
            {"group": "其余", **profile(rest)},
            {"group": "全部", **profile(t)},
        ]
    )
    sess = pd.crosstab(t.index.isin(streak.index), t["session"], normalize="index").round(2)
    sess.index = ["其余", "最长连胜"] if len(sess) == 2 else sess.index

    # 3. 三种口径
    client = ThetaClient(concurrency=1)
    ts_exit, ps_exit, ts_note, ps_note = [], [], [], []
    for _, r in t.iterrows():
        path = option_path(root, r, client)
        e1, n1 = time_stop_exit(r, path)
        e2, n2 = premium_stop_exit(r, path)
        ts_exit.append(e1), ts_note.append(n1), ps_exit.append(e2), ps_note.append(n2)
    t["exit_time_stop"], t["note_time_stop"] = ts_exit, ts_note
    t["exit_prem_stop"], t["note_prem_stop"] = ps_exit, ps_note
    pnl_ts = recompute(t, t["exit_time_stop"])
    pnl_ps = recompute(t, t["exit_prem_stop"])
    variants = pd.DataFrame(
        [
            {"口径": "实际", **stats(t["pnl_net"])},
            {"口径": "剔除最大 5 笔亏损", **stats(t.drop(index=top5.index)["pnl_net"])},
            {
                "口径": "30 分钟时间止损",
                **stats(pnl_ts),
                "触发笔数": int((t["note_time_stop"] == "time_stop").sum()),
                "无报价": int((t["note_time_stop"] == "no_quote").sum()),
            },
            {
                "口径": "30% 权利金止损",
                **stats(pnl_ps),
                "触发笔数": int((t["note_prem_stop"] == "premium_stop").sum()),
                "无报价": int((t["note_prem_stop"] == "no_quote").sum()),
            },
        ]
    )

    # 4. 随机对照
    rc = random_control(t, bars, n_random)

    # 5. 分布
    pnl = t["pnl_net"]
    loss_med = float((-pnl[pnl < 0]).median())
    big = pnl < -2 * loss_med
    worst_decile = pnl.nsmallest(max(1, len(pnl) // 10))
    dist = {
        "skew": round(float(skew(pnl)), 3),
        "excess_kurtosis": round(float(kurtosis(pnl)), 3),
        "loss_median_usd": round(loss_med, 2),
        "big_loss_def": "亏损 > 2 × 亏损中位数",
        "big_loss_count": int(big.sum()),
        "big_loss_share_of_trips": round(float(big.mean()), 3),
        "big_loss_share_of_total_loss": round(float(pnl[big].sum() / pnl[pnl < 0].sum()), 3),
        "worst_10pct_share_of_total_loss": round(float(worst_decile.sum() / pnl[pnl < 0].sum()), 3),
    }

    cols = ["symbol", "entry_et", "exit_et", "hold_min", "qty", "entry_px", "exit_px", "pnl_net"]
    fmt = lambda df: df[cols].assign(  # noqa: E731
        entry_et=df["entry_et"].dt.strftime("%Y-%m-%d %H:%M"),
        exit_et=df["exit_et"].dt.strftime("%Y-%m-%d %H:%M"),
        hold_min=df["hold_min"].round(0),
        pnl_net=df["pnl_net"].round(2),
    )
    lines = [
        "# 交易史报告（第二轮）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"往返 {len(t)} 笔（全部为多头期权），"
        f"未配对单边 {len(unmatched)} 笔不计入 · 耗时 {time.monotonic() - t0:.0f}s",
        "",
        "## 1. 累计净盈亏",
        "",
        f"![累计净盈亏](research/{png.name})",
        "",
        f"合计 {pnl.sum():.2f} USD；累计曲线最大回撤 {max_dd:.2f} USD。最大 5 笔亏损：",
        "",
        df_to_md(fmt(top5)),
        "",
        f"这 5 笔合计 {top5['pnl_net'].sum():.2f} USD，占全部亏损的 "
        f"{top5['pnl_net'].sum() / pnl[pnl < 0].sum():.1%}。",
        "",
        "## 2. 最长连续盈利区间",
        "",
        f"按出场时间排序，最长连胜 {len(streak)} 笔："
        f"{streak['exit_et'].min():%Y-%m-%d %H:%M} ～ {streak['exit_et'].max():%Y-%m-%d %H:%M}，"
        f"合计 {streak['pnl_net'].sum():.2f} USD。",
        "价外程度 otm_pct：>0 为价外（call 行权价高于现价、put 低于现价），用入场分钟的正股收盘价；"
        "day_ret / day_rv 为入场当日标的开盘→收盘涨跌与 1 分钟收益年化波动。"
        "无正股 K 的标的（DRAM 等）不计入这三列。DTE 均值被少数 2027-01 长期合约拉高，看中位数。",
        "",
        df_to_md(prof),
        "",
        "入场时段分布（行内占比）：",
        "",
        df_to_md(sess.reset_index().rename(columns={"index": "group"})),
        "",
        "连胜区间明细：",
        "",
        df_to_md(fmt(streak)),
        "",
        "## 3. 三种口径重算",
        "",
        "b、c 使用 ThetaData 历史 1 分钟 NBBO（每笔合约逐日拉取并缓存），卖出按 bid；"
        "费用沿用实际费用。",
        "时间止损只影响持仓 >30 分钟的交易；权利金止损在入场后第 2 分钟起检查，"
        "以中间价触发、bid 成交；「无报价」的交易保留实际出场。",
        "",
        df_to_md(variants),
        "",
        "## 4. 随机入场对照（1,000 次）",
        "",
        "方法：同日期范围、同笔数、同标的、同 call/put 与张数，入场日与分钟随机（盘中），"
        "持仓时长从全部 157 笔的真实分布抽样；"
        f"盈亏 = 张数 × 100 × {DELTA} × 正股价差 × 方向 − D-013 费用"
        "（不含 theta，真实与随机同一模型）。",
        "",
        df_to_md(pd.DataFrame([rc])) if rc else "（无可用正股数据）",
        "",
        (
            f"**真实交易（模型口径）位于随机分布的第 {rc['real_model_percentile']} 百分位**；"
            f"同一批交易的实际净盈亏为 {rc['real_actual_net']:.2f} USD"
            f"（模型口径 {rc['real_model_net']:.2f}，差异来自 theta、实际 delta 与成交价）。"
            f"随机分布 5%–95% 区间 [{rc['random_p05']:.0f}, {rc['random_p95']:.0f}] 很宽"
            "（MU 价格高、抽样持仓含多日），检验力弱；"
            "结论只能是「与随机入场不可区分」，不能反过来说明择时有害或有益。"
        )
        if rc
        else "",
        "",
        "## 5. 盈亏分布",
        "",
        df_to_md(pd.DataFrame([dist])),
        "",
        "## 复现",
        "",
        "```bash",
        "bash deploy/sync_from_vps.sh --now        # journal parquet",
        "bash scripts/run_theta_terminal.sh        # 另开终端，b/c 需要",
        "uv run python -m zr.journal.analyze_history_2",
        "```",
        "",
    ]
    text = "\n".join(lines)
    out_md.write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/TRADE_HISTORY_REPORT_2.md")
    p.add_argument("--n-random", type=int, default=N_RANDOM)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_report(get_settings().zr_data_root, Path(a.out), a.n_random)
    print(f"写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
