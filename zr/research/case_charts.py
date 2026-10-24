# ruff: noqa: E501  （HTML 模板与图上说明文字为长字符串）
"""owner_rule_2 交易案例图（正股视角）→ docs/research/cases/。

交易：owner_rule_2 触发（R-049 定义、口径 C），出场时刻按 R-052 选定组（期权中间价
+6.5% / −30% / 30 分钟上限，负责人挂单策略；只用来确定出场时间，只保留期权实际成交
的信号）。入场价 / 出场价 = 触发 / 出场时刻的正股价；盈亏 = 正股朝交易方向的涨跌（%）。
- MU：R-052 第 2 步落盘的交易（research/exit_finalize/option.parquet，负责人挂单执行），
  正股价 = 秒级平价反推价。
- QQQ：没有秒级正股价与期权逐笔缓存 → 用 1 分钟 K（已完成 K 的收盘）与本地 1 分钟期权
  报价（options_1m）在分钟网格上复现 owner_rule_2 的三个条件与 R-052 出场（盘口 = 1 分钟
  报价，只有报价越过挂单价才成交）。
每个标的取盈利、亏损各最近 5 笔（按时间倒序，不挑选）。图：2 分钟 K（口径 C：含盘前盘后
连续、不按日重置）+ BOLL(20,2) + MA5 / MA10，副图成交量（20 日同时段均量）、MACD(6,13,6)、
KDJ(9,3,3)、RSI(2) / RSI(14)（Wilder）。
运行：uv run python -m zr.research.case_charts
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from zr.backtest import intrabar as ib  # noqa: E402
from zr.backtest.fill_limit import Book  # noqa: E402
from zr.config import get_settings  # noqa: E402
from zr.data.calendar import trading_days  # noqa: E402
from zr.data.thetadata import lb_symbol  # noqa: E402
from zr.features import entry_features as ef  # noqa: E402
from zr.features.bars import ET, load_extended_bars, resample_bars  # noqa: E402
from zr.features.indicators import kdj, wilder_rsi  # noqa: E402
from zr.research import exit_finalize as xf  # noqa: E402
from zr.research import exit_mining as xm  # noqa: E402

log = logging.getLogger("zr.research.case_charts")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "cases"
NS = 1_000_000_000
N_EACH = 5
MU_RANGE = (dt.date(2026, 2, 1), dt.date(2026, 4, 28))
QQQ_RANGE = (dt.date(2026, 9, 1), dt.date(2026, 10, 21))
CAL = (dt.date(2026, 6, 1), dt.date(2026, 8, 31))
# owner_rule_2（config/intrabar/owner_rule_2.yaml）
RET600_BP = 14.133933
OPT_CHG = -0.380505
REASON = {"tp": "止盈", "sl": "止损", "cap": "到时", "flat": "清仓（15:50 / 0DTE 10:45）"}
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------- MU ----------------


def mu_trades(root: Path) -> pd.DataFrame:
    o = pd.read_parquet(root / "research" / "exit_finalize" / "option.parquet")
    o = o[(o["rule"] == "owner_rule_2") & (o["exec"] == "owner") & (o["filled"] > 0)].copy()
    o["day"] = pd.to_datetime(o["day"]).dt.date
    o = o[(o["day"] >= MU_RANGE[0]) & (o["day"] <= MU_RANGE[1])]
    rows = []
    ctx = ef.Ctx.load(root)
    ext = load_extended_bars(root, "MU", clean=False)
    h1 = _macd_hist(ext["close"])  # 1 分钟 MACD(6,13,6) 柱原值（口径 C）
    h1_close = pd.DatetimeIndex(h1.index).as_unit("ns").asi8 + 60 * NS
    for day, g in o.groupby("day"):
        grid = ib.second_grid(root, "MU", day)
        S = pd.Series(grid["S"].to_numpy()).ffill().to_numpy()
        t = grid["t"].to_numpy()
        d = ef.load_day(ctx, day)
        fs = ef.day_features(d, ("bar", "sec", "opt"))
        for r in g.itertuples():
            k0 = int(np.searchsorted(t, r.decision_ns, side="right")) - 1
            k1 = int(np.searchsorted(t, r.exit_ns, side="right")) - 1
            dd = np.array([r.dir])
            rows.append(
                {
                    "und": "MU",
                    "day": day,
                    "dir": int(r.dir),
                    "entry_ns": int(r.decision_ns),
                    "exit_ns": int(r.exit_ns),
                    "px0": float(S[k0]),
                    "px1": float(S[k1]),
                    "reason": r.reason,
                    "symbol": r.symbol,
                    "c_ret600_pct": float(ef._dir_rows(fs["s_ret600_bp"], np.array([k0]), dd)[0])
                    / 100,
                    "c_macd_side": float(
                        ef._dir_rows(fs["b1_m6_cross_side"], np.array([k0]), dd)[0]
                    ),
                    "c_macd_hist": float(
                        h1.iloc[int(np.searchsorted(h1_close, r.decision_ns, side="right")) - 1]
                    ),
                    "c_opt_chg_pct": float(
                        ef._dir_rows(fs["o_mid_chg60s_pct"], np.array([k0]), dd)[0]
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df["ret_pct"] = df["dir"] * (df["px1"] / df["px0"] - 1) * 100
    return df


# ---------------- QQQ（分钟网格） ----------------


def _macd_hist(c: pd.Series) -> pd.Series:
    dif = c.ewm(span=6, adjust=False).mean() - c.ewm(span=13, adjust=False).mean()
    return dif - dif.ewm(span=6, adjust=False).mean()


def qqq_day(root: Path, day: dt.date, ext: pd.DataFrame, rng) -> list[dict]:
    p = root / "options_1m" / "underlying=QQQ" / f"date={day}" / "quotes.parquet"
    if not p.exists():
        return []
    q = pd.read_parquet(p)
    q["expiry"] = pd.to_datetime(q["expiry"]).dt.date
    q = q[(q["bid"] > 0) & (q["ask"] >= q["bid"])]
    q["t"] = pd.DatetimeIndex(q["ts"]).as_unit("ns").asi8
    q["mid"] = (q["bid"] + q["ask"]) / 2
    e1 = ext[(ext.index.tz_convert(ET).date <= day)]
    hist = _macd_hist(e1["close"])  # 1 分钟 MACD(6,13,6) 柱，口径 C
    rth = e1[(e1.index.tz_convert(ET).date == day)]
    rth = rth[
        (rth.index.tz_convert(ET).time >= dt.time(9, 30))
        & (rth.index.tz_convert(ET).time < dt.time(16, 0))
    ]
    if len(rth) < 30:
        return []
    close_ns = pd.DatetimeIndex(rth.index).as_unit("ns").asi8 + 60 * NS
    c = rth["close"].to_numpy(float)
    h = hist.reindex(rth.index).to_numpy()
    front = min(e for e in q["expiry"].unique() if e >= day)
    qf = q[q["expiry"] == front]
    books: dict = {}

    def book(sym):
        if sym not in books:
            x = q[q["symbol"] == sym]
            books[sym] = Book.from_frames(x) if len(x) else None
        return books[sym]

    def leg_mid(right, strike, tns):
        x = qf[(qf["right"] == right) & (qf["strike"] == strike)]
        i = np.searchsorted(x["t"].to_numpy(), tns, side="right") - 1
        return float(x["mid"].to_numpy()[i]) if i >= 0 else math.nan

    t_grid = np.arange(
        pd.Timestamp(f"{day} 09:30", tz=ET).value, pd.Timestamp(f"{day} 16:00", tz=ET).value, NS
    )
    flat = int(np.searchsorted(t_grid, pd.Timestamp(f"{day} 15:50", tz=ET).value))
    out, busy_until, last = [], 0, {1: -(10**18), -1: -(10**18)}
    prev = {1: False, -1: False}
    for i in range(10, len(rth)):
        tn = int(close_ns[i])
        hm = pd.Timestamp(tn, tz="UTC").tz_convert(ET).time()
        for d in (1, -1):
            ret600 = d * (c[i] / c[i - 10] - 1) * 1e4
            side = d * np.sign(h[i]) if np.isfinite(h[i]) else -1
            ks = np.sort(qf["strike"].unique())
            atm = float(ks[np.argmin(np.abs(ks - c[i]))])
            right = "call" if d == 1 else "put"
            m1, m0 = leg_mid(right, atm, tn), leg_mid(right, atm, tn - 60 * NS)
            chg = (m1 / m0 - 1) * 100 if m0 and np.isfinite(m0) and np.isfinite(m1) else math.nan
            ok = ret600 >= RET600_BP and side >= 0 and np.isfinite(chg) and chg >= OPT_CHG
            ok = ok and dt.time(9, 32) <= hm <= dt.time(15, 40)
            edge = ok and not prev[d]
            prev[d] = ok
            if not edge or tn < busy_until or tn - last[d] < 300 * NS:
                continue
            last[d] = tn
            exps = sorted(e for e in q["expiry"].unique() if e >= day)
            if hm > dt.time(10, 15):
                exps = [e for e in exps if e != day]
            if not exps:
                continue
            e = exps[0]
            ke = np.sort(q[q["expiry"] == e]["strike"].unique())
            cand = ke[ke > c[i]] if d == 1 else ke[ke < c[i]][::-1]
            if not len(cand):
                continue
            sym = lb_symbol("QQQ", e, right, float(cand[0]))
            bk = book(sym)
            if bk is None:
                continue
            n, px0, te, _ = ib.place(
                bk, 1, 1.0, tn, {"mode": "owner"}, sym, 0.9, int(t_grid[flat]), rng
            )
            if n <= 0 or te is None:
                busy_until = tn
                continue
            i0 = int(np.searchsorted(t_grid, te, side="right"))
            stop = flat
            if e == day:
                stop = min(
                    stop, int(np.searchsorted(t_grid, pd.Timestamp(f"{day} 10:45", tz=ET).value))
                )
            f = xm.state_arrays(bk, t_grid, i0, px0)["st_float_pct"]
            j, why = xf.exit_index(f, i0, stop, 30.0, 30.0)
            tdec = int(t_grid[min(j, len(t_grid) - 1)])
            order = {"mode": "taker", "lag_s": 2.0} if why == "flat" else {"mode": "owner"}
            xn, px1, xt, _ = ib.place(bk, -1, 1.0, tdec, order, sym, 0.9, None, rng)
            xt = xt or tdec
            busy_until = xt
            ji = int(np.searchsorted(close_ns, xt, side="right")) - 1
            out.append(
                {
                    "und": "QQQ",
                    "day": day,
                    "dir": d,
                    "entry_ns": tn,
                    "exit_ns": int(xt),
                    "px0": float(c[i]),
                    "px1": float(c[max(ji, i)]),
                    "reason": why,
                    "symbol": sym,
                    "c_ret600_pct": ret600 / 100,
                    "c_macd_side": float(side),
                    "c_macd_hist": float(h[i]),
                    "c_opt_chg_pct": chg,
                }
            )
    return out


def qqq_trades(root: Path, lo: dt.date, hi: dt.date) -> pd.DataFrame:
    ext = load_extended_bars(root, "QQQ", clean=False)
    rng = np.random.default_rng(0)
    rows = []
    for day in trading_days(lo, hi):
        if CAL[0] <= day <= CAL[1]:
            continue
        rows += qqq_day(root, day, ext, rng)
    df = pd.DataFrame(rows)
    if len(df):
        df["ret_pct"] = df["dir"] * (df["px1"] / df["px0"] - 1) * 100
    return df


# ---------------- 选取 ----------------


def pick(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("entry_ns", ascending=False)
    w = df[df["ret_pct"] > 0].head(N_EACH).assign(group="盈利")
    lo = df[df["ret_pct"] < 0].head(N_EACH).assign(group="亏损")
    return pd.concat([w, lo], ignore_index=True)


# ---------------- 作图 ----------------


def bars2(root: Path, und: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    one = load_extended_bars(root, und, clean=False)
    b = resample_bars(one, "2min")
    c = b["close"]
    x = pd.DataFrame(index=b.index)
    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    x["mid"], x["up"], x["lo"] = mid, mid + 2 * sd, mid - 2 * sd
    x["ma5"], x["ma10"] = c.rolling(5).mean(), c.rolling(10).mean()
    dif = c.ewm(span=6, adjust=False).mean() - c.ewm(span=13, adjust=False).mean()
    dea = dif.ewm(span=6, adjust=False).mean()
    x["dif"], x["dea"], x["hist"] = dif, dea, dif - dea
    k = kdj(b["high"], b["low"], c, 9, 3, 3)
    x["k"], x["d"], x["j"] = k["k"], k["d"], k["j"]
    x["rsi2"], x["rsi14"] = wilder_rsi(c, 2), wilder_rsi(c, 14)
    et = b.index.tz_convert(ET)
    slot = et.hour * 60 + et.minute
    day = et.date
    v = pd.DataFrame({"slot": slot, "day": day, "v": b["volume"].to_numpy()}, index=b.index)
    piv = v.pivot_table(index="day", columns="slot", values="v", aggfunc="sum")
    avg = piv.rolling(20, min_periods=5).mean().shift(1)  # 前 20 个交易日同时段均量（不含当日）
    x["vavg"] = [
        avg.at[d, s] if d in avg.index and s in avg.columns else np.nan
        for d, s in zip(day, slot, strict=True)
    ]
    return b, x


def chart(tr: dict, b: pd.DataFrame, x: pd.DataFrame, path: Path) -> None:
    lab = pd.DatetimeIndex(b.index)
    t_open = lab.as_unit("ns").asi8
    ie = int(np.searchsorted(t_open, tr["entry_ns"], side="right")) - 1
    ix = int(np.searchsorted(t_open, tr["exit_ns"], side="right")) - 1
    a, z = max(ie - 20, 0), min(ix + 10, len(b) - 1)
    bb, xx = b.iloc[a : z + 1], x.iloc[a : z + 1]
    n = len(bb)
    xs = np.arange(n)
    fig = plt.figure(figsize=(16, 12), dpi=100)
    gs = fig.add_gridspec(5, 1, height_ratios=[5, 1.3, 1.5, 1.3, 1.3], hspace=0.08)
    ax = [fig.add_subplot(gs[i]) for i in range(5)]
    for k in range(1, 5):
        ax[k].sharex(ax[0])
    up = bb["close"] >= bb["open"]
    for i, (o_, h_, l_, c_) in enumerate(bb[["open", "high", "low", "close"]].to_numpy()):
        col = "#16a34a" if up.iloc[i] else "#dc2626"
        ax[0].vlines(i, l_, h_, color=col, lw=1)
        ax[0].add_patch(Rectangle((i - 0.35, min(o_, c_)), 0.7, max(abs(c_ - o_), 1e-6), color=col))
    ax[0].plot(xs, xx["mid"], color="#2563eb", lw=1.2, label="BOLL 中轨")
    ax[0].plot(xs, xx["up"], color="#f97316", lw=1.0, label="上轨")
    ax[0].plot(xs, xx["lo"], color="#16a34a", lw=1.0, label="下轨")
    ax[0].plot(xs, xx["ma5"], color="#a855f7", lw=0.9, label="MA5")
    ax[0].plot(xs, xx["ma10"], color="#eab308", lw=0.9, label="MA10")
    for axx in ax:
        axx.axvspan(ie - a - 0.5, ie - a + 0.5, color="#3b82f6", alpha=0.15)
        axx.axvspan(ix - a - 0.5, ix - a + 0.5, color="#f59e0b", alpha=0.2)
    for px, i, col, txt in (
        (tr["px0"], ie - a, "#1d4ed8", "入场"),
        (tr["px1"], ix - a, "#b45309", "出场"),
    ):
        ax[0].axhline(px, color=col, ls="--", lw=0.9)
        ax[0].annotate(
            f"{txt} {px:.2f}",
            xy=(i, px),
            xytext=(i + 1.5, px),
            color=col,
            fontsize=10,
            arrowprops={"arrowstyle": "->", "color": col},
        )
    ax[0].legend(loc="upper right", fontsize=8, ncol=5)
    ax[0].set_title(
        f"{tr['und']} 2 分钟 K（口径 C：含盘前盘后连续、指标不按日重置）  BOLL(20,2) ddof0 · MA5"
        " / MA10",
        fontsize=11,
        loc="left",
    )
    vc = ["#16a34a" if u else "#dc2626" for u in up]
    ax[1].bar(xs, bb["volume"], color=vc, width=0.7)
    ax[1].plot(xs, xx["vavg"], color="#6b7280", lw=1, label="20 日同时段均量")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].set_ylabel("成交量", fontsize=9)
    hc = ["#16a34a" if v >= 0 else "#dc2626" for v in xx["hist"]]
    ax[2].bar(xs, xx["hist"], color=hc, width=0.7)
    ax[2].plot(xs, xx["dif"], color="#2563eb", lw=1, label="DIF")
    ax[2].plot(xs, xx["dea"], color="#f97316", lw=1, label="DEA")
    ax[2].axhline(0, color="#9ca3af", lw=0.6)
    ax[2].legend(loc="upper left", fontsize=8)
    ax[2].set_ylabel("MACD(6,13,6)\n标准 EMA", fontsize=9)
    ax[3].plot(xs, xx["k"], color="#2563eb", lw=1, label="K")
    ax[3].plot(xs, xx["d"], color="#f97316", lw=1, label="D")
    ax[3].plot(xs, xx["j"], color="#a855f7", lw=1, label="J")
    ax[3].legend(loc="upper left", fontsize=8)
    ax[3].set_ylabel("KDJ(9,3,3)", fontsize=9)
    ax[4].plot(xs, xx["rsi2"], color="#dc2626", lw=1, label="RSI(2)")
    ax[4].plot(xs, xx["rsi14"], color="#2563eb", lw=1, label="RSI(14)")
    for lv in (30, 70):
        ax[4].axhline(lv, color="#9ca3af", lw=0.6, ls=":")
    ax[4].legend(loc="upper left", fontsize=8)
    ax[4].set_ylabel("RSI（Wilder）", fontsize=9)
    et = lab[a : z + 1].tz_convert(ET)
    step = max(1, n // 12)
    ticks = list(range(0, n, step))
    ax[4].set_xticks(ticks)
    ax[4].set_xticklabels([et[i].strftime("%Y-%m-%d\n%H:%M") for i in ticks], fontsize=8)
    for k in range(4):
        plt.setp(ax[k].get_xticklabels(), visible=False)
    ax[4].set_xlabel("美东时间（K 线标签 = 开盘时刻）", fontsize=9)
    e0 = pd.Timestamp(tr["entry_ns"], tz="UTC").tz_convert(ET)
    e1 = pd.Timestamp(tr["exit_ns"], tz="UTC").tz_convert(ET)
    info = (
        f"{tr['und']}  {e0:%Y-%m-%d}  {'做多' if tr['dir'] == 1 else '做空'}\n"
        f"入场 {e0:%H:%M:%S}  @ {tr['px0']:.2f}\n"
        f"出场 {e1:%H:%M:%S}  @ {tr['px1']:.2f}\n"
        f"正股涨跌（交易方向）{tr['ret_pct']:+.3f}%\n"
        f"持仓 {(tr['exit_ns'] - tr['entry_ns']) / 60e9:.1f} 分钟  出场原因：{REASON.get(tr['reason'], tr['reason'])}"
        "\n"
        f"入场时 owner_rule_2 条件：\n"
        f"  近 10 分钟方向化涨跌 {tr['c_ret600_pct']:+.3f}%（≥ {RET600_BP / 100:.3f}%）\n"
        f"  1 分钟 MACD 柱 {tr['c_macd_hist']:+.4g}（方向侧 {tr['c_macd_side']:+.0f}，≥ 0）\n"
        f"  期权中间价 60 秒变化 {tr['c_opt_chg_pct']:+.2f}%（≥ {OPT_CHG:.2f}%）"
    )
    ax[0].text(
        0.005,
        0.98,
        info,
        transform=ax[0].transAxes,
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round", "fc": "white", "ec": "#9ca3af", "alpha": 0.92},
    )
    fig.savefig(path, dpi=100, bbox_inches=None)
    plt.close(fig)


# ---------------- 汇总 ----------------


def index_html(sel: pd.DataFrame) -> str:
    def card(r):
        e0 = pd.Timestamp(r.entry_ns, tz="UTC").tz_convert(ET)
        cls = "win" if r.ret_pct > 0 else "loss"
        return (
            f'<a class="card {cls}" href="{r.file}" target="_blank"><img src="{r.file}" alt="{r.file}">'
            f"<div>{e0:%Y-%m-%d %H:%M} {'多' if r.dir == 1 else '空'} {r.ret_pct:+.3f}%</div></a>"
        )

    cols = []
    for und in ("MU", "QQQ"):
        s = sel[sel["und"] == und]
        parts = [f"<section><h2>{und}</h2>"]
        for g in ("盈利", "亏损"):
            gg = s[s["group"] == g]
            parts.append(f"<h3>{g}（{len(gg)} 笔）</h3><div class='grid'>")
            parts += [card(r) for r in gg.itertuples()]
            parts.append("</div>")
        parts.append("</section>")
        cols.append("".join(parts))
    rows = "".join(
        f"<tr><td>{r.und}</td><td>{r.group}</td><td>{pd.Timestamp(r.entry_ns, tz='UTC').tz_convert(ET):%Y-%m-%d %H:%M:%S}"
        "</td>"
        f"<td>{'做多' if r.dir == 1 else '做空'}</td><td>{r.px0:.2f}</td><td>{r.px1:.2f}</td>"
        f"<td>{r.ret_pct:+.3f}%</td><td>{(r.exit_ns - r.entry_ns) / 60e9:.1f}</td>"
        f"<td>{REASON.get(r.reason, r.reason)}</td><td><a href='{r.file}'>图</a></td></tr>"
        for r in sel.itertuples()
    )
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>owner_rule_2 案例</title>
<style>
:root {{ --bg:#fff; --fg:#111827; --mut:#6b7280; --win:#16a34a; --loss:#dc2626; --bd:#e5e7eb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#111827; --fg:#f9fafb; --mut:#9ca3af; --bd:#374151; }} }}
body {{ background:var(--bg); color:var(--fg); font-family:-apple-system,"PingFang SC",sans-serif; margin:16px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px; }}
.card {{ display:block; border:1px solid var(--bd); border-radius:6px; padding:6px; color:var(--fg); text-decoration:none; }}
.card img {{ width:100%; display:block; }}
.card.win div {{ color:var(--win); }} .card.loss div {{ color:var(--loss); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:24px; }}
td,th {{ border:1px solid var(--bd); padding:4px 6px; text-align:right; }}
p.note {{ color:var(--mut); font-size:13px; }}
</style></head><body>
<h1>owner_rule_2 交易案例（正股视角）</h1>
<p class="note">最近 N 笔，按时间倒序，不挑选。规则与数据见 CASES_README.md。盈亏 = 正股从入场到出场朝交易方向的涨跌（%）。</p>
<div class="cols">{"".join(cols)}</div>
<table><tr><th>标的</th><th>组</th><th>入场（美东）</th><th>方向</th><th>入场价</th><th>出场价</th><th>正股涨跌</th><th>持仓（分钟）</th><th>出场原因</th><th>图</th></tr>{rows}</table>
</body></html>
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    OUT.mkdir(parents=True, exist_ok=True)
    notes = []
    mu = pick(mu_trades(root))
    lo = QQQ_RANGE[0]
    qq = qqq_trades(root, lo, QQQ_RANGE[1])
    while (
        qq.empty or (qq["ret_pct"] > 0).sum() < N_EACH or (qq["ret_pct"] < 0).sum() < N_EACH
    ) and lo > dt.date(2025, 1, 2):
        nlo = (pd.Timestamp(lo) - pd.DateOffset(months=1)).date()
        more = qqq_trades(root, nlo, lo - dt.timedelta(days=1))
        qq = pd.concat([qq, more], ignore_index=True)
        notes.append(f"QQQ 不足 5 / 5，向前扩展到 {nlo}（跳过 2026-06～08）")
        lo = nlo
    qqp = pick(qq)
    sel = pd.concat([mu, qqp], ignore_index=True)
    files = []
    for und, g in sel.groupby("und"):
        b, x = bars2(root, und)
        for r in g.to_dict("records"):
            e0 = pd.Timestamp(r["entry_ns"], tz="UTC").tz_convert(ET)
            fn = f"{und}_{e0:%Y%m%d_%H%M%S}_{'long' if r['dir'] == 1 else 'short'}_{'win' if r['ret_pct'] > 0 else 'loss'}.png"
            chart(r, b, x, OUT / fn)
            files.append((r["entry_ns"], und, fn))
    fm = {(u, t): f for t, u, f in files}
    sel["file"] = [fm[(u, t)] for u, t in zip(sel["und"], sel["entry_ns"], strict=True)]
    sel.to_csv(OUT / "cases.csv", index=False)
    (OUT / "index.html").write_text(index_html(sel), encoding="utf-8")
    counts = sel.groupby(["und", "group"]).size().to_dict()
    log.info("案例：%s；%s", counts, notes)
    (OUT / "notes.txt").write_text("\n".join(notes) or "无", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
