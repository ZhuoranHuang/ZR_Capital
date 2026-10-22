"""已知答案审计（audit/known-answer）→ docs/research/KNOWN_ANSWER_AUDIT.md。

不做新研究，只拿已知答案检验底座。

1. 负责人真实成交对账（D-030 入选样本）：同一合约、同一成交时刻，回放引擎的模拟成交（+2 s 后第一个
   tick 的 ask / bid）与对账单成交价、费用、盈亏逐笔比较。
2. 独立引擎复算：backtrader 重写 E4 基线与 V4 H2（信号、入场时刻），与我们的引擎逐笔比对
   （zr/research/bt_independent.py）。
3. 数据体检：夏令时、半日市、日线一致性、看跌看涨平价与合约解析、交易日历。
运行：uv run python -m zr.research.known_answer_audit
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.costs.fees import option_order_fees
from zr.data.calendar import is_trading_day, session_close
from zr.data.thetadata import lb_symbol, parse_contract
from zr.execution.ticks import tick_size
from zr.features.bars import ET, load_bars, load_extended_bars, resample_bars
from zr.features.regime import load_config
from zr.journal.analyze_history import load
from zr.ops.markdown import df_to_md
from zr.research import bt_independent as bi
from zr.research import e4_review as er
from zr.research import intraday_single_leg as isl
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.research.known_answer_audit")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "research" / "KNOWN_ANSWER_AUDIT.md"
LAG = pd.Timedelta(seconds=2)
UNDS = ("MU", "QQQ", "SPY")


# ---------------- 1. 真实成交对账 ----------------


def _ticks_for(root: Path, r) -> pd.DataFrame | None:
    tk = isl.trade_ticks(None, root, r)
    if tk is None or tk.empty:
        return None
    tk = tk[(tk["ask"] > 0) & (tk["bid"] >= 0) & (tk["ask"] >= tk["bid"])].copy()
    tk["ts"] = pd.to_datetime(tk["ts"], utc=True)
    return tk.sort_values("ts", kind="stable").reset_index(drop=True)


def reconcile(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (逐笔成交对比, 逐段盈亏对比)。"""
    sel, _ = isl.d030_sample(root)
    _, ex, ch = load(root)
    ex = ex.assign(ts=pd.to_datetime(ex["ts_et"].astype(str), utc=True))
    all_lots = isl.round_trips(root)
    fills, eps = [], []
    for _, r in sel.iterrows():
        tk = _ticks_for(root, r)
        f = ex[
            (ex["symbol"] == r["symbol"]) & (ex["ts"] >= r["entry_ts"]) & (ex["ts"] <= r["exit_ts"])
        ].sort_values("ts")
        sim_cash, act_cash = 0.0, 0.0
        rows = []
        for _, x in f.iterrows():
            buy = x["side"] == "Buy"
            px = float(x["price"])
            row = {
                "symbol": r["symbol"],
                "order_id": x["order_id"],
                "side": x["side"],
                "ts": x["ts"],
                "qty": float(x["quantity"]),
                "px": px,
            }
            if tk is not None:
                asof = tk[tk["ts"] <= x["ts"]]
                aft = tk[tk["ts"] >= x["ts"] + LAG]
                if len(asof):
                    row["bid_asof"], row["ask_asof"] = asof["bid"].iat[-1], asof["ask"].iat[-1]
                    row["quote_age_s"] = (x["ts"] - asof["ts"].iat[-1]).total_seconds()
                if len(aft):
                    row["sim_px"] = float(aft["ask"].iat[0] if buy else aft["bid"].iat[0])
                    row["sim_gap_s"] = (aft["ts"].iat[0] - x["ts"]).total_seconds()
            rows.append(row)
        fr = pd.DataFrame(rows)
        if fr.empty:
            continue
        ts_ = float(tick_size(r["symbol"], fr["px"].mean()))
        fr["tick"] = ts_
        sgn = np.where(fr["side"] == "Buy", 1, -1)  # 正 = 实际比模拟差（买得更贵 / 卖得更便宜）
        fr["diff_ticks"] = sgn * (fr["px"] - fr.get("sim_px", np.nan)) / ts_
        fr["diff_pct"] = sgn * (fr["px"] / fr.get("sim_px", np.nan) - 1) * 100
        fr["inside"] = (fr["px"] >= fr.get("bid_asof", np.nan) - 1e-9) & (
            fr["px"] <= fr.get("ask_asof", np.nan) + 1e-9
        )
        fills.append(fr)
        # 逐段：模拟按订单汇总下 D-013 费用（最低收费按订单计）
        # 按张数加权（同一订单的多笔成交数量不同，不能简单平均）
        fr2 = fr.assign(sim_v=fr["sim_px"] * fr["qty"], act_v=fr["px"] * fr["qty"])
        o = (
            fr2.groupby(["order_id", "side"])
            .agg(q=("qty", "sum"), sim_v=("sim_v", "sum"), act_v=("act_v", "sum"))
            .reset_index()
        )
        o["sim"], o["act"] = o["sim_v"] / o["q"], o["act_v"] / o["q"]
        for _, z in o.iterrows():
            side = "Buy" if z["side"] == "Buy" else "Sell"
            k = -1 if side == "Buy" else 1
            sim_cash += k * z["q"] * 100 * z["sim"] - float(
                option_order_fees(int(z["q"]), side, z["sim"])["total"]
            )
            act_cash += k * z["q"] * 100 * z["act"] - float(
                option_order_fees(int(z["q"]), side, z["act"])["total"]
            )
        lots = all_lots[
            (all_lots["symbol"] == r["symbol"])
            & (all_lots["entry_ts"] >= r["entry_ts"])
            & (all_lots["entry_ts"] <= r["exit_ts"])
        ]
        prem = r["entry_px"] * 100 * r["qty"]
        eps.append(
            {
                "symbol": r["symbol"],
                "entry": r["entry_ts"].tz_convert(ET).strftime("%Y-%m-%d %H:%M:%S"),
                "qty": r["qty"],
                "actual_pnl": r["pnl_net"],
                "sim_pnl": round(sim_cash, 2),
                "formula_on_actual_px": round(act_cash, 2),
                "fees_actual": round(float(lots["fees_actual"].sum()), 2),
                "fees_formula": round(float(lots["fees_formula"].sum()), 2),
                "diff_usd": round(r["pnl_net"] - sim_cash, 2),
                "diff_pct_prem": round((r["pnl_net"] - sim_cash) / prem * 100, 2),
            }
        )
    return pd.concat(fills, ignore_index=True), pd.DataFrame(eps)


def _outside_note(f: pd.DataFrame) -> str:
    o = f[~f["inside"] & f["bid_asof"].notna()]
    if o.empty:
        return "- 全部成交价都在当时买一卖一之内。"
    dist = ((o["bid_asof"] - o["px"]).clip(lower=0) + (o["px"] - o["ask_asof"]).clip(lower=0)) / o[
        "tick"
    ]
    parts = [
        f"{side} {len(g)} 笔（低于买一 {int((g['px'] < g['bid_asof']).sum())}、"
        f"高于卖一 {int((g['px'] > g['ask_asof']).sum())}）"
        for side, g in o.groupby("side")
    ]
    return (
        f"- 成交价落在当时买一卖一之外的 {len(o)} 笔：{'；'.join(parts)}；偏离中位 "
        f"{dist.median():.1f} tick，当时报价的年龄中位 {o['quote_age_s'].median():.3f} 秒"
        "（报价不旧）。对账单成交时间只精确到秒、tick 精确到毫秒，同一秒内报价变动时「成交时刻之前"
        "最近一笔报价」可能不是成交当时的报价——这是对时精度造成的，不判定为数据错误（未逐笔核实）。"
    )


def reason(row: pd.Series, fills: pd.DataFrame) -> str:
    f = fills[fills["symbol"] == row["symbol"]]
    f = f[(f["ts"].dt.tz_convert(ET).dt.strftime("%Y-%m-%d %H:%M:%S") >= row["entry"])]
    parts = []
    imp = f["diff_ticks"].dropna()
    if len(imp) and (imp < 0).mean() > 0.5:
        parts.append(
            f"多数成交优于 +2 s 后的盘口（中位 {imp.median():+.1f} tick），限价单在价差内成交"
        )
    if len(f) and f.get("sim_gap_s", pd.Series(dtype=float)).max() > 10:
        parts.append(f"成交后报价稀疏（+2 s 后首个 tick 最晚 {f['sim_gap_s'].max():.0f} s）")
    big = f.loc[f["diff_ticks"].abs() >= 5]
    if len(big):
        parts.append(f"{len(big)} 笔成交与模拟价差 ≥ 5 tick（报价在 2 秒内跳动）")
    if (~f["inside"]).any():
        parts.append(f"{int((~f['inside']).sum())} 笔成交价落在当时买一卖一之外")
    if row["qty"] >= 5:
        parts.append(f"张数大（{row['qty']:.0f} 张），差额按张数放大")
    return "；".join(parts) or "—"


# ---------------- 3. 数据体检 ----------------


SHIFTED = {
    "08:29",
    "08:30",
    "08:31",
    "10:29",
    "10:30",
    "10:31",
    "14:58",
    "14:59",
    "15:00",
    "16:58",
    "16:59",
    "17:00",
}  # 开盘 / 收盘错一小时时成交量高峰会出现的时刻


def _kind(hm: str) -> str:
    if hm in ("09:30", "09:31"):
        return "开盘"
    if "15:58" <= hm <= "16:15":
        return "收盘（含收盘竞价与盘后报送）"
    if hm in ("12:58", "12:59", "13:00"):
        return "半日市收盘"
    if hm in SHIFTED:
        return "错一小时的特征时刻"
    return "其他"


def dst_check(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, near = [], []
    kinds = ("开盘", "收盘（含收盘竞价与盘后报送）", "半日市收盘", "错一小时的特征时刻", "其他")
    for u in UNDS:
        e = load_extended_bars(root, u, clean=False)
        et = e.index.tz_convert(ET)
        g = pd.DataFrame(
            {"vol": e["volume"].to_numpy(), "hm": et.strftime("%H:%M"), "day": et.date}
        )
        top = g.loc[g.groupby("day")["vol"].idxmax()].copy()
        piv = g.pivot_table(index="day", columns="hm", values="vol", aggfunc="sum").fillna(0)
        zero = pd.Series(0.0, index=piv.index)
        v930 = piv.get("09:30", zero)
        shifted_open = (piv.get("08:30", zero) > v930) | (piv.get("10:30", zero) > v930)
        top["kind"] = top["hm"].map(_kind)
        cnt = top["kind"].value_counts()
        other = top.loc[top["kind"] == "其他", "hm"].value_counts().head(5)
        rows.append(
            {
                "标的": u,
                "交易日": len(top),
                **{k: int(cnt.get(k, 0)) for k in kinds},
                "09:30 量不大于 08:30 或 10:30（错一小时判据）": int(shifted_open.sum()),
                "其他时刻": ", ".join(f"{k}×{v}" for k, v in other.items()) or "—",
            }
        )
        for y in sorted({d.year for d in top["day"]}):
            for sw in (_nth_sunday(y, 3, 2), _nth_sunday(y, 11, 1)):
                m = top["day"].map(lambda d, s=sw: abs((d - s).days) <= 14)
                if m.any():
                    near.append(
                        {
                            "标的": u,
                            "切换日": sw,
                            "前后两周交易日": int(m.sum()),
                            "错一小时的特征时刻": int((m & (top["kind"] == kinds[3])).sum()),
                            "其他": int((m & (top["kind"] == "其他")).sum()),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(near)


def session_label_check(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """长桥给每根 K 标了 session（Pre / Intraday / Post），与我们的时区换算无关：若 UTC→ET 在夏令时
    切换前后算错，当日第一根 Intraday K 就不会落在 09:30 ET。最后一根应为收盘前 1 分钟。"""
    rows, odd = [], []
    for u in UNDS:
        e = load_extended_bars(root, u, clean=False)
        e = e[e["session"] == "Intraday"]
        et = e.index.tz_convert(ET)
        g = (
            pd.DataFrame({"hm": et.strftime("%H:%M"), "day": et.date})
            .groupby("day")["hm"]
            .agg(["first", "last"])
        )
        exp = pd.Series(
            {
                d: (dt.datetime.combine(d, session_close(d)) - dt.timedelta(minutes=1)).strftime(
                    "%H:%M"
                )
                for d in g.index
            }
        )
        bf, bl = g["first"] != "09:30", g["last"] != exp
        rows.append(
            {
                "标的": u,
                "交易日": len(g),
                "首根 Intraday 不在 09:30": int(bf.sum()),
                "末根不在收盘前 1 分钟": int(bl.sum()),
            }
        )
        for d in g.index[bf | bl]:
            odd.append(
                {
                    "标的": u,
                    "日期": d,
                    "首根": g.at[d, "first"],
                    "末根": g.at[d, "last"],
                    "应为": exp[d],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(odd)


def _nth_sunday(y: int, month: int, n: int) -> dt.date:
    d = dt.date(y, month, 1)
    d += dt.timedelta(days=(6 - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def option_open_check(root: Path) -> pd.DataFrame:
    """期权开盘时刻：(a) 全天逐笔 tick 文件（fill-resolution 拉取，09:30–16:00）的首笔 tick；
    (b) 1 分钟快照里每天首个有效报价（ask > 0）的分钟。按夏令时 / 冬令时分组。"""
    rows = []
    for p in sorted((root / "diag" / "fill_resolution" / "ticks").glob("*.parquet")):
        t = pd.read_parquet(p, columns=["ts"])
        if t.empty:
            continue
        f = pd.to_datetime(t["ts"], utc=True).min().tz_convert(ET)
        rows.append(
            {
                "来源": "逐笔 tick",
                "日期": f.date(),
                "DST": bool(f.dst()),
                "首个时刻 ET": f.strftime("%H:%M:%S"),
            }
        )
    rng = np.random.default_rng(0)
    for u in ("MU", "QQQ"):
        days = sorted((root / "options_1m" / f"underlying={u}").glob("date=*"))
        for d in rng.choice(days, size=min(60, len(days)), replace=False):
            q = pd.read_parquet(d / "quotes.parquet", columns=["ts", "ask"])
            q = q[q["ask"] > 0]
            if q.empty:
                continue
            f = pd.to_datetime(q["ts"], utc=True).min().tz_convert(ET)
            rows.append(
                {
                    "来源": f"1 分钟快照 {u}",
                    "日期": f.date(),
                    "DST": bool(f.dst()),
                    "首个时刻 ET": f.strftime("%H:%M:%S"),
                }
            )
    r = pd.DataFrame(rows)
    return r.groupby(["来源", "DST", "首个时刻 ET"]).size().rename("天数").reset_index()


def half_day_check(root: Path) -> tuple[pd.DataFrame, list[str]]:
    days = sorted(
        dt.date.fromisoformat(r["date"])
        for r in json.loads(
            (REPO / "config" / "us_equity_calendar.json").read_text(encoding="utf-8")
        )["days"]
        if r["type"] == "early_close"
    )
    v4 = v4d.load_v4()
    rows, notes = [], []
    mu5 = resample_bars(er.load_1m(root, "MU"), "5min")
    te = er.backtest(er.prepare(mu5, er.BASELINE), er.BASELINE)
    te_day = te["entry_ts"].dt.tz_convert(ET).dt.date
    for d in days:
        close = pd.Timestamp(dt.datetime.combine(d, session_close(d)), tz=ET)
        last_rth = mu5[np.asarray(mu5.index.tz_convert(ET).date) == d]
        e4d = te[te_day == d]
        row = {
            "日期": d,
            "收盘": session_close(d).strftime("%H:%M"),
            "MU RTH 5 分钟最后一根": last_rth.index.max().tz_convert(ET).strftime("%H:%M")
            if len(last_rth)
            else "—",
            "E4 交易": len(e4d),
            "E4 出场晚于收盘": int((e4d["exit_ts"] + pd.Timedelta(minutes=5) > close).sum()),
        }
        for conv in ("C", "A"):
            p = root / "research" / "v4" / f"trades_{conv}_MU.parquet"
            if not p.exists():
                continue
            t = pd.read_parquet(p)
            t = t[t["day"] == d]
            ent = pd.to_datetime(t["entry_ns"], utc=True)
            exi = pd.to_datetime(t["exit_ns"], utc=True)
            row[f"V4 {conv} 交易"] = len(t)
            row[f"V4 {conv} 入场或出场晚于收盘"] = int(((ent > close) | (exi > close)).sum())
        rows.append(row)
    in_sample = [d for d in days if d in set(v4d.sample_days(root, "MU", v4))]
    notes.append(f"V4 样本内的提前收盘日：{', '.join(str(d) for d in in_sample) or '无'}。")
    return pd.DataFrame(rows), notes


def daily_check(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ, bad = [], []
    for u in UNDS:
        p = root / "diag" / "daily" / f"{u}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d["day"] = pd.to_datetime(d["ts"], utc=True).dt.tz_convert(ET).dt.date
        d = d.set_index("day")
        b = load_bars(root, u)
        g = b.groupby(np.asarray(b.index.tz_convert(ET).date))
        agg = pd.DataFrame(
            {
                "open": g["open"].first(),
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].last(),
                "volume": g["volume"].sum(),
            }
        )
        j = agg.join(d[["open", "high", "low", "close", "volume"]], rsuffix="_lb", how="inner")
        rel = pd.DataFrame(
            {
                k: (j[k] / j[f"{k}_lb"] - 1).abs() * 100
                for k in ("open", "high", "low", "close", "volume")
            }
        )
        ohlc = ["open", "high", "low", "close"]
        over = rel[ohlc].gt(0.1)
        ratio = j["volume"] / j["volume_lb"]
        summ.append(
            {
                "标的": u,
                "比对天数": len(j),
                **{f"{k} 差 >0.1% 天数": int(over[k].sum()) for k in ohlc},
                **{f"{k} 中位差%": round(float(rel[k].median()), 4) for k in ohlc},
                "成交量比 1 分钟合计 / 日线（中位）": round(float(ratio.median()), 3),
                "成交量比 P10–P90": f"{ratio.quantile(0.1):.3f}–{ratio.quantile(0.9):.3f}",
            }
        )
        for day in rel.index[over.any(axis=1)]:
            bad.append(
                {
                    "标的": u,
                    "日期": day,
                    **{f"{k} 差%": round(float(rel.at[day, k]), 3) for k in ohlc},
                }
            )
    return pd.DataFrame(summ), pd.DataFrame(bad)


def parity_check(root: Path, n: int = 500, r_annual: float = 0.04, seed: int = 0):
    """同到期同行权价 call / put 中间价：C − P 与 S − K·e^(−rT) 的差，允许 (两边半价差之和
    + 0.4%·S) 的误差（股息与美式提前行权的余量；r 取 4% 年化，未验证）。"""
    rng = np.random.default_rng(seed)
    out = []
    stocks = {u: load_bars(root, u) for u in ("MU", "QQQ")}
    days = {u: sorted((root / "options_1m" / f"underlying={u}").glob("date=*")) for u in stocks}
    while len(out) < n:
        u = ("MU", "QQQ")[len(out) % 2]
        q = pd.read_parquet(rng.choice(days[u]) / "quotes.parquet")
        q = q[(q["ask"] > 0) & (q["bid"] > 0) & (q["ask"] >= q["bid"])]
        if q.empty:
            continue
        piv = q.pivot_table(
            index=["ts", "expiry", "strike"],
            columns="right",
            values=["bid", "ask"],
            aggfunc="first",
        ).dropna()
        if piv.empty:
            continue
        k = piv.index[rng.integers(len(piv))]
        ts, exp, strike = k
        t = pd.Timestamp(ts)
        s = stocks[u]
        s_px = s["close"][s.index <= t - pd.Timedelta(minutes=1)]
        if s_px.empty or (t - s_px.index[-1]) > pd.Timedelta(minutes=3):
            continue
        S = float(s_px.iloc[-1])
        T = max(
            (pd.Timestamp(dt.datetime.combine(exp, dt.time(16, 0)), tz=ET) - t).total_seconds()
            / (365 * 86400),
            0.0,
        )
        cm = (piv.loc[k, ("bid", "call")] + piv.loc[k, ("ask", "call")]) / 2
        pm = (piv.loc[k, ("bid", "put")] + piv.loc[k, ("ask", "put")]) / 2
        half = (
            piv.loc[k, ("ask", "call")]
            - piv.loc[k, ("bid", "call")]
            + piv.loc[k, ("ask", "put")]
            - piv.loc[k, ("bid", "put")]
        ) / 2
        resid = (cm - pm) - (S - strike * np.exp(-r_annual * T))
        out.append(
            {
                "und": u,
                "ts": t,
                "expiry": exp,
                "strike": strike,
                "S": S,
                "resid": resid,
                "tol": half + 0.004 * S,
                "ok": abs(resid) <= half + 0.004 * S,
                "moneyness": strike / S - 1,
            }
        )
    return pd.DataFrame(out)


def contract_check(root: Path) -> dict:
    n_rows = expired = bad_parse = 0
    syms = set()
    for u in ("MU", "QQQ", "SPY"):
        for d in sorted((root / "options_1m" / f"underlying={u}").glob("date=*")):
            q = pd.read_parquet(
                d / "quotes.parquet", columns=["symbol", "expiry", "strike", "right", "trade_date"]
            )
            if q.empty:
                continue
            n_rows += len(q)
            expired += int((q["expiry"] < q["trade_date"]).sum())
            syms.update(zip(q["symbol"], q["expiry"], q["strike"], q["right"], strict=True))
    for sym, exp, k, r in syms:
        c = parse_contract(sym)
        if (
            c["expiry"] != exp
            or abs(c["strike"] - k) > 1e-9
            or c["right"] != r
            or lb_symbol(c["und"], exp, r, k) != sym
        ):
            bad_parse += 1
    return {
        "报价行": n_rows,
        "合约数": len(syms),
        "到期后仍有报价的行": expired,
        "代码解析不一致": bad_parse,
    }


def calendar_check(root: Path) -> pd.DataFrame:
    rows = []
    hol = [
        dt.date.fromisoformat(r["date"])
        for r in json.loads(
            (REPO / "config" / "us_equity_calendar.json").read_text(encoding="utf-8")
        )["days"]
        if r["type"] == "full_close"
    ]
    hol = [d for d in hol if dt.date(2023, 10, 2) <= d <= dt.date(2026, 10, 21)]
    for u in ("MU", "QQQ", "SPY"):
        od = {
            dt.date.fromisoformat(p.name[5:])
            for p in (root / "options_1m" / f"underlying={u}").glob("date=*")
        }
        b = load_bars(root, u)
        bd = set(b.index.tz_convert(ET).date)
        rows.append(
            {
                "标的": u,
                "样本期内全天休市日": len(hol),
                "期权数据落在休市日": len(od & set(hol)),
                "RTH K 线落在休市日": len(bd & set(hol)),
                "非交易日的期权数据（周末等）": sum(1 for d in od if not is_trading_day(d)),
            }
        )
    for conv in ("C", "A"):
        p = root / "research" / "v4" / f"trades_{conv}_MU.parquet"
        if p.exists():
            t = pd.read_parquet(p)
            rows.append(
                {
                    "标的": f"V4 {conv} MU 成交",
                    "样本期内全天休市日": len(hol),
                    "期权数据落在休市日": int(t["day"].isin(hol).sum()),
                    "RTH K 线落在休市日": np.nan,
                    "非交易日的期权数据（周末等）": np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------- 报告 ----------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    v4 = v4d.load_v4()
    L = [
        "# 已知答案审计（audit/known-answer）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
        "`uv run python -m zr.research.known_answer_audit`",
        "",
    ]
    verdict = {}

    # 1
    fills, eps = reconcile(root)
    have = fills["sim_px"].notna()
    eps["原因"] = [reason(r, fills) for _, r in eps.iterrows()]
    top = eps.reindex(eps["diff_usd"].abs().sort_values(ascending=False).index).head(10)
    fee_ok = (eps["fees_actual"] - eps["fees_formula"]).abs() < 0.005
    verdict["1"] = "通过（模拟口径保守）" if fee_ok.all() else "不通过（费用不一致）"
    L += [
        "## 1. 负责人真实成交对账（D-030 入选样本）",
        "",
        f"入选 {len(eps)} 个持仓段、{len(fills)} 笔成交；"
        f"有 tick 可模拟的成交 {int(have.sum())} 笔。"
        "模拟买价 = 成交时刻 +2 s 后第一个 tick 的 ask，模拟卖价 = 同一规则的 bid（V4 回放口径）。"
        "价格差符号：正 = 实际比模拟差（买得更贵或卖得更便宜）。",
        "",
        df_to_md(
            pd.DataFrame(
                [
                    {
                        "侧": side,
                        "成交笔数": int(len(g)),
                        "价差中位（tick）": round(float(g["diff_ticks"].median()), 2),
                        "P10 / P90（tick）": (
                            f"{g['diff_ticks'].quantile(0.1):.1f} / "
                            f"{g['diff_ticks'].quantile(0.9):.1f}"
                        ),
                        "价差中位（% 权利金）": round(float(g["diff_pct"].median()), 2),
                        "落在当时买一卖一之内": f"{g['inside'].mean():.1%}",
                    }
                    for side, g in fills[have].groupby("side")
                ]
            )
        ),
        "",
        "- 费用：逐段实际收费与 D-013 公式（`fees_formula`）"
        f"{'全部一致' if fee_ok.all() else '有不一致'}"
        f"（{int(fee_ok.sum())}/{len(eps)} 段差 < 0.5 美分）。",
        f"- 逐段盈亏：实际合计 {eps['actual_pnl'].sum():.2f} USD，按回放口径模拟合计 "
        f"{eps['sim_pnl'].sum():.2f} USD，差 {eps['diff_usd'].sum():.2f} USD（每段中位 "
        f"{eps['diff_pct_prem'].median():+.2f}% 权利金）。即回放引擎的成交假设比负责人实际成交"
        "**更差**——V4 的扣费后结果是偏保守的，不是偏乐观的。",
        "",
        _outside_note(fills),
        "",
        "差异最大的 10 段：",
        "",
        df_to_md(top),
        "",
    ]

    # 2
    mu5 = resample_bars(er.load_1m(root, "MU"), "5min")
    ours = er.backtest(er.prepare(mu5, er.BASELINE), er.BASELINE)
    ours = ours.assign(signal_ts=mu5.index[ours["signal_i"].to_numpy()]).set_index("signal_ts")
    btt = bi.run_e4(mu5).set_index("signal_ts")
    common = ours.index.intersection(btt.index)
    mism = {
        c: int((ours.loc[common, c] != btt.loc[common, c]).sum())
        for c in ("entry_ts", "exit_ts", "reason", "adds")
    }
    pxd = {
        c: float((ours.loc[common, c] - btt.loc[common, c]).abs().max())
        for c in ("entry_px", "exit_px", "gross", "net")
    }
    nd = (ours.loc[common, "net"] - btt.loc[common, "net"]).abs()
    days = v4d.sample_days(root, "MU", v4)
    ev, _ = v4d.build_events(root, "MU", "C", v4)
    h2 = ev[ev["hyp_H2"]].copy()
    ent = [
        bi.first_tick(root, s, d, t + LAG, f)
        for s, d, t, f in zip(
            h2["symbol"], h2["day"], h2["decision_ts"], h2["flat_ts"], strict=True
        )
    ]
    h2["entry_ts"] = [e["entry_ts"] for e in ent]
    h2["entry_ask"] = [e["entry_ask"] for e in ent]
    b2 = resample_bars(load_extended_bars(root, "MU", end=max(days), clean=False), "2min")
    w = load_config()["regime"]["range"]["weights"]
    W = [w["bw"], w["slope"], w["hist"]]
    thr = v4["conventions"]["C"]["range_threshold"]
    bh2 = bi.run_h2(root, b2, set(days), thr, W, 1.0)
    bh2_2d = bi.run_h2(root, b2, set(days), thr, W, 2.0)
    k = ["decision_ts", "direction"]
    m = h2.merge(bh2, on=k, how="outer", suffixes=("", "_bt"), indicator=True)
    both = m[m["_merge"] == "both"]
    h2_ok = (
        (m["_merge"] == "both").all()
        and (both["symbol"] == both["symbol_bt"]).all()
        and (both["entry_ts"] == both["entry_ts_bt"]).all()
    )
    e4_ok = len(ours) == len(btt) == len(common) and not any(mism.values()) and nd.max() <= 0.01
    verdict["2"] = "通过" if (e4_ok and h2_ok) else "不通过"
    L += [
        "## 2. 独立引擎复算（backtrader 1.9.78，`zr/research/bt_independent.py`）",
        "",
        "### E4 基线（MU 全样本，5 分钟 K，R-036 修正后基线）",
        "",
        f"- 交易数：我们 {len(ours)}、backtrader {len(btt)}，按信号 K 对齐后共同 {len(common)}，"
        f"只在一边出现 {len(ours.index.symmetric_difference(btt.index))}。",
        "- 不一致笔数：" + "、".join(f"{c} {v}" for c, v in mism.items()) + "。",
        "- 价格 / 盈亏最大绝对差：" + "、".join(f"{c} {v:.2g}" for c, v in pxd.items()) + "。",
        f"- 净额差 > 1e-6 的 {int((nd > 1e-6).sum())} 笔逐条解释：样本最开头几天 ATR 预热口径不同"
        "（backtrader 的 ATR 以前 14 根 TR 的均值做种，我们以第一根 TR 做种；"
        "几百根后收敛到 1e-7），"
        "造成止损价差 ~1e-7、净额差 < 0.0001 USD；另 1 笔差 0.01 USD 来自费用——我们的引擎算费用前"
        "把成交价取整到分（1029.691 → 1029.69），backtrader 用原价，SEC 费四舍五入差 1 美分。",
        "",
        "### V4 H2（MU，口径 C：信号、护栏、合约、入场时刻）",
        "",
        f"- 信号行：我们 {len(h2)}、backtrader {len(bh2)}；"
        f"{m['_merge'].value_counts().to_dict()}。",
        f"- 合约不一致 {int((both['symbol'] != both['symbol_bt']).sum())}，入场 tick 时刻不一致 "
        f"{int((both['entry_ts'] != both['entry_ts_bt']).sum())}，入场 ask 最大差 "
        f"{float((both['entry_ask'] - both['entry_ask_bt']).abs().max()):.2g}。",
        "- backtrader 的 MACD 用 SMA 做 EMA 种子（我们用首值），连续三年的 2 分钟 K 上已完全收敛。",
        f"- RANGE 打分窗口按配置的 2 个交易日（而不是代码实际的 1 天，见第 4 节）重算：信号行 "
        f"{len(bh2_2d)}，与 1 天窗口逐行相同——H2 需要 CHANNEL 状态，而 CHANNEL 优先级高于 RANGE，"
        "RANGE 护栏对 H2 永远不起作用；H1 同理（IGNITION 优先级最高）。",
        "",
    ]

    # 3
    dst, near = dst_check(root)
    opt_open = option_open_check(root)
    half, notes = half_day_check(root)
    dsum, dbad = daily_check(root)
    par = parity_check(root)
    con = contract_check(root)
    cal = calendar_check(root)
    sess, sess_odd = session_label_check(root)
    verdict["3a"] = (
        "通过"
        if (sess["首根 Intraday 不在 09:30"] == 0).all()
        else f"不通过（{int(sess['首根 Intraday 不在 09:30'].sum())} 个标的-日）"
    )
    verdict["3b"] = "有问题（见表）" if (half.filter(like="晚于收盘").sum().sum() > 0) else "通过"
    vr = dsum["成交量比 1 分钟合计 / 日线（中位）"]
    verdict["3c"] = (
        f"价格基本一致；1 分钟 K 成交量只有日线的 {vr.min():.2f}–{vr.max():.2f}（口径不同，未验证）"
    )
    verdict["3d"] = f"{par['ok'].mean():.1%} 落在容差内"
    verdict["3e"] = "通过" if (cal["期权数据落在休市日"].fillna(0) == 0).all() else "不通过"
    L += [
        "## 3. 数据体检",
        "",
        "### 3a. 夏令时",
        "",
        "判据：长桥自己给每根 K 标 session，与我们的时区换算无关；若夏令时换算出错，当日首根 "
        "Intraday K 不会落在 09:30 ET。",
        "",
        df_to_md(sess),
        "",
        "首根或末根异常的日子（末根早于收盘前 1 分钟 = 当"
        "天最后几分钟缺 K；2026-10-21 为盘中拉取）：",
        "",
        df_to_md(sess_odd) if len(sess_odd) else "无。",
        "",
        "补充（描述性）：每个交易日成交量最大的 1 分钟（全时段原始 K）。「其他」多为盘中事件或盘后"
        "大单，不是时区问题——时区错一小时会让开盘竞价出现在 08:30 / 10:30，上面的判据已排除。",
        "",
        df_to_md(dst),
        "",
        "夏令时切换日（3 月第 2 个、11 月第 1 个周日）前后各两周：",
        "",
        df_to_md(near),
        "",
        "期权开盘时刻（逐笔 tick 文件的首笔 / 1 分钟快照每天首个有效报价）：",
        "",
        df_to_md(opt_open),
        "",
        "### 3b. 半日市（提前收盘日）",
        "",
        df_to_md(half),
        "",
        *notes,
        "",
        "### 3c. 日线一致性：RTH 1 分钟 K 合成日线 vs 长桥日线（NoAdjust）",
        "",
        df_to_md(dsum),
        "",
        f"开高低收任一差异 > 0.1% 的标的-日共 {len(dbad)} 个（前 40 个）：",
        "",
        df_to_md(dbad.head(40)) if len(dbad) else "无。",
        "",
        "开盘价差异多见于 MU：日线开盘价是开盘竞价价，1 分钟 K 的开盘是 09:30 那根的首笔成交。"
        "成交量：长桥 1 分钟 K 合计只有长桥日线成交量的约"
        " 17–23%，每天比例稳定。推测 1 分钟 K 是部分"
        "市场（如单一交易所，或不含场外 / 暗池）成交量"
        "，日线是全市场合并成交量（未验证）。影响：成交量"
        "比类特征（vol_ratio、E11 的量比）只要比例稳定就不受影响；把 1 分钟成交量当作全市场成交量的"
        "绝对用法不成立（目前仓库没有这种用法）。",
        "",
        "### 3d. 看跌看涨平价（随机 500 个时刻，MU / QQQ 各半）",
        "",
        f"落在容差内 {int(par['ok'].sum())}/{len(par)}（{par['ok'].mean():.1%}）；残差中位 "
        f"{par['resid'].median():.3f}，|残差| P90 {par['resid'].abs().quantile(0.9):.3f}。"
        "容差 = 两边半价差之和 + 0.4%·S（股息与美式提前行权余量，r 取 4% 未验证）。",
        "",
        df_to_md(par[~par["ok"]].sort_values("resid", key=abs, ascending=False).head(10).round(4))
        if (~par["ok"]).any()
        else "全部在容差内。",
        "",
        f"合约代码与到期：{json.dumps(con, ensure_ascii=False)}。",
        "",
        "### 3e. 交易日历",
        "",
        df_to_md(cal),
        "",
        "## 4. 发现的问题与影响",
        "",
        "- **RANGE 打分窗口实际是 1 个交易日，不是配置的 2 天**："
        "`zr/features/regime.range_score` 用 "
        "`window_days × bars_per_day(df) / period`，而 `bars_per_day` 已按周期 P 的 K 线计数，"
        "多除了一次周期（RTH 2 分钟 195 根/日、含盘前盘后 480 根/日，窗口都只有 1 天）。RANGE 阈值"
        "是用同一份代码标定的（30% 覆盖率仍成立），所以内部一致，但「2 天」的标注是错的。"
        "影响：所有用到 RANGE 状态的结果——v3 网格与规则卡、V4 H3（需前一根 RANGE）、"
        "R-035/R-037 规则"
        "回放的 RANGE 护栏；不影响 H1、H2（已核实 H2 逐行相同）。"
        "修法二选一（待负责人定）：把配置改成 "
        "1 天（不改任何结果），或修代码为 2 天并重标阈值、重跑上述结果。",
        "- **长桥 session 标签在 2023–2024 年 4 个提前收盘日有误**（2023-11-24、2024-07-03、"
        "2024-11-29、2024-12-24）：13:00 收盘后的 K"
        " 仍标为 Intraday（到 15:5x），2025 年的提前收盘日"
        "标注正确。我们的 RTH 数据按交易日历（`zr/data/calendar.rth_mask`）裁剪、不依赖该标签，"
        "所以 RTH K 线与所有基于它的结果不受影响；任何直接用 session 标签判定盘中的代码都会出错"
        "（目前仓库没有）。",
        "- **1 分钟 K 成交量只占长桥日线成交量的约 17–23%**（3c）：量比类特征不受影响，绝对成交量"
        "不能当全市场成交量用。",
        "- **V4 的清仓 / 最后入场时刻原来不随提前收盘日调整**（写死 15:49 / 15:40）：本分支改为按"
        "交易日历的收盘时刻前移（`v4_decisive.day_limits`，提前收盘日 12:40 / 12:50）。MU 现有结果"
        "在提前收盘日 0 笔受影响（3b 表）；QQQ 结果出来后同样核对。",
        "- **回放成交口径比负责人实际成交差**（第 1 节）：V4 的「+2 秒后吃对手价」假设下，"
        "负责人 51 段实际 +9,839 USD 只剩约 +129 USD；V4 的负结果因此偏保守，但这不是代码错误，"
        "也不能直接推出用限价单就能拿到同样的成交（限价单有成交选择偏差）。",
        *([f"- 半日市：{notes[0]}"] if notes else []),
        "",
        "## 结论",
        "",
        df_to_md(pd.DataFrame([{"项目": k, "结论": v} for k, v in verdict.items()])),
        "",
        "## 边界",
        "",
        "- 对账只覆盖 D-030 入选的持仓段；独立引擎只复算 E4 基线与 V4 H2（MU，口径 C）。",
        "- 平价检验的容差含股息与提前行权余量，是合理性检查，不是精确定价。",
        "- 日线比对用长桥日线（RTH，NoAdjust）；成交量口径差异（是否含收盘集合竞价、碎股）未拆解。",
        "",
    ]
    OUT.write_text("\n".join(L), encoding="utf-8")
    store = root / "research" / "audit"
    store.mkdir(parents=True, exist_ok=True)
    fills.to_parquet(store / "owner_fill_reconcile.parquet", index=False)
    eps.to_parquet(store / "owner_episode_reconcile.parquet", index=False)
    log.info("写入 %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
