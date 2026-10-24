"""限价单成交模型校准（feat/fill-model 第 1 节）：负责人订单 × ThetaData 逐笔报价与 OPRA 成交。

样本（按订单，不按成交笔）：D-030 入选日内单腿持仓段涉及的全部订单 + 同期全部 Canceled / Expired
的期权限价单。每单的数据窗口 = [提交（盘前提交则取开盘）− 60 s,
结束（最后成交 / 撤单 / 过期）+ 300 s]，
截到当日 RTH；完全在盘外提交并撤掉的订单不拉数据，单独计数。
运行：uv run python -m zr.research.fill_model pull   （拉取并缓存）
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.calendar import session_close
from zr.data.thetadata import ET, ThetaClient, parse_contract
from zr.journal.analyze_history import load
from zr.journal.intraday_list import build as build_list

log = logging.getLogger("zr.research.fill_model")
REPO = Path(__file__).resolve().parents[2]
PRE = pd.Timedelta(seconds=60)
POST = pd.Timedelta(seconds=300)
LIMIT_TYPES = ("LO", "LIT")


def _utc(s: pd.Series, naive_is_utc: bool) -> pd.Series:
    t = pd.to_datetime(s.astype(str))
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC" if naive_is_utc else ET)
    return t.dt.tz_convert("UTC")


def order_sample(root: Path) -> pd.DataFrame:
    """返回订单表：order_id、symbol、side、type、status、limit、qty、filled_qty、submit、end、
    group（d030 / canceled / expired），以及合约信息。"""
    o, ex, _ = load(root)
    ex = ex.assign(ts=_utc(ex["ts_et"], naive_is_utc=False))
    ep = build_list(root)
    sel = ep[ep["category"] == "入选"]
    ids = set()
    for _, r in sel.iterrows():
        m = (
            (ex["symbol"] == r["symbol"])
            & (ex["ts"] >= r["open_ts"].tz_convert("UTC"))
            & (ex["ts"] <= r["close_ts"].tz_convert("UTC"))
        )
        ids |= set(ex.loc[m, "order_id"])
    o = o.assign(
        submit=_utc(o["submitted_et"], naive_is_utc=False),
        updated=_utc(o["updated_at"], naive_is_utc=True),
    )
    last_fill = ex.groupby("order_id")["ts"].max()
    d030 = o[o["order_id"].isin(ids)].assign(group="d030")
    canc = o[(o["status"] == "Canceled") & o["order_type"].isin(LIMIT_TYPES)].assign(
        group="canceled"
    )
    expd = o[(o["status"] == "Expired") & o["order_type"].isin(LIMIT_TYPES)].assign(group="expired")
    s = pd.concat([d030, canc, expd], ignore_index=True).drop_duplicates("order_id")
    s["end"] = np.where(s["status"] == "Filled", s["order_id"].map(last_fill), s["updated"])
    s["end"] = pd.to_datetime(s["end"], utc=True)
    c = s["symbol"].map(parse_contract)
    s["und"] = [x["und"] for x in c]
    s["expiry"] = [x["expiry"] for x in c]
    s["strike"] = [x["strike"] for x in c]
    s["right"] = [x["right"] for x in c]
    s["day"] = s["submit"].dt.tz_convert(ET).dt.date
    return s.rename(
        columns={
            "price": "limit",
            "quantity": "qty",
            "executed_quantity": "filled_qty",
            "order_type": "type",
        }
    )[
        [
            "order_id",
            "group",
            "symbol",
            "und",
            "expiry",
            "strike",
            "right",
            "side",
            "type",
            "status",
            "limit",
            "qty",
            "filled_qty",
            "submit",
            "end",
            "day",
            "time_in_force",
        ]
    ]


def window(r) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """RTH 内的数据窗口；订单完全在盘外则 None。跨日（GTC）只看提交当日。"""
    d = r["submit"].tz_convert(ET).date()
    op = pd.Timestamp(dt.datetime.combine(d, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    cl = pd.Timestamp(dt.datetime.combine(d, session_close(d)), tz=ET).tz_convert("UTC")
    start = max(r["submit"], op) - PRE
    end = min(r["end"] + POST, cl) if pd.notna(r["end"]) else cl
    if start >= cl or end <= op:
        return None
    return max(start, op - PRE), end


def _path(root: Path, oid, kind: str) -> Path:
    return root / "diag" / "owner_orders" / f"{oid}_{kind}.parquet"


def pull(root: Path, s: pd.DataFrame, concurrency: int = 4) -> dict:
    client = ThetaClient(concurrency=concurrency)
    stats = {"orders": len(s), "off_hours": 0, "pulled": 0, "cached": 0, "failed": 0}
    lock = threading.Lock()
    jobs = []
    for _, r in s.iterrows():
        w = window(r)
        if w is None:
            stats["off_hours"] += 1
            continue
        jobs.append((r, w))

    def bump(k):
        with lock:
            stats[k] += 1

    def one(job):
        r, (a, b) = job
        if (
            _path(root, r["order_id"], "quotes").exists()
            and _path(root, r["order_id"], "trades").exists()
        ):
            bump("cached")
            return
        kw = {
            "start_time": a.tz_convert(ET).strftime("%H:%M:%S.000"),
            "end_time": b.tz_convert(ET).strftime("%H:%M:%S.000"),
        }
        try:
            q = client.history_quote(
                r["und"],
                r["expiry"],
                r["day"],
                interval="tick",
                strike=f"{r['strike']:.3f}",
                right=r["right"],
                **kw,
            )
            t = client.history_trade(
                r["und"], r["expiry"], r["day"], f"{r['strike']:.3f}", r["right"], **kw
            )
        except Exception as e:  # noqa: BLE001 - 单单失败记录后继续
            log.warning("拉取失败 %s %s：%s", r["order_id"], r["symbol"], e)
            bump("failed")
            return
        p = _path(root, r["order_id"], "quotes")
        p.parent.mkdir(parents=True, exist_ok=True)
        qc = [c for c in ("ts", "bid", "ask", "bid_size", "ask_size") if c in q.columns]
        tc = [c for c in ("ts", "price", "size", "exchange", "condition") if c in t.columns]
        (q[qc] if qc else pd.DataFrame(columns=["ts", "bid", "ask"])).to_parquet(p, index=False)
        (t[tc] if tc else pd.DataFrame(columns=["ts", "price", "size"])).to_parquet(
            _path(root, r["order_id"], "trades"), index=False
        )
        bump("pulled")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, jobs))
    return stats


def pull_v4_trades(root: Path, und: str = "MU", concurrency: int = 4) -> dict:
    """为 V4 已缓存的每个 tick 时间窗（diag/v4_ticks）拉同一合约同一时间窗的 OPRA 逐笔成交。"""
    client = ThetaClient(concurrency=concurrency)
    base = root / "diag" / "v4_ticks" / f"underlying={und}"
    out = root / "diag" / "v4_trades" / f"underlying={und}"
    jobs = []
    for f in sorted(base.glob("date=*/*.parquet")):
        d = dt.date.fromisoformat(f.parent.name[5:])
        dst = out / f.parent.name / f.name
        if not dst.exists():
            jobs.append((f, d, dst))
    stats = {"windows": len(jobs), "pulled": 0, "failed": 0}
    lock = threading.Lock()

    def one(job):
        f, d, dst = job
        sym, se = f.stem.rsplit("_", 1)
        a, b = se.split("-")
        c = parse_contract(sym)
        try:
            t = client.history_trade(
                c["und"],
                c["expiry"],
                d,
                f"{c['strike']:.3f}",
                c["right"],
                start_time=f"{a[:2]}:{a[2:4]}:{a[4:6]}.000",
                end_time=f"{b[:2]}:{b[2:4]}:{b[4:6]}.000",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("成交拉取失败 %s：%s", f.name, e)
            with lock:
                stats["failed"] += 1
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        cols = [x for x in ("ts", "price", "size", "exchange", "condition") if x in t.columns]
        (t[cols] if cols else pd.DataFrame(columns=["ts", "price", "size"])).to_parquet(
            dst, index=False
        )
        with lock:
            stats["pulled"] += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, jobs))
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pull", "pull-v4-trades", "calibrate", "calibrate-orders"])
    ap.add_argument("--und", default="MU", help="pull-v4-trades 的标的")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    s = order_sample(root)
    log.info("订单样本：%s", s.groupby(["group", "status"]).size().to_dict())
    if args.cmd == "pull":
        log.info("拉取：%s", pull(root, s))
    elif args.cmd == "pull-v4-trades":
        log.info("V4 成交拉取：%s", pull_v4_trades(root, args.und))
    elif args.cmd == "calibrate":  # 按挂单段的校准与报告（取代原按订单的版本）
        from zr.research import fill_segments

        return fill_segments.main()
    elif args.cmd == "calibrate-orders":
        f = order_features(root, s)
        coef, tab = fit_fill_prob(f)
        store = root / "research" / "fill_model"
        store.mkdir(parents=True, exist_ok=True)
        f.to_parquet(store / "owner_order_features.parquet", index=False)
        head = [
            "# 限价单成交模型校准（负责人订单，feat/fill-model 第 1 节）",
            "",
            f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · 复现 "
            "`uv run python -m zr.research.fill_model pull` → `calibrate`",
            "",
        ]
        out = REPO / "docs" / "research" / "FILL_MODEL_ORDER_LEVEL.md"
        out.write_text("\n".join(head + report_calibration(f, coef, tab)), encoding="utf-8")
        log.info("写入 %s", out)
    return 0


# ---------------- 第 1 节：校准 ----------------

DELTAS_FILL = (10, 30, 60)
DELTAS_CANCEL = (30, 60, 300)
HORIZONS = (2, 5, 10, 30, 60, 300)


def _load_ticks(root: Path, oid) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    qp, tp = _path(root, oid, "quotes"), _path(root, oid, "trades")
    if not qp.exists():
        return None
    q = pd.read_parquet(qp)
    q = q[(q["ask"] > 0) & (q["bid"] > 0) & (q["ask"] >= q["bid"])].copy()
    if q.empty:
        return None
    q["ts"] = pd.to_datetime(q["ts"], utc=True)
    q = q.sort_values("ts", kind="stable").reset_index(drop=True)
    q["ns"] = pd.DatetimeIndex(q["ts"]).as_unit("ns").asi8
    t = pd.read_parquet(tp) if tp.exists() else pd.DataFrame(columns=["ts", "price", "size"])
    if len(t):
        t["ts"] = pd.to_datetime(t["ts"], utc=True)
        t = t.sort_values("ts", kind="stable").reset_index(drop=True)
    return q, t


def _asof(q: pd.DataFrame, ts: pd.Timestamp):
    i = int(np.searchsorted(q["ns"].to_numpy(), ts.value, side="right")) - 1
    return q.iloc[max(i, 0)]


def order_features(root: Path, s: pd.DataFrame) -> pd.DataFrame:
    from zr.execution.ticks import tick_size

    _, ex, _ = load(root)
    ex = ex.assign(ts=_utc(ex["ts_et"], naive_is_utc=False))
    first_fill = ex.groupby("order_id")["ts"].min()
    rows = []
    for _, r in s.iterrows():
        w = window(r)
        base = {
            k: r[k]
            for k in (
                "order_id",
                "group",
                "symbol",
                "und",
                "side",
                "type",
                "status",
                "limit",
                "qty",
                "filled_qty",
            )
        }
        if w is None:
            rows.append({**base, "skip": "盘外"})
            continue
        d = _load_ticks(root, r["order_id"])
        if d is None:
            rows.append({**base, "skip": "无有效报价"})
            continue
        q, t = d
        sgn = 1 if r["side"] == "Buy" else -1
        t0 = max(r["submit"], w[0] + PRE)
        a0 = _asof(q, t0)
        mid0, spr = (a0["bid"] + a0["ask"]) / 2, a0["ask"] - a0["bid"]
        tick = float(tick_size(r["symbol"], mid0))
        lim = float(r["limit"]) if pd.notna(r["limit"]) else np.nan
        prev = _asof(q, t0 - pd.Timedelta(seconds=30))
        out = {
            **base,
            "skip": "",
            "t0": t0,
            "bid0": a0["bid"],
            "ask0": a0["ask"],
            "mid0": mid0,
            "spread": spr,
            "spread_ticks": spr / tick,
            "spread_pct": spr / mid0 * 100,
            "tick": tick,
            "offset_ticks": sgn * (lim - mid0) / tick,
            "offset_spread": sgn * (lim - mid0) / spr if spr > 0 else np.nan,
            "marketable": bool((lim >= a0["ask"]) if sgn == 1 else (lim <= a0["bid"])),
            "mid_dir30_ticks": sgn * (mid0 - (prev["bid"] + prev["ask"]) / 2) / tick,
            "filled": r["filled_qty"] > 0,
            "full": r["filled_qty"] >= r["qty"],
            "live_s": (r["end"] - t0).total_seconds() if pd.notna(r["end"]) else np.nan,
        }
        ff = first_fill.get(r["order_id"])
        if pd.notna(ff) and out["filled"]:
            out["fill_s"] = max((ff - t0).total_seconds(), 0.0)
            af = _asof(q, ff)
            m_f = (af["bid"] + af["ask"]) / 2
            for dlt in DELTAS_FILL:
                aa = _asof(q, ff + pd.Timedelta(seconds=dlt))
                out[f"adv_{dlt}s_ticks"] = sgn * ((aa["bid"] + aa["ask"]) / 2 - m_f) / tick
                out[f"adv_{dlt}s_pct"] = sgn * ((aa["bid"] + aa["ask"]) / 2 / m_f - 1) * 100
        if r["status"] in ("Canceled", "Expired") and pd.notna(r["end"]):
            ac = _asof(q, r["end"])
            m_c = (ac["bid"] + ac["ask"]) / 2
            for dlt in DELTAS_CANCEL:
                aa = _asof(q, r["end"] + pd.Timedelta(seconds=dlt))
                out[f"miss_{dlt}s_ticks"] = sgn * ((aa["bid"] + aa["ask"]) / 2 - m_c) / tick
                out[f"miss_{dlt}s_pct"] = sgn * ((aa["bid"] + aa["ask"]) / 2 / m_c - 1) * 100
        # 触价：卖一 ≤ 挂单价（买）或 OPRA 成交价 ≤ 挂单价（买）首次出现的时刻
        qq = q[q["ts"] >= t0]
        touch_q = qq["ts"][(qq["ask"] <= lim) if sgn == 1 else (qq["bid"] >= lim)]
        tt = t[t["ts"] >= t0] if len(t) else t
        touch_t = (
            tt["ts"][(tt["price"] <= lim) if sgn == 1 else (tt["price"] >= lim)]
            if len(tt)
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        through = (
            tt["ts"][(tt["price"] < lim) if sgn == 1 else (tt["price"] > lim)]
            if len(tt)
            else pd.Series(dtype="datetime64[ns, UTC]")
        )
        out["through_trade_s"] = (through.iloc[0] - t0).total_seconds() if len(through) else np.nan
        out["touch_quote_s"] = (touch_q.iloc[0] - t0).total_seconds() if len(touch_q) else np.nan
        out["touch_trade_s"] = (touch_t.iloc[0] - t0).total_seconds() if len(touch_t) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def fit_fill_prob(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """成交概率：对每个时长 T，样本 = 存活至少 T 秒或在 T 内成交的订单，y = T 内成交；
    特征 = 挂单位置（÷ spread）、log T、最近 30 秒中间价方向（tick，按方向）、spread（tick）。
    样本小，用带 L2 的逻辑回归，系数只作描述。"""
    from sklearn.linear_model import LogisticRegression

    g = f[(f["skip"] == "") & f["type"].isin(LIMIT_TYPES) & f["offset_spread"].notna()].copy()
    rows = []
    for T in HORIZONS:
        filled_in = g["filled"] & (g["fill_s"] <= T)
        alive = g["live_s"].fillna(np.inf) >= T
        m = filled_in | alive
        for _, r in g[m].iterrows():
            rows.append(
                {
                    "T": T,
                    "y": int(filled_in.loc[r.name]),
                    "offset_spread": r["offset_spread"],
                    "logT": np.log(T),
                    "mid_dir30_ticks": r["mid_dir30_ticks"],
                    "spread_ticks": r["spread_ticks"],
                    "marketable": r["marketable"],
                }
            )
    d = pd.DataFrame(rows)
    X = d[["offset_spread", "logT", "mid_dir30_ticks", "spread_ticks"]].to_numpy(float)
    mdl = LogisticRegression(C=1.0, max_iter=1000).fit(X, d["y"])
    coef = pd.DataFrame(
        {
            "特征": [
                "截距",
                "挂单位置 ÷ spread",
                "log 等待秒数",
                "最近 30 秒中间价方向（tick）",
                "spread（tick）",
            ],
            "系数": np.r_[mdl.intercept_, mdl.coef_[0]].round(3),
        }
    )
    d["p_hat"] = mdl.predict_proba(X)[:, 1]
    d["pos_bucket"] = pd.cut(
        d["offset_spread"],
        [-np.inf, -0.25, 0.0, 0.25, 0.5, np.inf],
        labels=[
            "≤ −0.25（靠近己方报价）",
            "(−0.25, 0]",
            "(0, 0.25]",
            "(0.25, 0.5]（靠近对手价）",
            "> 0.5（越过对手价）",
        ],
    )
    tab = (
        d.groupby(["T", "pos_bucket"], observed=True)
        .agg(订单数=("y", "size"), 实际成交率=("y", "mean"), 模型成交率=("p_hat", "mean"))
        .round(3)
    )
    return coef, tab.reset_index()


def _q(x: pd.Series) -> str:
    x = x.dropna()
    return (
        f"{x.median():.2f}（P25 {x.quantile(0.25):.2f} / P75 {x.quantile(0.75):.2f}，n={len(x)}）"
        if len(x)
        else "—"
    )


def report_calibration(f: pd.DataFrame, coef: pd.DataFrame, tab: pd.DataFrame) -> list[str]:
    from zr.ops.markdown import df_to_md

    g = f[f["skip"] == ""].copy()
    g["filled"] = g["filled"].astype(bool)
    lim = g[g["type"].isin(LIMIT_TYPES)]
    fl, uf = lim[lim["filled"]], lim[~lim["filled"]]
    before = uf["touch_trade_s"] <= uf["live_s"]
    thr_before = uf["through_trade_s"] <= uf["live_s"]
    by_pos = lim.assign(
        pos=pd.cut(lim["offset_spread"], [-np.inf, -1, -0.5, -0.25, 0, 0.25, 0.5, np.inf])
    ).groupby("pos", observed=True)
    pos_tab = pd.DataFrame(
        {
            "订单数": by_pos.size(),
            "成交率": by_pos["filled"].mean().round(3),
            "成交耗时中位（秒）": by_pos["fill_s"].median().round(1),
        }
    ).reset_index()
    adv = pd.DataFrame(
        [
            {
                "成交后": f"+{d} 秒",
                "中间价变化中位（tick，按方向）": round(float(fl[f"adv_{d}s_ticks"].median()), 2),
                "中位（% 中间价）": round(float(fl[f"adv_{d}s_pct"].median()), 2),
                "不利（< 0）占比": round(float((fl[f"adv_{d}s_ticks"] < 0).mean()), 2),
            }
            for d in DELTAS_FILL
        ]
    )
    canc = g[g["status"].isin(["Canceled", "Expired"])]
    miss = pd.DataFrame(
        [
            {
                "撤单后": f"+{d} 秒",
                "中间价变化中位（tick，按方向）": round(
                    float(canc[f"miss_{d}s_ticks"].median()), 2
                ),
                "中位（% 中间价）": round(float(canc[f"miss_{d}s_pct"].median()), 2),
                "错过有利行情（> 0）占比": round(float((canc[f"miss_{d}s_ticks"] > 0).mean()), 2),
            }
            for d in DELTAS_CANCEL
        ]
    )
    ctx = (
        lim.groupby("group")
        .agg(
            订单数=("order_id", "size"),
            成交率=("filled", "mean"),
            spread_tick中位=("spread_ticks", "median"),
            spread_pct中位=("spread_pct", "median"),
            挂单距中间价tick中位=("offset_ticks", "median"),
            挂单位置除spread中位=("offset_spread", "median"),
            提交即可成交占比=("marketable", "mean"),
            存活秒中位=("live_s", "median"),
        )
        .round(3)
        .reset_index()
    )
    return [
        "## 1. 样本",
        "",
        df_to_md(g.groupby(["group", "type", "side"]).size().rename("订单数").reset_index()),
        "",
        f"另有 {int((f['skip'] == '盘外').sum())} 单完全在盘外提交并撤掉（期权不交易），"
        f"{int((f['skip'] == '无有效报价').sum())} 单窗口内无有效报价，未纳入。"
        "负责人 92 笔撤单中 88 笔是限价单（其余为市价 / 条件单）。分析单位是订单。",
        "",
        "## 2. 提交时的盘口与挂单位置（限价单）",
        "",
        "挂单位置 = (挂单价 − 中间价) × 方向 ÷ sprea"
        "d：0 = 挂在中间价，+0.5 = 挂在对手价（买单挂卖一），"
        "−0.5 = 挂在己方最优价（买单挂买一），< −0.5 = 挂在己方最优价之外。",
        "",
        df_to_md(ctx),
        "",
        "## 3. 成交与挂单位置",
        "",
        df_to_md(pos_tab),
        "",
        f"- 成交耗时（已成交的限价单）：{_q(fl['fill_s'])} 秒；部分成交 "
        f"{int((lim['filled'] & ~lim['full'].astype(bool)).sum())} 单。",
        f"- 撤单 / 过期前存活时长：{_q(uf['live_s'])} 秒。",
        "",
        "## 4. 「触价即成交」规则的检验",
        "",
        f"- 已成交 {len(fl)} 单：成交时刻之前，盘口卖一（买单）已 ≤ 挂单价的 "
        f"{int((fl['touch_quote_s'] <= fl['fill_s'] + 1).sum())} 单；有 OPRA 成交价 ≤ 挂单价的 "
        f"{int((fl['touch_trade_s'] <= fl['fill_s'] + 1).sum())}"
        f" 单。**只看盘口会漏掉一半以上的成交**"
        "——负责人的成交大多是别人的市价单打到挂单价，盘口并没有越过挂单价。",
        f"- 未成交 {len(uf)} 单：撤单前已有 OPRA 成交价 ≤ 挂单价（触价）的 {int(before.sum())} 单，"
        f"其中成交价严格越过挂单价的 {int(thr_before.sum())} 单。",
        f"- 合起来：撤单前被成交价触到的限价单里，实际成交 {len(fl)}"
        f" / {len(fl) + int(before.sum())}"
        f"（{len(fl) / max(len(fl) + int(before.sum()), 1):.0%}"
        f"）。只「触到」但没「越过」时可能因排队"
        "没轮到；模拟时把「越过」记为确定成交、「触到」记为按此比例成交。",
        "",
        "## 5. 成交后的逆向选择（按方向：> 0 = 成交后价格朝有利方向走）",
        "",
        df_to_md(adv),
        "",
        "## 6. 撤单后的错过机会（按方向：> 0 = 撤掉之后价格朝原本想要的方向走）",
        "",
        df_to_md(miss),
        "",
        "## 7. 成交概率拟合（逻辑回归，带 L2；样本小，系数只作描述）",
        "",
        "样本：每个等待时长 T ∈ {2, 5, 10, 30, 60, 300} 秒，取在 T 内成交或存活至少 T 秒的限价单，"
        "y = T 内是否成交。",
        "",
        df_to_md(coef),
        "",
        "拟合与实际（按挂单位置分组）：",
        "",
        df_to_md(tab),
        "",
        "## 边界",
        "",
        "- 负责人 2026-06-09～2026-08-19 的订单（D"
        "-030 入选段 + 同期撤单 / 过期限价单），以 MU 为主；"
        "单一交易者、单一券商（长桥）的执行，不代表一般限价单成交率。",
        "- 订单时间：成交用对账单成交时刻（秒级），撤单 / 过期用订单最后更新时刻；tick 为毫秒级，"
        "存在对时误差。",
        "- 「最近 30 秒中间价方向」只用提交前的报价；"
        "成交后 / 撤单后的变化是事后统计，不进模型特征。",
        "",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
