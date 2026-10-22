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
    ap.add_argument("cmd", choices=["pull", "pull-v4-trades"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    s = order_sample(root)
    log.info("订单样本：%s", s.groupby(["group", "status"]).size().to_dict())
    if args.cmd == "pull":
        log.info("拉取：%s", pull(root, s))
    elif args.cmd == "pull-v4-trades":
        log.info("V4 成交拉取：%s", pull_v4_trades(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
