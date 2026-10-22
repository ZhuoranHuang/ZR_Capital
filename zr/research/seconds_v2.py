"""秒级分析（METHOD_V2 规格 10，R-027；规格 5 的 10 秒 / 30 秒标签）。

数据：录制器逐笔推送（2026-10-21 起）。

- 正股价格：Quote 推送的 last_done，时间 = 本地接收时间（recv_ts_utc_ns）。
- 期权盘口：Depth 推送的一档 bid/ask（长桥 depth 只有一档，`order_num` 恒 0，见 CLAUDE.md）。
- 事件：正股在 N 秒内变动 ≥ x%（N ∈ {5, 10, 30}，x ∈ {0.05%, 0.1%, 0.2%}），同一 (N, x)
  下相邻事件间隔 ≥ N 秒。
- 合约：当日最近到期（≥ 当日）上的最近一档略虚值（call 行权价 > 现价、put < 现价），
  方向与正股变动同向。
- 输出：事件后 10 / 30 / 60 / 120 秒期权 bid 相对事件时刻 bid 的变化、同向比例、可成交性
  （该时刻 bid>0 且 bid 量>0）；以及数据集事件（1 分钟 bar 事件）在 10 / 30 秒的正股方向化收益。
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ET = dt.timezone(dt.timedelta(hours=-4))  # 仅用于显示；计算全部用 UTC ns
WINDOWS_S = (10, 30, 60, 120)
N_S = (5, 10, 30)
X_PCT = (0.0005, 0.001, 0.002)
SYM_RE = re.compile(r"^(?P<und>[A-Z]+)(?P<exp>\d{6})(?P<right>[CP])(?P<strike>\d+)\.US$")


def recorder_days(root: Path) -> list[dt.date]:
    base = root / "recorder"
    return (
        sorted(dt.date.fromisoformat(p.name[5:]) for p in base.glob("date=*"))
        if base.is_dir()
        else []
    )


def load_kind(root: Path, day: dt.date, und: str, kind: str) -> pd.DataFrame:
    files = sorted(
        (root / "recorder" / f"date={day.isoformat()}" / f"underlying={und}").glob(
            f"{kind}_*.parquet"
        )
    )
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values(
        "recv_ts_utc_ns"
    )


def stock_prices(q: pd.DataFrame, und: str) -> pd.Series:
    s = q[q["symbol"] == f"{und}.US"]
    px = [json.loads(p).get("last_done") for p in s["payload"]]
    out = pd.Series(px, index=s["recv_ts_utc_ns"].to_numpy(), dtype=float).dropna()
    return out[~out.index.duplicated(keep="last")]


def option_books(depth: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, list] = {}
    for sym, ts, payload in zip(
        depth["symbol"], depth["recv_ts_utc_ns"], depth["payload"], strict=True
    ):
        if not SYM_RE.match(sym):
            continue
        p = json.loads(payload)
        b, a = (p.get("bids") or [{}])[0], (p.get("asks") or [{}])[0]
        out.setdefault(sym, []).append((ts, b.get("price"), b.get("volume"), a.get("price")))
    return {
        s: pd.DataFrame(v, columns=["ts", "bid", "bid_vol", "ask"]).set_index("ts").sort_index()
        for s, v in out.items()
    }


def asof(series_index: np.ndarray, t: np.ndarray) -> np.ndarray:
    i = np.searchsorted(series_index, t, side="right") - 1
    return i


def detect_moves(px: pd.Series, n_s: int, x: float) -> pd.DataFrame:
    t = px.index.to_numpy()
    v = px.to_numpy()
    j = asof(t, t - n_s * 1_000_000_000)
    ok = j >= 0
    ref = np.where(ok, v[np.maximum(j, 0)], np.nan)
    chg = v / ref - 1
    hit = np.flatnonzero(np.abs(chg) >= x)
    rows, last = [], -np.inf
    for i in hit:
        if t[i] - last < n_s * 1_000_000_000:
            continue
        rows.append({"ts": t[i], "price": v[i], "chg": chg[i], "direction": int(np.sign(chg[i]))})
        last = t[i]
    return pd.DataFrame(rows)


def pick_contract(
    books: dict[str, pd.DataFrame], day: dt.date, spot: float, direction: int
) -> str | None:
    cand = []
    for s in books:
        m = SYM_RE.match(s)
        exp = dt.datetime.strptime(m["exp"], "%y%m%d").date()
        if exp < day:
            continue
        right = 1 if m["right"] == "C" else -1
        k = int(m["strike"]) / 1000
        if right != direction or (direction == 1 and k <= spot) or (direction == -1 and k >= spot):
            continue
        cand.append((exp, abs(k - spot), s))
    return min(cand)[2] if cand else None


def option_response(ev: pd.DataFrame, books: dict[str, pd.DataFrame], day: dt.date) -> pd.DataFrame:
    rows = []
    for e in ev.itertuples(index=False):
        sym = pick_contract(books, day, e.price, e.direction)
        if sym is None:
            continue
        b = books[sym]
        ti = b.index.to_numpy()
        i0 = asof(ti, np.array([e.ts]))[0]
        if i0 < 0 or not np.isfinite(b["bid"].iloc[i0]) or b["bid"].iloc[i0] is None:
            continue
        bid0 = float(b["bid"].iloc[i0])
        if not bid0 > 0:
            continue
        r = {"ts": e.ts, "symbol": sym, "direction": e.direction, "bid0": bid0}
        for w in WINDOWS_S:
            iw = asof(ti, np.array([e.ts + w * 1_000_000_000]))[0]
            bw = b["bid"].iloc[iw]
            vw = b["bid_vol"].iloc[iw]
            r[f"bid_chg_{w}s"] = (
                float(bw) / bid0 - 1 if bw is not None and np.isfinite(bw) else np.nan
            )
            r[f"tradable_{w}s"] = bool(bw is not None and bw > 0 and vw is not None and vw > 0)
        rows.append(r)
    return pd.DataFrame(rows)


def run(root: Path, und: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    days = recorder_days(root)
    summary, detail = [], []
    meta = {"days": [d.isoformat() for d in days]}
    for day in days:
        q = load_kind(root, day, und, "quote")
        d = load_kind(root, day, und, "depth")
        if q.empty or d.empty:
            continue
        px = stock_prices(q, und)
        books = option_books(d)
        meta[f"{day}_stock_pushes"] = len(px)
        meta[f"{day}_option_contracts"] = len(books)
        meta[f"{day}_span_min"] = (
            round((px.index.max() - px.index.min()) / 6e10, 1) if len(px) else 0
        )
        for n_s in N_S:
            for x in X_PCT:
                ev = detect_moves(px, n_s, x)
                resp = option_response(ev, books, day) if len(ev) else pd.DataFrame()
                row = {
                    "day": day.isoformat(),
                    "N_s": n_s,
                    "x_pct": x * 100,
                    "events": len(ev),
                    "with_option": len(resp),
                }
                for w in WINDOWS_S:
                    if len(resp):
                        c = resp[f"bid_chg_{w}s"]
                        row[f"same_dir_{w}s"] = round(float((c > 0).mean()), 3)
                        row[f"mean_bid_chg_{w}s"] = round(float(c.mean()), 4)
                        row[f"tradable_{w}s"] = round(float(resp[f"tradable_{w}s"].mean()), 3)
                summary.append(row)
                if len(resp):
                    detail.append(resp.assign(N_s=n_s, x_pct=x * 100))
    return (
        pd.DataFrame(summary),
        (pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()),
        meta,
    )


def second_labels(root: Path, und: str, events: pd.DataFrame) -> pd.DataFrame:
    """数据集事件（1 分钟 bar）在录制日的 10 / 30 秒正股方向化收益：决策时刻 = bar 收盘 = ts + 60
    s。
    """
    rows = []
    for day in recorder_days(root):
        q = load_kind(root, day, und, "quote")
        if q.empty:
            continue
        px = stock_prices(q, und)
        t = px.index.to_numpy()
        v = px.to_numpy()
        ev = events[pd.to_datetime(events["ts"]).dt.tz_convert("America/New_York").dt.date == day]
        for e in ev.itertuples(index=False):
            t0 = pd.Timestamp(e.ts).value + 60_000_000_000
            i0 = asof(t, np.array([t0]))[0]
            if i0 < 0 or t0 - t[i0] > 5_000_000_000:  # 决策时刻前 5 秒内无报价则跳过
                continue
            r = {"ts": e.ts, "detector": e.detector, "direction": e.direction}
            for w in (10, 30):
                iw = asof(t, np.array([t0 + w * 1_000_000_000]))[0]
                r[f"ret_{w}s"] = (v[iw] / v[i0] - 1) * e.direction
            rows.append(r)
    return pd.DataFrame(rows)
