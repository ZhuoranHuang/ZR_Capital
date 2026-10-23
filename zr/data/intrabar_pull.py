"""MU 秒级正股价（看跌看涨平价反推）的逐日构建，供盘中回测引擎（zr/backtest/intrabar.py）使用。

每个交易日：合约对 = 当日最近到期日（含 0DTE）的看涨 + 看跌、同一行权价；行权价随现价换档——
按长桥 1 分钟 K 的价格逐分钟选最接近的行权价，只有价格离当前行权价超过 0.75 个行权价间距
（相邻行权价之差，来自当日期权链）才换档。每段向 ThetaData 拉两条腿的逐笔 NBBO
（段起点前 10 秒起），在两条腿合并的报价时间轴上反推 S = C_mid − P_mid + K·e^(−rT)
（r = 0.04，到期 16:00 ET，不计股息与提前行权；任一条腿超过 10 秒未更新则不出值）。
落盘：$ZR_DATA_ROOT/intrabar/underlying=MU/date=YYYY-MM-DD/S.parquet
（ts、S、strike、expiry、c_mid、p_mid、c_spread、p_spread），每日质检写 qa.csv：RTH 秒级覆盖率
（该秒有 10 秒内更新过的反推价的比例）、与长桥 1 分钟收盘的误差（bp）。
已有 S.parquet 的日子跳过（断点续传）。按日期倒序。
运行：uv run python -m zr.data.intrabar_pull --ranges 2026-09-01:2026-10-21,2025-01-01:2026-05-31
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from zr.config import get_settings
from zr.data.calendar import session_close, trading_days
from zr.data.thetadata import ET, ThetaClient
from zr.features.bars import load_bars
from zr.research import v4_decisive as v4d

log = logging.getLogger("zr.data.intrabar_pull")
R = 0.04
NS = 1_000_000_000
MAX_AGE = 10 * NS
HYST = 0.75


def base_dir(root: Path, und: str = "MU") -> Path:
    return root / "intrabar" / f"underlying={und}"


def segments(
    minute_px: pd.Series, strikes: np.ndarray
) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """逐分钟选行权价（带滞后），返回 [(段起, 段止, 行权价)]（UTC）。"""
    out, cur, start = [], None, None
    for ts, px in minute_px.items():
        j = int(np.argmin(np.abs(strikes - px)))
        near = float(strikes[j])
        if cur is None:
            cur, start = near, ts
            continue
        k = int(np.searchsorted(strikes, cur))
        lo = strikes[max(k - 1, 0)]
        hi = strikes[min(k + 1, len(strikes) - 1)]
        gap = max(min(cur - lo if cur > lo else np.inf, hi - cur if hi > cur else np.inf), 0.01)
        if abs(px - cur) > HYST * gap and near != cur:
            out.append((start, ts, cur))
            cur, start = near, ts
    if cur is not None:
        out.append((start, minute_px.index[-1] + pd.Timedelta(minutes=1), cur))
    return out


def _mid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df[(df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])]
    ts = v4d.ns(pd.to_datetime(df["ts"], utc=True))
    o = np.argsort(ts, kind="stable")
    b, a = df["bid"].to_numpy(float)[o], df["ask"].to_numpy(float)[o]
    return ts[o], (b + a) / 2, a - b


def implied(c: tuple, p: tuple, strike: float, expiry: dt.date) -> pd.DataFrame:
    t = np.unique(np.r_[c[0], p[0]])
    cols = {}
    for name, (ts, mid, spr) in (("c", c), ("p", p)):
        i = np.searchsorted(ts, t, side="right") - 1
        ok = (i >= 0) & ((t - ts[np.clip(i, 0, None)]) <= MAX_AGE)
        cols[f"{name}_mid"] = np.where(ok, mid[np.clip(i, 0, None)], np.nan)
        cols[f"{name}_spread"] = np.where(ok, spr[np.clip(i, 0, None)], np.nan)
    exp_close = pd.Timestamp(f"{expiry} 16:00", tz=ET).value
    tt = np.maximum((exp_close - t) / (365 * 24 * 3600e9), 0.0)
    s = cols["c_mid"] - cols["p_mid"] + strike * np.exp(-R * tt)
    df = pd.DataFrame({"ts": t, "S": s, "strike": strike, "expiry": expiry, **cols})
    return df[np.isfinite(df["S"])]


_EXPS: list[dt.date] = []
_STRIKES: dict[dt.date, np.ndarray] = {}


def expirations(client: ThetaClient) -> list[dt.date]:
    """期权链到期日（ThetaData list 端点，含已过期日期）。"""
    if not _EXPS:
        _EXPS.extend(client.list_expirations("MU"))
    return _EXPS


def strikes_for(client: ThetaClient, exp: dt.date) -> np.ndarray:
    if exp not in _STRIKES:
        rows = client.get_json(
            "/option/list/strikes", symbol="MU", expiration=exp.strftime("%Y%m%d")
        )
        _STRIKES[exp] = np.sort(np.array([float(r["strike"]) for r in rows]))
    return _STRIKES[exp]


def build_day(root: Path, client: ThetaClient, day: dt.date, bars: pd.DataFrame) -> dict | None:
    exps = [e for e in expirations(client) if e >= day]
    if not exps or bars.empty:
        return None
    exp = exps[0]
    strikes = strikes_for(client, exp)
    if not len(strikes):
        return None
    px = bars["close"].shift(1).fillna(bars["open"])  # 每分钟用上一分钟收盘（首分钟用开盘）选档
    segs = segments(px, strikes)

    def pull(seg, right):
        a, b, k = seg
        df = client.history_quote(
            "MU",
            exp,
            day,
            interval="tick",
            strike=f"{k:.3f}",
            right=right,
            start_time=(a - pd.Timedelta(seconds=10)).tz_convert(ET).strftime("%H:%M:%S.000"),
            end_time=b.tz_convert(ET).strftime("%H:%M:%S.000"),
        )
        return _mid(df) if len(df) and {"bid", "ask", "ts"} <= set(df.columns) else None

    jobs = [(sg, r) for sg in segs for r in ("call", "put")]
    with ThreadPoolExecutor(max_workers=client.concurrency) as pool:
        res = list(pool.map(lambda j: pull(*j), jobs))
    parts = []
    for i, sg in enumerate(segs):
        c, p = res[2 * i], res[2 * i + 1]
        if c is None or p is None:
            continue
        d = implied(c, p, sg[2], exp)
        a, b = v4d.ns(pd.DatetimeIndex([sg[0], sg[1]]))
        parts.append(d[(d["ts"] >= a) & (d["ts"] < b)])
    if not parts:
        return {"day": str(day), "segments": len(segs), "rows": 0}
    out = pd.concat(parts, ignore_index=True).sort_values("ts", kind="stable")
    dd = base_dir(root) / f"date={day}"
    dd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dd / "S.parquet", index=False)
    # 质检
    open_ns = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).value
    close_ns = pd.Timestamp(dt.datetime.combine(day, session_close(day)), tz=ET).value
    grid = np.arange(open_ns, close_ns, NS)
    ts = out["ts"].to_numpy()
    i = np.searchsorted(ts, grid, side="right") - 1
    cov = float(((i >= 0) & ((grid - ts[np.clip(i, 0, None)]) <= MAX_AGE)).mean())
    mc = v4d.ns(bars.index) + 60 * NS
    j = np.searchsorted(ts, mc, side="right") - 1
    ok = (j >= 0) & ((mc - ts[np.clip(j, 0, None)]) <= MAX_AGE)
    err = (out["S"].to_numpy()[np.clip(j, 0, None)][ok] / bars["close"].to_numpy()[ok] - 1) * 1e4
    return {
        "day": str(day),
        "expiry": str(exp),
        "segments": len(segs),
        "rows": len(out),
        "coverage": round(cov, 4),
        "n_minutes": int(ok.sum()),
        "err_bp_median": round(float(np.median(err)), 3) if len(err) else math.nan,
        "abs_err_bp_median": round(float(np.median(np.abs(err))), 3) if len(err) else math.nan,
        "abs_err_bp_p90": round(float(np.quantile(np.abs(err), 0.9)), 3) if len(err) else math.nan,
    }


def upsert_qa(root: Path, row: dict) -> None:
    p = base_dir(root) / "qa.csv"
    old = pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame()
    if len(old):
        old = old[old["day"] != row["day"]]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values("day").to_csv(
        p, index=False
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", default="2026-09-01:2026-10-21,2025-01-01:2026-05-31")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    days = []
    for r in args.ranges.split(","):
        a, b = (dt.date.fromisoformat(x) for x in r.split(":"))
        days += trading_days(a, b)
    days = sorted(set(days), reverse=True)
    allbars = load_bars(root, "MU", min(days), max(days))
    bday = np.asarray(allbars.index.tz_convert(ET).date)
    client = ThetaClient(concurrency=args.concurrency)
    done = 0
    for day in days:
        if (base_dir(root) / f"date={day}" / "S.parquet").exists():
            continue
        try:
            row = build_day(root, client, day, allbars[bday == day])
        except Exception as e:  # noqa: BLE001 - 单日失败不中断，记 qa 后继续
            log.warning("%s 失败：%s", day, e)
            row = {"day": str(day), "error": str(e)[:200]}
        if row is not None:
            upsert_qa(root, row)
            done += 1
            log.info("%s：%s（本次 %d 天）", day, row, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
