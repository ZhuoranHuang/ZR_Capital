# ruff: noqa: E501  （中文日志与报告行较长）
"""R-057 Massive 数据层（只在 VPS 上运行；密钥只在 VPS 的 .env，本模块不读出、不打印密钥）。

阶段：
- verify：§1.1 核对——3 个日文件（字段、是否不复权、时间戳时区、是否含盘前盘后、大小与行数），
  退市股抽查（TWTR 2022-10、SIVB 2023-03），与长桥 MU / QQQ / SPY 1 分钟 K 在重叠日期逐根比对。
  结果写 $ZR_DATA_ROOT/sip/massive/verify.json；原始文件核对完即删。
运行（VPS）：cd ~/ZR_Capital && env ZR_DATA_ROOT=/home/ubuntu/zr_data PYTHONPATH=<代码副本> .venv/bin/python -P -m zr.data.sip_massive --phase verify
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as pacsv

from zr.data import massive as mv

log = logging.getLogger("zr.sip_massive")
ET = "America/New_York"
EXTRA = ("MU", "QQQ", "SPY", "SMH", "AMD")
LB_CMP = ("MU", "QQQ", "SPY")


def read_aggs(path: Path) -> pd.DataFrame:
    """读一个 flat file（csv.gz）。ticker 读成字符串，其余按文件推断。"""
    t = pacsv.read_csv(
        path, convert_options=pacsv.ConvertOptions(column_types={"ticker": "string"})
    )
    return t.to_pandas()


def ts_et(ns: pd.Series) -> pd.Series:
    return pd.to_datetime(ns, unit="ns", utc=True).dt.tz_convert(ET)


# ---------------- §1.1 核对 ----------------


def describe_minute(df: pd.DataFrame, path: Path) -> dict:
    t = ts_et(df["window_start"])
    mins = t.dt.hour * 60 + t.dt.minute
    return {
        "file_bytes": path.stat().st_size,
        "rows": len(df),
        "tickers": int(df["ticker"].nunique()),
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "window_start_min_et": str(t.min()),
        "window_start_max_et": str(t.max()),
        "window_start_max_utc": str(pd.to_datetime(df["window_start"].max(), unit="ns", utc=True)),
        "rows_pre_0930": int((mins < 570).sum()),
        "rows_rth": int(((mins >= 570) & (mins < 960)).sum()),
        "rows_post_1600": int((mins >= 960).sum()),
        "all_on_minute_grid": bool((df["window_start"] % 60_000_000_000 == 0).all()),
    }


def ticker_day(df: pd.DataFrame, sym: str) -> dict:
    x = df[df["ticker"] == sym]
    if x.empty:
        return {"present": False}
    t = ts_et(x["window_start"])
    rth = x[(t.dt.hour * 60 + t.dt.minute >= 570) & (t.dt.hour * 60 + t.dt.minute < 960)]
    tr = ts_et(rth["window_start"])
    return {
        "present": True,
        "bars": len(x),
        "rth_bars": len(rth),
        "first_et": str(t.min()),
        "last_et": str(t.max()),
        "rth_open": float(rth.loc[tr.idxmin(), "open"]) if len(rth) else None,
        "rth_close": float(rth.loc[tr.idxmax(), "close"]) if len(rth) else None,
        "rth_volume": int(rth["volume"].sum()),
        "all_volume": int(x["volume"].sum()),
    }


def compare_lb(m: pd.DataFrame, lb: pd.DataFrame) -> dict:
    """Massive 1 分钟 vs 长桥 1 分钟（长桥只取有成交的 K；占位 K 量为 0 不比）。偏移 −1/0/+1 分钟的收盘一致率用来核对时间戳对齐。"""
    m = m.set_index("ts")
    lbv = lb[lb["volume"] > 0].set_index("ts")
    out: dict = {"massive_bars": len(m), "lb_traded_bars": len(lbv)}
    for k in (-1, 0, 1):
        j = m[["close"]].join(
            lbv[["close"]].shift(freq=f"{k}min"), how="inner", lsuffix="_m", rsuffix="_l"
        )
        out[f"close_equal_rate_shift{k:+d}"] = (
            float((np.abs(j["close_m"] - j["close_l"]) < 1e-9).mean()) if len(j) else None
        )
    j = m.join(lbv, how="inner", lsuffix="_m", rsuffix="_l")
    out["joined"] = len(j)
    out["only_massive"] = int(len(m.index.difference(lbv.index)))
    out["only_lb_traded"] = int(len(lbv.index.difference(m.index)))
    out["only_massive_volume_share"] = float(
        m.loc[m.index.difference(lbv.index), "volume"].sum() / max(m["volume"].sum(), 1)
    )
    if not len(j):
        return out
    et = j.index.tz_convert(ET)
    rth = (et.hour * 60 + et.minute >= 570) & (et.hour * 60 + et.minute < 960)
    for seg, mask in (("rth", rth), ("ext", ~rth)):
        s = j[mask]
        if not len(s):
            continue
        r: dict = {"n": len(s)}
        for c in ("open", "high", "low", "close"):
            d = s[f"{c}_m"] - s[f"{c}_l"]
            bp = d / s[f"{c}_l"] * 1e4
            r[c] = {
                "equal_rate": float((d.abs() < 1e-9).mean()),
                "abs_usd_p50": float(d.abs().median()),
                "abs_usd_p95": float(d.abs().quantile(0.95)),
                "abs_usd_max": float(d.abs().max()),
                "abs_bp_p95": float(bp.abs().quantile(0.95)),
                "abs_bp_max": float(bp.abs().max()),
            }
        vr = s["volume_m"] / s["volume_l"].where(s["volume_l"] > 0)
        r["volume"] = {
            "equal_rate": float((s["volume_m"] == s["volume_l"]).mean()),
            "ratio_p05": float(vr.quantile(0.05)),
            "ratio_p50": float(vr.median()),
            "ratio_p95": float(vr.quantile(0.95)),
            "sum_ratio": float(s["volume_m"].sum() / max(s["volume_l"].sum(), 1)),
        }
        out[seg] = r
    return out


def lb_bars(root: Path, sym: str, day: dt.date) -> pd.DataFrame:
    p = root / "bars_1m" / f"underlying={sym}" / f"year={day.year}" / "bars.parquet"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_parquet(p)
    d = d[d["ts"].dt.tz_convert(ET).dt.date == day]
    return d[["ts", "open", "high", "low", "close", "volume"]]


def all_day_keys(ff: mv.FlatFiles, prefix: str) -> list[str]:
    objs, _ = ff.list(prefix + "/")
    return sorted(o["key"] for o in objs if o["key"].endswith(".csv.gz"))


def first_accessible(ff: mv.FlatFiles, keys: list[str]) -> str | None:
    """订阅档位可下载的最早文件（HEAD 二分；假设可下载区间是连续的尾段）。"""
    if not keys or not ff.exists(keys[-1]):
        return None
    lo, hi = 0, len(keys) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ff.exists(keys[mid]):
            hi = mid
        else:
            lo = mid + 1
    return keys[lo]


def verify(root: Path, days3: list[dt.date | None], cmp_days: list[dt.date | None]) -> dict:
    """days3 / cmp_days 里的 None 表示最新可得日。"""
    ff = mv.FlatFiles()
    rest = mv.Rest()
    tmp = root / "sip" / "massive" / "tmp_verify"
    tmp.mkdir(parents=True, exist_ok=True)
    res: dict = {"run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}

    # 可得范围
    _, years = ff.list(mv.MINUTE_PREFIX + "/", "/")
    res["minute_years"] = years
    first_year = sorted(years)[0]
    _, months = ff.list(first_year, "/")
    objs, _ = ff.list(sorted(months)[0])
    res["minute_first_files"] = sorted(o["key"] for o in objs)[:3]
    last_year = sorted(years)[-1]
    _, months = ff.list(last_year, "/")
    objs, _ = ff.list(sorted(months)[-1])
    res["minute_last_files"] = sorted(o["key"] for o in objs)[-3:]
    keys = all_day_keys(ff, mv.MINUTE_PREFIX)
    res["minute_files_listed"] = len(keys)
    res["minute_first_accessible"] = first_accessible(ff, keys)
    _, dyears = ff.list(mv.DAY_PREFIX + "/", "/")
    res["day_years"] = dyears

    def fetch(day: dt.date, prefix: str) -> Path:
        """下载失败抛 MassiveError（调用方记录）。"""
        p = tmp / f"{prefix.split('/')[-1]}_{day}.csv.gz"
        t0 = time.monotonic()
        n = ff.download(mv.day_key(day, prefix), p)
        res.setdefault("downloads", []).append(
            {"key": mv.day_key(day, prefix), "bytes": n, "seconds": round(time.monotonic() - t0, 2)}
        )
        return p

    latest = dt.date.fromisoformat(res["minute_last_files"][-1].split("/")[-1][:10])
    res["latest_day"] = str(latest)
    days3 = [latest if d is None else d for d in days3]
    cmp_days = [latest if d is None else d for d in cmp_days]

    # 3 个日文件
    res["files"] = {}
    for day in days3:
        try:
            pm, pd_ = fetch(day, mv.MINUTE_PREFIX), fetch(day, mv.DAY_PREFIX)
        except mv.MassiveError as e:
            res["files"][str(day)] = {"error": str(e)}
            continue
        t0 = time.monotonic()
        m = read_aggs(pm)
        read_s = time.monotonic() - t0
        dd = read_aggs(pd_)
        r = {"minute": describe_minute(m, pm), "minute_read_seconds": round(read_s, 2)}
        r["day"] = {
            "file_bytes": pd_.stat().st_size,
            "rows": len(dd),
            "columns": {c: str(dd[c].dtype) for c in dd.columns},
            "window_start_et": sorted({str(x) for x in ts_et(dd["window_start"]).head(3)}),
        }
        r["tickers"] = {}
        for sym in ("AAPL", "NVDA", "TSLA", "AMZN", "GOOGL", "SPY", "MU"):
            tdm = ticker_day(m, sym)
            drow = dd[dd["ticker"] == sym]
            if len(drow):
                tdm["day_file"] = {
                    c: (float(drow.iloc[0][c]) if c != "ticker" else sym)
                    for c in ("open", "high", "low", "close", "volume")
                }
            r["tickers"][sym] = tdm
        res["files"][str(day)] = r
        del m, dd
        pm.unlink()
        pd_.unlink()

    # 退市股抽查
    res["delisted"] = {}
    for sym, day in (("TWTR", dt.date(2022, 10, 20)), ("SIVB", dt.date(2023, 3, 8))):
        pm, pd_ = fetch(day, mv.MINUTE_PREFIX), fetch(day, mv.DAY_PREFIX)
        m, dd = read_aggs(pm), read_aggs(pd_)
        r = {"minute": ticker_day(m, sym), "in_day_file": bool((dd["ticker"] == sym).any())}
        r["ref_inactive"] = [
            {
                k: x.get(k)
                for k in (
                    "ticker",
                    "name",
                    "type",
                    "primary_exchange",
                    "active",
                    "delisted_utc",
                    "composite_figi",
                    "share_class_figi",
                    "cik",
                )
            }
            for x in rest.tickers(ticker=sym, active="false")
        ]
        # 时点查询：date 参数是否返回当日在市（含后来退市）的普通股
        try:
            pit = rest.tickers(ticker=sym, date=str(day))
            r["ref_point_in_time"] = [
                {
                    k: x.get(k)
                    for k in ("ticker", "type", "primary_exchange", "active", "delisted_utc")
                }
                for x in pit
            ]
        except mv.MassiveError as e:
            r["ref_point_in_time_error"] = str(e)
        res["delisted"][sym] = r
        del m, dd
        pm.unlink()
        pd_.unlink()

    # reference 规模与时点快照
    ref: dict = {}
    for ex in ("XNYS", "XNAS"):
        t0 = time.monotonic()
        a = rest.tickers(type="CS", exchange=ex, active="true")
        b = rest.tickers(type="CS", exchange=ex, active="false")
        ref[ex] = {"active": len(a), "inactive": len(b), "seconds": round(time.monotonic() - t0, 1)}
        ref[ex]["inactive_delisted_min"] = (
            min((x.get("delisted_utc") or "9") for x in b) if b else None
        )
        ref[ex]["inactive_with_delisted_utc"] = sum(1 for x in b if x.get("delisted_utc"))
        for pdate in ("2017-06-15", "2022-10-20"):
            t0 = time.monotonic()
            try:
                pit = rest.tickers(type="CS", exchange=ex, date=pdate)
            except mv.MassiveError as e:
                ref[ex][f"pit_{pdate}_error"] = str(e)
                continue
            ref[ex][f"pit_{pdate}"] = len(pit)
            ref[ex][f"pit_{pdate}_inactive_now"] = sum(1 for x in pit if not x.get("active"))
            ref[ex][f"pit_{pdate}_seconds"] = round(time.monotonic() - t0, 1)
    ref["rest_calls"] = rest.calls
    res["reference"] = ref
    sp = rest.splits(ticker="NVDA")
    res["splits_nvda"] = [
        {k: x.get(k) for k in ("ticker", "execution_date", "split_from", "split_to")} for x in sp
    ]

    # 与长桥逐根比对
    res["lb_compare"] = {}
    for day in cmp_days:
        pm = fetch(day, mv.MINUTE_PREFIX)
        m = read_aggs(pm)
        m = m[m["ticker"].isin(LB_CMP)].copy()
        m["ts"] = pd.to_datetime(m["window_start"], unit="ns", utc=True)
        for sym in LB_CMP:
            lb = lb_bars(root, sym, day)
            if lb.empty:
                res["lb_compare"][f"{sym} {day}"] = {"lb_missing": True}
                continue
            res["lb_compare"][f"{sym} {day}"] = compare_lb(
                m[m["ticker"] == sym][["ts", "open", "high", "low", "close", "volume"]], lb
            )
        pm.unlink()
    tmp.rmdir()
    return res


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["verify"], required=True)
    p.add_argument(
        "--days3", nargs="+", default=["2017-06-15", "2021-11-15", "2024-06-07", "latest"]
    )
    p.add_argument(
        "--cmp-days",
        nargs="+",
        default=["2023-11-15", "2024-04-15", "2024-12-16", "2025-04-09", "2025-10-15", "latest"],
    )
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    if a.phase == "verify":

        def parse(xs: list[str]) -> list[dt.date | None]:
            return [None if x == "latest" else dt.date.fromisoformat(x) for x in xs]

        res = verify(root, parse(a.days3), parse(a.cmp_days))
        out = root / "sip" / "massive" / "verify.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
