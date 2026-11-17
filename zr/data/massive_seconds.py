# ruff: noqa: E501  （中文说明较长）
"""Massive 秒级聚合 K（REST /v2/aggs/ticker/<T>/range/1/second/<日>/<日>，不复权）。只在 VPS 上运行；
密钥只从 VPS 的 .env 读，不打印。端点来源：https://massive.com/docs/rest/stocks/aggregates/custom-bars（未验证，首次运行实测）。

- probe：探查当前档位能取多久的历史、含不含盘前盘后，结果写 $ZR_DATA_ROOT/bars_1s_sip/probe.json。
- pull：逐标的逐交易日拉取，落盘 $ZR_DATA_ROOT/bars_1s_sip/underlying=<代码>/date=<日>/bars.parquet
  （ts = UTC 秒 K 开始时刻；只有有成交的秒才有行）。交易日取自 flat files 日聚合文件列表。
  断点续传：文件已存在即跳过（无数据的日子写 0 行文件）；接口返回 401 / 403（订阅失效）立即停止。
  进度写 bars_1s_sip/PROGRESS.md（随同步任务传回 Mac）。
运行（VPS）：cd ~/ZR_Capital && env ZR_DATA_ROOT=/home/ubuntu/zr_data PYTHONPATH=<代码副本> .venv/bin/python -P -m zr.data.massive_seconds --phase probe|pull
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from zr.data import massive as mv

log = logging.getLogger("zr.massive_seconds")
SYMBOLS = ("MU", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "SMH")
ET = "America/New_York"
MIN_FREE_GB = 4.0


def out_root(root: Path) -> Path:
    return Path(root) / "bars_1s_sip"


def fetch_day(rest: mv.Rest, sym: str, day: dt.date) -> pd.DataFrame:
    rows = list(
        rest.paged(
            f"/v2/aggs/ticker/{sym}/range/1/second/{day}/{day}",
            {"adjusted": "false", "sort": "asc", "limit": 50000},
        )
    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "transactions",
            ]
        )
    d = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(d["t"], unit="ms", utc=True),
            "symbol": sym,
            "open": d["o"].astype(float),
            "high": d["h"].astype(float),
            "low": d["l"].astype(float),
            "close": d["c"].astype(float),
            "volume": d["v"].astype(float),
            "vwap": d["vw"].astype(float) if "vw" in d else float("nan"),
            "transactions": d["n"].astype("Int64") if "n" in d else pd.NA,
        }
    )
    return out.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def trading_days(start: str) -> list[dt.date]:
    ff = mv.FlatFiles()
    objs, _ = ff.list(mv.DAY_PREFIX + "/")
    ds = sorted(
        dt.date.fromisoformat(o["key"].split("/")[-1][:10])
        for o in objs
        if o["key"].endswith(".csv.gz")
    )
    return [d for d in ds if str(d) >= start]


def probe(root: Path) -> dict:
    rest = mv.Rest()
    res: dict = {"run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "days": {}}
    days_all = trading_days("2003-01-01")
    latest = days_all[-1]
    cands = [
        dt.date(2010, 1, 4),
        dt.date(2016, 1, 4),
        dt.date(2019, 1, 2),
        dt.date(2021, 11, 12),
        dt.date(2021, 11, 15),
        dt.date(2023, 6, 5),
        latest,
    ]
    for day in cands:
        t0 = time.monotonic()
        try:
            c0 = rest.calls
            d = fetch_day(rest, "SPY", day)
            et = d["ts"].dt.tz_convert(ET) if len(d) else None
            m = (et.dt.hour * 60 + et.dt.minute) if len(d) else None
            res["days"][str(day)] = {
                "rows": len(d),
                "requests": rest.calls - c0,
                "seconds": round(time.monotonic() - t0, 2),
                "first_et": str(et.min()) if len(d) else None,
                "last_et": str(et.max()) if len(d) else None,
                "rows_pre": int((m < 570).sum()) if len(d) else 0,
                "rows_rth": int(((m >= 570) & (m < 960)).sum()) if len(d) else 0,
                "rows_post": int((m >= 960).sum()) if len(d) else 0,
            }
        except mv.MassiveError as e:
            res["days"][str(day)] = {"error": str(e)[:200], "status": e.status}
    o = out_root(root)
    o.mkdir(parents=True, exist_ok=True)
    (o / "probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


class Stop(Exception):
    pass


def pull(root: Path, symbols: tuple[str, ...], start: str, workers: int = 4) -> dict:
    rest = mv.Rest()
    o = out_root(root)
    days = trading_days(start)
    todo = [
        (s, d)
        for s in symbols
        for d in days
        if not (o / f"underlying={s}" / f"date={d}" / "bars.parquet").exists()
    ]
    total, done, rows = len(todo), 0, 0
    lock = threading.Lock()
    stop: dict = {}
    t0 = time.monotonic()

    def work(job):
        nonlocal done, rows
        s, d = job
        if stop:
            return
        if shutil.disk_usage(o).free / 1e9 < MIN_FREE_GB:
            stop["reason"] = "磁盘剩余不足"
            return
        try:
            df = fetch_day(rest, s, d)
        except mv.MassiveError as e:
            if e.status in (401, 403):
                stop["reason"] = f"订阅失效或无权限（HTTP {e.status}）：{s} {d}"
            else:
                log.error("%s %s 失败：%s", s, d, e)
            return
        p = o / f"underlying={s}" / f"date={d}" / "bars.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
        with lock:
            done += 1
            rows += len(df)
            if done % 200 == 0 or done == total:
                el = time.monotonic() - t0
                rem = (total - done) * el / done / 3600
                (o / "PROGRESS.md").write_text(
                    f"# Massive 秒级 K 拉取进度\n\n- 完成 {done:,} / {total:,} 个标的日（最新 {s} {d}），累计 {rows:,} 行\n"
                    f"- 已用 {el / 3600:.2f} 小时，预计剩余 {rem:.2f} 小时\n- 磁盘剩余 {shutil.disk_usage(o).free / 1e9:.1f} GB\n"
                    f"- 更新 {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n",
                    encoding="utf-8",
                )
                log.info("%d/%d %s %s 剩余约 %.2f 小时", done, total, s, d, rem)

    o.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    return {
        "total": total,
        "done": done,
        "rows": rows,
        "stopped": stop.get("reason"),
        "hours": round((time.monotonic() - t0) / 3600, 2),
    }


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["probe", "pull"], required=True)
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    ap.add_argument("--start", default="2021-11-15")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    res = probe(root) if a.phase == "probe" else pull(root, tuple(a.symbols), a.start, a.workers)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
