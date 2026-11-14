# ruff: noqa: E501  （中文日志与报告行较长）
"""R-057 Massive 逐日处理（只在 VPS 上运行；Massive 密钥只在 VPS 的 .env，本模块不读出、不打印密钥）。

原始文件来自归档目录 $ZR_DATA_ROOT/massive_raw/<S3 键>（zr.data.massive_archive 按日期顺序下载；本模块只读，
文件未到就等待，不自己下载，避免与归档进程写同一文件）。输出 $ZR_DATA_ROOT/sip/massive/：

- pass1（按日期顺序）：
  daily/YYYY/<日>.parquet：当日每个代码一行——日聚合（d_*，开 = 常规时段首笔、官方收盘、全天合并量）、
    常规时段 1 分钟聚合（r_*，09:30 至当日收盘，提前收盘日按交易所日程）、开盘区间（or_*，09:30–09:34 五根 1 分钟 K：
    开 = 首根开盘，收 = 末根收盘，高低取极值，量 = 合计；or_first_min = 首根所在分钟）。
  extra1m/YYYY/<日>.parquet：MU、QQQ、SPY、SMH、AMD 全时段 1 分钟 K（含盘前盘后，§1.4）。
- meta（REST）：ref/cs_tickers.parquet（XNYS、XNAS 普通股，active=true 与 false）；ref/pit/<周首个交易日>.parquet
  （reference tickers 时点查询 date=该日，type=CS）；ref/splits.parquet。
- pass2（按日期顺序，需要 Mac 上 zr.research.sip_select 产出并上传的 derived/pass2_rows.parquet）：
  bars1/YYYY/<日>.parquet：当日前 50 名的常规时段 1 分钟 K（§1.2(c)）；
  cand_out/YYYY/<日>.parquet：基础版候选（合格 ∩ 非十字星）在 (保守, 0.01) 与 (乐观, 0) 两种设定下的成交结果
  （zr.research.sip_orb.sim_core，与 SIP 前 N 同一执行代码）。
断点续传：state/pass1_done.txt、state/pass2_done.txt。进度：PROGRESS.md（随同步任务传回 Mac）。

运行（VPS）：cd ~/ZR_Capital && env ZR_DATA_ROOT=/home/ubuntu/zr_data PYTHONPATH=<代码副本> .venv/bin/python -P -m zr.data.sip_stream --phase pass1|meta|pass2 [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from zr.data import massive as mv
from zr.data.massive_archive import raw_root
from zr.data.sip_massive import read_aggs
from zr.research import sip_orb as so

log = logging.getLogger("zr.sip_stream")
ET = "America/New_York"
EXTRA = ("MU", "QQQ", "SPY", "SMH", "AMD")
OR_START, OR_END = 570, 575  # 09:30 ≤ 分钟 < 09:35
PASS2_CFGS = (("cons", 0.01), ("opt", 0.0))
MIN_FREE_GB = 3.0


def out_root(root: Path) -> Path:
    return root / "sip" / "massive"


# ---------------- 纯函数 ----------------


def et_minutes(window_start_ns: pd.Series) -> np.ndarray:
    t = pd.to_datetime(window_start_ns.to_numpy(), unit="ns", utc=True).tz_convert(ET)
    return np.asarray(t.hour * 60 + t.minute)


def session_close_min(day: dt.date) -> int:
    from zr.data.calendar import session_close

    c = session_close(day)
    return c.hour * 60 + c.minute


def _agg(x: pd.DataFrame, p: str) -> pd.DataFrame:
    g = x.groupby("ticker", sort=False)
    out = g.agg(
        **{
            f"{p}_open": ("open", "first"),
            f"{p}_high": ("high", "max"),
            f"{p}_low": ("low", "min"),
            f"{p}_close": ("close", "last"),
            f"{p}_volume": ("volume", "sum"),
            f"{p}_bars": ("open", "size"),
        }
    )
    if p == "or":
        out["or_first_min"] = g["m"].first()
    return out


def day_tables(minute: pd.DataFrame, close_min: int) -> pd.DataFrame:
    """一个日文件的 1 分钟 K（需列 ticker、window_start、open、high、low、close、volume）→ 每个代码的
    常规时段聚合（r_*）与开盘区间（or_*）。只用当日数据；常规时段 = [09:30, 收盘)。"""
    x = minute.assign(m=et_minutes(minute["window_start"]), volume=minute["volume"].astype(float))
    x = x.sort_values(["ticker", "window_start"], kind="mergesort")
    rth = x[(x["m"] >= OR_START) & (x["m"] < close_min)]
    orb = rth[rth["m"] < OR_END]
    return _agg(rth, "r").join(_agg(orb, "or"), how="left")


def session_of(m: np.ndarray, close_min: int) -> np.ndarray:
    return np.where(m < OR_START, "Pre", np.where(m < close_min, "Intraday", "Post"))


def extra_bars(minute: pd.DataFrame, close_min: int, syms=EXTRA) -> pd.DataFrame:
    x = minute[minute["ticker"].isin(syms)].copy()
    x["ts"] = pd.to_datetime(x["window_start"].to_numpy(), unit="ns", utc=True)
    x["session"] = session_of(et_minutes(x["window_start"]), close_min)
    x["volume"] = x["volume"].astype(float)
    cols = ["ts", "ticker", "open", "high", "low", "close", "volume", "transactions", "session"]
    return (
        x[cols]
        .rename(columns={"ticker": "symbol"})
        .sort_values(["symbol", "ts"])
        .reset_index(drop=True)
    )


def rth_groups(minute: pd.DataFrame, close_min: int, syms: set) -> dict:
    """{代码: (分钟, o, h, l, c, 原始行)}，只含常规时段、按时间升序。"""
    x = minute[minute["ticker"].isin(syms)]
    m = et_minutes(x["window_start"])
    keep = (m >= OR_START) & (m < close_min)
    x = x[keep].assign(m=m[keep]).sort_values(["ticker", "window_start"], kind="mergesort")
    out = {}
    for s, g in x.groupby("ticker", sort=False):
        out[s] = (
            g["m"].to_numpy(),
            g["open"].to_numpy(float),
            g["high"].to_numpy(float),
            g["low"].to_numpy(float),
            g["close"].to_numpy(float),
            g,
        )
    return out


def outcomes(groups: dict, rows: pd.DataFrame, cfgs=PASS2_CFGS) -> pd.DataFrame:
    """rows：symbol / dir / or_high / or_low / atr_prev。对每个设定跑 sim_core（自然方向），未触发的不出行。"""
    out = []
    for r in rows.itertuples(index=False):
        g = groups.get(r.symbol)
        if g is None:
            continue
        m, o, h, lo, c, _ = g
        for algo, slip in cfgs:
            res = so.sim_core(
                m, o, h, lo, c, int(r.dir), r.or_high, r.or_low, r.atr_prev, slip, algo == "cons"
            )
            if res is None:
                continue
            e, _, x, j, k, stopped = res
            out.append(
                (
                    r.symbol,
                    algo,
                    slip,
                    int(r.dir),
                    float(e),
                    float(x),
                    bool(stopped),
                    int(m[j]),
                    int(m[k]),
                )
            )
    return pd.DataFrame(
        out,
        columns=[
            "symbol",
            "algo",
            "slip",
            "dir",
            "entry",
            "exit",
            "stopped",
            "entry_min",
            "exit_min",
        ],
    )


# ---------------- I/O ----------------


def wait_file(p: Path, timeout: float = 900) -> None:
    t0 = time.monotonic()
    while not p.exists():
        if time.monotonic() - t0 > timeout:
            raise TimeoutError(f"等待归档文件超时：{p}")
        time.sleep(5)


def write_atomic(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)


def done_set(p: Path) -> set[str]:
    return set(p.read_text(encoding="utf-8").split()) if p.exists() else set()


def mark_done(p: Path, day: dt.date) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{day}\n")


class Progress:
    def __init__(self, path: Path, phase: str, total: int):
        self.path, self.phase, self.total = path, phase, total
        self.t0 = time.monotonic()
        self.n = 0
        self.secs: list[float] = []

    def step(self, day: dt.date, secs: float) -> None:
        self.n += 1
        self.secs.append(secs)
        if self.n % 10 == 0 or self.n == self.total:
            per = float(np.mean(self.secs[-50:]))
            rem = (self.total - self.n) * per / 3600
            free = shutil.disk_usage(self.path.parent).free / 1e9
            self.path.write_text(
                f"# R-057 Massive 逐日处理进度\n\n- 阶段：{self.phase}\n- 完成：{self.n}/{self.total}（最新 {day}）\n"
                f"- 每日耗时（近 50 日均值）：{per:.1f} 秒；预计剩余 {rem:.2f} 小时\n- 磁盘剩余：{free:.1f} GB\n"
                f"- 更新：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}\n",
                encoding="utf-8",
            )
            log.info(
                "%s %d/%d %s %.1fs/日 剩余约 %.2f 小时",
                self.phase,
                self.n,
                self.total,
                day,
                per,
                rem,
            )


def pass1(root: Path, limit: int | None = None) -> dict:
    o = out_root(root)
    done_p = o / "state" / "pass1_done.txt"
    done = done_set(done_p)
    days = [d for d in plan_days(root) if str(d) not in done]
    if limit:
        days = days[:limit]
    pr = Progress(o / "PROGRESS.md", "pass1", len(days))
    anomalies = []
    for day in days:
        t0 = time.monotonic()
        if shutil.disk_usage(o.parent).free / 1e9 < MIN_FREE_GB:
            raise RuntimeError("磁盘剩余不足，停止")
        from zr.data.calendar import is_trading_day

        if not is_trading_day(day):
            anomalies.append(f"{day} 有文件但交易日历不是交易日")
        pm = raw_root(root) / mv.day_key(day, mv.MINUTE_PREFIX)
        pdd = raw_root(root) / mv.day_key(day, mv.DAY_PREFIX)
        wait_file(pm)
        wait_file(pdd)
        cm = session_close_min(day)
        minute = read_aggs(pm)
        dayf = read_aggs(pdd)
        t = day_tables(minute, cm)
        d = dayf.set_index("ticker")[["open", "high", "low", "close", "volume", "transactions"]]
        d.columns = ["d_open", "d_high", "d_low", "d_close", "d_volume", "d_transactions"]
        d["d_volume"] = d["d_volume"].astype(float)
        daily = d.join(t, how="outer").reset_index().rename(columns={"index": "ticker"})
        daily.insert(0, "date", day)
        daily["close_min"] = cm
        write_atomic(daily, o / "daily" / f"{day.year}" / f"{day}.parquet")
        write_atomic(extra_bars(minute, cm), o / "extra1m" / f"{day.year}" / f"{day}.parquet")
        mark_done(done_p, day)
        pr.step(day, time.monotonic() - t0)
    return {
        "days": len(days),
        "anomalies": anomalies,
        "mean_sec": float(np.mean(pr.secs)) if pr.secs else None,
    }


def plan_days(root: Path) -> list[dt.date]:
    """处理顺序：flat files 列表里 ≥ 2021-11-15 的分钟文件日期（与归档同一份列表）。"""
    ff = mv.FlatFiles()
    o, _ = ff.list(mv.MINUTE_PREFIX + "/")
    ds = sorted(
        dt.date.fromisoformat(x["key"].split("/")[-1][:10])
        for x in o
        if x["key"].endswith(".csv.gz")
    )
    first = out_root(root) / "state" / "first_day.txt"
    start = (
        dt.date.fromisoformat(first.read_text().strip())
        if first.exists()
        else dt.date(2021, 11, 15)
    )
    return [d for d in ds if d >= start]


def meta(root: Path) -> dict:
    rest = mv.Rest()
    o = out_root(root) / "ref"
    o.mkdir(parents=True, exist_ok=True)
    keep = (
        "ticker",
        "name",
        "type",
        "primary_exchange",
        "active",
        "delisted_utc",
        "composite_figi",
        "share_class_figi",
        "cik",
        "list_date",
    )
    rows = []
    for ex in ("XNYS", "XNAS"):
        for act in ("true", "false"):
            for x in rest.tickers(type="CS", exchange=ex, active=act):
                rows.append({k: x.get(k) for k in keep})
    write_atomic(pd.DataFrame(rows), o / "cs_tickers.parquet")
    n_cs = len(rows)
    sp = rest.splits(**{"execution_date.gte": "2021-01-01"})
    write_atomic(
        pd.DataFrame(
            [
                {k: x.get(k) for k in ("ticker", "execution_date", "split_from", "split_to")}
                for x in sp
            ]
        ),
        o / "splits.parquet",
    )
    # 每周首个交易日的时点快照
    days = plan_days(root)
    weeks: dict[tuple[int, int], dt.date] = {}
    for d in days:
        weeks.setdefault(tuple(d.isocalendar()[:2]), d)
    pit_dir = o / "pit"
    n = 0
    for d in sorted(weeks.values()):
        p = pit_dir / f"{d}.parquet"
        if p.exists():
            continue
        rows = []
        for ex in ("XNYS", "XNAS"):
            for x in rest.tickers(type="CS", exchange=ex, date=str(d)):
                rows.append(
                    {
                        "ticker": x.get("ticker"),
                        "primary_exchange": x.get("primary_exchange"),
                        "type": x.get("type"),
                    }
                )
        write_atomic(pd.DataFrame(rows), p)
        n += 1
        if n % 20 == 0:
            log.info("pit 快照 %d/%d（%s）", n, len(weeks), d)
    # 抽查：12 个交易日的逐日时点快照（核对按周快照的近似）
    rng = np.random.default_rng(57)
    chk = sorted(rng.choice(np.array(days, dtype=object), size=min(12, len(days)), replace=False))
    for d in chk:
        p = pit_dir / "daily_check" / f"{d}.parquet"
        if p.exists():
            continue
        rows = []
        for ex in ("XNYS", "XNAS"):
            for x in rest.tickers(type="CS", exchange=ex, date=str(d)):
                rows.append(
                    {"ticker": x.get("ticker"), "primary_exchange": x.get("primary_exchange")}
                )
        write_atomic(pd.DataFrame(rows), p)
    return {"cs_records": n_cs, "splits": len(sp), "weeks": len(weeks), "rest_calls": rest.calls}


def pass2(root: Path, limit: int | None = None) -> dict:
    o = out_root(root)
    rows_all = pd.read_parquet(o / "derived" / "pass2_rows.parquet")
    rows_all["date"] = pd.to_datetime(rows_all["date"]).dt.date
    done_p = o / "state" / "pass2_done.txt"
    done = done_set(done_p)
    by_day = {d: g for d, g in rows_all.groupby("date")}
    days = [d for d in sorted(by_day) if str(d) not in done]
    if limit:
        days = days[:limit]
    pr = Progress(o / "PROGRESS.md", "pass2", len(days))
    for day in days:
        t0 = time.monotonic()
        g = by_day[day]
        pm = raw_root(root) / mv.day_key(day, mv.MINUTE_PREFIX)
        wait_file(pm)
        cm = session_close_min(day)
        minute = read_aggs(pm)
        groups = rth_groups(minute, cm, set(g["symbol"]))
        top = g[g["rank"] > 0]
        bars = [groups[s][5] for s in top["symbol"] if s in groups]
        if bars:
            b = pd.concat(bars, ignore_index=True)
            b = b.assign(
                ts=pd.to_datetime(b["window_start"].to_numpy(), unit="ns", utc=True),
                volume=b["volume"].astype(float),
            )
            b = b[
                ["ts", "ticker", "open", "high", "low", "close", "volume", "transactions"]
            ].rename(columns={"ticker": "symbol"})
            write_atomic(b, o / "bars1" / f"{day.year}" / f"{day}.parquet")
        res = outcomes(groups, g[g["base"]])
        res.insert(0, "date", day)
        write_atomic(res, o / "cand_out" / f"{day.year}" / f"{day}.parquet")
        mark_done(done_p, day)
        pr.step(day, time.monotonic() - t0)
    return {"days": len(days), "mean_sec": float(np.mean(pr.secs)) if pr.secs else None}


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pass1", "meta", "pass2"], required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    fn = {"pass1": pass1, "meta": meta, "pass2": pass2}[a.phase]
    res = fn(root, a.limit) if a.phase != "meta" else fn(root)
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
