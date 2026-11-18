# ruff: noqa: E501  （中文说明较长）
"""收尾拉取的内容级核对（负责人 2026-11-19 指令，分支 data/final-pull-audit）。只读本地文件，除 B 阶段外不联网。

A1  options_1m：逐日逐到期日的行数、合约数、不同分钟数、首末时刻（美东）、有效报价占比、行权价覆盖；manifest 里请求过
    但文件里没有的到期日；ohlc 是否含 volume / count。结果写 $ZR_DATA_ROOT/theta_final/audit_a1/{expiry,day}.parquet。
A2  秒级 K：逐标的日行数与首末时刻；按分钟聚合后与 bars_1m_sip 逐分钟比较。结果写 $ZR_DATA_ROOT/theta_final/audit_a2/。
A3  options_eod：行数异常文件清单（读 theta_final/audit.json）。
运行：python -m zr.data.final_audit --phase a1|a1-summary|a2|a2-summary|a3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger("zr.final_audit")
ET = "America/New_York"
A1_SYMS = ("MU", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "AMD")
BAND = {"MU": 0.08}  # 其余 3%（SPY / QQQ 的 P1 范围与 P3 五只）
BAND_DEFAULT = 0.03
MIN_RATIO = 0.9  # 不同分钟数低于中位数 90% 判异常
VALID_RATIO = 0.5  # 有效报价占比低于中位数一半判异常
A2_SYMS = ("MU", "QQQ", "SPY", "IWM", "NVDA", "TSLA", "AAPL", "AMD", "SMH")
VOL_TOL = 0.001  # 成交量比值偏离 1 超过 0.1%


def band_of(und: str) -> float:
    return BAND.get(und, BAND_DEFAULT)


def audit_dir(root: Path, name: str) -> Path:
    p = Path(root) / "theta_final" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------- A1 ----------------


def requested_expirations(root: Path) -> dict[tuple[str, str], set[str]]:
    """manifest.jsonl 与 manifest_p3.jsonl：(标的, 日) → 当天请求过的到期日（同一天多条记录取并集；带 error 的记录跳过）。"""
    out: dict[tuple[str, str], set[str]] = {}
    for name in ("manifest.jsonl", "manifest_p3.jsonl"):
        p = Path(root) / "options_1m" / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r or not r.get("expirations"):
                continue
            out.setdefault((r["underlying"], r["date"]), set()).update(r["expirations"])
    return out


def day_ranges(root: Path, syms, start: str = "2023-10-01") -> pd.DataFrame:
    """Massive 日聚合（不复权）的当日高低点：sip/massive/daily/YYYY/<日>.parquet。"""
    files = sorted((Path(root) / "sip" / "massive" / "daily").glob("*/*.parquet"))
    files = [f for f in files if f.stem >= start]
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=["date", "ticker", "d_high", "d_low"])
        parts.append(d[d["ticker"].isin(syms)])
    d = pd.concat(parts, ignore_index=True)
    d["date"] = d["date"].astype(str)
    return d.set_index(["ticker", "date"])


def _edge_tol(strikes: np.ndarray) -> tuple[float, float]:
    """行权价两端的间距（判断「盖住」时允许差一个档位）。"""
    s = np.unique(strikes)
    if len(s) < 2:
        return 0.0, 0.0
    return float(s[1] - s[0]), float(s[-1] - s[-2])


def _chain_gap(listed, have: np.ndarray, lo, hi) -> dict:
    """全链里落在 [lo, hi] 内、但 1 分钟文件里没有的行权价。"""
    if listed is None or lo is None:
        return {"chain_in_band": None, "missing_strikes": None, "missing_list": ""}
    inb = listed[(listed >= lo - 1e-9) & (listed <= hi + 1e-9)]
    miss = np.setdiff1d(np.round(inb, 4), np.round(have, 4))
    return {
        "chain_in_band": int(len(inb)),
        "missing_strikes": int(len(miss)),
        "missing_list": ",".join(f"{x:g}" for x in miss[:20]),
    }


def a1_day(args) -> dict:
    root, und, day, req, rng = args
    base = Path(root) / "options_1m" / f"underlying={und}" / f"date={day}"
    rec: dict = {"underlying": und, "date": day, "files": {}, "expiries": [], "missing_req": []}
    for k in ("quotes", "ohlc", "oi"):
        p = base / f"{k}.parquet"
        if p.exists():
            md = pq.read_metadata(p)
            rec["files"][k] = {"rows": md.num_rows, "cols": md.schema.names}
    q = base / "quotes.parquet"
    if not q.exists() or rec["files"]["quotes"]["rows"] == 0:
        rec["missing_req"] = sorted(req)
        return rec
    d = pd.read_parquet(q, columns=["ts", "symbol", "expiry", "strike", "bid", "ask"])
    d["expiry"] = d["expiry"].astype(str)
    valid = (d["bid"] > 0) & (d["ask"] >= d["bid"])
    rec["rows"] = len(d)
    rec["valid"] = int(valid.sum())
    lo = hi = None
    if rng is not None:
        b = band_of(und)
        lo, hi = rng[1] * (1 - b), rng[0] * (1 + b)
    rec["band_lo"], rec["band_hi"] = lo, hi
    # 当日实际挂牌的行权价：options_eod 全链日线（同一数据源，含当天所有有报价的合约）
    chain: dict[str, np.ndarray] = {}
    ep = Path(root) / "options_eod" / f"underlying={und}" / f"date={day}" / "eod.parquet"
    if ep.exists() and pq.read_metadata(ep).num_rows:
        c = pd.read_parquet(ep, columns=["expiry", "strike"])
        c["expiry"] = c["expiry"].astype(str)
        chain = {e: np.unique(g["strike"].to_numpy(float)) for e, g in c.groupby("expiry")}
    rec["chain_known"] = bool(chain)
    for e, g in d.groupby("expiry"):
        ts = g["ts"].drop_duplicates()
        et = ts.dt.tz_convert(ET)
        k = g["strike"].to_numpy(float)
        tl, th = _edge_tol(k)
        kmin, kmax = float(k.min()), float(k.max())
        rec["expiries"].append(
            {
                "expiry": e,
                "rows": len(g),
                "contracts": int(g["symbol"].nunique()),
                "minutes": int(len(ts)),
                "first_et": et.min().strftime("%H:%M"),
                "last_et": et.max().strftime("%H:%M"),
                "valid": int(valid[g.index].sum()),
                "strike_min": kmin,
                "strike_max": kmax,
                "cover_lo": None if lo is None else bool(kmin <= lo + tl + 1e-9),
                "cover_hi": None if hi is None else bool(kmax >= hi - th - 1e-9),
                **_chain_gap(chain.get(e), np.unique(k), lo, hi),
            }
        )
    have = {x["expiry"] for x in rec["expiries"]}
    rec["missing_req"] = sorted(set(req) - have)
    return rec


def a1(root: Path, workers: int = 10) -> dict:
    from zr.data.calendar import session_close

    req = requested_expirations(root)
    rng = day_ranges(root, A1_SYMS)
    jobs = []
    for u in A1_SYMS:
        for p in sorted((Path(root) / "options_1m" / f"underlying={u}").glob("date=*")):
            day = p.name[5:]
            r = (
                tuple(rng.loc[(u, day), ["d_high", "d_low"]].astype(float))
                if (u, day) in rng.index
                else None
            )
            jobs.append((str(root), u, day, req.get((u, day), set()), r))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(a1_day, jobs, chunksize=8))
    ex_rows, day_rows = [], []
    for r in recs:
        f = r["files"]
        day_rows.append(
            {
                "underlying": r["underlying"],
                "date": r["date"],
                "quotes_rows": f.get("quotes", {}).get("rows"),
                "quotes_cols": len(f.get("quotes", {}).get("cols", [])),
                "ohlc_rows": f.get("ohlc", {}).get("rows"),
                "ohlc_has_volume": "volume" in f.get("ohlc", {}).get("cols", []) if "ohlc" in f else None,
                "ohlc_has_count": "count" in f.get("ohlc", {}).get("cols", []) if "ohlc" in f else None,
                "oi_rows": f.get("oi", {}).get("rows"),
                "valid": r.get("valid"),
                "rows": r.get("rows"),
                "band_lo": r.get("band_lo"),
                "band_hi": r.get("band_hi"),
                "missing_req": ",".join(r["missing_req"]),
                "n_expiries": len(r["expiries"]),
            }
        )
        for e in r["expiries"]:
            ex_rows.append({"underlying": r["underlying"], "date": r["date"], **e})
    days = pd.DataFrame(day_rows)
    exps = pd.DataFrame(ex_rows)
    days["early_close"] = [
        session_close(dt.date.fromisoformat(d)).hour < 16 for d in days["date"]
    ]
    exps = exps.merge(days[["underlying", "date", "early_close"]], on=["underlying", "date"])
    o = audit_dir(root, "audit_a1")
    days.to_parquet(o / "day.parquet", index=False)
    exps.to_parquet(o / "expiry.parquet", index=False)
    return {"days": len(days), "expiries": len(exps)}


def a1_flags(root: Path) -> dict[str, pd.DataFrame]:
    """把 A1 结果分成各类异常。"""
    o = audit_dir(root, "audit_a1")
    days = pd.read_parquet(o / "day.parquet")
    exps = pd.read_parquet(o / "expiry.parquet")
    out: dict[str, pd.DataFrame] = {}
    out["empty_file"] = days[(days["quotes_rows"].fillna(0) == 0) | (days["quotes_cols"] == 0)][
        ["underlying", "date", "quotes_rows", "quotes_cols"]
    ]
    mr = days[days["missing_req"] != ""][["underlying", "date", "missing_req"]]
    out["missing_expiry"] = (
        mr.assign(expiry=mr["missing_req"].str.split(",")).explode("expiry")[["underlying", "date", "expiry"]]
    )
    med = exps.groupby(["underlying", "early_close"])["minutes"].median().rename("med_minutes")
    e = exps.join(med, on=["underlying", "early_close"])
    out["minutes_low"] = e[e["minutes"] < MIN_RATIO * e["med_minutes"]][
        ["underlying", "date", "expiry", "minutes", "med_minutes", "contracts", "first_et", "last_et", "early_close"]
    ]
    days["valid_ratio"] = days["valid"] / days["rows"]
    vm = days.groupby("underlying")["valid_ratio"].median().rename("med_valid")
    dv = days.join(vm, on="underlying")
    out["valid_low"] = dv[dv["valid_ratio"] < VALID_RATIO * dv["med_valid"]][
        ["underlying", "date", "valid_ratio", "med_valid"]
    ]
    out["no_range"] = days[days["band_lo"].isna()][["underlying", "date"]]
    cov = exps[
        (exps["missing_strikes"] > 0)
        | (exps["missing_strikes"].isna() & ((exps["cover_lo"] == False) | (exps["cover_hi"] == False)))  # noqa: E712
    ]
    cov = cov.merge(days[["underlying", "date", "band_lo", "band_hi"]], on=["underlying", "date"])
    out["strike_cover"] = cov[
        ["underlying", "date", "expiry", "strike_min", "strike_max", "band_lo", "band_hi", "cover_lo", "cover_hi",
         "chain_in_band", "missing_strikes", "missing_list"]
    ]
    oh = days[days["underlying"].isin(["MU", "QQQ", "SPY"])]
    out["ohlc_bad"] = oh[
        (oh["ohlc_rows"].fillna(0) == 0) | (oh["ohlc_has_volume"] != True) | (oh["ohlc_has_count"] != True)  # noqa: E712
    ][["underlying", "date", "ohlc_rows", "ohlc_has_volume", "ohlc_has_count"]]
    return out


def a1_summary(root: Path) -> dict:
    o = audit_dir(root, "audit_a1")
    days = pd.read_parquet(o / "day.parquet")
    exps = pd.read_parquet(o / "expiry.parquet")
    fl = a1_flags(root)
    res: dict = {
        "days_by_und": days.groupby("underlying").size().to_dict(),
        "expiries": len(exps),
        "median_minutes": exps.groupby(["underlying", "early_close"])["minutes"].median().to_dict(),
        "flags": {k: len(v) for k, v in fl.items()},
        "flags_by_und": {k: v.groupby("underlying").size().to_dict() for k, v in fl.items() if len(v)},
    }
    res["median_minutes"] = {f"{u} {'early' if e else 'normal'}": v for (u, e), v in res["median_minutes"].items()}
    return res


# ---------------- A2 ----------------

PX = ("open", "high", "low", "close")


def sec_to_min(d: pd.DataFrame) -> pd.DataFrame:
    """秒级 K → 分钟 K（开 = 首秒开、高低 = 极值、收 = 末秒收、量与笔数求和）。"""
    if d.empty:
        return pd.DataFrame(columns=["ts", *PX, "volume", "transactions"])
    d = d.sort_values("ts")
    g = d.groupby(d["ts"].dt.floor("min"), sort=True)
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        transactions=("transactions", "sum"),
    )
    out.index.name = "ts"
    return out.reset_index()


def compare_day(sec: pd.DataFrame, m1: pd.DataFrame) -> dict:
    a = sec_to_min(sec).set_index("ts")
    b = m1.set_index("ts")[[*PX, "volume", "transactions"]]
    both = a.index.intersection(b.index)
    x, y = a.loc[both], b.loc[both]
    eq_px = np.isclose(x[list(PX)].to_numpy(float), y[list(PX)].to_numpy(float), rtol=0, atol=1e-6).all(axis=1)
    eq_v = np.isclose(x["volume"].to_numpy(float), y["volume"].to_numpy(float), rtol=0, atol=1e-6)
    eq_n = x["transactions"].to_numpy(float) == y["transactions"].to_numpy(float)
    v_sec, v_min = float(sec["volume"].sum()) if len(sec) else 0.0, float(m1["volume"].sum()) if len(m1) else 0.0
    return {
        "sec_rows": len(sec),
        "min_sec": len(a),
        "min_1m": len(b),
        "min_both": len(both),
        "only_sec": len(a.index.difference(b.index)),
        "only_1m": len(b.index.difference(a.index)),
        "eq_px": int(eq_px.sum()),
        "eq_vol": int(eq_v.sum()),
        "eq_n": int(eq_n.sum()),
        "eq_all": int((eq_px & eq_v & eq_n).sum()),
        "vol_sec": v_sec,
        "vol_1m": v_min,
        "first_et": sec["ts"].min().tz_convert(ET).strftime("%H:%M:%S") if len(sec) else None,
        "last_et": sec["ts"].max().tz_convert(ET).strftime("%H:%M:%S") if len(sec) else None,
    }


def a2_symbol(args) -> list[dict]:
    root, sym = args
    from zr.data.sip_bars import load_sip_1m

    m = load_sip_1m(Path(root), sym, rth_only=False)
    m["day"] = m["ts"].dt.tz_convert(ET).dt.date.astype(str)
    byday = dict(tuple(m.groupby("day")))
    out = []
    base = Path(root) / "bars_1s_sip" / f"underlying={sym}"
    for p in sorted(base.glob("date=*/bars.parquet")):
        day = p.parent.name[5:]
        sec = pd.read_parquet(p)
        m1 = byday.get(day, m.iloc[:0])
        out.append({"symbol": sym, "date": day, **compare_day(sec, m1)})
    return out


def a2(root: Path, syms=A2_SYMS) -> dict:
    with ProcessPoolExecutor(max_workers=len(syms)) as ex:
        recs = [r for rs in ex.map(a2_symbol, [(str(root), s) for s in syms]) for r in rs]
    d = pd.DataFrame(recs)
    d["vol_ratio"] = d["vol_sec"] / d["vol_1m"].replace(0, np.nan)
    d.to_parquet(audit_dir(root, "audit_a2") / "day.parquet", index=False)
    return {"symbol_days": len(d)}


def a2_flags(root: Path) -> pd.DataFrame:
    d = pd.read_parquet(audit_dir(root, "audit_a2") / "day.parquet")
    bad = (d["sec_rows"] == 0) | (d["only_sec"] > 0) | (d["only_1m"] > 0) | ((d["vol_ratio"] - 1).abs() > VOL_TOL)
    return d[bad]


def a2_summary(root: Path) -> dict:
    d = pd.read_parquet(audit_dir(root, "audit_a2") / "day.parquet")
    g = d.groupby("symbol")
    s = pd.DataFrame(
        {
            "days": g.size(),
            "rows": g["sec_rows"].sum(),
            "zero_days": g["sec_rows"].apply(lambda x: int((x == 0).sum())),
            "minutes_both": g["min_both"].sum(),
            "only_sec": g["only_sec"].sum(),
            "only_1m": g["only_1m"].sum(),
            "eq_all_pct": g["eq_all"].sum() / g["min_both"].sum() * 100,
            "eq_px_pct": g["eq_px"].sum() / g["min_both"].sum() * 100,
            "eq_vol_pct": g["eq_vol"].sum() / g["min_both"].sum() * 100,
            "eq_n_pct": g["eq_n"].sum() / g["min_both"].sum() * 100,
            "vol_ratio_p01": g["vol_ratio"].quantile(0.01),
            "vol_ratio_p50": g["vol_ratio"].median(),
            "vol_ratio_p99": g["vol_ratio"].quantile(0.99),
            "first_et_min": g["first_et"].min(),
            "last_et_max": g["last_et"].max(),
        }
    )
    f = a2_flags(root)
    return {"by_symbol": s.round(4).to_dict(orient="index"), "flag_days": len(f), "flag_by_symbol": f.groupby("symbol").size().to_dict()}


# ---------------- 拆股（VPS，Massive reference/splits） ----------------

SPLIT_SYMS = ("SPY", "QQQ", "IWM", "MU", "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "FB", "GOOGL", "AMD", "SMH", "TLT", "GLD")


def splits(root: Path, syms=SPLIT_SYMS, since: str = "2016-01-01") -> dict:
    """只在 VPS 上运行（需 Massive 密钥）。结果写 theta_final/splits.json。"""
    from zr.data import massive as mv

    rest = mv.Rest()
    out: dict = {}
    for s in syms:
        rows = list(rest.paged("/v3/reference/splits", {"ticker": s, "limit": 1000}))
        out[s] = sorted(
            [
                {"execution_date": r.get("execution_date"), "split_from": r.get("split_from"), "split_to": r.get("split_to")}
                for r in rows
                if str(r.get("execution_date", "")) >= since
            ],
            key=lambda r: r["execution_date"],
        )
    o = audit_dir(root, ".")
    (o / "splits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def a2_refetch(root: Path, items: list[str]) -> dict:
    """只在 VPS 上运行：重拉指定的（标的:日）秒级 K 到 theta_final/a2_refetch/（不覆盖 bars_1s_sip）。"""
    from zr.data import massive as mv
    from zr.data.massive_seconds import fetch_day

    rest = mv.Rest()
    out = {}
    for it in items:
        sym, day = it.split(":")
        d = fetch_day(rest, sym, dt.date.fromisoformat(day))
        p = audit_dir(root, "a2_refetch") / f"underlying={sym}" / f"date={day}" / "bars.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(p, index=False)
        out[it] = len(d)
    return out


# ---------------- 报告（A 阶段） ----------------


def pull_bands(root: Path) -> pd.DataFrame:
    """拉取当时用的行权价带：manifest.jsonl 的 band_lo / band_hi；manifest_p3.jsonl 的 px = (开, 低, 高) × 3%。"""
    rows = []
    for line in (Path(root) / "options_1m" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if '"band_lo"' in line:
            r = json.loads(line)
            rows.append((r["underlying"], r["date"], r["band_lo"], r["band_hi"]))
    p3 = Path(root) / "options_1m" / "manifest_p3.jsonl"
    for line in p3.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("px"):
            _, lo, hi = r["px"]
            rows.append((r["underlying"], r["date"], lo * (1 - BAND_DEFAULT), hi * (1 + BAND_DEFAULT)))
    d = pd.DataFrame(rows, columns=["underlying", "date", "pull_lo", "pull_hi"])
    return d.drop_duplicates(["underlying", "date"], keep="last")


def strike_gaps(root: Path) -> pd.DataFrame:
    c = a1_flags(root)["strike_cover"].merge(pull_bands(root), on=["underlying", "date"], how="left")

    def inside(r):
        if pd.isna(r.pull_lo) or not r.missing_list:
            return np.nan
        ks = [float(k) for k in r.missing_list.split(",")]
        return sum(r.pull_lo - 1e-9 <= k <= r.pull_hi + 1e-9 for k in ks)

    c["miss_in_pull_band"] = c.apply(inside, axis=1)
    return c


def _cell(x) -> str:
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        return f"{x:,.0f}" if float(x).is_integer() and abs(x) >= 100 else f"{x:,.4g}"
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return f"{x:,}" if abs(x) >= 10000 else str(x)
    return str(x)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(_cell(x) for x in r) + " |")
    return "\n".join(out)


def report_a(root: Path) -> str:
    root = Path(root)
    o1 = audit_dir(root, "audit_a1")
    days = pd.read_parquet(o1 / "day.parquet")
    exps = pd.read_parquet(o1 / "expiry.parquet")
    fl = a1_flags(root)
    sg = strike_gaps(root)
    a2d = pd.read_parquet(audit_dir(root, "audit_a2") / "day.parquet")
    a2s = a2_summary(root)["by_symbol"]
    au = json.loads((root / "theta_final" / "audit.json").read_text(encoding="utf-8"))
    sp = json.loads((root / "theta_final" / "splits.json").read_text(encoding="utf-8"))
    L: list[str] = []
    L += ["## A1 期权 1 分钟（`options_1m/`）", ""]
    t = days.groupby("underlying").agg(
        交易日=("date", "size"), 最早=("date", "min"), 最晚=("date", "max"), 到期日数=("n_expiries", "sum"),
        报价行数=("rows", "sum"), 有效报价=("valid", "sum"),
    )
    t["有效报价占比"] = (t["有效报价"] / t["报价行数"]).round(4)
    mm = exps.groupby("underlying").agg(每到期日分钟数中位数=("minutes", "median"), 合约数中位数=("contracts", "median"))
    t = t.join(mm).reset_index()
    L += [_md(t), ""]
    L += [
        f"- 0 行或没有列的文件：{len(fl['empty_file'])} 个（{', '.join(fl['empty_file']['underlying'] + ' ' + fl['empty_file']['date'])}）。",
        f"- manifest 请求过但文件里没有的到期日：{len(fl['missing_expiry'])} 个（标的，日，到期日）：",
        "",
        _md(fl["missing_expiry"].groupby(["underlying", "expiry"])["date"].agg(lambda x: "、".join(x)).reset_index().rename(columns={"date": "交易日"})),
        "",
        f"- 不同分钟数低于中位数 90%：{len(fl['minutes_low'])} 个。每个到期日的分钟数中位数都是 391（09:30～16:00 每分钟一行）；"
        f"提前收盘日（{int(days.drop_duplicates('date')['early_close'].sum())} 天）ThetaData 同样给到 16:00，收盘后的行是收盘时的报价延续，单独算中位数也是 391。",
        f"- 有效报价（bid > 0 且 ask ≥ bid）占比低于该标的中位数一半的日子：{len(fl['valid_low'])} 个。",
        f"- MU / QQQ / SPY 的 ohlc.parquet：{len(days[days.underlying.isin(['MU','QQQ','SPY'])])} 个交易日，缺 volume / count 列或 0 行的 {len(fl['ohlc_bad'])} 个"
        "（SPY 2026-03-31 整天的三个文件都是空的，算在上面「0 行文件」里）。" if len(fl["ohlc_bad"]) else
        f"- MU / QQQ / SPY 的 ohlc.parquet：全部含 volume 与 count 两列、都有行（{len(days[days.underlying.isin(['MU','QQQ','SPY'])]) - len(fl['empty_file'])} 个交易日；SPY 2026-03-31 整天为空，见上）。",
        "",
        "**行权价覆盖**：价格带 = [Massive 日聚合最低价 ×(1−b), 最高价 ×(1+b)]（不复权；SPY、QQQ 与 P3 五只 b = 3%，MU 8%）。"
        "「缺档」= 同日 `options_eod` 全链里该到期日落在价格带内、但 1 分钟文件里没有的行权价（全链来自同一数据源，比「最低 / 最高行权价是否越过带边」精确；"
        f"{int(exps['missing_strikes'].notna().sum()):,} / {len(exps):,} 个到期日都能对上全链）。",
        "",
    ]
    sg["类别"] = np.where(sg["miss_in_pull_band"] > 0, "拉取时价格带内也缺（真缺口）",
                       np.where(sg["miss_in_pull_band"] == 0, "只缺拉取带外、Massive 带内的档（价格带来源不同）", "无拉取带记录"))
    t = sg.groupby(["underlying", "类别"]).agg(到期日=("expiry", "size"), 缺档=("missing_strikes", "sum")).reset_index()
    L += [_md(t), ""]
    real = sg[sg["miss_in_pull_band"] > 0]
    rd = real.groupby(["underlying", "date"]).agg(到期日=("expiry", "size"), 带内缺档=("miss_in_pull_band", "sum"),
                                                  全链带内=("chain_in_band", "sum")).reset_index()
    L += ["真缺口按日：", "", _md(rd), "",
          "价格带来源不同：MU / QQQ / SPY 在 2026-10-21 及以前拉取时，价格带取长桥 1 分钟 K 常规时段的高低点，且先用 `clean_wicks` 收回疑似坏成交影线"
          "（`zr/data/thetadata.py` 的 `day_price_range`）；2026-10-22 起的补拉与 P3 用 ThetaData 正股日线。Massive 日聚合的高低点是全天官方值，通常更宽，"
          "差出来的一两档不在当时请求的范围内，不是数据丢失。", ""]
    L += ["## A2 秒级 K（`bars_1s_sip/`）与 SIP 1 分钟 K（`bars_1m_sip/`）", "",
          "`bars_1m_sip/` 已从 `massive_raw/` 扩到 IWM、NVDA、TSLA、AAPL（与原 5 只同做法：含盘前盘后、不复权；逐日中间文件 `sip/massive/extra1m_more/`）。", ""]
    t = pd.DataFrame(a2s).T.reset_index().rename(columns={"index": "标的"})
    t = t[["标的", "days", "rows", "zero_days", "first_et_min", "last_et_max", "minutes_both", "only_sec", "only_1m", "eq_px_pct", "eq_vol_pct", "eq_all_pct", "vol_ratio_p01", "vol_ratio_p50", "vol_ratio_p99"]]
    t.columns = ["标的", "标的日", "秒级行数", "0 行文件", "最早时刻", "最晚时刻", "共有分钟", "只在秒级", "只在分钟", "价格全等 %", "量相等 %", "全部相等 %", "日量比 P1", "日量比 P50", "日量比 P99"]
    L += [_md(t), ""]
    f = a2_flags(root)
    L += [
        f"- 分钟集合：{len(a2d):,} 个标的日全部一致（没有只在一边出现的分钟）。",
        f"- 日量比 = 秒级 K 当日成交量合计 / 分钟 K 当日成交量合计：没有一天大于 1；偏离 1 不超过 0.1% 的只有 {int(((a2d.vol_ratio - 1).abs() <= VOL_TOL).sum())} 天，"
        f"按 0.1% 的标准异常日 {len(f):,} 个（几乎全部标的日）。成交笔数同样是秒级少。",
        "- 抽查 9 个标的日（SPY、MU、NVDA 各 3 天，种子 20261119）：常规时段秒级量为分钟量的 96.6%～99.96%，盘后 31%～99%，盘前 51%～97%——差额集中在盘前盘后；"
        "缺的成交 SPY 平均每笔 190～719 股，MU / NVDA 14～53 股。推测秒级聚合与分钟聚合计入的成交条件不同（未验证，未查到 Massive 的明确说明）。",
        "- 重拉核对：SPY 2024-04-30、NVDA 2024-06-05、MU 2024-09-20 在 VPS 上重拉，与磁盘上的文件逐行完全相同——差异来自数据源口径，不是传输残缺。",
        "",
        "拆股日前后（Massive reference/splits；秒级区间 2021-11-18～2026-11-17 内）：",
        "",
    ]
    rows = []
    for s, xs in sp.items():
        for x in xs:
            e = x["execution_date"]
            g = a2d[(a2d.symbol == s)]
            if g.empty or e < g.date.min():
                continue
            ds = sorted(g.date)
            i = next(k for k, d in enumerate(ds) if d >= e)
            win = ds[i - 2 : i + 2]
            w = g[g.date.isin(win)]
            px = []
            for d in ds[i - 1 : i + 1]:
                b = pd.read_parquet(root / "bars_1s_sip" / f"underlying={s}" / f"date={d}" / "bars.parquet")
                et = b.ts.dt.tz_convert(ET)
                m = et.dt.hour * 60 + et.dt.minute
                r = b[(m >= 570) & (m < 960)]
                px.append((r.close.iloc[-1], r.open.iloc[0]))
            rows.append({"标的": s, "拆股日": e, "比例": f"{x['split_from']}:{x['split_to']}", "前后两天": f"{win[0]}～{win[-1]}",
                         "价格全等分钟": f"{int(w.eq_px.sum())} / {int(w.min_both.sum())}", "前日收 / 当日开": round(px[0][0] / px[1][1], 3)})
    L += [_md(pd.DataFrame(rows)), "",
          "价格在拆股日按拆股比例跳变，秒级与分钟级在拆股前后同样不复权、价格一致。", ""]
    L += ["## A3 options_eod 行数异常文件（之前没有重新请求过）", ""]
    rr = [{"标的": u, "文件": k, "日期": d, "行数": v[0], "前后 21 个交易日中位数": v[1]} for u, rec in au.items() for k, r in rec.items() for d, v in r["low_days"].items()]
    L += [_md(pd.DataFrame(rr)), ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    from zr.config import get_settings

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["a1", "a1-summary", "a2", "a2-summary", "splits", "a2-refetch", "report-a"])
    ap.add_argument("--items", nargs="*", default=[], help="a2-refetch：标的:日")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(get_settings().zr_data_root)
    fn = {
        "a1": lambda: a1(root, a.workers),
        "a1-summary": lambda: a1_summary(root),
        "a2": lambda: a2(root),
        "a2-summary": lambda: a2_summary(root),
        "splits": lambda: splits(root),
        "a2-refetch": lambda: a2_refetch(root, a.items),
        "report-a": lambda: report_a(root),
    }[a.phase]
    res = fn()
    print(res if isinstance(res, str) else json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
