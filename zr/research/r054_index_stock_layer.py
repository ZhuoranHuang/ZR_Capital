# ruff: noqa: E501  （中文报告行较长）
"""R-054：三条调研规则在 QQQ / SPY 正股层的复制（扣费前）→ docs/research/R054_INDEX_STOCK_LAYER.md。

预登记见 RESEARCH_LOG R-054（运行前写入并提交）。规则条件、参数、会话窗口、ORB 每日 1 次全部读自
config/intrabar/survey_{orb_breakout,rsi2_pullback,ema_pullback}.yaml，不改任何数。
- 指标：已完成 2 分钟 K（resample_bars）。公式与 zr.features.entry_features.svy_features / bar_block 相同
  （SMA200 / SMA5 距离、EMA50 / EMA200 排列、EMA50 收回与 5 根前同侧、Wilder RSI(2)、Wilder ADX(14)）；
  entry_features 依赖秒级平价价（只有 MU），所以在本模块按 1 分钟 K 重写，并做截断检验。
- 口径 C（主）：含盘前盘后的长桥原始 1 分钟 K（load_extended_bars(clean=False)），全序列连续、不按日重置；
  口径 D-027（敏感性）：RTH-only 1 分钟 K（load_bars），指标按日重置。
- 开盘区间：RTH 1 分钟 K 09:30–09:34 的最高 / 最低，方向 = 09:34 收盘 / 09:30 开盘 − 1；09:35 起可用；
  突破用信号 K 收盘价。
- 信号：2 分钟 K 收盘时刻（标签 + 2 分钟）在会话窗口 [first_entry, last_entry] 内且条件成立的上升沿，
  同规则同方向 300 秒内不重复（R-049 _edges）；ORB 每天只取最早 1 个。
- 入场：信号收盘时刻那根 RTH 1 分钟 K 的开盘价；不存在或 ≥ flat 丢弃计数。收益 h 分钟 = 入场后第 h 根
  1 分钟 K 的收盘 / 入场价 − 1（方向化，%）；超出收盘丢弃计数。MFE 同期。
- 基线：每个触发 50 个随机分钟（同日、会话窗口内、触发前后 30 分钟内、同方向），排除与当日任一触发
  （三条规则、两个方向，同口径）相距 ≤ 2 分钟的时刻；同样「该分钟 1 分钟 K 开盘」入场。
运行：python -m zr.research.r054_index_stock_layer（计算并落盘）；--report（生成报告）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from zr.backtest.intrabar import calibration_window
from zr.config import get_settings
from zr.data.calendar import session_close
from zr.features.bars import ET, et_date, load_bars, load_extended_bars, resample_bars
from zr.features.indicators import adx, wilder_rsi

log = logging.getLogger("zr.research.r054")
REPO = Path(__file__).resolve().parents[2]
OUT_MD = REPO / "docs" / "research" / "R054_INDEX_STOCK_LAYER.md"
RULES = ("survey_orb_breakout", "survey_rsi2_pullback", "survey_ema_pullback")
RULE_CN = {"survey_orb_breakout": "开盘区间突破", "survey_rsi2_pullback": "RSI(2) 回调", "survey_ema_pullback": "均线回踩"}
UNDS = ("QQQ", "SPY")
CONVS = ("C", "D027")
HS = (1, 2, 5, 10, 15, 30)
REFRACTORY_S = 300
N_RAND = 50
RAND_WIN_MIN = 30
RAND_EXCL_MIN = 2
MIN_DAYS = 10
BH_Q = 0.10
NS = 10**9
MIN_NS = 60 * NS
OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal}


def store(root: Path) -> Path:
    p = root / "research" / "r054"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_rules() -> dict[str, dict]:
    out = {}
    for n in RULES:
        out[n] = yaml.safe_load((REPO / "config" / "intrabar" / f"{n}.yaml").read_text(encoding="utf-8"))
    return out


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ---------------- 特征（每根已完成 2 分钟 K 一个值；(多, 空) 两个方向） ----------------


def bar_feats(b: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """与 entry_features.svy_features / bar_block 相同的公式；返回 名 → (d=+1 值, d=−1 值)。"""
    c, h, lo = b["close"], b["high"], b["low"]
    sma200 = c.rolling(200, min_periods=200).mean()
    sma5 = c.rolling(5, min_periods=5).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    side = np.sign((c - e50).to_numpy())
    prev = np.r_[np.nan, side[:-1]]
    lag5 = np.r_[np.full(min(5, len(side)), np.nan), side[:-5]][: len(side)]
    s200 = ((c / sma200 - 1) * 1e4).to_numpy()
    s5 = ((c / sma5 - 1) * 1e4).to_numpy()
    al = ((e50 / e200 - 1) * 1e4).to_numpy()
    rsi = wilder_rsi(c, 2).to_numpy()
    ax = adx(h, lo, c, 14)["adx"].to_numpy()
    return {
        "svy_sma200_dist_bp": (s200, -s200),
        "svy_sma5_dist_bp": (s5, -s5),
        "svy_ema50_200_align_bp": (al, -al),
        "svy_ema50_cross": (((side > 0) & (prev <= 0)) * 1.0, ((side < 0) & (prev >= 0)) * 1.0),
        "svy_ema50_side_lag5": ((lag5 > 0) * 1.0, (lag5 < 0) * 1.0),
        "b2_rsi2": (rsi, 100 - rsi),
        "b2_adx14": (ax, ax),
    }


def _two_min(b1: pd.DataFrame) -> pd.DataFrame:
    if b1.empty:
        return b1
    b = resample_bars(b1, "2min")
    return b[b["close"].notna()]


def orb_feats(rth_day: pd.DataFrame, open_ns: int, close_px: np.ndarray, close_ns: np.ndarray):
    """开盘区间 5 分钟：RTH 1 分钟 K 标签 09:30–09:34；收盘时刻 ≥ 09:35 才可用。"""
    lab = rth_day.index.as_unit("ns").asi8
    seg = rth_day[(lab >= open_ns) & (lab < open_ns + 5 * MIN_NS)]
    n = len(close_px)
    if seg.empty:
        nan = np.full(n, np.nan)
        return {"svy_or5_break_bp": (nan, nan), "svy_or5_dir_bp": (nan, nan)}
    hi, lo = float(seg["high"].max()), float(seg["low"].min())
    odir = (float(seg["close"].iloc[-1]) / float(seg["open"].iloc[0]) - 1) * 1e4
    avail = close_ns >= open_ns + 5 * MIN_NS
    up = np.where(avail, (close_px / hi - 1) * 1e4, np.nan)
    dn = np.where(avail, (lo / close_px - 1) * 1e4, np.nan)
    od = np.where(avail, odir, np.nan)
    return {"svy_or5_break_bp": (up, dn), "svy_or5_dir_bp": (od, -od)}


def _edges_time(on: np.ndarray, t_ns: np.ndarray, refractory_s: int = REFRACTORY_S) -> np.ndarray:
    """上升沿 + 不应期（同 owner_rules_oos._edges，按时间）。返回下标。"""
    rise = np.flatnonzero(on & ~np.r_[False, on[:-1]])
    out, last = [], None
    for k in rise:
        if last is None or t_ns[k] - last >= refractory_s * NS:
            out.append(k)
            last = t_ns[k]
    return np.array(out, dtype=int)


def compute_signals(ext1: pd.DataFrame, rth1: pd.DataFrame, conv: str, rules: dict, days=None) -> pd.DataFrame:
    """1 分钟 K → 触发表（day, rule, dir, sig_ns = 信号 K 收盘时刻）。只用收盘时刻 ≤ 信号时刻的 K。"""
    rows = []
    rth_days = et_date(rth1.index)
    if conv == "C":
        b2 = _two_min(ext1)
        feats_all = bar_feats(b2)
        b2_day = et_date(b2.index)
    day_list = sorted(set(rth_days)) if days is None else days
    for day in day_list:
        rd = rth1[rth_days == day]
        if rd.empty:
            continue
        if conv == "C":
            m = b2_day == day
            bd = b2[m]
            fd = {k: (a[m], b[m]) for k, (a, b) in feats_all.items()}
        else:
            bd = _two_min(rd)
            fd = bar_feats(bd)
        if bd.empty:
            continue
        open_ns = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).value
        cl_ns = bd.index.as_unit("ns").asi8 + 2 * MIN_NS
        fd.update(orb_feats(rd, open_ns, bd["close"].to_numpy(float), cl_ns))
        mod = (cl_ns - open_ns) // MIN_NS + 9 * 60 + 30  # 收盘时刻（美东分钟）
        for name, r in rules.items():
            s = r["session"]
            in_win = (mod >= _hm(s["first_entry"])) & (mod <= _hm(s["last_entry"]))
            trig = {}
            for d, j in ((1, 0), (-1, 1)):
                cond = in_win.copy()
                for c in r["entry"]["all"]:
                    v = fd[c["name"]][j]
                    with np.errstate(invalid="ignore"):
                        cond &= OPS[c["op"]](v, c["value"]) & np.isfinite(v)
                trig[d] = _edges_time(cond, cl_ns)
            cap = r.get("max_entries_per_day")
            if cap:
                allk = sorted((cl_ns[k], d) for d, ks in trig.items() for k in ks)[: int(cap)]
                trig = {d: [k for k in trig[d] if (cl_ns[k], d) in allk] for d in trig}
            for d, ks in trig.items():
                for k in ks:
                    rows.append({"day": day, "rule": name, "dir": d, "sig_ns": int(cl_ns[k])})
    return pd.DataFrame(rows, columns=["day", "rule", "dir", "sig_ns"])


# ---------------- 入场、收益、随机基线 ----------------


def minute_arrays(rth_day: pd.DataFrame, day: dt.date) -> dict:
    """当日 RTH 1 分钟 K 排到 09:30 起的分钟格（缺失 = NaN）。"""
    open_ns = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).value
    close_ns = pd.Timestamp(dt.datetime.combine(day, session_close(day)), tz=ET).value
    n = int((close_ns - open_ns) // MIN_NS)
    k = (rth_day.index.as_unit("ns").asi8 - open_ns) // MIN_NS
    ok = (k >= 0) & (k < n)
    out = {"open_ns": open_ns, "n": n}
    for c in ("open", "high", "low", "close"):
        a = np.full(n, np.nan)
        a[k[ok]] = rth_day[c].to_numpy(float)[ok]
        out[c] = a
    return out


def outcomes(ma: dict, k: int, d: int, flat_k: int) -> dict | None:
    """信号收盘时刻 = 分钟格 k（该分钟 K 开盘入场）。返回各 h 的方向化收益与 MFE（%）；入场无效 → None。"""
    if k < 0 or k >= min(flat_k, ma["n"]) or not np.isfinite(ma["open"][k]):
        return None
    p0 = ma["open"][k]
    out = {"entry": p0}
    for h in HS:
        e = k + h - 1
        if e >= ma["n"] or not np.isfinite(ma["close"][e]):
            out[f"ret{h}"] = math.nan
            out[f"mfe{h}"] = math.nan
            continue
        out[f"ret{h}"] = d * (ma["close"][e] / p0 - 1) * 100
        if d == 1:
            out[f"mfe{h}"] = (np.nanmax(ma["high"][k : e + 1]) / p0 - 1) * 100
        else:
            out[f"mfe{h}"] = (1 - np.nanmin(ma["low"][k : e + 1]) / p0) * 100
    return out


def random_pool(k: int, trig_ks: np.ndarray, win_lo: int, win_hi: int, valid: np.ndarray) -> np.ndarray:
    """候选分钟：会话窗口 [win_lo, win_hi] 内、|j − k| ≤ 30、与任一触发相距 > 2 分钟、该分钟可入场。"""
    j = np.arange(max(win_lo, k - RAND_WIN_MIN), min(win_hi, k + RAND_WIN_MIN) + 1)
    j = j[(j >= 0) & (j < len(valid))]
    if len(trig_ks):
        far = np.min(np.abs(j[:, None] - np.asarray(trig_ks)[None, :]), axis=1) > RAND_EXCL_MIN
        j = j[far]
    return j[valid[j]]


def evaluate(sig: pd.DataFrame, rth1: pd.DataFrame, rules: dict, seed: int, tag: str) -> tuple[pd.DataFrame, dict]:
    rows, cnt = [], {"entry_dropped": 0, "trig": len(sig)}
    rth_days = et_date(rth1.index)
    for day, sd in sig.groupby("day"):
        ma = minute_arrays(rth1[rth_days == day], day)
        o = ma["open_ns"]
        tk = ((sd["sig_ns"].to_numpy() - o) // MIN_NS).astype(int)
        valid = np.isfinite(ma["open"])
        rng = np.random.default_rng([seed, day.toordinal(), sum(map(ord, tag))])
        for (_, r), k in zip(sd.iterrows(), tk, strict=True):
            rule = rules[r["rule"]]
            s = rule["session"]
            flat_k = _hm(s["flat"]) - (9 * 60 + 30)
            res = outcomes(ma, k, r["dir"], flat_k)
            if res is None:
                cnt["entry_dropped"] += 1
                continue
            lo, hi = _hm(s["first_entry"]) - 570, min(_hm(s["last_entry"]) - 570, flat_k - 1)
            vv = valid.copy()
            vv[min(flat_k, len(vv)) :] = False
            pool = random_pool(k, tk, lo, hi, vv)
            rk = rng.choice(pool, size=min(N_RAND, len(pool)), replace=False) if len(pool) else []
            rr = [outcomes(ma, int(j), r["dir"], flat_k) for j in rk]
            row = {"day": day, "rule": r["rule"], "dir": r["dir"], "sig_ns": r["sig_ns"], "k": k, "n_rand": len(rr), **res}
            for h in HS:
                v = np.array([x[f"ret{h}"] for x in rr], float)
                v = v[np.isfinite(v)]
                row[f"rand{h}"] = float(v.mean()) if len(v) else math.nan
            rows.append(row)
    return pd.DataFrame(rows), cnt


def run(root: Path, seed: int = 0) -> None:
    rules = load_rules()
    lo, hi = calibration_window()
    out = []
    counts = []
    for und in UNDS:
        rth1 = load_bars(root, und)
        ext1 = load_extended_bars(root, und, clean=False)
        rd = et_date(rth1.index)
        keep = ~((rd >= lo) & (rd <= hi))  # D-033
        rth1 = rth1[keep]
        days = sorted(set(et_date(rth1.index)))
        for conv in CONVS:
            sig = compute_signals(ext1, rth1, conv, rules, days)
            ev, cnt = evaluate(sig, rth1, rules, seed, f"{und}{conv}")
            ev["und"], ev["conv"] = und, conv
            out.append(ev)
            counts.append({"und": und, "conv": conv, "days": len(days), **cnt})
            log.info("%s %s：触发 %d，入场丢弃 %d", und, conv, cnt["trig"], cnt["entry_dropped"])
    df = pd.concat(out, ignore_index=True)
    df.assign(day=df["day"].astype(str)).to_parquet(store(root) / "triggers.parquet", index=False)
    pd.DataFrame(counts).to_csv(store(root) / "counts.csv", index=False)
    # 真实数据截断检验：QQQ 口径 C，抽 5 天、每天 4 个时刻
    rth1 = load_bars(root, "QQQ")
    ext1 = load_extended_bars(root, "QQQ", clean=False)
    rng = np.random.default_rng(seed)
    days = sorted(set(et_date(rth1.index)))
    pick = sorted(rng.choice(len(days), 5, replace=False))
    res = []
    for i in pick:
        day = days[i]
        lo_d = day - dt.timedelta(days=20)
        e = ext1[(et_date(ext1.index) >= lo_d) & (et_date(ext1.index) <= day)]
        r = rth1[(et_date(rth1.index) >= lo_d) & (et_date(rth1.index) <= day)]
        for conv in CONVS:
            full = compute_signals(e, r, conv, rules, [day])
            for hh in (10, 11, 13, 15):
                t = pd.Timestamp(dt.datetime.combine(day, dt.time(hh, 1)), tz=ET).value
                tr = compute_signals(_cut(e, t), _cut(r, t), conv, rules, [day])
                a = _rows(full[full["sig_ns"] <= t])
                b = _rows(tr[tr["sig_ns"] <= t])
                res.append({"day": str(day), "conv": conv, "t": f"{hh}:01", "n_full": len(a), "n_cut": len(b), "same": a == b})
    pd.DataFrame(res).to_csv(store(root) / "truncation.csv", index=False)


def _rows(x: pd.DataFrame) -> list[tuple]:
    """逐条比较用（空表的 dtype 不同不应算不一致）。"""
    return [(str(r.day), r.rule, int(r.dir), int(r.sig_ns)) for r in x.itertuples()]


def _cut(b1: pd.DataFrame, t_ns: int) -> pd.DataFrame:
    """只保留收盘时刻（标签 + 60 秒）≤ t 的 1 分钟 K。"""
    return b1[b1.index.as_unit("ns").asi8 + MIN_NS <= t_ns]


# ---------------- 统计与报告 ----------------


def day_t(ex: np.ndarray, days: np.ndarray) -> tuple[float, float, float, int]:
    """先按日取均值，再对日均值做单侧 t（H0 ≤ 0）。返回 (日均值的均值, t, p, 天数)。"""
    s = pd.Series(ex).groupby(days).mean()
    G = len(s)
    if G < MIN_DAYS:
        return (float(s.mean()) if G else math.nan), math.nan, 1.0, G
    m, sd = float(s.mean()), float(s.std(ddof=1))
    if sd == 0:
        return m, math.nan, 1.0, G
    t = m / (sd / math.sqrt(G))
    return m, t, float(stats.t.sf(t, G - 1)), G


def bh_q(p: np.ndarray) -> np.ndarray:
    """BH 调整后的 q 值。"""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def cell_stats(sub: pd.DataFrame, h: int) -> dict:
    x, r = sub[f"ret{h}"].to_numpy(float), sub[f"rand{h}"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(r)
    ex = x[ok] - r[ok]
    m, t, p, G = day_t(ex, sub["day"].to_numpy()[ok])
    return {"n": int(ok.sum()), "days": G, "excess": m, "ret": float(np.nanmean(x[ok])) if ok.any() else math.nan,
            "rand": float(np.nanmean(r[ok])) if ok.any() else math.nan, "t": t, "p": p,
            "mfe": float(np.nanmean(sub[f"mfe{h}"].to_numpy(float)[ok])) if ok.any() else math.nan,
            "dropped_h": int((~np.isfinite(x)).sum())}


def thresholds(root: Path) -> tuple[dict, dict]:
    """COST_REPORT 门槛：(und, dir, h) → (均值, 中位, n)；以及各标的成本数据窗口。"""
    from zr.research import cost_report as cr

    out, win = {}, {}
    for und in UNDS:
        fs = sorted(cr.out_dir(root, und).glob("date=*.parquet"))
        if not fs:
            continue
        win[und] = (dt.date.fromisoformat(fs[0].name[5:15]), dt.date.fromisoformat(fs[-1].name[5:15]))
        d, _, _ = cr.load_all(root, und)
        for d_, right in ((1, 0), (-1, 1)):
            for h in cr.HS:
                a = cr._agg(d, "thr_pct", dte=0, m=0, right=right, h=h, q=1)
                out[(und, d_, h)] = (a["mean"], a["p50"], a["n"])
    return out, win


def _f(x, nd=3, sign=True):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def report(root: Path) -> str:
    from zr.ops.markdown import df_to_md

    df = pd.read_parquet(store(root) / "triggers.parquet")
    df["day"] = pd.to_datetime(df["day"]).dt.date
    counts = pd.read_csv(store(root) / "counts.csv")
    trunc = pd.read_csv(store(root) / "truncation.csv")
    thr, win = thresholds(root)
    lo_c, hi_c = calibration_window()
    rng_all = {u: (min(df[df.und == u]["day"]), max(df[df.und == u]["day"])) for u in UNDS}
    ndays = {(r.und, r.conv): int(r.days) for r in counts.itertuples()}
    B = "**全部为扣费前**。1 分钟 K、2 分钟 K 收盘确认、下一根（信号收盘时刻那根）1 分钟 K 开盘入场；随机基线 = 同日、会话窗口内、触发前后 30 分钟、同方向的 50 个随机分钟（排除任一触发 ±2 分钟），同样开盘入场。"
    L = ["# R-054：三条调研规则在 QQQ / SPY 正股层的复制", "",
         "`python -m zr.research.r054_index_stock_layer` → `--report`（本文）。预登记：RESEARCH_LOG R-054（运行前提交）。规则读自 `config/intrabar/survey_*.yaml`，一个数未调。", "",
         f"数据：长桥 1 分钟 K（bars_1m），QQQ {rng_all['QQQ'][0]}～{rng_all['QQQ'][1]}、SPY {rng_all['SPY'][0]}～{rng_all['SPY'][1]}；按 D-033 剔除 {lo_c}～{hi_c}。"
         f"交易日：QQQ {ndays[('QQQ', 'C')]}、SPY {ndays[('SPY', 'C')]}。主口径 C（含盘前盘后原始 K、不按日重置）；敏感性 D-027（RTH-only、按日重置）。", "", B, ""]
    # 1 触发数
    rows = []
    for und in UNDS:
        for conv in CONVS:
            for rule in RULES:
                s = df[(df.und == und) & (df.conv == conv) & (df.rule == rule)]
                c = counts[(counts.und == und) & (counts.conv == conv)].iloc[0]
                rows.append({"标的": und, "口径": conv, "规则": RULE_CN[rule], "交易日": int(c.days), "触发（可入场）": len(s),
                             "多 / 空": f"{int((s.dir == 1).sum())} / {int((s.dir == -1).sum())}", "每日": f"{len(s) / c.days:.2f}"})
    mu = None
    p = root / "research" / "owner_rules_oos" / "stock.parquet"
    if p.exists():
        m = pd.read_parquet(p)
        nd = m["day"].nunique()
        for rule in RULES:
            s = m[m.rule == rule]
            v1 = int((np.isfinite(s["ret60"]) & np.isfinite(s["rand_ret60"])).sum())
            rows.append({"标的": "MU（R-049）", "口径": "C（秒级）", "规则": RULE_CN[rule], "交易日": nd, "触发（可入场）": f"{len(s)}（1 分钟有效 {v1}）",
                         "多 / 空": f"{int((s.dir == 1).sum())} / {int((s.dir == -1).sum())}", "每日": f"{len(s) / nd:.2f}"})
        mu = True
    cd = counts.copy()
    L += ["## 1 触发数对照（量级 sanity check，不作为通过标准）", "",
          "表头：标的 QQQ / SPY，日期见上，口径 C 与 D-027，扣费前。MU 一行读自 R-049 落盘结果（`research/owner_rules_oos/stock.parquet`，388 个样本外交易日；任务书引用的 RSI2 3,506 是 1 分钟 horizon 的有效样本数）。", "",
          df_to_md(pd.DataFrame(rows)), "",
          "入场丢弃（信号收盘时刻没有 RTH 1 分钟 K 或已到 flat）：" + "；".join(f"{r.und} {r.conv} {int(r.entry_dropped)} / {int(r.trig)}" for r in cd.itertuples()), ""]
    if mu is None:
        L += ["（R-049 落盘结果不存在，MU 行缺。）", ""]
    # 2 主表
    def cells(conv, sample):
        out = []
        for und in UNDS:
            sub_u = sample(df[(df.und == und) & (df.conv == conv)], und)
            for rule in RULES:
                for d in (1, -1):
                    sub = sub_u[(sub_u.rule == rule) & (sub_u.dir == d)]
                    for h in HS:
                        st = cell_stats(sub, h)
                        out.append({"und": und, "rule": rule, "dir": d, "h": h, **st})
        return pd.DataFrame(out)

    def fam_sample(x, und):
        if und in win:
            a, b = win[und]
            return x[(x["day"] >= a) & (x["day"] <= b)]
        return x.iloc[0:0]

    main = cells("C", fam_sample)
    main["q"] = bh_q(main["p"].to_numpy())
    verdict = []
    for r in main.itertuples():
        tm = thr.get((r.und, r.dir, r.h))
        if not np.isfinite(r.excess) or r.days < MIN_DAYS:
            v = "不显著"
        elif r.q > BH_Q:
            v = "不显著"
        elif tm is None or not np.isfinite(tm[0]):
            v = "不可进入（无门槛数据）"
        elif r.excess > tm[0]:
            v = "可进入期权层"
        else:
            v = "不可进入"
        verdict.append(v)
    main["verdict"] = verdict
    tab = []
    for r in main.itertuples():
        tm = thr.get((r.und, r.dir, r.h))
        tab.append({"标的": r.und, "规则": RULE_CN[r.rule], "方向": "多" if r.dir == 1 else "空", "h（分钟）": r.h,
                    "触发 / 交易日": f"{r.n} / {r.days}", "超额（%）": _f(r.excess, 4), "触发均值（%）": _f(r.ret, 4), "随机均值（%）": _f(r.rand, 4),
                    "t": _f(r.t, 2), "p": _f(r.p, 4, False), "q": _f(r.q, 4, False),
                    "门槛均值（%）": _f(tm[0], 3, False) if tm else "无", "门槛中位（%）": _f(tm[1], 3, False) if tm else "无", "判定": r.verdict})
    wq, ws = win.get("QQQ"), win.get("SPY")
    passed = main[main.verdict == "可进入期权层"]
    L += ["## 2 主表（口径 C，72 格 BH 家族，q = 0.10）", "",
          f"表头：QQQ {wq[0]}～{wq[1]}、SPY {ws[0]}～{ws[1]}（各自 COST_REPORT 期权成本数据窗口；均剔除 D-033 窗口）；口径 C；扣费前；样本 = 触发数 / 交易日数。"
          "超额 = 触发收益 − 其 50 个随机分钟收益均值，先按日取均值再取均值；p = 日均值单侧 t（H0 ≤ 0）；q = 72 格 BH。"
          "门槛 = COST_REPORT 同标的 0DTE 平值 1 张、同 h 的回本门槛（全部起始分钟的均值 / 中位；多对看涨、空对看跌），只有 h = 5 / 15 / 30 有。"
          "判定：超额 > 门槛均值且 q ≤ 0.10 → 可进入期权层；q > 0.10 → 不显著；其余 → 不可进入。", "", B, "",
          df_to_md(pd.DataFrame(tab)), "",
          f"**可进入期权层：{len(passed)} 格**" + ("。" if passed.empty else "：" + "、".join(f"{r.und} {RULE_CN[r.rule]} {'多' if r.dir == 1 else '空'} {r.h} 分钟" for r in passed.itertuples())), ""]
    sg = main[(main.q <= BH_Q) & np.isfinite(main.excess)]
    nb = int((sg["ret"].abs() < sg["rand"].abs()).sum())
    L += [f"**基线结构提示**：q ≤ 0.10 的 {len(sg)} 格中，{nb} 格的触发自身收益均值的绝对值小于随机均值的绝对值——超额主要来自随机基线为负，而不是触发后正股朝方向走。"
          "RSI(2) / 均线回踩是回调后入场，触发前 30 分钟内的随机分钟其前向窗口包含这段回调本身，基线因此系统性偏负（R-049 同一设计，同样适用）。"
          f"这些格子触发自身收益均值最大 {_f(sg['ret'].max(), 4)}%，均低于对应门槛。", ""]
    sig5 = main[(main.h == 5)]
    L += ["假设检验（预登记原文：5 分钟超额为正，且 C 与 D-027 同号）的 5 分钟格：" + "；".join(
        f"{r.und} {RULE_CN[r.rule]} {'多' if r.dir == 1 else '空'} {_f(r.excess, 4)}%（q {_f(r.q, 3, False)}）" for r in sig5.itertuples()), ""]
    # SPY 2024-08 之后
    def after(x, und):
        if und == "SPY" and "SPY" in win:
            return x[x["day"] > win["SPY"][1]]
        return x.iloc[0:0]
    sa = cells("C", after)
    sa = sa[sa.und == "SPY"]
    if len(sa):
        dd = df[(df.und == "SPY") & (df.conv == "C") & (df["day"] > win["SPY"][1])]["day"]
        tab = [{"规则": RULE_CN[r.rule], "方向": "多" if r.dir == 1 else "空", "h": r.h, "触发 / 交易日": f"{r.n} / {r.days}",
                "超额（%）": _f(r.excess, 4), "t": _f(r.t, 2), "p": _f(r.p, 4, False)} for r in sa.itertuples()]
        L += [f"### SPY {dd.min()}～{dd.max()}（无对应期权成本数据，不与门槛比较、不进 BH 家族，只描述）", "", B, "", df_to_md(pd.DataFrame(tab)), ""]
    # 3 D-027 敏感性
    def all_s(x, und):
        return x
    c_all = cells("C", fam_sample)
    d_all = cells("D027", fam_sample)
    mg = c_all.merge(d_all, on=["und", "rule", "dir", "h"], suffixes=("_c", "_d"))
    both = mg[np.isfinite(mg.excess_c) & np.isfinite(mg.excess_d)]
    inc = both[np.sign(both.excess_c) != np.sign(both.excess_d)]
    nod = mg[~np.isfinite(mg.excess_d)]
    tab = [{"标的": r.und, "规则": RULE_CN[r.rule], "方向": "多" if r.dir == 1 else "空", "h": r.h, "C 超额（%）": _f(r.excess_c, 4),
            "D-027 超额（%）": _f(r.excess_d, 4), "D-027 触发 / 交易日": f"{r.n_d} / {r.days_d}"} for r in inc.itertuples()]
    L += ["## 3 D-027 敏感性（只看符号，描述，不进 BH）", "",
          f"表头：样本窗口同主表；D-027 = RTH-only 1 分钟 K、指标按日重置；扣费前。72 格中两个口径都有值的 {len(both)} 格，符号不一致 {len(inc)} 格（下表）；D-027 无样本 {len(nod)} 格"
          "（RSI(2) 规则要求 SMA200，按日重置的 RTH 2 分钟 K 一天只有 195 根，SMA200 永远为空 → D-027 下该规则 0 触发）。", "", B, "",
          df_to_md(pd.DataFrame(tab)) if tab else "无不一致格。", ""]
    # 4 年份
    tab = []
    cc = df[df.conv == "C"].copy()
    cc["year"] = [d.year for d in cc["day"]]
    for und in UNDS:
        for rule in RULES:
            for d in (1, -1):
                for y in sorted(cc["year"].unique()):
                    sub = cc[(cc.und == und) & (cc.rule == rule) & (cc.dir == d) & (cc.year == y)]
                    if sub.empty:
                        continue
                    row = {"标的": und, "规则": RULE_CN[rule], "方向": "多" if d == 1 else "空", "年": y, "触发 / 交易日": f"{len(sub)} / {sub['day'].nunique()}"}
                    for h in (5, 15, 30):
                        row[f"超额 {h} 分钟（%）"] = _f(cell_stats(sub, h)["excess"], 4)
                    tab.append(row)
    L += ["## 4 按年份拆分（口径 C，全部日期，描述）", "", "表头：QQQ / SPY 全部可用日期（剔除 D-033 窗口），口径 C，扣费前；超额同主表口径（按日均值）。", "", B, "", df_to_md(pd.DataFrame(tab)), ""]
    # 5 MFE、多空合并（描述）
    tab = []
    for und in UNDS:
        sub_u = fam_sample(df[(df.und == und) & (df.conv == "C")], und)
        for rule in RULES:
            sub = sub_u[sub_u.rule == rule]
            for h in HS:
                st = cell_stats(sub, h)
                tab.append({"标的": und, "规则": RULE_CN[rule], "h": h, "触发 / 交易日": f"{st['n']} / {st['days']}", "多空合并超额（%）": _f(st["excess"], 4),
                            "p": _f(st["p"], 4, False), "平均 MFE（%）": _f(st["mfe"], 4, False), "h 超出收盘丢弃": st["dropped_h"]})
    L += ["## 5 多空合并与 MFE（口径 C，主表窗口，描述，不进 BH）", "", B, "", df_to_md(pd.DataFrame(tab)), ""]
    # 截断
    L += ["## 6 无前视检验", "",
          f"真实数据截断检验（QQQ，随机 5 天 × 4 个时刻 × 2 口径：把 1 分钟 K 截到收盘时刻 ≤ t 后重算信号，与全量结果中 ≤ t 的信号逐条比较）：{int(trunc['same'].sum())} / {len(trunc)} 一致，"
          f"比较的信号共 {int(trunc['n_full'].sum())} 条。另有合成数据的单元测试（tests/test_r054_rules.py）。", ""]
    L += ["## 7 边界", "",
          f"- 标的 QQQ、SPY；1 分钟 K（长桥），{rng_all['QQQ'][0]}～{rng_all['QQQ'][1]}，剔除 D-033 校准窗口 {lo_c}～{hi_c}；末日 2026-10-21 数据只到盘中（之后的 horizon 丢弃计数）。",
          "- 2 分钟 K 收盘确认；入场 = 信号收盘时刻那根 1 分钟 K 的开盘价（不是逐秒价，也不含点差与费用）；收益与超额全部为**扣费前**正股收益。",
          "- 开盘区间用 RTH 1 分钟 K（R-049 在 MU 上用秒级平价价），突破用 2 分钟 K 收盘价；与 R-049 的 MU 结果不是同一实现，只能作量级对照。",
          "- 指标在 C 口径下全序列连续计算（R-049 每天只取此前 14 个日历日的 K 预热）。",
          "- 随机基线：触发前后 30 分钟、同方向、同会话窗口的 50 个随机分钟，排除当日任一触发（三条规则、两个方向）±2 分钟；池子不足 50 时全取。",
          f"- SPY 期权成本数据只覆盖 {ws[0]}～{ws[1]}：SPY 门槛对比与 BH 家族只用该窗口；之后的 SPY 结果单列、不比较。",
          "- 门槛是 COST_REPORT 的 0DTE 平值 1 张、全部起始分钟的均值；触发时刻的门槛分布未单独匹配。h = 1 / 2 / 10 分钟没有门槛数据。",
          "- 随机基线取触发前后 ±30 分钟，回调类规则的触发前窗口包含回调本身，基线偏负、超额偏高（见第 2 节提示）；预登记设计未改。",
          "- 本报告不涉及期权层回放；「可进入期权层」只表示值得做期权层检验，不是盈利结论。", ""]
    txt = "\n".join(L)
    OUT_MD.write_text(txt, encoding="utf-8")
    return txt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = get_settings().zr_data_root
    if a.report:
        report(root)
        print(f"写入 {OUT_MD}")
        return 0
    run(root, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
