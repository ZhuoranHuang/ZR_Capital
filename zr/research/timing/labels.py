# ruff: noqa: E501  （中文文档字符串与注释较长）
"""分钟网格、反转信号、L1–L4 标签与 L4 出场引擎（夜间择时研究 R-055 归档）。

分钟网格：每个交易日 09:30 至收盘逐分钟（缺失分钟按前收补 O=H=L=C、量 0），行号连续。
决策行 t（标签 09:32–15:45），入场行 e = t + 1（开盘价 ± 1 tick）。
L4 出场（负责人 2026-11-11 替换版，见 PREREG）：在「有利方向相对收益」坐标里统一处理多空。
原因码：1 止盈、2 止损、3 反转信号、4 兜底时间、5 收盘（不足 H）、6 回吐。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zr.research.timing.data import CAPS, GRID, HS, MS, TICK, et_parts, ext, rth, split_of

TP, SL, REV, TIME, EOD, GB = 1, 2, 3, 4, 5, 6


# ---------------- 分钟网格 ----------------


def minute_grid(root: Path, und: str, end=None, allow_holdout: bool = False) -> pd.DataFrame:
    from zr.data.calendar import session_close

    b = rth(root, und, end, allow_holdout)
    d, m = et_parts(b.index)
    parts = []
    for day in sorted(set(d)):
        cl = session_close(day)
        n = (cl.hour * 60 + cl.minute) - 570
        x = b[d == day]
        mm = m[d == day] - 570
        f = pd.DataFrame(
            index=np.arange(n), columns=["open", "high", "low", "close", "volume"], dtype=float
        )
        ok = (mm >= 0) & (mm < n)
        f.iloc[mm[ok]] = x[["open", "high", "low", "close", "volume"]].to_numpy(float)[ok]
        f["filled"] = f["close"].isna()
        f["close"] = f["close"].ffill().bfill()
        for c in ("open", "high", "low"):
            f[c] = f[c].fillna(f["close"])
        f["volume"] = f["volume"].fillna(0.0)
        f["day"], f["mod"] = day, np.arange(n) + 570
        parts.append(f)
    g = pd.concat(parts, ignore_index=True)
    base = pd.DatetimeIndex(pd.to_datetime(g["day"].astype(str))).tz_localize("America/New_York")
    g["ts"] = (base + pd.to_timedelta(g["mod"].to_numpy(), unit="m")).tz_convert("UTC")
    first = np.r_[True, g["day"].to_numpy()[1:] != g["day"].to_numpy()[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
    cnt = g.groupby("day")["mod"].transform("size").to_numpy()
    g["day_end"] = start + cnt  # 当日最后一行 + 1（行号）
    g["split"] = split_of(g["day"].to_numpy())
    return g


# ---------------- 反转信号（周期 P 的 K，口径 C） ----------------


def reversal_rows(g: pd.DataFrame, e1: pd.DataFrame, period: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (next_long, next_short)：对每一行 k，第一个「收盘时刻严格晚于 k 行开盘」的反转 K 收盘
    所在分钟行号 r（在 r 行开盘成交）；不存在 / 不在当日 → 大数。"""
    from zr.features.bars import resample_bars

    b = e1 if period == 1 else resample_bars(e1, f"{period}min")
    b = b[b["close"].notna()]
    c = b["close"]
    mid = c.rolling(20, min_periods=20).mean()
    dif = c.ewm(span=6, adjust=False).mean() - c.ewm(span=13, adjust=False).mean()
    hist = dif - dif.ewm(span=6, adjust=False).mean()
    h0, h1 = hist.to_numpy(), hist.shift(1).to_numpy()
    c0, c1, c2 = c.to_numpy(), c.shift(1).to_numpy(), c.shift(2).to_numpy()
    m0, m1 = mid.to_numpy(), mid.shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        rl = ((h0 < 0) & (h1 >= 0)) | ((c0 < c1) & (c1 < c2)) | ((c0 < m0) & (c1 >= m1))
        rs = ((h0 > 0) & (h1 <= 0)) | ((c0 > c1) & (c1 > c2)) | ((c0 > m0) & (c1 <= m1))
    close_t = b.index + pd.Timedelta(minutes=period)
    key = pd.Series(np.arange(len(g)), index=pd.DatetimeIndex(g["ts"]))
    rows = key.reindex(close_t).to_numpy()  # 收盘时刻落在 RTH 网格的哪一行（开盘时刻 = 收盘时刻）
    big = np.iinfo(np.int64).max // 4
    out = []
    for flag in (rl, rs):
        f = np.zeros(len(g), bool)
        ok = flag & np.isfinite(rows)
        f[rows[ok].astype(int)] = True
        idx = np.where(f, np.arange(len(g)), big)
        nxt = np.minimum.accumulate(idx[::-1])[::-1]  # ≥ k 的第一个
        nxt = np.r_[nxt[1:], big]  # > k
        nxt = np.where(nxt < g["day_end"].to_numpy(), nxt, big)
        out.append(nxt.astype(np.int64))
    return out[0], out[1]


# ---------------- L4 出场引擎 ----------------


def l4(
    G: dict,
    e: np.ndarray,
    d,
    m,
    H: int,
    kind: str,
    rev_next=None,
    entry_px=None,
    slip: float = TICK,
    giveback: float = 0.4,
):
    """G：o/h/l/c/day_end 数组（网格行）。e：入场行；d：±1；m：相对幅度（标量或数组）。
    kind ∈ a/b/c/d。返回 dict(entry, exit, ret_pct（毛，含滑点）, hold_min, reason, exit_row)。"""
    e = np.asarray(e, np.int64)
    n = len(e)
    d = np.broadcast_to(np.asarray(d, float), (n,)).copy()
    m = np.broadcast_to(np.asarray(m, float), (n,)).copy()
    o, hi, lo, c, dend = G["o"], G["h"], G["l"], G["c"], G["day_end"]
    last = dend[e] - 1
    ent = (o[e] + d * slip) if entry_px is None else np.asarray(entry_px, float)
    if kind == "d":
        xr = e + H
        eod = xr > last
        xr = np.where(eod, last, xr)
        raw = np.where(eod, c[xr], o[xr])
        reason = np.where(eod, EOD, TIME)
        t_exit = np.where(eod, xr + 1, xr)
        return _pack(ent, raw - d * slip, d, t_exit - e, reason, xr)
    cap = e + H - 1
    eod = cap > last
    cap = np.where(eod, last, cap)
    K = max(H - 1, 1)
    off = np.arange(1, K + 1)
    rows = e[:, None] + off[None, :]
    valid = rows <= cap[:, None]
    rr = np.minimum(rows, len(o) - 1)
    oh, hh, ll = o[rr], hi[rr], lo[rr]
    E = ent[:, None]
    D = d[:, None]
    fav_hi = np.where(D > 0, hh / E - 1, 1 - ll / E)
    adv_lo = np.where(D > 0, ll / E - 1, 1 - hh / E)
    op = np.where(D > 0, oh / E - 1, 1 - oh / E)
    M = m[:, None]
    big = 10**9
    # 下行触发水位：止损 −m；回吐（c）max(gb, −m)
    if kind == "c":
        prior = np.maximum.accumulate(np.where(valid, fav_hi, -np.inf), axis=1)
        prior = np.c_[np.full(n, -np.inf), prior[:, :-1]]  # 此前各根（不含当根）的最大有利
        gb = np.where(prior > 0, (1 - giveback) * prior, -np.inf)
        lvl_dn = np.maximum(gb, -M)
        dn_reason = np.where(gb > -M, GB, SL)
    else:
        lvl_dn = np.broadcast_to(-M, fav_hi.shape)
        dn_reason = np.full(fav_hi.shape, SL)
    hit_dn = valid & (adv_lo <= lvl_dn)
    hit_up = valid & (fav_hi >= M) if kind in ("a", "b") else np.zeros_like(valid)
    any_ = hit_dn | hit_up
    j = np.where(any_.any(1), any_.argmax(1), big)
    jj = np.minimum(j, K - 1)
    ar = np.arange(n)
    dn_first = hit_dn[ar, jj]  # 同根两边都碰按止损
    o_j = op[ar, jj]
    lv = lvl_dn[ar, jj]
    rel_dn = np.where(o_j <= lv, o_j, lv)
    rel_up = np.where(o_j >= m, o_j, m)
    rel_px = np.where(dn_first, rel_dn, rel_up)
    rsn_px = np.where(dn_first, dn_reason[ar, jj], TP)
    trig_row = e + 1 + jj  # 触发根（行号）
    has_px = j < big
    # 反转信号（a）：r 行开盘成交，r ≤ 触发根 → 反转优先；r ≤ cap
    if kind == "a":
        r = np.asarray(rev_next)[e]
        has_rev = r <= cap
        rev_first = has_rev & (~has_px | (r <= trig_row))
    else:
        r = np.full(n, big)
        rev_first = np.zeros(n, bool)
    raw_px = ent * (1 + d * rel_px)
    exit_raw = np.where(rev_first, o[np.minimum(r, len(o) - 1)], np.where(has_px, raw_px, c[cap]))
    reason = np.where(rev_first, REV, np.where(has_px, rsn_px, np.where(eod, EOD, TIME)))
    x_row = np.where(rev_first, r, np.where(has_px, trig_row, cap))
    t_exit = np.where(rev_first, r, x_row + 1)  # 开盘成交 = 该行；收盘成交 = 下一行
    return _pack(ent, exit_raw - d * slip, d, t_exit - e, reason, x_row)


def _pack(ent, ex, d, hold, reason, xr):
    return {
        "entry": ent,
        "exit": ex,
        "ret": d * (ex - ent) / ent * 100,
        "hold": hold.astype(np.int16),
        "reason": np.asarray(reason, np.int8),
        "xrow": xr,
    }


def arrays(g: pd.DataFrame) -> dict:
    return {
        "o": g["open"].to_numpy(float),
        "h": g["high"].to_numpy(float),
        "l": g["low"].to_numpy(float),
        "c": g["close"].to_numpy(float),
        "day_end": g["day_end"].to_numpy(np.int64),
    }


def atr_m(g: pd.DataFrame, e1: pd.DataFrame) -> np.ndarray:
    """MU 的 ATR 档：m = 2 × ATR14（口径 C 1 分钟 K，已完成）÷ 收盘价，对齐到网格行（该行 K 收盘时可得）。"""
    from zr.features.indicators import atr

    a = atr(e1["high"], e1["low"], e1["close"], 14) / e1["close"]
    s = pd.Series(a.to_numpy(), index=e1.index).reindex(pd.DatetimeIndex(g["ts"]))
    return 2 * s.to_numpy()


# ---------------- 构建 ----------------


def build(
    root: Path, und: str, m_main: float, end=None, allow_holdout: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (标签表（每个决策分钟一行）, 分钟网格（含反转信号行号、MU 的 ATR 档））。"""
    g = minute_grid(root, und, end, allow_holdout)
    e1 = ext(root, und, end, allow_holdout)
    G = arrays(g)
    rev = {p: reversal_rows(g, e1, p) for p in (1, 2, 5)}
    for p, (a, b) in rev.items():
        g[f"rev{p}_long"], g[f"rev{p}_short"] = a, b
    mod = g["mod"].to_numpy()
    t = np.flatnonzero((mod >= GRID[0]) & (mod <= GRID[1]) & ~g["filled"].to_numpy())
    t = t[t + 1 < G["day_end"][t]]
    e = t + 1
    lab = pd.DataFrame(
        {"row": t, "day": g["day"].to_numpy()[t], "mod": mod[t], "split": g["split"].to_numpy()[t]}
    )
    o, c, dend = G["o"], G["c"], G["day_end"]
    # L1（主口径与入场点曲线）
    entries = {
        "e1": (e, o[e]),
        "c0": (e, c[t]),
        "e2": (t + 2, None),
        "e3": (t + 3, None),
        "e6": (t + 6, None),
    }
    for key, (er, px) in entries.items():
        ok_e = er < dend[t]
        er_c = np.minimum(er, len(o) - 1)
        p0 = o[er_c] if px is None else px
        for h in HS + (60,):
            x = er_c + h - 1  # 第 h 根 K 的收盘（c0 档按 e1 的时间轴）
            ok = ok_e & (x < dend[t])
            v = np.where(ok, c[np.minimum(x, len(c) - 1)] / p0 - 1, np.nan) * 100
            if key == "e1" or h in HS:
                lab[f"L1_{key}_{h}"] = v.astype(np.float32)
        if key == "e1":
            lab["L1_e1_close"] = ((c[dend[t] - 1] / o[e] - 1) * 100).astype(np.float32)
    # L2 / L3
    ms = list(MS)
    m_atr = atr_m(g, e1) if und == "MU" else None
    for H in CAPS:
        K = H - 1
        rows = e[:, None] + np.arange(1, K + 1)[None, :]
        valid = rows < dend[t][:, None]
        rr = np.minimum(rows, len(o) - 1)
        up = G["h"][rr] / o[e][:, None] - 1
        dn = G["l"][rr] / o[e][:, None] - 1
        for mk, mv in [(f"{x * 100:.2f}", x) for x in ms] + (
            [("atr", m_atr[t])] if m_atr is not None else []
        ):
            mv = np.broadcast_to(np.asarray(mv, float), (len(t),))[:, None]
            hu, hd = valid & (up >= mv), valid & (dn <= -mv)
            first_u = np.where(hu.any(1), hu.argmax(1), 999)
            first_d = np.where(hd.any(1), hd.argmax(1), 999)
            code = np.where(
                (first_u == 999) & (first_d == 999),
                0,
                np.where(first_u < first_d, 1, np.where(first_d < first_u, -1, 2)),
            )
            lab[f"L2_{mk}_{H}"] = code.astype(np.int8)
            lab[f"L2t_{mk}_{H}"] = (np.minimum(first_u, first_d) + 1).astype(np.int16)
        w = e[:, None] + np.arange(0, H)[None, :]
        vw = w < dend[t][:, None]
        ww = np.minimum(w, len(o) - 1)
        lab[f"L3_up_{H}"] = (np.where(vw, G["h"][ww], -np.inf).max(1) / o[e] - 1) * 100
        lab[f"L3_dn_{H}"] = (np.where(vw, G["l"][ww], np.inf).min(1) / o[e] - 1) * 100
    # L4 主口径：出场 (a) 兜底 15，反转周期 2 全部 m；周期 1、5 只做所选 m
    lab["entry_long"], lab["entry_short"] = o[e] + TICK, o[e] - TICK
    for p in (1, 2, 5):
        mlist = ms + (["atr"] if m_atr is not None else []) if p == 2 else [m_main]
        for mv in mlist:
            mk = "atr" if mv == "atr" else f"{mv * 100:.2f}"
            marr = m_atr[t] if mv == "atr" else mv
            for dn_, dv in (("long", 1), ("short", -1)):
                r = l4(G, e, dv, marr, 15, "a", rev_next=g[f"rev{p}_{dn_}"].to_numpy())
                lab[f"L4a15_p{p}_{mk}_{dn_}_x"] = r["exit"]
                lab[f"L4a15_p{p}_{mk}_{dn_}_ret"] = r["ret"].astype(np.float32)
                lab[f"L4a15_p{p}_{mk}_{dn_}_hold"] = r["hold"]
                lab[f"L4a15_p{p}_{mk}_{dn_}_rsn"] = r["reason"]
    # L4(a) 入场点曲线（所选 m、反转周期 2、兜底 15）
    mk = f"{m_main * 100:.2f}"
    for key in ("c0", "e2", "e3", "e6"):
        er = {"c0": e, "e2": t + 2, "e3": t + 3, "e6": t + 6}[key]
        ok = er < dend[t]
        er_c = np.where(ok, er, e)
        for dn_, dv in (("long", 1), ("short", -1)):
            px = (c[t] + dv * TICK) if key == "c0" else None
            r = l4(G, er_c, dv, m_main, 15, "a", rev_next=g[f"rev2_{dn_}"].to_numpy(), entry_px=px)
            lab[f"L4a15_{key}_{mk}_{dn_}_entry"] = np.where(ok, r["entry"], np.nan)
            lab[f"L4a15_{key}_{mk}_{dn_}_x"] = np.where(ok, r["exit"], np.nan)
            lab[f"L4a15_{key}_{mk}_{dn_}_ret"] = np.where(ok, r["ret"], np.nan).astype(np.float32)
    keep = [
        "day",
        "mod",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "filled",
        "day_end",
        "split",
    ] + [c_ for c_ in g.columns if c_.startswith("rev")]
    if m_atr is not None:
        g["m_atr"] = m_atr
        keep.append("m_atr")
    return lab, g[keep]
