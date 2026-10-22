"""状态层（METHOD_V3 第 2 节）：每根 K 一个标签。全部只用 ≤t 的数据。

指标（周期 P 的 K 线上）：BOLL(n, k)、MACD(fast, slow, signal)，柱 = DIF − DEA。
标签（优先级 IGNITION > CHANNEL > RANGE > OTHER），方向 dir ∈ {+1, −1, 0}：
- IGNITION：带宽连续 expand_bars 根扩张，且连续 ≥ outer_bars 根收盘 %b > pctb_hi（上行）/ <
  pctb_lo（下行），
  且这几根 MACD 柱同向（上行 > 0）逐根放大。点火之后只要收盘仍在 %b > pctb_hi（下行 < pctb_lo），
  状态延续。
- CHANNEL(N, M)：中轨斜率连续 ≥ N 根同向；近 M 根中 ≥ side_frac 收盘在中轨同侧；带宽未收缩
  （bw[t] ≥ bw_shrink_tol × bw[t−N]）。
- RANGE（D-026 打分制，取代三条件同时满足——AND 版覆盖率只有 4%，过严）：带宽、|中轨斜率|
  （÷ 中轨）、|MACD 柱|（÷ 收盘）各自在近 window_days 个交易日的百分位秩（0–1，越低越「不活跃」），
  按 weights 加权求和，分数 ≤ score_threshold 即 RANGE。窗口的 bar 数由数据的实际 bar/日折算
  （不写死 390，兼容含盘前盘后的连续 K 线）。阈值固定为一个常数，在 2024 年数据上标定后用于全部年份
  （不逐年重新标定），不构成前视。
- OTHER：其余。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zr.features import indicators as ind
from zr.features.bars import ET

CONFIG = Path(__file__).resolve().parents[2] / "config" / "method_v3.yaml"
OTHER, IGNITION, CHANNEL, RANGE = 0, 1, 2, 3
STATE_NAMES = {OTHER: "OTHER", IGNITION: "IGNITION", CHANNEL: "CHANNEL", RANGE: "RANGE"}


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _consec(flag: np.ndarray) -> np.ndarray:
    """截至每根（含）连续 True 的根数。"""
    out = np.zeros(len(flag), dtype=int)
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        out[i] = run
    return out


def indicators(
    df: pd.DataFrame,
    cfg: dict,
    macd_key: str = "macd",
    daily_reset: bool = True,
    warmup: str = "nan",
) -> pd.DataFrame:
    """周期 P 的 K 线 → BOLL / MACD / 衍生量。

    daily_reset=True（默认，D-027 底座）：BOLL/MACD 每个交易日从头算，df 为 RTH-only bars。
    daily_reset=False + 含盘前盘后 bars = 连续全时段口径。2026-10-22 复核（V4_CHECKPOINTS.md）：
    负责人的 3 个截图时刻在 App 显示精度（3 位小数）下两种口径都 15/15 精确相等，**截图区分不了**；
    INDICATOR_PARITY.md 里「含盘前盘后误差 44%」是本地 2 分钟分箱锚在 04:01 的 bug，不是 App 行为。
    只有「RTH-only 跨日连续」被截图排除（误差 2.38%）。

    warmup：BOLL 不足 n 根时的处理。"nan"（不足 n 根为空值）或 "partial"（按已有根数算）。
    MACD 的 EMA 从第一根就有值，不受影响。App 的实际处理未核实，见 docs/research/V4_CHECKPOINTS.md。
    """
    if warmup not in ("nan", "partial"):
        raise ValueError(f"warmup 只能是 nan / partial：{warmup}")
    mp = 1 if warmup == "partial" else None
    m = cfg[macd_key]
    boll_n, boll_k = cfg["boll"]["n"], cfg["boll"]["k"]

    def _one_day(g: pd.DataFrame) -> pd.DataFrame:
        b = ind.bollinger(g["close"], boll_n, boll_k, min_periods=mp)
        mc = ind.macd(g["close"], m["fast"], m["slow"], m["signal"])
        return pd.concat([b, mc], axis=1)

    day = np.asarray(df.index.tz_convert(ET).date)
    if daily_reset:
        parts = [_one_day(g) for _, g in df.groupby(day)]
        raw = pd.concat(parts).sort_index()
    else:
        raw = _one_day(df)
    out = pd.DataFrame(index=df.index)
    out["mid"], out["upper"], out["lower"] = raw["mid"], raw["upper"], raw["lower"]
    out["hw"] = (raw["upper"] - raw["lower"]) / 2
    out["bw"] = raw["width"]
    out["pctb"] = raw["pct_b"]
    # 斜率按日分组求 diff：daily_reset 下每日第一根没有同日前值，NaN（不跨日比较）
    out["slope"] = (
        raw.groupby(day)["mid"].diff() / raw["mid"]
        if daily_reset
        else (raw["mid"].diff() / raw["mid"])
    )
    out["dif"], out["dea"] = raw["dif"], raw["dea"]
    out["hist"] = out["dif"] - out["dea"]
    return out


def bars_per_day(df: pd.DataFrame) -> float:
    """数据里每个交易日的实际 bar 数（中位数），供「近 N 个交易日」窗口折算成 bar 数用。

    不写死 390（RTH 分钟数）：盘前盘后连续 K 线每日 bar 数更多，用这个函数让窗口口径自动适配。
    """
    day = np.asarray(df.index.tz_convert(ET).date)
    return float(pd.Series(day).value_counts().median())


def range_score(x: pd.DataFrame, df: pd.DataFrame, rg: dict, period: int) -> pd.Series:
    """RANGE 打分（D-026）：三项百分位秩的加权和，越低越「不活跃」。只用 ≤t 的滚动窗口。"""
    w = max(1, round(rg["window_days"] * bars_per_day(df)))  # bars_per_day 已按周期 P 计数（D-031）
    pr = lambda s: s.rolling(w, min_periods=max(20, w // 4)).rank(pct=True)  # noqa: E731
    wt = rg["weights"]
    return (
        wt["bw"] * pr(x["bw"])
        + wt["slope"] * pr(x["slope"].abs())
        + wt["hist"] * pr((x["hist"] / df["close"]).abs())
    )


def _channel(x: pd.DataFrame, close: np.ndarray, n: int, m: int, frac: float, tol: float):
    s = x["slope"].to_numpy()
    up_run, dn_run = _consec(s > 0), _consec(s < 0)
    above = pd.Series((close > x["mid"].to_numpy()).astype(float), index=x.index)
    below = pd.Series((close < x["mid"].to_numpy()).astype(float), index=x.index)
    fa = above.rolling(m, min_periods=m).mean().to_numpy()
    fb = below.rolling(m, min_periods=m).mean().to_numpy()
    bw = x["bw"].to_numpy()
    bw_n = np.r_[np.full(n, np.nan), bw[:-n]] if len(bw) > n else np.full(len(bw), np.nan)
    with np.errstate(invalid="ignore"):
        keep = bw >= tol * bw_n
        up = (up_run >= n) & (fa >= frac) & keep
        dn = (dn_run >= n) & (fb >= frac) & keep
    return up, dn


def regimes(df: pd.DataFrame, x: pd.DataFrame, cfg: dict, period: int) -> pd.DataFrame:
    """返回 state / dir（默认通道参数）以及每个 (N, M) 的通道方向 chan_N_M ∈ {+1, −1, 0}。"""
    rc = cfg["regime"]
    ig, ch, rg = rc["ignition"], rc["channel"], rc["range"]
    close = df["close"].to_numpy()
    pctb, hist, bw = x["pctb"].to_numpy(), x["hist"].to_numpy(), x["bw"].to_numpy()
    n = len(df)
    with np.errstate(invalid="ignore"):
        expand = _consec(np.r_[False, bw[1:] > bw[:-1]]) >= ig["expand_bars"]
        hi_run = _consec(pctb > ig["pctb_hi"])
        lo_run = _consec(pctb < ig["pctb_lo"])
        h_up = _consec(np.r_[False, (hist[1:] > hist[:-1]) & (hist[1:] > 0)])
        h_dn = _consec(np.r_[False, (hist[1:] < hist[:-1]) & (hist[1:] < 0)])
    k = ig["outer_bars"]
    start_up = expand & (hi_run >= k) & (h_up >= k)
    start_dn = expand & (lo_run >= k) & (h_dn >= k)
    ign = np.zeros(n, dtype=int)
    for i in range(n):  # 点火后沿外侧区间延续
        if start_up[i] or (i and ign[i - 1] == 1 and pctb[i] > ig["pctb_hi"]):
            ign[i] = 1
        elif start_dn[i] or (i and ign[i - 1] == -1 and pctb[i] < ig["pctb_lo"]):
            ign[i] = -1
    out = pd.DataFrame(index=df.index)
    for cn in ch["slope_n"]:
        for cm in ch["side_m"]:
            up, dn = _channel(x, close, cn, cm, ch["side_frac"], ch["bw_shrink_tol"])
            out[f"chan_{cn}_{cm}"] = np.where(up, 1, np.where(dn, -1, 0))
    score = range_score(x, df, rg, period)
    out["range_score"] = score
    rng = (score <= rg["score_threshold"]).to_numpy()
    dn_, dm_ = ch["default"]
    cdef = out[f"chan_{dn_}_{dm_}"].to_numpy()
    state = np.full(n, OTHER)
    sdir = np.zeros(n, dtype=int)
    state[rng] = RANGE
    state[cdef != 0], sdir[cdef != 0] = CHANNEL, cdef[cdef != 0]
    state[ign != 0], sdir[ign != 0] = IGNITION, ign[ign != 0]
    out["state"], out["dir"] = state, sdir
    out["ign"] = ign
    return out


def state_features(df: pd.DataFrame, x: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """进模型的状态特征（第 8 节）：状态、方向、持续根数、带宽扩张率、外轨连续根数、柱及变化、回踩
    %b 与带半宽比。
    """
    st = reg["state"].to_numpy() * 10 + reg["dir"].to_numpy()
    dur = np.ones(len(st), dtype=int)
    for i in range(1, len(st)):
        dur[i] = dur[i - 1] + 1 if st[i] == st[i - 1] else 1
    pctb = x["pctb"].to_numpy()
    above = _consec(pctb > 1)
    below = _consec(pctb < 0)
    lowb = (df["low"] - x["lower"]) / (2 * x["hw"]).replace(0, np.nan)
    highb = (df["high"] - x["lower"]) / (2 * x["hw"]).replace(0, np.nan)
    return pd.DataFrame(
        {
            "rg_state": reg["state"].to_numpy(),
            "rg_dir": reg["dir"].to_numpy(),
            "rg_duration": dur,
            "rg_bw_expand": x["bw"].pct_change().to_numpy(),
            "rg_outer_run": np.where(above > 0, above, -below),
            "rg_hist": (x["hist"] / df["close"]).to_numpy(),
            "rg_hist_chg": (x["hist"].diff() / df["close"]).to_numpy(),
            "rg_low_pctb": lowb.to_numpy(),
            "rg_high_pctb": highb.to_numpy(),
            "rg_low_mid_hw": ((df["low"] - x["mid"]) / x["hw"].replace(0, np.nan)).to_numpy(),
            "rg_high_mid_hw": ((df["high"] - x["mid"]) / x["hw"].replace(0, np.nan)).to_numpy(),
        },
        index=df.index,
    )


def bar_close_minute(index: pd.DatetimeIndex, period: int) -> np.ndarray:
    """K 线收盘时刻的分钟索引（09:30 = 0）；周期 P 的 bar 标签 t 在 t + P 收盘。"""
    et = index.tz_convert(ET)
    return (et.hour * 60 + et.minute - 570).to_numpy() + period
