"""参数扫描（R-006～R-009）：E1 变体 × 趋势定义 A–E、E2 15 分钟、E5 k、E9 抄底。

运行：uv run python -m zr.research.scan [--config config/scan.yaml]
输出：docs/research/SCAN_TABLE.md；ZR_DATA_ROOT/research/w2/scan_labels.parquet
每个变体都走同一条 triple-barrier 标签与日聚合 t 值，和 run.py 口径一致。
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from zr.config import get_settings
from zr.features import indicators as ind
from zr.features.bars import load_bars
from zr.features.features import compute_features
from zr.ops.markdown import df_to_md
from zr.research.events import _day
from zr.research.labels import clustered_t, rolling_sigma, triple_barrier
from zr.research.run import fee_adjusted_pnl

log = logging.getLogger("zr.research.scan")
ET = ZoneInfo("America/New_York")


# ---------- 趋势定义 A–E（返回 +1 上升 / −1 下降 / 0 无）----------


def trend_A(df, f, p):  # MACD 三条件（原 E1）
    up = (f["macd_dif_atr"] > f["macd_dea_atr"]) & (f["macd_dif_atr"] > 0) & (f["bb_mid_slope"] > 0)
    dn = (f["macd_dif_atr"] < f["macd_dea_atr"]) & (f["macd_dif_atr"] < 0) & (f["bb_mid_slope"] < 0)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def trend_B(df, f, p):  # 价格在 VWAP 上且 SMA20 > SMA60
    s20, s60 = ind.sma(df["close"], 20), ind.sma(df["close"], 60)
    up = (df["close"] > f["vwap"]) & (s20 > s60)
    dn = (df["close"] < f["vwap"]) & (s20 < s60)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def trend_C(df, f, p):  # 两个低点抬高且两个高点抬高（局部极值用 w 根窗口确认，信号滞后 w 根）
    w = p.get("swing_w", 5)
    hi, lo = df["high"], df["low"]
    # 中心 bar 为前后 w 根的最高/最低；确认需再等 w 根，因此 shift(w) 后才"可见"
    is_hi = (hi == hi.rolling(2 * w + 1, center=True).max()).shift(w).fillna(False).astype(bool)
    is_lo = (lo == lo.rolling(2 * w + 1, center=True).min()).shift(w).fillna(False).astype(bool)
    sh = hi.shift(w).where(is_hi)
    sl = lo.shift(w).where(is_lo)
    last_hi, last_lo = sh.ffill(), sl.ffill()
    prev_hi = sh.dropna().shift(1).reindex(df.index).ffill()
    prev_lo = sl.dropna().shift(1).reindex(df.index).ffill()
    day = _day(df.index)
    same_day = day == day.shift(2 * w)  # 极值须在当日形成
    up = (last_hi > prev_hi) & (last_lo > prev_lo) & same_day
    dn = (last_hi < prev_hi) & (last_lo < prev_lo) & same_day
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def trend_D(df, f, p):  # ADX(14) > 25 且 +DI > −DI
    a = ind.adx(df["high"], df["low"], df["close"], 14)
    thr = p.get("adx_min", 25)
    up = (a["adx"] > thr) & (a["plus_di"] > a["minus_di"])
    dn = (a["adx"] > thr) & (a["plus_di"] < a["minus_di"])
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


def trend_E(df, f, p):  # A + 120 根均线过滤
    t = trend_A(df, f, p)
    ma = ind.sma(df["close"], 120)
    up = (t == 1) & (df["close"] > ma)
    dn = (t == -1) & (df["close"] < ma)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.index)


TRENDS = {"A": trend_A, "B": trend_B, "C": trend_C, "D": trend_D, "E": trend_E}


# ---------- 变体检测器 ----------


def e1_variant(df, f, p) -> pd.Series:
    """通道回踩：趋势 × ε × 确认根数。确认 1 根 = 触及后下一根收盘仍在中轨之上才触发。"""
    eps, confirm = p["eps"], p.get("confirm", 0)
    trend = TRENDS[p["trend"]](df, f, p)
    mid = f["bb_mid"]
    touch_lo = (df["low"] - mid).abs() / mid <= eps
    touch_hi = (df["high"] - mid).abs() / mid <= eps
    long = (trend == 1) & touch_lo & (df["close"] > mid)
    short = (trend == -1) & touch_hi & (df["close"] < mid)
    if confirm == 1:
        long = long.shift(1, fill_value=False) & (df["close"] > mid) & (trend == 1)
        short = short.shift(1, fill_value=False) & (df["close"] < mid) & (trend == -1)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def e2_variant(df, f, p) -> pd.Series:
    from zr.research.events import e2_opening_range_breakout

    return e2_opening_range_breakout(df, f, {"range_min": p["range_min"]})


def e5_variant(df, f, p) -> pd.Series:
    from zr.research.events import e5_cusum

    return e5_cusum(df, f, {"k": p["k"], "sigma_window": p.get("sigma_window", 60)})


def e9_dip_buy(df, f, p) -> pd.Series:
    """BinHV45 式抄底：收盘跌穿下轨、实体 < 带宽×ratio、相对前 N 根高点回撤 > thr。

    做空对称用上轨。
    """
    n, ratio, thr = p.get("lookback", 20), p["body_ratio"], p["drawdown"]
    band = (f["bb_width"] * f["bb_mid"]).replace(0, np.nan)  # 带宽绝对值
    lower = f["bb_mid"] - band / 2
    upper = f["bb_mid"] + band / 2
    body = (df["close"] - df["open"]).abs()
    small_body = body < band * ratio
    hi_n = df["high"].rolling(n, min_periods=n).max().shift(1)
    lo_n = df["low"].rolling(n, min_periods=n).min().shift(1)
    long = (df["close"] < lower) & small_body & (df["close"] / hi_n - 1 < -thr)
    short = (df["close"] > upper) & small_body & (df["close"] / lo_n - 1 > thr)
    # 同一日内连续满足只取首次
    long = long & ~long.shift(1, fill_value=False)
    short = short & ~short.shift(1, fill_value=False)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def build_variants(cfg: dict) -> list[dict]:
    v = []
    e1 = cfg["E1"]
    for eps, confirm, trend in product(e1["eps"], e1["confirm"], e1["trend"]):
        v.append(
            {
                "detector": "E1",
                "variant": f"eps={eps:.4%}|confirm={confirm}|trend={trend}",
                "fn": e1_variant,
                "p": {"eps": eps, "confirm": confirm, "trend": trend, **e1.get("extra", {})},
            }
        )
    for rm in cfg["E2"]["range_min"]:
        v.append(
            {
                "detector": "E2",
                "variant": f"range_min={rm}",
                "fn": e2_variant,
                "p": {"range_min": rm},
            }
        )
    for k in cfg["E5"]["k"]:
        v.append({"detector": "E5", "variant": f"k={k}", "fn": e5_variant, "p": {"k": k}})
    e9 = cfg["E9"]
    for br, ddn in product(e9["body_ratio"], e9["drawdown"]):
        v.append(
            {
                "detector": "E9",
                "variant": f"body_ratio={br}|drawdown={ddn:.1%}",
                "fn": e9_dip_buy,
                "p": {"body_ratio": br, "drawdown": ddn, "lookback": e9.get("lookback", 20)},
            }
        )
    return v


def scan_symbol(symbol: str, cfg: dict, root: Path) -> pd.DataFrame:
    und = symbol.split(".")[0]
    df = load_bars(
        root, und, dt.date.fromisoformat(str(cfg["start"])), dt.date.fromisoformat(str(cfg["end"]))
    )
    f = compute_features(df)
    sigma = rolling_sigma(df["close"], cfg["labels"]["sigma_window"])
    n_days = df.index.tz_convert(ET).normalize().nunique()
    rows = []
    for var in build_variants(cfg["variants"]):
        sig = var["fn"](df, f, var["p"])
        hit = sig[sig != 0]
        events = pd.DataFrame({"ts": hit.index, "direction": hit.to_numpy()})
        for h in cfg["labels"]["horizons"]:
            lab = triple_barrier(df["close"], events, h, cfg["labels"]["k"], sigma).dropna(
                subset=["ret_signed"]
            )
            if lab.empty:
                rows.append(
                    {
                        "detector": var["detector"],
                        "variant": var["variant"],
                        "symbol": symbol,
                        "H": h,
                        "n": 0,
                        "per_day": 0.0,
                    }
                )
                continue
            spot = df["close"].reindex(lab["ts"]).to_numpy()
            pnl = fee_adjusted_pnl(
                lab["ret_signed"], pd.Series(spot, index=lab.index), und, cfg["fee_model"]
            )
            days = lab["ts"].dt.tz_convert(ET).dt.date
            t, nd = clustered_t(lab["ret_signed"], days)
            rows.append(
                {
                    "detector": var["detector"],
                    "variant": var["variant"],
                    "symbol": symbol,
                    "H": h,
                    "n": len(lab),
                    "per_day": round(len(lab) / n_days, 2),
                    "hit_rate": round(float((lab["ret_signed"] > 0).mean()), 3),
                    "mean_ret_bp": round(float(lab["ret_signed"].mean() * 1e4), 2),
                    "t_daily": round(t, 2),
                    "days": nd,
                    "mean_pnl_fee_adj_usd": round(float(pnl.mean()), 2),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/scan.yaml")
    parser.add_argument("--symbols", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = get_settings().zr_data_root
    symbols = args.symbols.split(",") if args.symbols else cfg["symbols"]
    t0 = time.monotonic()
    tabs = []
    for s in symbols:
        t1 = time.monotonic()
        tabs.append(scan_symbol(s, cfg, root))
        log.info("%s 扫描完成 %.0fs", s, time.monotonic() - t1)
    tab = pd.concat(tabs, ignore_index=True)
    store = root / "research" / "w2"
    store.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(store / "scan_table.parquet", index=False)
    n_var = tab.groupby(["detector", "variant"]).ngroups
    n_trials = len(tab)
    bonf_t = 4.0
    top = tab[(tab["n"] >= 200)].sort_values("t_daily", ascending=False).head(15)
    e1 = tab[(tab["detector"] == "E1") & (tab["H"] == 5)].copy()
    e1_ok = e1[(e1["per_day"] >= 3) & (e1["per_day"] <= 15)].sort_values(
        ["symbol", "t_daily"], ascending=[True, False]
    )
    lines = [
        "# 参数扫描表（R-006～R-009）",
        "",
        f"生成：{dt.datetime.now(dt.UTC).isoformat(timespec='seconds')} UTC · "
        f"`{args.config}` · 耗时 {time.monotonic() - t0:.0f}s",
        f"变体 {n_var} 个 × 标的 {len(symbols)} × H {cfg['labels']['horizons']} "
        f"= {n_trials} 次试验。"
        f"Bonferroni（{n_trials}）下 t 阈值 ≈ {bonf_t}；表中 t_daily 为日聚合 t。"
        "扣费后 P&L 口径同 EVENT_LABEL_TABLE。",
        "",
        "## E1：触发 3–15 次/日的组合（H=5），按 t 排序",
        "",
        df_to_md(
            e1_ok[
                [
                    "symbol",
                    "variant",
                    "per_day",
                    "hit_rate",
                    "mean_ret_bp",
                    "t_daily",
                    "mean_pnl_fee_adj_usd",
                ]
            ]
        ),
        "",
        "## 全部试验中 t_daily 最高的 15 个（样本 ≥ 200）",
        "",
        df_to_md(top),
        "",
        "## 全表",
        "",
        df_to_md(tab),
        "",
    ]
    out = Path(cfg["out_dir"]) / "SCAN_TABLE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("写入 %s（%d 次试验，%.0fs）", out, n_trials, time.monotonic() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
