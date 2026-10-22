"""审计（METHOD_V2 规格 10，R-026）。

1. 植入信号：随机 5% 事件的 MFE_10 加 +20 bp 并重算 ever_up_0.5_10，同时给出指示特征 planted=1；
   同一 walk-forward 训练后，模型应把这些事件排到前面（植入组的样本外 AUC 远高于 0.5）。这检验的是
   数据 → 标签 → 模型 → 评估这条管线能否传递一个已知信号，不是检验市场里有没有信号。
2. 标签打乱：每个月内随机打乱目标后重训，样本外 AUC 应在 0.5 附近。
3. 前视检查清单：列出每组特征 / 标签的无前视测试并实际运行，附 pytest 结果。
4. 手工复算 10 个事件：独立重读原始 parquet，用最直接的写法重算 MFE_10 与真实期权盈亏（持仓上限 30）
  ，
   与数据集中的值比较。
5. 1 / 2 / 5 分钟分周期：E1T 在 1、2、5 分钟 K 上的事件数 / 日与 10 分钟方向化收益。
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from zr.costs.fees import option_order_fees
from zr.features.bars import ET, load_bars, resample_bars
from zr.features.channel import channel_states
from zr.features.features import compute_features
from zr.research import outcomes as oc
from zr.research.model_v2 import design, uniqueness_weights, walk_forward
from zr.research.replay_rule import pullback_events

LOOKAHEAD_TESTS = [
    ("正股特征 23 列", "tests/test_features.py"),
    ("通道 A/B/C/D/F 与质量", "tests/test_channel.py"),
    ("结构上下文 / tf5 / 日历 / 跨资产 / 共振", "tests/test_features_v2.py"),
    ("期权特征（对齐 t+1 报价）", "tests/test_option_features.py"),
    ("指标（ATR / KDJ / ADX 等）", "tests/test_indicators.py"),
    ("标签 v2（窗口 t+1..t+H、同日）", "tests/test_labels_v2.py"),
    ("期权结果表（决策分钟 ask、bid 出场）", "tests/test_outcomes.py"),
    ("交易日历 / RTH", "tests/test_session.py"),
]


def planted_signal(
    ds: pd.DataFrame,
    und: str,
    frac: float = 0.05,
    bump: float = 0.002,
    seed: int = 0,
    last_months: int = 8,
) -> dict:
    d = ds[ds["ever_up_0.5_10"].notna() & ds["mfe_10"].notna()].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    planted = rng.random(len(d)) < frac
    s_h = d["sigma_1m"].to_numpy() * np.sqrt(10)
    mfe = d["mfe_10"].to_numpy() + np.where(planted, bump, 0.0)
    y = (mfe >= 0.5 * s_h).astype(int)
    X = design(d, und)
    X["planted"] = planted.astype(float)
    w = uniqueness_weights(d["day"], d["bar_minute"].to_numpy(), 10)
    keep = _last_months_mask(d["day"], last_months + 12)
    oos, _ = walk_forward(X[keep], pd.Series(y[keep]), w[keep], d.loc[keep, "day"])
    ok = oos.notna().to_numpy()
    yk, pk, plk = y[keep][ok], oos.to_numpy()[ok], planted[keep][ok]
    return {
        "n_oos": int(ok.sum()),
        "planted_share": round(float(plk.mean()), 4),
        "auc_all": round(float(roc_auc_score(yk, pk)), 4),
        "auc_planted_vs_rest": round(float(roc_auc_score(plk, pk)), 4),
        "mean_p_planted": round(float(pk[plk].mean()), 4),
        "mean_p_rest": round(float(pk[~plk].mean()), 4),
        "y_rate_planted": round(float(yk[plk].mean()), 4),
        "y_rate_rest": round(float(yk[~plk].mean()), 4),
    }


def _last_months_mask(day: pd.Series, n: int) -> np.ndarray:
    m = pd.to_datetime(day).dt.to_period("M")
    months = sorted(m.unique())[-n:]
    return m.isin(months).to_numpy()


def shuffled_labels(
    ds: pd.DataFrame, und: str, target: str = "ever_up_0.5_10", seed: int = 0, last_months: int = 8
) -> dict:
    d = ds[ds[target].notna()].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    month = pd.to_datetime(d["day"]).dt.to_period("M")
    y = d[target].astype(int).to_numpy().copy()
    for m in month.unique():
        idx = np.flatnonzero((month == m).to_numpy())
        y[idx] = y[rng.permutation(idx)]
    X = design(d, und)
    w = uniqueness_weights(d["day"], d["bar_minute"].to_numpy(), 10)
    keep = _last_months_mask(d["day"], last_months + 12)
    oos, folds = walk_forward(X[keep], pd.Series(y[keep]), w[keep], d.loc[keep, "day"])
    ok = oos.notna().to_numpy()
    return {
        "n_oos": int(ok.sum()),
        "auc": round(float(roc_auc_score(y[keep][ok], oos.to_numpy()[ok])), 4),
        "fold_auc_min": round(float(pd.DataFrame(folds)["auc"].min()), 4),
        "fold_auc_max": round(float(pd.DataFrame(folds)["auc"].max()), 4),
    }


def lookahead_checklist(repo: Path) -> pd.DataFrame:
    rows = []
    for name, path in LOOKAHEAD_TESTS:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", path], cwd=repo, capture_output=True, text=True
        )
        last = (r.stdout.strip().splitlines() or [""])[-1]
        rows.append({"组": name, "测试": path, "结果": last, "通过": r.returncode == 0})
    return pd.DataFrame(rows)


def _manual_option_pnl(
    root: Path,
    und: str,
    day: dt.date,
    minute: int,
    symbol: str,
    direction: int,
    close: np.ndarray,
    mid: np.ndarray,
    cap: int = 30,
) -> float:
    """独立实现：直接读 quotes.parquet，按规则逐分钟走一遍（不复用 outcomes.simulate）。"""
    q = pd.read_parquet(
        root / "options_1m" / f"underlying={und}" / f"date={day.isoformat()}" / "quotes.parquet"
    )
    q = q[(q["symbol"] == symbol) & (q["bid"] > 0) & (q["ask"] > 0)].copy()
    et = q["ts"].dt.tz_convert(ET)
    q["m"] = et.dt.hour * 60 + et.dt.minute - 570
    q = q.set_index("m")
    ask0 = float(q.loc[minute, "ask"])
    cash = -ask0 * 100 * 2 - float(option_order_fees(2, "Buy", ask0)["total"])
    left, half, last_bid = 2, False, None
    end = min(minute + cap, oc.FORCE_EXIT)
    for j in range(minute + 1, end + 1):
        if j in q.index:
            last_bid = float(q.loc[j, "bid"])
            if not half and last_bid >= ask0 * 1.2:
                cash += last_bid * 100 - float(option_order_fees(1, "Sell", last_bid)["total"])
                left, half = 1, True
        c, m = close[j], mid[j]
        if (
            np.isfinite(c)
            and np.isfinite(m)
            and ((direction == 1 and c < m) or (direction == -1 and c > m))
        ):
            break
    cash += last_bid * 100 * left - float(option_order_fees(left, "Sell", last_bid)["total"])
    return cash


def manual_recompute(
    root: Path, und: str, ds: pd.DataFrame, outcomes: pd.DataFrame, n: int = 10, seed: int = 7
) -> pd.DataFrame:
    stock = load_bars(root, und)
    mid_all = compute_features(stock)["bb_mid"]
    cand = ds[ds["net_cap30"].notna()].sample(
        n=min(n, int(ds["net_cap30"].notna().sum())), random_state=seed
    )
    rows = []
    sday = np.asarray(stock.index.tz_convert(ET).date)
    for _, r in cand.iterrows():
        day = r["day"]
        m = sday == day
        b = stock[m]
        idx = oc.minute_index(b.index)
        close = np.full(oc.MINUTES, np.nan)
        mid = np.full(oc.MINUTES, np.nan)
        close[idx] = b["close"].to_numpy()
        mid[idx] = mid_all[m].to_numpy()
        hi = np.full(oc.MINUTES, np.nan)
        lo = np.full(oc.MINUTES, np.nan)
        hi[idx], lo[idx] = b["high"].to_numpy(), b["low"].to_numpy()
        t = int(r["bar_minute"])
        c0 = close[t]
        w_hi, w_lo = hi[t + 1 : t + 11], lo[t + 1 : t + 11]
        mfe = (np.nanmax(w_hi) / c0 - 1) if r["direction"] == 1 else (1 - np.nanmin(w_lo) / c0)
        o = outcomes[
            (outcomes["day"] == day)
            & (outcomes["minute"] == t + 1)
            & (outcomes["direction"] == r["direction"])
            & (outcomes["cap"] == 30)
        ]
        sym = o["symbol"].iloc[0] if len(o) else None
        pnl = (
            _manual_option_pnl(root, und, day, t + 1, sym, int(r["direction"]), close, mid)
            if sym
            else np.nan
        )
        rows.append(
            {
                "ts_et": pd.Timestamp(r["ts"]).tz_convert(ET).strftime("%Y-%m-%d %H:%M"),
                "detector": r["detector"],
                "direction": int(r["direction"]),
                "symbol": sym,
                "mfe_10_dataset": round(float(r["mfe_10"]), 6),
                "mfe_10_manual": round(float(mfe), 6),
                "net_cap30_dataset": round(float(r["net_cap30"]), 2),
                "net_cap30_manual": round(float(pnl), 2),
            }
        )
    t = pd.DataFrame(rows)
    t["mfe_match"] = (t["mfe_10_dataset"] - t["mfe_10_manual"]).abs() < 1e-6
    t["pnl_match"] = (t["net_cap30_dataset"] - t["net_cap30_manual"]).abs() < 0.011
    return t


def timeframe_table(root: Path, und: str) -> pd.DataFrame:
    df1 = load_bars(root, und)
    rows = []
    for rule, k in (("1min", 1), ("2min", 2), ("5min", 5)):
        df = df1 if k == 1 else resample_bars(df1, rule)
        f = compute_features(df)
        ch = channel_states(df, f)
        sig = pullback_events(df, f, ch["chan_A"])
        day = np.asarray(df.index.tz_convert(ET).date)
        h = max(1, 10 // k)
        fwd = df["close"].shift(-h) / df["close"] - 1
        same = pd.Series(day, index=df.index).shift(-h).to_numpy() == day
        ev = sig[sig != 0]
        r = (fwd[ev.index] * ev).where(same[df.index.get_indexer(ev.index)])
        r = r.dropna()
        rows.append(
            {
                "timeframe": rule,
                "events": int(len(ev)),
                "per_day": round(len(ev) / len(set(day)), 2),
                "fwd_10min_mean_bp": round(float(r.mean() * 1e4), 3),
                "hit_rate": round(float((r > 0).mean()), 3),
                "n_with_fwd": int(len(r)),
            }
        )
    return pd.DataFrame(rows)
