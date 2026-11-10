# ruff: noqa: E501  （中文文档字符串与注释较长）
"""分钟级择时研究工具（zr/research/timing，R-055 归档）的最小测试：截断、时点断言、L4 与 zr.sim.stock 对账、
统计函数已知答案、L-002（日内排序偏差）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import zr.sim.stock as zs
from zr.costs.fees import stock_order_fees
from zr.research import leakcheck as lc
from zr.research.timing import features as F
from zr.research.timing import stats as S
from zr.research.timing.data import ACCT, net_pct, round_trip_fees
from zr.research.timing.events import boll_macd_event
from zr.research.timing.labels import EOD, GB, REV, SL, TIME, TP, l4
from zr.research.timing.trend import noise_signal, random_dirs
from zr.sim.engine import Account

BARS = lc.synthetic_bars(n_days=18, seed=5)


# ---------------- 截断与时点 ----------------


def test_features_and_event_no_lookahead():
    lc.assert_no_lookahead_frame(lambda d: F.build(d, None), BARS, cuts=4)
    for P in (1, 2, 5):
        lc.assert_no_lookahead_frame(boll_macd_event(P, 0.0), BARS, cuts=4)


def test_trend_signal_no_lookahead():
    def fn(d):
        t = d.index.tz_convert("America/New_York")
        g = d.reset_index(drop=True).assign(
            day=np.asarray(t.date), mod=np.asarray(t.hour * 60 + t.minute)
        )
        return noise_signal(g, 1.0).set_axis(d.index)

    lc.assert_no_lookahead_frame(fn, BARS, cuts=4)


def test_window_features_unavailable_before_window_completes():
    X = F.build(BARS, None)
    t = BARS.index.tz_convert("America/New_York")
    m = np.asarray(t.hour * 60 + t.minute)
    for col, first_ok in {
        **{f"or{n}_hi_dist": 570 + n - 1 for n in (5, 15, 30)},
        "first30_ret": 599,
    }.items():
        assert not np.isfinite(X[col].to_numpy()[m < first_ok]).any()
        assert np.isfinite(X[col].to_numpy()[(m >= first_ok) & (m < 940)]).any()


# ---------------- L4 引擎 ----------------


def _grid(bars: pd.DataFrame) -> dict:
    d = np.asarray(bars.index.tz_convert("America/New_York").date)
    start = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    cnt = np.diff(np.r_[start, len(d)])
    return {
        "o": bars["open"].to_numpy(float),
        "h": bars["high"].to_numpy(float),
        "l": bars["low"].to_numpy(float),
        "c": bars["close"].to_numpy(float),
        "day_end": np.repeat(start + cnt, cnt),
    }


def test_l4_matches_zr_sim_stock():
    G = _grid(BARS)
    rng = np.random.default_rng(0)
    d_all = np.asarray(BARS.index.tz_convert("America/New_York").date)
    orig = zs.size_shares
    try:
        for k in range(60):
            e = int(rng.integers(5, len(BARS) - 5))
            d = 1 if k % 2 else -1
            kind, H, m = (
                ("b" if k % 4 < 2 else "d"),
                (5, 15, 30)[k % 3],
                (0.001, 0.002, 0.003)[k % 3],
            )
            r = l4(G, np.array([e]), d, m, H, kind)
            ent, ex = float(r["entry"][0]), float(r["exit"][0])
            sh = int(np.floor(ACCT["B"] / ent))
            pnl = sh * d * (ex - ent) - float(
                round_trip_fees(np.array([sh]), np.array([ent]), np.array([ex]), np.array([d]))[0]
            )
            day = d_all == d_all[e]
            s0 = int(np.flatnonzero(day)[0])
            x = BARS[day]
            bars = pd.DataFrame(
                {
                    "ts_ns": x.index.as_unit("ns").asi8,
                    **{c: x[c].to_numpy() for c in ("open", "high", "low", "close")},
                }
            )
            rule = (
                {"kind": "bracket", "tp_pct": m, "sl_pct": m, "max_bars": H - 1}
                if kind == "b"
                else {"kind": "time", "bars": H}
            )
            zs.size_shares = lambda *a, _sh=sh, **kw: _sh
            tr = zs.simulate_day(
                bars,
                [(e - s0, d)],
                Account(equity_usd=ACCT["B"]),
                {"first_entry": "09:30", "last_entry": "16:00", "flat": "16:00"},
                exit_rule=rule,
            ).trades[0]
            assert abs(ent - tr.entry_px) < 1e-9 and abs(ex - tr.exit_px) < 1e-6 and tr.qty == sh
            assert abs(pnl - tr.pnl_usd) <= 0.01
    finally:
        zs.size_shares = orig


def _G(o, h, l_, c):
    n = len(o)
    return {
        "o": np.array(o, float),
        "h": np.array(h, float),
        "l": np.array(l_, float),
        "c": np.array(c, float),
        "day_end": np.full(n, n),
    }


def test_l4_reversal_giveback_hand_cases():
    flat = [100.0] * 12
    g1 = _G(flat, [100.05] * 12, [99.95] * 12, flat)
    nxt = np.full(12, 10**9)
    nxt[1] = 4
    r = l4(g1, np.array([1]), 1, 0.002, 15, "a", rev_next=nxt)
    assert r["reason"][0] == REV and r["hold"][0] == 3 and abs(r["exit"][0] - 99.99) < 1e-9
    g3 = _G(flat, [100.05] * 12, [100.0 if i != 3 else 99.6 for i in range(12)], flat)
    r = l4(g3, np.array([1]), 1, 0.002, 15, "a", rev_next=nxt)
    assert r["reason"][0] == SL and abs(r["exit"][0] - (100.01 * 0.998 - 0.01)) < 1e-9
    h, lo = [100.0] * 12, [100.0] * 12
    h[2], lo[2], h[3], lo[3] = 100.51, 100.30, 100.35, 100.20
    g4 = _G(flat[:2] + [100.30, 100.34] + flat[4:], h, lo, flat)
    r = l4(g4, np.array([1]), 1, 0.003, 15, "c")
    assert (
        r["reason"][0] == GB
        and abs(r["exit"][0] - (100.01 * (1 + 0.6 * (100.51 / 100.01 - 1)) - 0.01)) < 1e-9
    )
    g6 = _G(flat, [100.0] * 12, [100.0 if i != 5 else 99.7 for i in range(12)], flat)
    assert l4(g6, np.array([1]), -1, 0.002, 15, "b")["reason"][0] == TP
    assert l4(g1, np.array([1]), 1, 0.01, 5, "b")["reason"][0] == TIME
    assert l4(g1, np.array([8]), 1, 0.01, 15, "b")["reason"][0] == EOD


# ---------------- 统计函数已知答案 ----------------


def test_pooled_mean_and_cluster_variance_known_answer():
    # x = [1, 2, 3, 6]，日 [a, a, b, b]：r̄ = 3；S − r̄n = (3 − 6, 9 − 6) → Σ² = 18；/N² = 1.125；× D/(D−1) = 2.25
    r, var = S.cluster_var(np.array([1.0, 2, 3, 6]), np.array(["a", "a", "b", "b"]))
    assert r == 3 and var == pytest.approx(2.25)
    s = S.pooled(np.array([1.0, 2, 3]), np.array(["a", "a", "b"]), n_boot=0)
    # r̄ = 2；(3 − 4, 3 − 2) → Σ² = 2；/9 × 2 = 0.4444；按天等权 = (1.5 + 3) / 2
    assert (
        s["mean"] == 2
        and s["se"] ** 2 == pytest.approx(4 / 9)
        and s["day_eq"] == pytest.approx(2.25)
    )
    assert s["t"] == pytest.approx(2 / np.sqrt(4 / 9))


def test_daily_ic_global_known_answer_and_l002_bias():
    n = 50
    f = np.tile(np.arange(n, dtype=float), 2)
    day = np.repeat([0, 1], n)
    ic = S.daily_ic_global(f, f.copy(), day)
    assert ic.mean() == pytest.approx(
        1.0, rel=1e-12
    )  # 同序：z 用总体标准差（ddof=0）标准化，Σz²/N = 1
    # L-002：纯随机游走、特征 = 当日开盘以来收益（水平量）、标签 = 15 分钟未来收益
    rng = np.random.default_rng(0)
    D, m, h = 200, 390, 15
    p = np.cumsum(rng.normal(0, 1e-3, (D, m)), 1)
    y = np.full((D, m), np.nan)
    y[:, : m - h] = p[:, h:] - p[:, : m - h]
    dd = np.repeat(np.arange(D), m)
    assert S.daily_ic_withinday(p.ravel(), y.ravel(), dd).mean() < -0.2
    assert abs(S.daily_ic_global(p.ravel(), y.ravel(), dd).mean()) < 0.03


def test_net_pct_uses_stock_order_fees():
    pct, sh = net_pct(np.array([100.01]), np.array([100.51]), np.array([1]), "B")
    fee = float(
        stock_order_fees(int(sh[0]), "Buy", 100.01)["total"]
        + stock_order_fees(int(sh[0]), "Sell", 100.51)["total"]
    )
    assert sh[0] == 999 and pct[0] == pytest.approx((999 * 0.5 - fee) / (999 * 100.01) * 100)


def test_random_dirs_permute_keeps_counts():
    d = np.array([1, 1, 1, -1, 1])
    R = random_dirs(d, 200, "permute", seed=1)
    assert (R.sum(1) == d.sum()).all() and not (R == d).all()
    H = random_dirs(d, 2000, "half", seed=1)
    assert abs(H.mean()) < 0.05
