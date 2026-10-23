"""R-045 日内动量：噪声区、信号与回测逻辑的合成数据测试。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import zr.research.spy_intraday_momentum as sim


def _panel(n_days: int, path_fn, open_px: float = 100.0) -> sim.Panel:
    """n_days 个完整交易日（390 根），收盘 = path_fn(日, 分钟)；VWAP 用同一收盘的累计均值。"""
    days = [dt.date(2025, 1, 6) + dt.timedelta(days=i) for i in range(n_days)]
    close = np.array([[path_fn(d, m) for m in range(390)] for d in range(n_days)], float)
    vwap = np.cumsum(close, axis=1) / np.arange(1, 391)
    open_ = np.full(n_days, open_px)
    last = np.full(n_days, 389)
    return sim.Panel(days, close, vwap, open_, last, close[:, -1].copy())


def test_sigma_open_is_mean_of_prior_14_same_minute_moves():
    # 第 d 天每分钟的 |move| = 0.001 × (d + 1)
    p = _panel(20, lambda d, m: 100 * (1 + 0.001 * (d + 1)))
    s = sim.sigma_open(p.close, p.open_)
    assert np.isnan(s[:14]).all()  # 前 14 天无完整窗口（不含当日）
    assert s[14, 100] == pytest.approx(np.mean([0.001 * (k + 1) for k in range(14)]))
    assert s[19, 0] == pytest.approx(np.mean([0.001 * (k + 1) for k in range(5, 19)]))


def test_bands_use_max_open_prev_close_and_min():
    p = _panel(2, lambda d, m: 100.0)
    p.open_[:] = [100.0, 101.0]
    p.day_close[:] = [99.0, 100.0]
    sig = np.full((2, 390), 0.01)
    ub, lb = sim.bands(p, sig)
    assert ub[1, 0] == pytest.approx(101.0 * 1.01)  # max(开盘 101, 前收 99)
    assert lb[1, 0] == pytest.approx(99.0 * 0.99)  # min(101, 99)
    assert ub[0, 0] == pytest.approx(100.0 * 1.01)  # 第一天无前收：只用开盘


def test_signal_requires_band_and_vwap_and_opp_band_holds():
    assert sim.signal(102, 101, 99, 101.5, 0, "final") == 1
    assert sim.signal(102, 101, 99, 102.5, 1, "final") == 0  # 在带外但低于 VWAP → 平
    assert sim.signal(98, 101, 99, 98.5, 0, "final") == -1
    assert sim.signal(100, 101, 99, 100, 1, "final") == 0
    assert sim.signal(100, 101, 99, 100, 1, "opp_band") == 1  # 带内保持
    assert sim.signal(98, 101, 99, 100, 1, "opp_band") == -1  # 越过反向带 → 反手


def test_backtest_enters_at_first_check_outside_band_and_flattens_at_close():
    # 前 15 天平静（日内 ±0.05% 来回），第 16 天开盘后单边上涨 1%/小时
    def path(d, m):
        if d < 15:
            return 100 * (1 + 0.0005 * np.sin(m / 7))
        return 100 * (1 + 0.01 * m / 60)

    p = _panel(16, path)
    _, trades = sim.backtest(p, "band_vwap", "paper")
    t = trades[trades["day"] == p.days[15]]
    assert len(t) == 1 and t["dir"].iloc[0] == 1
    assert t["entry_m"].iloc[0] == sim.FIRST_CHECK  # 10:00 那根第一次判定即入场
    assert t["exit_m"].iloc[0] == 389  # 收盘平仓
    assert t["entry_px"].iloc[0] == pytest.approx(p.close[15, 30])
    shares = t["shares"].iloc[0]
    assert t["cost"].iloc[0] == pytest.approx(2 * (max(0.35, 0.0035 * shares) + 0.001 * shares))


def test_dynamic_sizing_caps_leverage_at_4():
    p = _panel(16, lambda d, m: 100.0 + 1e-6 * d)  # 日收益极小 → 2% / σ 远大于 4
    lev = sim.TARGET_VOL / sim.daily_vol(p)[15]
    assert lev > 4 and min(sim.MAX_LEV, lev) == 4


def test_random_direction_keeps_costs_and_is_reproducible():
    t = pd.DataFrame(
        {
            "shares": [10, 10],
            "entry_px": [100.0, 100.0],
            "exit_px": [101.0, 99.0],
            "cost": [1.0, 1.0],
        }
    )
    a = sim.random_direction(t, n=50, seed=0)
    b = sim.random_direction(t, n=50, seed=0)
    assert np.array_equal(a, b) and set(np.round(a, 6)) <= {18.0, -2.0, -22.0}
