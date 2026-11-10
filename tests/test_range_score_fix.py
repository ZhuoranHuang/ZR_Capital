"""regime.range_score 前视修复：窗口只用当日之前已完成交易日的根数中位数（首日名义值），
任意截断下输出不变（教训 L-001 / LEAKCHECK_FEATURES.md）。"""

from __future__ import annotations

import pandas as pd
import pytest

from zr.features import regime as rg
from zr.features.bars import resample_bars
from zr.research import leakcheck as lc

CFG = rg.load_config()
RNG = CFG["regime"]["range"]


def _regime_frame(period: int):
    def fn(d: pd.DataFrame) -> pd.DataFrame:
        b = d if period == 1 else resample_bars(d, f"{period}min")
        x = rg.indicators(b, CFG)
        reg = rg.regimes(b, x, CFG, period)
        s = rg.state_features(b, x, reg).set_axis(b.index)
        r = rg.range_score(x, b, RNG, period)
        parts = [reg.add_prefix("reg_"), s.add_prefix("sf_"), r.rename("range_score")]
        return pd.concat(parts, axis=1)

    return fn


@pytest.mark.parametrize("n_days", [1, 2, 5])
def test_range_score_regimes_state_features_no_lookahead(n_days):
    bars = lc.synthetic_bars(n_days=n_days, seed=n_days)
    lc.assert_no_lookahead_frame(_regime_frame(1), bars, cuts=10)


def test_nominal_bars_per_day_by_period_and_session():
    rth = lc.synthetic_bars(n_days=1)
    assert rg.nominal_bars_per_day(rth.index, 1) == 390
    assert rg.nominal_bars_per_day(rth.index, 2) == 195
    assert rg.nominal_bars_per_day(rth.index, 5) == 78
    ext = rth.copy()
    ext.index = ext.index - pd.Timedelta(hours=5)  # 首根落在 04:30 → 含盘前盘后口径
    assert rg.nominal_bars_per_day(ext.index, 2) == 480


def test_window_per_day_uses_only_prior_days():
    bars = lc.synthetic_bars(n_days=4)
    b = bars.drop(bars.index[391:400])  # 第 2 天少 9 根
    w = rg.range_windows(b, 1, RNG)
    day = b.index.tz_convert(rg.ET).date
    days = sorted(set(day))
    per = {d: set(w[day == d]) for d in days}
    assert per[days[0]] == {390}  # 名义值
    assert per[days[1]] == {390}  # 中位数（只有第 1 天 390）
    assert per[days[2]] == {round((390 + 381) / 2)}  # 第 1、2 天中位数
    assert per[days[3]] == {390}  # 390、381、390 的中位数


def test_explicit_bars_per_day_overrides():
    bars = lc.synthetic_bars(n_days=3)
    x = rg.indicators(bars, CFG)
    a = rg.range_score(x, bars, RNG, 1, bars_per_day=200)
    w = round(RNG["window_days"] * 200)
    pr = lambda s: s.rolling(w, min_periods=max(20, w // 4)).rank(pct=True)  # noqa: E731
    wt = RNG["weights"]
    want = wt["bw"] * pr(x["bw"]) + wt["slope"] * pr(x["slope"].abs())
    want = want + wt["hist"] * pr((x["hist"] / bars["close"]).abs())
    pd.testing.assert_series_equal(a, want, check_names=False)
    reg = rg.regimes(bars, x, CFG, 1, bars_per_day=200)
    pd.testing.assert_series_equal(reg["range_score"], a, check_names=False)
