import datetime as dt

import numpy as np
import pandas as pd

from zr.features.bars import ET, daily_volume, resample_bars, volume_bars, volume_threshold


def _day_bars(day: dt.date, n: int = 390, base: float = 100.0, vol: int = 100) -> pd.DataFrame:
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    idx = pd.date_range(start, periods=n, freq="min", name="ts")
    px = base + np.arange(n) * 0.01
    return pd.DataFrame(
        {"open": px, "high": px + 0.05, "low": px - 0.05, "close": px + 0.01, "volume": vol},
        index=idx,
    )


def test_resample_does_not_cross_days():
    df = pd.concat([_day_bars(dt.date(2026, 10, 19)), _day_bars(dt.date(2026, 10, 20))])
    r5 = resample_bars(df, "5min")
    assert len(r5) == 2 * 78
    first = r5.iloc[0]
    assert first["open"] == df["open"].iloc[0] and first["close"] == df["close"].iloc[4]
    assert first["volume"] == 500 and first["high"] == df["high"].iloc[:5].max()
    assert r5.index[0].tz_convert(ET).time() == dt.time(9, 30)
    # 第二天第一根不包含第一天数据
    assert r5.index[78].tz_convert(ET).date() == dt.date(2026, 10, 20)


def test_volume_threshold_uses_only_prior_days():
    days = [dt.date(2026, 10, 1) + dt.timedelta(days=i) for i in range(25)]
    df = pd.concat([_day_bars(d, vol=100 + i) for i, d in enumerate(days)])
    thr = volume_threshold(df, lookback_days=20, divisor=100)
    assert thr.iloc[:20].isna().all()
    # 第 21 天阈值 = 前 20 天日均量 / 100；日量 = 390 × (100+i)
    expected = np.mean([390 * (100 + i) for i in range(20)]) / 100
    assert np.isclose(thr.iloc[20], expected)
    dv = daily_volume(df)
    assert dv.iloc[0] == 390 * 100


def test_volume_bars_accumulate_and_reset_daily():
    df = pd.concat(
        [
            _day_bars(dt.date(2026, 10, 19), n=10, vol=100),
            _day_bars(dt.date(2026, 10, 20), n=10, vol=100),
        ]
    )
    vb = volume_bars(df, threshold=250)
    # 每日 1000 量，阈值 250 → 每 3 根收一根（300），共 3 根/日，剩 100 不成 bar
    assert len(vb) == 6
    assert (vb["n_bars"] == 3).all() and (vb["volume"] == 300).all()
    assert vb["date"].nunique() == 2
    assert vb.iloc[0]["open"] == df["open"].iloc[0] and vb.iloc[0]["close"] == df["close"].iloc[2]
