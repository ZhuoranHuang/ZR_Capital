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


def _session_bars(
    day: dt.date, session: str, start_hm: str, n: int, vol: float = 100.0
) -> pd.DataFrame:
    h, m = map(int, start_hm.split(":"))
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(h, m)), tz=ET).tz_convert("UTC")
    idx = pd.date_range(start, periods=n, freq="min", name="ts")
    px = 100.0 + np.arange(n) * 0.001
    return pd.DataFrame(
        {
            "ts": idx,
            "symbol": "MU.US",
            "open": px,
            "high": px + 0.02,
            "low": px - 0.02,
            "close": px,
            "volume": vol,
            "turnover": px * vol,
            "session": session,
        },
    )


def test_load_extended_bars_includes_pre_and_post_and_filters_holidays(tmp_path):
    from zr.features.bars import load_extended_bars

    day = dt.date(2026, 10, 19)  # 周一，交易日
    holiday = dt.date(2026, 10, 24)  # 周六，非交易日（长桥可能生成零成交 bar）
    rows = pd.concat(
        [
            _session_bars(day, "Pre", "04:00", 330),  # 04:00-09:29
            _session_bars(day, "Intraday", "09:30", 390),
            _session_bars(day, "Post", "16:00", 240),  # 16:00-19:59
            _session_bars(holiday, "Intraday", "09:30", 390, vol=0.0),
        ],
        ignore_index=True,
    )
    p = tmp_path / "bars_1m" / "underlying=MU" / "year=2026"
    p.mkdir(parents=True)
    rows.to_parquet(p / "bars.parquet", index=False)
    out = load_extended_bars(tmp_path, "MU")
    et = out.index.tz_convert(ET)
    assert set(et.date) == {day}  # 周六的记录被交易日历过滤掉
    assert out["session"].value_counts().to_dict() == {"Pre": 330, "Intraday": 390, "Post": 240}
    assert et.strftime("%H:%M").min() == "04:00"
    assert et.strftime("%H:%M").max() == "19:59"


def test_clean_extended_bars_drops_zero_volume_and_range_spike():
    from zr.data.bar_quality import clean_extended_bars

    day = dt.date(2026, 10, 19)
    df = _session_bars(day, "Pre", "04:00", 60).set_index("ts")
    df.index = df.index.tz_convert("UTC")
    df.loc[df.index[10], "volume"] = 0.0  # 成交量为 0
    df.loc[df.index[30], ["high", "low"]] = [200.0, 50.0]  # 单根振幅 > 5×ATR
    out = clean_extended_bars(df)
    assert len(out) < len(df)
    assert df.index[10] not in out.index
    assert df.index[30] not in out.index
