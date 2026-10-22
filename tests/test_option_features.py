import datetime as dt

import numpy as np
import pandas as pd
import pytest

from zr.features import option_features as of
from zr.features.bars import ET

DAY = dt.date(2026, 9, 15)
SIGMA = 0.20


def _stock(day: dt.date, spot: float = 500.0, n: int = 60) -> pd.DataFrame:
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    idx = pd.date_range(start, periods=n, freq="min", name="ts")
    close = np.full(n, spot)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000}, index=idx
    )


def _opt_day(
    day: dt.date, stock: pd.DataFrame, call_vol: float = 10.0, put_vol: float = 5.0
) -> dict:
    strikes = np.arange(490.0, 511.0, 1.0)
    exp = day
    q_rows, o_rows, oi_rows = [], [], []
    for ts in stock.index:
        for k in strikes:
            for r in ("call", "put"):
                # 报价时间戳 = bar 时间 + 1 分钟（Theta 1m 语义），价格由 BS(σ=0.2) 生成
                qts = ts + pd.Timedelta(minutes=1)
                t = of.years_to_expiry(pd.Series([qts]), pd.Series([exp]))[0]
                px = of.bs_price(500.0, k, t, SIGMA, r)
                q_rows.append(
                    {
                        "ts": qts,
                        "expiry": exp,
                        "strike": k,
                        "right": r,
                        "bid": px - 0.005,
                        "ask": px + 0.005,
                    }
                )
                o_rows.append(
                    {
                        "ts": ts,
                        "expiry": exp,
                        "strike": k,
                        "right": r,
                        "volume": call_vol if r == "call" else put_vol,
                    }
                )
    for k in strikes:
        for r in ("call", "put"):
            oi_rows.append(
                {
                    "ts": stock.index[0],
                    "symbol": f"S{k}{r}",
                    "expiry": exp,
                    "strike": k,
                    "right": r,
                    "open_interest": 100.0 if r == "call" else 50.0,
                }
            )
    return {
        "quotes": pd.DataFrame(q_rows),
        "ohlc": pd.DataFrame(o_rows),
        "oi": pd.DataFrame(oi_rows),
    }


def test_iv_roundtrip():
    t = 2 / 365
    for r in ("call", "put"):
        px = of.bs_price(500, 501, t, 0.25, r)
        assert abs(of.implied_vol(px, 500, 501, t, r) - 0.25) < 1e-4
    assert np.isnan(of.implied_vol(0.0, 500, 501, t, "call"))


def test_day_features_volume_ratio_and_iv():
    stock = _stock(DAY)
    opt = _opt_day(DAY, stock)
    f = of.day_features(stock, opt, DAY)
    # ATM±2 = 5 个行权价 × (10 call + 5 put)
    assert (f["opt_near_vol"] == 5 * 15).all()
    assert np.allclose(f["opt_cp_lr_near"], np.log(51 / 26))
    assert np.allclose(f["opt_cp_lr_all"], np.log((21 * 10 + 1) / (21 * 5 + 1)))
    assert f["opt_front_dte"].iloc[0] == 0
    assert (f["opt_atm_iv"].dropna() - SIGMA).abs().max() < 0.01
    assert f["opt_atm_iv_chg5"].iloc[:5].isna().all()


def test_atm_follows_spot():
    stock = _stock(DAY)
    stock["close"] = np.linspace(495, 505, len(stock))
    opt = _opt_day(DAY, stock)
    opt["ohlc"].loc[opt["ohlc"]["strike"] == 495.0, "volume"] = 1000.0
    f = of.day_features(stock, opt, DAY)
    # 起点 spot≈495 时 495 在 ATM±2 内，终点 spot≈505 时不在
    assert f["opt_near_vol"].iloc[0] > f["opt_near_vol"].iloc[-1]


def test_no_lookahead_on_last_minute():
    stock = _stock(DAY)
    opt = _opt_day(DAY, stock)
    base = of.day_features(stock, opt, DAY)
    last_bar = stock.index[-1]
    o2 = {k: v.copy() for k, v in opt.items()}
    o2["ohlc"].loc[o2["ohlc"]["ts"] == last_bar, "volume"] *= 50
    o2["quotes"].loc[o2["quotes"]["ts"] == last_bar + pd.Timedelta(minutes=1), ["bid", "ask"]] *= 3
    bumped = of.day_features(stock, o2, DAY)
    cols = ["opt_near_vol", "opt_cp_lr_near", "opt_atm_iv", "opt_atm_iv_chg5"]
    pd.testing.assert_frame_equal(base[cols].iloc[:-1], bumped[cols].iloc[:-1])
    assert bumped["opt_near_vol"].iloc[-1] != base["opt_near_vol"].iloc[-1]


def test_compute_option_features_multi_day_ratio_and_oi(tmp_path):
    days = [
        DAY + dt.timedelta(days=i) for i in range(10) if (DAY + dt.timedelta(days=i)).weekday() < 5
    ]
    stocks = []
    for i, d in enumerate(days):
        s = _stock(d, n=10)
        stocks.append(s)
        opt = _opt_day(d, s, call_vol=10.0 * (i + 1), put_vol=5.0)
        opt["oi"]["open_interest"] *= 1 + 0.1 * i
        p = tmp_path / "options_1m/underlying=SPY" / f"date={d.isoformat()}"
        p.mkdir(parents=True)
        for k, v in opt.items():
            v.to_parquet(p / f"{k}.parquet", index=False)
    stock = pd.concat(stocks)
    f = of.compute_option_features(tmp_path, "SPY", stock)
    assert list(f.columns) == of.OPTION_FEATURE_COLUMNS
    day = pd.Series(stock.index.tz_convert(ET).date, index=stock.index)
    # 前 5 天没有足够历史 → ratio NaN；第 6 天起有值
    assert f.loc[day == days[0], "opt_near_vol_ratio20"].isna().all()
    assert f.loc[day == days[5], "opt_near_vol_ratio20"].notna().all()
    # OI 水平：call 100 / put 50 → 2；21 个行权价 OI 相同 → ATM±3 占 7/21
    assert np.allclose(f["opt_cp_oi"], 2.0)
    assert np.allclose(f["opt_atm3_oi_share"], 7 / 21)


@pytest.mark.parametrize("missing", ["quotes", "ohlc", "oi"])
def test_missing_file_day_is_skipped(tmp_path, missing):
    s = _stock(DAY, n=5)
    opt = _opt_day(DAY, s)
    p = tmp_path / "options_1m/underlying=SPY" / f"date={DAY.isoformat()}"
    p.mkdir(parents=True)
    for k, v in opt.items():
        if k != missing:
            v.to_parquet(p / f"{k}.parquet", index=False)
    f = of.compute_option_features(tmp_path, "SPY", s)
    assert f.isna().all().all()


def _oi(rows):
    return pd.DataFrame(
        [
            {"symbol": f"{e}{r}{k}", "expiry": e, "strike": k, "right": r, "open_interest": v}
            for e, k, r, v in rows
        ]
    )


def test_oi_level_features_values():
    """D-021：最大 OI 行权价距离按间距归一、ATM±3 占比、call/put 比。"""
    stock = _stock(DAY, spot=100.0, n=3)
    stock["close"] = [100.0, 101.0, 103.0]
    rows = [(DAY, k, r, 10) for k in np.arange(95.0, 106.0, 1.0) for r in ("call", "put")]
    rows.append((DAY, 98.0, "put", 500))  # 最大 OI 在 98
    oi = _oi(rows)
    f = of.oi_level_features(oi, stock["close"], 100.0, 0.08, DAY)
    assert np.allclose(f["opt_maxoi_dist"], [2.0, 3.0, 5.0])
    total = 22 * 10 + 500
    assert np.isclose(f["opt_atm3_oi_share"].iloc[0], (14 * 10 + 500) / total)  # 97–103 含 98
    assert np.isclose(f["opt_atm3_oi_share"].iloc[2], 12 * 10 / total)  # 100–105（上端截断）
    assert np.allclose(f["opt_cp_oi"], 110 / 610)


def test_oi_level_features_ignore_strikes_outside_open_window():
    """拉取带宽随当日高低点变化；开盘价 ±b 之外的合约不影响特征（无前视）。"""
    stock = _stock(DAY, spot=100.0, n=3)
    base_rows = [(DAY, k, r, 10) for k in np.arange(95.0, 106.0, 1.0) for r in ("call", "put")]
    a = of.oi_level_features(_oi(base_rows), stock["close"], 100.0, 0.08, DAY)
    far = base_rows + [(DAY, 120.0, "call", 10_000), (DAY, 80.0, "put", 10_000)]
    b = of.oi_level_features(_oi(far), stock["close"], 100.0, 0.08, DAY)
    pd.testing.assert_frame_equal(a, b)
    # 已到期合约也不计
    old = base_rows + [(DAY - dt.timedelta(days=1), 100.0, "call", 10_000)]
    c = of.oi_level_features(_oi(old), stock["close"], 100.0, 0.08, DAY)
    pd.testing.assert_frame_equal(a, c)


def test_strike_spacing_from_data():
    ks = np.r_[np.arange(90.0, 100.0, 0.5), np.arange(100.0, 130.0, 2.5)]
    assert of.strike_spacing(ks, 95.0) == 0.5
    assert of.strike_spacing(ks, 120.0) == 2.5
