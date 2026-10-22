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
    # OI：第一天无前值；第二天变化 = 1.1/1.0 − 1
    assert f.loc[day == days[0], "opt_oi_chg"].isna().all()
    assert np.allclose(f.loc[day == days[1], "opt_oi_chg"], 0.1)
    assert np.allclose(f["opt_pc_oi"], 0.5)


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


def test_oi_chg_same_tenor_not_same_contract(tmp_path):
    """D-019：比较同 DTE 的不同合约，而非同一合约（同一合约随到期临近的增长不计）。"""
    d0, d1 = dt.date(2024, 3, 4), dt.date(2024, 3, 5)
    rows = {
        # 昨日：DTE 1（3/5 到期）OI 100，DTE 2（3/6 到期）OI 40
        d0: [(dt.date(2024, 3, 5), 100), (dt.date(2024, 3, 6), 40)],
        # 今日：3/6 合约（DTE 1）OI 涨到 150（同一合约 +275%），新的 DTE 2（3/7）OI 60
        d1: [(dt.date(2024, 3, 6), 150), (dt.date(2024, 3, 7), 60)],
    }
    for d, lst in rows.items():
        p = tmp_path / "options_1m/underlying=MU" / f"date={d.isoformat()}"
        p.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "symbol": f"X{e}",
                    "expiry": e,
                    "strike": 100.0,
                    "right": "call",
                    "open_interest": v,
                }
                for e, v in lst
            ]
        ).to_parquet(p / "oi.parquet", index=False)
    t = of.oi_daily(tmp_path, "MU", [d0, d1])
    assert np.isnan(t.loc[d0, "opt_oi_chg"])
    # DTE 1：150 vs 100；DTE 2：60 vs 40 → (150+60)/(100+40) − 1 = 0.5
    assert np.isclose(t.loc[d1, "opt_oi_chg"], 0.5)
    assert t.loc[d1, "opt_oi_n_dte"] == 2
