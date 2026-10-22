import datetime as dt

import numpy as np
import pandas as pd

from zr.features.bars import ET
from zr.research import labels_v2 as lv


def _day(close: np.ndarray, day: dt.date = dt.date(2026, 9, 16)) -> pd.DataFrame:
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 30)), tz=ET).tz_convert("UTC")
    idx = pd.date_range(start, periods=len(close), freq="min", name="ts")
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 1},
        index=idx,
    )


def test_mfe_mae_directional_and_same_day_only():
    close = np.full(390, 100.0)
    close[101:106] = [100.5, 101.0, 99.0, 100.0, 100.0]
    df = pd.concat([_day(close), _day(np.full(390, 200.0), dt.date(2026, 9, 17))])
    ts = pd.Series([df.index[100], df.index[100], df.index[389]])
    mfe, mae = lv.mfe_mae(df, ts, np.array([1, -1, 1]), 3)
    assert np.isclose(mfe[0], 101.1 / 100 - 1) and np.isclose(mae[0], 98.9 / 100 - 1)
    assert np.isclose(mfe[1], 1 - 98.9 / 100) and np.isclose(mae[1], 1 - 101.1 / 100)
    assert np.isnan(mfe[2])  # 当日最后一根之后没有同日 bar（不跨到次日 200）


def test_stock_labels_columns_and_ever_up():
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.001, 390))
    df = _day(close)
    ev = pd.DataFrame({"ts": df.index[[100, 200, 300]], "direction": [1, -1, 1]})
    lab = lv.stock_labels(df, ev)
    for h in lv.H_MIN:
        assert {f"mfe_{h}", f"mae_{h}", f"tb_label_{h}", f"ever_up_0.5_{h}"} <= set(lab.columns)
    # ever_up 单调：x 越小越容易达到
    assert (lab["ever_up_0.3_10"] >= lab["ever_up_1.0_10"]).all()
    # MFE 随 H 不减
    assert (lab["mfe_30"] >= lab["mfe_5"] - 1e-12).all()


def test_option_labels_join_on_decision_minute():
    ev = pd.DataFrame(
        {"ts": [pd.Timestamp("2026-09-16 10:00", tz=ET).tz_convert("UTC")], "direction": [1]}
    )
    outc = pd.DataFrame(
        {
            "day": [dt.date(2026, 9, 16)] * 2,
            "minute": [31, 31],
            "direction": [1, 1],
            "cap": [10, 30],
            "net": [5.0, 8.0],
            "dte": [0, 0],
            "d001_violation": [False, False],
            "bid_max_ret_5": [0.2, 0.2],
        }
    )
    lab = lv.option_labels(ev, outc, np.array([30]))  # 10:00 = 第 30 分钟，决策分钟 31
    assert (
        lab.loc[0, "net_cap10"] == 5.0
        and lab.loc[0, "net_cap30"] == 8.0
        and lab.loc[0, "bid_max_ret_5"] == 0.2
    )
