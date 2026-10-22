"""前视审计第 1 节「成交时点」：X1–X7 每一种离场信号，逐一确认

触发 bar 收盘 → 用触发后第一个快照成交（= 触发 bar 收盘分钟 + 1），不早不晚。
入场同理：决策时刻（信号 bar 收盘）+ 2 秒后的第一个快照（= 收盘分钟 + 1）。
"""

import datetime as dt

import numpy as np
import pandas as pd

from zr.features.regime import load_config
from zr.research import exits as ex
from zr.research import sim_v3 as sv

DAY = dt.date(2025, 6, 10)
CFG = load_config()
PERIOD = 2


def _idx(n, start="09:30"):
    i = pd.date_range(f"{DAY} {start}", periods=n, freq=f"{PERIOD}min", tz="America/New_York")
    return i.tz_convert("UTC")


def _od(bid: np.ndarray, ask: np.ndarray, expiry=None) -> sv.OptDay:
    expiry = expiry or (DAY + dt.timedelta(days=1))
    sym = "X"
    return sv.OptDay(
        DAY,
        [expiry],
        {(expiry, "call"): (np.array([101.0]), [sym]), (expiry, "put"): (np.array([99.0]), [sym])},
        {sym: bid},
        {sym: ask},
        {sym: (bid + ask) / 2},
        {sym: np.zeros(sv.MINUTES, bool)},
    )


def _cfg():
    c = {**CFG, "execution": {**CFG["execution"]}}
    c["exits"] = {
        **CFG["exits"],
        "X6": {"tp": [0.99], "sl": [0.99], "basis": "mid"},
        "X7": {"cap_min": [300]},
    }
    return c


def _fill_via_trig(
    rule_name: str,
    trig_minutes_long: np.ndarray,
    trig_minutes_short: np.ndarray,
    entry_m: int,
    direction: int,
) -> int:
    """用给定的触发分钟集合跑一次 evaluate()，返回出场的 bid 快照分钟（= 相对 entry 的 hold +
    entry_m）。
    """
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    od = _od(bid, ask)
    trig = {rule_name: {1: np.sort(trig_minutes_long), -1: np.sort(trig_minutes_short)}}
    ev = pd.DataFrame({"e": [entry_m], "direction": [direction], "spot": [100.0]})
    r = sv.evaluate(od, ev, trig, _cfg())
    return entry_m + int(r["hold"][0, 0, 0, 0, 0])


def _check_exit_fires_and_fills(
    df: pd.DataFrame,
    x: pd.DataFrame,
    cfg: dict,
    rule: str,
    expected_bar_long: int | None,
    expected_bar_short: int | None,
    entry_m: int = 10,
):
    exs = ex.exit_signals(df, x, cfg)
    lg, sh = exs[rule]
    if expected_bar_long is not None:
        assert lg[expected_bar_long], f"{rule} 多头未在预期 bar {expected_bar_long} 触发"
        assert not lg[:expected_bar_long].any(), f"{rule} 多头提前触发"
        close_min = df.index.tz_convert("America/New_York")
        cm = (close_min.hour * 60 + close_min.minute - 570).to_numpy() + PERIOD
        fill = _fill_via_trig(rule, np.array([cm[expected_bar_long]]), np.array([]), entry_m, 1)
        assert fill == cm[expected_bar_long] + 1, (
            f"{rule} 多头成交时点错误：{fill} != {cm[expected_bar_long] + 1}"
        )
    if expected_bar_short is not None:
        assert sh[expected_bar_short], f"{rule} 空头未在预期 bar {expected_bar_short} 触发"
        assert not sh[:expected_bar_short].any(), f"{rule} 空头提前触发"
        close_min = df.index.tz_convert("America/New_York")
        cm = (close_min.hour * 60 + close_min.minute - 570).to_numpy() + PERIOD
        fill = _fill_via_trig(rule, np.array([]), np.array([cm[expected_bar_short]]), entry_m, -1)
        assert fill == cm[expected_bar_short] + 1, f"{rule} 空头成交时点错误"


def _manual_x(n, dif, dea, mid, pctb, hw=1.0):
    idx = _idx(n)
    return pd.DataFrame(
        {
            "dif": dif,
            "dea": dea,
            "hist": np.subtract(dif, dea),
            "mid": mid,
            "upper": np.add(mid, hw),
            "lower": np.subtract(mid, hw),
            "hw": hw,
            "pctb": pctb,
            "bw": 2 * hw,
            "slope": 0.0,
        },
        index=idx,
    )


def _bars(close, high=None, low=None, open_=None):
    n = len(close)
    idx = _idx(n)
    close = np.asarray(close, float)
    o = np.asarray(open_, float) if open_ is not None else np.r_[close[0], close[:-1]]
    h = np.asarray(high, float) if high is not None else np.maximum(o, close) + 0.01
    lo = np.asarray(low, float) if low is not None else np.minimum(o, close) - 0.01
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": close, "volume": 1.0}, index=idx)


def test_x1_sma_and_ema_cross():
    close = np.r_[np.full(10, 100.0), np.linspace(100, 90, 10)]
    df = _bars(close)
    x = _manual_x(
        len(df),
        dif=np.zeros(len(df)),
        dea=np.zeros(len(df)),
        mid=np.full(len(df), 100.0),
        pctb=np.full(len(df), 0.5),
    )
    for kind in CFG["exits"]["X1"]["kind"]:
        for m in CFG["exits"]["X1"]["ma"]:
            from zr.features import indicators as ind

            ma = (ind.sma(df["close"], m) if kind == "sma" else ind.ema(df["close"], m)).to_numpy()
            below = np.flatnonzero(df["close"].to_numpy() < ma)
            if len(below) == 0:
                continue
            bar = int(below[0])
            _check_exit_fires_and_fills(df, x, CFG, f"X1_{kind}{m}", bar, None, entry_m=2)
    close_up = close[::-1].copy()
    df2 = _bars(close_up)
    for kind in CFG["exits"]["X1"]["kind"]:
        for m in CFG["exits"]["X1"]["ma"]:
            from zr.features import indicators as ind

            ma = (
                ind.sma(df2["close"], m) if kind == "sma" else ind.ema(df2["close"], m)
            ).to_numpy()
            above = np.flatnonzero(df2["close"].to_numpy() > ma)
            if len(above) == 0:
                continue
            bar = int(above[0])
            _check_exit_fires_and_fills(df2, x, CFG, f"X1_{kind}{m}", None, bar, entry_m=2)


def test_x2_cross_shrink_flip():
    n = 20
    df = _bars(np.full(n, 100.0))
    # cross: DIF 从高于 DEA 变为低于（第 10 根发生）
    dif = np.r_[np.full(10, 1.0), np.full(10, -1.0)]
    dea = np.zeros(n)
    x = _manual_x(n, dif, dea, mid=np.full(n, 100.0), pctb=np.full(n, 0.5))
    _check_exit_fires_and_fills(df, x, CFG, "X2_cross", 10, None, entry_m=2)
    x2 = _manual_x(n, -dif, dea, mid=np.full(n, 100.0), pctb=np.full(n, 0.5))
    _check_exit_fires_and_fills(df, x2, CFG, "X2_cross", None, 10, entry_m=2)

    # shrink_k：柱连续 k 根下降
    hist = np.r_[np.full(5, 5.0), 5 - np.arange(1, 10) * 0.5, np.full(6, -0.5)]
    x3 = pd.DataFrame(
        {
            "dif": hist,
            "dea": np.zeros(n),
            "hist": hist,
            "mid": np.full(n, 100.0),
            "upper": 101.0,
            "lower": 99.0,
            "hw": 1.0,
            "pctb": 0.5,
            "bw": 2.0,
        },
        index=_idx(n),
    )
    for k in (1, 2, 3):
        falling = np.zeros(n, dtype=bool)
        run = 0
        for i in range(1, n):
            run = run + 1 if hist[i] < hist[i - 1] else 0
            falling[i] = run >= k
        bar = int(np.flatnonzero(falling)[0])
        _check_exit_fires_and_fills(df, x3, CFG, f"X2_shrink{k}", bar, None, entry_m=2)

    # flip：柱由正变负
    histf = np.r_[np.full(10, 1.0), np.full(10, -1.0)]
    xf = pd.DataFrame(
        {
            "dif": histf,
            "dea": np.zeros(n),
            "hist": histf,
            "mid": np.full(n, 100.0),
            "upper": 101.0,
            "lower": 99.0,
            "hw": 1.0,
            "pctb": 0.5,
            "bw": 2.0,
        },
        index=_idx(n),
    )
    _check_exit_fires_and_fills(df, xf, CFG, "X2_flip", 10, None, entry_m=2)
    _check_exit_fires_and_fills(
        df, xf.assign(hist=-histf, dif=-histf), CFG, "X2_flip", None, 10, entry_m=2
    )


def test_x3_mid1_mid2():
    n = 15
    close = np.r_[np.full(8, 100.0), np.full(7, 95.0)]
    df = _bars(close)
    mid = np.full(n, 98.0)
    x = _manual_x(n, dif=np.zeros(n), dea=np.zeros(n), mid=mid, pctb=np.full(n, 0.5))
    below = np.flatnonzero(close < mid)
    _check_exit_fires_and_fills(df, x, CFG, "X3_mid1", int(below[0]), None, entry_m=2)
    # mid2 需要连续 2 根且同日
    _check_exit_fires_and_fills(df, x, CFG, "X3_mid2", int(below[1]), None, entry_m=2)
    close_up = np.r_[np.full(8, 100.0), np.full(7, 105.0)]
    df2 = _bars(close_up)
    above = np.flatnonzero(close_up > mid)
    _check_exit_fires_and_fills(df2, x, CFG, "X3_mid1", None, int(above[0]), entry_m=2)


def test_x4_back_inside():
    n = 10
    df = _bars(np.full(n, 100.0))
    pctb = np.r_[np.full(5, 1.2), np.full(5, 0.5)]  # 多头：外轨之外，第 5 根回到带内
    x = _manual_x(n, dif=np.zeros(n), dea=np.zeros(n), mid=np.full(n, 100.0), pctb=pctb)
    inside = np.flatnonzero(pctb < 0.7)
    _check_exit_fires_and_fills(df, x, CFG, "X4_back_inside", int(inside[0]), None, entry_m=1)
    pctb_s = np.r_[np.full(5, -0.2), np.full(5, 0.5)]
    xs = _manual_x(n, dif=np.zeros(n), dea=np.zeros(n), mid=np.full(n, 100.0), pctb=pctb_s)
    inside_s = np.flatnonzero(pctb_s > 0.3)
    _check_exit_fires_and_fills(df, xs, CFG, "X4_back_inside", None, int(inside_s[0]), entry_m=1)


def test_x5_higher_low_break():
    # 第 1 个摆动低点 90（第 2 行，i=4 用 w=2 确认）；第 2 个摆动低点 95（第 7 行，i=9 确认，
    # 95>90 是"更高低点"→ 水位=95，从第 9 行起生效）；此后收盘一直 >95，直到第 14 行跌破（80.5<95）
    # 。
    low = np.array([100, 99, 90, 94, 98, 101, 103, 95, 99, 104, 106, 108, 110, 112, 80])
    high = low + 1.0
    close = low + 0.5
    df = _bars(close, high=high, low=low)
    x = _manual_x(
        len(df),
        dif=np.zeros(len(df)),
        dea=np.zeros(len(df)),
        mid=np.full(len(df), 100.0),
        pctb=np.full(len(df), 0.5),
    )
    exs = ex.exit_signals(df, x, CFG)
    lg, sh = exs["X5_higher_low"]
    fire = np.flatnonzero(lg)
    assert len(fire) > 0, "X5_higher_low 多头从未触发（摆动点样例需要调整）"
    bar = int(fire[0])
    assert not lg[:bar].any()
    close_min = df.index.tz_convert("America/New_York")
    cm = (close_min.hour * 60 + close_min.minute - 570).to_numpy() + PERIOD
    fill = _fill_via_trig(
        "X5_higher_low", np.array([cm[bar]]), np.array([]), entry_m=2, direction=1
    )
    assert fill == cm[bar] + 1


def test_x6_fires_on_mid_fills_next_bid_snapshot():
    """X6 按 mid 判定（50 分钟触发），1 张在下一快照（51）成交；
    cap=12 让剩余 1 张在 52 分钟到期。
    """
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    bid[50], ask[50] = (
        1.3,
        1.5,
    )  # mid=1.4 >= 1.1*(1+0.2)=1.32 → 触发（判定用 mid，不用 bid/ask 本身）
    bid[51] = 1.25  # 触发快照(50)本身不可用于成交，用下一快照(51)的 bid
    od = _od(bid, ask)
    cfg = {
        **CFG,
        "execution": CFG["execution"],
        "exits": {
            **CFG["exits"],
            "X6": {"tp": [0.2], "sl": [0.99], "basis": "mid"},
            "X7": {"cap_min": [12]},
        },
    }
    ev = pd.DataFrame({"e": [40], "direction": [1], "spot": [100.0]})
    trig = {"none": {1: np.array([], int), -1: np.array([], int)}}
    r = sv.evaluate(od, ev, trig, cfg)
    assert r["hold"][0, 0, 0, 0, 0] == 12  # 剩余 1 张持有到 cap（e+12=52），早于 cap 的 tp 腿已先出
    ask0 = 1.1
    exp = (
        -ask0 * 200
        - sv._BUY[2]
        + 1.25 * 100
        - sv.sell_fee(1, 1.25)  # tp 腿：51 分钟 bid
        + 1.0 * 100
        - sv.sell_fee(1, 1.0)  # cap 腿：52 分钟 bid（未改动，仍 1.0）
    )
    assert np.isclose(r["net"][0, 0, 0, 0, 0], exp, atol=1e-3)


def test_x7_cap_exits_exactly_at_scheduled_minute_no_extra_lag():
    bid = np.full(sv.MINUTES, 1.0)
    ask = np.full(sv.MINUTES, 1.1)
    od = _od(bid, ask)
    cfg = {
        **CFG,
        "execution": CFG["execution"],
        "exits": {
            **CFG["exits"],
            "X6": {"tp": [0.99], "sl": [0.99], "basis": "mid"},
            "X7": {"cap_min": [20]},
        },
    }
    ev = pd.DataFrame({"e": [40], "direction": [1], "spot": [100.0]})
    trig = {"none": {1: np.array([], int), -1: np.array([], int)}}
    r = sv.evaluate(od, ev, trig, cfg)
    assert (
        r["hold"][0, 0, 0, 0, 0] == 20
    )  # 直接到 e+cap，无额外 +1 滞后（计划性时间止损不需要反应时间）


def test_entry_fills_at_signal_bar_close_plus_one_minute():
    """入场 = 信号 K 收盘 + 2 秒后的第一个快照 = 收盘分钟 + 1（Theta 1 分钟快照粒度下无法区分 +2s）
    。
    """
    df = _bars(np.full(20, 100.0))
    close_min = df.index.tz_convert("America/New_York")
    cm = (close_min.hour * 60 + close_min.minute - 570).to_numpy() + PERIOD

    class _Fake:
        pass

    pdat = _Fake()
    pdat.close_min = cm
    pdat.bars = df
    pdat.entries = {"sig": np.where(np.arange(len(df)) == 5, 1, 0).astype(np.int8)}
    ev, by_cfg = sv.day_events(pdat, np.ones(len(df), dtype=bool), last_entry=sv.hm("15:40"))
    assert len(ev) == 1
    assert ev["e"].iloc[0] == cm[5] + 1
    assert ev["direction"].iloc[0] == 1
    assert ev["spot"].iloc[0] == df["close"].iloc[5]
    assert "sig" in by_cfg
