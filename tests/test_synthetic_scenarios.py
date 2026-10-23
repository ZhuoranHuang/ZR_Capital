"""合成场景测试（audit/base-final 第 3 节）：人为构造的 2 分钟 K + 期权 tick，已知答案逐项断言。

走生产代码路径：indicators（口径 C：连续、不按日重置）→ regimes →
v4_decisive.hypothesis_signals / exit_arrays / guard_mask → v4_decisive.simulate（tick 级成交）。
日期 2026-08-04（周二，正常交易日），K 线从 08:00 ET 起（45 根盘前 K 给 BOLL 预热，与口径 C 一致）。

(a) 通道回踩：CHANNEL↑ 中一根下影探到中轨上方 0.1 带半宽的阴线 + 阳线 → H2（E1）在 10:22 那根；
    离场 (c') = X5 跌破前一更高低点 → 10:42 那根。
(b) 开盘急跌：09:30 起连续收在下轨之外 → 09:32（第 2 根）IGNITION↓ 且 H1（E10，k=2）；
    离场 (c) = X4 %b 回到 [0.1, 0.9] → 09:46。
(c) 横盘放量突破：RANGE 到 10:28，10:30 收盘破前 20 根高点、量比 4 → E11；下一根仍在突破位之上 →
    H3 在 10:32；下一根收回突破位以内 → 无 H3。
(d) X1–X7：同 (a) 的入场，各离场规则的触发 K 与 tick 成交价（触发 + 2 s 后第一个 bid）。
做空：价格关于 100 镜像（high/low 互换），信号与离场方向互换、K 位置不变。
镜像下状态并非逐根对称：带宽 bw、中轨斜率都除以中轨（价格水平），镜像改变价格水平，
临界 K 的状态可能不同（path_a 3 根、path_c 1 根；上涨中同样宽度的带 bw 递减、下跌中递增，
「带宽扩张」对下跌略敏感）——这是定义本身的性质，见 BASE_FINAL_AUDIT.md 第 3 节。
信号、离场与场景关键 K 的状态逐根对称。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zr.costs.fees import option_order_fees
from zr.features.bars import ET
from zr.features.regime import CHANNEL, IGNITION, RANGE, indicators, regimes
from zr.research import patterns_v3 as pt
from zr.research import v4_decisive as v4d
from zr.research.sim_v3 import _BUY

V4 = v4d.load_v4()
CFG = v4d.state_cfg(V4, "C")
DAY = pd.Timestamp("2026-08-04").date()
N_PRE = 45  # 08:00–09:28 共 45 根盘前 K
P0 = 100.0


def mk(closes, vol=None, wick=0.03) -> pd.DataFrame:
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range(pd.Timestamp("2026-08-04 08:00", tz=ET), periods=len(c), freq="2min")
    df = pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, c) + wick,
            "low": np.minimum(o, c) - wick,
            "close": c,
            "volume": np.full(len(c), 1000.0) if vol is None else np.asarray(vol, float),
        },
        index=idx.tz_convert("UTC"),
    )
    df.index.name = "ts"
    return df


def mirror(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    m["open"], m["close"] = 2 * P0 - df["open"], 2 * P0 - df["close"]
    m["high"], m["low"] = 2 * P0 - df["low"], 2 * P0 - df["high"]
    return m


def run(df: pd.DataFrame):
    x = indicators(df, CFG, daily_reset=False, warmup="nan")
    reg = regimes(df, x, CFG, V4["period"])
    sig = v4d.hypothesis_signals(df, x, reg, CFG, V4)
    exits = v4d.exit_arrays(df, x, CFG)
    guards = v4d.guard_mask(df, reg, V4, {DAY})
    return x, reg, sig, exits, guards


def hm(df: pd.DataFrame, i: int) -> str:
    return df.index[i].tz_convert(ET).strftime("%H:%M")


def first_after(arr: np.ndarray, i: int) -> int | None:
    f = np.flatnonzero(arr[i + 1 :])
    return int(f[0]) + i + 1 if len(f) else None


def oriented(df, d):
    return df if d == 1 else mirror(df)


def exit_side(exits, rule, d):
    return exits[rule][0 if d == 1 else 1]


# ---------------- 场景路径 ----------------


def path_a() -> pd.DataFrame:
    pre = [P0 + 0.05 * np.sin(i / 2.0) for i in range(N_PRE)]
    k = np.arange(1, 26)
    up = list(P0 + 0.15 * k + 0.25 * np.sin(k * 1.1))  # 09:30–10:18 带波动的上行
    closes = pre + up + [up[-1] - 0.15, up[-1] + 0.05]  # 10:20 阴线（长下影）、10:22 阳线
    closes += [up[-1] + 0.05 + 0.1 * j for j in range(1, 6)]
    closes += [up[-1] + 0.55 - 0.35 * j for j in range(1, 8)]
    df = mk(closes)
    x = indicators(df, CFG, daily_reset=False, warmup="nan")
    r = N_PRE + len(up)  # 阴线：下影探到中轨上方 0.1 带半宽
    df.iloc[r, df.columns.get_loc("low")] = float(x["mid"].iloc[r] + 0.1 * x["hw"].iloc[r])
    return df


def path_b() -> pd.DataFrame:
    pre = [P0 + 0.05 * np.sin(i / 2.0) for i in range(N_PRE)]
    drop = [99.6, 99.0, 98.2, 97.3, 96.6, 96.2, 96.0, 96.1, 96.6, 97.2, 97.6, 97.8, 97.9, 98.0]
    return mk(pre + drop)


def path_c(confirm: bool = True) -> pd.DataFrame:
    i0 = np.arange(N_PRE)
    pre = list(P0 + 0.6 * np.sin(i0 / 3.0) + 0.01 * i0)  # 盘前活跃
    flat = list(100.4 + 0.03 * np.sin(np.arange(30) * 2.1))  # 09:30–10:28 窄幅横盘
    brk = [100.9, 101.05 if confirm else 100.40, 101.2, 101.1]
    closes = pre + flat + brk
    vol = np.full(len(closes), 1000.0)
    vol[N_PRE + len(flat)] = 4000.0
    return mk(closes, vol=vol, wick=0.02)


A_ENTRY, B_SIG, C_RAW = 71, 46, 75


# ---------------- (a) ----------------


@pytest.mark.parametrize("d", [1, -1])
def test_a_channel_pullback_e1_then_exit_c_prime(d):
    df = oriented(path_a(), d)
    x, reg, sig, exits, (time_ok, ok_l, ok_s) = run(df)
    i = A_ENTRY
    assert hm(df, i) == "10:22"
    assert reg["state"].iloc[i] == CHANNEL and reg["dir"].iloc[i] == d
    assert np.flatnonzero(sig["H2"] == d).tolist() == [i]
    assert not (sig["H2"] == -d).any()
    assert time_ok[i] and (ok_l if d == 1 else ok_s)[i]
    # 回踩低点（两根 K 低点的较低者）的位置：%b ∈ [0.45, 0.75]、距中轨 ≤ 0.2 带半宽
    lo, hi = df["low"].to_numpy(), df["high"].to_numpy()
    ext = min(lo[i - 1], lo[i]) if d == 1 else max(hi[i - 1], hi[i])
    lower, hw, mid = x["lower"].iloc[i], x["hw"].iloc[i], x["mid"].iloc[i]
    pb = (ext - lower) / (2 * hw)
    assert 0.45 <= (pb if d == 1 else 1 - pb) <= 0.75 and abs(ext - mid) <= 0.2 * hw
    # (c')：X5 跌破前一更高低点 102.896（10:16 那根确认的摆动低点；空头镜像为 97.104）
    j = first_after(exit_side(exits, "X5_higher_low", d), i)
    assert hm(df, j) == "10:42"
    level = 102.896 if d == 1 else 2 * P0 - 102.896
    c = df["close"].to_numpy()
    assert d * (c[j] - level) < 0 and d * (c[j - 1] - level) >= 0


# ---------------- (b) ----------------


@pytest.mark.parametrize("d", [-1, 1])
def test_b_opening_drop_ignition_second_bar_e10_then_exit_c(d):
    df = path_b() if d == -1 else mirror(path_b())
    x, reg, sig, exits, (time_ok, ok_l, ok_s) = run(df)
    st, sd = reg["state"].to_numpy(), reg["dir"].to_numpy()
    rth = np.flatnonzero(df.index.tz_convert(ET).strftime("%H:%M") >= "09:30")
    assert hm(df, rth[0]) == "09:30" and st[rth[0]] != IGNITION  # 第 1 根：带宽才开始扩张
    i = B_SIG
    assert hm(df, i) == "09:32" and st[i] == IGNITION and sd[i] == d
    pb = x["pctb"].to_numpy()
    assert (d * (pb[i - 1 : i + 1] - 0.5) > 0.5).all()  # 连续 2 根收在外轨之外
    assert np.flatnonzero(sig["H1"] != 0).tolist() == [i] and sig["H1"][i] == d
    assert time_ok[i] and (ok_l if d == 1 else ok_s)[i]
    j = first_after(exit_side(exits, "X4_inside_0.1_0.9", d), i)
    assert hm(df, j) == "09:46"
    assert 0.1 < pb[j] < 0.9 and not (0.1 < pb[j - 1] < 0.9)


# ---------------- (c) ----------------


@pytest.mark.parametrize("d", [1, -1])
def test_c_range_breakout_e11_and_h3_delayed_confirmation(d):
    df = oriented(path_c(), d)
    x, reg, sig, exits, (time_ok, ok_l, ok_s) = run(df)
    h = V4["hypotheses"]["H3"]
    raw = pt.entry_signal(
        df,
        x,
        reg,
        {"entry": "E11", "M": h["M"], "v": h["v"]},
        {"entries": {"E11": {"vol_window": h["vol_window"]}}},
    )
    i = C_RAW
    assert hm(df, i) == "10:30" and reg["state"].iloc[i - 1] == RANGE
    assert np.flatnonzero(raw).tolist() == [i] and raw[i] == d
    c = df["close"].to_numpy()
    lvl = (df["high"] if d == 1 else df["low"]).iloc[i - h["M"] : i]
    brk = lvl.max() if d == 1 else lvl.min()
    assert d * (c[i] - brk) > 0
    assert np.flatnonzero(sig["H3"]).tolist() == [i + 1] and sig["H3"][i + 1] == d
    assert hm(df, i + 1) == "10:32" and d * (c[i + 1] - brk) > 0
    assert time_ok[i + 1] and (ok_l if d == 1 else ok_s)[i + 1]
    # 下一根收回突破位以内 → 不确认
    df2 = oriented(path_c(confirm=False), d)
    _, _, sig2, _, _ = run(df2)
    assert not sig2["H3"].any()


# ---------------- (d) X1–X7 ----------------

T_TICK0 = pd.Timestamp("2026-08-04 10:00", tz=ET)


def ticks(shift_from: str | None = None, shift: float = 0.0) -> v4d.Ticks:
    """每秒一个报价：bid = 1.90 + 1e-5 × 秒数，ask = bid + 0.10；
    shift_from 起 bid/ask 同加 shift。"""
    s = np.arange(0, 2 * 3600)
    bid = 1.90 + 1e-5 * s
    if shift_from is not None:
        k = int((pd.Timestamp(f"2026-08-04 {shift_from}", tz=ET) - T_TICK0).total_seconds())
        bid[k:] += shift
    ts = v4d.ns(pd.DatetimeIndex(T_TICK0 + pd.to_timedelta(s, unit="s")).tz_convert("UTC"))
    return v4d.Ticks(ts, bid, bid + 0.10)


def px_at(tk: v4d.Ticks, t: pd.Timestamp, side: str) -> float:
    k = int(np.searchsorted(tk.ts, t.value, side="left"))
    return float(tk.ask[k] if side == "ask" else tk.bid[k])


def close_ts(df, i) -> pd.Timestamp:
    return df.index[i] + pd.Timedelta(minutes=V4["period"])


def expected_net(ask0, fills) -> float:
    """fills = [(张数, bid)]；费用走 zr.costs.fees（与回放里的 sell_fee 实现独立）。"""
    cash = -ask0 * 100 * V4["execution"]["qty"] - _BUY[V4["execution"]["qty"]]
    for n, b in fills:
        cash += n * b * 100 - float(option_order_fees(n, "Sell", b)["total"])
    return cash


FLAT = pd.Timestamp("2026-08-04 15:49", tz=ET)
LAG = pd.Timedelta(seconds=V4["execution"]["lag_s"])


@pytest.mark.parametrize("d", [1, -1])
@pytest.mark.parametrize(
    "rule, want",
    [
        ("X1_sma5", "10:34"),
        ("X2_shrink2", "10:24"),
        ("X3_mid2", "10:40"),
        ("X4_inside_0.1_0.9", "10:24"),
        ("X5_higher_low", "10:42"),
    ],
)
def test_d_structural_exit_trigger_bar_and_fill(rule, want, d):
    df = oriented(path_a(), d)
    x, _, _, exits, _ = run(df)
    i, c = A_ENTRY, df["close"]
    arr = exit_side(exits, rule, d)
    j = first_after(arr, i)
    assert hm(df, j) == want
    # 规则的独立复算（只对简单规则）
    if rule == "X1_sma5":
        sma = c.rolling(5).mean().to_numpy()
        assert d * (c.iloc[j] - sma[j]) < 0 and all(
            d * (c.iloc[k] - sma[k]) >= 0 for k in range(i + 1, j)
        )
    if rule == "X3_mid2":
        mid = c.rolling(20).mean().to_numpy()
        below = d * (c.to_numpy() - mid) < 0
        assert below[j] and below[j - 1] and not (below[j - 1] and below[j - 2])
    if rule == "X2_shrink2":
        hs = d * x["hist"].to_numpy()
        assert hs[j] < hs[j - 1] < hs[j - 2]
    tk = ticks()
    dec, trig = close_ts(df, i), close_ts(df, j)
    struct = v4d.ns(pd.DatetimeIndex(df.index[arr] + pd.Timedelta(minutes=V4["period"])))
    r = v4d.simulate(tk, dec.value, struct, FLAT.tz_convert("UTC").value, V4)
    ask0, b = px_at(tk, dec + LAG, "ask"), px_at(tk, trig + LAG, "bid")
    assert r["reason"] == "structure" and not r["tp_used"]
    assert r["entry_ns"] == (dec + LAG).value and r["exit_ns"] == (trig + LAG).value
    assert r["ask0"] == pytest.approx(ask0)
    assert r["net_usd"] == pytest.approx(expected_net(ask0, [(2, b)]))


def test_d_x6_take_profit_sells_one_then_structure():
    df = path_a()
    _, _, _, exits, _ = run(df)
    arr = exits["X5_higher_low"][0]
    struct = v4d.ns(pd.DatetimeIndex(df.index[arr] + pd.Timedelta(minutes=2)))
    tk = ticks("10:30:00", +0.60)  # mid ≈ 2.57 ≥ 1.2 × ask0
    dec = close_ts(df, A_ENTRY)
    r = v4d.simulate(tk, dec.value, struct, FLAT.tz_convert("UTC").value, V4)
    ask0 = px_at(tk, dec + LAG, "ask")
    t_tp = pd.Timestamp("2026-08-04 10:30:00", tz=ET)
    trig = close_ts(df, first_after(arr, A_ENTRY))
    b1, b2 = px_at(tk, t_tp + LAG, "bid"), px_at(tk, trig + LAG, "bid")
    assert r["tp_used"] and r["reason"] == "structure"
    assert r["net_usd"] == pytest.approx(expected_net(ask0, [(1, b1), (1, b2)]))


def test_d_x6_stop_loss_sells_all():
    df = path_a()
    _, _, _, exits, _ = run(df)
    arr = exits["X5_higher_low"][0]
    struct = v4d.ns(pd.DatetimeIndex(df.index[arr] + pd.Timedelta(minutes=2)))
    tk = ticks("10:30:00", -0.70)  # mid ≈ 1.28 ≤ 0.7 × ask0
    dec = close_ts(df, A_ENTRY)
    r = v4d.simulate(tk, dec.value, struct, FLAT.tz_convert("UTC").value, V4)
    t_sl = pd.Timestamp("2026-08-04 10:30:00", tz=ET)
    ask0, b = px_at(tk, dec + LAG, "ask"), px_at(tk, t_sl + LAG, "bid")
    assert r["reason"] == "sl" and r["exit_ns"] == (t_sl + LAG).value
    assert r["net_usd"] == pytest.approx(expected_net(ask0, [(2, b)]))


def test_d_x7_cap_30_minutes():
    df = path_a()
    tk = ticks()
    dec = close_ts(df, A_ENTRY)
    r = v4d.simulate(tk, dec.value, np.array([], np.int64), FLAT.tz_convert("UTC").value, V4)
    te = dec + LAG
    cap = te + pd.Timedelta(minutes=V4["cap_min"])
    assert r["reason"] == "cap" and r["exit_ns"] == (cap + LAG).value
    assert hm(df, A_ENTRY) == "10:22" and cap.tz_convert(ET).strftime("%H:%M:%S") == "10:54:02"
    ask0, b = px_at(tk, te, "ask"), px_at(tk, cap + LAG, "bid")
    assert r["net_usd"] == pytest.approx(expected_net(ask0, [(2, b)]))


def test_d_flat_beats_cap_when_earlier():
    df = path_a()
    tk = ticks()
    dec = close_ts(df, A_ENTRY)
    flat = pd.Timestamp("2026-08-04 10:40", tz=ET)
    r = v4d.simulate(tk, dec.value, np.array([], np.int64), flat.tz_convert("UTC").value, V4)
    assert r["reason"] == "flat" and r["exit_ns"] == (flat + LAG).value


@pytest.mark.parametrize("path, n_state_diff", [(path_a, 3), (path_b, 0), (path_c, 1)])
def test_mirror_swaps_every_signal_and_exit(path, n_state_diff):
    df = path()
    _, reg, sig, exits, _ = run(df)
    _, reg_m, sig_m, exits_m, _ = run(mirror(df))
    for h in sig:
        assert (sig[h] == -sig_m[h]).all(), h
    for rule in exits:
        lg, sh = exits[rule]
        lg_m, sh_m = exits_m[rule]
        assert (lg == sh_m).all() and (sh == lg_m).all(), rule
    same = reg["state"].to_numpy() == reg_m["state"].to_numpy()
    assert int((~same).sum()) == n_state_diff  # 带宽 / 斜率按价格水平归一化（见模块说明）
    assert (reg["dir"].to_numpy()[same] == -reg_m["dir"].to_numpy()[same]).all()
