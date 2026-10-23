"""纸面护栏助手（zr/live）：旁路、纸面离场单、护栏、风控、系统接管离场、kill switch、OMS 闸门。"""

from __future__ import annotations

import copy
import datetime as dt
import json
import types
from pathlib import Path

import pandas as pd
import pytest

from zr.execution import oms
from zr.features.bars import ET
from zr.features.regime import IGNITION, OTHER, RANGE
from zr.live import report
from zr.live.core import Entry
from zr.live.io import AccountPoller
from zr.live.paper import PaperSell, limit_px
from zr.live.runner import load_cfg, make_assistant
from zr.live.tap import TapReader, encode, ts_ns

DAY = dt.date(2026, 10, 22)
CALL = "MU261023C900000.US"
PUT = "MU261023P880000.US"
ZDTE = "QQQ261022C710000.US"
RULE = types.SimpleNamespace(price="mid", T=5.0)


def ns(hms: str) -> int:
    return int(pd.Timestamp(f"{DAY} {hms}", tz=ET).value)


# ---------------- tap ----------------


def test_tap_encode_filters_and_compacts():
    unds = {"MU.US"}
    t = encode(
        "MU.US",
        "trade",
        {
            "trades": [
                {"timestamp": "2026-10-22T14:00:01", "price": 900.0, "volume": 5, "trade_type": "I"}
            ]
        },
        1,
        unds,
    )
    assert t == {"r": 1, "s": "MU.US", "k": "t", "x": [["2026-10-22T14:00:01", 900.0, 5, "I"]]}
    d = encode(
        CALL,
        "depth",
        {"bids": [{"price": 1.0, "volume": 3}], "asks": [{"price": 1.1, "volume": 4}]},
        2,
        unds,
    )
    assert d["b"] == 1.0 and d["a"] == 1.1 and d["bs"] == 3 and d["as"] == 4
    assert encode("QQQ.US", "trade", {"trades": [{}]}, 3, unds) is None  # 不在 tap 标的里
    assert encode(CALL, "quote", {}, 4, unds) is None
    assert ts_ns("2026-10-22T14:00:01") == int(pd.Timestamp("2026-10-22 14:00:01", tz="UTC").value)


def test_tap_reader_keeps_partial_line(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n{"b":')
    r = TapReader(p)
    assert r.poll() == [{"a": 1}]
    with open(p, "a") as fh:
        fh.write("2}\n")
    assert r.poll() == [{"b": 2}]


# ---------------- 纸面离场单 ----------------


def test_paper_sell_stop_limit_trade_through_and_ladder():
    st = PaperSell(CALL, 2, "sl", "stop", 0, RULE)
    st.on_quote(1, 1.00, 1.20)
    assert st.done and st.avg_px == 1.00
    lm = PaperSell(CALL, 2, "tp", "limit", 0, RULE)
    lm.on_quote(1, 1.00, 1.20)
    assert lm.px == pytest.approx(limit_px(1.00, 1.20, "mid", CALL)) == pytest.approx(1.10)
    lm.on_trade(2, 1.15, 1)  # 成交价越过挂单价 → 1 张
    lm.on_quote(3, 1.10, 1.25, bid_size=5)  # 买一 = 挂单价 → 剩余
    assert lm.done and lm.avg_px == pytest.approx(1.10)
    ld = PaperSell(CALL, 1, "cap", "limit", 0, RULE)
    ld.on_quote(1, 1.00, 1.20)
    ld.on_quote(int(6e9), 1.00, 1.20)  # 超时 → 改价 中间价 − 1 tick = 1.09
    assert ld.stage == "reprice" and ld.px == pytest.approx(1.09)
    ld.on_quote(int(12e9), 0.98, 1.18)  # 再超时 → 按买一
    assert ld.done and ld.avg_px == pytest.approx(0.98)


# ---------------- 助手 ----------------


@pytest.fixture
def cfg():
    return copy.deepcopy(load_cfg())


def mk(cfg):
    msgs, logs = [], []
    a = make_assistant(cfg, DAY, msgs.append, logs.append)
    return a, msgs, logs


def rec(state=OTHER, d=0, **ex):
    flags = {
        k: (False, False) for k in ("X3_mid2", "X2_shrink2", "X4_inside_0.1_0.9", "X5_higher_low")
    }
    flags.update(ex)
    return {
        "ts": ns("10:00:00"),
        "state": state,
        "dir": d,
        "ex": flags,
        "completed_ns": ns("10:02:00"),
    }


def codes(a):
    return [x.code for x in a.alerts]


def test_guardrails_counter_ignition_before_first_bar_range(cfg):
    a, _, _ = mk(cfg)
    a.feeds["MU.US"].last = rec(IGNITION, -1)
    a.on_quote(ns("09:31:00"), CALL, 1.0, 1.1)
    a.on_entry(Entry(ns("09:31:00"), CALL, 2, 1.1))
    assert "逆点火开仓" in codes(a) and "首根 K 收盘前开仓" in codes(a)
    lv = {x.code: x.level for x in a.alerts}
    assert lv["逆点火开仓"] == "warn" and lv["首根 K 收盘前开仓"] == "warn"
    b, _, _ = mk(cfg)
    b.feeds["MU.US"].last = rec(RANGE, 0)
    b.on_entry(Entry(ns("10:40:00"), PUT, 2, 1.1))
    assert [(x.code, x.level) for x in b.alerts] == [("RANGE 开仓", "notice")]
    c, _, _ = mk(cfg)
    c.feeds["MU.US"].last = rec(IGNITION, -1)
    c.on_entry(Entry(ns("10:40:00"), PUT, 2, 1.1))  # 顺点火方向的看跌：不警告
    assert codes(c) == []


def test_zero_dte_late_notice_and_half(cfg):
    a, _, _ = mk(cfg)
    a.feeds["QQQ.US"].last = rec()
    a.on_quote(ns("11:00:00"), ZDTE, 1.0, 1.1)
    p = a.on_entry(Entry(ns("11:00:00"), ZDTE, 4, 1.1))
    assert "0DTE 10:15 后开仓" in codes(a)
    assert p.committed == 2 and p.orders[0].reason.startswith("0DTE")


def test_per_trade_cap_and_day_loss_block(cfg):
    cfg["paper_equity_usd"] = 1000
    a, _, _ = mk(cfg)
    a.feeds["MU.US"].last = rec()
    a.on_quote(ns("10:40:00"), CALL, 1.0, 1.1)
    a.on_entry(Entry(ns("10:40:00"), CALL, 2, 1.1))
    assert "超单笔上限" in codes(a)  # 220 > 1000 × 5%
    a.on_quote(ns("10:41:00"), CALL, 0.2, 0.3)  # 中间价跌 77% → −30% 止损全出
    assert a.positions[1].closed_ns is not None
    a.on_entry(Entry(ns("10:42:00"), CALL, 1, 0.3))  # 已亏 ≥ 100（10%）→ 禁止开仓，立即平仓
    assert "禁止开仓" in codes(a)
    assert a.positions[2].closed_ns is not None and a.positions[2].orders[0].kind == "stop"


def test_premium_rules_half_then_all(cfg):
    a, _, _ = mk(cfg)
    a.feeds["MU.US"].last = rec()
    p = a.on_entry(Entry(ns("10:40:00"), CALL, 2, 1.00))
    a.on_quote(ns("10:41:00"), CALL, 1.20, 1.30)  # 中间价 1.25 ≥ 1.2 → 卖 1（限价，挂 1.25）
    assert p.tp_done and p.committed == 1
    a.on_quote(ns("10:41:01"), CALL, 1.25, 1.30)
    assert p.sold == 1
    a.on_quote(ns("10:42:00"), CALL, 0.60, 0.70)  # 中间价 0.65 ≤ 0.7 → 剩余全出（按买一）
    assert p.closed_ns is not None and p.orders[-1].avg_px == 0.60
    assert p.realized() == pytest.approx(
        1.25 * 100
        + 0.60 * 100
        - 1.00 * 200
        - report.buy_fee(2, 1.0)
        - report.sell_fee(1, 1.25)
        - report.sell_fee(1, 0.60)
    )


def test_structure_macd_cap_zero_dte_and_flat(cfg):
    a, _, _ = mk(cfg)
    a.feeds["MU.US"].last = rec()
    p = a.on_entry(Entry(ns("10:40:00"), PUT, 4, 1.00))
    a.on_quote(ns("10:40:01"), PUT, 1.00, 1.04)
    a.on_bar("MU.US", rec(X2_shrink2=(False, True)))  # 看跌 → 空头侧：柱连续缩短 → 减半
    assert p.macd_done and p.committed == 2
    a.on_bar("MU.US", rec(X2_shrink2=(False, True)))
    assert p.committed == 2  # 只减一次
    a.on_bar(
        "MU.US", {**rec(X5_higher_low=(False, True)), "completed_ns": ns("10:44:00")}
    )  # 结构止损：剩余全出（止损类）
    assert p.committed == 4 and p.orders[-1].kind == "stop" and p.orders[-1].done
    b, _, _ = mk(cfg)
    q = b.on_entry(Entry(ns("10:40:00"), CALL, 1, 1.00))
    b.on_quote(ns("10:40:01"), CALL, 1.00, 1.04)
    b.on_clock(ns("11:10:01"))
    assert q.orders and q.orders[0].reason == "持仓 30 分钟上限"
    c, _, _ = mk(cfg)
    z = c.on_entry(Entry(ns("15:20:00"), ZDTE, 1, 1.00))
    c.on_quote(ns("15:20:01"), ZDTE, 1.00, 1.04)
    c.on_clock(ns("15:30:00"))
    assert z.orders[0].reason == "0DTE 15:30 前清仓"
    d, _, _ = mk(cfg)
    f = d.on_entry(Entry(ns("15:45:00"), CALL, 1, 1.00))
    d.on_quote(ns("15:45:01"), CALL, 1.00, 1.04)
    d.on_clock(ns("15:50:00"))
    assert f.orders[0].reason == "15:50 全部清仓" and f.closed_ns is not None


def test_kill_switch_flattens_everything(cfg):
    a, msgs, _ = mk(cfg)
    for s in (CALL, PUT):
        a.on_quote(ns("10:40:00"), s, 1.0, 1.1)
        a.on_entry(Entry(ns("10:40:00"), s, 2, 1.1))
    a.kill(ns("10:41:00"))
    assert a.killed and all(p.closed_ns is not None for p in a.positions.values())
    assert any("kill switch" in m for m in msgs)


def test_daily_report_renders(cfg):
    a, _, _ = mk(cfg)
    a.on_quote(ns("10:40:00"), CALL, 1.0, 1.1)
    a.on_entry(Entry(ns("10:40:00"), CALL, 2, 1.1))
    a.on_owner_sell(ns("10:50:00"), CALL, 2, 1.3)
    text = report.build(a, DAY)
    assert "入场笔数：1" in text and CALL in text


def test_oms_refuses_without_gate_and_even_with_it(monkeypatch):
    monkeypatch.delenv("ZR_LIVE", raising=False)
    with pytest.raises(oms.LiveGateError):
        oms.submit_order(cli_live=True)
    monkeypatch.setenv("ZR_LIVE", "1")
    with pytest.raises(oms.LiveGateError):
        oms.submit_order(cli_live=False)
    with pytest.raises(NotImplementedError):
        oms.submit_order(cli_live=True)


def test_account_poller_detects_entries_and_owner_sells(tmp_path: Path):
    class Ord:
        def __init__(self, oid, sym, side, q, px):
            self.order_id, self.symbol, self.side = oid, sym, side
            self.executed_quantity, self.executed_price = q, px
            self.updated_at = dt.datetime(2026, 10, 22, 14, 40, 0)

    orders = [Ord("1", CALL, "Buy", 2, 1.1), Ord("2", "MU.US", "Buy", 10, 900.0)]
    t = types.SimpleNamespace(today_orders=lambda: orders)
    seen = tmp_path / "seen.json"
    p = AccountPoller(t, seen)
    e, s = p.poll()
    assert [(x.symbol, x.qty, x.price) for x in e] == [(CALL, 2, 1.1)] and s == []
    assert p.poll() == ([], [])  # 不重复
    orders.append(Ord("3", CALL, "Sell", 1, 1.3))
    e, s = AccountPoller(t, seen).poll()  # 重启后读 seen 文件
    assert e == [] and s[0][1:] == (CALL, 1, 1.3)
    assert json.loads(seen.read_text())["1"] == 2


def test_incremental_engine_daily_reset_matches_batch_convention_a():
    """口径 A（可切换）：增量引擎 daily_reset=True 与批量 indicators(daily_reset=True) 逐根一致。"""
    import numpy as np

    from zr.features.bars import resample_bars
    from zr.features.regime import bars_per_day, indicators, regimes
    from zr.research import v4_decisive as v4d
    from zr.strategy.incremental import Bar, IncrementalEngine

    df = pd.read_csv(
        Path(__file__).parent / "fixtures" / "mu_2026_10_21_min1_rth.csv", parse_dates=["ts"]
    )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    b = resample_bars(df.set_index("ts").sort_index(), "2min")
    v4 = v4d.load_v4()
    cfg_a = v4d.state_cfg(v4, "A")
    x = indicators(b, cfg_a, daily_reset=True, warmup="nan")
    reg = regimes(b, x, cfg_a, 2)
    eng = IncrementalEngine(cfg_a, v4, max(1, round(bars_per_day(b))), set(), daily_reset=True)
    recs = [
        eng.on_bar(Bar(int(t), o, h, lo, c, v))
        for t, o, h, lo, c, v in zip(
            v4d.ns(b.index), b.open, b.high, b.low, b.close, b.volume, strict=True
        )
    ]
    assert [r["state"] for r in recs] == reg["state"].tolist()
    assert np.allclose([r["mid"] for r in recs], x["mid"], equal_nan=True)
    assert np.isnan(recs[18]["mid"]) and np.isfinite(recs[19]["mid"])


def test_tap_writer_from_recorder_handle(tmp_path: Path, monkeypatch):
    """录制器回调多写一行；不在 tap 标的里的不写；旧文件只保留 3 天。"""
    from zr.data.recorder import Recorder
    from zr.live.tap import TapWriter, tap_path

    now = dt.datetime(2026, 10, 22, 10, 0, tzinfo=ET)
    old = tmp_path / "2026-10-18.jsonl"
    old.write_text("x\n")
    tap = TapWriter(tmp_path, {"MU.US"}, now_fn=lambda: now)
    assert not old.exists()
    sink = types.SimpleNamespace(add=lambda row: None)
    rec = Recorder(types.SimpleNamespace(), sink, {}, 10, tap=tap)
    payload = {
        "trades": [
            {"timestamp": "2026-10-22T14:00:00", "price": 900.0, "volume": 1, "trade_type": ""}
        ]
    }
    monkeypatch.setattr("zr.data.recorder.to_json", lambda e: json.dumps(payload))
    ev = types.SimpleNamespace(timestamp=None, trades=[])
    rec._handle("trade", "MU.US", ev)
    rec._handle("trade", "QQQ.US", ev)
    tap.close()
    lines = tap_path(tmp_path, now.date()).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["s"] == "MU.US"
    assert rec.callback_errors == 0 and tap.errors == 0


def test_restart_restores_positions_from_log(cfg):
    a, _, logs = mk(cfg)
    a.on_quote(ns("10:40:00"), CALL, 1.0, 1.1)
    a.on_entry(Entry(ns("10:40:00"), CALL, 2, 1.0))
    a.on_quote(ns("10:41:00"), CALL, 1.25, 1.30)  # +20% → 卖 1（限价 1.28）
    a.on_quote(ns("10:41:01"), CALL, 1.28, 1.30)
    rows = json.loads(json.dumps(logs, default=str))
    b, _, _ = mk(cfg)
    assert b.restore(rows) == 1
    p = b.positions[1]
    assert p.open_qty == 1 and p.tp_done and p.closed_ns is None
    assert p.realized() == pytest.approx(a.positions[1].realized())
