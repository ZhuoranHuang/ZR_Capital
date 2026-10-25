"""成本报告纯函数：往返成本、门槛、跨式减半、缺失分钟与低价剔除。"""

from __future__ import annotations

import numpy as np
import pytest

from zr.costs.fees import option_order_fees
from zr.research.cost_report import changes, cost_rt, hold_leg, threshold_pct, valid_quotes


@pytest.mark.parametrize("n", [1, 2, 5])
def test_cost_rt_matches_option_order_fees(n):
    bid, ask = 1.20, 1.25
    want = (
        (ask - bid) * 100 * n
        + float(option_order_fees(n, "Buy", ask)["total"])
        + float(option_order_fees(n, "Sell", bid)["total"])
    )
    assert cost_rt(bid, ask, n) == pytest.approx(want, abs=1e-9)


def test_threshold_example():
    # delta 0.5、S 100、每张成本 1.00 美元 → 1 / (0.5 × 100 × 100) × 100 = 0.02%
    # 题设「2.00%」对应每股口径（mult = 1）；这里两种口径都验证
    per_share = threshold_pct(np.array([1.00]), np.array([0.5]), np.array([100.0]), mult=1)
    per_contract = threshold_pct(np.array([100.0]), np.array([-0.5]), np.array([100.0]))
    assert per_share[0] == pytest.approx(2.0)
    assert per_contract[0] == pytest.approx(2.0)


def test_straddle_halving():
    assert hold_leg(-2.00) == pytest.approx(-1.00)
    assert hold_leg(np.array([-2.00]))[0] == pytest.approx(-1.00)


def test_missing_minute_dropped_and_counted():
    bid = np.array([1.0, np.nan, 1.0, 1.0])
    ask = np.array([1.1, np.nan, 1.1, 1.2])
    v, c = valid_quotes(bid, ask)
    assert c["missing"] == 1 and c["valid"] == 3
    mid = (bid + ask) / 2
    chg, dropped = changes(mid, v, 1)
    # t=0 的 t+1 缺失 → 剔除（不用邻近分钟填充）；t=3 无 t+1 也剔除
    assert np.isnan(chg[0]) and np.isnan(chg[1])
    assert chg[2] == pytest.approx(0.05)
    assert dropped == 2


def test_low_mid_dropped_and_counted():
    v, c = valid_quotes(np.array([0.01, 0.04, 0.0, 0.5]), np.array([0.03, 0.05, 0.05, 0.6]))
    assert c["lowmid"] == 2  # mid 0.02、0.045
    assert c["bid0"] == 1 and c["invalid"] == 1
    assert v.tolist() == [False, False, False, True]
