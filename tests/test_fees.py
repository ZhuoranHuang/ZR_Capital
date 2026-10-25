from decimal import Decimal as D

from zr.costs.fees import option_order_fees, stock_order_fees


def test_option_buy_31_contracts_matches_rate_card_example():
    f = option_order_fees(31, "Buy", 0.50)
    assert f["OptionsCommission"] == D("13.95")
    assert f["OptionsPlatformFee"] == D("9.30")
    assert f["OptionsExchangeFee"] == D("5.58")
    assert f["OptionsRegulatoryFee"] == D("0.72")
    assert f["CATFee"] == D("0.01")
    assert f["OptionsClearingFee"] == D("0.78")
    assert "OptionsSFCFee" not in f
    assert f["total"] == D("30.34")


def test_option_minimums_small_orders():
    f1 = option_order_fees(1, "Buy", 2.0)
    assert f1["OptionsCommission"] == D("1.49")  # 0.45 < 1.49 最低
    assert f1["OptionsPlatformFee"] == D("0.30")
    assert f1["OptionsRegulatoryFee"] == D("0.03")  # 0.02295 向上进位
    assert f1["CATFee"] == D("0.01")
    assert f1["OptionsClearingFee"] == D("0.03")  # 0.025 → 0.03 (half up)
    assert f1["total"] == D("2.04")
    f3 = option_order_fees(3, "Buy", 2.0)
    assert f3["OptionsCommission"] == D("1.49")  # 1.35 < 1.49
    f4 = option_order_fees(4, "Buy", 2.0)
    assert f4["OptionsCommission"] == D("1.80")


def test_option_sell_adds_sec_and_taf():
    f = option_order_fees(2, "Sell", 1.25)
    # notional = 1.25*100*2 = 250 → SEC 0.00515 → 0.01 最低
    assert f["OptionsSFCFee"] == D("0.01")
    assert f["OptionsTradingActivityFee"] == D("0.01")  # 0.00658 → 0.01
    big = option_order_fees(10, "Sell", 50.0)  # notional 50000 → SEC 1.03
    assert big["OptionsSFCFee"] == D("1.03")
    assert big["OptionsTradingActivityFee"] == D("0.03")


def test_stock_fees():
    f = stock_order_fees(100, "Buy", 200.0)
    assert f["Commission"] == D("0")
    assert f["PlatformFee"] == D("1.00")  # max(0.5, 1.0) 且 < 0.0099*20000
    assert f["SettlementFee"] == D("0.30")
    assert f["total"] == D("1.30")
    cheap = stock_order_fees(10, "Sell", 0.5)  # notional 5
    assert cheap["PlatformFee"] == D("0.05")  # min(1.0, 0.0495)
    assert cheap["SettlementFee"] == D("0.03")  # min(0.03, 0.35)
    assert cheap["SECFee"] == D("0.01")
    assert cheap["TradingActivityFee"] == D("0.01")
    huge = stock_order_fees(100_000, "Sell", 10.0)
    assert huge["TradingActivityFee"] == D("9.79")


# ---------------- D-035 费用口径统一：向量化 / 转调实现与 D-013 逐值相等 ----------------

import numpy as np  # noqa: E402

_NS = (1, 2, 3, 5, 10, 20)
_PREMS = np.round(np.r_[0.0, 0.01, 0.05, 0.1, 0.33, 0.995, np.arange(0.01, 60.0, 0.37), 99.99], 2)


def test_vectorized_sell_fee_equals_d013():
    from zr.research.sim_v3 import sell_fee

    for n in _NS:
        got = sell_fee(np.full(len(_PREMS), n), _PREMS)
        exp = np.array([float(option_order_fees(n, "Sell", float(p))["total"]) for p in _PREMS])
        assert np.abs(got - exp).max() < 1e-9, n


def test_buy_table_equals_d013_any_n():
    from zr.research.sim_v3 import _BUY

    for n in _NS:
        for p in (0.05, 1.0, 25.0):
            assert abs(_BUY[n] - float(option_order_fees(n, "Buy", p)["total"])) < 1e-9


def test_intrabar_fees_equals_d013():
    from zr.backtest.intrabar import fees

    for n in (1, 3, 10):
        for p in (0.0, 1.05, 8.25, 36.55):
            for side in ("Buy", "Sell"):
                assert abs(fees(n, p, side) - float(option_order_fees(n, side, p)["total"])) < 1e-9


def test_research_stock_fee_equals_d013():
    from zr.research.popular_confluence import stock_fee

    for shares in (1, 100, 1000, 10000):
        for p in (0.95, 10.01, 90.0, 800.0):
            for side in ("Buy", "Sell"):
                exp = float(stock_order_fees(shares, side, p)["total"])
                assert abs(stock_fee(shares, side, p) - exp) < 1e-9
