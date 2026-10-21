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
