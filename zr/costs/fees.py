"""长桥（香港主体）美股 / 美股期权费用，纯函数。

费率来自负责人提供的账户费率卡（2026-10-21）；按订单计费的假设由
`zr/costs/validate_fees.py` 用 218 笔真实成交验证（结论见 docs/DECISIONS.md D-013）。

键名与长桥 `charge_detail` 的 fee code 一致，便于逐项对照。
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

Side = str  # "Buy" | "Sell"

D = Decimal
CENT = D("0.01")


def _r2(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def _ceil2(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_CEILING)


def option_order_fees(n_contracts: int, side: Side, premium: float | Decimal) -> dict[str, Decimal]:
    """单笔期权订单费用明细（USD）。premium 为每股权利金（合约乘数 100）。"""
    if n_contracts <= 0:
        raise ValueError("n_contracts 必须为正")
    n = D(n_contracts)
    prem = D(str(premium))
    fees: dict[str, Decimal] = {
        "OptionsCommission": max(_r2(D("0.45") * n), D("1.49")),
        "OptionsPlatformFee": max(_r2(D("0.30") * n), D("0.30")),
        "OptionsExchangeFee": _r2(D("0.18") * n),
        "OptionsRegulatoryFee": _ceil2(D("0.02295") * n),
        "CATFee": max(_r2(D("0.0003") * n), CENT),
        "OptionsClearingFee": _r2(D("0.025") * n),
    }
    if side == "Sell":
        notional = prem * D(100) * n
        fees["OptionsSFCFee"] = max(_r2(D("0.0000206") * notional), CENT)
        fees["OptionsTradingActivityFee"] = max(_r2(D("0.00329") * n), CENT)
    fees["total"] = sum(fees.values(), D(0))
    return fees


def stock_order_fees(shares: int, side: Side, price: float | Decimal) -> dict[str, Decimal]:
    """单笔美股订单费用明细（USD）。佣金终生免。"""
    if shares <= 0:
        raise ValueError("shares 必须为正")
    s = D(shares)
    px = D(str(price))
    notional = s * px
    fees: dict[str, Decimal] = {
        "Commission": D(0),
        "PlatformFee": _r2(min(max(D("0.005") * s, D("1.0")), D("0.0099") * notional)),
        "SettlementFee": _r2(min(D("0.003") * s, D("0.07") * notional)),
    }
    if side == "Sell":
        fees["SECFee"] = max(_r2(D("0.0000206") * notional), CENT)
        fees["TradingActivityFee"] = min(max(_r2(D("0.000195") * s), CENT), D("9.79"))
    fees["total"] = sum(fees.values(), D(0))
    return fees
