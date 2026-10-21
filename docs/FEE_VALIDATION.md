# 费用公式验证

订单 218 笔（有成交），其中多笔成交订单 41 笔。
残差 = 实际 − 公式。脚本：`uv run python -m zr.costs.validate_fees`。

## H1 按订单计最低收费（n = 整单成交张数）

| 项目 | 残差=0 的订单 | 残差均值 | 残差绝对值和 | 最大残差 |
|---|---|---|---|---|
| OptionsCommission | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsPlatformFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsExchangeFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsRegulatoryFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| CATFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsClearingFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsSFCFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsTradingActivityFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| total | 218/218 | 0.0000 | 0.00 | 0.00 |

## H2 按成交笔数各自计最低收费后求和

| 项目 | 残差=0 的订单 | 残差均值 | 残差绝对值和 | 最大残差 |
|---|---|---|---|---|
| OptionsCommission | 178/218 | -0.3483 | 75.94 | 5.20 |
| OptionsPlatformFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsExchangeFee | 218/218 | 0.0000 | 0.00 | 0.00 |
| OptionsRegulatoryFee | 182/218 | -0.0022 | 0.47 | 0.03 |
| CATFee | 177/218 | -0.0032 | 0.70 | 0.04 |
| OptionsClearingFee | 195/218 | -0.0012 | 0.26 | 0.02 |
| OptionsSFCFee | 205/218 | -0.0003 | 0.15 | 0.02 |
| OptionsTradingActivityFee | 195/218 | -0.0012 | 0.27 | 0.02 |
| total | 177/218 | -0.3565 | 77.71 | 5.29 |

## 多笔成交订单上的对比

- H1 total 残差绝对值和：0.00
- H2 total 残差绝对值和：77.71
