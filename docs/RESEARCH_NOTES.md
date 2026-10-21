# 文献笔记

DOI 由负责人补。年份、期刊以记忆为准，标「未验证」的条目待核。

| # | 作者 | 年份 | 出处 | 一句话结论 | 进入系统的方式 | DOI |
|---|---|---|---|---|---|---|
| 1 | López de Prado, M. | 2018 | *Advances in Financial Machine Learning*, Wiley | 事件采样（CUSUM）、triple-barrier 标签、元标签、purge/embargo、样本唯一性加权是一套配套方法。 | E5 检测器；第 2 层标签；第 3 层模型与验证；第 7 层评估。 | |
| 2 | Bailey, D. H.; López de Prado, M. | 2014 | *Journal of Portfolio Management* 40(5) | 多次回测后需用 Deflated Sharpe 修正选择偏差。 | 第 7 层评估。 | |
| 3 | Gao, L.; Han, Y.; Li, S. Z.; Zhou, G. | 2018 | *Journal of Financial Economics* 129(2) | SPY 日内：首 30 分钟收益对末 30 分钟收益有正向预测力。 | E7 日内动量。 | |
| 4 | Baltussen, G.; Da, Z.; Lammers, S.; Martens, M. | 2021 | *Journal of Financial Economics* 141(3) | 指数期货末 30 分钟动量在多市场成立，与做市商 gamma 对冲相关。 | E7 日内动量；E8 gamma 条件的理论依据。 | |
| 5 | Dim, C.; Eraker, B.; Vilkov, G. | 2024 | 工作论文（**未验证**是否已发表） | 做市商净 gamma 为正时日内价格均值回归，为负时动量。 | E8 VWAP 均值回归的启用条件；GEX 特征。 | |

## 局限

- 上述文献多数基于 2023 年前数据。0DTE 期权在 2022–2023 年后占 SPX/SPY 期权成交量比例大幅上升，日内 gamma 结构已改变；E7 / E8 必须在 2024 年以后的数据上重验。
- 文献 3、4 研究指数（期货 / ETF），对 MU / AMD 等单一股票的适用性未验证。
- GEX 计算依赖做市商持仓方向假设，文献 5 本身承认该假设粗糙。
