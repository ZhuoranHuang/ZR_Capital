# 公开策略调研（第二条，2026-10-23）

**受欢迎不等于有效，所有候选待样本外验证。** 本文只整理公开策略的规则、分类与「怎么判断趋势」，没有回测任何一条；
候选规则（`config/intrabar/survey_*.yaml`）在第三条 R-049 里按预登记做样本外检验。
负责人截图里的表现（5–7 笔、胜率 100%、月收益 8%–28%）不作为任何证据（见第 5 节）。

计数与表格由 `uv run python -m zr.research.strategy_survey` 从 `docs/research/strategy_survey_sources.yaml` 生成
（`<!-- AUTO -->` 标记之间的内容会被覆盖）；其余是手写正文。

## 1. 结论

- **分类**：收录 53 条公开策略 / 指标（分布见第 3 节）。趋势突破最多，其中**开盘区间突破**（8 条）与 **N 根通道突破**（7 条）
  是两大子类，二者只差 1 条；顺势回调里最常见的是**均线回踩再收回**（5 条）。
- **趋势怎么判断**：最常见的不是一种「方向」定义，而是「方向 + 强度」两件套——方向用「价在长均线一侧 / 均线排列 / 通道突破 /
  开盘区间方向」，强度用 **ADX（阈值 20–25）**；ADX 是全表使用最多的趋势判断方式（第 4 节）。高低点结构、高一级周期确认很少见。
- **对应负责人三类情形**：点火 ↔ 通道 / 开盘区间 / ATR 跟踪线的突破，常配 ADX 或挤压释放；通道回调 ↔ 均线排列定方向后回到快线
  （EMA20/50）或短周期超卖（RSI(2) < 10）再恢复；回避震荡 ↔ ADX 门槛、挤压（BOLL 在 KC 内）不交易。没有一条公开策略
  用「BOLL 中轨斜率 + %b」这一类负责人在 App 上看的组合来定义通道状态（掘金 BOLL_STOCK 的「三轨同向」最接近）。
- **三个截图策略**：原文都没找到。NSGA-II-MACD 只找到同名方法的学术论文（arXiv 2111.13364）；RSI 超卖反弹对应 Connors RSI(2)
  （日线规则，已给盘中改写与空头镜像）；Keltner 触边回归对应 StockSharp 样例（均值回归，与负责人「顺着 30–120 秒的走势入场」相反）。
- **候选**：8 条对称规则（第 6 节），R-049 必测 3 条：盘中 RSI(2) 改写、最常见的突破（开盘区间突破）、最常见的回调（EMA 回踩）。
  全部需要引擎新增特征（`features_needed`），参数一律照搬原策略默认值，未调参。
- **宣称的表现**：TradingView 热门脚本几乎都不给数字；给了数字的里，能找到独立复现的两条都在扣成本后变弱或消失
  （QQQ 5 分钟 ORB：MQL5 博客复现毛利 +0.131R、扣成本后 5 个市场无一显著；SPY 噪声区动量：本仓库 R-045 主口径成本下 SPY −5.35%）；
  Connors RSI(2) 在 SPX 日线上十年 +18.1%，远低于买入持有 +255.4%（Backtrex，第三方）。

## 2. 范围与边界

- **采集日期**：2026-10-23；热度是当天读数的快照。
- **TradingView**：「按热度排序的策略列表」页面是前端渲染，抓不到列表；改用站内搜索接口 `pubscripts-suggest-json`，按 44 个关键词
  （supertrend、macd、rsi 2、keltner、adx、pullback、breakout、vwap、squeeze、opening range、donchian、nsga、genetic algorithm 等）
  取结果，按点赞数挑趋势 / 回调 / 截图相关条目；源码由同一接口或 `pine-facade` 接口取得并逐条阅读（22 条全部读了源码）。
  **选样偏差**：搜索接口每词只返回有限条，热度高但标题不含关键词的脚本可能漏掉。
- **GitHub（≥ 500 star）**：freqtrade-strategies（5588）、QuantDinger（OpenByteInc，13056）、myquant/strategy（掘金官方，956）、
  je-suis-tm/quant-trading（11109，用于 Dual Thrust 规则）、alpacahq/example-scalping（852）、StockSharp（10972）；
  GitHub 搜索「intraday / keltner / adx / opening range / supertrend strategy」且 star ≥ 500 的仓库几乎没有独立策略库，
  大部分高 star 项目是回测框架（backtrader 23555、Lean 22175 等），框架自带样例只有均线交叉，未单列。
- **MQL5**：Market 的销量 / 排名不公开；搜索到的高曝光 EA 多为黄金网格 / 马丁（不在本调研范围，规则也不公开）。
  只收了 3 个公开描述了趋势 / 回调规则的 EA 与 1 篇文章，**少于 5 条**；规则来自搜索摘要（`secondary`），未逐页核对。
- **中文平台**：QuantDinger 收官方模板（读了 SQL 种子里的 Python 源码）；掘金收官方策略库 7 条（读了 info.md 说明）。
  **聚宽未访问**：经 VPS 出口访问 joinquant.com 返回「当前地区暂不支持访问」（非中国大陆 IP 被拒），高收藏策略未取得。
- **不收录**：只做一边且无法镜像的、网格 / 马丁、机器学习黑箱、多标的轮动 / 选股（单标的 MU 用不上）。
- 「每类 5–10 个」：TradingView 22 条、GitHub 22 条超过上限，是为了让五个入场类别都有样本；MQL5 4 条、论文与书 5 条。

<!-- AUTO:platform_counts -->
| platform | 条目 |
|---|---|
| TradingView | 22 |
| GitHub/freqtrade-strategies | 8 |
| GitHub/myquant(掘金) | 7 |
| GitHub/QuantDinger | 4 |
| MQL5 Market | 3 |
| GitHub/StockSharp | 2 |
| 论文/SSRN | 2 |
| GitHub/alpacahq | 1 |
| MQL5 Articles | 1 |
| QuantConnect | 1 |
| 书籍/StockCharts | 1 |
| 论文/arXiv | 1 |

读到的内容（pine/python/sql = 读了源码；desc = 作者描述；secondary = 二手摘要）：

| read | 条目 |
|---|---|
| desc | 9 |
| pine | 22 |
| python | 11 |
| secondary | 7 |
| sql | 4 |
<!-- /AUTO:platform_counts -->

## 3. 分类统计

按入场逻辑分五类：趋势突破（价格离开一个区间 / 通道 / 跟踪线）、顺势回调（先定方向，等逆向回撤结束再顺向入场）、
均值回归（押回到中枢）、多指标共振（多个指标同向才入场，入场时刻由某个交叉决定）、波动率过滤（以波动率收缩 / 扩张作为主条件）。

<!-- AUTO:category_counts -->
| category（代码） | 类别 | 条目 | 多空对称 | 只做一边（已给镜像） |
|---|---|---|---|---|
| breakout | 趋势突破 | 24 | 20 | 4 |
| pullback | 顺势回调 | 9 | 7 | 2 |
| mean_reversion | 均值回归 | 5 | 4 | 1 |
| confluence | 多指标共振 | 11 | 6 | 5 |
| vol_filter | 波动率过滤 | 4 | 4 | 0 |
| 合计 |  | 53 | 41 | 12 |
<!-- /AUTO:category_counts -->

子类（family，同一入场机制）：

<!-- AUTO:family_counts -->
| 类别 | family | 条目 |
|---|---|---|
| 趋势突破 | opening_range_breakout | 8 |
| 趋势突破 | channel_breakout | 7 |
| 趋势突破 | atr_trail_flip | 3 |
| 趋势突破 | ma_cross | 3 |
| 趋势突破 | ma_slope_flip | 1 |
| 趋势突破 | range_expansion | 1 |
| 趋势突破 | structure_break | 1 |
| 多指标共振 | macd_cross_filter | 4 |
| 多指标共振 | adx_di | 3 |
| 多指标共振 | ma_cross_filter | 2 |
| 多指标共振 | band_slope | 1 |
| 多指标共振 | optimized_combo | 1 |
| 均值回归 | band_revert | 4 |
| 均值回归 | ma_revert | 1 |
| 顺势回调 | ma_pullback | 5 |
| 顺势回调 | oscillator_pullback | 3 |
| 顺势回调 | macd_pullback | 1 |
| 波动率过滤 | squeeze_release | 3 |
| 波动率过滤 | breakout_confirm | 1 |
<!-- /AUTO:family_counts -->

<!-- AUTO:most_common -->
最常见的趋势突破 family：`opening_range_breakout`；最常见的顺势回调 family：`ma_pullback`（按上表条目数，代码判定）。
<!-- /AUTO:most_common -->

常见参数（从来源表归纳，见第 8 节逐条）：
- 周期：TradingView 脚本多数不指定周期；freqtrade 以 1 小时为主（1 分钟、5 分钟、4 小时各有）；掘金 / QuantDinger 以日线为主，
  日内只有开盘区间类（Hans123、SkyPark、Dual Thrust、R-Breaker）；论文 ORB 用 5 分钟。**真正为分钟级股票设计的规则只有开盘区间类与
  噪声区动量**，其余都是把日线 / 小时线规则拿到任意周期。
- 长均线：EMA / SMA 200（ChartArt、Connors、MACD Pullback Sniper、ADX|DMI Trend、TrendRider、MQL5 Donchian）；快线 EMA 20 / 50。
- ADX：14 期，阈值 20（QuantDinger、Sniper、Coral、TrendRider）或 23–25（ADXMomentum、ADX|DMI、hlhb、millerrh）。
- 通道：Donchian 20（出场 10）；KC 多为 EMA / SMA 20 ± 1.5–2.0 × ATR；BOLL(20, 2)。
- MACD：几乎都是 12/26/9（负责人 App 是 6/13/6）。
- ATR 跟踪：SuperTrend ATR 10 × 3，UT Bot ATR 10 × 1。

## 4. 趋势怎么判断

<!-- AUTO:trend_def_counts -->
| 趋势判断方式 | trend_def | 使用条目 | 趋势突破 | 顺势回调 | 均值回归 | 多指标共振 | 波动率过滤 |
|---|---|---|---|---|---|---|---|
| ADX / DI | adx | 16 | 2 | 6 | 0 | 6 | 2 |
| 动量 / MACD 符号 | momentum | 11 | 2 | 0 | 0 | 7 | 2 |
| 通道（Donchian / Keltner / BOLL / VWAP 带） | channel | 10 | 7 | 0 | 0 | 1 | 2 |
| 价在长均线一侧 | price_vs_ma | 10 | 4 | 4 | 0 | 2 | 0 |
| 开盘区间方向 | opening_range | 8 | 8 | 0 | 0 | 0 | 0 |
| 均线排列 | ma_alignment | 8 | 1 | 4 | 0 | 3 | 0 |
| 均线斜率 | ma_slope | 5 | 2 | 1 | 1 | 1 | 0 |
| 不判断趋势 | none | 4 | 0 | 0 | 4 | 0 | 0 |
| ATR 跟踪线状态 | atr_trail | 3 | 3 | 0 | 0 | 0 | 0 |
| 高一级周期确认 | htf | 2 | 0 | 2 | 0 | 0 | 0 |
| 高低点结构 | swing_structure | 1 | 1 | 0 | 0 | 0 | 0 |

同时用两种及以上趋势判断方式的条目：19 / 53。
<!-- /AUTO:trend_def_counts -->

归纳（数字见上表）：

1. **方向与强度分开判断**。方向：价在长均线一侧、均线排列、通道 / 开盘区间突破方向、ATR 跟踪线状态；强度：ADX（+DI / −DI 也给方向）。
   ADX 在顺势回调与多指标共振里尤其普遍——回调策略需要先确认「有趋势」再去买回撤。
2. **均线**：价在 200 均线一侧（慢、稳定）定大方向；EMA20 / 50 作为回调的「回踩位」；均线斜率（QuantDinger 三线、Hull、Coral）比排列少。
3. **通道**：突破策略把通道当作「趋势开始」的触发（Donchian、KC 上轨收盘、VWAP 1σ 带、噪声区），均值回归把同一条通道当作「过度」的信号——
   **同一个 Keltner 上轨，趋势派做多、回归派做空**（WunderTrading vs StockSharp），区别只在有没有方向 / 强度过滤。
4. **高低点结构**只有 LonesomeTheBlue 一条（收盘越过最近的 Lower High / Higher Low），而且 pivot 要右侧 5 根确认，实时只能滞后知道。
5. **高一级周期确认**只在 TrendRider（4 小时 / 日线）与 TrendingGoldea（H1）出现。
6. **开盘区间方向**是分钟级股票专属的趋势定义：第一段 5 分钟 / 30 分钟的方向即当日方向。

与负责人三类情形的对应：

<!-- AUTO:situation_counts -->
| 负责人情形 | 条目 | 按类别 |
|---|---|---|
| 点火 | 36 | 趋势突破 24，多指标共振 8，波动率过滤 4 |
| 通道回调 | 11 | 顺势回调 9，趋势突破 1，多指标共振 1 |
| 回避震荡 | 14 | 多指标共振 5，波动率过滤 4，均值回归 2，顺势回调 2，趋势突破 1 |
| 不对应 | 4 | 均值回归 3，多指标共振 1 |
<!-- /AUTO:situation_counts -->

- **点火（IGNITION）**：公开策略的对应物是「离开区间」——开盘区间突破、N 根通道突破、KC 上轨收盘、ATR 跟踪线翻转、挤压释放。
  共同点是**触发看价格越过最近的极值或波动包络**，过滤看 ADX / 成交量放大（millerrh 要求量 > 2 倍均量、TORB 的量能线）。
  负责人的入场前 30–120 秒有顺向移动（PRE_ENTRY_MICRO.md：ret_60s 负责人中位数 11.1 bp、自身随机百分位均值 0.61、t = 2.75），
  且离顺向极值近、离逆向极值远，与「越过极值」类触发方向一致。
- **通道回调（CHANNEL 回踩）**：对应顺势回调——先用均线排列 / 长均线一侧定方向，等价格回到快线（EMA20 / 50，或 low ≤ EMA × 1.02）、
  或短周期振荡器到极端（RSI(2) < 10、随机 < 30），再等**收回快线 / 阳线确认**才入场。Coral Pullback 把它写得最完整：
  先有一根整根在线上 → 回调收在线下 → 回调期间趋势线不翻色 → 再收回。
- **回避震荡（RANGE）**：公开策略几乎不显式识别「震荡」，而是用门槛把它挡在外面：ADX < 20–25 不做、ADX 需上升（millerrh、掘金 ADX_DMI）、
  挤压（BOLL 在 KC 内）期间不做、等释放；QuantDinger Resonance 用 RSI 上限 75 防追高。本仓库 V4 的 RANGE 打分（带宽、|中轨斜率|、|MACD 柱|
  的近 1 日分位）在公开策略里没有对应物。

## 5. 负责人截图里的三个策略

三张截图本身不在仓库里（`docs/` 与代码中检索「非支配 / NSGA / Keltner / 超卖反弹」无结果），以下只按负责人给的策略名检索原文；
**原文都没有找到**，下面是能找到的最接近的公开规则。

**截图声称的表现（5–7 笔、100% 胜率、月收益 8%–28%）不能作为证据。** 即使 7 笔全部盈利，真实胜率的 95% 单侧下界也只有约 0.65：

<!-- AUTO:screenshot_ci -->
| 截图声称笔数 n | 全部盈利时胜率 95% 单侧下界（Clopper–Pearson） |
|---|---|
| 5 | 0.55 |
| 6 | 0.61 |
| 7 | 0.65 |
<!-- /AUTO:screenshot_ci -->

这还没考虑：平台展示的是被挑出来的那几个（幸存者偏差）、参数是在同一段数据上优化的、月收益按几笔外推。

### 5.1 「非支配遗传算法-MACD 复合量化策略（趋势跟踪）」

- **原文**：TradingView 搜索「nsga」「genetic algorithm」「multi objective」「optimized macd」、中文网页搜索「非支配遗传算法 MACD 复合」均未找到
  同名策略。最接近的是方法论文 Prasad et al., *Optimal Technical Indicator-based Trading Strategies Using NSGA-II*（arXiv 2111.13364）：
  用 NSGA-II 在技术指标组合里搜索买卖规则，两个目标是**最大化 Sharpe、最小化最大回撤**，滚动 2 年训练 / 1 年测试，含交易成本；
  作者自述在**无大幅经济波动的平稳期**效果明显更好。另有单目标 GA 调 MACD 参数的研究（只见第三方书评，未读原文，未验证）。
- **过滤条件**：截图里的具体过滤条件没有入库，无法逐条记录（**待负责人提供截图或原文链接**）。公开同类 MACD 复合策略最常用的过滤是：
  长均线一侧（SMA / EMA 200，ChartArt、Sniper）、ADX(14) > 20（QuantDinger Resonance、Sniper）、RSI(14) 区间 50–75（Resonance）、
  KDJ 确认且 K < 85（QuantDinger MACD+KDJ）、零轴（掘金 MACD-STOCK：DIF / DEA 同在零轴一侧才交叉入场）。
- **过拟合风险：高**。(1) 多目标遗传算法在「指标 × 参数 × 过滤」的组合空间里搜索，有效试验次数通常数以千计且不公开，无法做多重检验校正
  （本项目对类似情况用 BH / Deflated Sharpe）；(2) 帕累托前沿上的点是事后挑选的；(3) 优化目标含最大回撤，会偏好在样本内恰好躲过大跌的规则；
  (4) 论文作者自己承认在剧烈波动期失效；(5) 5–7 笔的展示样本无法区分规则与运气。
- **候选**：`survey_macd_trend_filter.yaml` 用公开默认参数（MACD 12/26/9 柱过零 + EMA200 一侧 + ADX > 20），**不使用任何 GA 优化出的参数**。

### 5.2 「RSI 超卖反弹趋势策略（只做多）」 vs Larry Connors RSI(2)

- **原文**：同名策略未找到。中文网上的「RSI 超卖 + 趋势过滤 + 只做多」多为个人博客，阈值 30 / 35 / 40 不统一、无样本外检验（未逐篇列入来源表）。
- **Connors 原规则**（*Short Term Trading Strategies That Work*，2008；StockCharts ChartSchool 转述）：**日线**；
  (1) 价在 200 日 SMA 之上只做多、之下只做空；(2) RSI(2) 低于 10 买入（Connors 测试了 0–10，**低于 5 的后续收益更高**），高于 90 卖空（高于 95 更好）；
  (3) 收盘前或次日开盘下单（Connors 主张收盘前）；(4) 多头收盘上穿 5 日 SMA 离场，空头下穿离场；Connors 不主张止损。
- **已发表的检验**：书中为指数 / ETF 日线历史检验（未读原书，数字未核实）。第三方 Backtrex：SPX 日线 2016-10～2026-10、RSI(2) < 5 / > 95、
  每边 0.02% 佣金：73 笔、胜率 71.2%、平均盈利 1.22% / 平均亏损 2.10%、合计 +18.1%（年化 1.7%）、最大回撤 21.5%，同期买入持有 +255.4%；
  同规则 NAS100 +26.9%（买入持有 +528.3%）、EUR/USD −4.5%。盘中证据只有 ProRealCode 论坛贴称 1 分钟 / 15 分钟版本「灾难性」（二手、无方法）。
- **截图版与原版的差别**：截图版只做多（Connors 原版多空对称）；其余阈值未知。
- **空头镜像**：收盘 < SMA200、收盘 > SMA5、RSI(2) > 90 → 做空（买看跌）；收盘下穿 SMA5 离场。
- **候选**：`survey_rsi2_pullback.yaml`——原参数（SMA200、SMA5、RSI(2) ≤ 10 / ≥ 90）直接用在已完成的 2 分钟 K 上，多空都测。
  注意 200 根 2 分钟 K ≈ 400 分钟，在口径 C 下会跨到盘前 / 前一日；这是字面改写的结果，不另行换算。

### 5.3 「Keltner 通道触边回归」

- **原文**：同名策略未找到。最接近：StockSharp 样例 0031 *Keltner Reversion*（读了源码）：5 分钟，KC = EMA20 ± 2.0 × ATR(14)，
  收盘在通道外且 RSI(14) < 30（> 70）反向入场，回到通道内或 2 × ATR 止损离场；页面自述「年化约 130%」无方法、无成本、无样本外，不可信。
  Kevin Davey 版本（StockSharp 0953，二手摘要）：EMA10 ± 1.6 × ATR，收盘跌破下轨做多、升破上轨做空。
- **与趋势跟踪的对照**：同一条 Keltner 上轨，WunderTrading 的趋势版（收盘站上上轨 + DI 确认 → 做多）与回归版方向正好相反。
  负责人的入场是顺着刚发生的 30–120 秒走势、靠近顺向极值（PRE_ENTRY_MICRO.md），属于趋势侧；触边回归押的是相反方向。
- **候选**：`survey_keltner_revert.yaml`，作为均值回归对照（`oos_pick: false`），在同一引擎下与趋势类规则对照。

## 6. 候选规则（config/intrabar/survey_*.yaml）

<!-- AUTO:rule_table -->
| 文件 | 类别 | family | oos_pick | 负责人情形 | 入场条件（全部方向化） | 需新增特征 |
|---|---|---|---|---|---|---|
| config/intrabar/survey_donchian_breakout.yaml | 趋势突破 | channel_breakout | False | 点火 | donchian(bars=20) > 0 | donchian |
| config/intrabar/survey_ema_pullback.yaml | 顺势回调 | ma_pullback | True | 通道回调 | ma_align(ma=ema, fast=50, slow=200) > 0；ma_side_lag(ma=ema, period=50, lag=5) == 1；ma_cross(ma=ema, period=50) == 1；adx(period=14) > 20 | ma_align、ma_cross、ma_side_lag、adx |
| config/intrabar/survey_keltner_revert.yaml | 均值回归 | band_revert | False | 不对应 | kc_pos(ema_len=20, atr_len=14, atr_mult=2.0) < -1；rsi(period=14) < 30 | kc_pos、rsi |
| config/intrabar/survey_macd_trend_filter.yaml | 多指标共振 | macd_cross_filter | False | 点火 | macd_cross(fast=12, slow=26, signal=9) == 1；ma_dist(ma=ema, period=200, price=bar_close) > 0；adx(period=14) > 20 | macd_cross、ma_dist、adx |
| config/intrabar/survey_orb_breakout.yaml | 趋势突破 | opening_range_breakout | True | 点火 | or_dir(minutes=5) > 0；or_break(minutes=5) > 0 | or_break、or_dir |
| config/intrabar/survey_rsi2_pullback.yaml | 顺势回调 | oscillator_pullback | True | 通道回调 | ma_dist(ma=sma, period=200, price=bar_close) > 0；ma_dist(ma=sma, period=5, price=bar_close) < 0；rsi(period=2) <= 10 | ma_dist、rsi |
| config/intrabar/survey_squeeze_release.yaml | 波动率过滤 | squeeze_release | False | 点火 | squeeze_fire(bb_len=20, bb_mult=2.0, kc_len=20, kc_mult=1.5) == 1；sqz_mom(kc_len=20) > 0；sqz_mom_chg(kc_len=20) > 0 | squeeze_fire、sqz_mom、sqz_mom_chg |
| config/intrabar/survey_supertrend_flip.yaml | 趋势突破 | atr_trail_flip | False | 点火 | supertrend(atr_period=10, mult=3.0) == 1；supertrend_age(atr_period=10, mult=3.0) == 0 | supertrend、supertrend_age |
<!-- /AUTO:rule_table -->

说明：
- 全部**多空对称**（`direction: both`，条件全部方向化：看涨按上涨为正，看跌按下跌为正）；只做一边的原策略已按镜像改写。
- 指标在**已完成的 2 分钟 K**（口径 C：含盘前盘后连续、不按日重置）上算，只用收盘时刻 ≤ t 的 K；开盘区间与 Donchian 的「当前价」用逐秒隐含价。
- **参数照搬原策略默认值**（日线 / 小时线参数直接用在 2 分钟 K 上），没有在任何数据上调过——规则 11：全是待测参数；不调参是为了 R-049 不引入样本内选择。
- `features_needed` 给出引擎尚未实现的特征的精确定义（RSI 用 Wilder、ADX 用 Wilder、MACD 用标准 EMA 等），`engine_ready: false`；第三条实现后才能跑。
- 离场统一为负责人离场习惯（mid +10% 全卖 / −15% 全卖 / 30 分钟上限，一次性）；原策略自己的离场记在 `native_exit`，只作记录。
- 执行两种：(i) R-041 负责人挂单策略 + 负责人离场习惯；(ii) 吃单对照（`executions: [owner, taker]`，由 R-049 实现）。
- 合约：最近可用到期日（`dte_max: 7` 内最近的）、虚值 1 档；0DTE 按 D-001 10:15 后不开（R-049 另报去掉该限制的敏感性）。
- 测试窗口只含 2025-01-02～2026-05-31 与 2026-09-01～2026-10-21（D-033，模块校验不与 2026-06～08 重叠）。
- **R-049 必测（oos_pick）**：`survey_rsi2_pullback`、`survey_orb_breakout`（最常见突破）、`survey_ema_pullback`（最常见回调）。
  开盘区间突破（8 条）只比 N 根通道突破（7 条）多 1 条，「最常见」的判定很薄；`survey_donchian_breakout` 已写好，可作为补充。

## 7. 与负责人入场规律的对照（待第一条完成后补）

待 `docs/research/OWNER_ENTRY_MINING.md`（第一条）完成后，在此对照每条负责人规则与公开策略的趋势判断 / 触发 / 不交易条件，
并生成 `config/intrabar/fused_rule_*.yaml`。

## 8. 完整来源表

<!-- AUTO:source_table -->
| id | 平台 | 名称 | 热度 | 读取 | 周期 | 指标与参数 | 趋势定义 | 入场 | 离场 / 止损 | 过滤 | 宣称表现与检验方式 | 过拟合 / 前视风险 | 类别 | 子类 | 镜像 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tv_supertrend_kivanc | TradingView | [SuperTrend STRATEGY](https://www.tradingview.com/script/P5Gu6F8k/) | 25540 TV 点赞 | pine | 任意（作者未指定） | ATR(10)，乘数 3.0，源 hl2；ATR 用 Wilder（changeATR=true） | 上轨 / 下轨按 hl2 ∓ 3×ATR 单向收紧；收盘穿越上一根的反向轨 → 趋势翻转 | 趋势由 −1 翻为 +1 做多，+1 翻为 −1 做空（收盘确认） | 反向翻转即反手（始终在场） | 只有回测日期窗口 | 无 | 始终在场，震荡期反复翻转；参数无依据；收盘确认，无前视 | 趋势突破 | atr_trail_flip | 原策略对称 |
| tv_utbot_quantnomad | TradingView | [UT Bot Alerts（及 UT Bot Strategy 9006 赞）](https://www.tradingview.com/script/n8ss8BID/) | 60831 TV 点赞 | pine | 任意 | ATR(10)，Key Value 1（止损距离 = 1×ATR） | 收盘价相对 ATR 跟踪止损线的位置；跟踪线只朝趋势方向移动 | 收盘上穿跟踪线做多，下穿做空 | 反向穿越（反手） | 可选 Heikin Ashi 源 | 无（作者让用户自己回测） | 指标脚本（策略版由他人改写）；1×ATR 很近，2 分钟 K 上信号会非常频繁；HA 源会让信号价格与可成交价不一致 | 趋势突破 | atr_trail_flip | 原策略对称 |
| tv_alphatrend | TradingView | [AlphaTrend Strategy](https://www.tradingview.com/script/3wdQu7P3/) | 7135 TV 点赞 | pine | 任意 | SMA(TR,14)，乘数 1；MFI(14) ≥ 50（无量时用 RSI(14) ≥ 50） | MFI ≥ 50 时线按 low − ATR 只升不降，否则按 high + ATR 只降不升；线与 2 根前比较的交叉为翻转 | AlphaTrend 上穿自身 2 根前的值做多，下穿做空 | 反手 | 无 | 无 | 与 SuperTrend 同类；MFI 用成交量，盘前盘后低量会扭曲 | 趋势突破 | atr_trail_flip | 原策略对称 |
| tv_turtles_tmr0 | TradingView | [20 years old Turtles strategy still work!!](https://www.tradingview.com/script/Q1O23zJP/) | 6534 TV 点赞 | pine | 原版海龟为日线 | Donchian 入场 20 / 55，出场 10 / 20 | 价格突破过去 N 根最高 / 最低 | high > 前 20 根最高做多、low < 前 20 根最低做空（55 根为第二系统），盘中触及即入 | 反向触及 10 根（第二系统 20 根）极值 | 无；允许加仓 5 次 | 标题称「仍然有效」，无数字 | 盘中触及入场 + 收盘计算的 Pine 回测会高估成交；日线参数直接搬到分钟线没有依据 | 趋势突破 | channel_breakout | 原策略对称 |
| tv_orb_chrismoody | TradingView | [Opening Range Breakout with 2 Profit Targets](https://www.tradingview.com/script/kpBLjNlk/) | 10123 TV 点赞 | pine | 日内（作者示例 5–15 分钟） | 开盘区间高低点；两档目标 | 价格离开开盘区间的方向即当日方向 | 突破开盘区间上沿做多 / 下沿做空 | 两档固定目标，区间另一侧止损 | 无 | 无 | 指标脚本，不含成交假设 | 趋势突破 | opening_range_breakout | 原策略对称 |
| tv_orb_zzzcrypto | TradingView | [ORB - Opening Range Breakout](https://www.tradingview.com/script/uJsja4dX/) | 11836 TV 点赞 | pine | 日内，区间 5 分钟（默认时段 09:15–09:20，印度市场） | 开盘前 5 分钟高低点 | 同上 | 只画线，不含入场（用户自行判断突破） | 无 | 无 | 无 | 指标；时段写死为 NSE 开盘 | 趋势突破 | opening_range_breakout | 原策略对称 |
| tv_keltner_trend_wunder | TradingView | [Keltner Channel - Trend Based Strategy](https://www.tradingview.com/script/ezSBm5MS/) | 2516 TV 点赞 | pine | 1 小时（ETH/USDT） | KC = SMA(81) ± 2.5×SMA(TR,81)；DMI(19)、ADX 平滑 10；EMA46 vs SMA46 | EMA46 > SMA46 视为上行；+DI > −DI 且 +DI > 27 | 开盘在通道内、收盘站上上轨、+DI > −DI 且 +DI > 27、EMA46 > SMA46 → 做多（源码只有多头） | 收盘跌破下轨或 −4% 止损；+4.5%（15%）与 +20% 两档止盈 | 回测日期窗口 | 无数字；参数 81 / 2.5 / 19 / 27 / 46 明显是优化出来的 | 过拟合风险高（非整参数）；只做多 | 趋势突破 | channel_breakout | 开盘在通道内、收盘跌破下轨、−DI > +DI 且 −DI > 27、EMA46 < SMA46 → 做空；收盘站上上轨或 +4% 止损 |
| tv_vwap_trend_wbburgin | TradingView | [VWAP Trendfollow Strategy](https://www.tradingview.com/script/XoR8rQaW/) | 4609 TV 点赞 | pine | 日内（锚定 Session 的 VWAP） | 会话 VWAP ± 1σ 带（作者默认倍数），2σ 带止盈 | 收盘穿出 VWAP 1σ 带的方向；可选「VWAP 自身上行」过滤 | 收盘上穿上 1σ 带做多，下穿下 1σ 带做空 | 收盘从上方回落穿过上 2σ 带（多）/ 从下方回升穿过下 2σ 带（空）止盈，或反向信号 | 可选 VWAP 斜率同向 | 无 | 锚点选 Earnings 时用 lookahead_on 的财报序列（默认 Session 不受影响） | 趋势突破 | channel_breakout | 原策略对称 |
| tv_torb_zeiierman | TradingView | [Timely Opening Range Breakout Strategy [TORB]](https://www.tradingview.com/script/78AVOLmR/) | 3316 TV 点赞 | pine | 日内 | 会话区间高低；成交量动量线 PMMV 与其均线 | 收盘穿出会话区间 | 收盘上穿区间高点且 PMMV 高于均线和阈值 → 上突破；下方对称 | 指标，无 | 成交量活跃度阈值 | 无 | 指标；量能公式自定义，含 20 根循环确认 | 趋势突破 | opening_range_breakout | 原策略对称 |
| tv_hhll_lonesome | TradingView | [Higher High Lower Low Strategy](https://www.tradingview.com/script/vcg3prja/) | 11237 TV 点赞 | pine | 任意 | pivot 左右各 5 根 | 收盘高于最近一个 Lower High → 上行；低于最近一个 Higher Low → 下行 | 结构翻转（指标版） | 反向翻转 | 无 | 无 | pivot 需要右侧 5 根确认，结构点在 5 根之后才可知（Pine 画在过去位置，看图会产生前视错觉） | 趋势突破 | structure_break | 原策略对称 |
| tv_hull_suite | TradingView | [Hull Suite](https://www.tradingview.com/script/hg92pFwS/) | 22594 TV 点赞 | pine | 任意（作者：55 用于波段入场） | HMA(55) | HULL > HULL[2] 为上行（颜色） | 指标；常见用法是颜色翻转入场 | 反向翻转 | 可选高一级周期 HMA | 无 | 指标；高周期 security 未设 lookahead 参数（旧版默认 off） | 趋势突破 | ma_slope_flip | 原策略对称 |
| tv_double_ema_pullback | TradingView | [Double EMA Pullback - Buy Sell Signal](https://www.tradingview.com/script/4H3SSSTg/) | 5571 TV 点赞 | pine | 任意 | EMA50、EMA200；回看 5 根 | EMA50 > EMA200 为上行 | 收盘上穿 EMA50、EMA50 > EMA200、5 根前收盘在 EMA50 之上（即先在上方、回落到下方、再收回）→ 买；下方对称 → 卖 | 指标，无 | 无 | 无 | 指标；无出场 | 顺势回调 | ma_pullback | 原策略对称 |
| tv_coral_pullback | TradingView | [Coral Trend Pullback Strategy (TradeIQ)](https://www.tradingview.com/script/kc6Itas8/) | 1195 TV 点赞 | pine | 任意（来自 TradeIQ 视频） | Coral Trend（平滑均线）；可选 ADX/DI(14) 中线 20 | Coral 线上升（bfr > bfr[1]）为多 | Coral 上行；上穿后至少 1 根整根在线上；之后收盘跌回线下（回调）且回调期间 Coral 未翻色；再收盘上穿 Coral → 多；可选 +DI > −DI、ADX > 20 且上升 | 近 N 根摆动低点止损，止盈 = R:R × 止损 | ADX/DI 或 ASH+HawkEye 量能确认 | 无 | 条件链长；视频来源的回测未公开 | 顺势回调 | ma_pullback | 原策略对称 |
| tv_macd_pullback_sniper | TradingView | [MACD Pullback Sniper \| Trend + ADX Filtered Strategy](https://www.tradingview.com/script/V87nRt9v/) | 481 TV 点赞 | pine | 任意 | MACD(12,26,9)、EMA200、ADX(14) > 20、ATR(14) | 收盘在 EMA200 之上为多 | MACD 线上穿信号线、收盘 > EMA200、（可选）MACD 线 < 0（回调后的再启动）、ADX > 20 → 多；空头对称 | 2×ATR 止损、2R 止盈、反向交叉平仓 | ADX、零轴 | 无 | 热度低；代码含 0.05% 佣金与 2 tick 滑点，较诚实 | 顺势回调 | macd_pullback | 原策略对称 |
| tv_rsi2_chrismoody | TradingView | [CM RSI-2 Strategy（Upper 9398 赞 / Lower 24636 赞）](https://www.tradingview.com/script/xOm7jSPf/) | 24636 TV 点赞 | pine | 原文日线 | RSI(2)（Wilder），SMA5、SMA200；阈值 10 / 90 | 收盘在 SMA200 之上为多头环境 | 收盘 > SMA200、收盘 < SMA5、RSI(2) < 10 → 多；收盘 < SMA200、> SMA5、RSI(2) > 90 → 空 | 入场后 1–4 根内收盘回到 SMA5 另一侧（high 触及 SMA5） | 无 | 无 | 着色指标，不是回测；日线规则 | 顺势回调 | oscillator_pullback | 原策略对称 |
| tv_macd_sma200_chartart | TradingView | [MACD + SMA 200 Strategy](https://www.tradingview.com/script/yMCa3XZD/) | 31810 TV 点赞 | pine | 任意 | MACD 用 SMA(12,26)、信号 SMA(9)；SMA200 | 26 根前的收盘在 SMA200 之上为多；SMA26 < SMA200 时撤多单 | 柱上穿 0、MACD > 0、SMA12 > SMA26、close[26] > SMA200 → 在当根 low 挂停止单做多；空头对称（high 挂停止单） | 反向信号；日内最大亏损 50% | SMA26 vs SMA200 | 无 | MACD 用 SMA 而非 EMA；停止单挂在 low 上相当于立即成交 | 多指标共振 | macd_cross_filter | 原策略对称 |
| tv_bb_rsi_chartart | TradingView | [Bollinger + RSI, Double Strategy v1.1](https://www.tradingview.com/script/uCV8I4xA/) | 49048 TV 点赞 | pine | 任意 | RSI(6) 穿 50；BOLL(200, 2) | 不判断趋势（着色里用中轨斜率，但不进入下单条件） | RSI(6) 上穿 50 且收盘上穿 BOLL 下轨 → 在下轨挂停止单做多；RSI 下穿 50 且收盘下穿上轨 → 做空 | 反向信号 | 无 | 无 | 均值回归；同一根上两个交叉同时成立很少，信号稀少 | 均值回归 | band_revert | 原策略对称 |
| tv_ema_slope_cross_chartart | TradingView | [EMA Slope + EMA Cross Strategy](https://www.tradingview.com/script/XLGBqrZq/) | 7751 TV 点赞 | pine | 任意 | EMA2、EMA4、EMA20 | 名为趋势，但源码 long = 收盘下穿 EMA20 或（价与 EMA2 下行、下穿 EMA2 且 EMA4 上行） | 收盘下穿 EMA20 做多 / 上穿做空（与名称相反，实为逆势） | 反手 | 无 | 无 | 名称与代码不符的例子：热门 ≠ 规则如名 | 均值回归 | ma_revert | 原策略对称 |
| tv_squeeze_lazybear | TradingView | [Squeeze Momentum Indicator [LazyBear]](https://www.tradingview.com/script/nqQ1DT5a/) | 117512 TV 点赞 | pine | 任意 | BOLL(20,2)、KC(SMA20 ± 1.5×SMA(TR,20))、动量 = linreg(close − avg(avg(HH20,LL20), SMA20), 20) | 动量柱符号与变化（亮绿 / 暗绿 / 亮红 / 暗红） | 指标；常见用法：挤压（BOLL 在 KC 内）释放时顺动量方向入场 | 无 | 挤压状态 | 无 | 源码 BOLL 偏差用的是 multKC（1.5）而不是 mult（2.0），即实际 BOLL 是 1.5σ——热门指标里的实现错误 | 波动率过滤 | squeeze_release | 原策略对称 |
| tv_squeeze_strategy_freeman | TradingView | [Strategy based on Squeeze Momentum Indicator [LazyBear]](https://www.tradingview.com/script/qp9BoNyS/) | 3416 TV 点赞 | pine | 任意 | BOLL(14, 2.0)、KC(16, 1.5)、LazyBear 动量 | 动量柱为亮绿（>0 且上升）/ 亮红（<0 且下降） | 由非「已释放」变为「已释放」（BOLL 跑出 KC）的那根，动量亮绿做多 / 亮红做空 | 动量转暗（多头柱下降）即平 | 可选动量绝对值下限 | 无 | calc_on_every_tick=true，实时与历史行为不同（回测重绘风险） | 波动率过滤 | squeeze_release | 原策略对称 |
| tv_intraday_ts_marcovalente | TradingView | [Intraday TS ,BB + Buy/Sell + Squeeze Mom. + adx-dmi](https://www.tradingview.com/script/opkkK4oM/) | 5566 TV 点赞 | pine | 日内（作者：5 分钟及以上） | BOLL、开盘区间、LazyBear 挤压、DMI/ADX（≥ 29 为趋势，≤ 20 为无趋势） | +DI ≥ −DI 且 ADX ≥ 29 → 上行；ADX ≤ 20 → 无趋势 | 价格在 BOLL 上轨之上时在区间高点 +0.001 挂突破买单；下方对称 | 两档目标 = 入场到对侧轨距离的 1 倍 / 2 倍 | 挤压状态、ADX 分档 | 无 | 指标（画线），无成交假设 | 波动率过滤 | squeeze_release | 原策略对称 |
| tv_adx_dmi_millerrh | TradingView | [ADX \| DMI Trend](https://www.tradingview.com/script/4aQD9gqq/) | 761 TV 点赞 | pine | 任意 | DMI(14)、ADX 平滑 14、关键水平 23；EMA200；成交量 > 2×SMA(vol,20)；ATR(14)×2 跟踪、SAR | ADX 近 3 根上升且 > 23；+DI > −DI；收盘 ≥ EMA200 | 以上全部 + 成交量放大 → 多；空头对称 | ADX 较上一根下降、ATR 跟踪或 SAR 被触及 | 成交量、EMA200 | 无 | 出场条件「ADX 下降一根」非常敏感 | 多指标共振 | adx_di | 原策略对称 |
| ft_trendrider | GitHub/freqtrade-strategies | [TrendRiderStrategy](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/TrendRiderStrategy.py) | 5588 GitHub star（仓库） | python | 1 小时；4 小时与日线确认 | EMA(多档)、EMA50、EMA200；RSI；ADX(14)、DI；MACD(12,26,9)；BOLL(20,2) 宽度；OBV；ATR(14) | 收盘 > EMA200 且 EMA50 > EMA200（is_bull），4 小时同样 is_bull，日线 EMA200 | 多种入场标签：回踩慢 EMA（low ≤ EMA×1.02、收盘 > EMA、阳线）、EMA50 反弹、RSI 回调区间 + ADX > 阈值、RSI 回升穿越、快慢 EMA 金叉 + ADX > 18 等 | RSI 超买、快慢 EMA 死叉；6% 止损（可选 ATR） | BTC 市场情绪（加密专用） | 无 | 超参数（hyperopt）多；只做多 | 顺势回调 | ma_pullback | 收盘 < EMA200 且 EMA50 < EMA200；high ≥ EMA×0.98、收盘 < EMA、阴线 → 做空 |
| ft_adxmomentum | GitHub/freqtrade-strategies | [ADXMomentum（berlinguyinca）](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/berlinguyinca/ADXMomentum.py) | 5588 GitHub star（仓库） | python | 1 小时 | ADX(14)、+DI(25)、−DI(25)、MOM(14) | ADX > 25 且 +DI > 25 且 +DI > −DI | ADX > 25、MOM > 0、+DI > 25、+DI > −DI → 多（源码出场条件即空头镜像） | ADX > 25、MOM < 0、−DI > 25、−DI > +DI；25% 止损 | 无 | 无 | 只做多，出场即镜像 | 多指标共振 | adx_di | ADX > 25、MOM < 0、−DI > 25、−DI > +DI → 做空 |
| ft_hlhb | GitHub/freqtrade-strategies | [hlhb](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/hlhb.py) | 5588 GitHub star（仓库） | python | 4 小时 | RSI(10, hl2) 穿 50；EMA5 / EMA10 交叉；ADX > 25 | EMA5 上穿 EMA10 且 RSI 上穿 50 且 ADX > 25 | 三者同根成立 → 多 | 三者反向同根成立；32% 止损 | ADX | 无 | 三个交叉要求同一根，信号稀少；只做多 | 多指标共振 | ma_cross_filter | RSI 下穿 50、EMA5 下穿 EMA10、ADX > 25 → 做空 |
| ft_volatility_system | GitHub/freqtrade-strategies | [VolatilitySystem（futures）](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/futures/VolatilitySystem.py) | 5588 GitHub star（仓库） | python | 1 小时，指标在 3 小时重采样上算 | ATR(14)×2；收盘变化 | 单根收盘变化超过 2×ATR 视为趋势启动 | 收盘变化 > 前一根 2×ATR → 多；< −2×ATR → 空 | 反向信号 | 无 | 无 | 重采样合并时若用未完成的 3 小时 K 会有前视（源码用 resampled_merge，未逐行核实） | 趋势突破 | range_expansion | 原策略对称 |
| ft_trend_following | GitHub/freqtrade-strategies | [TrendFollowingStrategy（futures）](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/futures/TrendFollowingStrategy.py) | 5588 GitHub star（仓库） | python | 5 分钟 | EMA20；OBV | 收盘在 EMA20 之上 | 收盘上穿 EMA20 且 OBV 上升 → 多；下穿且 OBV 下降 → 空 | 反向 | OBV | 无 | 26.5% 止损（超参数） | 趋势突破 | ma_cross | 原策略对称 |
| ft_fadxsma | GitHub/freqtrade-strategies | [FAdxSmaStrategy（futures）](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/futures/FAdxSmaStrategy.py) | 5588 GitHub star（仓库） | python | 1 小时 | ADX(n)、短 / 长 SMA（hyperopt 参数） | ADX > 阈值 + SMA 交叉方向 | ADX > 阈值且短 SMA 上穿长 SMA → 多；下穿 → 空 | ADX < 阈值等；5% 止损 + 跟踪 | ADX | 无 | 参数全部 hyperopt | 多指标共振 | ma_cross_filter | 原策略对称 |
| ft_bbandrsi | GitHub/freqtrade-strategies | [BbandRsi（berlinguyinca）](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/berlinguyinca/BbandRsi.py) | 5588 GitHub star（仓库） | python | 1 小时 | RSI(14)；BOLL(20,2)（典型价） | 不判断 | RSI < 30 且收盘 < 下轨 → 多 | RSI > 70；ROI 10%；25% 止损 | 无 | 无 | 只做多的均值回归 | 均值回归 | band_revert | RSI > 70 且收盘 > 上轨 → 做空；RSI < 30 平仓 |
| ft_scalp | GitHub/freqtrade-strategies | [Scalp（berlinguyinca）](https://github.com/freqtrade/freqtrade-strategies/blob/main/user_data/strategies/berlinguyinca/Scalp.py) | 5588 GitHub star（仓库） | python | 1 分钟 | EMA5（high/close/low）；快速随机 STOCHF(5,3)；ADX > 30 | 只用 ADX > 30 判断「有趋势」，不判断方向 | 开盘 < EMA5(low)、ADX > 30、fastk 与 fastd < 30 且 fastk 上穿 fastd → 多 | 开盘 ≥ EMA5(high) 或 fastk / fastd 上穿 70；4% 止损 | ADX | 无 | ADX 不给方向，强下跌中也会抄底 | 顺势回调 | oscillator_pullback | 开盘 > EMA5(high)、ADX > 30、fastk 与 fastd > 70 且 fastk 下穿 fastd → 空 |
| gh_quantdinger_resonance | GitHub/QuantDinger | [Indicator Resonance（策略模板）](https://github.com/OpenByteInc/QuantDinger/blob/main/backend_api_python/migrations/strategy_v2_templates.sql) | 13056 GitHub star（仓库） | sql | 日线（QQQ） | MACD(12,26,9) 柱 > 0；50 < RSI(14) < 75；ADX(14) > 20 | 三项同时成立即「多头状态」 | 状态成立且空仓 → 买入 95% | 状态不再成立 → 清仓 | RSI 上限 75 防追高 | 无 | 只做多（direction_mode=long_only） | 多指标共振 | macd_cross_filter | 柱 < 0、25 < RSI < 50、ADX > 20 → 做空状态 |
| gh_quantdinger_macd_kdj | GitHub/QuantDinger | [MACD and KDJ Confirmation（策略模板）](https://github.com/OpenByteInc/QuantDinger/blob/main/backend_api_python/migrations/strategy_v2_templates.sql) | 13056 GitHub star（仓库） | sql | 4 小时（BTC 永续） | MACD(12,26,9)；KDJ(9,3,3)；超买 85 | MACD 柱 > 0 | 柱 > 0 且（柱由 ≤0 转 >0 或 K 上穿 D）且 K < 85 → 多 | 柱 ≤ 0 或 K 下穿 D；2% 止损、+5% 后 1% 跟踪 | K < 85 | 无 | 只做多 | 多指标共振 | macd_cross_filter | 柱 < 0 且（柱由 ≥0 转 <0 或 K 下穿 D）且 K > 15 → 空 |
| gh_quantdinger_turtle | GitHub/QuantDinger | [Turtle Trading（策略模板）](https://github.com/OpenByteInc/QuantDinger/blob/main/backend_api_python/migrations/strategy_v2_templates.sql) | 13056 GitHub star（仓库） | sql | 日线（SPY） | Donchian 入场 20、出场 10；ATR(14)×2 止损 | 收盘突破前一根的 20 根上沿 | 收盘 > 前一根 20 根上沿 → 多 | 收盘 < 前一根 10 根下沿或 < 入场价 − 2×ATR | 无 | 无 | 只做多 | 趋势突破 | channel_breakout | 收盘 < 20 根下沿 → 空；收盘 > 10 根上沿或 > 入场 + 2×ATR 平 |
| gh_quantdinger_three_lines_trend | GitHub/QuantDinger | [Bullish Three Averages With Trend Filter（策略模板）](https://github.com/OpenByteInc/QuantDinger/blob/main/backend_api_python/migrations/strategy_v2_templates.sql) | 13056 GitHub star（仓库） | sql | 日线（A 股） | MA5/10/20；MA60 及其 5 根斜率；实体 ≥ 2% | 收盘 > MA60 且 MA60 高于 5 根前（斜率为正） | 开盘 ≤ 三条均线最小值、收盘 ≥ 最大值（一阳穿三线）、实体 ≥ 2%、趋势成立 → 多 | 收盘 < MA20 或趋势失效 | 实体大小 | 无 | 只做多 | 趋势突破 | ma_cross | 开盘 ≥ 三线最大、收盘 ≤ 最小、实体 ≤ −2%、收盘 < MA60 且 MA60 下行 → 空 |
| gh_myquant_dual_thrust | GitHub/myquant(掘金) | [Dual Thrust](https://github.com/myquant/strategy/tree/master/Dual-Thrust) | 956 GitHub star（仓库） | python | 日内分钟线，区间按前 N 日日线 | Range = max(HH − LC, HC − LL)（前 N 日）；上轨 = 开盘 + k1×Range，下轨 = 开盘 − k2×Range | 价格离开以今日开盘为中心的波动区间 | 突破上轨做多，跌破下轨做空（反向突破则反手） | 反手或收盘平仓 | 无 | 无 | k1、k2 常被优化；info.md 为空，规则取自 quant-trading 源码（11109 star） | 趋势突破 | opening_range_breakout | 原策略对称 |
| gh_myquant_hans123 | GitHub/myquant(掘金) | [Hans123](https://github.com/myquant/strategy/tree/master/Hans123) | 956 GitHub star（仓库） | desc | 日内 | 开盘后 30 分钟高低点 | 开盘 30 分钟区间的突破方向 | 30 分钟后，突破上轨做多、跌破下轨做空 | 收盘平仓 | 文中提到可加包络带、时间、波幅过滤 | 无 | 无 | 趋势突破 | opening_range_breakout | 原策略对称 |
| gh_myquant_rbreaker | GitHub/myquant(掘金) | [R-Breaker](https://github.com/myquant/strategy/tree/master/R-Breaker) | 956 GitHub star（仓库） | desc | 1 / 5 分钟 | 前日 H/L/C 算 6 个价位 | 突破买入价 / 突破卖出价之外为趋势；观察价之间为反转区 | 空仓时突破 buy_break 做多 / 跌破 sell_break 做空；持仓时到达观察价后回落穿反转价则反手 | 反手或收盘 | 无 | 无 | 突破 + 反转混合 | 趋势突破 | channel_breakout | 原策略对称 |
| gh_myquant_skypark | GitHub/myquant(掘金) | [空中花园 SkyPark](https://github.com/myquant/strategy/tree/master/SkyPark) | 956 GitHub star（仓库） | desc | 日内 | 第一根 K 高低点；跳空 ≥ 1% | 高开 / 低开 ≥ 1% 当日，第一根 K 的突破方向 | 突破第一根 K 高点做多，跌破低点做空 | 收盘平仓 | 只在大幅跳空日 | 无 | 无 | 趋势突破 | opening_range_breakout | 原策略对称 |
| gh_myquant_adx_dmi | GitHub/myquant(掘金) | [ADX_DMI_STOCK](https://github.com/myquant/strategy/tree/master/ADX_DMI_STOCK) | 956 GitHub star（仓库） | desc | 日线 | ADX(14)、DMI；双均线 | ADX 上升代表趋势转强；+DI 上穿 −DI 为入场 | 双均线信号 + ADX 上升确认 | −DI 上穿 +DI | ADX 上升，用于避免横盘反复交易 | 无 | 文档只给思路 | 多指标共振 | adx_di | 原策略对称 |
| gh_myquant_boll | GitHub/myquant(掘金) | [BOLL_STOCK](https://github.com/myquant/strategy/tree/master/BOLL_STOCK) | 956 GitHub star（仓库） | desc | 日线 | BOLL 三轨方向 | 上中下三轨同时向上且价 > 中轨 | 三轨同上且价 > 中轨 → 买 | 三轨同下或价 < 中轨 | 无 | 无 | 只做多（A 股） | 多指标共振 | band_slope | 三轨同下且价 < 中轨 → 空 |
| gh_myquant_macd | GitHub/myquant(掘金) | [MACD-STOCK](https://github.com/myquant/strategy/tree/master/MACD-STOCK) | 956 GitHub star（仓库） | desc | 日线 | MACD（DIF、DEA） | DIF、DEA 同在零轴一侧 | DIF、DEA 均 > 0 且 DIF 上穿 DEA → 买；均 < 0 且下穿 → 卖 | 反向 | 零轴 | 无 | 无 | 多指标共振 | macd_cross_filter | 原策略对称 |
| gh_alpaca_scalping | GitHub/alpacahq | [example-scalping](https://github.com/alpacahq/example-scalping) | 852 GitHub star（仓库） | python | 1 分钟 | 20 分钟 SMA | 上一根收盘在 SMA20 下、当前上穿 | 1 分钟收盘上穿 20 分钟均线 → 限价买入，2 分钟未成交撤单 | 限价止盈卖出 / 收盘前清仓 | 无 | 无（作者明说是演示） | 只做多；示例 | 趋势突破 | ma_cross | 1 分钟收盘下穿 20 分钟均线 → 做空 |
| qc_orb_stocks_in_play | QuantConnect | [Opening Range Breakout for Stocks in Play](https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/) | 未取得 | desc | 5 分钟开盘区间 | 前 5 分钟高低点；开盘 5 分钟成交量 / 前 14 日同窗口均量（in play）；ATR | 第一根 5 分钟 K 收阳 → 只做上破，收阴 → 只做下破 | 阳线后以停止单在区间高点做多（阴线后在低点做空），只做最「in play」的 20 只 | ATR 止损或收盘平仓 | 股价 > 5、ATR > 0.5、相对成交量排名 | 2016 年 Sharpe 2.396（SPY 0.836）；25 组参数中 17 组 Sharpe 高于基准 | 只回测了 2016 一年；多标的选股，单标的 MU 不适用选股部分 | 趋势突破 | opening_range_breakout | 原策略对称 |
| paper_orb_qqq | 论文/SSRN | [Can Day Trading Really Be Profitable?（QQQ 5 分钟 ORB）](https://www.mql5.com/en/blogs/post/776235) | 未取得 | secondary | 5 分钟 | 第一根 5 分钟 K | 第一根 5 分钟 K 的方向 | 第一根收阳在第二根开盘做多，收阴做空（十字星不做） | 第一根 K 另一端止损，10R 目标或收盘 | 无 | 命中率 24%、+0.13R/笔、4 倍杠杆年化 33%（QQQ） | 原文只计佣金不计点差与滑点、无样本外；MQL5 博客在 5 个指数 CFD 2015-01～2026-06 复现：毛利复现（NQ +0.131R），扣成本后 5 个市场无一显著、4 个为负 | 趋势突破 | opening_range_breakout | 原策略对称 |
| paper_noise_area | 论文/SSRN | [Beat the Market（SPY 噪声区日内动量）](https://www.cxoadvisory.com/?p=45565) | 未取得 | secondary | 1 分钟数据，HH:00 / HH:30 判定 | 噪声区（前 14 日同一分钟平均离开盘幅度）、VWAP | 收盘在噪声区外且在 VWAP 同侧 | 收盘 > 上边界且 > VWAP → 多；< 下边界且 < VWAP → 空 | 当前边界或 VWAP 作跟踪止损；收盘平仓 | 波动率定仓 | 原文 2007-05～2024-04 显著为正 | 本仓库 R-045 已按原文复现：SPY 主口径成本 −5.35%、QQQ +32.76%，随机方向 93.3 分位，预测不成立（SPY_INTRADAY_MOMENTUM.md） | 趋势突破 | channel_breakout | 原策略对称 |
| paper_connors_rsi2 | 书籍/StockCharts | [Connors RSI(2)（Short Term Trading Strategies That Work, 2008）](https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2) | 未取得 | desc | 日线 | RSI(2)；SMA200；SMA5 | 价在 SMA200 之上只做多，之下只做空 | RSI(2) < 10（Connors 测试 0–10，< 5 更好）做多；> 90（> 95 更好）做空；收盘前或次日开盘下单 | 多头收盘上穿 SMA5，空头下穿 SMA5；Connors 不主张止损 | SMA200 | 书中为 SPX 等指数日线历史检验；Backtrex（第三方）SPX 2016-10～2026-10 日线 73 笔、胜率 71.2%、+18.1%（买入持有 +255.4%）、最大回撤 21.5% | 原规则是日线；ProRealCode 论坛贴称 1 分钟、15 分钟版本结果很差（二手、无方法） | 顺势回调 | oscillator_pullback | 原策略对称 |
| paper_nsga2 | 论文/arXiv | [Optimal Technical Indicator-based Trading Strategies Using NSGA-II](https://arxiv.org/abs/2111.13364) | 未取得 | desc | 日线 | 多个技术指标的组合（由 NSGA-II 搜索） | 由算法选出的指标组合决定，无固定定义 | 算法搜索出的买卖规则组合 | 同上 | 目标函数 = 最大化 Sharpe、最小化最大回撤 | 滚动 2 年训练 / 1 年测试；作者称在无大幅经济波动的平稳期明显更好，含交易成本 | 多目标搜索在大量组合中选最优，样本内过拟合与多重检验风险最高；作者自己承认在剧烈波动期失效 | 多指标共振 | optimized_combo | 原策略对称 |
| ss_keltner_reversion | GitHub/StockSharp | [Keltner Reversion（样例 0031）](https://doc.stocksharp.com/en/api-examples/0031_Keltner_Reversion) | 10972 GitHub star（仓库） | python | 5 分钟 | KC = EMA20 ± 2.0×ATR(14)；RSI(14) 30/70；止损 2×ATR（入场时冻结） | 不判断趋势 | 收盘在下轨之外且 RSI < 30 → 多；在上轨之外且 RSI > 70 → 空 | 回到通道内或止损 | RSI | 页面自述年化约 130%（无方法、无成本、无样本外） | 厂商样例；自述收益不可信 | 均值回归 | band_revert | 原策略对称 |
| ss_keltner_davey | GitHub/StockSharp | [Keltner Channel by Kevin Davey（样例 0953）](https://doc.stocksharp.com/en/api-examples/0953_Keltner_Channel_by_Kevin_Davey) | 10972 GitHub star（仓库） | secondary | 未写明 | KC = EMA10 ± 1.6×ATR | 不判断趋势 | 收盘跌破下轨做多，升破上轨做空 | 未取得 | 无 | 无 | 页面正文未成功抽取，规则来自搜索摘要 | 均值回归 | band_revert | 原策略对称 |
| mql5_trendinggoldea | MQL5 Market | [TrendingGoldea](https://www.mql5.com/en/market/product/188049) | 未取得 | secondary | XAUUSD M15，H1 过滤 | EMA20 / EMA50；ADX(14)；H1 EMA；ATR | EMA20 vs EMA50 方向、ADX 确认有趋势、H1 EMA 同向 | 价格回踩触及快 EMA 后顺势恢复时入场（不追价） | ATR 止损 / 目标 / 跟踪；部分止盈；日亏损上限 | 只在伦敦、纽约时段 | 厂商称多年 M15 回测与模拟盘前测，统计在图集里 | 厂商自述，未独立核实；规则细节来自搜索摘要 | 顺势回调 | ma_pullback | 原策略对称 |
| mql5_trend_pullback_scalper | MQL5 Market | [Trend Pullback Scalper](https://www.mql5.com/en/market/product/188612) | 未取得 | secondary | 外汇，MT5 | 快 / 慢 EMA、入场 EMA、ADX；K 线形态确认 | 快慢 EMA 定方向，ADX 定强度 | 回调到入场 EMA 附近后出现合格 K 线信号再入场 | 止损止盈；可选保本、跟踪、部分平仓 | 点差、日亏损、每日笔数上限 | 无 | 厂商自述 | 顺势回调 | ma_pullback | 原策略对称 |
| mql5_donchian_portfolio | MQL5 Market | [Donchian Portfolio Visual](https://www.mql5.com/en/market/product/184188) | 未取得 | secondary | H4，多品种 | Donchian；EMA200；ADX；ATR | Donchian 突破 + EMA200 方向 + ADX | 通道突破且 EMA200、ADX 同意 | 2.0 ATR 止盈、1.5 ATR 止损或较短出场通道反向突破 | 无网格、无马丁 | 无 | 厂商自述 | 趋势突破 | channel_breakout | 原策略对称 |
| mql5_article_3146 | MQL5 Articles | [Donchian 突破 + 确认指标（文章 3146）](https://www.mql5.com/en/articles/3146) | 未取得 | secondary | 未取得 | Donchian；三个确认指标之一为 ADX | 突破须至少一个趋势指标确认，避免假突破 | Donchian 突破 + 确认 | 未取得 | ADX 等 | 未取得 | 二手摘要 | 波动率过滤 | breakout_confirm | 原策略对称 |
<!-- /AUTO:source_table -->

## 9. TradingView 条目按点赞排序

<!-- AUTO:tv_top -->
| name | author | kind | 点赞（2026-10-23） | category | family |
|---|---|---|---|---|---|
| Squeeze Momentum Indicator [LazyBear] | LazyBear | indicator | 117512 | vol_filter | squeeze_release |
| UT Bot Alerts（及 UT Bot Strategy 9006 赞） | QuantNomad（原作 Yo_adriiiiaan） | indicator | 60831 | breakout | atr_trail_flip |
| Bollinger + RSI, Double Strategy v1.1 | ChartArt | strategy | 49048 | mean_reversion | band_revert |
| MACD + SMA 200 Strategy | ChartArt | strategy | 31810 | confluence | macd_cross_filter |
| SuperTrend STRATEGY | KivancOzbilgic | strategy | 25540 | breakout | atr_trail_flip |
| CM RSI-2 Strategy（Upper 9398 赞 / Lower 24636 赞） | ChrisMoody | indicator | 24636 | pullback | oscillator_pullback |
| Hull Suite | InSilico | indicator | 22594 | breakout | ma_slope_flip |
| ORB - Opening Range Breakout | zzzcrypto123 | indicator | 11836 | breakout | opening_range_breakout |
| Higher High Lower Low Strategy | LonesomeTheBlue | indicator | 11237 | breakout | structure_break |
| Opening Range Breakout with 2 Profit Targets | ChrisMoody | indicator | 10123 | breakout | opening_range_breakout |
| EMA Slope + EMA Cross Strategy | ChartArt | strategy | 7751 | mean_reversion | ma_revert |
| AlphaTrend Strategy | KivancOzbilgic | strategy | 7135 | breakout | atr_trail_flip |
| 20 years old Turtles strategy still work!! | tmr0 | strategy | 6534 | breakout | channel_breakout |
| Double EMA Pullback - Buy Sell Signal | hanabil | indicator | 5571 | pullback | ma_pullback |
| Intraday TS ,BB + Buy/Sell + Squeeze Mom. + adx-dmi | MarcoValente | indicator | 5566 | vol_filter | squeeze_release |
| VWAP Trendfollow Strategy | wbburgin | strategy | 4609 | breakout | channel_breakout |
| Strategy based on Squeeze Momentum Indicator [LazyBear] | 03.freeman | strategy | 3416 | vol_filter | squeeze_release |
| Timely Opening Range Breakout Strategy [TORB] | Zeiierman | indicator | 3316 | breakout | opening_range_breakout |
| Keltner Channel - Trend Based Strategy | WunderTrading | strategy | 2516 | breakout | channel_breakout |
| Coral Trend Pullback Strategy (TradeIQ) | kevinmck100 | strategy | 1195 | pullback | ma_pullback |
| ADX \| DMI Trend | millerrh | strategy | 761 | confluence | adx_di |
| MACD Pullback Sniper \| Trend + ADX Filtered Strategy | blitz_locked | strategy | 481 | pullback | macd_pullback |

TradingView 22 条中指标脚本 10 条（只画信号、无离场与成本）、策略脚本 12 条；点赞最高的 10 条里指标 7 条。
<!-- /AUTO:tv_top -->

热门 ≠ 规则如名：ChartArt「EMA Slope + EMA Cross Strategy」（7751 赞）源码是收盘**下穿** EMA20 做多，实为逆势；
LazyBear Squeeze（117512 赞，本次检索最高）源码里 BOLL 的偏差误用了 KC 的乘数 1.5，而不是输入的 2.0。

## 10. 外部来源

- TradingView 脚本：见第 8 节各行链接（2026-10-23 经 `pubscripts-suggest-json` / `pine-facade` 接口读取源码与点赞数）。
- freqtrade-strategies：<https://github.com/freqtrade/freqtrade-strategies>
- QuantDinger：<https://github.com/OpenByteInc/QuantDinger>（`backend_api_python/migrations/strategy_v2_templates.sql`）
- 掘金策略库：<https://github.com/myquant/strategy>；Dual Thrust 实现：<https://github.com/je-suis-tm/quant-trading>
- QuantConnect ORB：<https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/>
- ORB 论文复现（MQL5 博客）：<https://www.mql5.com/en/blogs/post/776235>
- 噪声区动量摘要：<https://www.cxoadvisory.com/?p=45565>；本仓库复现见 `SPY_INTRADAY_MOMENTUM.md`（R-045）
- Connors RSI(2)：<https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2>；
  Backtrex：<https://backtrex.com/en/backtests/connors-rsi-2-sp-500>；ProRealCode 讨论：<https://www.prorealcode.com/prorealtime-trading-strategies/rsi-2-strategy-larry-connors/>
- NSGA-II 论文：<https://arxiv.org/abs/2111.13364>
- StockSharp Keltner：<https://doc.stocksharp.com/en/api-examples/0031_Keltner_Reversion>、<https://doc.stocksharp.com/en/api-examples/0953_Keltner_Channel_by_Kevin_Davey>
- MQL5：<https://www.mql5.com/en/market/product/188049>、<https://www.mql5.com/en/market/product/188612>、
  <https://www.mql5.com/en/market/product/184188>、<https://www.mql5.com/en/articles/3146>
- 聚宽：<https://www.joinquant.com/>（地区限制，未取得内容）

## 复现

```bash
uv run python -m zr.research.strategy_survey
```
