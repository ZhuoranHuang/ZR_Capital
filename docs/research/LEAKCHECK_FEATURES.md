# zr/features 前视截断检验

`python -m zr.research.leakcheck` 生成，不手改。检验函数 `zr.research.leakcheck.assert_no_lookahead_frame`：截到第 k 行重算，前 k 个值必须与全量完全一致（rtol 1e-9，NaN 视为相等）。背景：RESEARCH_LOG 教训 L-001（2026-11 /tmp 特征研究的三处时序错误）。**未通过的函数只列出，未修改原函数。**

zr/features 公开函数 73 个：检验覆盖 50 个，不适用 23 个，未归类 0 个。检验项 29 个，未通过 1 个。

## 检验结果

| 检验 | 数据 | 输出列数 | 结果 | 不一致的列（截断点） | 说明 |
|---|---|---|---|---|---|
| indicators.sma | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.ema | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.wilder_rsi | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.macd | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 3 | 通过 |  |  |
| indicators.bollinger | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 5 | 通过 |  |  |
| indicators.true_range | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.atr | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.adx | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 3 | 通过 |  |  |
| indicators.kdj | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 3 | 通过 |  |  |
| indicators.realized_vol | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.session_vwap | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.bar_slope | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| indicators.cross_signal | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| features.compute_features | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 29 | 通过 |  |  |
| channel.three_line_slopes + parallelism | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 4 | 通过 |  |  |
| channel.channel_states | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 25 | 通过 |  |  |
| structure.compute_structure | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 10 | 通过 |  |  |
| structure.macd_state | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| structure.tf5_regime | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| structure.bars_since | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| pullback.depth_features + directional | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 6 | 通过 |  |  |
| regime.indicators / regimes / state_features / range_score | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 30 | **未通过（5 列）** | `range_score`（391, 392, 395, 405, 421, 494）；`reg_range_score`（391, 392, 395, 405, 421, 494）；`reg_state`（391, 392, 395, 405, 421, 494）；`sf_rg_duration`（391, 392, 395, 405, 421, 494）；`sf_rg_state`（391, 392, 395, 405, 421, 494） |  |
| cross_asset.compute_cross_asset | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 7 | 通过 |  |  |
| calendar_features.compute_calendar | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 4 | 通过 |  |  |
| bars.resample_bars（按完成时刻映射回分钟行） | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 5 | 通过 |  |  |
| bars.volume_bars（按完成时刻映射回分钟行） | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 3 | 通过 |  |  |
| bars.volume_threshold（按日期映射回分钟行） | 合成 1 分钟 K，25 个交易日，均匀 10 个 + 开盘附近 50 个截断点 | 1 | 通过 |  |  |
| option_features.day_features（含 oi_level_features） | 真实数据 |  | 通过 |  | MU 2025-03-05、2025-09-10 真实数据；报价按快照时刻、期权 bar 按收盘时刻、正股按收盘时刻截断；compute_option_features 逐日调用 day_features |
| entry_features 全部特征族（truncation_check） | 真实数据 |  | 通过 |  | MU 2025-03-05、2025-09-10 真实数据，每天 3 个时刻（秒级），模块自带 truncation_check（load_day(t_cut) 截断全部输入），两个方向逐特征比较 |

## 不适用（不是逐行时间序列特征）

| 函数 | 理由 |
|---|---|
| bars.daily_volume | 日级汇总（当日全天成交量），输出按日期索引、不逐行；在分钟行上使用必须先 shift(1) 按日（volume_threshold 已这样做并单独检验） |
| bars.et_date | 时间戳 → 美东日期的工具 |
| bars.load_bars | 数据加载器（全量读取，不是特征） |
| bars.load_extended_bars | 数据加载器 |
| calendar_features.expiry_days | 读取到期日表（静态日历） |
| calendar_features.release_times | 读取定时宏观发布时刻（静态配置） |
| channel.load_channel_config | 读取配置 |
| cross_asset.cross_asset_columns | 列名列表 |
| entry_features.family_of | 特征名 → 族名的字符串工具 |
| option_features.bs_price | Black–Scholes 定价（纯数学） |
| option_features.implied_vol | 隐含波动率反解（纯数学） |
| option_features.load_option_day | 数据加载器 |
| option_features.open_window | 按开盘价筛行权价的工具（在 day_features 检验内覆盖） |
| option_features.option_days | 列出有数据的日期 |
| option_features.strike_spacing | 行权价间距（纯数学） |
| option_features.years_to_expiry | 到期年限（纯数学） |
| pullback.load_config | 读取配置 |
| pullback.pullback_columns | 列名列表 |
| range_calibration.main | 离线标定脚本入口（年度打分，不产生分钟特征） |
| range_calibration.year_scores | 离线标定（按年汇总），不产生分钟特征 |
| regime.bar_close_minute | K 线收盘分钟的时间工具 |
| regime.bars_per_day | 每日 K 线根数（标定窗口用的常数） |
| regime.load_config | 读取配置 |

## 边界

- 截断检验只能发现「t 时刻的值用到了 t 之后的数据」。约定：第 i 行的值在第 i 根 K 收盘时可得（包含该根的收盘价）。
- 发现不了入场时点错误：用信号 K 的开盘价入场、9:31 入场却用前 15 分钟涨幅做过滤器——这些值本身没有前视，错在入场早于信息可得时刻（CLAUDE.md「前视检查」与 scripts/run_rule.py 的入场映射负责）。
- 合成数据只覆盖代码路径，不覆盖真实数据里的缺口、半日市以外的异常；截断点 = 均匀 10 个（偏移 7 行落在日内）+ 10 个交易日开盘后第 1 / 2 / 5 / 15 / 31 行。
- 依赖外部数据的分支（compute_calendar 的到期日标记、cross_asset 从磁盘读参考标的）未在合成数据上覆盖。
