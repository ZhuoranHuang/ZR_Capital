# 规则接口（规则 → 结果）

写一个 Python 规则文件，一条命令得到逐笔明细与汇总。只覆盖「规则 → 结果」：没有 YAML DSL、盘中实时模式、画图、随机对照。

```bash
python scripts/run_rule.py --rule rules/example_boll_macd.py --symbol MU \
  --start 2025-01-02 --end 2025-12-31 --mode stock --out reports/example
```

在 worktree 里运行时加 `PYTHONPATH=.`，否则 `zr` 会从主仓库导入。数据只读本地 `$ZR_DATA_ROOT`（`bars_1m`、`options_1m`），不联网。

## 1 规则文件

```python
PERIOD = 2                      # 1 / 2 / 5 分钟
DIRECTION = "both"              # long / short / both
EXIT = {"kind": "time", "bars": 10}
# 或 {"kind": "bracket", "tp_pct": 0.02, "sl_pct": 0.015, "max_bars": 30}
CONTRACT = {"dte_max": 1, "otm_steps": 1}   # 仅 option 模式用

def signal(bars: pd.DataFrame) -> pd.Series:
    """bars：PERIOD 分钟 K（索引 = 开盘时刻，UTC）。返回同长度 Series，值 ∈ {-1, 0, +1}。
    只允许使用当前 bar 及之前的 bar。"""
```

| 项 | 说明 |
|---|---|
| `bars` | RTH 1 分钟 K 经 `zr.features.bars.resample_bars` 聚合（按美东日、整分钟网格），**跨日连续**，不按日重置；需要按日重置的指标由规则自己按日期分组。列：open / high / low / close / volume。 |
| `signal` 返回值 | 长度必须等于 `len(bars)`，取值只能是 −1 / 0 / +1，不能有 NaN，否则 runner 报错。每个非零值都是一个信号；持仓中、冷却中的信号由引擎计入跳过，不需要规则自己去重。 |
| `DIRECTION` | `long` 只保留 +1，`short` 只保留 −1。 |
| `EXIT["bars"]`、`EXIT["max_bars"]` | 以 PERIOD 根计；runner 换算成 1 分钟根数（× PERIOD）。 |
| `tp_pct` / `sl_pct` | **一律指正股价格**（相对正股入场成交价），期权模式也一样，不是期权涨跌幅。也可以给绝对价 `tp_px` / `sl_px`。 |
| `{"kind": "signal", ...}` | 未实现，抛 `NotImplementedError`。 |

**无前视**：`tests/test_run_rule.py` 里的 `assert_no_lookahead(signal_fn, bars, cuts=3)` 对每个截断点 k 检查 `signal_fn(bars.iloc[:k])` 的前 k 个值与 `signal_fn(bars)` 的前 k 个值相同。新规则请在该文件加一条测试调用它（示例规则已有）。

## 2 时间与成交

信号出现在 PERIOD K 的**收盘时刻 T**（标签 + PERIOD 分钟），入场 = 1 分钟 K 中**开盘时刻等于 T** 的那一根，按其开盘价成交（找不到 → `no_bar`）。

| 出场 | 成交价（含滑点） | 出场时刻 |
|---|---|---|
| `time`，n 根 | 入场根之后第 n 根的开盘 | 该根**开盘** |
| `bracket` | 从入场根的下一根起逐根判：多头 high 判止盈、low 判止损（空头镜像），同根同时触发按止损；成交价 = 触发价，若该根开盘已越过触发价则按开盘；到 `max_bars` 按该根收盘 | 该根**收盘**（整根结束才确认） |
| 到 flat 仍持仓 | flat 时刻那根的开盘；该根不存在则用 flat 之前最后一根的收盘 | 该根开盘（后一种情况为收盘） |

滑点：入场 `open + 方向 × slippage_ticks × tick`；出场多头卖出减滑点、空头买回加滑点（默认 1 tick = 0.01）。出场逻辑只有一份实现：`zr.sim.stock.plan_exit`，正股模式与期权模式共用。

## 3 正股模式（`zr/sim/stock.py`）

- 复用 `zr.sim.engine` 的 `Account / Trade / DayResult`；权益逐笔累加、跨日延续，下一笔按新权益定股数。
- 股数 `size_shares`：有止损（bracket 的止损价）→ `floor(权益 × max_risk_pct / |入场 − 止损|)`；无止损 → `floor(权益 × max_notional_pct / 入场)`（`max_notional_pct` 默认 0.5，是 `size_shares` 的参数，`Account` 不含此字段）；再取 `min(…, floor(权益 / 入场), 100,000)`，不加杠杆。
- 费用 `zr.costs.fees.stock_order_fees`：开仓、平仓两侧各按其方向计算（多头 Buy 入 / Sell 出，空头 Sell 入 / Buy 出）。`pnl = 方向 × (出场 − 入场) × 股数 − 费用`。
- 允许做空，**不建模借券费**。
- 跳过（按此顺序，每个信号只计一次）：`no_bar`（无入场 K 或入场 K 不早于 flat）→ `outside_session`（入场 K 早于 first_entry 或晚于 last_entry）→ `in_position` → `cooldown`（平仓后不足 `cooldown_bars` 根 1 分钟 K）→ `max_entries` → `size_zero`。恒等式：`sum(skipped) + 成交笔数 == 信号数`。
- 引擎入口：`simulate_day(bars, signals, account, session, policy, *, exit_rule, symbol="")`，`bars` = DataFrame[ts_ns, open, high, low, close]（当日 RTH 1 分钟），`signals` = [(入场 K 下标, ±1)]。出场规则通过关键字参数 `exit_rule` 传入（信号元组不含规则）。

## 4 期权模式

1. 用 `plan_exit` 在正股 1 分钟 K 上算出该笔的出场时刻与原因；
2. 合约：最近可交易到期日（0 ≤ DTE ≤ `dte_max`，日历日），行权价最接近信号 K 收盘价的为平值，向虚值 `otm_steps` 档（看涨向上、看跌向下）；报价取 `options_1m`。无合约 → `no_contract`，**不回退到别的合约**；
3. 入场报价 = 信号时刻 + 2 秒之后 60 秒内的第一条报价：没有或无效 → `no_quote`；相对价差 > `max_spread_pct`（0.15）→ `wide_spread`；
4. [入场分钟, 出场分钟] 内该合约每分钟都要有报价行（数据缺口 → `no_quote`；只查数据有无，不按报价好坏剔除，避免把跌到 0 的单子筛掉）；
5. 交给 `zr.sim.engine.simulate_day`，出场 `{"kind": "at", "ts_ns": 正股出场时刻}`（不再偏移），`min_coverage = 0.0`、`max_entry_wait_s = 60`；引擎抛 `ValueError` → 该笔 `no_quote`，不中断整批；引擎的跳过原因原样计入；
6. 期权成交时刻早于正股出场时刻 → `early_exit`，剔除该笔。

账户层由 runner 承担（逐信号调用引擎，引擎的日级累计失效）：上一笔结束后的权益作为下一笔的 `equity_usd`；当日已实现亏损 ≥ 10% × 当日期初权益后不再开仓（`daily_loss`）；当日开仓次数上限（`max_entries`，默认 20，`StockPolicy.max_entries_per_day` 优先）；上一笔期权未平仓时的信号计 `in_position`。张数用引擎的 `size_position`（`Account` 默认 5% 风险预算、15% 权利金上限、10 张；at 出场无止损 → 按权利金口径），权利金高时会 `size_zero`。

跳过顺序：`no_bar → outside_session → daily_loss → max_entries → in_position → cooldown → no_contract → no_quote → wide_spread →`（引擎）`late_quote / size_zero → early_exit`。

**分辨率**：期权报价是 1 分钟快照；引擎的 2 秒延迟在 1 分钟报价下约等于延后到下一分钟（入场 = 信号后下一分钟报价的 ask，出场 = 正股出场时刻后下一分钟报价的 bid）。

## 5 输出

`trades.csv`：每个信号一行，`skip_reason` 为空的是成交。

`date, mode, symbol, contract, direction, signal_time, stock_entry_time, stock_entry_px, stock_exit_time, stock_exit_px, stock_exit_reason, option_entry_time, option_entry_px, option_exit_time, option_exit_px, qty, fees, pnl_usd, pnl_pct, skip_reason`

时间为美东 `HH:MM:SS`；`pnl_pct` = 盈亏 / 入场成本（正股：入场价 × 股数；期权：入场价 × 100 × 张数）。

`summary.md`：信号数、成交笔数、胜率、平均盈亏%、总美元（已扣费）、期初 / 期末权益、最大回撤、费用合计、跳过明细、数据分辨率与口径说明。

## 6 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rule`、`--symbol`、`--start`、`--end`、`--out` | 必填 | |
| `--mode` | stock | stock / option |
| `--equity` | 10000 | 期初权益（美元） |
| `--first-entry` / `--last-entry` / `--flat` | 09:30 / 15:40 / 15:50 | 美东时刻 |
| `--slippage-ticks` | 1 | 正股滑点（tick = 0.01） |
| `--cooldown-bars` | 0 | 平仓后冷却（1 分钟根数） |

**D-033**：命令行会剔除成交参数校准窗口（`config/calibration_window.yaml`，2026-06-01～08-31）内的交易日，剔除天数写进 summary。纯函数 `run_rule` 不做这一步。

## 7 不覆盖

借券费；盘中实时；随机对照（结果只是规则自己的盈亏，没有与随机入场比较，不能单独作为「有边」的证据）；期权逐笔成交（1 分钟快照）；正股的逐笔成交与排队（开盘价 + 固定 tick 滑点）。
