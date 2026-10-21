# legacy/ 审计与删除记录

`legacy/` 是 2026-10 之前的日频 MA+RSI 原型（66 个文件，180 KB），2026-10-21 由负责人确认删除。
删除前最后一个包含它的 commit 可用 `git log --diff-filter=D -- legacy` 找到；恢复：`git checkout <该 commit>^ -- legacy/`。

## 内容清单

| 路径 | 内容 | 新系统对应 |
|---|---|---|
| `legacy/backtest/{engine,metrics}.py` | 单资产逐行回测循环；Sharpe / Calmar / 回撤 / 胜率 / profit factor | 未迁移。新回测在 `zr/backtest/`（W2+）按事件 + 元标签设计重写 |
| `legacy/strategy/technical/ma_rsi_strategy.py` | 金叉死叉 + RSI 过滤 + 止损 | 方法论已被 D-010 否定（固定阈值指标组合），不迁移 |
| `legacy/utils/indicators.py` | SMA、Wilder RSI、cross_signal | 已迁移并扩展为 `zr/features/indicators.py`（EMA、MACD、布林、ATR、VWAP、斜率，11 个测试） |
| `legacy/data/downloader.py` | yfinance 适配 + 24h parquet 缓存 | 不迁移。数据源改为长桥 API（VPS）与 ThetaData |
| `legacy/config/` | YAML 策略参数 + loader | 不迁移。新配置：`config/universe.yaml` + `zr/config.py`（pydantic-settings） |
| `legacy/main.py` | 端到端 demo（AAPL 日线） | 不迁移 |
| `legacy/{dashboard,execution,monitor,portfolio,research,risk,tests}/` | 占位目录，仅 `__init__.py` | `zr/` 下同名包重建 |
| `legacy/ARCHITECTURE.md`、`legacy/requirements.txt` | 旧架构说明、依赖 | 被 `docs/`、`pyproject.toml` 取代 |

## 2026-05-21 代码审计结论（摘自 `docs/ROADMAP_LONGTERM.md`，保留供参考）

Phase 1 回测（6 笔交易，Sharpe −0.41）不可信：
1. 前视偏差：信号用第 i 日数据、在第 i 日收盘成交（已在 541e5af 修为次日开盘）。
2. 裸 `except:` 吞异常。
3. RSI 用简单均值而非 Wilder 平滑（已修）。
4. RSI 除零未处理（已修）。
5. yfinance MultiIndex 列补丁（已修）。
6. `profit_factor` 实际算的是盈亏比（已修）。
7. 无风险利率硬编码 0.04。
8. 全进全出仓位、无成本模型、止损用收盘价——设计简化。

这些经验已转化为新系统规则：指标无前视（`tests/test_indicators.py` 的 ATR 无前视测试）、费用用实测（D-012 / D-013）、仓位分批（D-011）。
