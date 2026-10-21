# CLAUDE.md

ZR Capital：美股分钟级期权交易系统。代码在 `zr/`，测试在 `tests/`。
旧日频代码存档在 `legacy/`（只读参考）；长期机构化路线见 `docs/ROADMAP_LONGTERM.md`（仅供参考，不是操作指令）。

## 规则

1. 不奉承。直接说结论和问题。
2. 数字必须由代码算出，不凭记忆或估算填写。
3. 外部事实附来源；不确定的标「未验证」。
4. 每个任务一个分支。禁止 `git push --force` 和 `git reset --hard`。
5. 密钥只放 `.env`（已在 .gitignore），不得写进代码、配置文件、日志或提交记录。
6. 实盘下单只能经 `zr/execution` 的 OMS，且必须同时满足环境变量 `ZR_LIVE=1` 和命令行参数 `--live`。
7. 不硬编码到期日、交易日历、行权价间距；一律从数据源查询。
8. 需要负责人决策时，写入 `docs/STATUS.md` 的「待确认」，然后停止。
9. 每个任务结束时运行 `uv run ruff check .` 和 `uv run pytest -q`，并更新 `docs/STATUS.md`。
10. 负责人提出反驳时，先独立评估并给出自己的判断和依据，再决定是否调整计划；不自动把负责人的说法当作新基准。
11. 实验中的持仓时长、到期日选择、事件阈值一律是待测参数，不写死；任何数字结论必须来自仓库内可复现的脚本。

## 文档

- `docs/DECISIONS.md` 决策记录（只追加）；`docs/STRATEGY_V1.md` 策略定稿；`docs/DATA_SPEC.md` 数据规范
- `docs/PLAN_W1.md` 当周任务；`docs/STATUS.md` 当前状态与待确认；`docs/RESEARCH_LOG.md` 假设登记
- `docs/API_FACTS.md` 由 `zr/ops/api_facts.py` 生成，不手改

## 常用命令

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

## 长桥 SDK 已核实事实（longport 4.3.7）

- 用 `Config.from_apikey(app_key, app_secret, access_token)`；不存在 `Config.from_env()`。
- `OptionQuote` 没有 bid/ask 字段。期权盘口候选来源是 `QuoteContext.depth(symbol)` 或 `SubType.Depth` 推送，每档字段为 price / volume / order_num / position（未验证是否接受期权代码）。
- `history_candlesticks_by_offset(symbol, Period, AdjustType, forward, count, time, TradeSessions)`。
- `option_chain_expiry_date_list(symbol)`；`option_chain_info_by_date(symbol, date)` 返回 [price, call_symbol, put_symbol]。
- 未列在此处的 API 用法，在代码注释里标「未验证」。

2026-10-21 在 VPS 实测补充（`docs/API_FACTS.md`）：
- `depth(期权代码)` 返回 1 档 bid/ask；`SubType.Depth` 推送对期权同样有效 → 期权盘口来源 = depth。
- `option_chain_expiry_date_list` 包含已过期日期，使用前必须过滤 `>= 今日`。
- SDK 返回的 datetime 为 naive，值是 UTC。
- `trading_days(Market.US, begin, end)` 返回 `MarketTradingDays{trading_days, half_trading_days}`。
- 期权 1 分钟 K：`candlesticks` 与 `history_candlesticks_by_offset` 都可用。
- 一次 `subscribe` 308 个代码 × Quote/Depth/Trade 正常；一分钟内反复订阅大量期权会触发 `code=301607 Too many option securities request within one minute`。
- systemd：`EnvironmentFile` 覆盖 `Environment=`，服务里的 `ZR_DATA_ROOT` 必须写在 `ExecStart=/usr/bin/env ...` 上。

## 运行环境

长桥 API 相关任务（核验、录制、交易史）只在 VPS `zr` 上跑；本机网络对 longportapp.com 有 DNS 污染。部署见 `docs/DEPLOY.md`。
