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
