# 状态

更新：2026-10-21 · 分支 `feat/w1-data`（基于 `feat/bootstrap`）

## 阶段

W1 数据层。代码骨架与三个数据脚本已写完并通过 ruff + pytest（22 用例），均未在真实 API 上跑过。

## 完成项

- Task 1.0 骨架：uv + Python 3.12、`pyproject.toml`、`zr/` 包、`zr/features/indicators.py` 与 11 个指标测试、`CLAUDE.md`（11 条规则）
- 旧日频代码迁入 `legacy/`，旧 CLAUDE.md 迁入 `docs/ROADMAP_LONGTERM.md`
- 文档：`DECISIONS.md`（D-001–D-010）、`STRATEGY_V1.md`、`RESEARCH_NOTES.md`、`DATA_SPEC.md`、`PLAN_W1.md`、`RESEARCH_LOG.md`
- `zr/broker/serialize.py`：SDK 对象 → Python / JSON
- Task 1.1 `zr/ops/api_facts.py`：只读核验脚本，输出 `docs/API_FACTS.md`（未运行）
- Task 1.2 `zr/data/recorder.py` + `scripts/run_recorder.sh`：订阅集构造、Parquet 落盘去重、QA 报告；`--dry-run` 输出 308 个订阅；6 个单元测试
- Task 1.3 `zr/journal/trade_history.py`：分段拉取订单 / 成交 / 费用，节流 1.05 s；5 个单元测试

## 阻塞

- `.env` 已从模板复制，值为空。1.1 / 1.3 / 录制器实跑都需负责人填 `LONGPORT_*`。
- 期权 depth 是否返回盘口依赖期权行情包；先跑 1.1 看 `quote_package_details`。
- `git push` 需要 GitHub 认证（keychain 无凭据、无 SSH key）；`feat/bootstrap` 与 `feat/w1-data` 均未推送。

## 待确认

- `legacy/` 是否删除（当前保留为只读参考）。
- D-001 的保证金规则来自 AI 客服，需人工客服确认 14:00 ET 后是否解除。
- D-008 依据的 BS 蒙特卡洛脚本尚未入库（R-002），入库前该依据标「未验证」。

## 下一步

1. 负责人填 `.env`，运行 `uv run python -m zr.ops.api_facts`，审阅 `docs/API_FACTS.md`。
2. 按 1.1 结果修正 recorder 中标「未验证」的调用，交易日实跑 `scripts/run_recorder.sh`。
3. 运行 `uv run python -m zr.journal.trade_history --since 2026-05-01 --options-only`。
4. Task 1.4 成本报告。
