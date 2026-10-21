# W1 计划：数据层

| Task | 内容 | 验收标准 | 状态 |
|---|---|---|---|
| 1.0 | 骨架：uv、pyproject、zr/、指标与测试、CLAUDE.md | ruff + pytest 通过；commit 在 `feat/bootstrap` | 完成 |
| 1.1 | API 核验：`zr/ops/api_facts.py` 只读跑一遍行情与交易接口 | `docs/API_FACTS.md` 生成；每项有结果或错误码；明确 (a) 期权 depth 是否返回 bid/ask (b) 1 分钟 bar 可回溯起点 (c) MU/AMD 到期日星期 (d) 行情包等级 | 脚本已写，等 .env |
| 1.2 | 录制器：`zr/data/recorder.py` + `scripts/run_recorder.sh` | `--dry-run` 打印订阅集；单元测试覆盖订阅集构造与 Parquet 写入去重；实跑一个交易日后 `docs/recorder_qa/` 有 QA 报告 | 代码与测试已写，等实跑 |
| 1.3 | 交易史：`zr/journal/trade_history.py --since 2026-05-01 --options-only` | 三个 parquet 落盘；charges 行数 = 有成交订单数；费用项 code/name/amount 完整 | 脚本已写，等 .env |
| 1.4 | 成本报告：按标的 / DTE / 时段统计 spread/mid、费用占权利金比例 | `docs/COST_REPORT_W1.md`，所有数字由 `zr/research/cost_report.py` 生成；给出 D-009 的判定 | 未开始，依赖 1.2 或 1.5 数据 |
| 1.5 | ThetaData loader：`zr/data/thetadata.py` | 按 DATA_SPEC 第 2 节落盘；抽样 3 天与录制器对比 bid/ask 差异 | 未开始，等账号 |

## 阻塞

- 1.1 / 1.3 需负责人填 `.env`（LONGPORT_*）
- 1.1 期权 depth 测试需开通期权行情包（具体包名在 1.1 输出 `quote_package_details` 后确认）
- 1.5 需 ThetaData 账号
