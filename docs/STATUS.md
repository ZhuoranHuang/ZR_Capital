# 状态

更新：2026-10-21 12:30 ET · 分支 `feat/w1-data`

## 阶段

W1 数据层。代码与文档齐备（ruff + pytest 26 用例通过）。**Task 1.1 / 1.3 实跑被网络阻塞**，见下。

## 完成项

- Task 1.0 骨架（`feat/bootstrap`）
- 文档：DECISIONS D-001–D-011、STRATEGY_V1（第 4、6 层已按 D-011 修订）、RESEARCH_NOTES、DATA_SPEC、UNIVERSE_SPEC、PLAN_W1、RESEARCH_LOG
- `config/universe.yaml` + `zr/data/universe.py`：标的池单一来源，`ZR_SYMBOLS` 作子集过滤；录制器已改为从 universe 读规则与上限
- `zr/ops/api_facts.py`、`zr/journal/trade_history.py`、`zr/data/recorder.py`：已写，未在真实 API 上成功跑过
- `.env` 检查：被 `.gitignore` 忽略，`git ls-files .env` 为空，从未入库

## 阻塞：长桥 API 不可达（2026-10-21 12:15 ET 实测）

`uv run python -m zr.ops.api_facts` 已跑，`docs/API_FACTS.md` 中全部 19 项失败，错误一致：
`error sending request for url (https://openapi.longport.cn/v1/socket/token): client error (Connect)`。

诊断（`dig` / `curl`，本机）：

| 主机 | 结果 |
|---|---|
| `openapi.longport.cn`（SDK 按 geotest 选中的默认） | 系统 / 1.1.1.1 / 8.8.8.8 均无 DNS 记录 |
| `openapi.longportapp.com`（官方文档默认） | 三个解析器各返回不同的无关 IP（Dropbox / Facebook 网段），连接超时 — DNS 污染特征 |
| `openapi-quote.longportapp.com`、`openapi-trade.longportapp.com` | 同上 |
| `openapi.longbridge.global` | 可达（401 token empty），但无官方文档背书，且其 quote/trade WS 子域不解析，不能用 |

本机有 5 个 utun 隧道，解析器 100.70.255.254。结论：SDK 和代码没问题，是当前网络路径（VPN / DNS）的问题。官方端点见 [longport 4.3.7 crate 文档](https://docs.rs/crate/longport/4.3.7)：`https://openapi.longportapp.com`、`wss://openapi-quote.longportapp.com/v2`、`wss://openapi-trade.longportapp.com/v2`，可用 `LONGPORT_HTTP_URL` / `LONGPORT_QUOTE_WS_URL` / `LONGPORT_TRADE_WS_URL` 覆盖。

以下结论因此**全部未得出**：期权行情权限、depth() 对期权代码是否返回 bid/ask、20 秒订阅计数、1 分钟 K 回溯起点、MU/AMD 到期日星期、交易史订单 / 成交 / 费用统计。

## 待确认

1. **网络**：由负责人决定——关掉 VPN 或改 DNS 后重跑；或确认账户归属区域并提供正确端点写入 `.env`（`LONGPORT_HTTP_URL` 等三项）。我不改系统 DNS / VPN 配置。
2. `legacy/` 是否删除。
3. D-001 保证金规则需人工客服确认。
4. D-008 依据的蒙特卡洛脚本未入库（R-002）。
5. GitHub push 认证（两个分支均未推送）。

## 下一步（网络恢复后）

1. `uv run python -m zr.ops.api_facts`（盘中跑，订阅计数才有意义），脱敏后 commit `docs/API_FACTS.md`。
2. `uv run python -m zr.journal.trade_history --since 2026-05-01 --options-only`。
3. 交易日实跑 `scripts/run_recorder.sh`。
