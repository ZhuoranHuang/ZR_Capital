# 状态

更新：2026-10-21 14:50 ET · 分支 `feat/deploy`（VPS 已部署同一 commit）

## 阶段

W1 数据层。VPS 上线，长桥 API 全部核验通过，交易史已导出。录制器今日 14:46 ET 手动启动跑到 16:05，明日起由 timer 接管。首次启动因 systemd 环境变量优先级问题写错目录（已修复，见 DEPLOY_LOG），数据已并入 `~/zr_data/recorder/`。

## 完成项

- 部署：`deploy/`（bootstrap、deploy/rollback、systemd ×3、sync、launchd）、`docs/DEPLOY.md`、`docs/DEPLOY_LOG.md`
- 录制器时段逻辑（trading_days 判交易日、9:25 ET 订阅、16:05 ET 退出）、`zr/ops/healthcheck.py`、`zr.data.universe` 周快照 CLI、`zr/ops/redact_api_facts.py`
- Task 1.1 API 核验：`docs/API_FACTS.md`（脱敏后入库，28 项全部成功）
- Task 1.3 交易史：VPS `~/zr_data/journal/{orders,executions,charges}.parquet`
- `.env`：本机与 VPS 各一份（600），从未入库
- 测试 39 个，ruff 通过

## Task 1.1 关键结论（2026-10-21 14:43 ET 盘中，VPS 运行）

| 问题 | 结论 | 依据（API_FACTS.md 小节） |
|---|---|---|
| 期权行情权限 | **生效**。`quote_package_details` 含 "OPRA US options real-time quotes with best bid and offer, for OpenAPI only"（US_OPRA…，有效期至 2026-11-19）；`quote_level` 中 USOP=LV1 | quote_package_details / quote_level |
| depth() 对期权是否返回 bid/ask | **是，1 档**。SPY261021C774000.US：bid 0.81×232，ask 0.82×235，`order_num`=0 | depth(SPY261021C774000.US) |
| Depth 推送对期权 | **是**。订阅 SubType.Depth 收到该合约 bid 0.81 / ask 0.82 | subscribe 20s → depth_sample |
| 20 秒订阅计数（SPY + 1 张 0DTE call） | quote 117 / depth 99 / trade 115 | subscribe 20s |
| 正股 1 分钟 K 回溯 | 4 标的各翻 30 页 × 1000 根 = 30000 根，到 2026-09-09 仍有数据，**上限未探到**；含 Pre/Post 时段 | earliest_minute_bar(*)（第一次运行，`--skip-bars` 前） |
| 期权 1 分钟 K | `candlesticks(Min_1,100)` 返回 100 根；`history_by_offset(Min_1,1000)` 返回 1000 根，最早 2026-10-19 15:34 UTC（合约挂牌后全程可取） | candlesticks / history_candlesticks_by_offset |
| MU/AMD 到期日 | **周一 / 周三 / 周五都有**：10-16 五、10-19 一、10-21 三、10-23 五、10-26 一、10-28 三。SPY/QQQ 每个交易日 | option_chain_expiry_date_list |
| 账户类型 | **真实账户**：2026-06-09～08-19 有 218 笔真实成交，费用项含 CAT/OCC/SFC 等监管费。当前净资产数量级 10^-1 HKD、持仓空、近 30 天无成交——**账户基本为空，请负责人确认是否为预期账户** | account_balance（脱敏）/ stock_positions / history_executions |
| 新发现 | `option_chain_expiry_date_list` **包含已过期日期**，必须过滤；SDK 时间戳为无时区 naive，值为 UTC（quote timestamp 18:43:49 = VPS UTC 时间） | nearest_atm_call 第一次误选 10-15；option_quote.timestamp |

## Task 1.3 交易史（`--since 2026-05-01 --options-only`）

由 VPS 上 pandas 统计（命令见 DEPLOY_LOG）：

- 订单 331：Filled 218 / Canceled 92 / Rejected 16 / Expired 5
- 成交 288 笔，974 张，日期 2026-06-09 ～ 2026-08-19
- 按标的（成交笔数 / 张数）：MU 232 / 730，DRAM 39 / 116，QQQ 9 / 102，NVDA 3 / 4，TQQQ 3 / 20，META 2 / 2。**预期"MU 最多其次 QQQ"：MU 对，第二是 DRAM 不是 QQQ**
- 费用合计 1039.86 USD（218 笔有成交订单的 `charge_detail.total_amount` 之和）
  - 佣金 504.05、平台费 292.20、交易所费 175.32、清算费 24.94、监管费 23.39、SFC 15.69、CAT 2.18、TAF 2.09
  - **实测全部费用 1.0676 USD/张/单边**（1039.86 / 974）；佣金 0.5175/张；平台费 0.30/张（与 D-006 官网费率一致）。D-006 的 0.98 估计偏低约 9%

## 定时器（VPS，UTC）

| timer | 下次触发 |
|---|---|
| zr-healthcheck | 每 5 分钟（18:50 UTC 起） |
| zr-recorder | Thu 2026-10-22 13:20 UTC |
| zr-universe | Fri 2026-10-23 21:30 UTC |

universe 快照手动跑过一次：真实期权链解析出 308 个订阅，与 dry-run 一致。

## 待确认

1. 账户：API key 对应账户净资产 ≈ 0、无持仓；是否为准备实盘的账户？
2. 正股 1 分钟 K 回溯上限：需要专门回填脚本一次性翻到头（预计数百页），是否现在做？
3. `legacy/` 是否删除。
4. D-001 保证金规则需人工客服确认。
5. D-008 蒙特卡洛脚本未入库（R-002）。
6. 本机网络：DNS 污染（见 DEPLOY_LOG）；API 相关工作一律在 VPS 跑。

## 下一步

1. 明日看 `~/zr_data/recorder_qa/2026-10-22.md` 与 alerts，确认 timer 全流程。
2. Task 1.4 成本报告：用 journal 的 974 张实测费用 + 录制器 spread 数据。
3. Task 1.5 ThetaData loader。
