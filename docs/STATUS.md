# 状态

更新：2026-10-21 15:35 ET · 分支 `feat/w1-analysis`（VPS 已检出同一分支，录制器进程仍是上午的代码，16:05 ET 退出后下次 timer 用新代码）

## 阶段

W1 数据层收尾。费用模型已用真实成交逐项验证为精确；交易史往返分析出炉；正股 1 分钟 K 回填 SPY 完成、其余运行中。

## 完成项（本轮）

- `legacy/` 已删除（负责人确认），审计记录 `docs/AUDIT.md`，commit 4a2c1f5 之前可恢复
- `zr/costs/fees.py` + `validate_fees.py` → `docs/FEE_VALIDATION.md`，D-013
- `zr/journal/analyze_history.py` → `docs/TRADE_HISTORY_REPORT.md`；`trade_history.py` 现在从 `order_detail` 取拒绝原因
- `zr/data/backfill_bars.py`（断点续传、按年 parquet）；`docs/DATA_INVENTORY.md`
- `deploy/sync_from_vps.sh` 修复 macOS rsync 参数；`docs/DEPLOY.md` 增「网络」一节
- 测试 48 个，ruff 通过

## 费用公式验证（`docs/FEE_VALIDATION.md`）

218 笔有成交订单、8 个费用项、逐项残差 **全部为 0**（按订单计最低收费与四舍五入，H1）。按成交笔数各自计最低收费（H2）在 41 笔多笔成交订单上残差绝对值和 77.71 USD，最大单笔 5.29。**结论：最低收费按订单计，不按成交笔数。** 往返分析里公式费用与实际费用合计同为 903.59 USD。

## 交易史分析（`docs/TRADE_HISTORY_REPORT.md`，2026-06-09 ～ 08-19）

- FIFO 往返 157 笔 / 425 张；未配对单边另列不计入
- **胜率 0.57，盈亏比 0.73**（平均盈利 442.87 / 平均亏损 −603.45），平均每笔 −10.32，合计 **−1619.59 USD**（扣费后）
- 按标的：MU 125 笔 +1820.31（胜率 0.59，盈亏比 0.73）；DRAM 24 笔 −2698.49；QQQ 3 笔 −985.79；其余零星
- 按 DTE：0DTE 23 笔 +6572.61（胜率 0.48，盈亏比 2.1）；**1DTE 73 笔 −8402.90（胜率 0.59，盈亏比 0.41）**
- 按持仓：≤30 分钟三档合计 +11,837；**30–60 分钟 12 笔 −11,398（胜率 0.25）**；>4h 29 笔 −1373
- 被拒 16 笔：原因 12 笔 "The order amount exceeds the maximum buying power"、4 笔 "Insufficient holdings…"；DTE=0 且 11:00 ET 后只有 2 笔（7-27 14:31、14:45）——**不集中在到期日 11:00 后**，主要是资金不足
- 撤销 92 笔：09:30–11:30 ET 共 40 笔（10:00 桶 15 笔最多），盘前 02:30–09:00 共 14 笔

这些数字是历史手工交易的描述，不是策略结论；进入策略的只有费用公式（D-013）和"1DTE 盈亏比 0.41 / 30–60 分钟持仓亏损集中"两个待验证线索（登记 RESEARCH_LOG R-003）。

## 正股 1 分钟 K 回填（VPS，`zr/data/backfill_bars.py`）

- **SPY 最早 2023-10-02 08:00 UTC**（04:00 ET 盘前）；780 页 779,200 根，去重后 747,019 行（2023: 62,400 / 2024: 251,460 / 2025: 239,460 / 2026: 193,699）；约 9.5 分钟
- 服务端在此之前不再返回数据（`stop_reason=no_progress`），即长桥 1 分钟 K 回溯约 2 年；DATA_SPEC 的"2023-01 至今"改为"2023-10 至今"，2023-01～09 若需要只能靠其他数据源
- QQQ / MU / AMD 排队运行中（19:28 UTC 起），完成后重新生成 `docs/DATA_INVENTORY.md`

## 录制与同步

- 10-21 录制器 14:46 ET 起跑（含路径修复后的重启），16:05 ET 退出后生成 `~/zr_data/recorder_qa/2026-10-21.md`——**本次更新时尚未到点，QA 数字待下次补**
- `deploy/sync_from_vps.sh`：11.5 秒、426 个文件、7.3 MB（第一次全量）；journal 二次同步 <5 秒
- launchd 每日同步任务：待负责人确认后安装（本机时区是 US/Pacific，不是新加坡；09:00 local = 新加坡 00:00，需决定改 Hour 还是改系统时区）

## 待确认

1. 本机网络三处修改（SOCKS 常驻、ssh config github ProxyJump、zshrc zrproxy）+ launchd 同步任务：内容已打印，等"确认"
2. 1 分钟 K 只能回溯到 2023-10：是否接受，或为 2023-01～09 另找数据源
3. 账户净资产 ≈ 0（上轮）
4. D-001 保证金规则人工客服确认；D-008 蒙特卡洛脚本未入库

## 下一步

1. 网络配置写入并验证；装 launchd 同步
2. 等 QQQ/MU/AMD 回填完成 → DATA_INVENTORY；10-21 QA 报告写入 STATUS
3. Task 1.4 成本报告（费用公式 + 录制器 spread）；Task 1.5 ThetaData
