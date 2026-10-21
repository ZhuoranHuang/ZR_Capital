# 状态

更新：2026-10-21 15:55 ET · 分支 `feat/w1-analysis`（VPS 已检出同一分支；录制器进程仍是 18:46 UTC 启动的旧代码，16:05 ET 退出后 deploy）

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
- QQQ、MU 已完成，最早同为 **2023-10-02 08:00 UTC**（各 747k 行）；AMD 运行中（19:46 UTC 到 2026-08-25）。`docs/DATA_INVENTORY.md` 为当前快照；收盘任务会在 VPS 重新生成到 `~/zr_data/reports/DATA_INVENTORY.md`

## 录制与同步

- 10-21 录制器 14:46 ET 起一直是同一进程（NRestarts=0），**全程写到 `~/ZR_Capital/data_store/`**：18:52 的 deploy 因 VPS 工作区脏文件静默失败，新 unit 没装上（DEPLOY_LOG 有更正）。已手动装好新 unit 并验证环境变量生效；收盘后由 VPS 一次性 systemd 任务 `zr-post-close-20261021`（20:10 UTC 触发，`deploy/post_close.sh`）完成：等 QA → 并入 `~/zr_data/` → deploy `feat/w1-analysis` → 回填进度与清单；日志 `~/zr_data/logs/post_close_2026-10-21.log`，明天读它补 QA 数字
- 录制器新增静默看门狗：盘中 120 秒无推送则清空本地订阅状态强制全量重订，次数写入 QA
- `deploy.sh` 改为 checkout 前自动 stash、每步显式报错；`api_facts` 默认输出改到 `ZR_DATA_ROOT/reports/`
- `deploy/sync_from_vps.sh`：11.5 秒、426 个文件、7.3 MB（第一次全量）；journal 二次同步 <5 秒
- 本机网络已改为经 VPS 跳板（负责人确认）：SOCKS 常驻 ✓、`zrproxy` ✓（PyPI 200、import longport ok）、launchd 每日同步已装并手动触发成功（4 文件 13.8 MB）。**GitHub SSH 还差一步**：公钥 `zr-lightsail` 未加到 GitHub 账号，`git ls-remote` 报 publickey；加之前 push 走 `zrproxy git push https://…`
- launchd 同步 Hour=9 按本机时区 US/Pacific 触发（= 新加坡 00:00/01:00），等负责人定

## 待确认

1. 把 `~/.ssh/id_ed25519.pub`（zr-lightsail）加到 GitHub 账号；launchd 同步 Hour 取 9（太平洋）还是改系统时区
2. 1 分钟 K 只能回溯到 2023-10：是否接受，或为 2023-01～09 另找数据源
3. 账户净资产 ≈ 0（上轮）
4. D-001 保证金规则人工客服确认；D-008 蒙特卡洛脚本未入库

## 下一步

1. 读 `ssh zr 'cat ~/zr_data/logs/post_close_2026-10-21.log'`，把 QA（每标的消息数、最大间隔、缺口）与 deploy 结果写进 STATUS；确认 10-22 13:20 UTC timer 启动的录制器写到 `~/zr_data/`
2. 公钥加到 GitHub 后验证 `git ls-remote`
3. Task 1.4 成本报告（费用公式 + 录制器 spread）；Task 1.5 ThetaData
