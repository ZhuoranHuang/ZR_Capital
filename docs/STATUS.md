# 状态

更新：2026-10-22 08:50 SGT · main

## 阶段

**W2 结束，W3 转为「录制 + 期权维度研究」，实盘推后**（D-017）。理由：正股分钟级特征的信号 ≤1.6 bp，
离单张 ATM 期权往返成本（≈6 USD/张，D-013 费用 + 2 tick 点差）差一个量级；元标签与条件分层都没把它变成正期望。
计划见 `docs/PLAN_W3.md`。

## W2 结论（2023-10-02～2026-10-21，SPY/QQQ/MU，正股 1 分钟 K）

| 问题 | 结论 | 文档 |
|---|---|---|
| 单检测器有没有边 | 8 个检测器 + 扫描 570 次试验：hit_rate 0.42–0.55，最大均值 ≤1.6 bp；t 最高 5.04（MU，0.9 bp）；扣费后全部为负（−3～−13 USD/张） | EVENT_LABEL_TABLE、SCAN_TABLE |
| E1 收紧能否降到 3–15 次/日 | 只有 MU 能（最佳 13.5/日、hit 0.49、t 3.22、−6.05 USD）；SPY/QQQ 最紧仍 ≥16/日 | SCAN_TABLE |
| 条件分层能否找到好格子 | 406 格中 t 最高 4.02（E3 MU 逆开盘方向，1.05 bp，n 1606）；无格子 hit ≥0.55；扣费全负 | CONDITIONAL_TABLE |
| 元标签能否挑出好事件 | 样本外 AUC 0.511（H=5）/ 0.506（H=15），最高十分位 −6.52 / −5.50 USD/张、50 个/日 | META_MODEL |
| 全样本特征有没有方向信息 | 月度 IC 0.005–0.010，正月 40–68% | BASELINE_IC |
| 网传 E4 / E9 | E4 无边（期权更差）；E9 正收益来自上涨市长期持有，全部跑输买入持有 | SCREENSHOT_STRATEGIES |
| 期权是不是贵 | SPY 0DTE 开盘 IV 均值 0.17，随后 RV 低 0.08–0.10，**22/22 天 RV<IV** | RV_VS_IV |

## 进行中

- **ThetaData 全量拉取**：2026-10-22 08:45 SGT 启动（`nohup caffeinate -i bash scripts/theta_full_pull.sh`），SPY→QQQ→MU，2023-10-02～2026-10-21，进度逐月写进 `docs/DATA_INVENTORY.md`；日志 `$ZR_DATA_ROOT/logs/theta_pull.log`。开头几天 ≈23 s/天。终端重启时客户端最多等 10 分钟后继续；超时或其他错误则停在当月，重跑脚本续传。
- **数据根已迁到外接 SSD**（APFS）：`/Volumes/ZHITAI TiPlus9100 2TB Media/zr_data`，4,141 个文件 / 345,158,096 字节核对一致；`.env` 只改了 `ZR_DATA_ROOT` 一行；同步、录制、全量拉取脚本经 `scripts/lib/data_root.sh` 解析，盘未挂载时报错退出。
- 录制器：VPS 10-22 13:20 UTC 起按 timer 运行（SPY/QQQ/MU，239 订阅）。

## 待办

1. **2026-10-25（SGT）删除旧数据目录** `./data_store`（确认 SSD 上数据完好、同步与拉取都写到 SSD 后）。
2. **全量拉取已停在 2024-01**（2026-10-21 18:11 PT）：负责人 18:12 重启终端时用的是旧版脚本（默认 SOCKS），子 JVM 带 `-DsocksProxyHost=127.0.0.1:1080`，gRPC 上游全部 `503 io exception`。脚本已改为默认直连（`--proxy` 才用 HTTP CONNECT）。**请负责人用新脚本重启终端**：`bash scripts/run_theta_terminal.sh`，确认 `curl http://127.0.0.1:25503/v3/option/list/expirations?symbol=SPY` 返回 200 后告诉我，我重跑 `scripts/theta_full_pull.sh` 续传（2023-12 的 4 个失败日会自动重拉）。
2b. **安全：Theta 终端的 REST/MCP 对局域网开放**——`host=127.0.0.1` 不限制监听（实测 `*.25503`，局域网 IP 访问 `/mcp/sse` 返回 200），macOS 防火墙关闭。三种处理方式见 DEPLOY.md「ThetaData」小节，均需 sudo，待负责人决定。
3. 全量到位后：在 2 年期权数据上重跑元标签（加期权特征），报告同样指标（PLAN_W3 线 a）。

## 待确认

1. **10-21 定稿的「门」的具体条件写进 DECISIONS**——仓库里没有书面记录，D-017 只引用了 W2 数字；请负责人给出原文，我补一条 D-018。
2. 账户净资产 ≈ 0，实盘前需入金（实盘已推后，不阻塞）。
3. 1 分钟 K 只能回溯到 2023-10-02；2023-01～09 若需要另找源。
4. D-008 蒙特卡洛脚本未入库（R-002）。
5. Mac 合盖或电池模式仍会睡眠；全量拉取期间请接电源、不合盖（`caffeinate -i` 只防空闲睡眠）。
6. **D-019 按字面实现后 MU 覆盖率 0%**（`zr/features/option_features.py` 的 `oi_daily`，2023-10-02～2025-01-08 共 320 天全为 NaN）：
   2026-07 前 MU 周度到期只有周五，相邻交易日的 DTE 集合（如周一 {4, 11}、周二 {3, 10}）不重叠。
   当前保持字面口径（特征为 NaN，模型按缺失处理）。备选：与**上周同星期**的同 DTE 合约比（DTE 集合一致），
   或按 DTE 区间分桶。请负责人选择是否改 D-019。
7. **防火墙**（2026-10-21 20:25 PT 复查）：防火墙已开启，但 `socketfilterfw --listapps` 显示 `~/thetadata/jdk/.../bin/java` 为「Allow incoming connections」；终端监听 `*:25503`，`config.toml` 的 host=127.0.0.1 不限制 MCP。本机访问自身局域网 IP 返回 200 不能作为证据（不经过应用防火墙），需从另一台设备访问 `http://<Mac 局域网 IP>:25503/mcp/sse` 复查。建议把 java 改为「阻止传入连接」（系统设置 → 网络 → 防火墙 → 选项）。
