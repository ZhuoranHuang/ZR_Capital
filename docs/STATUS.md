# 状态

更新：2026-10-22 09:30 SGT · 分支 `feat/w2-conditional`

## 阶段

W2 第二步（条件分层 + 元标签）跑完。结论：**所有检测器 / 条件格子 / 元标签十分位在扣费后都为负**；
正股层面存在统计显著但 ≤1.6 bp 的弱信号，离覆盖期权费用（≈6 USD/张）差一个量级。ThetaData 已接入并验证。

## W2 第二步结果（2023-10-02～2026-10-21，SPY/QQQ/MU）

| 文档 | 内容 | 关键数字 |
|---|---|---|
| `docs/research/SCAN_TABLE.md` | R-006～R-009，570 次试验（233 s） | E1 收紧后 3–15/日只在 MU 可达（11 组），最佳 ε=0.05%/确认 1/趋势 E：13.5/日、hit 0.49、t 3.22、扣费 −6.05；SPY/QQQ 最紧仍 ≥16/日。全表 t 最高 5.04（MU，0.9 bp）。E9 抄底 MU 2.6/日 hit 0.52 t 3.68 扣费 −3.8；回撤 2% 版扣费为正但样本 ≤251 |
| `docs/research/SCREENSHOT_STRATEGIES.md` | R-010/R-011 两个网传策略原规则回测 | E4：SPY −4031 / QQQ −3307 / MU +2547 USD（费用 1.4 万），期权表达三者全负（−3.4 万/张）。E9 YaleV3：三标的为正（+4.9 万）但持仓 10–114 日、盈亏比 0.3–0.4，全部跑输同期买入持有（SPY +81% / QQQ +102% / MU +1246%）；期权表达不适用 |
| `docs/research/CONDITIONAL_TABLE.md` | R-012，406 格、报告 220 格（n≥200） | t 最高 5 格：E3 MU 逆开盘方向 t 4.02（1.05 bp，n 1606）、E1 MU 跳空向上 3.34、E3 QQQ 逆开盘方向 3.28、E1 MU 跳空向下 3.15、E3 QQQ 11:30–14:00 3.03；hit ≤0.55，扣费全负 |
| `docs/research/META_MODEL.md` | R-013，sklearn HGB（LightGBM 缺 libomp 回退），purged WF 12m/1m/1d | H=5 样本外 AUC 0.511（25 月，96% >0.5），最高十分位 hit 0.506、**−6.52 USD/张、50/日**；H=15 AUC 0.506、−5.50 USD。校准贴线。MDA：bb_pct_b、macd_hist_atr、min_to_close、vwap_dist_atr |
| `docs/research/BASELINE_IC.md` | R-014 全样本基线 | 月度 Spearman IC 0.005–0.010，正月 40–68%，t ≤ 2.0 |

## ThetaData 验证（R-015）

- 终端：Options STANDARD，REST 127.0.0.1:25503；`scripts/run_theta_terminal.sh` 已改为先 export 用户目录 JDK 的 PATH/JAVA_HOME（find 定位 bin）。
- SPY 2026-09 全月 + 10-21：22 天、quotes 313 万行、**690 s（31 s/天）、42 MB（1.9 MB/天）**；全量估算 SPY 6.8 h / 1.5 GB，三标的 ≈ 20 h / 4.5 GB。
- 与录制器同合约同分钟对比（10-21，5797 个合约分钟）：对齐 Theta t ↔ 录制器 [t−1,t) 后 **精确相等 43%、1 tick 内 66%、平均 |Δ| 0.02**；长桥 Depth 是单档非 NBBO，差异集中在收盘分钟与深价外。
- RV−IV：09:31 ATM 0DTE IV 均值 0.17，随后 1 小时 RV 均值低 0.08、全天低 0.10，**22/22 天 RV<IV**，开盘半小时最接近。`docs/research/RV_VS_IV.md`
- 发现并修的接口事实：响应是 `{response:[{contract,data}]}` 嵌套、当日数据必须指定 expiration、无数据可能 200+文本、1m 时间戳语义（t 时刻 NBBO）。

## 阻塞 / 待确认

0. **ThetaData 是否全量拉取**（SPY/QQQ/MU 2023-10 至今，≈20 h / 4.5 GB，可分标的后台跑）。
0b. `~/thetadata/config.toml` 把 `host` 改为 127.0.0.1（diff 已打印等确认）；MCP server 与 REST 同端口且无关闭开关，改 host 即限制在本机。
1. 账户净资产 ≈ 0（上轮发现），实盘前需入金。
2. 1 分钟 K 只能回溯到 2023-10-02；2023-01～09 若需要另找源。
3. D-008 蒙特卡洛脚本未入库（R-002）。
4. **ThetaData 首次登录（阻塞验证）**：JDK 21 与 `ThetaTerminalv3.jar` 已就位（`~/thetadata/`，无需 sudo）。请负责人在自己的终端运行 `bash scripts/run_theta_terminal.sh` 并登录（或 `--api-key`）。登录后我跑 SPY 2026-09 验证：`uv run python -m zr.data.thetadata --symbols SPY.US --start 2026-09-01 --end 2026-09-30`，再 `uv run python -m zr.data.thetadata --symbols SPY.US --start 2026-10-21 --end 2026-10-21` + `uv run python -m zr.data.theta_vs_recorder --date 2026-10-21`，结果写 DATA_INVENTORY 后停下由负责人决定全量。

## 下一步

1. `feat/thetadata`：读官方文档 → 装 JDK 21 + ThetaTerminal → `zr/data/thetadata.py` → SPY 2026-09 验证 → 停下由负责人决定全量。
2. W2 第二步：参数扫描 + 元标签特征表（事件 × 特征 × 标签）；E8 门控等 OI。
3. 10-22 13:20 UTC timer 启动的录制器应写到 `~/zr_data/`（新 unit 已装）；healthcheck 盘中应无 stale 报警。
