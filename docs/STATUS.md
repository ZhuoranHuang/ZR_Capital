# 状态

更新：2026-10-22 08:45 SGT · main

## 阶段

W1 收尾完成；W2 第一步（事件 × 标签）已跑完三标的；ThetaData 代码就绪、验证等负责人登录。标的收敛为 **SPY / QQQ / MU**（D-016）。

## W2 第一步结果（`docs/research/EVENT_LABEL_TABLE.md`，2023-10-02～2026-10-21，三标的 24 秒）

- 代码：`zr/features/bars.py`（2/5 分钟聚合、成交量 bar）、`zr/features/features.py`（23 个正股特征，逐一无前视测试）、`zr/research/{events,labels,run}.py`、`config/{events,research_w2}.yaml`；96 个测试。
- 触发频率（次/日，SPY/QQQ/MU）：E1 153/137/64（eps 太松）、E2 0.54/0.49/0.45、E3 4.4/4.2/3.9、E4 0.36/0.33/0.35、E5 35/34/34、E6 0.31/0.34/0.39、E7 ≈1、E8 9.8/9.3/8.8。
- 原始成功率（hit_rate）全部在 0.42–0.54 之间；没有一个检测器在三标的上稳定 >0.5。
- 响应曲线见顶：E2 MU 15 分钟（+4.5 bp）、E3 MU 10 / SPY 15 分钟、E4 MU 5 分钟、E5 1–3 分钟即见顶后转负、E6 MU 10 分钟、E7 5–10 分钟、E8 QQQ 30 / SPY 15 分钟。
- 扣费后近似 P&L（单张 ATM、delta 0.5、D-013 费用 + 2 tick 点差）几乎全为负（−4～−13 USD/张）；唯一为正的是 E4 MU H=15/30（+1.9 / +4.9 USD，样本 274，不可信）。
- R-004：H=30 vs H=5 无一致方向，交易史「持仓越长越差」未在事件标签上复现。R-005：按离场时长分桶的差异是选择偏差，不作证据。
- 结论：当前八个检测器在正股层面都没有可覆盖期权费用的 alpha；下一步是参数扫描（E1 eps、E2 15 分钟版、E5 k）、E8 加 gamma 门控（需 ThetaData OI）、以及元标签而不是单检测器。

## W1 收尾（2026-10-22）

- 10-21 录制 QA（post_close 任务 20:10 UTC 执行，exit 0）：4044 文件 / 1,673,430 行 / 去重 102,750，重连 0、回调异常 0。正股消息数 SPY 12878/15047/12949、QQQ 12368/15239/12409、MU 13407/10558/13459、AMD 11670/7932/11716（quote/depth/trade），正股最大间隔 ≤12 s、无 >60 s 缺口。详表 `docs/DATA_INVENTORY.md`。会话数据已并入 `~/zr_data/recorder/`（76 MB），deploy `feat/w1-analysis` 在 post_close 中成功。
- 回填：SPY/QQQ/MU/AMD 全部到 2023-10-02（服务端上限），AMD 19:55 UTC 正常结束，数据保留不续跑。
- 标的集合：`.env.example`、`zr/config.py` 默认、`config/universe.yaml` 注释、VPS 与 Mac `.env` 的 `ZR_SYMBOLS` 行 → `SPY.US,QQQ.US,MU.US`；dry-run 订阅集 239（上限 450）。AMD 留在候选池。
- D-014（到期日保证金，长桥 FAQ Q3/Q6，0DTE 仓位减半硬规则）、D-015（tick 规则 → `zr/execution/ticks.py`，5 个测试）、D-016（标的收敛）。
- Mac 同步：`sync_from_vps.sh --now` 3626 文件 63 MB 14.5 s；launchd 每小时触发、SGT 09 点门控。
- 测试 53 个，ruff 通过。

## 阻塞 / 待确认

1. 账户净资产 ≈ 0（上轮发现），实盘前需入金。
2. 1 分钟 K 只能回溯到 2023-10-02；2023-01～09 若需要另找源。
3. D-008 蒙特卡洛脚本未入库（R-002）。
4. **ThetaData 首次登录（阻塞验证）**：JDK 21 与 `ThetaTerminalv3.jar` 已就位（`~/thetadata/`，无需 sudo）。请负责人在自己的终端运行 `bash scripts/run_theta_terminal.sh` 并登录（或 `--api-key`）。登录后我跑 SPY 2026-09 验证：`uv run python -m zr.data.thetadata --symbols SPY.US --start 2026-09-01 --end 2026-09-30`，再 `uv run python -m zr.data.thetadata --symbols SPY.US --start 2026-10-21 --end 2026-10-21` + `uv run python -m zr.data.theta_vs_recorder --date 2026-10-21`，结果写 DATA_INVENTORY 后停下由负责人决定全量。

## 下一步

1. `feat/thetadata`：读官方文档 → 装 JDK 21 + ThetaTerminal → `zr/data/thetadata.py` → SPY 2026-09 验证 → 停下由负责人决定全量。
2. W2 第二步：参数扫描 + 元标签特征表（事件 × 特征 × 标签）；E8 门控等 OI。
3. 10-22 13:20 UTC timer 启动的录制器应写到 `~/zr_data/`（新 unit 已装）；healthcheck 盘中应无 stale 报警。
