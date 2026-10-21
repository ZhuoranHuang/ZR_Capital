# 状态

更新：2026-10-22 07:40 SGT · main

## 阶段

W1 收尾完成，W2 开始（ThetaData 接入 `feat/thetadata`；事件与标签 `feat/w2-events`）。标的收敛为 **SPY / QQQ / MU**（D-016）。

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
4. ThetaData：终端首次登录由负责人在本机终端完成（见 `feat/thetadata` 的 STATUS 更新）。

## 下一步

1. `feat/thetadata`：读官方文档 → 装 JDK 21 + ThetaTerminal → `zr/data/thetadata.py` → SPY 2026-09 验证 → 停下由负责人决定全量。
2. `feat/w2-events`：bars/features/events/labels/run → `docs/research/EVENT_LABEL_TABLE.md`。
3. 10-22 13:20 UTC timer 启动的录制器应写到 `~/zr_data/`（新 unit 已装）；healthcheck 盘中应无 stale 报警。
