# RUNBOOK：纸面护栏助手（zr-assistant）

只做纸面：助手**不向券商提交任何订单**（`zr/execution/oms.py` 的提交入口一律拒绝）。
「系统离场」是在实时盘口上模拟的纸面单，Telegram 推送后由负责人自行决定是否在账户里操作。

## 组成

| 部件 | 位置 | 说明 |
|---|---|---|
| 录制器旁路 | `zr/live/tap.py`，`zr-recorder.service` 的 `ZR_LIVE_TAP_DIR` | 录制器把 MU / QQQ 正股逐笔、期权盘口与成交另写一份到 `/home/ubuntu/zr_data/live/tap/YYYY-MM-DD.jsonl`（保留 3 天）；不新增订阅 |
| 助手 | `zr/live/runner.py run`，`zr-assistant.service` / `.timer` | 13:15 UTC 启动，等到 09:20 ET；16:05 ET 写日报后退出 |
| 状态层 | `zr/strategy/incremental.py` | 口径 C（`config/assistant.yaml: convention`，可改 A）；开盘时用长桥历史 2 分钟 K 灌入 |
| 配置 | `config/assistant.yaml` | 风控、离场、护栏、纸面权益 |
| 输出 | `/home/ubuntu/zr_data/live/` | `state.json`（每 2 s）、`trades_YYYY-MM-DD.jsonl`、`DAILY_YYYY-MM-DD.md`；日报另写到仓库 `docs/live/` |

## 启动 / 停止

```bash
ssh zr
systemctl list-timers zr-assistant.timer zr-recorder.timer   # 确认下一次触发时间
sudo systemctl start zr-assistant.service                    # 手动启动（非交易日自动退出）
sudo systemctl stop zr-assistant.service                     # 停止（不会平纸面仓；要平仓用 kill switch）
journalctl -u zr-assistant -f                                # 实时日志
```

盘中手动测试（跳过交易日判断与开盘前等待）：
`cd ~/ZR_Capital && ZR_DATA_ROOT=/home/ubuntu/zr_data ~/.local/bin/uv run --no-sync python -m zr.live.runner run --now`

## kill switch

```bash
cd ~/ZR_Capital && ZR_DATA_ROOT=/home/ubuntu/zr_data ~/.local/bin/uv run --no-sync python -m zr.live.runner kill
```

写 `/home/ubuntu/zr_data/live/KILL`。助手在 0.2 s 内发现：全部纸面持仓按当时买一平掉、推送、写日报、退出。
`systemd` 的 `Restart=on-failure` 不会因正常退出而重启。第二天启动时自动删掉旧的 KILL 文件。

## 查看状态

```bash
cd ~/ZR_Capital && ZR_DATA_ROOT=/home/ubuntu/zr_data ~/.local/bin/uv run --no-sync python -m zr.live.runner status
tail -f /home/ubuntu/zr_data/live/trades_$(TZ=America/New_York date +%F).jsonl
```

`state.json`：每个标的最近一根 K 的状态、BOLL、MACD、MA5/10；持仓（未平张数、买一卖一、已实现 / 未实现）；当日盈亏；最近 10 条提醒。

## 入场来源

1. **模拟盘账户轮询**（默认，`intake.account: paper`）：每 3 s 只读 `today_orders`，期权买单成交数量增加 = 负责人入场；卖单 = 负责人自己平仓（记入日报的「若无接管」对比）。
   需要 VPS `.env` 里有 `LONGPORT_PAPER_ACCESS_TOKEN`（2026-10-22 检查：**没有**，见 STATUS 待确认）。没有时助手推送「账户轮询不可用」，只接受第 2 种。
2. **手动登记**：`uv run --no-sync python -m zr.live.runner enter MU261023C900000.US 2 13.25`（合约、张数、成交价；时刻 = 登记时刻）。

## 护栏与离场（config/assistant.yaml）

- 护栏（证据：`docs/research/GUARDRAIL_EVIDENCE.md`）：逆 IGNITION 开仓 → 警告；09:32 前开仓 → 警告；RANGE → 提示；0DTE 10:15 后 → 提示 + 纸面卖出一半；单笔成本 > 权益 5% → 警告；当日亏损 ≥ 10% 或本周 ≥ 20% → 禁止开仓（新入场立即纸面平仓）。
- 系统接管离场：30 分钟上限；期权中间价 +20% 卖一半、−30% 全出（按买一）；正股收盘跌破前一更高低点（看跌镜像）全出（按买一）；MACD 柱顺方向连续 2 根缩短卖一半；0DTE 15:30 前清仓；15:50 全部清仓（按买一）。
- 限价类离场：中间价挂单、5 s 未成交改价一次（中间价 − 1 tick）、再 5 s 按买一（`zr/live/paper.py`，规则同 `zr/backtest/fill_limit.py`）。

## 故障处理

| 现象 | 处理 |
|---|---|
| Telegram 没消息 | `journalctl -u zr-assistant | grep -i telegram`：「telegram disabled」= `.env` 缺 `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`；「发送失败」只记异常类型 |
| `state.json` 里某标的 `live: false` | 旁路文件没有该标的的逐笔：`ls -l /home/ubuntu/zr_data/live/tap/`；`systemctl status zr-recorder`；录制器 09:25 ET 才订阅，09:26 之后才会就绪 |
| 旁路文件不存在 | 录制器没带 `ZR_LIVE_TAP_DIR` 启动（旧单元文件）：`systemctl cat zr-recorder | grep TAP`；**盘中不要重启录制器**，下一个交易日生效 |
| 助手崩溃 | systemd 20 s 后自动重启；重启时从当日 `trades_*.jsonl` 恢复持仓（`[重启]` 推送），账户轮询不会重复入场（`intake_seen_*.json`） |
| 需要回放检查 | 本机：`ZR_DATA_ROOT=... uv run python -m zr.live.runner replay 2026-10-22 --entry "10:24:05,MU261023C895000.US,2"`（录制器数据 + `diag/official_bars/*_2m_all.parquet` 灌入），输出在 `$ZR_DATA_ROOT/live/replay/` |

## 部署

`bash deploy/deploy.sh main`（ruff + pytest 通过才切换）。首次部署后启用定时器：
`sudo systemctl enable --now zr-assistant.timer`。`deploy.sh` 在 09:20–16:05 ET 不重启正在运行的录制器与助手。
