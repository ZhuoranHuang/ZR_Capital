# 部署

VPS：AWS Lightsail 悉尼（ap-southeast-2），Ubuntu 24.04，2 GB，SSH 别名 `zr`（用户 ubuntu）。
分工：VPS 跑录制器、健康检查、API 核验、交易史拉取、universe 快照；Mac 只做研究、回测、数据拉取。
密钥只在 VPS 的 `~/ZR_Capital/.env`（600），不入库。

## 文件

| 文件 | 在哪跑 | 用途 |
|---|---|---|
| `deploy/vps_bootstrap.sh` | VPS | 首次初始化：apt、swap、uv、clone、systemd 单元（只 enable） |
| `deploy/deploy.sh <branch\|tag>` | VPS | 更新代码；ruff + pytest 通过才重启；失败回滚 |
| `deploy/systemd/zr-recorder.{service,timer}` | VPS | 每个工作日 13:20 UTC 启动录制器；程序自判交易日、等到 9:25 ET |
| `deploy/systemd/zr-universe.{service,timer}` | VPS | 每周五 21:30 UTC 写 universe 快照到 `~/zr_data/universe/` |
| `deploy/systemd/zr-healthcheck.{service,timer}` | VPS | 每 5 分钟；盘中无进程或 5 分钟无新文件 → `~/zr_data/alerts/` + journald |
| `deploy/sync_from_vps.sh` | Mac | rsync `~/zr_data/`（排除 logs）到 `$ZR_DATA_ROOT` |
| `deploy/launchd/com.zr.sync-from-vps.plist` | Mac | 每天新加坡 09:00 跑上面的同步 |

数据目录 `/home/ubuntu/zr_data/`：`recorder/`（Parquet）、`recorder_qa/`、`universe/`、`journal/`、`alerts/`、`logs/`、`raw/`。

## 首次部署

```bash
ssh zr 'bash -s' < deploy/vps_bootstrap.sh          # 可加分支名参数，默认 feat/deploy
scp .env zr:/home/ubuntu/ZR_Capital/.env            # 唯一允许复制密钥的命令
ssh zr 'chmod 600 ~/ZR_Capital/.env'
ssh zr 'cd ZR_Capital && ~/.local/bin/uv run python -m zr.ops.api_facts'   # 验证连通
ssh zr 'sudo systemctl start zr-recorder.timer zr-universe.timer zr-healthcheck.timer'
ssh zr 'systemctl list-timers --all | grep zr-'
```

## 日常更新

```bash
ssh zr 'bash ZR_Capital/deploy/deploy.sh main'      # 或任意分支 / tag
```

## 回滚

`deploy.sh` 在 ruff / pytest 失败时自动回到原 commit。手动回滚：

```bash
ssh zr 'bash ZR_Capital/deploy/deploy.sh <上一个 tag 或 commit>'
```

## 看日志

```bash
ssh zr 'journalctl -u zr-recorder -f'
ssh zr 'journalctl -u zr-healthcheck --since today'
ssh zr 'cat ~/zr_data/alerts/$(date -u +%F).jsonl'
ssh zr 'ls ~/zr_data/recorder_qa/'
```

## 手动跑一次

```bash
ssh zr 'cd ZR_Capital && ZR_DATA_ROOT=~/zr_data ~/.local/bin/uv run python -m zr.data.recorder --now --until 16:05'
ssh zr 'cd ZR_Capital && ZR_DATA_ROOT=~/zr_data ~/.local/bin/uv run python -m zr.ops.healthcheck --force'
ssh zr 'cd ZR_Capital && ZR_DATA_ROOT=~/zr_data ~/.local/bin/uv run python -m zr.data.universe'
```

## 迁移到新加坡（ap-southeast-1）

1. Lightsail 控制台 → 实例 → 快照 → 创建快照。
2. 快照 → 复制到其他区域 → 选 ap-southeast-1。
3. 在新加坡从快照创建实例（同规格或更大）。
4. 新实例附加静态 IP；更新本机 `~/.ssh/config` 中 `zr` 的 HostName。
5. `ssh zr 'bash ZR_Capital/deploy/deploy.sh main'`，确认 `.env` 仍在且 600。
6. 旧实例停掉 timer（`sudo systemctl disable --now zr-*.timer`）再删除，避免双录。

## 注意

- `zr-recorder.timer` 13:20 UTC 在夏令时是 09:20 ET、冬令时 08:20 ET，程序内部用 `America/New_York` 等到 09:25 再订阅，所以 DST 切换不需要改 timer。
- healthcheck 不查交易日历，美股假日会误报一次。
- systemd 的 `EnvironmentFile`（.env）会覆盖 `Environment=`；服务里 `ZR_DATA_ROOT` 必须放在 `ExecStart=/usr/bin/env ZR_DATA_ROOT=... ` 上，不要移回 `Environment=`。
- 2 GB 内存：录制器缓冲 5000 条 / 30 秒 flush，实测内存见 `docs/DEPLOY_LOG.md`。
