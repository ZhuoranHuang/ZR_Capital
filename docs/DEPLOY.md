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
| `deploy/launchd/com.zr.sync-from-vps.plist` | Mac | 每小时调用同步脚本；脚本用 `TZ=Asia/Singapore` 判断 09 点且当天未同步才执行（`$ZR_DATA_ROOT/.last_sync`） |

数据目录 `/home/ubuntu/zr_data/`：`recorder/`（Parquet）、`recorder_qa/`、`universe/`、`journal/`、`alerts/`、`logs/`、`raw/`。

## 首次部署

```bash
ssh zr 'bash -s' < deploy/vps_bootstrap.sh          # 可加分支名参数，默认 feat/deploy
scp .env zr:/home/ubuntu/ZR_Capital/.env            # 唯一允许复制密钥的命令
ssh zr 'chmod 600 ~/ZR_Capital/.env'
ssh zr 'cd ZR_Capital && ZR_DATA_ROOT=~/zr_data ~/.local/bin/uv run python -m zr.ops.api_facts'   # 验证连通，输出 ~/zr_data/reports/API_FACTS.md
scp zr:zr_data/reports/API_FACTS.md docs/API_FACTS.md && uv run python -m zr.ops.redact_api_facts docs/API_FACTS.md   # 脱敏后再 commit
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

## 网络（Mac 侧）

公司 VPN 下本机对 github.com、PyPI、longportapp.com 不稳定（DNS 污染，见 DEPLOY_LOG），统一经 VPS `zr` 中转；claude.ai 流量不经过跳板。

| 组件 | 位置 | 作用 |
|---|---|---|
| `~/Library/LaunchAgents/com.zr.socks.plist` | launchd 常驻 | `ssh -N -D 127.0.0.1:1080 zr`，KeepAlive 断线自动重连；日志 `/tmp/zr-socks.log` |
| `~/.ssh/config` → `Host github.com` | `ProxyJump zr` | git 走 SSH 经 VPS；仓库远程为 `git@github.com:ZhuoranHuang/ZR_Capital.git` |
| `~/.zshrc` → `zrproxy` 函数 | `zrproxy <命令>` | 注入 `ALL_PROXY/HTTPS_PROXY/HTTP_PROXY=socks5h://127.0.0.1:1080`，给 uv / pip / curl / 长桥 SDK 的 HTTP 调用用 |

前提：`~/.ssh/id_ed25519.pub` 必须加到 GitHub 账号（Settings → SSH and GPG keys），否则 `git ls-remote` 报 `Permission denied (publickey)`；加之前可用 `zrproxy git push https://github.com/ZhuoranHuang/ZR_Capital.git <branch>` 走 HTTPS + keychain。

规则：
- 本机需要外网的命令一律加 `zrproxy` 前缀：`zrproxy uv sync`、`zrproxy curl -I https://pypi.org`、`zrproxy uv run python -c "import longport"`。
- 长桥 WebSocket 实时订阅（录制器、api_facts 的订阅测试）一律在 VPS 跑，不在本机经代理跑。
- `deploy/sync_from_vps.sh` 走 ssh，不需要代理；手动立即同步用 `bash deploy/sync_from_vps.sh --now`。
- **时区规则**：本机系统时区被锁定为 US/Pacific，仓库内任何脚本/文档不得依赖系统默认时区；shell 用 `TZ=Asia/Singapore date` 或 `TZ=America/New_York date`，Python 用 `ZoneInfo`，systemd 用 `UTC`。

验证：
```bash
launchctl list | grep com.zr.socks          # PID 非 "-"
git ls-remote --heads origin | head -3
zrproxy curl -sI https://pypi.org | head -1
zrproxy uv run python -c "import longport; print('ok')"
```

恢复：`launchctl unload ~/Library/LaunchAgents/com.zr.socks.plist`；删掉 ssh config 的 `Host github.com` 块；`git remote set-url origin https://github.com/ZhuoranHuang/ZR_Capital.git`。

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
