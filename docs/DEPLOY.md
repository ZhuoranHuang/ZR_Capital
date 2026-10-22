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
| `scripts/lib/data_root.sh` | Mac | 本机脚本统一解析数据根（环境变量 → `.env` → `./data_store`）；路径在 `/Volumes/<卷>` 下而卷未挂载时失败退出 |
| `scripts/theta_full_pull.sh` | Mac | ThetaData 全量拉取（SPY→QQQ→MU，按月，续传，进度写 DATA_INVENTORY）；`nohup caffeinate -i bash scripts/theta_full_pull.sh &` |
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

## 本机数据根（Mac）

2026-10-22 起 `ZR_DATA_ROOT="/Volumes/ZHITAI TiPlus9100 2TB Media/zr_data"`（外接 APFS SSD，`Owners: Disabled`，即卷上不强制文件属主——单用户数据目录无影响）。
旧目录 `./data_store` 保留到 2026-10-25（SGT）后删除。外接盘未插时，所有本机脚本（同步、录制、全量拉取）直接报错退出，不会在系统盘上建假目录；Python 侧读写会因路径不存在而失败。
路径含空格：shell 里一律加引号；`.env` 里值带双引号。

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

## ThetaData（Mac 本地）

Options Standard 档：tick 级，2016 起，账户并发 4（本项目用 2）。来源 https://thetadata.net/docs/Articles/Getting-Started/Subscriptions.html（2026-10-22）。

| 组件 | 位置 | 说明 |
|---|---|---|
| Temurin JDK 21 | `~/thetadata/jdk/`（用户目录，无需 sudo） | `~/thetadata/jdk/Contents/Home/bin/java -version` → 21.0.12.1 |
| `ThetaTerminalv3.jar` | `~/thetadata/` | 来自 https://downloads.thetadata.us/ThetaTerminalv3.jar，启动时自动更新 |
| `scripts/run_theta_terminal.sh` | 仓库 | 启动终端（REST `127.0.0.1:25503`），**默认直连、不走代理**；`--proxy` 显式开启 HTTP CONNECT 代理（见下）；日志 `~/thetadata/logs/` |
| `deploy/launchd/com.zr.theta-terminal.plist.example` | 仓库 | 常驻示例，**未安装**；需先用 API key 方式认证（`~/thetadata/.env` 的 `THETADATA_API_KEY`，负责人自己写） |
| `zr/data/thetadata.py` | 仓库 | `bulk_quotes_1m` / `ticks_window`；CLI `uv run python -m zr.data.thetadata --start 2026-09-01 --end 2026-09-30 --symbols SPY.US` |
| `zr/data/theta_vs_recorder.py` | 仓库 | 与录制器同一合约同一分钟 bid/ask 对比 |

**网络 / 代理**（2026-10-22 负责人实测）：
- Theta 终端连上游 MDDS 用 **gRPC**，只认 **HTTP CONNECT** 代理。设 `socksProxyHost`（SOCKS）会导致 `MDDS upstream unavailable`——**不要让终端走 `com.zr.socks` 跳板，也不要用 `zrproxy` 启动终端**。
- 默认直连：本机经公司 VPN 直连已验证可用（10-21 17:02 启动的终端完成 22 天拉取）。
- 确需代理时只能用 HTTP CONNECT 代理：`THETA_HTTP_PROXY=host:port bash scripts/run_theta_terminal.sh --proxy`，脚本设置 `-Dhttps.proxyHost/-Dhttps.proxyPort`（及 `http.*`），并通过 `JAVA_TOOL_OPTIONS` 传给引导 JAR 拉起的子 JVM。现有 SSH `-D` 跳板是 SOCKS，不能直接用；若要经 VPS，需要在 VPS 上另起 HTTP CONNECT 代理（未部署）。
- Python 侧访问终端是 `127.0.0.1:25503` 本地 HTTP，不需要也不应该加代理。

`~/thetadata/config.toml` 的 `host` 已改为 `127.0.0.1`，**但实测不限制监听**（2026-10-22 更正此前「改 host 即限制在本机」的说法——那是未经验证的推断）：
- 终端日志写 `Starting server at: http://127.0.0.1:25503/`，实际 `netstat` 为 `tcp46 *.25503 LISTEN`；
- 从本机局域网地址访问：`http://192.168.8.121:25503/mcp/sse` → **200**，REST → 有响应；
- macOS 应用防火墙为关闭状态（`socketfilterfw --getglobalstate` → disabled）。
即同一网络内的任何人都能访问 REST 和 MCP（使用本账户的数据权限）。config.toml 与 jar 内置模板都没有关闭 MCP 的键（未验证是否有命令行开关）。
需要负责人决定的系统级措施（需 sudo，未执行）：
1. 打开 macOS 防火墙并阻止 java 入站：`sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on` 与 `--blockapp ~/thetadata/jdk/Contents/Home/bin/java`；或
2. 用 pf 只放行回环：在 `/etc/pf.conf` 加 `block in quick on ! lo0 proto tcp to any port 25503` 后 `sudo pfctl -f /etc/pf.conf -e`；或
3. 向 ThetaData 支持确认 v3 终端的监听地址 / 关闭 MCP 的正式方法。

首次使用：负责人在自己的终端里运行 `bash scripts/run_theta_terminal.sh` 并按提示登录（或 `--api-key`）；之后 `curl "http://127.0.0.1:25503/v3/option/list/expirations?symbol=SPY&format=json" | head -c 200` 有返回即可。macOS Gatekeeper 若拦截 jar，按文档在「隐私与安全性」里 Open Anyway。

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
