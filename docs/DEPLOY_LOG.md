# 部署日志

只追加。不含密钥、IP 以外的账户信息。

## 2026-10-21 首次部署（悉尼 ap-southeast-2）

实例：Ubuntu 24.04.4 LTS，2 vCPU，1906 MB 内存，58 GB 磁盘，x86_64。SSH 别名 `zr`。

`ssh zr 'bash -s' < deploy/vps_bootstrap.sh`，18:35–18:37 UTC，exit 0：

| 时间 UTC | 步骤 | 结果 |
|---|---|---|
| 18:35:19 | apt update / upgrade + git rsync fail2ban unattended-upgrades chrony | 60+ 包升级，无错误 |
| 18:37:04 | timezone UTC | 已是 UTC |
| 18:37:04 | 2 GB swap `/swapfile` | 创建并写入 fstab |
| 18:37:04 | uv 官方脚本 + Python 3.12 | uv 0.13.4 x86_64 |
| 18:37:06 | clone 仓库 → `feat/deploy` | a4d1900 |
| 18:37:12 | `uv sync --extra dev` | 完成 |
| 18:37:12 | systemd 单元安装 + enable | zr-recorder / zr-universe / zr-healthcheck 三个 timer enabled，未 start |

之后：`scp .env zr:~/ZR_Capital/.env`，`chmod 600`。

网络：从 VPS `openapi.longportapp.com` 正常解析并返回 401（可达，0.6 s）；`github.com` 0.05 s。
对比本机（Mac，新加坡）同一时段 longportapp.com DNS 被污染、github.com 超时——所以 push 走了 VPS 的 SOCKS 跳板（`ssh -D 1080 zr` + `git -c http.proxy=socks5h://127.0.0.1:1080`），凭据仍在本机 keychain。

## 2026-10-21 首次实跑

- 18:39 UTC `deploy.sh feat/deploy` → 1dd032d（修复 to_py 递归），18:43 → ba26c01（ATM 过滤过期日）；两次 ruff + pytest 在 VPS 通过。
- `api_facts` 跑 3 次：第 1 次 `RecursionError`（pyo3 枚举序列化），第 2 次全过但 ATM 选中已过期合约，第 3 次 `--skip-bars` 全过 → `docs/API_FACTS.md`（脱敏后入库）。
- `trade_history --since 2026-05-01 --options-only`：orders 331 / executions 288 / charges 1564 行 → `~/zr_data/journal/`。统计命令：pandas 读三个 parquet，按 `symbol` 正则取 underlying 分组。
- 18:45 UTC 启动三个 timer；`universe` 手动跑一次 → `~/zr_data/universe/universe_2026-10-21.json`（308 订阅）；`healthcheck` 手动跑一次正确报警（当时录制器未启动）。
- 18:46 UTC `systemctl start zr-recorder.service`，跑到 16:05 ET 自动退出。
- 内存：bootstrap 后 used 466 MB / 1906，swap 2 GB 基本未用。
- 18:47–18:55 UTC 录制器启动后 `~/zr_data/recorder` 无文件、healthcheck 报警。诊断：标准脚本订阅 308×3 在 10 秒收到 5545 条，说明推送正常；真实原因是 systemd `EnvironmentFile`（.env 的 `ZR_DATA_ROOT=./data_store`）覆盖了 `Environment=ZR_DATA_ROOT`，数据写到了 `~/ZR_Capital/data_store/recorder/`（8 分钟 300 个文件 5.6 MB）。修复：unit 改为 `ExecStart=/usr/bin/env ZR_DATA_ROOT=... uv run ...`。**更正（19:50 UTC）**：18:52 的 `deploy.sh` 实际因 VPS 工作区 `docs/API_FACTS.md` 被脚本改过而在 `git checkout` 处静默退出（输出被 grep 吞掉），新 unit 当时没装上、服务没重启；录制器自 18:46:51 起一直是同一个进程（NRestarts=0），全程写 `data_store/`。19:48 手动安装新 unit + daemon-reload，并用 `systemd-run` 同样 EnvironmentFile 验证 `ZR_DATA_ROOT=/home/ubuntu/zr_data` 生效；本场录完后把 `data_store/recorder*` rsync 到 `~/zr_data/`。deploy.sh 已改为 checkout 前自动 stash 并显式报错；api_facts 默认输出改到 `ZR_DATA_ROOT/reports/`。
- 诊断中发现：一分钟内重复订阅大量期权触发 `OpenApiException code=301607 Too many option securities request within one minute`；录制器每 5 分钟只订阅差集，正常不会触发。
- 录制器进程 RSS 约 183 MB。

## 2026-10-21 本机网络改为经 VPS 跳板（负责人确认后写入）

- 备份：`~/.ssh/config.bak-20261021`、`~/.zshrc.bak-20261021`。
- 写入：`~/Library/LaunchAgents/com.zr.socks.plist`（常驻 `ssh -N -D 127.0.0.1:1080 zr`，KeepAlive）；`~/.ssh/config` 追加 `Host github.com` + `ProxyJump zr`；`~/.zshrc` 追加 `zrproxy`；`~/Library/LaunchAgents/com.zr.sync-from-vps.plist`（Hour=9，本机时区 US/Pacific，待负责人决定）；`git remote set-url origin git@github.com:ZhuoranHuang/ZR_Capital.git`。
- 验证：SOCKS 监听 127.0.0.1:1080 ✓；`zrproxy curl -I https://pypi.org` → HTTP/2 200 ✓；`zrproxy uv run python -c "import longport"` ✓；GitHub HTTPS 经代理 200 ✓；`git ls-remote`（SSH 经 ProxyJump）到达 github.com 但 `Permission denied (publickey)` —— 公钥 `zr-lightsail` 尚未加到 GitHub 账号。`launchctl start com.zr.sync-from-vps` 手动触发一次正常。
