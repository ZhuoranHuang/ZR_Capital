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
