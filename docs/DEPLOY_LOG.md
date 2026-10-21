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
