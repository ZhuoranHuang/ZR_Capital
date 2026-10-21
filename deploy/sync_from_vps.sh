#!/usr/bin/env bash
# Mac 上运行：把 VPS 的 zr_data 同步到本地 $ZR_DATA_ROOT（排除 logs）。
# launchd 每日定时见 deploy/launchd/com.zr.sync-from-vps.plist
set -euo pipefail
cd "$(dirname "$0")/.."

ZR_DATA_ROOT="${ZR_DATA_ROOT:-./data_store}"
mkdir -p "$ZR_DATA_ROOT"
echo "[sync $(date -Iseconds)] zr:/home/ubuntu/zr_data/ -> $ZR_DATA_ROOT/"
rsync -az --partial --info=stats1 --exclude 'logs/' zr:/home/ubuntu/zr_data/ "$ZR_DATA_ROOT/"
