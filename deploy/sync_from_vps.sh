#!/usr/bin/env bash
# Mac 上运行：把 VPS 的 zr_data 同步到本地 $ZR_DATA_ROOT（排除 logs）。
# launchd 每小时调用一次（deploy/launchd/com.zr.sync-from-vps.plist）；本脚本只在新加坡时间 09 点、
# 且当天（新加坡日期）尚未成功同步时执行；其余情况退出 0。不依赖系统时区。
# 数据根：环境变量或 .env 的 ZR_DATA_ROOT（外接盘未挂载时失败退出，见 scripts/lib/data_root.sh）。
# 手动立即同步：bash deploy/sync_from_vps.sh --now
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=../scripts/lib/data_root.sh
source scripts/lib/data_root.sh
zr_resolve_data_root

mkdir -p "$ZR_DATA_ROOT"
STAMP="$ZR_DATA_ROOT/.last_sync"
SG_HOUR="$(TZ=Asia/Singapore date +%H)"
SG_DATE="$(TZ=Asia/Singapore date +%F)"

if [ "${1:-}" != "--now" ]; then
    [ "$SG_HOUR" = "09" ] || exit 0
    [ "$(cat "$STAMP" 2>/dev/null || true)" = "$SG_DATE" ] && exit 0
fi

echo "[sync $(TZ=Asia/Singapore date -Iseconds) SGT] zr:/home/ubuntu/zr_data/ -> $ZR_DATA_ROOT/"
# macOS 自带 rsync 不支持 --info=stats1，用 --stats
rsync -az --partial --stats --exclude 'logs/' zr:/home/ubuntu/zr_data/ "$ZR_DATA_ROOT/" \
    | grep -E 'Number of (regular )?files transferred|Total transferred file size|Total bytes received|total size'
echo "$SG_DATE" > "$STAMP"
echo "[sync] done $SG_DATE"
