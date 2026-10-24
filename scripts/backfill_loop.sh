#!/usr/bin/env bash
# 数据补齐后台循环（替代 launchd：launchd 启动的进程没有外接盘访问权限，见 STATUS）。
# 从已获得磁盘权限的终端会话启动：nohup bash scripts/backfill_loop.sh >/dev/null 2>&1 &
# 每 30 分钟调用 scripts/backfill_night.sh（全天运行、单实例锁、断点续传）；
# 全部补齐完成（backfill_state/ALL_DONE）后发一次 macOS 通知。本循环单实例（pid 文件）。重启后需重新启动。
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=lib/data_root.sh
source scripts/lib/data_root.sh
zr_resolve_data_root || exit 1
PID="$ZR_DATA_ROOT/logs/backfill_loop.pid"
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then
    echo "循环已在运行（pid $(cat "$PID")）"; exit 0
fi
echo $$ > "$PID"
trap 'rm -f "$PID"' EXIT
while true; do
    bash scripts/backfill_night.sh
    if [ -f "$ZR_DATA_ROOT/backfill_state/ALL_DONE" ] && [ ! -f "$ZR_DATA_ROOT/backfill_state/NOTIFIED" ]; then
        osascript -e 'display notification "数据补齐全部完成，见 reports/DATA_INVENTORY_BACKFILL.md" with title "ZR Capital"' || true
        date > "$ZR_DATA_ROOT/backfill_state/NOTIFIED"
    fi
    sleep 1800
done
