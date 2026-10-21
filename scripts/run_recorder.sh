#!/usr/bin/env bash
# 启动录制器（本机手动用；VPS 由 systemd zr-recorder 管理）。时间一律显式美东时区，不用系统默认。
set -euo pipefail
cd "$(dirname "$0")/.."

ZR_DATA_ROOT="${ZR_DATA_ROOT:-./data_store}"
mkdir -p "$ZR_DATA_ROOT/logs"
LOG="$ZR_DATA_ROOT/logs/recorder_$(TZ=America/New_York date +%Y-%m-%d).log"

echo "[$(TZ=America/New_York date -Iseconds) ET] start recorder" >> "$LOG"
exec uv run python -m zr.data.recorder "$@" >> "$LOG" 2>&1
