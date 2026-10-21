#!/usr/bin/env bash
# 启动录制器：美东时区，日志写到 $ZR_DATA_ROOT/logs/
set -euo pipefail
cd "$(dirname "$0")/.."

export TZ=America/New_York
ZR_DATA_ROOT="${ZR_DATA_ROOT:-./data_store}"
mkdir -p "$ZR_DATA_ROOT/logs"
LOG="$ZR_DATA_ROOT/logs/recorder_$(date +%Y-%m-%d).log"

echo "[$(date -Iseconds)] start recorder" >> "$LOG"
exec uv run python -m zr.data.recorder "$@" >> "$LOG" 2>&1
