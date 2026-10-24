#!/usr/bin/env bash
# 数据补齐夜间调度（zr/data/backfill_all.py）。launchd 每 30 分钟调用（deploy/launchd/com.zr.backfill-night.plist）；
# 只在新加坡时间 22:00–次日 08:00 运行，白天直接退出；单实例（pid 锁）；Theta Terminal 未运行时自动启动。
# 每日增量（新加坡时间 06:00 后）由 backfill_all 在同一进程里先做。手动：bash scripts/backfill_night.sh --now
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=lib/data_root.sh
source scripts/lib/data_root.sh
zr_resolve_data_root || exit 1

UV="${UV:-$HOME/.local/bin/uv}"
H="$(TZ=Asia/Singapore date +%H)"
if [ "${1:-}" != "--now" ] && [ "$((10#$H))" -ge 8 ] && [ "$((10#$H))" -lt 22 ]; then
    exit 0
fi
LOG_DIR="$ZR_DATA_ROOT/logs"
LOCK="$LOG_DIR/backfill.lock"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/backfill_all.log" 2>&1
log() { echo "[backfill $(TZ=Asia/Singapore date -Iseconds) SGT] $*"; }

if ! ( set -o noclobber; echo $$ > "$LOCK" ) 2>/dev/null; then
    if kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
        exit 0
    fi
    log "清理残留锁（pid $(cat "$LOCK" 2>/dev/null) 已不在）"
    echo $$ > "$LOCK"
fi
trap 'rm -f "$LOCK"' EXIT

up() { curl -s -o /dev/null --max-time 60 "http://127.0.0.1:25503/v3/option/list/expirations?symbol=SPY&format=json"; }
if ! up; then
    log "Theta Terminal 未响应，启动"
    nohup bash scripts/run_theta_terminal.sh >/dev/null 2>&1 &
    for _ in $(seq 1 18); do sleep 10; up && break; done
    up || { log "Theta Terminal 仍未响应，退出（下次再试）"; exit 1; }
fi
log "开始"
"$UV" run --no-sync python -m zr.data.backfill_all --until 08:00
log "结束 rc=$?"
