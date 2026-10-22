#!/usr/bin/env bash
# ThetaData 期权 1 分钟历史全量拉取（Mac 本地，需 Theta Terminal 运行在 127.0.0.1:25503）。
# 顺序 SPY → QQQ → MU，按自然月分段；已完成的交易日自动跳过（manifest + 三个 parquet 齐全），
# 中断后重跑即续传。每完成一个月在 docs/DATA_INVENTORY.md 的进度表追加一行。
#
# 后台运行：nohup bash scripts/theta_full_pull.sh >/dev/null 2>&1 &
# 日志：$ZR_DATA_ROOT/logs/theta_pull.log
# 参数（环境变量）：THETA_START（默认 2023-10-02）、THETA_END（默认今天美东日期）、THETA_SYMBOLS、
# THETA_CONCURRENCY（默认 4 = 账户并发上限；429 指数退避、503/断连最多等 10 分钟）
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
# shellcheck source=lib/data_root.sh
source scripts/lib/data_root.sh
zr_resolve_data_root || exit 1

UV="${UV:-$HOME/.local/bin/uv}"
START="${THETA_START:-2023-10-02}"
END="${THETA_END:-$(TZ=America/New_York date +%F)}"
SYMBOLS="${THETA_SYMBOLS:-SPY.US QQQ.US MU.US}"
CONC="${THETA_CONCURRENCY:-4}"
LOG_DIR="$ZR_DATA_ROOT/logs"
LOG="$LOG_DIR/theta_pull.log"
LOCK="$LOG_DIR/theta_pull.lock"
mkdir -p "$LOG_DIR"
exec >>"$LOG" 2>&1

log() { echo "[theta_full $(TZ=America/New_York date -Iseconds) ET] $*"; }

# 单实例
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    log "已有实例在运行（pid $(cat "$LOCK")），退出"
    exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if ! curl -s -o /dev/null --max-time 10 "http://127.0.0.1:25503/v3/option/list/expirations?symbol=SPY&format=json"; then
    log "Theta Terminal 未响应（127.0.0.1:25503），请先启动：bash scripts/run_theta_terminal.sh"
    exit 1
fi

log "开始 symbols=[$SYMBOLS] $START ~ $END 并发 $CONC 数据根 $ZR_DATA_ROOT"
for sym in $SYMBOLS; do
    # 月份列表（美东自然月）
    months="$(cd "$REPO" && "$UV" run --no-sync python - "$START" "$END" <<'PY'
import sys, datetime as dt
s, e = (dt.date.fromisoformat(x) for x in sys.argv[1:3])
d = s.replace(day=1)
while d <= e:
    nxt = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    print(max(d, s).isoformat(), min(nxt - dt.timedelta(days=1), e).isoformat())
    d = nxt
PY
)"
    while read -r m_start m_end; do
        [ -n "$m_start" ] || continue
        t0=$(date +%s)
        log "$sym $m_start ~ $m_end"
        (cd "$REPO" && "$UV" run --no-sync python -m zr.data.thetadata \
            --symbols "$sym" --start "$m_start" --end "$m_end" --concurrency "$CONC")
        rc=$?
        secs=$(( $(date +%s) - t0 ))
        (cd "$REPO" && "$UV" run --no-sync python -m zr.data.theta_progress \
            --underlying "${sym%%.US}" --start "$m_start" --end "$m_end" --seconds "$secs" --rc "$rc")
        if [ "$rc" -ne 0 ]; then
            log "${sym} ${m_start} 失败 rc=${rc}，停止（重跑本脚本续传）"
            exit "$rc"
        fi
    done <<< "$months"
done
log "全部完成"
