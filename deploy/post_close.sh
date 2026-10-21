#!/usr/bin/env bash
# 收盘后一次性任务（VPS）：等 QA 报告 → 把会话数据并入 ~/zr_data → deploy <branch> → 报告回填进度与清单。
# 调度示例（16:10 ET = 20:10 UTC 夏令时）：
#   sudo systemd-run --unit=zr-post-close-$(date -u +%Y%m%d) --on-calendar="2026-10-21 20:10:00 UTC" \
#        --uid=ubuntu --gid=ubuntu -E HOME=/home/ubuntu -p WorkingDirectory=/home/ubuntu/ZR_Capital \
#        /bin/bash /home/ubuntu/ZR_Capital/deploy/post_close.sh feat/w1-analysis
# 日志：~/zr_data/logs/post_close_YYYY-MM-DD.log
set -uo pipefail

BRANCH="${1:-main}"
REPO=/home/ubuntu/ZR_Capital
DATA=/home/ubuntu/zr_data
UV=/home/ubuntu/.local/bin/uv
DAY="$(TZ=America/New_York date +%F)"
LOG="$DATA/logs/post_close_$DAY.log"
mkdir -p "$DATA/logs"
exec >>"$LOG" 2>&1

log() { echo "[post_close $(date -u +%FT%TZ)] $*"; }
log "开始 branch=$BRANCH day=$DAY"

# 1. 等 QA 报告（录制器 16:05 ET 退出后写），最多等 30 分钟；两个可能的输出目录都看
QA=""
for _ in $(seq 1 120); do
    for cand in "$DATA/recorder_qa/$DAY.md" "$REPO/data_store/recorder_qa/$DAY.md"; do
        [ -f "$cand" ] && QA="$cand" && break 2
    done
    sleep 15
done
if [ -n "$QA" ]; then
    log "QA 报告：$QA"; cat "$QA"
else
    log "警告：30 分钟内没有 QA 报告；录制器状态：$(systemctl is-active zr-recorder.service)"
    journalctl -u zr-recorder --no-pager -o cat --since "6 hours ago" | grep -vE '^\||^\+' | tail -20
fi

# 2. 会话数据若写到了仓库下的 data_store（旧 unit），并入 ~/zr_data（原文件保留）
if [ -d "$REPO/data_store/recorder" ]; then
    rsync -a "$REPO/data_store/recorder/" "$DATA/recorder/"
    [ -d "$REPO/data_store/recorder_qa" ] && rsync -a "$REPO/data_store/recorder_qa/" "$DATA/recorder_qa/"
    log "已合并 data_store → zr_data"
fi
log "recorder parquet 文件数：$(find "$DATA/recorder" -name '*.parquet' | wc -l)，大小：$(du -sh "$DATA/recorder" | cut -f1)"
log "当日 alerts（美东日期）："; cat "$DATA/alerts/$DAY.jsonl" 2>/dev/null | tail -5 || true

# 3. deploy（录制器已退出，不会打断会话）
if systemctl is-active --quiet zr-recorder.service; then
    log "录制器仍在运行，跳过 deploy"
else
    cd "$REPO" && bash deploy/deploy.sh "$BRANCH"
    log "deploy 退出码 $?"
fi

# 4. 回填进度与清单
if pgrep -f zr.data.backfill_bars >/dev/null; then
    log "回填仍在运行："; grep -vE '^\||^\+' "$DATA"/logs/backfill_*.log | grep -E '完成|已到' | tail -4
else
    log "回填已结束："; grep -vE '^\||^\+' "$DATA"/logs/backfill_*.log | grep '完成' | tail -4
fi
cd "$REPO" && ZR_DATA_ROOT="$DATA" "$UV" run python -m zr.data.backfill_bars --inventory 2>/dev/null \
    | grep -vE '^\+|^\| (US|HK|CN|SG) |Quote Package|^\| Market' | tee "$DATA/reports/DATA_INVENTORY.md" 2>/dev/null \
    || mkdir -p "$DATA/reports"
log "定时器："; systemctl list-timers --all --no-pager | grep -E 'zr-' | awk '{print $1,$2,$3,$4,$NF}'
log "结束"
