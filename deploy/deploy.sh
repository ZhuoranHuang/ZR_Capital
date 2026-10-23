#!/usr/bin/env bash
# VPS 上更新代码并重启服务。用法：bash deploy/deploy.sh <branch|tag>
# ruff + pytest 全部通过才重启；任一步失败回到原 commit。
set -euo pipefail

REF="${1:?用法: deploy.sh <branch|tag>}"
REPO_DIR="/home/ubuntu/ZR_Capital"
UV="/home/ubuntu/.local/bin/uv"
cd "$REPO_DIR"

PREV="$(git rev-parse HEAD)"
log() { echo "[deploy $(date -u +%H:%M:%S)] $*"; }

rollback() {
    log "失败：$1 —— 回滚到 $PREV"
    git checkout -q --detach "$PREV"
    "$UV" sync --extra dev >/dev/null 2>&1 || true
    exit 1
}

git fetch -q --tags --prune origin || rollback "git fetch"
# 工作区若有改动（脚本生成的文档等），先 stash，避免 checkout 静默失败
if [ -n "$(git status --porcelain)" ]; then
    git stash push -q -u -m "deploy $(date -u +%FT%TZ)"
    log "工作区有改动，已 stash（git stash list 可查）"
fi
if git show-ref -q --verify "refs/remotes/origin/$REF"; then
    git checkout -q -B "$REF" "origin/$REF" || rollback "git checkout $REF"
else
    git checkout -q "$REF" || rollback "git checkout $REF"   # tag 或 commit
fi
log "checkout $REF -> $(git rev-parse --short HEAD)"

"$UV" sync --extra dev >/dev/null || rollback "uv sync"
"$UV" run ruff check . || rollback "ruff"
"$UV" run pytest -q || rollback "pytest"

sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
for t in zr-recorder.timer zr-universe.timer zr-healthcheck.timer zr-assistant.timer zr-shadow-daily.timer; do
    if systemctl is-enabled --quiet "$t"; then sudo systemctl restart "$t"; fi
done
# 盘中（09:20–16:05 ET）不重启录制器 / 助手：重启会丢推送、打断纸面持仓管理；新代码等下一次 timer
HM="$(TZ=America/New_York date +%H%M)"
for s in zr-recorder.service zr-assistant.service; do
    if systemctl is-active --quiet "$s"; then
        if [ "$HM" -ge 0920 ] && [ "$HM" -lt 1605 ]; then
            log "$s 正在运行且在盘中，不重启（下一次 timer 生效）"
        else
            sudo systemctl restart "$s"
            log "$s restarted"
        fi
    fi
done
log "ok $(git rev-parse --short HEAD)"
