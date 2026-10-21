#!/usr/bin/env bash
# VPS 首次初始化。在 VPS 上以 ubuntu 运行：ssh zr 'bash -s' < deploy/vps_bootstrap.sh
# 幂等：重复运行不会破坏已有状态。不写入任何密钥；.env 由 scp 单独复制。
set -euo pipefail

REPO_URL="https://github.com/ZhuoranHuang/ZR_Capital.git"
REPO_DIR="/home/ubuntu/ZR_Capital"
DATA_DIR="/home/ubuntu/zr_data"
BRANCH="${1:-feat/deploy}"
UV="/home/ubuntu/.local/bin/uv"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

log "apt update / upgrade"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git rsync fail2ban unattended-upgrades chrony curl ca-certificates
sudo systemctl enable --now fail2ban chrony >/dev/null
sudo dpkg-reconfigure -f noninteractive unattended-upgrades

log "timezone UTC"
sudo timedatectl set-timezone UTC

if ! swapon --show | grep -q /swapfile; then
    log "create 2G swap"
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
    log "swap exists"
fi

if [ ! -x "$UV" ]; then
    log "install uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
"$UV" python install 3.12 >/dev/null
log "uv $("$UV" --version)"

if [ ! -d "$REPO_DIR/.git" ]; then
    log "clone $REPO_URL"
    git clone -q "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch -q origin
git checkout -q -B "$BRANCH" "origin/$BRANCH"
log "repo at $(git rev-parse --short HEAD) ($BRANCH)"
"$UV" sync --extra dev >/dev/null
log "deps synced"

mkdir -p "$DATA_DIR"/{raw,journal,universe,logs,alerts}

log "install systemd units"
sudo cp deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable zr-recorder.timer zr-universe.timer zr-healthcheck.timer >/dev/null
# 只 enable 不 start；start 由负责人确认 .env 后执行

log "done. timers enabled (not started):"
systemctl list-unit-files 'zr-*' --no-pager
