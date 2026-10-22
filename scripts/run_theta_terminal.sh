#!/usr/bin/env bash
# 启动 Theta Terminal v3（本机）。REST 监听 127.0.0.1:25503。
# 引导 JAR 会用 PATH 里的 `java` 再拉起真正的终端，所以必须先把用户目录 JDK 的 bin 放到 PATH 最前并设 JAVA_HOME。
# 首次启动需登录：由负责人在这个终端里输入（或按官方文档用 --api-key / THETADATA_API_KEY / ~/thetadata/.env）。
# 本脚本不读取、不传递任何凭据。
# 来源：https://thetadata.net/docs/Articles/Getting-Started/Getting-Started.html（2026-10-22）
set -euo pipefail

THETA_DIR="${THETA_DIR:-$HOME/thetadata}"
JAR="$THETA_DIR/ThetaTerminalv3.jar"
LOG_DIR="$THETA_DIR/logs"
mkdir -p "$LOG_DIR"

# 定位用户目录 JDK（Temurin 解压后 macOS 为 jdk/Contents/Home/bin，Linux 为 jdk/bin）
JDK_BIN="$(find "$THETA_DIR/jdk" -type d -name bin -path '*Home/bin' 2>/dev/null | head -1)"
[ -n "$JDK_BIN" ] || JDK_BIN="$(find "$THETA_DIR/jdk" -maxdepth 2 -type d -name bin 2>/dev/null | head -1)"
[ -n "$JDK_BIN" ] && [ -x "$JDK_BIN/java" ] || { echo "找不到 JDK：$THETA_DIR/jdk 下无 bin/java"; exit 1; }
export JAVA_HOME="${JDK_BIN%/bin}"
export PATH="$JDK_BIN:$PATH"
[ -f "$JAR" ] || { echo "找不到 $JAR（下载：https://downloads.thetadata.us/ThetaTerminalv3.jar）"; exit 1; }
echo "JAVA_HOME=$JAVA_HOME"
java -version 2>&1 | head -1

cd "$THETA_DIR"
# 终端启动时会自动更新；网络经 SOCKS 跳板时用 -DsocksProxyHost（公司 VPN 下需要）
JAVA_OPTS="${JAVA_OPTS:-}"
if [ "${THETA_VIA_SOCKS:-1}" = "1" ]; then
    JAVA_OPTS="$JAVA_OPTS -DsocksProxyHost=127.0.0.1 -DsocksProxyPort=1080"
    export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -DsocksProxyHost=127.0.0.1 -DsocksProxyPort=1080"
fi
echo "[$(TZ=America/New_York date -Iseconds) ET] start ThetaTerminal (log: $LOG_DIR)"
# shellcheck disable=SC2086
exec java $JAVA_OPTS -jar "$JAR" "$@" 2>&1 | tee -a "$LOG_DIR/terminal_$(TZ=America/New_York date +%F).log"
