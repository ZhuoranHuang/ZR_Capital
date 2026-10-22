#!/usr/bin/env bash
# 启动 Theta Terminal v3（本机）。REST 监听 127.0.0.1:25503（~/thetadata/config.toml 的 host）。
# 引导 JAR 会用 PATH 里的 `java` 再拉起真正的终端，所以必须先把用户目录 JDK 的 bin 放到 PATH 最前并设 JAVA_HOME。
# 首次启动需登录：由负责人在这个终端里输入（或按官方文档用 --api-key / THETADATA_API_KEY / ~/thetadata/.env）。
# 本脚本不读取、不传递任何凭据。
# 来源：https://thetadata.net/docs/Articles/Getting-Started/Getting-Started.html（2026-10-22）
#
# 网络：默认直连（本机经公司 VPN 直连已验证可用，2026-10-21 22 天拉取）。
# 终端连上游 MDDS 用 gRPC，只认 HTTP CONNECT 代理；SOCKS（socksProxyHost）会导致
# "MDDS upstream unavailable"，不要用。确需代理时：
#   THETA_HTTP_PROXY=host:port bash scripts/run_theta_terminal.sh --proxy
# 会设置 -Dhttps.proxyHost/-Dhttps.proxyPort（及 http.*）。--proxy 由本脚本消费，不传给终端。
set -euo pipefail

USE_PROXY=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --proxy) USE_PROXY=1 ;;
        *) ARGS+=("$a") ;;
    esac
done

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
[ -f "$JAR" ] || { echo "找不到 ${JAR}（下载：https://downloads.thetadata.us/ThetaTerminalv3.jar）"; exit 1; }
echo "JAVA_HOME=$JAVA_HOME"
java -version 2>&1 | head -1

JAVA_OPTS="${JAVA_OPTS:-}"
if [ "$USE_PROXY" = "1" ]; then
    : "${THETA_HTTP_PROXY:?--proxy 需要 THETA_HTTP_PROXY=host:port（HTTP CONNECT 代理，不能是 SOCKS）}"
    PHOST="${THETA_HTTP_PROXY%:*}"
    PPORT="${THETA_HTTP_PROXY##*:}"
    PROXY_OPTS="-Dhttps.proxyHost=$PHOST -Dhttps.proxyPort=$PPORT -Dhttp.proxyHost=$PHOST -Dhttp.proxyPort=$PPORT"
    JAVA_OPTS="$JAVA_OPTS $PROXY_OPTS"
    # 引导 JAR 拉起的子 JVM 也要继承代理设置
    export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} $PROXY_OPTS"
    echo "代理：HTTP CONNECT $PHOST:$PPORT"
else
    echo "代理：无（直连）"
fi

cd "$THETA_DIR"
echo "[$(TZ=America/New_York date -Iseconds) ET] start ThetaTerminal (log: $LOG_DIR)"
# shellcheck disable=SC2086
exec java $JAVA_OPTS -jar "$JAR" ${ARGS[@]+"${ARGS[@]}"} 2>&1 \
    | tee -a "$LOG_DIR/terminal_$(TZ=America/New_York date +%F).log"
