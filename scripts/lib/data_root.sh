# shellcheck shell=bash
# 解析本机数据根目录并做安全检查。用法：source scripts/lib/data_root.sh; zr_resolve_data_root
# 优先级：环境变量 ZR_DATA_ROOT → 应用从 .env 解析出的值（zr.config，不打印其他配置）→ ./data_store。
# 外接盘保护：路径在 /Volumes/<卷名>/ 下而该卷未挂载时直接失败，
# 避免 mkdir -p 在系统盘上建出一个假的 /Volumes/<卷名> 目录。

zr_resolve_data_root() {
    local repo root vol
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    if [ -n "${ZR_DATA_ROOT:-}" ]; then
        root="$ZR_DATA_ROOT"
    elif command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
        root="$(cd "$repo" && "${UV:-$HOME/.local/bin/uv}" run --no-sync python -c \
            'from zr.config import get_settings; print(get_settings().zr_data_root)' 2>/dev/null)"
    fi
    root="${root:-$repo/data_store}"
    case "$root" in
        /*) ;;
        *) root="$repo/${root#./}" ;;
    esac
    if [[ "$root" == /Volumes/* ]]; then
        vol="/Volumes/$(echo "$root" | cut -d/ -f3)"
        if ! /sbin/mount | grep -qF " on $vol ("; then
            echo "数据盘未挂载：${vol}（ZR_DATA_ROOT=${root}）。插上外接盘后重试。" >&2
            return 1
        fi
    fi
    export ZR_DATA_ROOT="$root"
}
