#!/usr/bin/env bash
# 深度视频转换器引擎启动器（供 LaunchAgent 与手动启动复用）
# 自动寻找带完整依赖（flask/cv2/numpy）的 python3，随环境变化可迁移
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# launchd 环境 PATH 精简，需补全 homebrew 路径（ffmpeg 等）
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

find_py() {
    local c
    # 优先当前会话运行时（含 torch 全套），其次系统 python
    for c in \
        "/Users/quanjunqi/Library/Application Support/Doubao/sandbox_runtime/bases/"*/bin/python3 \
        /opt/homebrew/bin/python3.14 /opt/homebrew/bin/python3.13 \
        /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.10 \
        /opt/homebrew/bin/python3 /usr/bin/python3; do
        [ -x "$c" ] || continue
        if "$c" -c "import flask, cv2, numpy" >/dev/null 2>&1; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

PY="$(find_py)"
if [ -z "$PY" ]; then
    echo "[engine] 未找到带依赖的 python3（需要 flask/cv2/numpy）"
    exit 1
fi
echo "[engine] $(date '+%Y-%m-%d %H:%M:%S') 使用 python: $PY"
exec "$PY" "$HERE/server.py"
