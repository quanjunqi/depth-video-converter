#!/usr/bin/env bash
# 一键启动 DA3 本地服务（深度视频转换器浏览器版使用）
# 用法: bash start_server.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "[start] 未找到 python3"; exit 1; }
echo "[start] 启动 DA3 本地服务（浏览器中选择本地 DA3 模型时需保持本服务运行）..."
exec "$PY" server.py
