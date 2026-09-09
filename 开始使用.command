#!/usr/bin/env bash
# ============================================================
# 深度视频转换器 · 一键使用（双击本文件即可）
# 引擎为系统常驻服务，一般无需操作；此脚本仅作兜底
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

if ! curl -s -m 2 http://127.0.0.1:8765/api/health >/dev/null 2>&1; then
    launchctl kickstart -k gui/$(id -u)/com.depth-video-converter.server 2>/dev/null || true
    for i in $(seq 1 15); do
        sleep 1
        curl -s -m 2 http://127.0.0.1:8765/api/health >/dev/null 2>&1 && break
    done
fi

open http://127.0.0.1:8765
