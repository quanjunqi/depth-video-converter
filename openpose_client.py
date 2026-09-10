#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose 远程客户端（通过 HTTP 调用 Windows GPU 服务）
- COCO 18 关键点（含 neck）
- 多人骨架提取
- 纯骨架绘制（黑底，左右侧不同颜色）
- NPZ 导出（keypoints + num_persons）
- 服务地址可配置，不硬编码
"""
from __future__ import annotations

import os
import io
import json
import threading
from typing import Optional

import cv2
import numpy as np
import requests

# ---- 服务配置（可通过环境变量覆盖） ---------------------------------------
DEFAULT_SERVER_URL = os.environ.get(
    "OPENPOSE_SERVER_URL", "http://192.168.1.14:5050"
)
REQUEST_TIMEOUT = float(os.environ.get("OPENPOSE_TIMEOUT", "15"))

# ---- 置信度阈值 -----------------------------------------------------------
PERSON_CONF = 0.3    # 人物检测阈值（服务端已过滤，这里做二次校验）
KEYPOINT_CONF = 0.3  # 关键点阈值：低于此值该点不绘制

# ---- COCO 18 关键点定义 ---------------------------------------------------
# 0:nose 1:neck 2:right_shoulder 3:right_elbow 4:right_wrist
# 5:left_shoulder 6:left_elbow 7:left_wrist 8:right_hip 9:right_knee
# 10:right_ankle 11:left_hip 12:left_knee 13:left_ankle
# 14:right_eye 15:left_eye 16:right_ear 17:left_ear
KEYPOINT_NAMES = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
]

# COCO 18 骨架连线
COCO_SKELETON = [
    (0, 1),       # nose - neck
    (0, 14), (0, 15),      # nose - eyes
    (14, 16), (15, 17),    # eyes - ears
    (1, 2), (1, 5),        # neck - shoulders
    (2, 3), (3, 4),        # right arm
    (5, 6), (6, 7),        # left arm
    (2, 8), (5, 11),       # shoulders - hips
    (8, 9), (9, 10),       # right leg
    (11, 12), (12, 13),    # left leg
    (8, 11),               # hip - hip
]

# ---- 部位配色（BGR，黑底上清晰可辨） -------------------------------
COLOR_WHITE = (255, 255, 255)
COLOR_RIGHT_ARM = (0, 0, 255)    # 右臂：红色（人物右侧=画面左侧）
COLOR_LEFT_ARM = (0, 255, 0)     # 左臂：绿色
COLOR_RIGHT_LEG = (0, 255, 255)  # 右腿：黄色
COLOR_LEFT_LEG = (255, 255, 0)   # 左腿：青色

# 连线 → 颜色映射
_LINK_COLOR = {
    (2, 3): COLOR_RIGHT_ARM, (3, 4): COLOR_RIGHT_ARM,
    (5, 6): COLOR_LEFT_ARM, (6, 7): COLOR_LEFT_ARM,
    (8, 9): COLOR_RIGHT_LEG, (9, 10): COLOR_RIGHT_LEG,
    (11, 12): COLOR_LEFT_LEG, (12, 13): COLOR_LEFT_LEG,
}
# 关键点 → 颜色映射
_POINT_COLOR = {
    3: COLOR_RIGHT_ARM, 4: COLOR_RIGHT_ARM,
    6: COLOR_LEFT_ARM, 7: COLOR_LEFT_ARM,
    9: COLOR_RIGHT_LEG, 10: COLOR_RIGHT_LEG,
    12: COLOR_LEFT_LEG, 13: COLOR_LEFT_LEG,
}

# ---- 服务端健康状态缓存 ---------------------------------------------------
_server_ok: Optional[bool] = None
_server_lock = threading.Lock()


def check_server(server_url: str = DEFAULT_SERVER_URL) -> bool:
    """检查 OpenPose 服务是否可用。"""
    global _server_ok
    with _server_lock:
        try:
            r = requests.get(f"{server_url}/health", timeout=3)
            _server_ok = r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            _server_ok = False
        return _server_ok


def infer_pose(rgb_frame: np.ndarray,
               server_url: str = DEFAULT_SERVER_URL) -> np.ndarray:
    """
    单帧多人姿态推理（通过 HTTP 调用远程 OpenPose 服务）。
    返回: keypoints, shape (N, 18, 3)，N 为人数，3 为 (x, y, confidence)。
    未检测到人或服务不可用时返回 (0, 18, 3) 空数组。
    """
    # 编码为 JPEG 传输（压缩率 90，平衡速度和质量）
    ok, buf = cv2.imencode(".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return np.zeros((0, 18, 3), dtype=np.float32)

    try:
        r = requests.post(
            f"{server_url}/pose",
            files={"image": ("frame.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[openpose] 远程调用失败: {e}")
        return np.zeros((0, 18, 3), dtype=np.float32)

    people = data.get("people", [])
    if not people:
        return np.zeros((0, 18, 3), dtype=np.float32)

    # 转换为 (N, 18, 3) 数组
    result = []
    for person in people:
        kps = person.get("keypoints", [])
        if len(kps) < 18:
            continue
        arr = np.zeros((18, 3), dtype=np.float32)
        for i, kp in enumerate(kps[:18]):
            arr[i, 0] = kp.get("x", 0)
            arr[i, 1] = kp.get("y", 0)
            arr[i, 2] = kp.get("confidence", 0)
        result.append(arr)

    return np.array(result, dtype=np.float32) if result else np.zeros((0, 18, 3), dtype=np.float32)


def draw_skeleton(keypoints: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    绘制彩色骨架帧（黑底，左右侧用不同颜色区分）。
    - 关键点置信度 < KEYPOINT_CONF 该点不绘制，连线任一端缺失则跳过
    - 配色：头部/躯干=白，右臂=红，左臂=绿，右腿=黄，左腿=青
    返回: uint8 BGR 彩色图 (H, W, 3)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if keypoints is None or len(keypoints) == 0:
        return canvas

    for person in keypoints:
        visible = person[:, 2] >= KEYPOINT_CONF

        # 画连线
        for a, b in COCO_SKELETON:
            if visible[a] and visible[b]:
                color = _LINK_COLOR.get((a, b), COLOR_WHITE)
                pt1 = (int(person[a, 0]), int(person[a, 1]))
                pt2 = (int(person[b, 0]), int(person[b, 1]))
                cv2.line(canvas, pt1, pt2, color, 3, cv2.LINE_AA)

        # 火柴人圆头（基于 nose-neck 距离）
        if visible[0] and visible[1]:
            nose = (int(person[0, 0]), int(person[0, 1]))
            neck = (int(person[1, 0]), int(person[1, 1]))
            head_radius = int(np.hypot(nose[0] - neck[0], nose[1] - neck[1]) * 0.8)
            if head_radius > 3:
                cv2.circle(canvas, nose, head_radius, COLOR_WHITE, 3, cv2.LINE_AA)

        # 画关键点
        for i in range(18):
            if visible[i]:
                color = _POINT_COLOR.get(i, COLOR_WHITE)
                cv2.circle(canvas,
                           (int(person[i, 0]), int(person[i, 1])),
                           6, color, -1, cv2.LINE_AA)

    return canvas


def save_pose_npz(keypoints: np.ndarray, path: str) -> None:
    """保存单帧姿态 NPZ（keypoints + num_persons）。"""
    np.savez(path, keypoints=keypoints, num_persons=len(keypoints))


def load_pose_npz(path: str) -> tuple[np.ndarray, int]:
    """读取单帧姿态 NPZ，返回 (keypoints, num_persons)。"""
    data = np.load(path)
    return data["keypoints"], int(data["num_persons"])
