#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8-Pose 人体姿态估计模块
- 多人骨架提取（COCO 17 关键点 + 颈部插值）
- 纯骨架绘制（黑底白骨架）
- 模型常驻缓存（跨任务复用）
- NPZ 导出（keypoints + num_persons）
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

# ---- 路径 ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
POSE_WEIGHTS = os.path.join(HERE, "weights", "pose", "yolov8s-pose.pt")

# ---- 置信度阈值（两级） -------------------------------------------------
PERSON_CONF = 0.3    # 人物检测阈值：低于此值整人不绘制
KEYPOINT_CONF = 0.5  # 关键点阈值：低于此值该点不绘制，但保留该人其他点

# ---- COCO 17 关键点定义 -------------------------------------------------
# 0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear
# 5:l_shoulder 6:r_shoulder 7:l_elbow 8:r_elbow 9:l_wrist 10:r_wrist
# 11:l_hip 12:r_hip 13:l_knee 14:r_knee 15:l_ankle 16:r_ankle
KEYPOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# COCO 原生骨架连线（17 组）
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (1, 2),    # 面部（含左右眼连线）
    (5, 6),                                      # 左右肩
    (5, 7), (7, 9),                              # 左臂
    (6, 8), (8, 10),                             # 右臂
    (5, 11), (6, 12), (11, 12),                 # 躯干
    (11, 13), (13, 15),                          # 左腿
    (12, 14), (14, 16),                          # 右腿
]

# 插值补充连线：颈部(虚拟点) - 鼻子，让头部与躯干连接
# 颈部 = (左肩 + 右肩) / 2，仅当左右肩都可见时计算
NECK_NOSE_LINK = True  # 绘制颈-鼻连线

# 颈部虚拟点索引（用于绘制，不存入 NPZ）
NECK_IDX = 17

# ---- 部位配色（BGR，黑底上清晰可辨） -------------------------------
COLOR_WHITE = (255, 255, 255)   # 头部/躯干/颈部
COLOR_LEFT_ARM = (0, 255, 0)     # 左臂：绿色
COLOR_RIGHT_ARM = (0, 0, 255)    # 右臂：红色
COLOR_LEFT_LEG = (255, 255, 0)   # 左腿：青色
COLOR_RIGHT_LEG = (0, 255, 255)  # 右腿：黄色

# 连线 → 颜色映射
_LINK_COLOR = {
    (5, 7): COLOR_LEFT_ARM, (7, 9): COLOR_LEFT_ARM,    # 左臂
    (6, 8): COLOR_RIGHT_ARM, (8, 10): COLOR_RIGHT_ARM,  # 右臂
    (11, 13): COLOR_LEFT_LEG, (13, 15): COLOR_LEFT_LEG,  # 左腿
    (12, 14): COLOR_RIGHT_LEG, (14, 16): COLOR_RIGHT_LEG, # 右腿
}
# 关键点 → 颜色映射（肩/髋属躯干用白色，肘/腕/膝/踝跟随肢体）
_POINT_COLOR = {
    7: COLOR_LEFT_ARM, 9: COLOR_LEFT_ARM,      # 左肘/左腕
    8: COLOR_RIGHT_ARM, 10: COLOR_RIGHT_ARM,    # 右肘/右腕
    13: COLOR_LEFT_LEG, 15: COLOR_LEFT_LEG,      # 左膝/左踝
    14: COLOR_RIGHT_LEG, 16: COLOR_RIGHT_LEG,    # 右膝/右踝
}


def _pick_device() -> str:
    """选择推理设备：优先 MPS，其次 CPU。"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---- 模型常驻缓存 -------------------------------------------------------
_POSE_CACHE: dict | None = None
_POSE_LOCK = threading.Lock()


def get_pose_model():
    """获取 YOLOv8-Pose 模型（进程内常驻，首次加载后复用）。"""
    global _POSE_CACHE
    with _POSE_LOCK:
        if _POSE_CACHE is not None:
            return _POSE_CACHE
        from ultralytics import YOLO
        device = _pick_device()
        model = YOLO(POSE_WEIGHTS)
        model.to(device)
        _POSE_CACHE = (model, device)
        print(f"[pose] YOLOv8s-pose 模型已加载（{device}）")
        return _POSE_CACHE


def infer_pose(rgb_frame: np.ndarray) -> np.ndarray:
    """
    单帧多人姿态推理。
    返回: keypoints, shape (N, 17, 3)，N 为人数，3 为 (x, y, confidence)。
    未检测到人时返回 (0, 17, 3) 空数组。
    """
    model, _ = get_pose_model()
    results = model(rgb_frame, conf=PERSON_CONF, verbose=False)
    if not results or results[0].keypoints is None:
        return np.zeros((0, 17, 3), dtype=np.float32)
    kpts = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3): x,y,conf
    return kpts.astype(np.float32)


def draw_skeleton(keypoints: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    绘制彩色骨架帧（黑底，左右侧用不同颜色区分）。
    - 人物级过滤由 YOLO 推理时的 conf=PERSON_CONF 保证
    - 关键点置信度 < KEYPOINT_CONF 该点不绘制，连线任一端缺失则跳过
    - 颈部 = (左肩 + 右肩) / 2（插值补充）
    - 配色：头部/躯干=白，左臂=绿，右臂=红，左腿=青，右腿=黄
    返回: uint8 BGR 彩色图 (H, W, 3)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if keypoints is None or len(keypoints) == 0:
        return canvas  # 纯黑帧（无人时确保帧数对齐）

    for person in keypoints:
        visible = person[:, 2] >= KEYPOINT_CONF  # (17,) bool

        # 计算颈部（插值）：左右肩都可见时
        neck = None
        if visible[5] and visible[6]:
            neck = ((person[5, 0] + person[6, 0]) / 2,
                    (person[5, 1] + person[6, 1]) / 2)

        # 画连线（按部位着色）
        for a, b in COCO_SKELETON:
            if visible[a] and visible[b]:
                color = _LINK_COLOR.get((a, b), COLOR_WHITE)
                pt1 = (int(person[a, 0]), int(person[a, 1]))
                pt2 = (int(person[b, 0]), int(person[b, 1]))
                cv2.line(canvas, pt1, pt2, color, 3, cv2.LINE_AA)

        # 颈-鼻插值连线（白色）
        if NECK_NOSE_LINK and neck is not None and visible[0]:
            cv2.line(canvas,
                     (int(neck[0]), int(neck[1])),
                     (int(person[0, 0]), int(person[0, 1])),
                     COLOR_WHITE, 3, cv2.LINE_AA)

        # 火柴人圆头（白色空心圆）
        if visible[0]:
            head_radius = 0.0
            nose_x, nose_y = float(person[0, 0]), float(person[0, 1])
            if visible[1] and visible[2]:
                eye_dist = float(np.hypot(person[1, 0] - person[2, 0],
                                           person[1, 1] - person[2, 1]))
                head_radius = eye_dist * 1.8
            elif neck is not None:
                neck_dist = float(np.hypot(nose_x - neck[0], nose_y - neck[1]))
                head_radius = neck_dist * 1.1
            elif visible[5] and visible[6]:
                shoulder_dist = float(np.hypot(person[5, 0] - person[6, 0],
                                                person[5, 1] - person[6, 1]))
                head_radius = shoulder_dist * 0.35
            if head_radius > 3:
                cv2.circle(canvas,
                           (int(nose_x), int(nose_y)),
                           int(head_radius),
                           COLOR_WHITE, 3, cv2.LINE_AA)

        # 画关键点（按部位着色）
        for i in range(17):
            if visible[i]:
                color = _POINT_COLOR.get(i, COLOR_WHITE)
                cv2.circle(canvas,
                           (int(person[i, 0]), int(person[i, 1])),
                           6, color, -1, cv2.LINE_AA)

        # 画颈部关键点（白色）
        if neck is not None:
            cv2.circle(canvas, (int(neck[0]), int(neck[1])), 6, COLOR_WHITE, -1, cv2.LINE_AA)

    return canvas


def save_pose_npz(keypoints: np.ndarray, path: str) -> None:
    """
    保存单帧姿态 NPZ。
    包含: keypoints (N,17,3) + num_persons (int)
    """
    np.savez(path, keypoints=keypoints, num_persons=len(keypoints))


def load_pose_npz(path: str) -> tuple[np.ndarray, int]:
    """读取单帧姿态 NPZ，返回 (keypoints, num_persons)。"""
    data = np.load(path)
    return data["keypoints"], int(data["num_persons"])
