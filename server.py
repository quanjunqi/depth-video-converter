#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换器 · 本地 DA3 推理服务

浏览器前端（index.html）选择"本地 DA3 模型"时，通过本服务完成推理：
1. 接收前端上传的视频与参数（模型/深度方向/对比度/亮度/尺寸/帧率/逐帧/保留原声）
2. 调用原项目 da3_video.py 做 DA3 深度推理（逐帧或窗口模式），导出每帧深度 NPZ
3. 按参数渲染灰度深度视频（逐帧 min-max 归一化），可选 ffmpeg 混入原声
4. 向前端回报进度，提供结果下载

用法:
    python3 server.py            # 默认 127.0.0.1:8765
    bash start_server.sh         # 一键启动
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

# ---- 路径定位 ------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# 推理引擎 = 用户家目录下的 depth_video_converter 项目（DA3 官方代码所在）
DA3_HOME = os.environ.get("DA3_HOME") or HERE
DA3_SCRIPT = os.path.join(DA3_HOME, "da3_video.py")
# 本地模型缓存：本站点 weights/（DA3 模型已移动至此；原项目处为软链）
MODELS_DIR = os.path.join(HERE, "weights")

UPLOAD_DIR = os.path.join(HERE, "uploads")
OUTPUT_DIR = os.path.join(HERE, "outputs")
LOG_DIR = os.path.join(HERE, "logs")
DEPTH_DIR = os.path.join(HERE, "depth_tmp")
PREVIEW_DIR = os.path.join(HERE, "preview")
CLIPS_DIR = os.path.join(HERE, "clips")
AUDIO_OUTPUT_DIR = os.path.join(HERE, "audio_output")
JOBS_FILE = os.path.join(HERE, "jobs.json")
MAX_ACTIVE_JOBS = 2   # 最多同时运行的任务数
STATIC_DIR = os.path.join(HERE, "app")
for d in (UPLOAD_DIR, OUTPUT_DIR, LOG_DIR, DEPTH_DIR, PREVIEW_DIR, CLIPS_DIR, AUDIO_OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

PORT = int(os.environ.get("PORT", "8765"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB


@app.after_request
def no_cache(resp):
    """本地工具：静态文件不缓存，确保浏览器每次刷新拿到最新前端代码。"""
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# ---- 任务持久化：jobs.json 磁盘镜像，服务重启后任务列表不丢 ---------------
def _job_meta(job: dict) -> dict:
    """只保留可序列化字段（proc/输入输出路径等瞬态项不入库）。"""
    keep = ("id", "status", "progress", "current", "total", "model", "fps",
            "input_name", "created_at", "updated_at", "error", "info",
            "output_path", "log_path", "preview_path", "last_preview",
            "stage", "input_path", "pose_enabled", "pose_output", "pose_info")
    return {k: job.get(k) for k in keep if k in job}


def _save_jobs() -> None:
    try:
        with JOBS_LOCK:
            data = {jid: _job_meta(j) for jid, j in JOBS.items()}
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, JOBS_FILE)
    except Exception as e:  # noqa: BLE001
        print(f"[server] jobs.json 写入失败: {e}")


def _load_jobs() -> None:
    if not os.path.isfile(JOBS_FILE):
        return
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for jid, meta in data.items():
            job = dict(meta)
            job["id"] = jid
            job["proc"] = None
            if job.get("status") in ("queued", "running", "render"):
                job["status"] = "error"
                job["error"] = "服务重启，任务中断，请重新提交"
                job["updated_at"] = time.time()
            JOBS[jid] = job
        _save_jobs()
    except Exception as e:  # noqa: BLE001
        print(f"[server] jobs.json 加载失败: {e}")


PROGRESS_RE = None  # 在 _run 中按需编译


def _cleanup_job_files(job: dict) -> None:
    """删除任务产出的全部资产：输入视频/输出视频/日志/预览图/深度临时目录。"""
    for path in (job.get("input_path"), job.get("output_path"), job.get("log_path")):
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except Exception:  # noqa: BLE001
            pass
    try:
        pv = os.path.join(PREVIEW_DIR, job["id"] + ".jpg")
        if os.path.isfile(pv):
            os.remove(pv)
        pose_pv = os.path.join(PREVIEW_DIR, job["id"] + "_pose.png")
        if os.path.isfile(pose_pv):
            os.remove(pose_pv)
    except Exception:  # noqa: BLE001
        pass
    try:
        pose_out = job.get("pose_output")
        if pose_out and os.path.isfile(pose_out):
            os.remove(pose_out)
    except Exception:  # noqa: BLE001
        pass
    try:
        d = os.path.join(DEPTH_DIR, job["id"])
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass

# 本地已就绪的 DA3 模型 → (显示名, 模型 preset)
def _local_models() -> list[dict]:
    """扫描本站点 weights/ 下的 DA3 模型（huggingface 缓存目录）。"""
    out = []
    known = {
        "DA3-SMALL": ("DA3 Small（80M）", "da3-small"),
        "DA3-BASE": ("DA3 Base（120M）", "da3-base"),
        "DA3-LARGE": ("DA3 Large（350M）", "da3-large"),
        "DA3MONO-LARGE": ("DA3 Mono Large（350M）", "da3mono-large"),
        "DA3METRIC-LARGE": ("DA3 Metric Large（米制）", "da3metric-large"),
        "DA3-GIANT": ("DA3 Giant（1.15B）", "da3-giant"),
    }
    for name, (label, preset) in known.items():
        p = os.path.join(MODELS_DIR, f"models--depth-anything--{name}")
        if os.path.isdir(p):
            ok = bool(glob.glob(os.path.join(p, "blobs", "*")))
            out.append({"id": preset, "label": label, "ready": ok})
    return out


# ---- 渲染：NPZ 深度 → 灰度视频（对比度/亮度/方向参数） ----------------------
def _contrast_factor(c: int) -> float:
    """对比度公式，c ∈ [-100, 100]。"""
    return (259 * (c + 255)) / (255 * (259 - c))


def render_depth_video(npz_dir: str, src_video: str, out_path: str,
                       params: dict) -> dict:
    """按参数渲染灰度深度视频（ffmpeg 管道编码 H.264，避免 macOS cv2 mp4 写入失败）。

    params: invert / contrast / brightness / scale / fps / per_frame
            depth_clip（百分位截断，0=关闭）/ clahe_clip（CLAHE对比度限制，0=关闭）/ clahe_tile（CLAHE块大小）
    深度语义与 da3_video 一致：小值=近、大值=远；默认"近亮远暗"。
    """
    files = sorted(glob.glob(os.path.join(npz_dir, "depth_*.npz")))
    if not files:
        raise RuntimeError("未找到深度 NPZ 输出")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("未找到 ffmpeg，无法渲染输出视频")
    cap = cv2.VideoCapture(src_video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fps = float(params.get("fps") or 0) or 25.0
    scale = params.get("scale") or "1"
    if str(scale) == "min640":
        k = max(1.0, 640.0 / max(1, min(src_w, src_h)))
        ow = max(640, int(round(src_w * k)))
        oh = max(640, int(round(src_h * k)))
    else:
        s = float(scale or 1.0)
        ow = max(1, int(round(src_w * s)))
        oh = max(1, int(round(src_h * s)))
    ow -= ow % 2
    oh -= oh % 2
    invert = bool(params.get("invert"))
    contrast = int(params.get("contrast") or 0)
    brightness = int(params.get("brightness") or 0)
    per_frame = bool(params.get("per_frame", True))
    depth_clip = float(params.get("depth_clip") or 0)  # 百分位截断，0=关闭
    clahe_clip = float(params.get("clahe_clip") or 0)  # CLAHE对比度限制，0=关闭
    clahe_tile = int(params.get("clahe_tile") or 8)    # CLAHE块大小
    sharpen = float(params.get("sharpen") or 0)        # 边缘锐化强度，0=关闭
    sharpen_radius = float(params.get("sharpen_radius") or 2.0)  # 锐化模糊半径
    local_norm = float(params.get("local_norm") or 0)  # 局部归一化强度，0=关闭
    local_norm_radius = int(params.get("local_norm_radius") or 15)  # 局部窗口半径
    cf = _contrast_factor(contrast)

    # CLAHE 对象（仅在开启时创建）
    clahe = None
    if clahe_clip > 0:
        tile = max(2, min(32, clahe_tile))
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))

    # 稳定模式：全片 1%/99% 分位数（一次性），抑制帧间闪烁
    if not per_frame and files:
        all_d = []
        for fp in files[: min(len(files), 60)]:
            all_d.append(np.load(fp)["depth"].ravel())
        arr = np.concatenate(all_d)
        lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    else:
        lo = hi = None

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{ow}x{oh}",
           "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fp in files:
            d = np.load(fp)["depth"].astype(np.float32)
            if per_frame:
                if depth_clip > 0:
                    # 百分位截断：去掉离群值，让主体占据更多动态范围
                    cp = min(max(depth_clip, 0.1), 49.0)
                    lo = float(np.percentile(d, cp))
                    hi = float(np.percentile(d, 100 - cp))
                else:
                    lo, hi = float(np.min(d)), float(np.max(d))
            span = max(hi - lo, 1e-6)
            norm = np.clip((d - lo) / span, 0.0, 1.0)
            gray = (norm * 255.0 if invert else (1.0 - norm) * 255.0)
            gray = cv2.resize(gray, (ow, oh), interpolation=cv2.INTER_CUBIC)
            if contrast != 0 or brightness != 0:
                gf = gray.astype(np.float32)
                gf = cf * (gf - 128.0) + 128.0 + brightness
                gray = np.clip(gf, 0, 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
            # CLAHE 局部对比度增强（增强人物内部深度差异）
            if clahe is not None:
                gray = clahe.apply(gray)
            # Unsharp Mask 边缘锐化（增强人物轮廓，无块状伪影）
            if sharpen > 0:
                blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sharpen_radius)
                gray = cv2.addWeighted(gray, 1.0 + sharpen, blur, -sharpen, 0)
            # 局部深度归一化（解决肢体重叠时深度接近无法区分的问题）
            if local_norm > 0:
                k = max(3, local_norm_radius * 2 + 1)
                kernel = np.ones((k, k), np.uint8)
                local_min = cv2.erode(gray, kernel)
                local_max = cv2.dilate(gray, kernel)
                span = np.maximum(local_max.astype(np.float32) - local_min.astype(np.float32), 1.0)
                local_normed = ((gray.astype(np.float32) - local_min.astype(np.float32)) / span * 255.0)
                gray = cv2.addWeighted(gray, 1.0 - local_norm, local_normed.astype(np.uint8), local_norm, 0)
            proc.stdin.write(gray.tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"ffmpeg 渲染失败 (rc={rc})")
    return {"frames": len(files), "fps": fps, "size": f"{ow}x{oh}"}


def render_pose_video(npz_dir: str, src_video: str, out_path: str,
                       params: dict) -> dict:
    """渲染纯骨架视频（ffmpeg 管道编码，参数与深度视频完全一致：fps/分辨率/编码器）。

    读 pose_*.npz（keypoints + num_persons），用 yolo_pose.draw_skeleton 绘制黑底白骨架。
    """
    import yolo_pose as yp

    files = sorted(glob.glob(os.path.join(npz_dir, "pose_*.npz")))
    if not files:
        raise RuntimeError("未找到姿态 NPZ 输出")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("未找到 ffmpeg，无法渲染骨架视频")
    cap = cv2.VideoCapture(src_video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fps = float(params.get("fps") or 0) or 25.0
    scale = params.get("scale") or "1"
    if str(scale) == "min640":
        k = max(1.0, 640.0 / max(1, min(src_w, src_h)))
        ow = max(640, int(round(src_w * k)))
        oh = max(640, int(round(src_h * k)))
    else:
        s = float(scale or 1.0)
        ow = max(1, int(round(src_w * s)))
        oh = max(1, int(round(src_h * s)))
    ow -= ow % 2
    oh -= oh % 2

    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
           "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fp in files:
            data = np.load(fp)
            kpts = data["keypoints"]
            skeleton = yp.draw_skeleton(kpts, src_h, src_w)
            skeleton = cv2.resize(skeleton, (ow, oh), interpolation=cv2.INTER_CUBIC)
            proc.stdin.write(skeleton.astype(np.uint8).tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) < 1000:
        raise RuntimeError(f"ffmpeg 骨架渲染失败 (rc={rc})")
    return {"frames": len(files), "fps": fps, "size": f"{ow}x{oh}"}


def _mux_audio(video_path: str, src_video: str) -> bool:
    """用 ffmpeg 把源视频音轨混入深度视频（替换原文件）。"""
    if not shutil.which("ffmpeg"):
        return False
    tmp = video_path + ".snd.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", src_video,
         "-map", "0:v", "-map", "1:a?", "-c", "copy", "-shortest", tmp],
        capture_output=True)
    if r.returncode == 0 and os.path.isfile(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, video_path)
        return True
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def _make_preview(jid: str) -> str | None:
    """把最新一帧深度 NPZ 渲染成灰度预览图（近亮远暗），返回 URL 或 None。"""
    npzs = sorted(glob.glob(os.path.join(DEPTH_DIR, jid, "depth_*.npz")))
    if not npzs:
        return None
    fp = npzs[-1]
    job = JOBS.get(jid)
    if job and job.get("last_preview") == os.path.basename(fp):
        pp = job.get("preview_path")
        if pp and os.path.isfile(pp):
            return f"/preview/{jid}.jpg?t={int(os.path.getmtime(pp))}"
        return None
    try:
        d = np.load(fp)["depth"].astype(np.float32)
        lo, hi = float(np.min(d)), float(np.max(d))
        span = max(hi - lo, 1e-6)
        norm = np.clip((d - lo) / span, 0.0, 1.0)
        gray = ((1.0 - norm) * 255.0).astype(np.uint8)  # 近亮远暗
        h, w = gray.shape
        if w > 400:
            nh = int(round(h * 400 / w))
            gray = cv2.resize(gray, (400, nh), interpolation=cv2.INTER_AREA)
        out = os.path.join(PREVIEW_DIR, f"{jid}.jpg")
        cv2.imwrite(out, gray)
        if job:
            job["last_preview"] = os.path.basename(fp)
            job["preview_path"] = out
        return f"/preview/{jid}.jpg?t={int(os.path.getmtime(out))}"
    except Exception:
        return None


# ---- 任务执行 ------------------------------------------------------------
INFER_LOCK = threading.Lock()   # 进程内推理串行锁（MPS 单设备安全）


def _run(job: dict, p: dict) -> None:
    log_path = job["log_path"]
    def log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    npz_dir = os.path.join(DEPTH_DIR, job["id"])
    try:
        import da3_video as dv   # 同目录引擎模块（模型进程内常驻）

        in_path = job["input_path"]
        model = p.get("model", "da3-small")
        window, overlap = (1, 0) if p.get("frame_wise") else (8, 2)
        pose_enabled = bool(p.get("pose_enabled", False))
        yp = None
        if pose_enabled:
            import yolo_pose as yp
        os.makedirs(npz_dir, exist_ok=True)

        log(f"引擎: {os.path.abspath(dv.__file__)}")
        log(f"模型: {model} | 窗口 {window} / 重叠 {overlap} | 处理分辨率 756"
            + (" | 同时输出骨架" if pose_enabled else ""))
        log(f"开始深度推理（{'逐帧' if window == 1 else '多视角窗口'}模式）...")

        t0 = time.time()
        cap = cv2.VideoCapture(in_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total <= 0:
            cap.release()
            raise RuntimeError("无法读取视频帧数")
        cap.release()

        job["status"] = "running"
        job["progress"] = 0.02
        job["pose_enabled"] = pose_enabled
        job["updated_at"] = time.time()
        _save_jobs()

        with INFER_LOCK:   # 推理串行：同时只跑一个任务（MPS 稳定）
            if job.get("stop_requested"):
                log("任务已被停止，跳过推理")
            else:
                model_obj, dev, label = dv.get_model(model)
                job["current"], job["total"] = 0, total
                _save_jobs()
                cap = cv2.VideoCapture(in_path)
                half = window // 2
                for i in range(total):
                    if job.get("stop_requested"):
                        log(f"收到停止指令，中断推理（已处理 {i}/{total} 帧）")
                        break
                    if window <= 1:
                        ok, bgr = cap.read()
                        if not ok:
                            break
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        depth = dv.infer_depth(model_obj, [rgb], 756)[0]
                        cur_rgb = rgb
                    else:
                        idxs = [max(0, min(total - 1, i + off)) for off in range(-half, half + 1)]
                        frames = []
                        for ix in idxs:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
                            ok, bgr = cap.read()
                            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok
                                          else np.zeros((src_h, src_w, 3), np.uint8))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        depths = dv.infer_depth(model_obj, frames, 756)
                        cur_idx = idxs.index(i) if i in idxs else half
                        depth = depths[cur_idx]
                        cur_rgb = frames[cur_idx]
                    np.savez(os.path.join(npz_dir, f"depth_{i:06d}.npz"), depth=depth)
                    # 姿态估计（同时输出骨架）
                    if pose_enabled and yp is not None:
                        kpts = yp.infer_pose(cur_rgb)
                        yp.save_pose_npz(kpts, os.path.join(npz_dir, f"pose_{i:06d}.npz"))
                        if (i + 1) % 5 == 0 or (i + 1) == total:
                            pose_frame = yp.draw_skeleton(kpts, src_h, src_w)
                            cv2.imwrite(os.path.join(PREVIEW_DIR, f"{job['id']}_pose.png"), pose_frame)
                    job["current"] = i + 1
                    if (i + 1) % 5 == 0 or (i + 1) == total:
                        el = time.time() - t0
                        eta = el / max(i + 1, 1) * (total - i - 1)
                        log(f"推理进度 {i + 1}/{total}  已写 {i + 1} 帧  耗时 {el:.1f}s  预计剩余 {eta:.1f}s")
                    if (i + 1) % 20 == 0:
                        job["progress"] = round((i + 1) / max(total, 1) * 0.9, 3)
                        job["updated_at"] = time.time()
                        _save_jobs()
                cap.release()

        if job.get("stop_requested"):
            job["status"] = "stopped"
            job["progress"] = round(job.get("current", 0) / max(total, 1), 3)
            job["updated_at"] = time.time()
            _save_jobs()
            log(f"已手动停止（已处理 {job.get('current', 0)}/{total} 帧）")
            return

        job["stage"] = "render"
        job["status"] = "render"
        log("深度推理完成，开始渲染灰度视频...")
        job["progress"] = 0.92
        job["updated_at"] = time.time()
        _save_jobs()
        info = render_depth_video(npz_dir, in_path, job["output_path"], p)
        if p.get("keep_audio"):
            ok = _mux_audio(job["output_path"], in_path)
            log(f"保留原声: {'成功' if ok else '失败/源视频无音轨或缺少 ffmpeg'}")
        # 同时输出骨架视频
        if pose_enabled:
            pose_out = os.path.join(OUTPUT_DIR, job["id"] + "_pose.mp4")
            pose_info = render_pose_video(npz_dir, in_path, pose_out, p)
            job["pose_output"] = pose_out
            job["pose_info"] = pose_info
            log(f"骨架视频完成: {pose_out} ({pose_info['frames']}帧 {pose_info['size']})")
        job["progress"] = 1.0
        job["info"] = info
        job["status"] = "done"
        job["updated_at"] = time.time()
        _save_jobs()
        log(f"完成: {job['output_path']} ({info['frames']} 帧 {info['size']} @ {info['fps']}fps)")
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)
        job["updated_at"] = time.time()
        _save_jobs()
        log(f"错误: {e}")
    finally:
        shutil.rmtree(npz_dir, ignore_errors=True)
        if os.path.exists(os.path.join(OUTPUT_DIR, job["id"] + "_gray_tmp.mp4")):
            os.remove(os.path.join(OUTPUT_DIR, job["id"] + "_gray_tmp.mp4"))


def _status(job: dict) -> dict:
    tail = ""
    try:
        with open(job["log_path"], "r", errors="replace") as f:
            tail = "".join(f.readlines()[-8:]).rstrip()
    except FileNotFoundError:
        pass
    st = job.get("status", "queued")
    if job["proc"] is not None:
        code = job["proc"].poll()
        if code is None:
            st = "running"
        elif code == 0 and st != "done":
            st = "render"
        elif code != 0:
            st = "error"
    return {"id": job["id"], "status": st, "progress": job.get("progress", 0.0),
            "log_tail": tail, "error": job.get("error"),
            "output_exists": bool(job["output_path"] and os.path.isfile(job["output_path"])),
            "pose_enabled": bool(job.get("pose_enabled")),
            "info": job.get("info")}


# ---- API ----------------------------------------------------------------
@app.route("/")
def index_page():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:fp>")
def static_files(fp: str):
    """托管站点静态资源（app/）与逐帧预览图（preview/），仅限站点内文件。"""
    # 逐帧预览图（深度帧面板）
    if fp.startswith("preview/"):
        name = os.path.basename(fp)
        pv = os.path.join(PREVIEW_DIR, name)
        if os.path.isfile(pv):
            return send_from_directory(PREVIEW_DIR, name)
        return jsonify({"error": "not found"}), 404
    safe = os.path.normpath(os.path.join(STATIC_DIR, fp))
    if safe.startswith(STATIC_DIR + os.sep) and os.path.isfile(safe):
        return send_from_directory(STATIC_DIR, fp)
    return jsonify({"error": "not found"}), 404


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "name": "深度视频转换器 · DA3 本地服务",
                    "models": _local_models(),
                    "engine": os.path.isfile(DA3_SCRIPT)})


@app.route("/api/convert", methods=["POST"])
def convert():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify({"error": "未收到视频文件"}), 400
    ext = os.path.splitext(f.filename)[1].lower() or ".mp4"
    jid = uuid.uuid4().hex[:10]
    in_path = os.path.join(UPLOAD_DIR, f"{jid}{ext}")
    f.save(in_path)

    def _b(name, default=None):
        v = request.form.get(name, default)
        return v

    params = {
        "model": _b("model", "da3-small"),
        "invert": _b("invert", "false") == "true",
        "contrast": int(_b("contrast", "0")),
        "brightness": int(_b("brightness", "0")),
        "scale": _b("scale", "1"),   # 数字倍数 或 min640（短边不低于 640，只放大不缩小）
        "fps": float(_b("fps", "0")) if _b("fps", "0") not in ("", "0") else None,
        "frame_wise": _b("frame_wise", "false") == "true",
        "keep_audio": _b("keep_audio", "false") == "true",
        "per_frame": _b("per_frame", "true") == "true",
        "pose_enabled": _b("pose_enabled", "false") == "true",
        "depth_clip": float(_b("depth_clip", "0")),     # 百分位截断，0=关闭
        "clahe_clip": float(_b("clahe_clip", "0")),      # CLAHE对比度限制，0=关闭
        "clahe_tile": int(_b("clahe_tile", "8")),         # CLAHE块大小
        "sharpen": float(_b("sharpen", "0")),             # 边缘锐化强度，0=关闭
        "sharpen_radius": float(_b("sharpen_radius", "2")),  # 锐化模糊半径
        "local_norm": float(_b("local_norm", "0")),       # 局部归一化强度，0=关闭
        "local_norm_radius": int(_b("local_norm_radius", "15")),  # 局部窗口半径
    }
    if not params["fps"]:
        # 自动：保持原视频帧率，避免输出时长变化
        cap = cv2.VideoCapture(in_path)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        cap.release()
        params["fps"] = src_fps
    out_path = os.path.join(OUTPUT_DIR, f"{jid}_depth.mp4")
    now = time.time()
    job = {"id": jid, "proc": None, "input_path": in_path,
           "output_path": out_path, "log_path": os.path.join(LOG_DIR, f"{jid}.log"),
           "progress": 0.0, "status": "queued", "stage": "infer", "error": None,
           "info": None, "model": params["model"], "fps": params["fps"],
           "input_name": f.filename or os.path.basename(in_path),
           "created_at": now, "updated_at": now}
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values()
                     if j.get("status") in ("queued", "running", "render"))
        if active >= MAX_ACTIVE_JOBS:
            os.remove(in_path)
            return jsonify({"error": f"最多同时运行 {MAX_ACTIVE_JOBS} 个任务，"
                                    "请等待当前任务完成后再试"}), 429
        JOBS[jid] = job
    _save_jobs()
    threading.Thread(target=_run, args=(job, params), daemon=True).start()
    return jsonify({"job_id": jid})


@app.route("/api/progress/<jid>")
def progress(jid: str):
    job = JOBS.get(jid)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    # 推理阶段从日志解析实时进度
    s = _status(job)
    if s["status"] in ("running", "render"):
        try:
            with open(job["log_path"], "r", errors="replace") as f:
                txt = f.read()
            import re
            m = re.findall(r"推理进度 (\d+)/(\d+)", txt)
            if m:
                done, total = int(m[-1][0]), int(m[-1][1])
                s["progress"] = round(done / max(total, 1) * 0.9, 3)
                s["current"] = done
                s["total"] = total
        except FileNotFoundError:
            pass
        if s["status"] == "running":
            url = _make_preview(jid)
            if url:
                s["latest_frame"] = url
            # 骨架帧预览（PNG，避免 JPEG 压缩伪影）
            if job.get("pose_enabled"):
                pose_png = os.path.join(PREVIEW_DIR, f"{jid}_pose.png")
                if os.path.isfile(pose_png):
                    s["latest_pose_frame"] = f"/preview/{jid}_pose.png?t={int(os.path.getmtime(pose_png))}"
    # 进度写回任务记录并落盘（刷新/服务重启后仍可恢复）
    job["status"] = s["status"]
    job["progress"] = s["progress"]
    if s.get("current") is not None:
        job["current"], job["total"] = s["current"], s["total"]
    job["updated_at"] = time.time()
    _save_jobs()
    return jsonify(s)


@app.route("/api/jobs")
def jobs_list():
    """任务列表（最近优先，最多保留 6 条）。"""
    with JOBS_LOCK:
        items = sorted(JOBS.values(),
                       key=lambda j: j.get("created_at", 0), reverse=True)[:6]
        views = []
        for job in items:
            v = _job_meta(job)
            st = _status(job)["status"]
            v["status"] = st
            if st in ("running", "render"):
                v["latest_frame"] = _make_preview(job["id"])
                if job.get("pose_enabled"):
                    pose_png = os.path.join(PREVIEW_DIR, f"{job['id']}_pose.png")
                    if os.path.isfile(pose_png):
                        v["latest_pose_frame"] = f"/preview/{job['id']}_pose.png?t={int(os.path.getmtime(pose_png))}"
            v["can_download"] = (st == "done" and job.get("output_path")
                                 and os.path.isfile(job["output_path"]))
            v["can_download_pose"] = (st == "done" and job.get("pose_enabled")
                                       and job.get("pose_output")
                                       and os.path.isfile(job["pose_output"]))
            views.append(v)
    return jsonify({"jobs": views})


@app.route("/api/jobs/delete", methods=["POST"])
def jobs_delete():
    """批量删除任务（同时清理该任务产出的全部文件）。运行中的任务不可删。"""
    data = request.get_json(silent=True) or {}
    ids = data.get("ids") or []
    if not ids:
        return jsonify({"error": "未指定任务"}), 400
    with JOBS_LOCK:
        for jid in ids:
            job = JOBS.get(jid)
            if job and job.get("status") in ("queued", "running", "render"):
                return jsonify({"error": f"任务「{job.get('input_name') or jid}」正在运行，无法删除"}), 409
        for jid in ids:
            job = JOBS.pop(jid, None)
            if job:
                _cleanup_job_files(job)
    _save_jobs()
    return jsonify({"ok": True, "deleted": len(ids)})


@app.route("/api/jobs/<jid>/stop", methods=["POST"])
def stop_job(jid):
    """停止正在运行或排队的任务。"""
    with JOBS_LOCK:
        job = JOBS.get(jid)
        if not job:
            return jsonify({"error": "任务不存在"}), 404
        if job.get("status") in ("done", "error", "stopped"):
            return jsonify({"error": "任务已结束，无法停止"}), 409
        job["stop_requested"] = True
    return jsonify({"ok": True})


def _safe_project_id(pid: str) -> str:
    """清理项目编号，仅保留字母数字和 -_，防路径穿越。"""
    pid = "".join(c for c in (pid or "") if c.isalnum() or c in "-_")
    return pid or time.strftime("clip_%Y%m%d_%H%M%S")


@app.route("/api/clip", methods=["POST"])
def clip_video():
    """视频分段裁剪：按指定秒数将视频切成多段，按项目编号存到 clips/ 目录。"""
    video = request.files.get("video")
    if not video or not video.filename:
        return jsonify({"error": "未上传视频"}), 400

    seg_raw = request.form.get("segment_seconds", "10")
    try:
        seg_sec = float(seg_raw)
        if seg_sec <= 0:
            return jsonify({"error": "分段时长必须大于 0 秒"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "分段时长格式错误"}), 400

    if not shutil.which("ffmpeg"):
        return jsonify({"error": "未找到 ffmpeg，无法裁剪视频"}), 500

    project_id = _safe_project_id(request.form.get("project_id", ""))
    out_dir = os.path.join(CLIPS_DIR, project_id)
    os.makedirs(out_dir, exist_ok=True)

    ext = video.filename.rsplit(".", 1)[-1].lower() if "." in video.filename else "mp4"
    src_path = os.path.join(out_dir, f"_source.{ext}")
    video.save(src_path)

    out_pattern = os.path.join(out_dir, "segment_%03d.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-c", "copy",
        "-f", "segment",
        "-segment_time", str(seg_sec),
        "-segment_start_number", "1",
        "-reset_timestamps", "1",
        "-avoid_negative_ts", "make_zero",
        out_pattern,
    ]
    rc = subprocess.call(cmd)
    if rc != 0:
        return jsonify({"error": f"ffmpeg 分段失败 (返回码 {rc})"}), 500

    segments = []
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("segment_") and f.endswith(".mp4"):
            fp = os.path.join(out_dir, f)
            segments.append({
                "name": f,
                "size": os.path.getsize(fp),
                "url": f"/api/clip/download/{project_id}/{f}",
            })

    return jsonify({
        "project_id": project_id,
        "output_dir": out_dir,
        "segments": segments,
        "count": len(segments),
        "segment_seconds": seg_sec,
    })


@app.route("/api/clip/download/<project_id>/<filename>")
def clip_download(project_id, filename):
    """下载裁剪后的分段视频。"""
    project_id = _safe_project_id(project_id)
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "非法文件名"}), 400
    fp = os.path.join(CLIPS_DIR, project_id, filename)
    if not os.path.isfile(fp):
        return jsonify({"error": "文件不存在"}), 404
    return send_file(fp, as_attachment=True)


@app.route("/api/download/<jid>")
def download(jid: str):
    job = JOBS.get(jid)
    if not job:
        return jsonify({"error": "任务不存在"}), 404
    dtype = request.args.get("type", "depth")
    stem = os.path.splitext(os.path.basename(job["input_path"]))[0]
    if dtype == "pose":
        pose_path = job.get("pose_output")
        if not pose_path or not os.path.isfile(pose_path):
            return jsonify({"error": "骨架视频不存在"}), 404
        return send_file(pose_path, as_attachment=True,
                         download_name=f"{stem}_pose.mp4")
    if not os.path.isfile(job["output_path"]):
        return jsonify({"error": "结果不存在"}), 404
    return send_file(job["output_path"], as_attachment=True,
                     download_name=f"{stem}_depth.mp4")


# ============================================================
# 视频音频处理 API（只保留人声 / 只保留环境音 / 无声音）
# ============================================================
import audio_processor as ap


@app.route("/api/audio-process/status")
def audio_process_status():
    """检查 Demucs 可用性"""
    return jsonify(ap.check_demucs())


@app.route("/api/audio-process/upload", methods=["POST"])
def audio_process_upload():
    """上传视频"""
    if "video" not in request.files:
        return jsonify({"error": "未找到视频文件"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "文件名为空"}), 400
    safe_name = os.path.basename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    f.save(save_path)
    return jsonify({"filename": safe_name, "path": save_path})


@app.route("/api/audio-process/start", methods=["POST"])
def audio_process_start():
    """开始音频处理"""
    data = request.get_json(force=True)
    video_path = data.get("video_path", "")
    mode = data.get("mode", "mute")  # vocals_only | ambient_only | mute
    model = data.get("model", "htdemucs")

    if not video_path or not os.path.exists(video_path):
        # 尝试用 filename
        filename = data.get("filename", "")
        if filename:
            video_path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
        if not os.path.exists(video_path):
            return jsonify({"error": "视频文件不存在"}), 400

    if mode not in ("vocals_only", "ambient_only", "mute"):
        return jsonify({"error": "无效的处理模式"}), 400

    task_id = ap.create_task(video_path, mode, AUDIO_OUTPUT_DIR, model)
    return jsonify({"task_id": task_id})


@app.route("/api/audio-process/task/<task_id>")
def audio_process_task(task_id):
    """查询任务状态"""
    task = ap.get_task(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(task)


@app.route("/api/audio-process/download/<task_id>")
def audio_process_download(task_id):
    """下载处理结果"""
    task = ap.get_task(task_id)
    if not task or task["status"] != "done":
        return jsonify({"error": "任务未完成"}), 404
    out_path = task["result"].get("output_path", "")
    if not out_path or not os.path.exists(out_path):
        return jsonify({"error": "结果文件不存在"}), 404
    return send_file(out_path, as_attachment=True, download_name=os.path.basename(out_path))


@app.route("/api/audio-process/download-stem/<task_id>/<stem>")
def audio_process_download_stem(task_id, stem):
    """下载分离的音轨（vocals / ambient）"""
    task = ap.get_task(task_id)
    if not task or task["status"] != "done":
        return jsonify({"error": "任务未完成"}), 404
    key = "vocals_path" if stem == "vocals" else "ambient_path"
    path = task["result"].get(key, "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "音轨不存在"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


def _warmup_default_model() -> None:
    """服务启动时后台加载并预热默认模型（MPS 首次推理编译很慢，提前完成）。"""
    try:
        import da3_video as dv
        with INFER_LOCK:
            m, dev, label = dv.get_model("da3-small")
            img = np.zeros((420, 640, 3), np.uint8)
            dv.infer_depth(m, [img], 756)
        print(f"[server] 默认模型 {label} 已预热（{dev}）")
    except Exception as e:  # noqa: BLE001
        print(f"[server] 模型预热失败（任务时仍会按需加载）: {e}")


if __name__ == "__main__":
    _load_jobs()
    threading.Thread(target=_warmup_default_model, daemon=True).start()
    print(f"深度视频转换器 DA3 本地服务: http://127.0.0.1:{PORT}")
    print(f"已就绪模型: {[m['id'] for m in _local_models()] or '（无，请检查 weights/ 目录）'}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
