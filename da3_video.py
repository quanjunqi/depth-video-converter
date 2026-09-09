#!/usr/bin/env python3
"""深度视频转换器 · DA3 深度估计引擎

基于官方 Depth Anything 3 推理 API，对视频逐帧（或窗口多视角）估计深度：
- 输出灰度深度视频（近亮远暗）或 Turbo 彩色深度视频
- 可导出每帧深度 NPZ（供上层渲染使用）

用法示例:
  python3 da3_video.py input.mp4 -o output.mp4 --model da3-small --style gray \\
      --window 1 --overlap 0 --process-res 756 --save-depth ./depth_npz
"""
import argparse
import glob
import os
import sys
import threading
import time

# torch / cv2 均链接 OpenMP，macOS 下需允许重复加载，否则 OMP Error #15
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
# 官方推理库（depth_anything_3）源码路径，可用 DA3_SRC 覆盖
DA3_SRC = os.environ.get("DA3_SRC") or os.path.join(
    os.path.expanduser("~"), "da3-video", "depth-anything-3", "src")
if DA3_SRC not in sys.path:
    sys.path.insert(0, DA3_SRC)

MODELS = {
    # preset: (HuggingFace repo, 显示名, 许可证, 类型)
    "da3-small": ("depth-anything/DA3-SMALL", "DA3 Small（80M）", "Apache-2.0", "relative"),
    "da3-base": ("depth-anything/DA3-BASE", "DA3 Base（120M）", "Apache-2.0", "relative"),
    "da3-large": ("depth-anything/DA3-LARGE-1.1", "DA3 Large（350M）", "CC BY-NC 4.0", "relative"),
    "da3mono-large": ("depth-anything/DA3MONO-LARGE", "DA3 Mono Large（350M）", "Apache-2.0", "relative"),
    "da3metric-large": ("depth-anything/DA3METRIC-LARGE", "DA3 Metric Large（米制）", "Apache-2.0", "metric"),
    "da3-giant": ("depth-anything/DA3-GIANT-1.1", "DA3 Giant（1.15B）", "Apache-2.0", "relative"),
}


def log(msg: str) -> None:
    print(f"[da3-video] {msg}", flush=True)


def pick_device() -> str:
    if torch_available() and torch_mps():
        return "mps"
    return "cuda" if torch_available() and torch_cuda() else "cpu"


def torch_available() -> bool:
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


def torch_mps() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_model(preset: str):
    """加载 DA3 模型（官方 DepthAnything3，权重来自本地 weights/ 缓存）。"""
    if preset not in MODELS:
        raise SystemExit(f"未知模型 preset: {preset}，可选: {', '.join(MODELS)}")
    repo, label, lic, mtype = MODELS[preset]
    log(f"加载本地模型 {repo}（{label} · {lic}）...")
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(repo, cache_dir=WEIGHTS_DIR)
    model.eval()
    dev = pick_device()
    if dev != "cpu":
        try:
            model.to(dev)
        except Exception as e:
            log(f"设备切换 {dev} 失败（{e}），使用 CPU")
            dev = "cpu"
    log(f"模型就绪：{preset} @ {dev}")
    return model, dev, label


def depth_to_gray(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    """深度 → 灰度（逐帧 min-max 全范围拉伸；小值=近，默认近亮远暗）。"""
    d = depth.astype(np.float32)
    lo, hi = float(np.min(d)), float(np.max(d))
    span = max(hi - lo, 1e-6)
    norm = np.clip((d - lo) / span, 0.0, 1.0)
    gray = (norm * 255.0 if invert else (1.0 - norm) * 255.0)
    return gray.astype(np.uint8)


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    """深度 → Turbo 彩色（近=暖色）。"""
    gray = depth_to_gray(depth)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def open_writer(path: str, w: int, h: int, fps: float):
    """ffmpeg 管道写 H.264（macOS 下 cv2 mp4v 写入不可靠）。"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc


import subprocess  # noqa: E402


# ---- 进程内复用接口（server.py 常驻服务使用：模型只加载一次） ----
_MODEL_CACHE: dict[str, tuple] = {}
MODEL_LOCK = threading.Lock()


def get_model(preset: str):
    """按 preset 取模型（进程内缓存，最多常驻 2 个最近使用的模型）。"""
    with MODEL_LOCK:
        if preset in _MODEL_CACHE:
            _MODEL_CACHE[preset] = _MODEL_CACHE.pop(preset)  # 移到最近使用
            return _MODEL_CACHE[preset]
        model, dev, label = load_model(preset)
        if len(_MODEL_CACHE) >= 2:
            oldest = next(iter(_MODEL_CACHE))
            _MODEL_CACHE.pop(oldest)
            log(f"释放常驻模型 {oldest}")
        _MODEL_CACHE[preset] = (model, dev, label)
        return _MODEL_CACHE[preset]


def infer_depth(model, rgb_frames, process_res: int = 756):
    """单帧或窗口多帧推理，返回与输入顺序对齐的深度列表（float32 HxW，小=近）。"""
    pred = model.inference(rgb_frames, process_res=process_res,
                           process_res_method="upper_bound_resize")
    return [np.asarray(d, dtype=np.float32) for d in pred.depth]


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 视频深度估计引擎")
    ap.add_argument("input", help="输入视频")
    ap.add_argument("-o", "--output", required=True, help="输出视频路径")
    ap.add_argument("--model", default="da3-small", choices=list(MODELS), help="模型 preset")
    ap.add_argument("--style", default="gray", choices=["gray", "color", "side"], help="输出样式")
    ap.add_argument("--window", type=int, default=1, help="多视角窗口大小（1=逐帧）")
    ap.add_argument("--overlap", type=int, default=0, help="窗口重叠")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--save-depth", default=None, help="同时导出每帧深度 NPZ 到该目录")
    ap.add_argument("--fps", type=float, default=None, help="输出帧率（默认取源视频）")
    ap.add_argument("--process-res", type=int, default=504, help="模型处理分辨率（越大越精细）")
    # 兼容原版参数（高级增强暂未实现，解析保留）
    ap.add_argument("--no-smooth", action="store_true", help="关闭时序平滑")
    ap.add_argument("--enhance", action="store_true", help="导演式深度增强（暂未启用）")
    ap.add_argument("--bg-stabilize", action="store_true", help="背景稳定（暂未启用）")
    ap.add_argument("--ground-ratio", type=float, default=0.70, help="地面分界")
    ap.add_argument("--seg-model", choices=["u2net", "isnet"], default="u2net", help="分割模型")
    ap.add_argument("--seg-interval", type=int, default=1, help="分割间隔")
    args = ap.parse_args()

    if args.enhance or args.bg_stabilize:
        log("[警告] --enhance/--bg-stabilize 高级增强在当前版本未启用，将输出纯净深度图")

    if not os.path.isfile(args.input):
        raise SystemExit(f"输入视频不存在: {args.input}")
    if not os.path.isdir(DA3_SRC):
        raise SystemExit(f"未找到官方推理库: {DA3_SRC}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"无法读取视频: {args.input}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    fps = args.fps or src_fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    log(f"读取视频 {args.input} ...")
    log(f"共 {total} 帧 @ {W}x{H}，fps={src_fps:.2f}，窗口 {args.window} 帧 / 重叠 {args.overlap} 帧")

    model, dev, label = load_model(args.model)
    if args.save_depth:
        os.makedirs(args.save_depth, exist_ok=True)
        for old in glob.glob(os.path.join(args.save_depth, "depth_*.npz")):
            os.remove(old)

    out_w = W * 2 if args.style == "side" else W
    writer = open_writer(args.output, out_w, H, fps)

    window = max(1, args.window)
    t0 = time.time()
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or i >= total:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if window <= 1:
            pred = model.inference([rgb], process_res=args.process_res,
                                   process_res_method="upper_bound_resize")
            depth = pred.depth[0]
        else:
            half = window // 2
            idxs = [max(0, min(total - 1, i + off)) for off in range(-half, half + 1)]
            frames = _read_frames(cap, idxs, W, H)
            pred = model.inference(frames, process_res=args.process_res,
                                   process_res_method="upper_bound_resize")
            cur_pos = idxs.index(i) if i in idxs else half
            depth = pred.depth[cur_pos]

        depth = np.asarray(depth, dtype=np.float32)

        if args.save_depth:
            fp = os.path.join(args.save_depth, f"depth_{i:06d}.npz")
            np.savez(fp, depth=depth)

        if args.style == "gray":
            frame_out = depth_to_gray(depth)
            frame_out = cv2.resize(frame_out, (W, H), interpolation=cv2.INTER_CUBIC)
        elif args.style == "color":
            frame_out = depth_to_color(depth)
            frame_out = cv2.resize(frame_out, (W, H), interpolation=cv2.INTER_CUBIC)
        else:  # side
            gray = depth_to_gray(depth)
            gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_CUBIC)
            side = np.hstack([bgr, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
            frame_out = side
        writer.stdin.write(frame_out.astype(np.uint8).tobytes())

        i += 1
        if i % 5 == 0 or i == total:
            el = time.time() - t0
            eta = el / max(i, 1) * (total - i)
            log(f"推理进度 {i}/{total}  已写 {i} 帧  耗时 {el:.1f}s  预计剩余 {eta:.1f}s")

    cap.release()
    writer.stdin.close()
    rc = writer.wait()
    if rc != 0:
        log(f"[警告] ffmpeg 编码返回 {rc}")
    if not os.path.isfile(args.output) or os.path.getsize(args.output) < 1000:
        raise SystemExit(f"输出视频写入失败: {args.output}")
    log(f"完成：{args.output}（{i} 帧，总耗时 {time.time()-t0:.1f}s）")


def _read_frames(cap, idxs, W, H):
    """按索引读取帧（仅在窗口模式使用，逐帧重读成本低时可接受）。"""
    out = []
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for ix in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
        ok, bgr = cap.read()
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else np.zeros((H, W, 3), np.uint8))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return out


if __name__ == "__main__":
    main()
