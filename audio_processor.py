"""
视频音频处理引擎
支持三种模式：只保留人声、只保留环境音、无声音
人声/环境音分离基于 Demucs v4 (htdemucs)，未安装时仅静音模式可用
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

FFMPEG = "/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else "ffmpeg"

# 任务状态存储
_tasks = {}
_tasks_lock = threading.Lock()


def check_demucs() -> dict:
    """检查 Demucs 是否可用"""
    try:
        result = subprocess.run(
            ["python3", "-c", "import demucs; print(demucs.__version__)"],
            capture_output=True, text=True, timeout=10
        )
        available = result.returncode == 0
        version = result.stdout.strip() if available else None
        return {"demucs_available": available, "version": version}
    except Exception:
        return {"demucs_available": False, "version": None}


def _run_ffmpeg(cmd: list, desc: str = "") -> tuple[bool, str]:
    """运行 ffmpeg 命令"""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return False, proc.stderr[-500:] if proc.stderr else f"返回码 {proc.returncode}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "处理超时"
    except Exception as e:
        return False, str(e)


def extract_audio(video_path: str, out_path: str, fmt: str = "wav") -> bool:
    """从视频提取音轨"""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", video_path, "-vn", "-acodec", "pcm_s16le" if fmt == "wav" else "libmp3lame", out_path]
    ok, err = _run_ffmpeg(cmd, "提取音轨")
    return ok


def separate_with_demucs(audio_path: str, out_dir: str, model: str = "htdemucs") -> dict | None:
    """用 Demucs 分离人声和伴奏，返回 {vocals: path, no_vocals: path}"""
    try:
        cmd = [
            "python3", "-m", "demucs",
            "-n", model,
            "-o", out_dir,
            "--two-stems", "vocals",
            audio_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            return None

        # Demucs 输出结构: out_dir/model/audio_name/vocals.wav, no_vocals.wav
        audio_stem = Path(audio_path).stem
        result_dir = Path(out_dir) / model / audio_stem
        vocals = result_dir / "vocals.wav"
        no_vocals = result_dir / "no_vocals.wav"

        result = {}
        if vocals.exists():
            result["vocals"] = str(vocals)
        if no_vocals.exists():
            result["no_vocals"] = str(no_vocals)
        return result if result else None
    except Exception:
        return None


def replace_audio(video_path: str, audio_path: str, out_path: str) -> bool:
    """替换视频音轨"""
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", out_path
    ]
    ok, err = _run_ffmpeg(cmd, "替换音轨")
    return ok


def remove_audio(video_path: str, out_path: str) -> bool:
    """去除视频音轨"""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", video_path, "-c", "copy", "-an", out_path]
    ok, err = _run_ffmpeg(cmd, "去除音轨")
    return ok


def process_video(task_id: str, video_path: str, mode: str, out_dir: str, model: str = "htdemucs"):
    """
    处理视频音轨
    mode: vocals_only | ambient_only | mute
    """
    global _tasks

    def _update(status, progress=0, message="", result=None, error=None):
        with _tasks_lock:
            _tasks[task_id].update({
                "status": status, "progress": progress,
                "message": message, "result": result, "error": error,
                "updated_at": time.time()
            })

    try:
        os.makedirs(out_dir, exist_ok=True)
        video_name = Path(video_path).stem
        out_path = os.path.join(out_dir, f"{video_name}_{mode}.mp4")

        if mode == "mute":
            _update("running", 0.2, "去除音轨中...")
            ok = remove_audio(video_path, out_path)
            if not ok:
                _update("error", error="去除音轨失败")
                return
            _update("done", 1.0, "完成", {"output_path": out_path, "mode": mode})
            return

        # 人声/环境音模式需要 Demucs
        demucs_info = check_demucs()
        if not demucs_info["demucs_available"]:
            _update("error", error="Demucs 未安装，无法分离人声/环境音。请运行: pip install demucs")
            return

        _update("running", 0.1, "提取音轨中...")
        audio_path = os.path.join(out_dir, f"{video_name}_audio.wav")
        if not extract_audio(video_path, audio_path):
            _update("error", error="提取音轨失败")
            return

        _update("running", 0.3, "分离人声中（首次运行会下载模型，约80MB）...")
        sep_result = separate_with_demucs(audio_path, out_dir, model)
        if not sep_result:
            _update("error", error="人声分离失败")
            return

        _update("running", 0.8, "合成视频中...")
        if mode == "vocals_only":
            audio_src = sep_result.get("vocals")
        else:  # ambient_only
            audio_src = sep_result.get("no_vocals")

        if not audio_src or not os.path.exists(audio_src):
            _update("error", error=f"未找到{'人声' if mode == 'vocals_only' else '环境音'}音轨")
            return

        ok = replace_audio(video_path, audio_src, out_path)
        if not ok:
            _update("error", error="合成视频失败")
            return

        # 清理临时文件
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        _update("done", 1.0, "完成", {
            "output_path": out_path, "mode": mode,
            "vocals_path": sep_result.get("vocals"),
            "ambient_path": sep_result.get("no_vocals")
        })

    except Exception as e:
        _update("error", error=str(e))


def create_task(video_path: str, mode: str, out_dir: str, model: str = "htdemucs") -> str:
    """创建处理任务"""
    global _tasks
    task_id = f"audio_{int(time.time() * 1000)}"
    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id, "video_path": video_path,
            "mode": mode, "status": "pending", "progress": 0,
            "message": "等待中", "result": None, "error": None,
            "created_at": time.time(), "updated_at": time.time()
        }
    t = threading.Thread(target=process_video, args=(task_id, video_path, mode, out_dir, model), daemon=True)
    t.start()
    return task_id


def get_task(task_id: str) -> dict | None:
    """获取任务状态"""
    with _tasks_lock:
        return _tasks.get(task_id)
