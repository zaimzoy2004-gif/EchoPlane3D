"""Small video helpers shared by the sample scripts and the upload endpoint."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def write_mp4(frames_u8: np.ndarray, path: Path, fps: float) -> None:
    """Encode frames ``[T, H, W]`` (gray) or ``[T, H, W, 3]`` (BGR) uint8 as
    browser-playable H.264 / yuv420p."""
    frames_u8 = np.ascontiguousarray(frames_u8)
    if frames_u8.ndim == 3:
        pix_fmt = "gray"
        t, h, w = frames_u8.shape
    elif frames_u8.ndim == 4 and frames_u8.shape[3] == 3:
        pix_fmt = "bgr24"
        t, h, w, _ = frames_u8.shape
    else:
        raise ValueError("frames must be [T, H, W] or [T, H, W, 3]")
    if h % 2 or w % 2:
        raise ValueError("H.264 yuv420p needs even dimensions")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "12", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    proc = subprocess.run(cmd, input=frames_u8.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[-400:]}")


def read_video_gray(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.asarray(out)
