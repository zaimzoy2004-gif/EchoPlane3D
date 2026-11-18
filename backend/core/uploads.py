"""Ingest user-supplied echo clips for the workbench.

A clip is validated, temporally sub-sampled to at most
``config.MAX_UPLOAD_FRAMES`` frames, down-scaled if very large, re-encoded as
H.264 and stored in ``backend/data/uploads/`` with a metadata side-car in the
same format as the bundled samples.

Calibration
-----------
Without ``mm_per_pixel`` the clip is *uncalibrated*: volumes are reported as a
pixel-scale volume index (computed with 1 px = 1 unit), not millilitres.  LVEF
is a ratio of two volumes measured with the same scale, so it does not depend
on the calibration.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

import cv2
import numpy as np

from . import config
from .video_io import ffmpeg_available, write_mp4


class UploadError(ValueError):
    """Raised for anything wrong with the user's file (maps to HTTP 4xx)."""


def _safe_label(filename: str) -> str:
    stem = Path(filename or "clip").stem
    stem = re.sub(r"[^A-Za-z0-9 ._-]+", "", stem).strip()[:48]
    return stem or "clip"


def ingest_upload(stream: BinaryIO, filename: str, mm_per_pixel: Optional[float] = None, view: str = "A4C") -> dict:
    """Validate and store an uploaded clip; return its sample metadata."""
    ext = Path(filename or "").suffix.lower()
    if ext not in config.UPLOAD_EXTENSIONS:
        raise UploadError(f"unsupported file type '{ext or '?'}' — upload {' or '.join(config.UPLOAD_EXTENSIONS)}")
    if mm_per_pixel is not None and not (0.01 <= float(mm_per_pixel) <= 5.0):
        raise UploadError("mm_per_pixel must be between 0.01 and 5.0")
    if view not in ("A4C", "A2C"):
        raise UploadError("view must be A4C or A2C")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to store uploaded clips")
    config.ensure_dirs()

    limit = config.MAX_UPLOAD_MB * 2**20
    digest = hashlib.sha1()
    with tempfile.NamedTemporaryFile(suffix=ext, dir=config.UPLOADS_DIR, delete=True) as tmp:
        size = 0
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                raise UploadError(f"file is larger than {config.MAX_UPLOAD_MB} MB")
            digest.update(chunk)
            tmp.write(chunk)
        tmp.flush()
        if size == 0:
            raise UploadError("the uploaded file is empty")

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise UploadError("the file could not be decoded as a video")
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        while len(frames) < 4000:                      # hard stop for pathological files
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

    n_src = len(frames)
    if n_src < 2:
        raise UploadError("the clip must contain at least 2 decodable frames")
    if not (1.0 <= src_fps <= 240.0):
        src_fps = 25.0

    # temporal sub-sampling: keep the whole clip duration, at most MAX_UPLOAD_FRAMES frames
    keep = min(n_src, config.MAX_UPLOAD_FRAMES)
    idx = np.unique(np.round(np.linspace(0, n_src - 1, keep)).astype(int))
    frames = [frames[i] for i in idx]
    fps = src_fps * len(frames) / n_src

    # spatial down-scaling + even dimensions
    h, w = frames[0].shape[:2]
    scale = min(1.0, config.MAX_UPLOAD_SIDE_PX / max(h, w))
    nw, nh = int(round(w * scale)) // 2 * 2, int(round(h * scale)) // 2 * 2
    if nw < 32 or nh < 32:
        raise UploadError("frames are too small (need at least 32x32 px)")
    if (nw, nh) != (w, h):
        frames = [cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in frames]
    arr = np.asarray(frames, dtype=np.uint8)

    calibrated = mm_per_pixel is not None
    mmpp = float(mm_per_pixel) * (w / nw) if calibrated else 1.0     # rescale calibration with the image

    sid = f"upload_{digest.hexdigest()[:12]}"
    video_path = config.UPLOADS_DIR / f"{sid}.mp4"
    write_mp4(arr, video_path, fps)
    meta = {
        "id": sid,
        "name": f"Upload · {_safe_label(filename)}",
        "kind": "upload",
        "view": view,
        "subject_id": sid,
        "subject_name": f"Upload · {_safe_label(filename)}",
        "description": "User-uploaded clip. " + (
            "Calibrated with the mm/pixel value supplied at upload."
            if calibrated else
            "Uncalibrated: volumes are a pixel-scale volume index, not millilitres; LVEF is scale-invariant."),
        "video": video_path.name,
        "width": nw,
        "height": nh,
        "n_frames": int(arr.shape[0]),
        "fps": round(fps, 4),
        "mm_per_pixel": round(mmpp, 6),
        "calibrated": calibrated,
        "volume_unit": "mL" if calibrated else "index",
        "source_frames": n_src,
        "source_size": [w, h],
        "subsampled": n_src > arr.shape[0],
        "ground_truth": None,
    }
    video_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def delete_upload(sample_id: str) -> bool:
    """Remove an uploaded clip and its side-car (uploads only)."""
    if not re.fullmatch(r"upload_[0-9a-f]{12}", sample_id):
        return False
    removed = False
    for suffix in (".mp4", ".json"):
        f = config.UPLOADS_DIR / f"{sample_id}{suffix}"
        if f.is_file():
            f.unlink()
            removed = True
    return removed
