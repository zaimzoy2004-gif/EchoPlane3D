"""Access to the open CAMUS echocardiography dataset.

Files are fetched from the public (no-login) Girder collection of the CREATIS
human-heart project and cached under ``backend/data/camus_raw/`` so that every
later run — sample preparation and the paper benchmarks — works offline.

CAMUS: S. Leclerc et al., "Deep Learning for Segmentation using an Open
Large-Scale Dataset in 2D Echocardiography", IEEE TMI 38(9), 2019,
doi:10.1109/TMI.2019.2900516.  The dataset's own terms apply to the downloaded
files; they are not redistributed with this repository.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
NIFTI_FOLDER_ID = "63fde55f73e9f004868fb7ac"          # CAMUS_public / database_nifti
USER_AGENT = "echoplane3d-sample-fetch/1.1"
VIEWS = {"A4C": "4CH", "A2C": "2CH"}
CITATION = (
    'S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al. "Deep Learning for Segmentation using an Open '
    'Large-Scale Dataset in 2D Echocardiography", IEEE Trans. Med. Imaging 38(9):2198-2210, 2019. '
    "doi:10.1109/TMI.2019.2900516"
)
SOURCE = "CAMUS_public, https://humanheart-project.creatis.insa-lyon.fr/database/ (public Girder collection)"
LABEL_LV, LABEL_MYO, LABEL_LA = 1, 2, 3


def _http(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _index_path() -> Path:
    return config.CAMUS_RAW_DIR / "_folders.json"


def patient_folders(refresh: bool = False) -> dict[str, str]:
    """``{patient name: Girder folder id}`` for the whole collection (cached)."""
    config.CAMUS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    idx = _index_path()
    if idx.is_file() and not refresh:
        return json.loads(idx.read_text())
    folders = json.loads(_http(f"{API}/folder?parentType=folder&parentId={NIFTI_FOLDER_ID}&limit=1000&sort=name"))
    mapping = {f["name"]: f["_id"] for f in folders}
    idx.write_text(json.dumps(mapping))
    return mapping


def _items(patient: str) -> dict[str, str]:
    pdir = config.CAMUS_RAW_DIR / patient
    pdir.mkdir(parents=True, exist_ok=True)
    cache = pdir / "_items.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    folders = patient_folders()
    if patient not in folders:
        raise KeyError(f"{patient} not found in CAMUS_public")
    items = {i["name"]: i["_id"] for i in json.loads(_http(f"{API}/item?folderId={folders[patient]}&limit=100"))}
    cache.write_text(json.dumps(items))
    return items


def fetch(patient: str, filename: str) -> Path:
    """Local path of one CAMUS file, downloading it on first use."""
    path = config.CAMUS_RAW_DIR / patient / filename
    if path.is_file() and path.stat().st_size > 0:
        return path
    items = _items(patient)
    if filename not in items:
        raise FileNotFoundError(f"{filename} is not part of {patient}")
    data = _http(f"{API}/item/{items[filename]}/download", timeout=180)
    path.write_bytes(data)
    return path


def read_info(patient: str, view: str) -> dict:
    """Parse ``Info_4CH.cfg`` / ``Info_2CH.cfg`` (ED, ES, NbFrame, ImageQuality, EF, FrameRate ...)."""
    info = {}
    for line in fetch(patient, f"Info_{VIEWS[view]}.cfg").read_text().strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def _read_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return arr, (float(img.GetSpacing()[0]), float(img.GetSpacing()[1]))


def _isotropic(frames: np.ndarray, masks: Optional[np.ndarray], sx: float, sy: float):
    """Resample to square pixels (CAMUS files are sometimes anisotropic)."""
    import cv2

    mmpp = min(sx, sy)
    if abs(sx - sy) > 1e-4:
        _, h, w = frames.shape
        nw, nh = int(round(w * sx / mmpp)), int(round(h * sy / mmpp))
        frames = np.asarray([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR) for f in frames])
        if masks is not None:
            masks = np.asarray([cv2.resize(m.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) for m in masks])
    return frames, masks, mmpp


def load_sequence(patient: str, view: str) -> dict:
    """Half-cycle sequence with per-frame reference labels.

    Returns ``frames`` uint8 [T, H, W], ``labels`` uint8 [T, H, W] (1 LV cavity,
    2 myocardium, 3 LA), isotropic ``mm_per_pixel`` and the parsed ``info``."""
    tag = VIEWS[view]
    seq, (sx, sy) = _read_nifti(fetch(patient, f"{patient}_{tag}_half_sequence.nii.gz"))
    gt, _ = _read_nifti(fetch(patient, f"{patient}_{tag}_half_sequence_gt.nii.gz"))
    if seq.ndim != 3 or gt.shape != seq.shape:
        raise RuntimeError(f"unexpected CAMUS array shapes {seq.shape} / {gt.shape}")
    frames = np.clip(seq, 0, 255).astype(np.uint8)
    labels = np.rint(gt).astype(np.uint8)
    frames, labels, mmpp = _isotropic(frames, labels, sx, sy)
    return {"patient": patient, "view": view, "frames": frames, "labels": labels, "mm_per_pixel": mmpp,
            "info": read_info(patient, view)}


def load_ed_es(patient: str, view: str) -> dict:
    """The two expert-annotated key frames of a view: ``{"ED": (image, labels), "ES": (...)}``."""
    tag = VIEWS[view]
    out = {"patient": patient, "view": view, "info": read_info(patient, view)}
    for phase in ("ED", "ES"):
        img, (sx, sy) = _read_nifti(fetch(patient, f"{patient}_{tag}_{phase}.nii.gz"))
        gt, _ = _read_nifti(fetch(patient, f"{patient}_{tag}_{phase}_gt.nii.gz"))
        if img.ndim == 3:
            img, gt = img[0], gt[0]
        frames, labels, mmpp = _isotropic(np.clip(img, 0, 255).astype(np.uint8)[None], np.rint(gt).astype(np.uint8)[None], sx, sy)
        out[phase] = (frames[0], labels[0])
        out["mm_per_pixel"] = mmpp
    return out


def ef_class(ef_pct: Optional[float]) -> str:
    """Heart-failure style bands used to stratify the demo cohort."""
    if ef_pct is None:
        return "unknown"
    if ef_pct >= 55:
        return "normal (>=55 %)"
    if ef_pct >= 50:
        return "low-normal (50-54 %)"
    if ef_pct >= 41:
        return "mildly reduced (41-49 %)"
    return "reduced (<=40 %)"
