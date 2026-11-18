"""EchoPlane3D FastAPI backend.

Run with:  uvicorn backend.main:app --host 127.0.0.1 --port 8000

EchoPlane3D is an educational and research simulation prototype.  It is not a
medical device and none of its outputs are clinical measurements.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .core import config, guide, sam2_tracker, uploads
from .core.simpsons_math import (
    biplane_volume_from_disks,
    foreshorten_volume,
    foreshortening_sweep,
    simulate_apical_foreshortening,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("echoplane.api")

config.ensure_dirs()
app = FastAPI(
    title="EchoPlane3D",
    version="1.1.0",
    description="Offline 2D/3D echocardiography training workbench (educational / research prototype, "
                "not a medical device).",
)

DISCLAIMER = "Educational and research simulation prototype — not a medical device; not for diagnosis."


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
def _load_samples() -> dict[str, dict]:
    """Scan the bundled samples and the uploads folder.  Only IDs found here
    are ever accepted from clients, so request strings never touch the
    filesystem directly."""
    samples: dict[str, dict] = {}
    for folder in (config.SAMPLES_DIR, config.UPLOADS_DIR):
        for side in sorted(folder.glob("*.json")):
            if side.name == "manifest.json":
                continue
            try:
                meta = json.loads(side.read_text())
            except json.JSONDecodeError:
                log.warning("skipping unreadable sample metadata %s", side)
                continue
            video = folder / str(meta.get("video", ""))
            if "id" not in meta or not video.is_file():
                continue
            meta.setdefault("view", "A4C")
            meta.setdefault("subject_id", meta["id"])
            meta.setdefault("subject_name", meta.get("name", meta["id"]))
            meta.setdefault("calibrated", True)
            meta.setdefault("volume_unit", "mL" if meta["calibrated"] else "index")
            meta["_video_path"] = video
            samples[meta["id"]] = meta
    # phantoms first, then real data, then uploads; healthier ventricle first, A4C before A2C
    kind_order = {"phantom": 0, "camus": 1, "upload": 2}
    view_order = {"A4C": 0, "A2C": 1}

    def sort_key(kv):
        meta = kv[1]
        gt = meta.get("ground_truth") or {}
        ef = gt.get("camus_biplane_ef_pct") or gt.get("lvef_pct") or 0.0
        return (kind_order.get(meta.get("kind"), 9), -float(ef), meta["subject_id"], view_order.get(meta["view"], 9))

    return dict(sorted(samples.items(), key=sort_key))


def _get_sample(sample_id: str) -> dict:
    samples = _load_samples()
    if sample_id not in samples:
        raise HTTPException(status_code=404, detail=f"unknown sample '{sample_id}'")
    return samples[sample_id]


def _public_sample(meta: dict) -> dict:
    out = {k: v for k, v in meta.items() if not k.startswith("_") and k not in ("gt_masks",)}
    sid = meta["id"]
    out["video_url"] = f"/api/samples/{sid}/video"
    out["frame_url_template"] = f"/api/samples/{sid}/frames/{{index}}.jpg"
    return out


def _subjects(samples: dict[str, dict]) -> list[dict]:
    """Group sequences by subject so the UI can pair A4C and A2C of one patient."""
    subjects: dict[str, dict] = {}
    for meta in samples.values():
        s = subjects.setdefault(meta["subject_id"], {
            "id": meta["subject_id"], "name": meta["subject_name"], "kind": meta.get("kind"),
            "ef_class": meta.get("ef_class"), "views": {},
        })
        s["views"][meta["view"]] = meta["id"]
    return list(subjects.values())


def _paired_sample(sample: dict, samples: dict[str, dict]) -> Optional[dict]:
    for other in samples.values():
        if other["subject_id"] == sample["subject_id"] and other["id"] != sample["id"] and other["view"] != sample["view"]:
            return other
    return None


@app.get("/api/health")
def health() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "status": "ok",
        "disclaimer": DISCLAIMER,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "sam2": config.validate_sam2_paths(),
        "model_loaded": sam2_tracker._predictor is not None,
        "vram": sam2_tracker.gpu_memory_mb(),
        "vram_budget_mb": config.VRAM_BUDGET_MB,
        "n_samples": len(_load_samples()),
        "heart_model": config.HEART_GLB_PATH.is_file(),
        "guide_engine": guide.engine_status(),
    }


@app.get("/api/samples")
def list_samples() -> dict:
    samples = _load_samples()
    public = [_public_sample(m) for m in samples.values()]
    return {"samples": public, "count": len(public), "subjects": _subjects(samples)}


@app.get("/api/samples/{sample_id}/video")
def sample_video(sample_id: str) -> FileResponse:
    return FileResponse(_get_sample(sample_id)["_video_path"], media_type="video/mp4")


@app.get("/api/samples/{sample_id}/frames/{index}.jpg")
def sample_frame(sample_id: str, index: int) -> FileResponse:
    meta = sam2_tracker.extract_frames(_get_sample(sample_id)["_video_path"])
    if not (0 <= index < meta["n_frames"]):
        raise HTTPException(status_code=404, detail=f"frame {index} out of range (0..{meta['n_frames'] - 1})")
    return FileResponse(
        Path(meta["frames_dir"]) / f"{index:05d}.jpg",
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
@app.post("/api/upload")
def upload_clip(
    file: UploadFile = File(..., description=".mp4 or .avi echo clip"),
    mm_per_pixel: Optional[float] = Form(default=None, description="optional calibration, mm per pixel"),
    view: str = Form(default="A4C"),
) -> dict:
    try:
        meta = uploads.ingest_upload(file.file, file.filename or "", mm_per_pixel, view)
    except uploads.UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    meta["_video_path"] = config.UPLOADS_DIR / meta["video"]
    return {"sample": _public_sample(meta)}


@app.delete("/api/upload/{sample_id}")
def delete_clip(sample_id: str) -> dict:
    if not uploads.delete_upload(sample_id):
        raise HTTPException(status_code=404, detail=f"no uploaded clip '{sample_id}'")
    return {"deleted": sample_id}


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
class SegmentRequest(BaseModel):
    sample_id: str
    click: Optional[list[float]] = Field(default=None, description="[x, y] on frame 0; omit for auto-centroid")
    points: Optional[list[list[float]]] = Field(default=None, description="multi-point prompt [[x, y], ...]")
    labels: Optional[list[int]] = Field(default=None, description="1 = cavity, 0 = not cavity, one per point")
    auto_snap: bool = Field(default=True, description="move a single click on bright wall into the adjacent blood pool")
    force: bool = Field(default=False, description="ignore the cache and re-run SAM 2.1")
    cached_only: bool = Field(default=False, description="return a cached result if present, never touch the GPU")

    @field_validator("click")
    @classmethod
    def _two_numbers(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("click must be [x, y]")
        return v


def _biplane(result: dict, sample: dict, samples: dict[str, dict]) -> Optional[dict]:
    """Biplane method of disks when the same subject's other apical view has
    also been segmented.  ED and ES are taken per view (the two clips are
    different beats)."""
    other_sample = _paired_sample(sample, samples)
    if other_sample is None:
        return None
    other = sam2_tracker.load_latest_result(Path(other_sample["_video_path"]).stem)
    if not other or "diameters_mm" not in other["frames_data"][0]:
        return None
    a4c, a2c = (result, other) if sample["view"] == "A4C" else (other, result)

    def vol(phase: str) -> tuple[float, float]:
        f4 = a4c["frames_data"][a4c[f"{phase}_frame_idx"]]
        f2 = a2c["frames_data"][a2c[f"{phase}_frame_idx"]]
        v = biplane_volume_from_disks(f4["diameters_mm"], f4["length_mm"], f2["diameters_mm"], f2["length_mm"])
        longest = max(f4["length_mm"], f2["length_mm"])
        mismatch = abs(f4["length_mm"] - f2["length_mm"]) / longest * 100.0 if longest > 0 else 0.0
        return v, mismatch

    edv, mis_ed = vol("ed")
    esv, mis_es = vol("es")
    if edv <= 0:
        return None
    return {
        "method": "biplane method of disks, V = pi/4 * sum(a_i * b_i) * max(L_A4C, L_A2C) / 20",
        "edv_ml": round(edv, 2),
        "esv_ml": round(esv, 2),
        "lvef_pct": round((edv - esv) / edv * 100.0, 2),
        "paired_sample_id": other_sample["id"],
        "paired_view": other_sample["view"],
        "length_mismatch_pct_ed": round(mis_ed, 1),
        "length_mismatch_pct_es": round(mis_es, 1),
        "length_warning": mis_ed > 10.0 or mis_es > 10.0,
    }


def _enrich(result: dict, sample: dict, samples: dict[str, dict]) -> dict:
    """Attach reference values and biplane estimates, keeping estimates and
    references in separately named fields."""
    out = dict(result)
    gt = sample.get("ground_truth") or {}
    out["available"] = True
    out["view"] = sample["view"]
    out["subject_id"] = sample["subject_id"]
    out["ground_truth"] = sample.get("ground_truth")
    # (a) algorithmic estimates
    out["single_plane_lvef"] = result["lvef_pct"]
    bp = _biplane(result, sample, samples)
    out["biplane"] = bp
    out["biplane_lvef"] = bp["lvef_pct"] if bp else None
    # (b) dataset reference annotations
    out["reference_single_plane_lvef"] = gt.get("lvef_pct")
    out["reference_biplane_lvef"] = gt.get("reference_biplane_lvef_pct")
    out["reference_source"] = gt.get("method")
    # (c) value reported with the dataset by its clinical readers (CAMUS only)
    out["dataset_reported_biplane_ef"] = gt.get("camus_biplane_ef_pct")
    out["disclaimer"] = DISCLAIMER
    return out


@app.post("/api/segment")
def segment(req: SegmentRequest) -> JSONResponse:
    samples = _load_samples()
    if req.sample_id not in samples:
        raise HTTPException(status_code=404, detail=f"unknown sample '{req.sample_id}'")
    sample = samples[req.sample_id]
    video = sample["_video_path"]
    try:
        if req.cached_only:
            pts, lbl, _ = sam2_tracker.normalise_prompt(req.click, req.points, req.labels)
            cache = sam2_tracker._cache_path(
                video, pts, lbl, req.auto_snap, float(sample.get("mm_per_pixel", config.DEFAULT_MM_PER_PIXEL)),
                config.N_SIMPSON_DISKS,
            )
            if not cache.is_file():
                return JSONResponse({"available": False, "sample_id": req.sample_id})
        result = sam2_tracker.segment_video(
            video, click=req.click, points=req.points, labels=req.labels, auto_snap=req.auto_snap, force=req.force
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        log.exception("segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(_enrich(result, sample, samples))


# --------------------------------------------------------------------------- #
# Foreshortening
# --------------------------------------------------------------------------- #
class ForeshorteningRequest(BaseModel):
    true_edv: float = Field(gt=0, description="true end-diastolic volume, mL")
    true_esv: float = Field(gt=0, description="true end-systolic volume, mL")
    tilt_deg: float = Field(ge=0, le=60, description="off-axis tilt of the apical plane, degrees")
    volumes_ml: Optional[list[float]] = Field(default=None, description="optional V(t) curve to foreshorten")


def _foreshortening(true_edv: float, true_esv: float, tilt_deg: float, volumes_ml: Optional[list[float]]) -> dict:
    try:
        result = simulate_apical_foreshortening(true_edv, true_esv, tilt_deg)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["sweep"] = foreshortening_sweep(true_edv, true_esv, 30.0, 1.0)
    if volumes_ml is not None:
        result["curve_ml"] = [round(foreshorten_volume(max(float(v), 0.0), true_edv, tilt_deg), 3) for v in volumes_ml]
    return result


@app.post("/api/foreshortening")
def foreshortening_post(req: ForeshorteningRequest) -> dict:
    return _foreshortening(req.true_edv, req.true_esv, req.tilt_deg, req.volumes_ml)


@app.get("/api/foreshortening")
def foreshortening_get(
    true_edv: float = Query(gt=0), true_esv: float = Query(gt=0), tilt_deg: float = Query(ge=0, le=60)
) -> dict:
    return _foreshortening(true_edv, true_esv, tilt_deg, None)


# --------------------------------------------------------------------------- #
# Clinical guide
# --------------------------------------------------------------------------- #
class QuizAnswer(BaseModel):
    id: str
    answer: int


class GuideRequest(BaseModel):
    message: str = Field(default="", max_length=500)
    preset: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)
    quiz_answer: Optional[QuizAnswer] = None


def _guide_samples() -> list[dict]:
    return [
        {"id": m["id"], "name": m.get("name", m["id"]), "kind": m.get("kind"), "view": m["view"],
         "subject_id": m["subject_id"], "ef_class": m.get("ef_class")}
        for m in _load_samples().values()
    ]


@app.get("/api/guide")
def guide_info() -> dict:
    return {
        "welcome": guide.WELCOME,
        "presets": guide.PRESETS,
        "views": {k: v["label"] for k, v in guide.VIEWS.items()},
        "engine": guide.engine_status(),
    }


@app.post("/api/guide")
def guide_turn(req: GuideRequest) -> dict:
    return guide.respond(
        message=req.message,
        preset=req.preset,
        context=req.context,
        quiz_answer=req.quiz_answer.model_dump() if req.quiz_answer else None,
        samples=_guide_samples(),
    )


# --------------------------------------------------------------------------- #
# Frontend (must be mounted last so /api/* wins)
# --------------------------------------------------------------------------- #
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
