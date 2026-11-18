"""SAM 2.1 (hiera-tiny) video tracker for the left-ventricular cavity.

``segment_video`` prompts SAM 2.1 on frame 0 — automatic LV centroid, one
click (optionally snapped from bright wall into the adjacent blood pool), or
several positive / negative points — resolves SAM's part/whole ambiguity to
the blood-pool mask, propagates it through the clip, runs 20-disk Simpson
volumetry on every frame and returns EDV / ESV / LVEF plus everything the UI
needs to draw overlays.

VRAM discipline (6 GB laptop GPU, 2.5 GB budget)
* hiera-tiny weights, autocast to bfloat16 (falls back to float16),
* decoded frames stay in system RAM (``offload_video_to_cpu``),
* ``predictor.reset_state`` + ``torch.cuda.empty_cache()`` after every run,
* a process-wide lock so two requests can never share the GPU.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from . import config
from .simpsons_math import compute_simpsons_20_disks

log = logging.getLogger("echoplane.sam2")

TRACKER_VERSION = 6          # bump to invalidate cached results
OBJ_ID = 1

_predictor = None
_predictor_lock = threading.Lock()
_gpu_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def get_predictor():
    """Build the SAM 2.1 video predictor once and keep it on the GPU."""
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            import torch
            from sam2.build_sam import build_sam2_video_predictor

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available — EchoPlane3D needs the GPU for SAM 2.1")
            if not config.SAM2_CHECKPOINT.is_file():
                raise FileNotFoundError(f"SAM2 checkpoint not found: {config.SAM2_CHECKPOINT}")
            t0 = time.perf_counter()
            # Same post-processing overrides SAM 2 applies by default, except
            # GPU hole filling (fill_hole_area): it needs SAM 2's optional
            # compiled extension, and clean_mask() fills holes on the CPU anyway.
            _predictor = build_sam2_video_predictor(
                config.SAM2_CONFIG,
                str(config.SAM2_CHECKPOINT),
                device=config.DEVICE,
                apply_postprocessing=False,
                hydra_overrides_extra=[
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                    "++model.binarize_mask_from_pts_for_mem_enc=true",
                    "++model.fill_hole_area=0",
                ],
            )
            log.info("SAM 2.1 hiera-tiny loaded in %.1f s", time.perf_counter() - t0)
        return _predictor


def unload_predictor() -> None:
    """Drop the model and release its VRAM (used by tests / shutdown)."""
    global _predictor
    with _predictor_lock:
        _predictor = None
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def gpu_memory_mb() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 2**20,
        "reserved_mb": torch.cuda.memory_reserved() / 2**20,
    }


# --------------------------------------------------------------------------- #
# Video helpers
# --------------------------------------------------------------------------- #
def _file_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()[:12]


def extract_frames(video_path: str | Path) -> dict:
    """Decode a video into ``<cache>/frames/<stem>_<key>/00000.jpg ...`` (the
    JPEG-folder layout SAM 2 reads natively).  Cached across runs."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    out_dir = config.FRAMES_CACHE_DIR / f"{video_path.stem}_{_file_key(video_path)}"
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if len(list(out_dir.glob("*.jpg"))) == meta["n_frames"]:
            meta["frames_dir"] = str(out_dir)
            return meta

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = 0
    w = h = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n >= config.MAX_VIDEO_FRAMES:
            cap.release()
            raise ValueError(f"video has more than {config.MAX_VIDEO_FRAMES} frames")
        h, w = frame.shape[:2]
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    cap.release()
    if n == 0:
        raise RuntimeError(f"no frames decoded from {video_path}")
    meta = {"n_frames": n, "width": w, "height": h, "fps": fps}
    meta_path.write_text(json.dumps(meta))
    meta["frames_dir"] = str(out_dir)
    return meta


def load_calibration(video_path: str | Path) -> dict:
    """Read the ``<video>.json`` side-car (mm_per_pixel, fps, ground truth)."""
    side = Path(video_path).with_suffix(".json")
    if side.is_file():
        try:
            return json.loads(side.read_text())
        except json.JSONDecodeError:
            log.warning("unreadable calibration side-car %s", side)
    return {}


# --------------------------------------------------------------------------- #
# Auto-centroid prompt
# --------------------------------------------------------------------------- #
def auto_centroid(gray: np.ndarray) -> tuple[float, float]:
    """Locate a click point inside the LV cavity of an apical view.

    The LV blood pool is the largest dark region that is fully enclosed by
    brighter tissue and sits in the upper-central part of the sector.  Returns
    ``(x, y)`` in full-resolution pixel coordinates.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 256.0 / max(h, w)
    small = cv2.resize(gray, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    blur = cv2.GaussianBlur(small, (0, 0), 2.5)

    # ultrasound sector = convex hull of everything that is not pure black
    lit = (cv2.GaussianBlur(small, (0, 0), 4.0) > 6).astype(np.uint8)
    sector = np.zeros_like(lit)
    pts = cv2.findNonZero(lit)
    if pts is None:
        return w / 2.0, h / 2.0
    cv2.fillConvexPoly(sector, cv2.convexHull(pts), 1)
    inner = cv2.erode(sector, np.ones((9, 9), np.uint8))
    if inner.sum() < 50:
        return w / 2.0, h / 2.0

    thr, _ = cv2.threshold(blur[inner > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = ((blur < thr) & (sector > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    yy, xx = np.mgrid[0:sh, 0:sw]
    sx_min, sx_max = np.nonzero(sector.any(axis=0))[0][[0, -1]]
    sy_min, sy_max = np.nonzero(sector.any(axis=1))[0][[0, -1]]
    cx_sector = 0.5 * (sx_min + sx_max)
    span_x = max(sx_max - sx_min, 1)
    span_y = max(sy_max - sy_min, 1)

    def prior(x, y):
        horiz = np.exp(-(((x - cx_sector) / (0.30 * span_x)) ** 2))
        vert = np.exp(-(((y - (sy_min + 0.36 * span_y)) / (0.26 * span_y)) ** 2))   # LV sits above the atria
        return horiz * vert

    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=4)
    border = (sector > 0) & (inner == 0)
    best, best_score = None, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.012 * sector.sum():
            continue
        comp = labels == i
        touching = (comp & border).sum() / max(area, 1)
        if touching > 0.02:
            continue
        score = area * prior(cents[i][0], cents[i][1])
        if score > best_score:
            best, best_score = comp, score

    def deepest_dark_point():
        # deepest dark point near the expected LV position (works when wall drop-out leaves the LV unenclosed)
        dt_all = cv2.distanceTransform((dark & inner).astype(np.uint8), cv2.DIST_L2, 5) * prior(xx, yy)
        yb, xb = np.unravel_index(int(np.argmax(dt_all)), dt_all.shape)
        return float(xb), float(yb)

    if best is None:
        x, y = deepest_dark_point()
        return float((x + 0.5) / scale), float((y + 0.5) / scale)

    # The best *enclosed* pool can be the left atrium when the LV wall drops out and the LV is not
    # enclosed. If that pool sits low in the sector and there is a deep dark region clearly above
    # it, where the LV is expected, prompt there instead.
    ys_b, xs_b = np.nonzero(best)
    if prior(xs_b.mean(), ys_b.mean()) < 0.40:
        x, y = deepest_dark_point()
        if y < ys_b.mean() - 0.15 * span_y:
            return float((x + 0.5) / scale), float((y + 0.5) / scale)

    dt = cv2.distanceTransform(best.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(best)
    cx, cy = xs.mean(), ys.mean()
    ix, iy = int(round(cx)), int(round(cy))
    if not (0 <= iy < sh and 0 <= ix < sw and best[iy, ix] and dt[iy, ix] >= 0.5 * dt.max()):
        iy, ix = np.unravel_index(int(np.argmax(dt)), dt.shape)
        cx, cy = float(ix), float(iy)
    return float((cx + 0.5) / scale), float((cy + 0.5) / scale)


def _sector_mask(gray: np.ndarray) -> np.ndarray:
    """Convex hull of the insonified region (everything that is not pure black)."""
    lit = (cv2.GaussianBlur(gray, (0, 0), 6.0) > 6).astype(np.uint8)
    sector = np.zeros_like(lit)
    pts = cv2.findNonZero(lit)
    if pts is None:
        return np.ones_like(lit, dtype=bool)
    cv2.fillConvexPoly(sector, cv2.convexHull(pts), 1)
    return sector.astype(bool)


def analyse_frame(gray: np.ndarray) -> dict:
    """Shared image analysis: sector, smoothed image, Otsu tissue/blood
    threshold and the map of dark (blood-pool-like) pixels."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    sector = _sector_mask(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    if sector.sum() < 16:
        return {"sector": sector, "inner": sector, "blur": blur, "thr": 0.0, "dark": np.zeros_like(sector)}
    thr, _ = cv2.threshold(blur[sector].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_open = max(5, int(round(0.012 * max(gray.shape))) | 1)
    dark = ((blur < thr) & sector).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    k_in = max(9, int(round(0.03 * max(gray.shape))) | 1)
    inner = cv2.erode(sector.astype(np.uint8), np.ones((k_in, k_in), np.uint8)).astype(bool)
    return {"sector": sector, "inner": inner, "blur": blur, "thr": float(thr), "dark": dark.astype(bool)}


# --------------------------------------------------------------------------- #
# Sector-constrained cavity snapping
# --------------------------------------------------------------------------- #
def snap_to_cavity(gray: np.ndarray, xy: Sequence[float], radius_px: Optional[float] = None,
                   analysis: Optional[dict] = None) -> dict:
    """Move a click that landed on echogenic tissue into the adjacent blood pool.

    A click on the bright myocardium makes SAM segment the horseshoe-shaped
    wall.  If the clicked pixel is not inside a dark, sizeable region of the
    sector, the nearest such region within ``radius_px`` (default 34 px on a
    512-px frame) is located — failing that, by marching from the click toward
    the sector centre line — and the prompt is moved to a well-interior point
    of that region close to the original click.

    Returns ``{"snapped", "point", "distance_px", "reason"}``; ``point`` is the
    original click when no snap was needed or possible.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(h, w) / 512.0
    radius = float(radius_px) if radius_px else 34.0 * scale
    a = analysis or analyse_frame(gray)
    x0, y0 = float(xy[0]), float(xy[1])
    x, y = int(round(x0)), int(round(y0))
    info = {"snapped": False, "point": [x0, y0], "distance_px": 0.0, "reason": "", "radius_px": round(radius, 1)}
    if not (0 <= x < w and 0 <= y < h) or not a["sector"][y, x]:
        info["reason"] = "click is outside the ultrasound sector"
        return info

    dark = a["dark"].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    min_area = 0.008 * float(a["sector"].sum())
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    lab = int(labels[y, x])
    if lab in big and cv2.distanceTransform(dark, cv2.DIST_L2, 5)[y, x] >= 2.0 * scale:
        info["reason"] = "click is already inside a blood pool"
        return info
    if not big:
        info["reason"] = "no blood pool found in the sector"
        return info

    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - x0) ** 2 + (yy - y0) ** 2
    dist = {i: float(np.sqrt(d2[labels == i].min())) for i in big}
    near = {i: d for i, d in dist.items() if d <= radius}
    how = "nearest blood pool within the search radius"
    if not near:
        # radial march along the scan line toward the sector centre line
        cols = np.nonzero(a["sector"][y])[0]
        cx = 0.5 * (cols[0] + cols[-1])
        step = 1 if cx >= x else -1
        for xi in range(x, int(round(cx)) + step, step):
            li = int(labels[y, xi])
            if li in big:
                near = {li: float(abs(xi - x))}
                how = "first blood pool toward the sector centre line"
                break
        if not near or min(near.values()) > 3.0 * radius:
            info["reason"] = "no blood pool near the click"
            return info

    d_min = min(near.values())
    tied = [i for i, d in near.items() if d <= d_min + 6.0 * scale]      # e.g. a click on the septum
    comp_id = max(tied, key=lambda i: stats[i, cv2.CC_STAT_AREA])        # prefer the larger cavity (LV)
    comp = labels == comp_id
    depth = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
    reach = comp & (d2 <= (near[comp_id] + 0.9 * radius) ** 2)
    ys, xs = np.nonzero(reach)
    j = int(np.argmax(depth[reach]))
    px, py = float(xs[j]), float(ys[j])
    info.update(snapped=True, point=[px, py], distance_px=round(float(np.hypot(px - x0, py - y0)), 1),
                reason=f"moved from echogenic tissue to the {how}")
    return info


# --------------------------------------------------------------------------- #
# Candidate-mask selection
# --------------------------------------------------------------------------- #
def choose_cavity_mask(gray: np.ndarray, candidates: np.ndarray, scores: Sequence[float],
                       pos_points: Sequence[Sequence[float]], neg_points: Sequence[Sequence[float]] = (),
                       analysis: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Resolve SAM's part / whole ambiguity to the blood-pool mask.

    On ultrasound the top-scoring candidate is frequently the whole sector, and
    a click near the wall can return the horseshoe-shaped myocardium.  A blood
    pool is dark, darker than the rim around it, compact, and lies inside the
    sector.  Each candidate that honours the prompt (contains every positive
    point, excludes every negative one) and covers at most half of the sector
    is scored by

        (rim - interior contrast) * (1 - bright fraction) * solidity * containment

    and the best one is kept.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    a = analysis or analyse_frame(gray)
    sector, inner, blur, thr = a["sector"], a["inner"], a["blur"], a["thr"]
    sector_area = max(int(sector.sum()), 1)
    k = max(9, int(round(0.03 * max(gray.shape))) | 1)
    kernel = np.ones((k, k), np.uint8)
    h, w = gray.shape

    def inside(m, p):
        x, y = int(round(p[0])), int(round(p[1]))
        return 0 <= y < h and 0 <= x < w and bool(m[y, x])

    rows, cleaned = [], []
    first = pos_points[0] if len(pos_points) else None
    for i, cand in enumerate(candidates):
        m = clean_mask(np.asarray(cand) > 0, prefer_xy=first)
        cleaned.append(m)
        area = int(m.sum())
        row = {"index": i, "sam_score": round(float(scores[i]), 4), "area_frac": round(area / sector_area, 4),
               "contrast": None, "bright_frac": None, "solidity": None, "containment": None, "score": None,
               "honours_prompt": False, "eligible": False}
        if area > 0:
            row["honours_prompt"] = all(inside(m, p) for p in pos_points) and not any(inside(m, p) for p in neg_points)
            ring = cv2.dilate(m.astype(np.uint8), kernel).astype(bool) & ~m & sector
            interior = float(blur[m].mean())
            rim = float(blur[ring].mean()) if ring.any() else 0.0
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = float(cv2.contourArea(cv2.convexHull(np.vstack(cnts)))) if cnts else 0.0
            solidity = min(1.0, area / hull_area) if hull_area > 0 else 0.0
            containment = float((m & inner).sum()) / area
            bright = float((blur[m] >= thr).mean())
            row.update(contrast=round(rim - interior, 2), bright_frac=round(bright, 4), solidity=round(solidity, 4),
                       containment=round(containment, 4))
            row["eligible"] = bool(row["honours_prompt"] and area / sector_area <= 0.5 and rim - interior > 0)
            if row["eligible"]:
                row["score"] = round((rim - interior) * (1.0 - bright) * solidity * containment, 3)
        rows.append(row)

    eligible = [r for r in rows if r["eligible"]]
    if eligible:
        best = max(eligible, key=lambda r: r["score"])
        rule = "dark-lumen contrast x solidity x sector containment"
    else:
        honour = [r for r in rows if r["honours_prompt"] and 0 < r["area_frac"] <= 0.5]
        small = [r for r in rows if 0 < r["area_frac"] <= 0.5]
        best = max(honour or small or rows, key=lambda r: r["sam_score"])
        rule = "fallback: top SAM score"
    return cleaned[best["index"]], {"chosen": best["index"], "rule": rule, "candidates": rows}


# --------------------------------------------------------------------------- #
# Mask post-processing
# --------------------------------------------------------------------------- #
def clean_mask(mask: np.ndarray, prefer_xy: Optional[Sequence[float]] = None) -> np.ndarray:
    """Keep one connected component (the one under ``prefer_xy`` if given,
    else the largest) and fill its holes."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    keep = None
    if prefer_xy is not None:
        x, y = int(round(prefer_xy[0])), int(round(prefer_xy[1]))
        if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and labels[y, x] > 0:
            keep = labels[y, x]
    if keep is None:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == keep).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def mask_contour(mask: np.ndarray, epsilon: float = 0.6) -> list[list[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon, True)
    return c.reshape(-1, 2).astype(int).tolist()


def mask_solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    hull = float(cv2.contourArea(cv2.convexHull(np.vstack(cnts))))
    return float(min(1.0, mask.sum() / hull)) if hull > 0 else 0.0


# --------------------------------------------------------------------------- #
# Prompt handling
# --------------------------------------------------------------------------- #
def normalise_prompt(click=None, points=None, labels=None) -> tuple[Optional[list], Optional[list], str]:
    """Validate the three ways of prompting and return ``(points, labels, mode)``
    with mode in ``auto`` (no prompt), ``single`` (one positive click) or
    ``multi`` (several positive / negative points)."""
    if points is not None and len(points) > 0:
        pts = []
        for p in points:
            if len(p) != 2 or not all(np.isfinite(float(c)) for c in p):
                raise ValueError("every point must be [x, y]")
            pts.append([float(p[0]), float(p[1])])
        lbl = [1] * len(pts) if labels is None else [int(v) for v in labels]
        if len(lbl) != len(pts):
            raise ValueError("labels must have one entry per point")
        if any(v not in (0, 1) for v in lbl):
            raise ValueError("labels must be 1 (cavity) or 0 (not cavity)")
        if 1 not in lbl:
            raise ValueError("at least one positive (label 1) point is required")
        if len(pts) > 24:
            raise ValueError("at most 24 prompt points are supported")
        return pts, lbl, ("single" if len(pts) == 1 else "multi")
    if click is not None:
        if len(click) != 2 or not all(np.isfinite(float(c)) for c in click):
            raise ValueError("click must be [x, y]")
        return [[float(click[0]), float(click[1])]], [1], "single"
    return None, None, "auto"


def predict_frame_mask(predictor, frame_bgr: np.ndarray, points: Sequence[Sequence[float]], labels: Sequence[int],
                       select: str = "cavity", dtype_name: Optional[str] = None,
                       image_predictor=None) -> tuple[np.ndarray, dict]:
    """Segment one frame with the SAM 2.1 image head (shares the video model's
    weights).  ``select='cavity'`` applies :func:`choose_cavity_mask`;
    ``select='naive'`` returns what stock SAM gives (its top-scoring mask).

    Pass an ``image_predictor`` whose image is already set to ``frame_bgr`` to
    try several prompts on one frame without recomputing the embedding (used
    by the benchmark); the caller then owns ``reset_predictor()``."""
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name or config.AUTOCAST_DTYPE]
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    lbl = np.asarray(labels, dtype=np.int32).reshape(-1)
    owned = image_predictor is None
    if owned:
        image_predictor = SAM2ImagePredictor(predictor)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            if owned:
                image_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            multimask = select == "cavity" or len(pts) == 1
            cands, scores, _ = image_predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=multimask)
    finally:
        if owned:
            image_predictor.reset_predictor()
    pos = [p for p, l in zip(pts.tolist(), lbl.tolist()) if l == 1]
    neg = [p for p, l in zip(pts.tolist(), lbl.tolist()) if l == 0]
    if select == "naive":
        i = int(np.argmax(scores))
        mask = clean_mask(np.asarray(cands[i]) > 0, prefer_xy=pos[0] if pos else None)
        return mask, {"chosen": i, "rule": "stock SAM: highest predicted IoU",
                      "candidates": [{"index": j, "sam_score": round(float(s), 4)} for j, s in enumerate(scores)]}
    return choose_cavity_mask(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cands, scores, pos, neg)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _propagate(predictor, frames_dir: str, points, labels, dtype_name: str) -> tuple[dict[int, np.ndarray], dict]:
    """Frame-0 prompt -> SAM 2.1 candidate masks -> cavity-scale mask -> video propagation."""
    import torch

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    masks: dict[int, np.ndarray] = {}
    state = None
    try:
        frame0 = cv2.imread(str(Path(frames_dir) / "00000.jpg"))
        seed, prompt_info = predict_frame_mask(predictor, frame0, points, labels, select="cavity", dtype_name=dtype_name)
        if not seed.any():
            raise RuntimeError("SAM 2.1 found no object under the prompt — click inside the LV cavity")
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            state = predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=config.OFFLOAD_VIDEO_TO_CPU,
                offload_state_to_cpu=config.OFFLOAD_STATE_TO_CPU,
            )
            predictor.add_new_mask(inference_state=state, frame_idx=0, obj_id=OBJ_ID, mask=seed)
            for frame_idx, _obj_ids, logits in predictor.propagate_in_video(state):
                masks[int(frame_idx)] = (logits[0, 0] > 0.0).cpu().numpy()
    finally:
        # mandatory VRAM hygiene after every video run
        if state is not None:
            predictor.reset_state(state)
            state.clear()
        del state
        torch.cuda.empty_cache()
    return masks, prompt_info


def _prompt_key(points, labels, auto_snap: bool) -> str:
    if points is None:
        return "auto"
    body = ";".join(f"{p[0]:.1f},{p[1]:.1f},{l}" for p, l in zip(points, labels))
    return f"{body}|snap={int(bool(auto_snap) and len(points) == 1)}"


def _cache_path(video_path: Path, points, labels, auto_snap: bool, mm_per_pixel: float, n_disks: int) -> Path:
    raw = (f"{_file_key(video_path)}|{_prompt_key(points, labels, auto_snap)}|{mm_per_pixel:.6f}|{n_disks}|"
           f"{config.SAM2_CHECKPOINT.name}|v{TRACKER_VERSION}")
    return config.CACHE_DIR / f"seg_{video_path.stem}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}.json"


def latest_result_path(stem: str) -> Path:
    return config.CACHE_DIR / "latest" / f"{stem}.json"


def load_latest_result(stem: str) -> Optional[dict]:
    """Most recent segmentation of a sample (any prompt), used for biplane pairing."""
    pointer = latest_result_path(stem)
    try:
        target = config.CACHE_DIR / json.loads(pointer.read_text())["cache_file"]
        return json.loads(target.read_text())
    except (OSError, KeyError, json.JSONDecodeError):
        return None


def _remember_latest(stem: str, cache_file: Path) -> None:
    pointer = latest_result_path(stem)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"cache_file": cache_file.name}))


def load_reference_masks(video_path: Path) -> Optional[np.ndarray]:
    """Bit-packed reference LV masks stored next to a sample (``<stem>_gt.npz``)."""
    side = video_path.with_name(video_path.stem + "_gt.npz")
    if not side.is_file():
        return None
    try:
        z = np.load(side)
        shape = tuple(int(v) for v in z["shape"])
        return np.unpackbits(z["packed"], count=int(np.prod(shape))).reshape(shape).astype(bool)
    except Exception:  # corrupt side-car: quality check simply has no reference
        return None


def assess_quality(volumes: Sequence[float], masks: Sequence[np.ndarray], ed: int, es: int, lvef: float,
                   reference_masks: Optional[np.ndarray] = None, reference_ed: Optional[int] = None) -> dict:
    """Heuristic tracking-quality flags shown as a warning badge in the UI.
    These are plausibility checks, not a validated quality score."""
    vol = np.asarray(volumes, dtype=np.float64)
    edv = float(vol[ed]) if vol.size else 0.0
    flags, messages = [], []
    jump = float(np.max(np.abs(np.diff(vol))) / edv * 100.0) if vol.size > 1 and edv > 0 else 0.0
    if jump > 20.0:
        flags.append("volume_jump")
        messages.append(f"Volume changes by {jump:.0f} % of EDV between two consecutive frames — tracking may have slipped.")
    if np.any(vol <= 0):
        flags.append("empty_frames")
        messages.append(f"{int(np.sum(vol <= 0))} frame(s) have no mask.")
    solidity = mask_solidity(masks[ed]) if len(masks) else 0.0
    if solidity < 0.80:
        flags.append("non_compact_mask")
        messages.append(f"End-diastolic mask solidity is {solidity:.2f} — it may include wall or leak out of the cavity.")
    if not (5.0 <= lvef <= 85.0):
        flags.append("implausible_lvef")
        messages.append(f"LVEF of {lvef:.0f} % is outside the plausible 5–85 % range.")
    area_ratio = None
    if reference_masks is not None and len(reference_masks) == len(masks):
        ref_idx = reference_ed if reference_ed is not None and 0 <= reference_ed < len(masks) else ed
        ref_area = float(reference_masks[ref_idx].sum())
        if ref_area > 0:
            area_ratio = float(masks[ref_idx].sum()) / ref_area
            if area_ratio < 0.5:
                flags.append("under_segmentation")
                messages.append(f"Tracked cavity area is {area_ratio * 100:.0f} % of the reference annotation at end-diastole.")
            elif area_ratio > 1.6:
                flags.append("over_segmentation")
                messages.append(f"Tracked cavity area is {area_ratio * 100:.0f} % of the reference annotation at end-diastole.")
    return {
        "level": "warning" if flags else "good",
        "flags": flags,
        "messages": messages,
        "max_frame_jump_pct_of_edv": round(jump, 1),
        "ed_mask_solidity": round(solidity, 3),
        "area_ratio_vs_reference": None if area_ratio is None else round(area_ratio, 3),
        "note": "Heuristic plausibility checks, not a validated quality metric.",
    }


def segment_video(
    video_path: str | Path,
    click: Optional[Sequence[float]] = None,
    mm_per_pixel: Optional[float] = None,
    use_cache: bool = True,
    force: bool = False,
    n_disks: int = config.N_SIMPSON_DISKS,
    return_masks: bool = False,
    points: Optional[Sequence[Sequence[float]]] = None,
    labels: Optional[Sequence[int]] = None,
    auto_snap: bool = True,
) -> dict:
    """Track the LV cavity through ``video_path`` and quantify it.

    Prompting (frame 0)
    -------------------
    * nothing            -> automatic LV centroid,
    * ``click=[x, y]``   -> one positive point; with ``auto_snap`` a click on
      echogenic wall is moved into the adjacent blood pool first,
    * ``points`` + ``labels`` -> multi-point prompt (1 = cavity, 0 = not
      cavity), used exactly as given.

    Returns
    -------
    dict with ``frames_data, edv_ml, esv_ml, lvef_pct, ed_frame_idx,
    es_frame_idx, vram_mb, inference_time_sec`` (+ prompt, quality and
    calibration metadata).
    """
    import torch

    video_path = Path(video_path)
    config.ensure_dirs()
    calib = load_calibration(video_path)
    if mm_per_pixel is None:
        mm_per_pixel = float(calib.get("mm_per_pixel", config.DEFAULT_MM_PER_PIXEL))
    if not (mm_per_pixel > 0):
        raise ValueError("mm_per_pixel must be > 0")
    pts, lbl, mode = normalise_prompt(click, points, labels)

    cache_file = _cache_path(video_path, pts, lbl, auto_snap, mm_per_pixel, n_disks)
    if use_cache and not force and not return_masks and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text())
            cached["cached"] = True
            _remember_latest(video_path.stem, cache_file)
            return cached
        except json.JSONDecodeError:
            cache_file.unlink(missing_ok=True)

    meta = extract_frames(video_path)
    w, h, n_frames = meta["width"], meta["height"], meta["n_frames"]
    fps = float(calib.get("fps") or meta["fps"] or 25.0)
    frame0_gray = cv2.imread(str(Path(meta["frames_dir"]) / "00000.jpg"), cv2.IMREAD_GRAYSCALE)

    snap_info = {"snapped": False, "reason": "auto-snap not applicable", "distance_px": 0.0}
    click_original = None
    if mode == "auto":
        pts, lbl = [list(auto_centroid(frame0_gray))], [1]
        click_source = "auto-centroid"
    else:
        for p in pts:
            if not (0 <= p[0] < w and 0 <= p[1] < h):
                raise ValueError(f"point {p} is outside the {w}x{h} frame")
        click_source = "user" if mode == "single" else "user multi-point"
        if mode == "single":
            click_original = list(pts[0])
            if auto_snap:
                snap_info = snap_to_cavity(frame0_gray, pts[0])
                if snap_info["snapped"]:
                    pts = [list(snap_info["point"])]
            else:
                snap_info["reason"] = "auto-snap disabled"

    with _gpu_lock:
        predictor = get_predictor()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        dtype_used = config.AUTOCAST_DTYPE
        try:
            masks, prompt_info = _propagate(predictor, meta["frames_dir"], pts, lbl, dtype_used)
        except (RuntimeError, TypeError) as exc:
            if dtype_used == "bfloat16" and "out of memory" not in str(exc).lower() and "no object" not in str(exc):
                log.warning("bfloat16 autocast failed (%s) — retrying with float16", exc)
                dtype_used = "float16"
                masks, prompt_info = _propagate(predictor, meta["frames_dir"], pts, lbl, dtype_used)
            else:
                raise
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 2**20
        peak_reserved = torch.cuda.max_memory_reserved() / 2**20
        after = gpu_memory_mb()

    first_pos = next(p for p, l in zip(pts, lbl) if l == 1)
    frames_data, volumes, clean = [], [], []
    for i in range(n_frames):
        m = clean_mask(masks.get(i, np.zeros((h, w), bool)), prefer_xy=first_pos if i == 0 else None)
        res = compute_simpsons_20_disks(m, mm_per_pixel, n_disks)
        volumes.append(res["volume_ml"])
        clean.append(m)
        frames_data.append(
            {
                "frame_idx": i,
                "time_sec": round(i / fps, 4),
                "volume_ml": round(res["volume_ml"], 3),
                "area_mm2": round(res["area_mm2"], 2),
                "length_mm": round(res["length_mm"], 2),
                "diameters_mm": [round(d, 2) for d in res["diameters_mm"]],
                "long_axis": res["long_axis"],
                "chords": res["chords"],
                "contour": mask_contour(m),
            }
        )

    vol = np.asarray(volumes, dtype=np.float64)
    if not np.any(vol > 0):
        raise RuntimeError("SAM 2.1 returned an empty mask on every frame — try clicking inside the LV cavity")
    ed = int(np.argmax(vol))
    valid = np.where(vol > 0, vol, np.inf)
    es = int(np.argmin(valid))
    edv, esv = float(vol[ed]), float(vol[es])
    lvef = (edv - esv) / edv * 100.0 if edv > 0 else 0.0

    reference = calib.get("ground_truth") or {}
    quality = assess_quality(vol, clean, ed, es, lvef, load_reference_masks(video_path), reference.get("ed_frame_idx"))
    calibrated = bool(calib.get("calibrated", "mm_per_pixel" in calib))

    result = {
        "video": video_path.name,
        "sample_id": calib.get("id", video_path.stem),
        "width": w,
        "height": h,
        "n_frames": n_frames,
        "fps": fps,
        "mm_per_pixel": mm_per_pixel,
        "calibrated": calibrated,
        "volume_unit": "mL" if calibrated else "index",
        "prompt_mode": mode,
        "points": [[round(p[0], 1), round(p[1], 1)] for p in pts],
        "labels": list(lbl),
        "click": [round(first_pos[0], 1), round(first_pos[1], 1)],
        "click_original": None if click_original is None else [round(click_original[0], 1), round(click_original[1], 1)],
        "click_source": click_source,
        "snap": snap_info,
        "prompt": prompt_info,
        "frames_data": frames_data,
        "volumes_ml": [round(float(v), 3) for v in vol],
        "edv_ml": round(edv, 2),
        "esv_ml": round(esv, 2),
        "lvef_pct": round(lvef, 2),
        "single_plane_lvef": round(lvef, 2),
        "ed_frame_idx": ed,
        "es_frame_idx": es,
        "quality": quality,
        "vram_mb": round(peak_reserved, 1),             # peak reserved by PyTorch during this run
        "vram_allocated_mb": round(peak_alloc, 1),      # peak live tensors
        "vram_after_mb": round(after["allocated_mb"], 1),
        "vram_budget_mb": config.VRAM_BUDGET_MB,
        "inference_time_sec": round(inference_time, 3),
        "fps_inference": round(n_frames / inference_time, 2),
        "ms_per_frame": round(inference_time / n_frames * 1000.0, 1),
        "model": config.SAM2_CHECKPOINT.name,
        "autocast_dtype": dtype_used,
        "n_disks": n_disks,
        "cached": False,
    }
    if use_cache:
        cache_file.write_text(json.dumps(result))
        _remember_latest(video_path.stem, cache_file)
    if return_masks:
        result = dict(result)
        result["_masks"] = np.asarray(clean)
    return result
