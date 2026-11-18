#!/usr/bin/env python3
"""Create the sample ultrasound library in backend/data/samples/.

1. Two *calibrated* beating-ventricle phantom cine loops (apical 4-chamber
   layout, synthetic speckle) whose LV cavity is the exact long-axis section of
   a half prolate ellipsoid, so ground-truth EDV / ESV / LVEF are analytic:
       phantom_normal  EDV 120 mL, ESV  48.0 mL  -> LVEF 60 %
       phantom_hfref   EDV 210 mL, ESV 142.8 mL  -> LVEF 32 %
2. A curated demo cohort of eight real patients from the open CAMUS dataset,
   each with BOTH apical views (A4C + A2C = 16 sequences), spanning normal,
   mildly reduced and reduced ejection fraction and good / medium / poor image
   quality.  Files come from the public (no-login) Girder API and are cached
   in backend/data/camus_raw/, so a second run needs no network.  If the
   server is unreachable the script warns and carries on with the phantoms.

Every sample is ``<id>.mp4`` (H.264) + ``<id>.json`` (mm_per_pixel, fps,
subject, view, reference values ...) + ``<id>_gt.npz`` (bit-packed reference
LV masks).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.core import camus, config  # noqa: E402
from backend.core.simpsons_math import (  # noqa: E402
    biplane_volume_from_disks,
    compute_simpsons_20_disks,
    lv_dimensions_from_volume,
)
from backend.core.video_io import ffmpeg_available, read_video_gray, write_mp4  # noqa: E402,F401

# ------------------------------ phantom specs ------------------------------- #
SIZE = 512                 # square frame, px
MM_PER_PIXEL = 0.34
FPS = 25
N_FRAMES = 50              # two beats at 60 bpm
FRAMES_PER_BEAT = 25
FAN_APEX = (256.0, 4.0)    # (x, y) px
FAN_HALF_ANGLE_DEG = 45.0
FAN_DEPTH_PX = 500.0
LV_AXIS_X = 264.0          # px
LV_APEX_Y = 70.0           # px (fixed; the base descends toward it in systole)

PHANTOMS = [
    dict(
        id="phantom_normal",
        name="Phantom · Normal (LVEF 60 %)",
        description="Calibrated half-ellipsoid LV phantom, normal systolic function.",
        edv_ml=120.0, esv_ml=48.0, ed_length_to_radius=3.5, wall_ed_mm=9.0, seed=11,
    ),
    dict(
        id="phantom_hfref",
        name="Phantom · HFrEF (LVEF 32 %)",
        description="Calibrated half-ellipsoid LV phantom, dilated ventricle with reduced ejection fraction.",
        edv_ml=210.0, esv_ml=142.8, ed_length_to_radius=2.9, wall_ed_mm=8.0, seed=23,
    ),
]

# ------------------------------- CAMUS specs -------------------------------- #
# Demo cohort: (patient, why it was picked).  EF and quality come from the dataset's own Info files.
CAMUS_COHORT = [
    "patient0017",   # normal EF, good quality
    "patient0004",   # normal EF, medium quality
    "patient0005",   # mildly reduced EF, good quality
    "patient0020",   # mildly reduced EF, medium quality
    "patient0022",   # mildly reduced EF, poor quality
    "patient0038",   # reduced EF, good quality
    "patient0019",   # reduced EF, medium quality
    "patient0008",   # reduced EF, poor quality
]
CAMUS_VIEWS = ("A4C", "A2C")


# =============================== utilities ================================== #
def volume_fraction(tau: float) -> float:
    """Normalised LV volume g in [0, 1] over one cardiac cycle (tau in [0, 1)).
    g = 1 at end-diastole (tau = 0), g = 0 at end-systole (tau = 0.36)."""
    knots = [(0.0, 1.0), (0.36, 0.0), (0.60, 0.72), (0.84, 0.80), (1.0, 1.0)]
    tau = tau % 1.0
    for (t0, g0), (t1, g1) in zip(knots[:-1], knots[1:]):
        if t0 <= tau <= t1:
            u = (tau - t0) / (t1 - t0)
            return g0 + (g1 - g0) * 0.5 * (1 - math.cos(math.pi * u))
    return 1.0


def save_gt_masks(path: Path, masks: np.ndarray) -> None:
    np.savez_compressed(path, packed=np.packbits(masks.astype(bool), axis=None), shape=np.array(masks.shape))


def load_gt_masks(path: Path) -> np.ndarray:
    z = np.load(path)
    shape = tuple(int(v) for v in z["shape"])
    return np.unpackbits(z["packed"], count=int(np.prod(shape))).reshape(shape).astype(bool)


# ================================ phantom =================================== #
def _half_ellipse(xx, yy, cx, base_y, b_px, a_px):
    """Half ellipse with flat base at base_y and apex a_px above it."""
    return (((xx - cx) / b_px) ** 2 + ((base_y - yy) / a_px) ** 2 <= 1.0) & (yy <= base_y)


def _ellipse(xx, yy, cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def render_phantom(spec: dict) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(spec["seed"])
    s = MM_PER_PIXEL
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    yy += 0.0
    # pixel-centre convention: pixel (i, j) covers [j, j+1) x [i, i+1); centre at +0.5
    xc, yc = xx + 0.5, yy + 0.5

    dx, dy = xc - FAN_APEX[0], yc - FAN_APEX[1]
    rad = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dx, dy))
    fan = (np.abs(ang) <= FAN_HALF_ANGLE_DEG) & (rad <= FAN_DEPTH_PX) & (rad >= 6)
    fan_soft = cv2.GaussianBlur(fan.astype(np.float32), (0, 0), 1.0)
    depth_gain = (1.0 - 0.25 * np.clip(rad / FAN_DEPTH_PX, 0, 1)).astype(np.float32)

    edv, esv = spec["edv_ml"], spec["esv_ml"]
    a_ed, b_ed = lv_dimensions_from_volume(edv, edv, spec["ed_length_to_radius"])

    frames, masks, volumes = [], [], []
    noise = None
    for f in range(N_FRAMES):
        g = volume_fraction((f % FRAMES_PER_BEAT) / FRAMES_PER_BEAT)
        vol = esv + (edv - esv) * g
        a, b = lv_dimensions_from_volume(vol, edv, spec["ed_length_to_radius"])
        a_px, b_px = a / s, b / s
        base_y = LV_APEX_Y + a_px
        wall = spec["wall_ed_mm"] * (b_ed / b) ** 0.8 / s            # thickens in systole
        valve = 3.5 / s

        lv = _half_ellipse(xc, yc, LV_AXIS_X, base_y, b_px, a_px)
        lv_outer = _half_ellipse(xc, yc, LV_AXIS_X, base_y, b_px + wall, a_px + wall)
        peri = _half_ellipse(xc, yc, LV_AXIS_X, base_y, b_px + wall + 5, a_px + wall + 5)

        # right ventricle: shorter, hugging the septum on screen-left
        rv_b, rv_a = 0.78 * b_px, 0.74 * a_px
        rv_cx = LV_AXIS_X - (b_px + wall) - 0.80 * rv_b
        rv = _half_ellipse(xc, yc, rv_cx, base_y, rv_b, rv_a) & ~lv_outer
        rv_wall = _half_ellipse(xc, yc, rv_cx, base_y, rv_b + 4.5 / s, rv_a + 4.5 / s) & ~lv_outer

        # atrioventricular plane (closed valves + annulus) and the atria below it
        av_plane = (yc > base_y) & (yc <= base_y + valve) & (xc > rv_cx - rv_b - 4.5 / s) & (xc < LV_AXIS_X + b_px + wall)
        atr = 1.0 + 0.10 * (1 - g)                                     # atria fill while ventricles empty
        la_rx, la_ry = 21.0 / s * atr, 20.0 / s * atr
        la_cy = base_y + valve + la_ry + 1.0 / s
        la = _ellipse(xc, yc, LV_AXIS_X, la_cy, la_rx, la_ry)
        la_wall = _ellipse(xc, yc, LV_AXIS_X, la_cy, la_rx + 3.5 / s, la_ry + 3.5 / s)
        ra_rx, ra_ry = 18.0 / s * atr, 19.0 / s * atr
        ra_cx = min(rv_cx, LV_AXIS_X - la_rx - 3.5 / s - ra_rx - 4.0 / s)
        ra_cy = base_y + valve + ra_ry + 1.0 / s
        ra = _ellipse(xc, yc, ra_cx, ra_cy, ra_rx, ra_ry)
        ra_wall = _ellipse(xc, yc, ra_cx, ra_cy, ra_rx + 3.0 / s, ra_ry + 3.0 / s)

        echo = np.full((SIZE, SIZE), 0.20, np.float32)                 # background tissue
        echo[peri] = 0.78                                              # bright pericardium rim
        echo[rv_wall] = 0.55
        echo[la_wall | ra_wall] = 0.55
        echo[lv_outer] = 0.60                                          # myocardium
        echo[rv] = 0.045
        echo[la | ra] = 0.045
        echo[av_plane] = 0.85
        echo[lv] = 0.04                                                # blood pool
        echo = cv2.GaussianBlur(echo, (0, 0), 0.8)

        # temporally correlated, fully developed speckle (Rayleigh amplitude)
        fresh = rng.standard_normal((2, SIZE, SIZE)).astype(np.float32)
        noise = fresh if noise is None else 0.78 * noise + math.sqrt(1 - 0.78**2) * fresh
        re = cv2.GaussianBlur(noise[0], (0, 0), sigmaX=1.7, sigmaY=1.0)
        im = cv2.GaussianBlur(noise[1], (0, 0), sigmaX=1.7, sigmaY=1.0)
        amp = np.hypot(re, im)
        amp /= amp.mean()

        img = echo * amp * depth_gain + 0.006 * np.abs(rng.standard_normal((SIZE, SIZE)).astype(np.float32))
        img = np.log1p(18.0 * img) / math.log1p(18.0)                  # log compression
        img = cv2.GaussianBlur(img, (0, 0), 0.6) * fan_soft
        frames.append(np.clip(img * 255.0, 0, 255).astype(np.uint8))
        masks.append(lv)
        volumes.append(vol)
    return np.asarray(frames), np.asarray(masks), volumes


def build_phantom(spec: dict) -> dict:
    frames, masks, volumes = render_phantom(spec)
    out = config.SAMPLES_DIR
    write_mp4(frames, out / f"{spec['id']}.mp4", FPS)
    save_gt_masks(out / f"{spec['id']}_gt.npz", masks)

    ed, es = int(np.argmax(volumes)), int(np.argmin(volumes))
    ys, xs = np.nonzero(masks[0])
    sim = [compute_simpsons_20_disks(m, MM_PER_PIXEL)["volume_ml"] for m in masks]
    cal_err = float(np.max(np.abs(np.array(sim) - np.array(volumes)) / np.array(volumes)) * 100)
    meta = {
        "id": spec["id"],
        "name": spec["name"],
        "kind": "phantom",
        "view": "A4C",
        "subject_id": spec["id"],
        "subject_name": spec["name"],
        "ef_class": camus.ef_class((spec["edv_ml"] - spec["esv_ml"]) / spec["edv_ml"] * 100),
        "calibrated": True,
        "volume_unit": "mL",
        "description": spec["description"],
        "video": f"{spec['id']}.mp4",
        "gt_masks": f"{spec['id']}_gt.npz",
        "width": SIZE,
        "height": SIZE,
        "n_frames": N_FRAMES,
        "fps": FPS,
        "mm_per_pixel": MM_PER_PIXEL,
        "heart_rate_bpm": 60,
        "lv_seed_point": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        "ground_truth": {
            "method": "analytic half prolate ellipsoid, V = 2/3*pi*b^2*a",
            "edv_ml": spec["edv_ml"],
            "esv_ml": spec["esv_ml"],
            "lvef_pct": round((spec["edv_ml"] - spec["esv_ml"]) / spec["edv_ml"] * 100, 2),
            "ed_frame_idx": ed,
            "es_frame_idx": es,
            "volumes_ml": [round(v, 3) for v in volumes],
            "simpson_on_gt_mask_max_err_pct": round(cal_err, 3),
            # the phantom is a solid of revolution, so biplane and single-plane truths coincide
            "reference_biplane_lvef_pct": round((spec["edv_ml"] - spec["esv_ml"]) / spec["edv_ml"] * 100, 2),
        },
    }
    (out / f"{spec['id']}.json").write_text(json.dumps(meta, indent=2))
    print(f"  {spec['id']}: {N_FRAMES} frames, GT EDV {spec['edv_ml']} / ESV {spec['esv_ml']} mL, "
          f"LVEF {meta['ground_truth']['lvef_pct']} %, Simpson-on-GT max err {cal_err:.2f} %")
    return meta


# ================================= CAMUS ==================================== #
def _even(frames: np.ndarray, masks: np.ndarray):
    """Pad to even dimensions for H.264 (keeps pixel coordinates unchanged)."""
    _, h, w = frames.shape
    ph, pw = h % 2, w % 2
    if ph or pw:
        frames = np.pad(frames, ((0, 0), (0, ph), (0, pw)))
        masks = np.pad(masks, ((0, 0), (0, ph), (0, pw)))
    return frames, masks


def build_camus_sequence(patient: str, view: str) -> dict:
    """One CAMUS sequence -> mp4 + reference masks; returns metadata plus the
    per-frame disk measurements needed for the reference biplane value."""
    d = camus.load_sequence(patient, view)
    info, mmpp = d["info"], d["mm_per_pixel"]
    frames, masks = _even(d["frames"], d["labels"] == camus.LABEL_LV)
    tag = camus.VIEWS[view].lower()
    sid = f"camus_{patient}_{tag}"
    fps = float(info.get("FrameRate", 50.0))
    out = config.SAMPLES_DIR
    write_mp4(frames, out / f"{sid}.mp4", fps)
    save_gt_masks(out / f"{sid}_gt.npz", masks)

    disks = [compute_simpsons_20_disks(m, mmpp) for m in masks]
    vols = [r["volume_ml"] for r in disks]
    ed, es = int(np.argmax(vols)), int(np.argmin(vols))
    ys, xs = np.nonzero(masks[0])
    ef_reported = float(info["EF"]) if "EF" in info else None
    meta = {
        "id": sid,
        "name": f"CAMUS · {patient} · {view}",
        "kind": "camus",
        "view": view,
        "subject_id": f"camus_{patient}",
        "subject_name": f"CAMUS {patient} · EF {info.get('EF', '?')} % · {camus.ef_class(ef_reported).split(' (')[0]}",
        "ef_class": camus.ef_class(ef_reported),
        "image_quality": info.get("ImageQuality"),
        "calibrated": True,
        "volume_unit": "mL",
        "description": f"Real {view} half-cycle (ED -> ES) from the open CAMUS dataset, image quality "
                       f"{info.get('ImageQuality', 'n/a')}. Reference values are computed from the dataset's "
                       f"LV masks; the EF reported with the dataset (biplane, by its clinical readers) is "
                       f"{info.get('EF', 'n/a')} %.",
        "video": f"{sid}.mp4",
        "gt_masks": f"{sid}_gt.npz",
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "n_frames": int(frames.shape[0]),
        "fps": fps,
        "mm_per_pixel": round(mmpp, 6),
        "lv_seed_point": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        "ground_truth": {
            "method": f"20-disk single-plane Simpson on the CAMUS reference LV masks ({view})",
            "edv_ml": round(vols[ed], 2),
            "esv_ml": round(vols[es], 2),
            "lvef_pct": round((vols[ed] - vols[es]) / vols[ed] * 100, 2),
            "ed_frame_idx": ed,
            "es_frame_idx": es,
            "volumes_ml": [round(v, 3) for v in vols],
            "camus_biplane_ef_pct": ef_reported,
        },
        "camus_info": info,
        "source": camus.SOURCE,
        "citation": camus.CITATION,
    }
    meta["_disks"] = {"ed": disks[ed], "es": disks[es]}
    return meta


def build_camus_patient(patient: str) -> list[dict]:
    """Both apical views of one patient + the reference biplane value that ties them together."""
    metas = {view: build_camus_sequence(patient, view) for view in CAMUS_VIEWS}
    d4, d2 = metas["A4C"].pop("_disks"), metas["A2C"].pop("_disks")
    edv = biplane_volume_from_disks(d4["ed"]["diameters_mm"], d4["ed"]["length_mm"], d2["ed"]["diameters_mm"], d2["ed"]["length_mm"])
    esv = biplane_volume_from_disks(d4["es"]["diameters_mm"], d4["es"]["length_mm"], d2["es"]["diameters_mm"], d2["es"]["length_mm"])
    for meta in metas.values():
        meta["ground_truth"].update(
            reference_biplane_edv_ml=round(edv, 2),
            reference_biplane_esv_ml=round(esv, 2),
            reference_biplane_lvef_pct=round((edv - esv) / edv * 100, 2),
            reference_biplane_method="biplane method of disks on the CAMUS reference masks of both views",
        )
        (config.SAMPLES_DIR / f"{meta['id']}.json").write_text(json.dumps(meta, indent=2))
    return list(metas.values())


# ================================== main ==================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-camus", action="store_true", help="do not import CAMUS sequences")
    args = ap.parse_args(argv)

    if not ffmpeg_available():
        print("ERROR: ffmpeg is required to encode the sample videos", file=sys.stderr)
        return 2
    config.ensure_dirs()
    manifest = {"samples": [], "subjects": [], "camus": {"attempted": not args.skip_camus, "fetched": [], "errors": {}}}

    total = len(PHANTOMS) + (0 if args.skip_camus else len(CAMUS_COHORT))
    bar = tqdm(total=total, desc="preparing samples", unit="subject")
    done = 0

    def step(label: str) -> None:
        nonlocal done
        done += 1
        bar.update(1)
        tqdm.write(f"[{done}/{total} subjects | {done / total * 100:5.1f}%] {label}")

    for spec in PHANTOMS:
        meta = build_phantom(spec)
        manifest["samples"].append(meta["id"])
        manifest["subjects"].append(meta["subject_id"])
        step(f"{spec['id']}: analytic phantom, LVEF {meta['ground_truth']['lvef_pct']} %")

    if not args.skip_camus:
        for patient in CAMUS_COHORT:
            try:
                metas = build_camus_patient(patient)
                manifest["samples"] += [m["id"] for m in metas]
                manifest["subjects"].append(metas[0]["subject_id"])
                manifest["camus"]["fetched"].append(patient)
                g = metas[0]["ground_truth"]
                step(f"{patient}: A4C quality {metas[0]['image_quality']}, A2C quality {metas[1]['image_quality']}, "
                     f"reference biplane LVEF {g['reference_biplane_lvef_pct']} % (dataset-reported EF {g['camus_biplane_ef_pct']} %)")
            except Exception as exc:  # network down, layout changed ... -> phantoms still work
                manifest["camus"]["errors"][patient] = f"{type(exc).__name__}: {exc}"
                step(f"WARNING: could not import {patient}: {type(exc).__name__}: {exc}")
    bar.close()

    (config.SAMPLES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {len(manifest['samples'])} sequences / {len(manifest['subjects'])} subjects in {config.SAMPLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
