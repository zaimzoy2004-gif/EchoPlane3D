#!/usr/bin/env python3
"""Reproducible validation for the EchoPlane3D paper.

Every number that appears in ``paper/`` comes from the files this script
writes to ``results/``.  Nothing is copied from earlier runs.

Experiment A — analytical phantoms & foreshortening (CPU, plus SAM 2.1 on the two phantom clips)
    results/table1_phantom_verification.csv     20-disk Simpson on a prolate ellipsoid (5 spacings x 5 rotations)
    results/table1b_foreshortening_sweep.csv    closed-form model vs numerical plane slice, 0-30 deg tilt
    results/table1c_phantom_tracking.csv        SAM 2.1 tracking of the two calibrated phantom cine loops

Experiment B — CAMUS (reference = the dataset's expert LV masks; NOT clinical ground truth)
    30 patients never viewed during development (patient0041 onward), 10 per A4C image quality
    (Good / Medium / Poor); ED and ES frames of A4C and A2C = 120 frames x 6 prompting strategies.
    results/camus_patient_metrics.csv           one row per (patient, view, phase, strategy): DSC, HD95, volumes
    results/camus_patient_volumes.csv           one row per (patient, strategy): single-plane + biplane EDV/ESV/LVEF
    results/table2_camus_validation.csv         DSC / HD95 by strategy x image quality
    results/table2b_camus_agreement.csv         MAE, bias, 95 % LoA, Pearson r, ICC(2,1) for volumes and LVEF
    results/table2c_showcase_tracking.csv       full-cycle tracking of the 8-patient demo cohort (tuned-on data!)
    results/table2d_heldout_tracking.csv        full-cycle tracking (the app's real pipeline) of the 30 held-out
                                                patients, A4C + A2C half cycles, automatic and snapped prompts
    results/table2e_heldout_tracking_agreement.csv  volume / LVEF agreement of that tracking

Experiment C — hardware telemetry on this machine
    results/table3_hardware_telemetry.csv

Figures (300 DPI PNG + PDF): results/figures/fig1..fig4
Provenance: results/benchmark_manifest.json

Usage:  /home/burha/echo_env/bin/python3 scripts/run_paper_benchmarks.py [--skip-gpu] [--n-per-quality 10]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.core import camus, config  # noqa: E402
from backend.core.simpsons_math import (  # noqa: E402
    biplane_volume_from_disks,
    compute_biplane_simpson,
    compute_simpsons_20_disks,
    foreshortened_length,
    lv_dimensions_from_volume,
    prolate_ellipsoid_volume_ml,
    simulate_apical_foreshortening,
    synth_ellipse_mask,
)

RESULTS = config.RESULTS_DIR
FIGURES = RESULTS / "figures"
SEED = 20261118
QUALITIES = ("Good", "Medium", "Poor")
FIRST_HELD_OUT = 41                       # patients 0001-0040 were looked at while building the demo cohort
STRATEGIES = {
    "naive_click": "Naive single click (stock SAM 2.1: highest predicted IoU)",
    "snapped_click": "Cavity-snapped single click (EchoPlane3D)",
    "multi_point": "Multi-point: 3 cavity + 1 wall point (EchoPlane3D)",
    "automatic": "Automatic prompt, no reference input (EchoPlane3D)",
    "naive_wall_click": "Click on the myocardium, stock SAM 2.1",
    "snapped_wall_click": "Click on the myocardium, cavity-snapped (EchoPlane3D)",
}
MAIN_STRATEGIES = ("naive_click", "snapped_click", "multi_point", "automatic")


# =========================================================================== #
# helpers
# =========================================================================== #
def banner(phase: str, pct: float, msg: str) -> None:
    print(f"[PROGRESS] Benchmark {phase}: {pct:.0f}% complete — {msg}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})
    print(f"  wrote {path.relative_to(config.PROJECT_ROOT)} ({len(rows)} rows)")


def _fmt(v):
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return json.dumps(v)
    return v


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)


def hd95(a: np.ndarray, b: np.ndarray, mm_per_pixel: float) -> float:
    """Symmetric 95th-percentile Hausdorff distance between mask boundaries, in mm
    (max of the two directed 95th percentiles; NaN if either mask is empty)."""
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    sa, sb = _surface(a), _surface(b)
    d_ab = ndimage.distance_transform_edt(~sb)[sa]
    d_ba = ndimage.distance_transform_edt(~sa)[sb]
    return float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)) * mm_per_pixel)


def icc_2_1(est: np.ndarray, ref: np.ndarray) -> float:
    """ICC(2,1) between an estimate and a reference (see :func:`icc_2_1_matrix`)."""
    return icc_2_1_matrix(np.column_stack([est, ref]))


def icc_2_1_matrix(y: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement
    (Shrout & Fleiss 1979).  ``y`` is targets x raters."""
    y = np.asarray(y, dtype=np.float64)
    n, k = y.shape
    if n < 3:
        return float("nan")
    grand = y.mean()
    ssr = k * np.sum((y.mean(axis=1) - grand) ** 2)
    ssc = n * np.sum((y.mean(axis=0) - grand) ** 2)
    sse = np.sum((y - grand) ** 2) - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else float("nan")


def agreement(est, ref) -> dict:
    est, ref = np.asarray(est, float), np.asarray(ref, float)
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref = est[ok], ref[ok]
    n = int(ok.sum())
    out = {"n": n, "n_excluded": int((~ok).sum())}
    if n < 3:
        return out | {k: float("nan") for k in ("mae", "bias", "sd_diff", "loa_low", "loa_high", "pearson_r", "pearson_p", "icc_2_1")}
    diff = est - ref
    sd = float(np.std(diff, ddof=1))
    r, p = (stats.pearsonr(est, ref) if np.ptp(est) > 0 and np.ptp(ref) > 0 else (float("nan"), float("nan")))
    return out | {
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "sd_diff": sd,
        "loa_low": float(np.mean(diff) - 1.96 * sd),
        "loa_high": float(np.mean(diff) + 1.96 * sd),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "icc_2_1": icc_2_1(est, ref),
    }


def lvef(edv: float, esv: float) -> float:
    return (edv - esv) / edv * 100.0 if edv > 0 and esv > 0 else float("nan")


def proc_mem_mb() -> dict:
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            k, v = line.split(":")
            out[k] = round(int(v.split()[0]) / 1024.0, 1)
    return {"process_rss_mb": out.get("VmRSS"), "process_peak_rss_mb": out.get("VmHWM")}


# =========================================================================== #
# Experiment A
# =========================================================================== #
def experiment_a_math() -> tuple[list[dict], list[dict]]:
    banner("A (phantoms)", 0, "20-disk Simpson on a prolate ellipsoid, 5 spacings x 5 rotations")
    exact = prolate_ellipsoid_volume_ml(25.0, 40.0)
    rows = []
    for mmpp in (0.15, 0.25, 0.34, 0.40, 0.60):
        for ang in (0, 17, 45, 90, 133):
            r = compute_simpsons_20_disks(synth_ellipse_mask(40.0, 25.0, mmpp, ang), mmpp)
            rows.append({"case": "prolate_ellipsoid_R25_halfL40", "mm_per_pixel": mmpp, "rotation_deg": ang,
                         "exact_volume_ml": exact, "simpson_volume_ml": r["volume_ml"],
                         "error_pct": (r["volume_ml"] - exact) / exact * 100.0, "length_mm": r["length_mm"],
                         "max_diameter_mm": max(r["diameters_mm"])})
    worst = max(abs(r["error_pct"]) for r in rows)
    print(f"  ellipsoid: exact {exact:.4f} mL, worst |error| {worst:.3f} % over {len(rows)} cases")

    banner("A (phantoms)", 40, "foreshortening sweep: closed form vs numerical plane slice, 0-30 deg")
    sweep = []
    cases = {"phantom_normal": (120.0, 48.0), "phantom_hfref": (210.0, 142.8)}
    for name, (edv, esv) in cases.items():
        ratio = 3.5 if name == "phantom_normal" else 2.9
        for tilt in range(0, 31):
            m = simulate_apical_foreshortening(edv, esv, tilt, ed_length_to_radius=ratio)
            row = {"case": name, "tilt_deg": tilt, "true_edv_ml": edv, "true_esv_ml": esv,
                   "true_lvef_pct": m["true"]["lvef_pct"], "model_edv_ml": m["foreshortened"]["edv_ml"],
                   "model_esv_ml": m["foreshortened"]["esv_ml"], "model_lvef_pct": m["foreshortened"]["lvef_pct"],
                   "edv_error_pct": m["errors"]["edv_pct"], "esv_error_pct": m["errors"]["esv_pct"],
                   "lvef_error_pts": m["errors"]["lvef_abs_pts"]}
            if tilt % 5 == 0:
                # independent check: rasterise the tilted plane through the 3-D half ellipsoid and run Simpson on it
                th, mmpp, num = math.radians(tilt), 0.25, {}
                for phase, vol in (("edv", edv), ("esv", esv)):
                    a, b = lv_dimensions_from_volume(vol, edv, ratio)
                    vs = np.arange(0, a + 5, mmpp) + mmpp / 2
                    us = np.arange(-b - 5, b + 5, mmpp) + mmpp / 2
                    uu, vv = np.meshgrid(us, vs)
                    x, y, z = uu, vv * math.sin(th), vv * math.cos(th)
                    num[phase] = compute_simpsons_20_disks((x / b) ** 2 + (y / b) ** 2 + (z / a) ** 2 <= 1.0, mmpp)["volume_ml"]
                row.update(numerical_edv_ml=num["edv"], numerical_esv_ml=num["esv"],
                           numerical_lvef_pct=lvef(num["edv"], num["esv"]),
                           model_vs_numerical_edv_pct=(m["foreshortened"]["edv_ml"] - num["edv"]) / num["edv"] * 100)
            sweep.append(row)
    mesh = mesh_cut_check()
    for r in sweep:
        if r["case"] == "phantom_normal" and r["tilt_deg"] in mesh:
            r.update(mesh_apparent_length_mm=mesh[r["tilt_deg"]][0], formula_length_mm=mesh[r["tilt_deg"]][1])
    return rows, sweep


def mesh_cut_check() -> dict:
    """Apparent LV length of the 3-D heart model cut at each tilt vs the closed-form a'."""
    from scripts import build_heart_glb as hb

    meta = json.loads(config.HEART_PLANES_PATH.read_text())
    meshes = hb.load_world_meshes()
    p = meta["planes"]["A4C"]
    b, n, ax = (np.array(p[k]) for k in ("beam", "normal", "tilt_axis"))
    pivot = np.array(meta["pivot"])
    out = {}
    for tilt in (0, 5, 10, 15, 20, 25, 30):
        th = math.radians(tilt)
        b2, n2 = hb.rodrigues(b, ax, th), hb.rodrigues(n, ax, th)
        sec = meshes["LV"].section(plane_origin=pivot, plane_normal=n2)
        apparent = float((-(np.asarray(sec.vertices) - pivot) @ b2).max())
        out[tilt] = (apparent, foreshortened_length(meta["lv_length_mm"], meta["lv_radius_mm"], tilt))
    return out


def experiment_a_tracking(trk, poller_cls) -> tuple[list[dict], list[dict]]:
    from scripts.prepare_samples import load_gt_masks

    rows, telem = [], []
    for i, sid in enumerate(("phantom_normal", "phantom_hfref")):
        banner("A (phantoms)", 70 + 15 * i, f"SAM 2.1 full-cycle tracking of {sid} (automatic prompt)")
        meta = json.loads((config.SAMPLES_DIR / f"{sid}.json").read_text())
        res, smi = _track(trk, poller_cls, config.SAMPLES_DIR / meta["video"], click=None)
        gt = load_gt_masks(config.SAMPLES_DIR / meta["gt_masks"])
        d = [dice(res["_masks"][k], gt[k]) for k in range(len(gt))]
        h = [hd95(res["_masks"][k], gt[k], meta["mm_per_pixel"]) for k in range(len(gt))]
        g = meta["ground_truth"]
        rows.append({"sample": sid, "n_frames": res["n_frames"], "prompt": res["click_source"],
                     "truth_edv_ml": g["edv_ml"], "est_edv_ml": res["edv_ml"], "edv_error_pct": (res["edv_ml"] - g["edv_ml"]) / g["edv_ml"] * 100,
                     "truth_esv_ml": g["esv_ml"], "est_esv_ml": res["esv_ml"], "esv_error_pct": (res["esv_ml"] - g["esv_ml"]) / g["esv_ml"] * 100,
                     "truth_lvef_pct": g["lvef_pct"], "est_lvef_pct": res["lvef_pct"], "lvef_error_pts": res["lvef_pct"] - g["lvef_pct"],
                     "dice_mean": float(np.mean(d)), "dice_min": float(np.min(d)), "hd95_median_mm": float(np.nanmedian(h)),
                     "quality_level": res["quality"]["level"]})
        telem.append(_telemetry_row("phantom full-cycle tracking", sid, res, smi))
    return rows, telem


# =========================================================================== #
# GPU plumbing shared by B and C
# =========================================================================== #
def _track(trk, poller_cls, video: Path, click=None):
    poller = poller_cls()
    poller.start()
    try:
        res = trk.segment_video(video, click=click, force=True, use_cache=False, return_masks=True)
    finally:
        poller.stop()
    return res, poller.peak


def _telemetry_row(kind: str, sample: str, res: dict, smi_peak: float) -> dict:
    return {"run": kind, "sample": sample, "n_frames": res["n_frames"], "width": res["width"], "height": res["height"],
            "prompt": res["click_source"], "inference_time_sec": res["inference_time_sec"],
            "ms_per_frame": res["ms_per_frame"], "fps": res["fps_inference"],
            "vram_reserved_peak_mb": res["vram_mb"], "vram_allocated_peak_mb": res["vram_allocated_mb"],
            "vram_allocated_after_mb": res["vram_after_mb"], "nvidia_smi_peak_mib": smi_peak,
            "autocast_dtype": res["autocast_dtype"], "within_2500mb_budget": res["vram_mb"] < config.VRAM_BUDGET_MB,
            **proc_mem_mb()}


# =========================================================================== #
# Experiment B — CAMUS cohort selection
# =========================================================================== #
def select_cohort(n_per_quality: int) -> dict[str, list[str]]:
    banner("B (CAMUS)", 0, f"selecting {n_per_quality} held-out patients per A4C image quality (from patient{FIRST_HELD_OUT:04d})")
    names = sorted(n for n in camus.patient_folders() if n.startswith("patient") and int(n[7:]) >= FIRST_HELD_OUT)
    excluded = set()
    try:
        from scripts.prepare_samples import CAMUS_COHORT
        excluded = set(CAMUS_COHORT)
    except ImportError:
        pass
    chosen = {q: [] for q in QUALITIES}
    bar = tqdm(names, desc="scanning CAMUS Info files", unit="patient")
    for name in bar:
        if name in excluded:
            continue
        q = camus.read_info(name, "A4C").get("ImageQuality")
        if q in chosen and len(chosen[q]) < n_per_quality:
            chosen[q].append(name)
            bar.set_postfix_str(" ".join(f"{k[0]}{len(v)}" for k, v in chosen.items()))
        if all(len(v) >= n_per_quality for v in chosen.values()):
            break
    bar.close()
    for q, v in chosen.items():
        if len(v) < n_per_quality:
            print(f"  WARNING: only {len(v)} {q} patients found")
    return chosen


# =========================================================================== #
# Experiment B — prompts derived from the reference mask ("simulated user")
# =========================================================================== #
def oracle_prompts(lv: np.ndarray, myo: np.ndarray) -> dict:
    """Prompts a careful user would give, derived from the reference annotation.

    interior: deepest point of the LV cavity; wall: myocardium pixel closest to it;
    multi: three points at 25/50/75 % of the cavity's principal axis (+) and the wall point (-)."""
    depth = ndimage.distance_transform_edt(lv)
    iy, ix = np.unravel_index(int(np.argmax(depth)), depth.shape)
    interior = [float(ix), float(iy)]
    my, mx = np.nonzero(myo)
    if len(mx):
        j = int(np.argmin((mx - ix) ** 2 + (my - iy) ** 2))
        wall = [float(mx[j]), float(my[j])]
    else:                                            # no myocardium label: step outside the cavity
        ys, xs = np.nonzero(~lv & ndimage.binary_dilation(lv, iterations=4))
        j = int(np.argmin((xs - ix) ** 2 + (ys - iy) ** 2))
        wall = [float(xs[j]), float(ys[j])]
    ys, xs = np.nonzero(lv)
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    e1 = evecs[:, 1]
    t = (pts - c) @ e1
    positives = []
    for f in (0.25, 0.5, 0.75):
        p = c + e1 * (t.min() + f * (t.max() - t.min()))
        k = int(np.argmin(np.sum((pts - p) ** 2, axis=1)))           # nearest cavity pixel
        positives.append([float(pts[k, 0]), float(pts[k, 1])])
    return {"interior": interior, "wall": wall, "multi_points": positives + [wall], "multi_labels": [1, 1, 1, 0]}


def run_strategies(trk, predictor, image_predictor, gray: np.ndarray, lv: np.ndarray, myo: np.ndarray, dtype) -> list[dict]:
    """Segment one frame with every strategy, reusing one image embedding."""
    import cv2
    import torch

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        image_predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    torch.cuda.synchronize()
    embed_ms = (time.perf_counter() - t0) * 1000
    o = oracle_prompts(lv, myo)
    plans = {
        "naive_click": ([o["interior"]], [1], "naive", None),
        "snapped_click": ([o["interior"]], [1], "cavity", "snap"),
        "multi_point": (o["multi_points"], o["multi_labels"], "cavity", None),
        "automatic": ([list(trk.auto_centroid(gray))], [1], "cavity", None),
        "naive_wall_click": ([o["wall"]], [1], "naive", None),
        "snapped_wall_click": ([o["wall"]], [1], "cavity", "snap"),
    }
    rows = []
    for name, (pts, lbl, select, snap) in plans.items():
        moved = 0.0
        if snap:
            s = trk.snap_to_cavity(gray, pts[0])
            if s["snapped"]:
                moved = s["distance_px"]
                pts = [s["point"]]
        t1 = time.perf_counter()
        mask, info = trk.predict_frame_mask(predictor, bgr, pts, lbl, select=select, image_predictor=image_predictor)
        torch.cuda.synchronize()
        x, y = (int(round(v)) for v in pts[0])
        rows.append({"strategy": name, "mask": mask, "decode_ms": (time.perf_counter() - t1) * 1000, "embed_ms": embed_ms,
                     "first_point": [round(pts[0][0], 1), round(pts[0][1], 1)], "n_points": len(pts),
                     "first_point_in_reference_lv": bool(lv[y, x]) if 0 <= y < lv.shape[0] and 0 <= x < lv.shape[1] else False,
                     "snap_moved_px": moved, "selection_rule": info["rule"]})
    image_predictor.reset_predictor()
    return rows


def experiment_b_frames(trk, cohort: dict[str, list[str]]) -> tuple[list[dict], list[dict], list[float], list[float]]:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = trk.get_predictor()
    image_predictor = SAM2ImagePredictor(predictor)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.AUTOCAST_DTYPE]
    patients = [(q, p) for q in QUALITIES for p in cohort[q]]
    frame_rows, vol_rows, embed_ms, decode_ms = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    for i, (stratum, patient) in enumerate(patients, 1):
        per_view = {}
        for view in ("A4C", "A2C"):
            d = camus.load_ed_es(patient, view)
            mmpp, quality = d["mm_per_pixel"], d["info"].get("ImageQuality")
            per_view[view] = {"mmpp": mmpp, "quality": quality, "info": d["info"], "ref": {}, "est": {s: {} for s in STRATEGIES}}
            for phase in ("ED", "ES"):
                gray, labels = d[phase]
                lv, myo = labels == camus.LABEL_LV, labels == camus.LABEL_MYO
                per_view[view]["ref"][phase] = lv
                for r in run_strategies(trk, predictor, image_predictor, gray, lv, myo, dtype):
                    m = r.pop("mask")
                    per_view[view]["est"][r["strategy"]][phase] = m
                    embed_ms.append(r["embed_ms"]) if r["strategy"] == "naive_click" else None
                    decode_ms.append(r["decode_ms"])
                    v_est = compute_simpsons_20_disks(m, mmpp)["volume_ml"]
                    v_ref = compute_simpsons_20_disks(lv, mmpp)["volume_ml"]
                    frame_rows.append({
                        "patient": patient, "cohort_stratum_a4c_quality": stratum, "view": view, "sequence_quality": quality,
                        "phase": phase, "strategy": r["strategy"], "dice": dice(m, lv), "hd95_mm": hd95(m, lv, mmpp),
                        "area_ratio": float(m.sum() / lv.sum()) if lv.sum() else float("nan"),
                        "volume_est_ml": v_est, "volume_ref_ml": v_ref, "mm_per_pixel": mmpp,
                        "first_point": r["first_point"], "n_points": r["n_points"],
                        "first_point_in_reference_lv": r["first_point_in_reference_lv"],
                        "snap_moved_px": r["snap_moved_px"], "selection_rule": r["selection_rule"],
                        "decode_ms": r["decode_ms"], "embed_ms": r["embed_ms"],
                    })
        vol_rows += patient_volumes(patient, stratum, per_view)
        tqdm.write(f"[{i}/{len(patients)} patients | {i / len(patients) * 100:5.1f}%] {patient} ({stratum}): "
                   + ", ".join(f"{s} DSC {np.mean([r['dice'] for r in frame_rows[-24:] if r['strategy'] == s]):.2f}"
                               for s in MAIN_STRATEGIES))
    print(f"  peak VRAM during frame experiment: {torch.cuda.max_memory_reserved() / 2**20:.0f} MB reserved")
    return frame_rows, vol_rows, embed_ms, decode_ms


def patient_volumes(patient: str, stratum: str, per_view: dict) -> list[dict]:
    rows = []
    ef_reported = per_view["A4C"]["info"].get("EF")
    a4, a2 = per_view["A4C"], per_view["A2C"]

    def bundle(m4: dict, m2: dict) -> dict:
        out = {}
        for view, m, mm in (("a4c", m4, a4["mmpp"]), ("a2c", m2, a2["mmpp"])):
            ed = compute_simpsons_20_disks(m["ED"], mm)["volume_ml"]
            es = compute_simpsons_20_disks(m["ES"], mm)["volume_ml"]
            out |= {f"edv_sp_{view}": ed or float("nan"), f"esv_sp_{view}": es or float("nan"), f"lvef_sp_{view}": lvef(ed, es)}
        bed = compute_biplane_simpson(m4["ED"], m2["ED"], a4["mmpp"], a2["mmpp"])
        bes = compute_biplane_simpson(m4["ES"], m2["ES"], a4["mmpp"], a2["mmpp"])
        out |= {"edv_bp": bed["volume_ml"] or float("nan"), "esv_bp": bes["volume_ml"] or float("nan"),
                "lvef_bp": lvef(bed["volume_ml"], bes["volume_ml"]),
                "length_mismatch_ed_pct": bed["length_mismatch_pct"] if bed["length_mismatch_pct"] is not None else float("nan")}
        return out

    ref = bundle(a4["ref"], a2["ref"])
    for s in STRATEGIES:
        est = bundle(a4["est"][s], a2["est"][s])
        row = {"patient": patient, "cohort_stratum_a4c_quality": stratum, "a4c_quality": a4["quality"],
               "a2c_quality": a2["quality"], "strategy": s,
               "dataset_reported_ef_pct": float(ef_reported) if ef_reported else float("nan")}
        row |= {f"est_{k}": v for k, v in est.items()} | {f"ref_{k}": v for k, v in ref.items()}
        rows.append(row)
    return rows


def summarise_frames(frame_rows: list[dict]) -> list[dict]:
    out = []
    for s in STRATEGIES:
        for stratum in ("All", *QUALITIES):
            rows = [r for r in frame_rows if r["strategy"] == s and (stratum == "All" or r["sequence_quality"] == stratum)]
            if not rows:
                continue
            d = np.array([r["dice"] for r in rows], float)
            h = np.array([r["hd95_mm"] for r in rows], float)
            hit = np.array([r["first_point_in_reference_lv"] for r in rows], bool)
            out.append({
                "strategy": s, "strategy_label": STRATEGIES[s], "sequence_quality": stratum, "n_frames": len(rows),
                "dice_mean": float(np.nanmean(d)), "dice_sd": float(np.nanstd(d, ddof=1)), "dice_median": float(np.nanmedian(d)),
                "dice_q1": float(np.nanpercentile(d, 25)), "dice_q3": float(np.nanpercentile(d, 75)),
                "hd95_median_mm": float(np.nanmedian(h)), "hd95_q1_mm": float(np.nanpercentile(h, 25)),
                "hd95_q3_mm": float(np.nanpercentile(h, 75)), "failures_dice_below_0_5": int(np.sum(d < 0.5)),
                "empty_masks": int(np.sum(~np.isfinite(h))), "first_point_in_reference_lv_pct": float(hit.mean() * 100),
            })
    return out


def summarise_agreement(vol_rows: list[dict]) -> list[dict]:
    measures = [("lvef_sp_a4c", "pts"), ("lvef_sp_a2c", "pts"), ("lvef_bp", "pts"), ("edv_sp_a4c", "mL"),
                ("esv_sp_a4c", "mL"), ("edv_bp", "mL"), ("esv_bp", "mL")]
    out = []
    for s in STRATEGIES:
        rows = [r for r in vol_rows if r["strategy"] == s]
        for m, unit in measures:
            a = agreement([r[f"est_{m}"] for r in rows], [r[f"ref_{m}"] for r in rows])
            out.append({"strategy": s, "measure": m, "unit": unit, "comparator": "reference masks (same Simpson code)"} | a)
        a = agreement([r["est_lvef_bp"] for r in rows], [r["dataset_reported_ef_pct"] for r in rows])
        out.append({"strategy": s, "measure": "lvef_bp", "unit": "pts", "comparator": "dataset-reported EF (Info file)"} | a)
    rows = [r for r in vol_rows if r["strategy"] == "naive_click"]           # reference rows are identical across strategies
    a = agreement([r["ref_lvef_bp"] for r in rows], [r["dataset_reported_ef_pct"] for r in rows])
    out.append({"strategy": "reference_masks", "measure": "lvef_bp", "unit": "pts",
                "comparator": "dataset-reported EF (Info file)"} | a)
    return out


# =========================================================================== #
# Experiment B — full-cycle tracking on the showcase cohort
# =========================================================================== #
def experiment_b_showcase(trk, poller_cls) -> tuple[list[dict], list[dict]]:
    from scripts.prepare_samples import load_gt_masks

    metas = [json.loads(p.read_text()) for p in sorted(config.SAMPLES_DIR.glob("camus_*.json"))]
    if not metas:
        print("  showcase cohort not prepared — skipped (run scripts/prepare_samples.py)")
        return [], []
    rows, telem, tracked = [], [], {}
    jobs = [(m, prompt) for m in metas for prompt in ("automatic", "snapped_click")]
    for i, (meta, prompt) in enumerate(tqdm(jobs, desc="showcase full-cycle tracking", unit="run"), 1):
        gt = load_gt_masks(config.SAMPLES_DIR / meta["gt_masks"])
        click = None
        if prompt == "snapped_click":
            depth = ndimage.distance_transform_edt(gt[0])
            iy, ix = np.unravel_index(int(np.argmax(depth)), depth.shape)
            click = [float(ix), float(iy)]
        res, smi = _track(trk, poller_cls, config.SAMPLES_DIR / meta["video"], click=click)
        n = min(len(gt), res["n_frames"])
        d = [dice(res["_masks"][k], gt[k]) for k in range(n)]
        h = [hd95(res["_masks"][k], gt[k], meta["mm_per_pixel"]) for k in range(n)]
        g = meta["ground_truth"]
        rows.append({"patient": meta["subject_id"].replace("camus_", ""), "view": meta["view"], "prompt": prompt,
                     "tuned_on_this_data": True, "sequence_quality": meta.get("image_quality"), "n_frames": n,
                     "dice_mean": float(np.nanmean(d)), "dice_min": float(np.nanmin(d)), "hd95_median_mm": float(np.nanmedian(h)),
                     "est_edv_ml": res["edv_ml"], "ref_edv_ml": g["edv_ml"], "est_esv_ml": res["esv_ml"], "ref_esv_ml": g["esv_ml"],
                     "est_lvef_sp": res["lvef_pct"], "ref_lvef_sp": g["lvef_pct"],
                     "ref_lvef_bp": g.get("reference_biplane_lvef_pct"), "dataset_reported_ef_pct": g.get("camus_biplane_ef_pct"),
                     "quality_flags": res["quality"]["flags"]})
        tracked[(meta["subject_id"], meta["view"], prompt)] = res
        telem.append(_telemetry_row("CAMUS full-cycle tracking", meta["id"], res, smi))
        res.pop("_masks", None)
        tqdm.write(f"[{i}/{len(jobs)} runs | {i / len(jobs) * 100:5.1f}%] {meta['id']} {prompt}: DSC {np.nanmean(d):.3f}, "
                   f"LVEF {res['lvef_pct']:.1f} vs ref {g['lvef_pct']:.1f}")
    # biplane from the two tracked views of each patient (ED / ES taken per view)
    for r in rows:
        key4, key2 = (f"camus_{r['patient']}", "A4C", r["prompt"]), (f"camus_{r['patient']}", "A2C", r["prompt"])
        if key4 in tracked and key2 in tracked:
            a4, a2 = tracked[key4], tracked[key2]
            vol = {}
            for ph in ("ed", "es"):
                f4, f2 = a4["frames_data"][a4[f"{ph}_frame_idx"]], a2["frames_data"][a2[f"{ph}_frame_idx"]]
                vol[ph] = biplane_volume_from_disks(f4["diameters_mm"], f4["length_mm"], f2["diameters_mm"], f2["length_mm"])
            r["est_lvef_bp"] = lvef(vol["ed"], vol["es"])
    return rows, telem


# =========================================================================== #
# Experiment B — full-cycle tracking of the held-out cohort (the app's real pipeline)
# =========================================================================== #
def experiment_b_heldout_tracking(trk, poller_cls, cohort: dict[str, list[str]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Track every held-out half cycle from a single frame-0 prompt, exactly as the workbench does,
    and compare per-frame masks and the resulting EDV / ESV / LVEF with the reference sequence masks."""
    from backend.core.video_io import write_mp4

    work = config.CACHE_DIR / "benchmark_heldout"
    work.mkdir(parents=True, exist_ok=True)
    patients = [(q, p) for q in QUALITIES for p in cohort[q]]
    seq_rows, telem, tracked, refs = [], [], {}, {}
    total = len(patients) * 4
    bar = tqdm(total=total, desc="held-out full-cycle tracking", unit="run")
    done = 0
    for stratum, patient in patients:
        for view in ("A4C", "A2C"):
            d = camus.load_sequence(patient, view)
            frames, lv, mmpp, info = d["frames"], d["labels"] == camus.LABEL_LV, d["mm_per_pixel"], d["info"]
            _, h, w = frames.shape
            if h % 2 or w % 2:
                frames = np.pad(frames, ((0, 0), (0, h % 2), (0, w % 2)))
                lv = np.pad(lv, ((0, 0), (0, h % 2), (0, w % 2)))
            stem = f"{patient}_{view.lower()}"
            video = work / f"{stem}.mp4"
            if not video.is_file():
                write_mp4(frames, video, float(info.get("FrameRate", 50.0)))
            video.with_suffix(".json").write_text(json.dumps({"id": stem, "mm_per_pixel": mmpp, "calibrated": True,
                                                              "fps": float(info.get("FrameRate", 50.0))}))
            ref_disks = [compute_simpsons_20_disks(m, mmpp) for m in lv]
            ref_vol = [r["volume_ml"] for r in ref_disks]
            r_ed, r_es = int(np.argmax(ref_vol)), int(np.argmin(ref_vol))
            refs[(patient, view)] = {"ed": ref_disks[r_ed], "es": ref_disks[r_es]}
            depth = ndimage.distance_transform_edt(lv[0])
            iy, ix = np.unravel_index(int(np.argmax(depth)), depth.shape)
            for prompt, click in (("automatic", None), ("snapped_click", [float(ix), float(iy)])):
                res, smi = _track(trk, poller_cls, video, click=click)
                n = min(len(lv), res["n_frames"])
                dsc = [dice(res["_masks"][k], lv[k]) for k in range(n)]
                hd = [hd95(res["_masks"][k], lv[k], mmpp) for k in range(n)]
                seq_rows.append({
                    "patient": patient, "cohort_stratum_a4c_quality": stratum, "view": view,
                    "sequence_quality": info.get("ImageQuality"), "prompt": prompt, "n_frames": n,
                    "prompt_in_reference_lv": bool(lv[0][int(round(res["click"][1])), int(round(res["click"][0]))]),
                    "dice_mean": float(np.nanmean(dsc)), "dice_min": float(np.nanmin(dsc)),
                    "dice_frame0": dsc[0], "hd95_median_mm": float(np.nanmedian(hd)),
                    "est_edv_ml": res["edv_ml"], "ref_edv_ml": ref_vol[r_ed], "est_esv_ml": res["esv_ml"], "ref_esv_ml": ref_vol[r_es],
                    "est_lvef_sp": res["lvef_pct"], "ref_lvef_sp": lvef(ref_vol[r_ed], ref_vol[r_es]),
                    "quality_level": res["quality"]["level"], "quality_flags": res["quality"]["flags"],
                    "dataset_reported_ef_pct": float(info["EF"]) if info.get("EF") else float("nan"),
                })
                tracked[(patient, view, prompt)] = {k: res[k] for k in ("frames_data", "ed_frame_idx", "es_frame_idx")}
                telem.append(_telemetry_row("CAMUS held-out full-cycle tracking", stem, res, smi))
                del res
                done += 1
                bar.update(1)
                tqdm.write(f"[{done}/{total} runs | {done / total * 100:5.1f}%] {stem} {prompt}: DSC {np.nanmean(dsc):.3f}, "
                           f"LVEF {seq_rows[-1]['est_lvef_sp']:.1f} vs ref {seq_rows[-1]['ref_lvef_sp']:.1f}")
    bar.close()

    patient_rows = []
    for stratum, patient in patients:
        r4, r2 = refs[(patient, "A4C")], refs[(patient, "A2C")]
        ref_bp = {ph: biplane_volume_from_disks(r4[ph]["diameters_mm"], r4[ph]["length_mm"], r2[ph]["diameters_mm"], r2[ph]["length_mm"])
                  for ph in ("ed", "es")}
        reported = next(r["dataset_reported_ef_pct"] for r in seq_rows if r["patient"] == patient)
        for prompt in ("automatic", "snapped_click"):
            t4, t2 = tracked[(patient, "A4C", prompt)], tracked[(patient, "A2C", prompt)]
            est_bp = {}
            for ph in ("ed", "es"):
                f4, f2 = t4["frames_data"][t4[f"{ph}_frame_idx"]], t2["frames_data"][t2[f"{ph}_frame_idx"]]
                est_bp[ph] = biplane_volume_from_disks(f4["diameters_mm"], f4["length_mm"], f2["diameters_mm"], f2["length_mm"])
            sp = {v: next(r for r in seq_rows if r["patient"] == patient and r["view"] == v and r["prompt"] == prompt) for v in ("A4C", "A2C")}
            patient_rows.append({
                "patient": patient, "cohort_stratum_a4c_quality": stratum, "prompt": prompt,
                "est_edv_bp": est_bp["ed"] or float("nan"), "ref_edv_bp": ref_bp["ed"],
                "est_esv_bp": est_bp["es"] or float("nan"), "ref_esv_bp": ref_bp["es"],
                "est_lvef_bp": lvef(est_bp["ed"], est_bp["es"]), "ref_lvef_bp": lvef(ref_bp["ed"], ref_bp["es"]),
                "est_lvef_sp_a4c": sp["A4C"]["est_lvef_sp"], "ref_lvef_sp_a4c": sp["A4C"]["ref_lvef_sp"],
                "est_lvef_sp_a2c": sp["A2C"]["est_lvef_sp"], "ref_lvef_sp_a2c": sp["A2C"]["ref_lvef_sp"],
                "est_edv_sp_a4c": sp["A4C"]["est_edv_ml"], "ref_edv_sp_a4c": sp["A4C"]["ref_edv_ml"],
                "est_esv_sp_a4c": sp["A4C"]["est_esv_ml"], "ref_esv_sp_a4c": sp["A4C"]["ref_esv_ml"],
                "dataset_reported_ef_pct": reported,
            })
    agree = []
    for prompt in ("automatic", "snapped_click"):
        rows = [r for r in patient_rows if r["prompt"] == prompt]
        for m, unit in (("lvef_bp", "pts"), ("lvef_sp_a4c", "pts"), ("lvef_sp_a2c", "pts"), ("edv_bp", "mL"),
                        ("esv_bp", "mL"), ("edv_sp_a4c", "mL"), ("esv_sp_a4c", "mL")):
            a = agreement([r[f"est_{m}"] for r in rows], [r[f"ref_{m}"] for r in rows])
            agree.append({"prompt": prompt, "measure": m, "unit": unit, "comparator": "reference masks (same Simpson code)"} | a)
        a = agreement([r["est_lvef_bp"] for r in rows], [r["dataset_reported_ef_pct"] for r in rows])
        agree.append({"prompt": prompt, "measure": "lvef_bp", "unit": "pts", "comparator": "dataset-reported EF (Info file)"} | a)
        plaus = [r for r in rows if 0 <= r["est_lvef_bp"] <= 90]
        agree[-1]["n_physically_plausible_lvef"] = len(plaus)
    return seq_rows, patient_rows, agree, telem


# =========================================================================== #
# figures
# =========================================================================== #
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e7e6e2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # reference palette slots 1-4, fixed order (light mode)
STRAT_LABEL = {"naive_click": "Naive click", "snapped_click": "Snapped click", "multi_point": "Multi-point",
               "automatic": "Automatic", "naive_wall_click": "Wall click, naive", "snapped_wall_click": "Wall click, snapped"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
        "axes.labelsize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


def _save(fig, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"  wrote results/figures/{name}.png + .pdf")


def fig1_architecture():
    plt = _style()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fill="#f4f3f0", edge=MUTED):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fill, ec=edge, lw=0.8))
        ax.text(x + 1.2, y + h - 1.4, title, fontsize=7.8, weight="bold", color=INK, va="top")
        for k, ln in enumerate(lines):
            ax.text(x + 1.2, y + h - 4.8 - 3.0 * k, ln, fontsize=6.6, color=INK2, va="top")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>", mutation_scale=8, color=INK2, lw=0.8,
                                     shrinkA=0, shrinkB=0))

    for x, txt in ((1, "Browser · vendored libraries, no build step"), (36, "FastAPI · localhost"), (63, "Core · Python, offline")):
        ax.text(x, 55.5, txt, fontsize=7.2, color=MUTED, va="top")
    box(1, 37, 31, 15, "2-D ultrasound pane", ["canvas playback, click / multi-point", "mask, long axis, 20 chords", "clip upload (.mp4 / .avi)"])
    box(1, 19.5, 31, 15, "3-D probe navigator", ["Three.js clipping plane + caps", "A4C / A2C / PLAX / PSAX + tilt", "LV contracts with V(t)"])
    box(1, 2, 31, 15, "Quantification + guide", ["Plotly V(t), reference ED / ES", "estimate vs reference table", "tutor chat, quiz, actions"])
    box(36, 2, 23, 50, "REST API", ["/api/samples", "/api/segment", "/api/foreshortening", "/api/guide", "/api/upload", "",
                                     "static frontend + models"], fill="#eef4fc", edge="#86b6ef")
    box(63, 40, 36, 12, "SAM 2.1 hiera-tiny tracker", ["bf16, RTX 3050 6 GB, 2.5 GB budget", "cavity snap + candidate scoring"])
    box(63, 25.5, 36, 11.5, "Simpson engine", ["single-plane + biplane, 20 disks", "foreshortening model"])
    box(63, 12, 36, 11, "Deterministic tutor", ["optional local Ollama (wording only)", "Pydantic action whitelist"])
    box(63, 1, 36, 8.5, "Data", ["phantoms · CAMUS cache · uploads"])
    for y in (44.5, 27, 9.5):
        arrow(32.6, y, 35.4, y)
    for y in (46, 31, 17.5, 5.2):
        arrow(59.6, y, 62.4, y)
    ax.add_patch(FancyArrowPatch((81, 39.6), (81, 37.4), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))
    ax.text(82, 38.5, "masks → volumes", fontsize=6.2, color=INK2, va="center")
    fig.suptitle("Figure 1. EchoPlane3D architecture", x=0.01, y=0.98, ha="left", fontsize=9.5, weight="bold", color=INK)
    _save(fig, "fig1_architecture_overview")
    plt.close(fig)


def fig2_bland_altman(vol_rows: list[dict], track_rows: list[dict]):
    plt = _style()
    panels = [
        ("Independent ED + ES frames · snapped click", [r for r in vol_rows if r["strategy"] == "snapped_click"]),
        ("Independent ED + ES frames · automatic", [r for r in vol_rows if r["strategy"] == "automatic"]),
        ("Full-cycle tracking · snapped click", [r for r in track_rows if r["prompt"] == "snapped_click"]),
        ("Full-cycle tracking · automatic", [r for r in track_rows if r["prompt"] == "automatic"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharex=True, sharey=True)
    lim = 60.0
    for ax, (title, rows) in zip(axes.flat, panels):
        est = np.array([r["est_lvef_bp"] for r in rows], float)
        ref = np.array([r["ref_lvef_bp"] for r in rows], float)
        ok = np.isfinite(est) & np.isfinite(ref)
        mean, diff = (est[ok] + ref[ok]) / 2, est[ok] - ref[ok]
        bias, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
        inside = np.abs(diff) <= lim
        ax.scatter(mean[inside], diff[inside], s=18, color=SERIES[0], edgecolor="white", linewidth=0.6, zorder=3)
        out = ~inside                                    # off-scale points: drawn at the edge as open triangles
        if out.any():
            ax.scatter(np.clip(mean[out], -10, 110), np.sign(diff[out]) * lim * 0.97, s=22, marker="^", facecolor="white",
                       edgecolor=SERIES[1], linewidth=1.1, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axhline(bias, color=INK, lw=1.0)
        for v in (bias - 1.96 * sd, bias + 1.96 * sd):
            if abs(v) < lim:
                ax.axhline(v, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(0.98, 0.97, f"bias {bias:+.1f}, LoA {bias - 1.96 * sd:+.0f} to {bias + 1.96 * sd:+.0f}\n"
                f"n = {ok.sum()}" + (f", {int(out.sum())} off scale" if out.any() else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK)
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.set_ylim(-lim, lim)
    for ax in axes[1]:
        ax.set_xlabel("Mean of estimate and reference LVEF (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimate − reference (points)")
    fig.suptitle("Figure 2. Biplane LVEF vs CAMUS reference masks, 30 held-out patients",
                 x=0.01, ha="left", fontsize=9.5, weight="bold", color=INK)
    fig.text(0.01, -0.01, "△ estimate − reference beyond ±60 points (plotted at the edge). Reference = CAMUS expert masks "
             "measured with the same Simpson code; not clinical ground truth.", fontsize=6.8, color=INK2)
    fig.tight_layout()
    _save(fig, "fig2_bland_altman_lvef")
    plt.close(fig)


def fig3_prompt_ablation(frame_rows: list[dict]):
    plt = _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1.25, 1.25, 0.9]})

    def bars(ax, strategies, metric, ylabel, colors):
        n = len(strategies)
        width = 0.8 / n
        for k, s in enumerate(strategies):
            meds, lo, hi = [], [], []
            for q in QUALITIES:
                v = np.array([r[metric] for r in frame_rows if r["strategy"] == s and r["sequence_quality"] == q], float)
                v = v[np.isfinite(v)]
                m = np.median(v) if len(v) else np.nan
                meds.append(m)
                lo.append(m - np.percentile(v, 25) if len(v) else 0)
                hi.append(np.percentile(v, 75) - m if len(v) else 0)
            x = np.arange(len(QUALITIES)) + (k - (n - 1) / 2) * width
            ax.bar(x, meds, width=width * 0.9, color=colors[k], label=STRAT_LABEL[s], edgecolor="white", linewidth=0.8)
            ax.errorbar(x, meds, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=0)
        ax.set_xticks(np.arange(len(QUALITIES)), QUALITIES)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)

    bars(axes[0], MAIN_STRATEGIES, "dice", "Dice (median, IQR)", SERIES)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("a  Overlap with reference", loc="left")
    bars(axes[1], MAIN_STRATEGIES, "hd95_mm", "HD95, mm (median, IQR)", SERIES)
    axes[1].set_title("b  Boundary distance", loc="left")
    bars(axes[2], ("naive_wall_click", "snapped_wall_click"), "dice", "Dice (median, IQR)", [SERIES[0], SERIES[1]])
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("c  Click on the wall", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.36, -0.08))
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(h, ["Naive", "Snapped"], loc="lower center", ncol=2, bbox_to_anchor=(0.86, -0.08))
    fig.suptitle("Figure 3. Prompting strategy by CAMUS image quality (ED + ES, A4C + A2C)",
                 x=0.01, ha="left", fontsize=9.5, weight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "fig3_prompt_ablation_by_quality")
    plt.close(fig)


def fig4_foreshortening(sweep: list[dict]):
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for k, (case, label) in enumerate((("phantom_normal", "Normal (true LVEF 60 %)"), ("phantom_hfref", "HFrEF (true LVEF 32 %)"))):
        rows = [r for r in sweep if r["case"] == case]
        t = [r["tilt_deg"] for r in rows]
        axes[0].plot(t, [r["lvef_error_pts"] for r in rows], color=SERIES[k], lw=2, label=label)
        num = [r for r in rows if "numerical_lvef_pct" in r]
        axes[0].plot([r["tilt_deg"] for r in num], [r["numerical_lvef_pct"] - r["true_lvef_pct"] for r in num], "o",
                     ms=4.5, mfc="white", mec=SERIES[k], mew=1.4)
        axes[0].annotate(label.split(" (")[0], (t[-1], rows[-1]["lvef_error_pts"]), xytext=(4, 0), textcoords="offset points",
                         fontsize=7.4, color=INK, va="center")
        axes[1].plot(t, [r["edv_error_pct"] for r in rows], color=SERIES[k], lw=2, label=f"{label.split(' (')[0]} EDV")
        axes[1].plot(t, [r["esv_error_pct"] for r in rows], color=SERIES[k], lw=2, ls=(0, (4, 2)), label=f"{label.split(' (')[0]} ESV")
    axes[0].set_title("a  LVEF over-estimation", loc="left")
    axes[0].set_xlabel("Apical plane tilt (°)")
    axes[0].set_ylabel("Foreshortened − true LVEF (points)")
    axes[0].legend(loc="upper left")
    axes[0].text(0.98, 0.04, "line: closed form · ○ numerical plane slice", transform=axes[0].transAxes, ha="right",
                 fontsize=7, color=INK2)
    axes[0].annotate("PCA long axis flips to the\nannulus (section wider than tall)", xy=(30, 1.62), xytext=(19.5, 0.35),
                     fontsize=6.8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    axes[0].set_xlim(0, 35)
    axes[1].set_title("b  Volume under-estimation", loc="left")
    axes[1].set_xlabel("Apical plane tilt (°)")
    axes[1].set_ylabel("Error vs true volume (%)")
    axes[1].legend(loc="lower left", ncol=1)
    fig.suptitle("Figure 4. Sensitivity of 20-disk Simpson volumetry to apical foreshortening (half-ellipsoid model)",
                 x=0.01, ha="left", fontsize=9.5, weight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "fig4_foreshortening_sensitivity")
    plt.close(fig)


# =========================================================================== #
# provenance
# =========================================================================== #
def manifest(args, cohort, timings: dict, gpu: bool) -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    ckpt_sha = None                                   # absent on CPU-only CI runners
    if config.SAM2_CHECKPOINT.is_file():
        h = hashlib.sha256()
        with config.SAM2_CHECKPOINT.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ckpt_sha = h.hexdigest()
    info = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sam2_repo_commit": run(["git", "-C", str(config.SAM2_REPO), "rev-parse", "HEAD"]),
        "sam2_checkpoint": config.SAM2_CHECKPOINT.name,
        "sam2_checkpoint_sha256": ckpt_sha,
        "sam2_config": config.SAM2_CONFIG,
        "autocast_dtype": config.AUTOCAST_DTYPE,
        "vram_budget_mb": config.VRAM_BUDGET_MB,
        "camus_source": camus.SOURCE,
        "camus_citation": camus.CITATION,
        "camus_held_out_cohort": cohort,
        "held_out_rule": f"first {args.n_per_quality} patients per A4C ImageQuality from patient{FIRST_HELD_OUT:04d}, "
                         "excluding the demo cohort; none of them was viewed while developing the prompting heuristics",
        "reference_definition": "CAMUS expert LV masks (label 1) processed by the same 20-disk Simpson code; "
                                "dataset-reported EF is compared separately and is NOT ground truth for the masks",
        "timings_sec": timings,
        "ollama_reachable": False,
    }
    if gpu:
        import torch

        info |= {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                 "nvidia_driver": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])}
    try:
        from backend.core import guide

        info["ollama_reachable"] = bool(guide.get_ollama().available())
    except Exception:
        pass
    return info


# =========================================================================== #
# main
# =========================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-gpu", action="store_true", help="CPU-only: Experiment A maths and figures 1 and 4")
    ap.add_argument("--n-per-quality", type=int, default=10)
    ap.add_argument("--skip-showcase", action="store_true", help="skip full-cycle tracking of the demo cohort")
    args = ap.parse_args(argv)
    np.random.seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    # ---------------- Experiment A ----------------
    t = time.perf_counter()
    table1, sweep = experiment_a_math()
    write_csv(RESULTS / "table1_phantom_verification.csv", table1)
    write_csv(RESULTS / "table1b_foreshortening_sweep.csv", sweep)
    fig1_architecture()
    fig4_foreshortening(sweep)
    timings["experiment_a_math"] = round(time.perf_counter() - t, 1)
    cohort: dict = {}
    if args.skip_gpu:
        (RESULTS / "benchmark_manifest_cpu_only.json").write_text(json.dumps(manifest(args, cohort, timings, False), indent=2))
        banner("done", 100, "CPU-only run complete")
        return 0

    import torch

    from backend.core import sam2_tracker as trk
    from scripts.benchmark_tracker import SmiPoller

    telemetry: list[dict] = []
    t = time.perf_counter()
    smi_idle = SmiPoller.read()
    trk.get_predictor()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    telemetry.append({"run": "model load", "sample": config.SAM2_CHECKPOINT.name, "inference_time_sec": time.perf_counter() - t,
                      "vram_allocated_after_mb": torch.cuda.memory_allocated() / 2**20, "nvidia_smi_idle_mib": smi_idle,
                      "gpu_total_mb": total / 2**20, "gpu_free_after_load_mb": free / 2**20, **proc_mem_mb()})
    rows, tel = experiment_a_tracking(trk, SmiPoller)
    write_csv(RESULTS / "table1c_phantom_tracking.csv", rows)
    telemetry += tel
    timings["experiment_a_tracking"] = round(time.perf_counter() - t, 1)

    # ---------------- Experiment B ----------------
    t = time.perf_counter()
    cohort = select_cohort(args.n_per_quality)
    timings["cohort_selection_and_download"] = round(time.perf_counter() - t, 1)
    banner("B (CAMUS)", 20, f"{sum(len(v) for v in cohort.values())} patients x 2 views x ED/ES x {len(STRATEGIES)} strategies")
    t = time.perf_counter()
    frame_rows, vol_rows, embed_ms, decode_ms = experiment_b_frames(trk, cohort)
    timings["experiment_b_frames"] = round(time.perf_counter() - t, 1)
    write_csv(RESULTS / "camus_patient_metrics.csv", frame_rows)
    write_csv(RESULTS / "camus_patient_volumes.csv", vol_rows)
    write_csv(RESULTS / "table2_camus_validation.csv", summarise_frames(frame_rows))
    write_csv(RESULTS / "table2b_camus_agreement.csv", summarise_agreement(vol_rows))
    telemetry.append({"run": "single-frame image encoder (set_image)", "sample": "CAMUS ED/ES frames", "n_frames": len(embed_ms),
                      "ms_per_frame": float(np.mean(embed_ms)), "ms_p95": float(np.percentile(embed_ms, 95))})
    telemetry.append({"run": "single-frame prompt decode + candidate selection", "sample": "CAMUS ED/ES frames",
                      "n_frames": len(decode_ms), "ms_per_frame": float(np.mean(decode_ms)), "ms_p95": float(np.percentile(decode_ms, 95))})
    fig3_prompt_ablation(frame_rows)

    banner("B (CAMUS)", 55, "full-cycle tracking of the 30 held-out patients (the workbench's real pipeline)")
    t = time.perf_counter()
    seq_rows, track_rows, track_agree, tel = experiment_b_heldout_tracking(trk, SmiPoller, cohort)
    write_csv(RESULTS / "table2d_heldout_tracking.csv", seq_rows)
    write_csv(RESULTS / "table2d_heldout_tracking_patients.csv", track_rows)
    write_csv(RESULTS / "table2e_heldout_tracking_agreement.csv", track_agree)
    telemetry += tel
    timings["experiment_b_heldout_tracking"] = round(time.perf_counter() - t, 1)
    fig2_bland_altman(vol_rows, track_rows)

    if not args.skip_showcase:
        banner("B (CAMUS)", 80, "full-cycle tracking of the 8-patient demo cohort (tuned-on data)")
        t = time.perf_counter()
        rows, tel = experiment_b_showcase(trk, SmiPoller)
        write_csv(RESULTS / "table2c_showcase_tracking.csv", rows)
        telemetry += tel
        timings["experiment_b_showcase"] = round(time.perf_counter() - t, 1)

    # ---------------- Experiment C ----------------
    banner("C (telemetry)", 90, "writing hardware telemetry")
    write_csv(RESULTS / "table3_hardware_telemetry.csv", telemetry)
    trk.unload_predictor()
    timings["total"] = round(time.perf_counter() - t_all, 1)
    (RESULTS / "benchmark_manifest.json").write_text(json.dumps(manifest(args, cohort, timings, True), indent=2))
    print("  wrote results/benchmark_manifest.json")
    banner("done", 100, f"all experiments finished in {timings['total']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
