"""Left-ventricular volumetry: 20-disk Modified Simpson's Rule + apical
foreshortening simulation.

Method of disks (single plane)
------------------------------
1.  PCA of the mask's pixel coordinates (in mm) gives the long-axis
    eigenvector ``e1`` and the orthogonal short-axis direction ``e2``.
2.  The cavity is cut into ``n_disks`` slabs of equal height ``h = L / n``
    perpendicular to ``e1``.
3.  Each slab's diameter ``d_i`` is its area-equivalent chord
    (slab area / h) which is robust to ragged segmentation edges.
4.  ``V = sum_i  pi/4 * d_i^2 * h``   (mm^3 -> mL by / 1000).

Foreshortening model
--------------------
The LV cavity is modelled as a half prolate ellipsoid (flat base at the mitral
annulus, apex at distance ``a``; base radius ``b``):

    x^2/b^2 + y^2/b^2 + z^2/a^2 <= 1 ,  z >= 0        V = 2/3 * pi * b^2 * a

A correctly aligned apical plane contains the z axis.  A foreshortened plane
still passes through the centre of the mitral annulus but is tilted by
``theta`` off the true long axis (probe one rib-space too high), so it never
reaches the true apex.  Its intersection with the cavity is a half ellipse
with the *same* base radius ``b`` but a shortened apparent length

    a' = 1 / sqrt( cos^2(theta)/a^2 + sin^2(theta)/b^2 )

Simpson's rule applied to that view therefore reports ``V' = V * a'/a``.
Because the end-systolic cavity is more elongated (larger a/b) than the
end-diastolic one, ESV is under-estimated more than EDV, so LVEF is
over-estimated — the classic clinical signature of apical foreshortening.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

__all__ = [
    "compute_simpsons_20_disks",
    "compute_biplane_simpson",
    "biplane_volume_from_disks",
    "simulate_apical_foreshortening",
    "foreshortening_sweep",
    "lv_dimensions_from_volume",
    "foreshortened_length",
    "synth_ellipse_mask",
    "prolate_ellipsoid_volume_ml",
]

# Shape priors for the half-ellipsoid LV model -------------------------------
ED_LENGTH_TO_RADIUS = 3.5      # a/b at end-diastole (L ~ 89 mm, D ~ 51 mm for 120 mL)
LONGITUDINAL_EXPONENT = 0.25   # a(t) = a_ed * (V/EDV)^0.25 ; b(t) = b_ed * (V/EDV)^0.375


def _spacing(mm_per_pixel) -> tuple[float, float]:
    """Return (sx, sy) in mm; accepts a scalar or an (x, y) pair."""
    if isinstance(mm_per_pixel, (int, float, np.floating, np.integer)):
        s = float(mm_per_pixel)
        return s, s
    sx, sy = mm_per_pixel
    return float(sx), float(sy)


def _empty_result(n_disks: int) -> dict:
    return {
        "volume_ml": 0.0,
        "area_mm2": 0.0,
        "length_mm": 0.0,
        "disk_height_mm": 0.0,
        "diameters_mm": [0.0] * n_disks,
        "long_axis": None,
        "chords": [],
        "centroid": None,
        "axis_angle_deg": 0.0,
        "n_disks": n_disks,
    }


def compute_simpsons_20_disks(mask, mm_per_pixel=1.0, n_disks: int = 20) -> dict:
    """Single-plane Modified Simpson's Rule on a binary mask.

    Parameters
    ----------
    mask : 2-D array-like, truthy inside the LV cavity (rows = y, cols = x).
    mm_per_pixel : float or (mm_x, mm_y).
    n_disks : number of disks (20 by convention).

    Returns
    -------
    dict with ``volume_ml`` plus everything the UI needs to draw the result in
    *pixel* coordinates: ``long_axis`` ``[[x0, y0], [x1, y1]]`` (apex-side
    first, i.e. the endpoint nearer the top of the image) and ``chords`` — a
    list of ``n_disks`` segments ``[[xa, ya], [xb, yb]]``.
    """
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError("mask must be 2-D")
    ys, xs = np.nonzero(m)
    if xs.size < max(3 * n_disks, 16):
        return _empty_result(n_disks)

    sx, sy = _spacing(mm_per_pixel)
    pts = np.stack([xs * sx, ys * sy], axis=1).astype(np.float64)  # mm
    centroid = pts.mean(axis=0)
    centred = pts - centroid

    # --- PCA: long axis = eigenvector with the largest eigenvalue ----------
    cov = centred.T @ centred / centred.shape[0]
    evals, evecs = np.linalg.eigh(cov)  # ascending
    e1 = evecs[:, 1]
    if e1[1] < 0 or (e1[1] == 0 and e1[0] < 0):  # orient "down the screen"
        e1 = -e1
    e2 = np.array([-e1[1], e1[0]])

    t = centred @ e1
    s = centred @ e2

    # Pixels have a footprint: pad the extent by half a pixel along e1.
    half_px = 0.5 * (abs(e1[0]) * sx + abs(e1[1]) * sy)
    t_min = float(t.min() - half_px)
    t_max = float(t.max() + half_px)
    length = t_max - t_min
    h = length / n_disks

    # Anti-aliased binning: every pixel spreads its area over the slabs its
    # footprint [t - half_px, t + half_px] overlaps.  Hard (nearest-slab)
    # binning aliases against the pixel grid and makes diameters jitter ±6 %.
    lo = (t - half_px - t_min) / h
    hi = (t + half_px - t_min) / h
    width = hi - lo
    first = np.floor(lo).astype(np.int64)
    span = int(math.ceil(float(width.max()))) + 1
    counts = np.zeros(n_disks)
    s_sum = np.zeros(n_disks)
    for k in range(span):
        b = first + k
        frac = np.clip(np.minimum(hi, b + 1) - np.maximum(lo, b), 0.0, None) / width
        ok = (frac > 0) & (b >= 0) & (b < n_disks)
        if not ok.any():
            continue
        counts += np.bincount(b[ok], weights=frac[ok], minlength=n_disks)
        s_sum += np.bincount(b[ok], weights=frac[ok] * s[ok], minlength=n_disks)
    slab_area = counts * sx * sy                  # mm^2
    diam = slab_area / h                          # area-equivalent chord, mm
    volume_mm3 = float(np.sum(np.pi / 4.0 * diam**2 * h))

    s_mid = np.divide(s_sum, counts, out=np.zeros(n_disks), where=counts > 1e-9)

    def to_px(p_mm: np.ndarray) -> list[float]:
        return [round(float(p_mm[0] / sx), 2), round(float(p_mm[1] / sy), 2)]

    chords = []
    for i in range(n_disks):
        t_c = t_min + (i + 0.5) * h
        base = centroid + t_c * e1 + s_mid[i] * e2
        half = 0.5 * diam[i] * e2
        chords.append([to_px(base - half), to_px(base + half)])

    long_axis = [to_px(centroid + t_min * e1), to_px(centroid + t_max * e1)]

    return {
        "volume_ml": volume_mm3 / 1000.0,
        "area_mm2": float(xs.size * sx * sy),
        "length_mm": float(length),
        "disk_height_mm": float(h),
        "diameters_mm": [float(d) for d in diam],
        "long_axis": long_axis,
        "chords": chords,
        "centroid": to_px(centroid),
        "axis_angle_deg": float(math.degrees(math.atan2(e1[1], e1[0]))),
        "n_disks": n_disks,
    }


# --------------------------------------------------------------------------- #
# Biplane method of disks (Lang et al., ASE/EACVI 2015)
# --------------------------------------------------------------------------- #
def biplane_volume_from_disks(
    diameters_a4c_mm: Sequence[float], length_a4c_mm: float,
    diameters_a2c_mm: Sequence[float], length_a2c_mm: float,
) -> float:
    """Biplane Simpson volume in mL from per-disk diameters of two orthogonal
    apical views (disk i of each view is at the same fractional level, apex
    first):

        V = pi/4 * sum_i a_i * b_i * max(L_A4C, L_A2C) / n
    """
    a = np.asarray(diameters_a4c_mm, dtype=np.float64)
    b = np.asarray(diameters_a2c_mm, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("both views need the same, non-zero number of disks")
    length = max(float(length_a4c_mm), float(length_a2c_mm))
    return float(np.pi / 4.0 * np.sum(a * b) * length / a.size / 1000.0)


def compute_biplane_simpson(mask_a4c, mask_a2c, mm_per_pixel_a4c=1.0, mm_per_pixel_a2c=None,
                            n_disks: int = 20) -> dict:
    """True biplane method of disks from an A4C and an A2C cavity mask.

    Also returns both single-plane volumes and the long-axis length mismatch
    between the views (the guideline tolerance is 10 %; a larger mismatch
    suggests one view is foreshortened).
    """
    if mm_per_pixel_a2c is None:
        mm_per_pixel_a2c = mm_per_pixel_a4c
    r4 = compute_simpsons_20_disks(mask_a4c, mm_per_pixel_a4c, n_disks)
    r2 = compute_simpsons_20_disks(mask_a2c, mm_per_pixel_a2c, n_disks)
    if r4["volume_ml"] <= 0 or r2["volume_ml"] <= 0:
        return {"volume_ml": 0.0, "length_mm": 0.0, "length_a4c_mm": r4["length_mm"], "length_a2c_mm": r2["length_mm"],
                "length_mismatch_pct": None, "single_plane_a4c_ml": r4["volume_ml"],
                "single_plane_a2c_ml": r2["volume_ml"], "n_disks": n_disks}
    volume = biplane_volume_from_disks(r4["diameters_mm"], r4["length_mm"], r2["diameters_mm"], r2["length_mm"])
    longest = max(r4["length_mm"], r2["length_mm"])
    return {
        "volume_ml": volume,
        "length_mm": longest,
        "length_a4c_mm": r4["length_mm"],
        "length_a2c_mm": r2["length_mm"],
        "length_mismatch_pct": abs(r4["length_mm"] - r2["length_mm"]) / longest * 100.0,
        "single_plane_a4c_ml": r4["volume_ml"],
        "single_plane_a2c_ml": r2["volume_ml"],
        "n_disks": n_disks,
    }


# --------------------------------------------------------------------------- #
# Half-ellipsoid LV model + foreshortening
# --------------------------------------------------------------------------- #
def prolate_ellipsoid_volume_ml(radius_mm: float, half_length_mm: float) -> float:
    """Volume of a full prolate ellipsoid, 4/3*pi*R^2*(L/2), in mL."""
    return 4.0 / 3.0 * math.pi * radius_mm**2 * half_length_mm / 1000.0


def lv_dimensions_from_volume(
    volume_ml: float,
    edv_ml: float,
    ed_length_to_radius: float = ED_LENGTH_TO_RADIUS,
    longitudinal_exponent: float = LONGITUDINAL_EXPONENT,
) -> tuple[float, float]:
    """Return (a, b) in mm — apex-to-base length and base radius — of the
    half-ellipsoid cavity holding ``volume_ml``, for a ventricle whose
    end-diastolic volume is ``edv_ml``.

    End-diastole fixes the shape (a_ed = ratio * b_ed).  Contraction splits
    between long-axis shortening ``(V/EDV)^k`` and radial shortening
    ``(V/EDV)^((1-k)/2)`` so that b^2 * a always tracks the volume exactly.
    With k = 0.25, an EF of 60 % gives ~20 % longitudinal and ~29 % radial
    shortening — close to normal strain values.
    """
    if edv_ml <= 0:
        raise ValueError("edv_ml must be > 0")
    if volume_ml <= 0:
        raise ValueError("volume_ml must be > 0")
    b_ed = (edv_ml * 1000.0 / (2.0 / 3.0 * math.pi * ed_length_to_radius)) ** (1.0 / 3.0)
    a_ed = ed_length_to_radius * b_ed
    ratio = volume_ml / edv_ml
    a = a_ed * ratio**longitudinal_exponent
    b = b_ed * ratio ** ((1.0 - longitudinal_exponent) / 2.0)
    return a, b


def foreshortened_length(a_mm: float, b_mm: float, tilt_deg: float) -> float:
    """Apparent apex-to-base length when the imaging plane (through the base
    centre) is tilted ``tilt_deg`` away from the true long axis."""
    th = math.radians(tilt_deg)
    return 1.0 / math.sqrt(math.cos(th) ** 2 / a_mm**2 + math.sin(th) ** 2 / b_mm**2)


def simulate_apical_foreshortening(
    true_edv: float,
    true_esv: float,
    tilt_deg: float,
    ed_length_to_radius: float = ED_LENGTH_TO_RADIUS,
    longitudinal_exponent: float = LONGITUDINAL_EXPONENT,
) -> dict:
    """Predict what Simpson's rule would report from a foreshortened apical view.

    Parameters
    ----------
    true_edv, true_esv : true end-diastolic / end-systolic volumes in mL.
    tilt_deg : off-axis tilt of the imaging plane (0 = true apex in plane).
    """
    true_edv = float(true_edv)
    true_esv = float(true_esv)
    tilt_deg = float(tilt_deg)
    if not (math.isfinite(true_edv) and math.isfinite(true_esv) and math.isfinite(tilt_deg)):
        raise ValueError("inputs must be finite numbers")
    if true_edv <= 0:
        raise ValueError("true_edv must be > 0")
    if true_esv <= 0 or true_esv > true_edv:
        raise ValueError("true_esv must satisfy 0 < true_esv <= true_edv")
    if not (0.0 <= tilt_deg < 90.0):
        raise ValueError("tilt_deg must be in [0, 90)")

    a_ed, b_ed = lv_dimensions_from_volume(true_edv, true_edv, ed_length_to_radius, longitudinal_exponent)
    a_es, b_es = lv_dimensions_from_volume(true_esv, true_edv, ed_length_to_radius, longitudinal_exponent)

    a_ed_f = foreshortened_length(a_ed, b_ed, tilt_deg)
    a_es_f = foreshortened_length(a_es, b_es, tilt_deg)

    edv_f = true_edv * a_ed_f / a_ed
    esv_f = true_esv * a_es_f / a_es
    ef_true = (true_edv - true_esv) / true_edv * 100.0
    ef_f = (edv_f - esv_f) / edv_f * 100.0

    return {
        "tilt_deg": tilt_deg,
        "true": {
            "edv_ml": true_edv,
            "esv_ml": true_esv,
            "lvef_pct": ef_true,
            "ed_length_mm": a_ed,
            "es_length_mm": a_es,
            "ed_diameter_mm": 2 * b_ed,
            "es_diameter_mm": 2 * b_es,
        },
        "foreshortened": {
            "edv_ml": edv_f,
            "esv_ml": esv_f,
            "lvef_pct": ef_f,
            "ed_length_mm": a_ed_f,
            "es_length_mm": a_es_f,
            "ed_diameter_mm": 2 * b_ed,
            "es_diameter_mm": 2 * b_es,
        },
        "errors": {
            "edv_pct": (edv_f - true_edv) / true_edv * 100.0,
            "esv_pct": (esv_f - true_esv) / true_esv * 100.0,
            "lvef_abs_pts": ef_f - ef_true,
            "ed_length_pct": (a_ed_f - a_ed) / a_ed * 100.0,
            "es_length_pct": (a_es_f - a_es) / a_es * 100.0,
        },
        "model": {
            "shape": "half prolate ellipsoid, plane pivots at mitral-annulus centre",
            "ed_length_to_radius": ed_length_to_radius,
            "longitudinal_exponent": longitudinal_exponent,
        },
    }


def foreshorten_volume(
    volume_ml: float,
    edv_ml: float,
    tilt_deg: float,
    ed_length_to_radius: float = ED_LENGTH_TO_RADIUS,
    longitudinal_exponent: float = LONGITUDINAL_EXPONENT,
) -> float:
    """Volume Simpson's rule would report for a cavity of ``volume_ml`` (in a
    ventricle with end-diastolic volume ``edv_ml``) seen through a plane
    tilted ``tilt_deg`` off axis.  Used to foreshorten a whole V(t) curve."""
    if volume_ml <= 0:
        return 0.0
    a, b = lv_dimensions_from_volume(volume_ml, edv_ml, ed_length_to_radius, longitudinal_exponent)
    return volume_ml * foreshortened_length(a, b, tilt_deg) / a


def foreshortening_sweep(
    true_edv: float, true_esv: float, max_tilt_deg: float = 30.0, step_deg: float = 1.0
) -> dict:
    """Tabulate the foreshortening model over 0..max_tilt for plotting."""
    tilts = np.arange(0.0, max_tilt_deg + 1e-9, step_deg)
    rows = [simulate_apical_foreshortening(true_edv, true_esv, float(t)) for t in tilts]
    return {
        "tilt_deg": [float(t) for t in tilts],
        "edv_ml": [r["foreshortened"]["edv_ml"] for r in rows],
        "esv_ml": [r["foreshortened"]["esv_ml"] for r in rows],
        "lvef_pct": [r["foreshortened"]["lvef_pct"] for r in rows],
    }


# --------------------------------------------------------------------------- #
# Synthetic shapes (unit tests + phantom generator)
# --------------------------------------------------------------------------- #
def synth_ellipse_mask(
    semi_long_mm: float,
    semi_short_mm: float,
    mm_per_pixel: float,
    angle_deg: float = 0.0,
    half: bool = False,
    margin_px: int = 12,
    centre_offset_px: Sequence[float] = (0.0, 0.0),
) -> np.ndarray:
    """Rasterise the long-axis cross-section of a prolate ellipsoid.

    ``angle_deg`` is the long-axis direction measured from the +y (down)
    image axis.  ``half=True`` keeps only the apical half (flat base through
    the ellipse centre) — the cross-section of the half-ellipsoid LV model.
    A pixel is inside when its centre is inside the shape.
    """
    r = math.hypot(semi_long_mm, semi_short_mm) / mm_per_pixel
    size = int(2 * (r + margin_px)) + 1
    cy = cx = size / 2.0
    cx += centre_offset_px[0]
    cy += centre_offset_px[1]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = (xx - cx) * mm_per_pixel
    dy = (yy - cy) * mm_per_pixel
    th = math.radians(angle_deg)
    # long-axis unit vector (points down the screen when angle = 0)
    ux, uy = math.sin(th), math.cos(th)
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    inside = (along / semi_long_mm) ** 2 + (across / semi_short_mm) ** 2 <= 1.0
    if half:
        inside &= along <= 0.0  # apex toward the top of the image
    return inside
