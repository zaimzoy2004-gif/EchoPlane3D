#!/usr/bin/env python3
"""Procedurally build an anatomically oriented multi-structure heart.

Outputs
-------
frontend/models/heart_chambers.glb   nine named, watertight meshes, units = mm
frontend/models/heart_planes.json    probe pose + cut plane for A4C / A2C / PLAX / PSAX in the same frame

Method
------
Every structure is an implicit surface: a smooth union / difference of
ellipsoids and spline tubes evaluated on a 1.5 mm voxel grid and extracted
with marching cubes, then lightly Taubin-smoothed.  This is a *schematic
teaching model*, not a patient-derived anatomy: proportions follow textbook
adult values, and the LV cavity is deliberately kept an exact half prolate
ellipsoid so that the 3-D cut agrees with the closed-form foreshortening
model in ``backend/core/simpsons_math.py``.

    LV     half-ellipsoid cavity with mitral-inlet and aortic-outlet (LVOT) stubs
    IVS    interventricular septum, a curved slab between LV and RV
    RV     crescent wrapped round the septum + infundibulum (RVOT)
    LA     posterior-superior chamber with a left-atrial-appendage protrusion
    RA     right lateral atrium
    Aorta  root, ascending aorta, arch with three branch stubs, descending aorta
    PA     pulmonary trunk from the RVOT, bifurcating under the aortic arch
    SVC / IVC  venae cavae entering the RA along the body's vertical axis

Heart-local frame (mm)
----------------------
* +y is the LV long axis pointing from apex to base; the mitral-annulus centre
  is the origin and the LV apex sits at y = -LV_LENGTH.
* Angles phi go round the long axis, d(phi) = (cos phi, 0, -sin phi), laid out
  like the six basal wall segments seen from the apex:
      0 inferoseptal | 60 anteroseptal | 120 anterior
    180 anterolateral | 240 inferolateral | 300 inferior
  so the apical planes are the classic 60-degree rotations:
      A4C contains d(0)/d(180), PLAX (= A3C) d(60)/d(240), A2C d(120)/d(300).

World frame (what the GLB stores, patient-anatomical, three.js friendly)
-----------------------------------------------------------------------
+X = patient left, +Y = superior, +Z = anterior.  The apex points left,
inferior and anterior; the RV is the most anterior chamber.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from skimage import measure
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.core import config  # noqa: E402
from backend.core.simpsons_math import foreshortened_length  # noqa: E402

# ----------------------------- anatomy (mm) -------------------------------- #
LV_LENGTH = 88.0      # mitral annulus -> apex
LV_RADIUS = 26.0
LV_CAP = 10.0         # shallow basal dome closing the cavity
VOXEL_MM = 1.5
MAX_GLB_MB = 6.0

COLORS = {
    "LV": "#ef4444",
    "IVS": "#b45309",
    "RV": "#3b82f6",
    "LA": "#f9a8d4",
    "RA": "#93c5fd",
    "Aorta": "#f59e0b",
    "PA": "#8b5cf6",
    "SVC": "#22d3ee",
    "IVC": "#0891b2",
}
DESCRIPTIONS = {
    "LV": "Left ventricle (cavity): conical, thick-walled systemic pump",
    "IVS": "Interventricular septum between LV and RV",
    "RV": "Right ventricle: crescent wrapped round the septum, with the infundibulum (RVOT)",
    "LA": "Left atrium with the left atrial appendage (LAA)",
    "RA": "Right atrium",
    "Aorta": "Aortic root, ascending aorta, arch with three branch stubs, descending thoracic aorta",
    "PA": "Main pulmonary artery and its bifurcation under the aortic arch",
    "SVC": "Superior vena cava entering the right atrium from above",
    "IVC": "Inferior vena cava entering the right atrium from below",
}

APEX_DIR_WORLD = np.array([0.62, -0.60, 0.50])   # left, inferior, anterior
RV_DIR_WORLD = np.array([-0.55, 0.00, 0.83])     # right, anterior
Y = np.array([0.0, 1.0, 0.0])


def d(phi_deg: float) -> np.ndarray:
    p = math.radians(phi_deg)
    return np.array([math.cos(p), 0.0, -math.sin(p)])


def at(phi_deg: float, rho: float, y: float) -> np.ndarray:
    """Point at angle phi round the long axis, rho mm from it, height y."""
    return rho * d(phi_deg) + np.array([0.0, y, 0.0])


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# --------------------------- local -> world -------------------------------- #
def world_rotation() -> np.ndarray:
    wy = -unit(APEX_DIR_WORLD)                       # local +y (apex->base)
    wd30 = unit(RV_DIR_WORLD - np.dot(RV_DIR_WORLD, wy) * wy)
    wd120 = np.cross(wy, wd30)                       # y x d(phi) = d(phi + 90)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    wx = c * wd30 - s * wd120                        # d(0)
    wz = -s * wd30 - c * wd120                       # (0,0,1) = d(270)
    rot = np.stack([wx, wy, wz], axis=1)
    assert abs(np.linalg.det(rot) - 1.0) < 1e-9, "rotation must be proper"
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    return rot


ROT = world_rotation()
UP_LOCAL = ROT.T @ np.array([0.0, 1.0, 0.0])        # the body's superior direction in heart-local coordinates


# ----------------------- implicit-surface primitives ------------------------ #
def smin(a, b, k):
    """Polynomial smooth minimum (smooth union), blend radius k mm."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def f_bullet(P, centre, radial, tangential, down, up, phi_deg):
    """Ellipsoid with different semi-axes below / above its equator; ``radial``
    lies along d(phi), ``tangential`` along d(phi + 90).  Negative inside."""
    q = P - np.asarray(centre, dtype=np.float64)
    u, v, w = q @ d(phi_deg), q[:, 1], q @ d(phi_deg + 90)
    ry = np.where(v < 0, down, up)
    k = np.sqrt((u / radial) ** 2 + (v / ry) ** 2 + (w / tangential) ** 2)
    return (k - 1.0) * min(radial, tangential, up, down)


def f_lv_envelope(P, delta=0.0):
    return f_bullet(P, (0, 0, 0), LV_RADIUS + delta, LV_RADIUS + delta, LV_LENGTH + delta, LV_CAP + delta, 0)


def f_tube(P, points, radii, samples=48):
    """Distance to a spline centre-line minus a radius that varies along it."""
    pts = np.asarray(points, dtype=np.float64)
    seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    seg /= seg[-1]
    if len(pts) >= 3:
        centres = CubicSpline(seg, pts)(np.linspace(0, 1, samples))
    else:
        centres = pts[0] + np.linspace(0, 1, samples)[:, None] * (pts[1] - pts[0])
    rad = np.interp(np.linspace(0, 1, samples), seg, radii)
    best = np.full(len(P), np.inf)
    for i in range(samples - 1):
        a, b = centres[i], centres[i + 1]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        dist = np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - (rad[i] + t * (rad[i + 1] - rad[i]))
        np.minimum(best, dist, out=best)
    return best


# --------------------------- the nine structures --------------------------- #
RA_CENTRE = np.array([41.0, 30.0, 3.0])
RV_PHI = 30.0
IVS_HALF_ANGLE = 58.0


def f_lv(P):
    f = f_lv_envelope(P)
    f = smin(f, f_tube(P, [at(240, 5, 3), at(240, 8, 11)], [13.0, 12.5]), 4.0)        # mitral inlet
    f = smin(f, f_tube(P, [at(60, 12, -6), at(60, 16.5, 6)], [10.5, 10.5]), 4.0)      # LV outflow tract
    return f


def f_ivs(P):
    shell = np.maximum(f_lv_envelope(P, 8.0), -f_lv_envelope(P, 1.2))
    phi = np.degrees(np.arctan2(-P[:, 2], P[:, 0]))
    dphi = (phi - RV_PHI + 180.0) % 360.0 - 180.0
    rho = np.hypot(P[:, 0], P[:, 2])
    wedge = (np.abs(dphi) - IVS_HALF_ANGLE) * np.pi / 180.0 * np.maximum(rho, 1.0)
    band = np.maximum(P[:, 1] - 2.0, -72.0 - P[:, 1])
    return smax(smax(shell, wedge, 2.0), band, 2.0)


def f_rv(P):
    body = f_bullet(P, 30 * d(RV_PHI) + np.array([0, -2.0, 0]), 22, 34, 62, 10, RV_PHI)
    rvot = f_tube(P, [at(45, 34, -8), at(62, 39, 8), at(75, 41, 22)], [13.0, 12.0, 11.5])
    return smax(smin(body, rvot, 6.0), -f_lv_envelope(P, 9.2), 2.5)


def f_la(P):
    body = f_bullet(P, at(240, 14, 31), 21, 20, 20, 20, 240)
    laa = f_tube(P, [at(195, 20, 35), at(155, 29, 34), at(130, 34, 31)], [8.0, 6.5, 5.0])
    return smin(body, laa, 5.0)


def f_ra(P):
    return f_bullet(P, RA_CENTRE, 22, 20, 21, 21, 0)


def f_aorta(P):
    trunk = f_tube(
        P,
        [at(60, 17, 7), at(60, 17.5, 28), at(60, 17, 55), at(60, 10, 77), at(240, 8, 88), at(240, 30, 85),
         at(240, 48, 66), at(240, 54, 38), at(240, 56, 8)],
        [12.5, 11.0, 10.5, 10.0, 9.5, 9.0, 9.0, 9.0, 9.0],
        samples=80,
    )
    f = trunk
    for start, r in ((at(60, 3, 84), 5.5), (at(240, 9, 88), 4.5), (at(240, 21, 87), 4.8)):   # supra-aortic branch stubs
        f = smin(f, f_tube(P, [start, start + 26.0 * UP_LOCAL], [r, r * 0.9]), 3.0)
    return f


def f_pa(P):
    trunk = f_tube(P, [at(75, 41, 23), at(88, 38, 42), at(105, 31, 58), at(128, 23, 68)], [11.0, 10.8, 10.4, 10.0])
    lpa = f_tube(P, [at(128, 23, 68), at(170, 34, 71), at(200, 48, 67)], [8.0, 7.5, 7.0])
    rpa = f_tube(P, [at(128, 23, 68), at(200, 10, 69.5), at(300, 22, 68), at(330, 40, 66)], [8.0, 7.5, 7.2, 7.0])
    return smin(smin(trunk, lpa, 4.0), rpa, 4.0)


def f_svc(P):
    return f_tube(P, [RA_CENTRE + 20.5 * UP_LOCAL, RA_CENTRE + 62.0 * UP_LOCAL], [8.5, 8.0])


def f_ivc(P):
    return f_tube(P, [RA_CENTRE - 19.0 * UP_LOCAL, RA_CENTRE - 46.0 * UP_LOCAL], [9.5, 9.5])


STRUCTURES = {
    "LV": (f_lv, 2), "IVS": (f_ivs, 6), "RV": (f_rv, 10), "LA": (f_la, 10), "RA": (f_ra, 6),
    "Aorta": (f_aorta, 10), "PA": (f_pa, 10), "SVC": (f_svc, 6), "IVC": (f_ivc, 6),
}                                                   # name -> (implicit function, Taubin iterations)

LABEL_LOCAL = {                                     # where each 3-D text label sits (kept apart on purpose)
    "LV": at(180, 4, -40), "IVS": at(RV_PHI, 30, -52), "RV": at(RV_PHI, 45, -16), "LA": at(240, 14, 31),
    "RA": RA_CENTRE, "Aorta": at(60, 17, 40), "PA": at(95, 35, 50),
    "SVC": RA_CENTRE + 46.0 * UP_LOCAL, "IVC": RA_CENTRE - 36.0 * UP_LOCAL,
}


def polygonise(name: str, fn, smooth_iters: int) -> trimesh.Trimesh:
    """Marching cubes over the implicit function (coarse pass finds the bounding box)."""
    lo, hi = np.array([-95.0, -110.0, -95.0]), np.array([95.0, 140.0, 95.0])
    coarse = 5.0
    gx, gy, gz = (np.arange(lo[i], hi[i] + coarse, coarse) for i in range(3))
    G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = G[fn(G) < coarse]
    if len(inside) == 0:
        raise RuntimeError(f"{name}: implicit surface is empty")
    lo2, hi2 = inside.min(axis=0) - 2 * coarse, inside.max(axis=0) + 2 * coarse
    axes = [np.arange(lo2[i], hi2[i] + VOXEL_MM, VOXEL_MM) for i in range(3)]
    shape = tuple(len(a) for a in axes)
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    vol = np.empty(len(P))
    for s in range(0, len(P), 400_000):                 # chunked to keep RAM flat
        vol[s:s + 400_000] = fn(P[s:s + 400_000])
    vol = vol.reshape(shape)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, spacing=(VOXEL_MM,) * 3)
    mesh = trimesh.Trimesh(vertices=verts + lo2, faces=faces, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    if smooth_iters:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def build_local(verbose: bool = True) -> dict[str, trimesh.Trimesh]:
    meshes = {}
    bar = tqdm(STRUCTURES.items(), desc="marching cubes", unit="structure", disable=not verbose)
    for i, (name, (fn, iters)) in enumerate(bar, 1):
        bar.set_postfix_str(name)
        meshes[name] = polygonise(name, fn, iters)
        if verbose:
            tqdm.write(f"  [{i}/{len(STRUCTURES)} structures | {i / len(STRUCTURES) * 100:5.1f}%] {name:6s} "
                       f"{len(meshes[name].vertices):6d} vertices, {meshes[name].volume / 1000:6.1f} mL")
    return meshes


# ------------------------------- planes ------------------------------------ #
def plane_defs_local() -> dict:
    apex_probe = np.array([0.0, -(LV_LENGTH + 8.0), 0.0])
    ps = 82 * d(60)                                   # parasternal window, anteroseptal side
    defs = {
        "A4C": dict(
            label="Apical 4-Chamber",
            origin=apex_probe, beam=Y, right=d(180), depth_mm=150.0, sector_deg=80.0, apical=True,
            expect=["LV", "IVS", "RV", "LA", "RA"],
            description="Probe at the apex, beam through the crux of the heart. All four chambers and the "
                        "septum; LV on screen right, RV on screen left.",
        ),
        "A2C": dict(
            label="Apical 2-Chamber",
            origin=apex_probe, beam=Y, right=d(120), depth_mm=150.0, sector_deg=80.0, apical=True,
            expect=["LV", "LA"],
            description="Rotate ~60 deg counter-clockwise from A4C. Only LV and LA (with the appendage); "
                        "anterior wall on screen right, inferior wall on screen left.",
        ),
        "PLAX": dict(
            label="Parasternal Long-Axis",
            origin=ps + np.array([0, -18.0, 0]), beam=-d(60), right=Y, depth_mm=150.0, sector_deg=80.0, apical=False,
            expect=["LV", "IVS", "RV", "LA", "Aorta"],
            description="Probe at the left sternal border, plane along the LV long axis. RVOT on top, then "
                        "septum, LV, aortic root and LA; apex off-screen to the left.",
        ),
        "PSAX": dict(
            label="Parasternal Short-Axis (mid-papillary)",
            origin=ps + np.array([0, -40.0, 0]), beam=-d(60), right=d(150), depth_mm=140.0, sector_deg=80.0, apical=False,
            expect=["LV", "IVS", "RV"],
            description="Rotate 90 deg clockwise from PLAX. Circular LV with the septum and the crescent RV "
                        "wrapping it at the top-left.",
        ),
    }
    for p in defs.values():
        p["beam"] = unit(p["beam"])
        p["right"] = unit(p["right"])
        assert abs(np.dot(p["beam"], p["right"])) < 1e-9
        # camera sits on +normal so that screen-right == `right` and the probe is at the top
        p["normal"] = np.cross(p["beam"], p["right"])
        # positive rotation about tilt_axis slides the probe off the true apex (anterior / medial)
        p["tilt_axis"] = -p["right"] if p["apical"] else None
    return defs


def in_fan(points: np.ndarray, origin, beam, depth, sector_deg) -> np.ndarray:
    rel = points - origin
    dist = np.linalg.norm(rel, axis=1)
    cosang = (rel @ beam) / np.maximum(dist, 1e-9)
    return (dist <= depth) & (cosang >= math.cos(math.radians(sector_deg / 2)))


def chambers_cut(meshes, origin, normal, beam, depth, sector_deg) -> list[str]:
    hit = []
    for name, m in meshes.items():
        sec = m.section(plane_origin=origin, plane_normal=normal)
        if sec is None:
            continue
        if in_fan(np.asarray(sec.vertices), origin, beam, depth, sector_deg).sum() >= 8:
            hit.append(name)
    return hit


def rodrigues(v, axis, ang_rad):
    axis = unit(axis)
    return v * math.cos(ang_rad) + np.cross(axis, v) * math.sin(ang_rad) + axis * np.dot(axis, v) * (1 - math.cos(ang_rad))


def hex_rgba(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [255]


# -------------------------------- main -------------------------------------- #
def build(verbose: bool = True) -> dict:
    config.ensure_dirs()
    local = build_local(verbose)

    # centre the model on the bounding-box centre of all structures
    all_v = np.vstack([m.vertices for m in local.values()])
    centre_local = 0.5 * (all_v.min(axis=0) + all_v.max(axis=0))

    def to_world(p):
        return (np.asarray(p, dtype=np.float64) - centre_local) @ ROT.T

    def dir_world(v):
        return np.asarray(v, dtype=np.float64) @ ROT.T

    scene = trimesh.Scene()
    world = {}
    for name in COLORS:                                  # fixed order = render order in the viewer
        w = local[name].copy()
        w.vertices = to_world(local[name].vertices)
        assert w.is_watertight, f"{name} is not watertight"
        assert w.volume > 0
        mat = trimesh.visual.material.PBRMaterial(
            name=f"{name}_mat", baseColorFactor=hex_rgba(COLORS[name]), metallicFactor=0.0, roughnessFactor=0.55
        )
        w.visual = trimesh.visual.TextureVisuals(material=mat)
        w.metadata["name"] = name
        scene.add_geometry(w, node_name=name, geom_name=name)
        world[name] = w

    glb = scene.export(file_type="glb")                  # normals are recomputed in the browser (smaller file)
    assert len(glb) / 2**20 < MAX_GLB_MB, f"GLB is {len(glb) / 2**20:.1f} MB (limit {MAX_GLB_MB} MB)"
    config.HEART_GLB_PATH.write_bytes(glb)

    defs = plane_defs_local()
    planes = {}
    for key, p in defs.items():
        planes[key] = {
            "label": p["label"],
            "description": p["description"],
            "origin": to_world(p["origin"]).round(4).tolist(),
            "beam": dir_world(p["beam"]).round(6).tolist(),
            "right": dir_world(p["right"]).round(6).tolist(),
            "normal": dir_world(p["normal"]).round(6).tolist(),
            "tilt_axis": None if p["tilt_axis"] is None else dir_world(p["tilt_axis"]).round(6).tolist(),
            "depth_mm": p["depth_mm"],
            "sector_deg": p["sector_deg"],
            "apical": p["apical"],
            "expected_chambers": p["expect"],
        }
    meta = {
        "units": "mm",
        "model": "schematic implicit-surface teaching model (marching cubes, 1.5 mm voxels); not patient-derived",
        "world_axes": {"+X": "patient left", "+Y": "superior", "+Z": "anterior"},
        "pivot": to_world([0, 0, 0]).round(4).tolist(),          # mitral-annulus centre
        "apex": to_world([0, -LV_LENGTH, 0]).round(4).tolist(),
        "long_axis": dir_world(-Y).round(6).tolist(),            # base -> apex
        "lv_length_mm": LV_LENGTH,
        "lv_radius_mm": LV_RADIUS,
        "colors": COLORS,
        "chambers": {
            n: {
                "volume_ml": round(float(m.volume) / 1000, 1),
                "centroid": m.center_mass.round(2).tolist(),
                "label": to_world(LABEL_LOCAL[n]).round(2).tolist(),
                "description": DESCRIPTIONS[n],
                "vertices": int(len(m.vertices)),
            }
            for n, m in world.items()
        },
        "planes": planes,
    }
    config.HEART_PLANES_PATH.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"wrote {config.HEART_GLB_PATH} ({len(glb) / 2**20:.2f} MB)")
        print(f"wrote {config.HEART_PLANES_PATH}")
    return meta


def load_world_meshes() -> dict[str, trimesh.Trimesh]:
    scene = trimesh.load(config.HEART_GLB_PATH)
    meshes = {}
    for node in scene.graph.nodes_geometry:
        tf, geom = scene.graph[node]
        m = scene.geometry[geom].copy()
        m.apply_transform(tf)
        meshes[node] = m
    return meshes


def verify() -> None:
    """Reload the exported files and check geometry, names and every echo plane."""
    meta = json.loads(config.HEART_PLANES_PATH.read_text())
    size_mb = config.HEART_GLB_PATH.stat().st_size / 2**20
    meshes = load_world_meshes()
    print(f"GLB: {size_mb:.2f} MB, geometries: {sorted(meshes)}")
    assert sorted(meshes) == sorted(COLORS), sorted(meshes)
    assert size_mb < MAX_GLB_MB
    for name in COLORS:
        m = meshes[name]
        ext = m.bounds[1] - m.bounds[0]
        print(f"  {name:6s} verts={len(m.vertices):6d} faces={len(m.faces):6d} watertight={m.is_watertight} "
              f"vol={m.volume / 1000:6.1f} mL extents={np.round(ext, 1)}")
        assert m.is_watertight and m.volume > 0
        assert np.isfinite(m.vertices).all()
        assert len(m.split(only_watertight=False)) == 1, f"{name} must be one connected surface"
    allv = np.vstack([m.vertices for m in meshes.values()])
    size = allv.max(0) - allv.min(0)
    print("  overall extents (mm):", np.round(size, 1))
    assert 100 < size.max() < 260

    # anatomical orientation
    apex, pivot = np.array(meta["apex"]), np.array(meta["pivot"])
    a = unit(apex - pivot)
    cm = {n: m.center_mass for n, m in meshes.items()}
    print("  apex direction (L, S, A):", np.round(a, 3))
    assert a[0] > 0.3 and a[1] < -0.3 and a[2] > 0.3, "apex must point left, inferior, anterior"
    assert cm["RV"][2] > cm["LV"][2] and cm["RV"][0] < cm["LV"][0], "RV must be anterior and to the right of the LV"
    assert cm["LA"][2] < cm["LV"][2] and cm["LA"][1] > cm["LV"][1], "LA must be posterior and superior"
    assert cm["RA"][0] < cm["LA"][0], "RA must be to the right of the LA"
    assert cm["SVC"][1] > cm["RA"][1] > cm["IVC"][1], "SVC above and IVC below the RA"
    assert abs(unit(cm["SVC"] - cm["IVC"])[1]) > 0.97, "venae cavae must run along the body's vertical axis"
    assert cm["PA"][2] > cm["LA"][2], "pulmonary trunk lies anterior to the LA"
    lv_rv = np.linalg.norm(cm["IVS"] - cm["LV"]) + np.linalg.norm(cm["IVS"] - cm["RV"])
    assert lv_rv < 1.6 * np.linalg.norm(cm["LV"] - cm["RV"]) + 25, "septum must lie between the ventricles"

    # each echo plane must cut exactly the expected structures inside its sector
    for key, p in meta["planes"].items():
        o, n, b = map(np.array, (p["origin"], p["normal"], p["beam"]))
        got = chambers_cut(meshes, o, n, b, p["depth_mm"], p["sector_deg"])
        ok = sorted(got) == sorted(p["expected_chambers"])
        print(f"  plane {key:4s} cuts {sorted(got)}  expected {sorted(p['expected_chambers'])}  {'OK' if ok else 'MISMATCH'}")
        assert ok, key
        assert abs(np.dot(n, b)) < 1e-5

    # labels of the structures seen together in one view must not sit on top of each other
    for key, p in meta["planes"].items():
        names = p["expected_chambers"]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                gap = np.linalg.norm(np.array(meta["chambers"][names[i]]["label"]) - np.array(meta["chambers"][names[j]]["label"]))
                assert gap > 22.0, f"labels {names[i]} / {names[j]} are only {gap:.0f} mm apart in {key}"

    # 3-D foreshortening must agree with the closed-form model in simpsons_math
    p = meta["planes"]["A4C"]
    o, n, b, ax = map(np.array, (p["origin"], p["normal"], p["beam"], p["tilt_axis"]))
    for tilt in (0, 10, 20, 30):
        th = math.radians(tilt)
        b2, n2 = rodrigues(b, ax, th), rodrigues(n, ax, th)
        sec = meshes["LV"].section(plane_origin=pivot, plane_normal=n2)
        v = np.asarray(sec.vertices) - pivot
        apparent = float((-(v @ b2)).max())           # farthest LV wall toward the probe
        model = foreshortened_length(meta["lv_length_mm"], meta["lv_radius_mm"], tilt)
        err = abs(apparent - model) / model * 100
        print(f"  A4C tilt {tilt:2d} deg: apparent LV length {apparent:6.2f} mm, model {model:6.2f} mm ({err:.2f} % diff)")
        assert err < 1.5
    print("VERIFIED: GLB loads cleanly, 9 watertight structures, planes cut the expected anatomy")


if __name__ == "__main__":
    build()
    verify()
