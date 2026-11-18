"""Camera-geometry regression tests for the 3D Probe Navigator.

The camera poses are computed in ``frontend/app.js`` from
``frontend/models/heart_planes.json``.  These tests mirror that arithmetic in
Python (``cameraPose``, ``slerpDir``, ``tweenPose``) so a change to the plane
definitions that would make any view singular — ``up`` parallel to the line of
sight, eye on the target, eye travelling through the heart — fails here.

They check the *data and the maths*; the JavaScript itself is exercised by the
real-time browser run recorded in PROGRESS_LOG.md.
"""
import itertools
import json
import math

import numpy as np
import pytest

from backend.core import config

VIEWS = ["A4C", "A2C", "PLAX", "PSAX"]
MODES = ["anatomic", "echo"]
MIN_UP_ANGLE_DEG = 15.0      # app.js falls back below ~7 deg; home poses must stay well clear of it
HEART_RADIUS_MM = 120.0      # no chamber vertex is farther than this from the model centre


@pytest.fixture(scope="module")
def planes():
    return json.loads(config.HEART_PLANES_PATH.read_text())


def camera_pose(meta, view, mode):
    """Mirror of cameraPose() in frontend/app.js (tilt = 0)."""
    p = meta["planes"][view]
    origin, beam, right, normal = (np.array(p[k], dtype=float) for k in ("origin", "beam", "right", "normal"))
    if mode == "echo":
        target = origin + beam * p["depth_mm"] * 0.48
        return target + normal * 520.0, target, -beam
    up = np.array([0.0, 1.0, 0.0])
    d = normal + 0.3 * up + 0.22 * right
    return d / np.linalg.norm(d) * 560.0, np.zeros(3), up


def slerp_dir(a, b, t):
    """Mirror of slerpDir() in frontend/app.js."""
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 0.9995:
        v = a + (b - a) * t
        return v / np.linalg.norm(v)
    if dot < -0.9995:
        axis = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = axis - a * (axis @ a)
    else:
        axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)
    ang = math.acos(dot) * t
    return a * math.cos(ang) + np.cross(axis, a) * math.sin(ang) + axis * (axis @ a) * (1 - math.cos(ang))


def sight_up_angle(pos, target, up):
    sight = (target - pos) / np.linalg.norm(target - pos)
    return math.degrees(math.acos(min(1.0, abs(float(sight @ (up / np.linalg.norm(up)))))))


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("mode", MODES)
def test_home_pose_is_not_singular(planes, view, mode):
    pos, target, up = camera_pose(planes, view, mode)
    assert np.isfinite(pos).all() and np.isfinite(target).all() and np.isfinite(up).all()
    assert np.linalg.norm(pos - target) > 300, "eye must sit well outside the heart"
    assert np.linalg.norm(up) == pytest.approx(1.0, abs=1e-6)
    assert sight_up_angle(pos, target, up) >= MIN_UP_ANGLE_DEG, "camera.up nearly parallel to the line of sight"


@pytest.mark.parametrize("view", VIEWS)
def test_plane_frame_is_orthonormal(planes, view):
    p = planes["planes"][view]
    beam, right, normal = (np.array(p[k], dtype=float) for k in ("beam", "right", "normal"))
    for v in (beam, right, normal):
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)
    assert abs(beam @ right) < 1e-4 and abs(beam @ normal) < 1e-4 and abs(right @ normal) < 1e-4
    assert np.cross(beam, right) == pytest.approx(normal, abs=1e-4), "normal must equal beam x right"


def test_echo_aligned_view_looks_straight_at_the_cut(planes):
    for view in VIEWS:
        pos, target, up = camera_pose(planes, view, "echo")
        assert sight_up_angle(pos, target, up) == pytest.approx(90.0, abs=0.01)


def test_every_view_transition_stays_clear_of_the_heart(planes):
    """Arc interpolation between any two home poses: finite, never inside the
    heart, never singular."""
    keys = list(itertools.product(VIEWS, MODES))
    for a, b in itertools.permutations(keys, 2):
        p0, t0, u0 = camera_pose(planes, *a)
        p1, t1, u1 = camera_pose(planes, *b)
        o0, o1 = p0 - t0, p1 - t1
        for e in np.linspace(0, 1, 61):
            target = t0 + (t1 - t0) * e
            radius = np.linalg.norm(o0) + (np.linalg.norm(o1) - np.linalg.norm(o0)) * e
            pos = target + slerp_dir(o0 / np.linalg.norm(o0), o1 / np.linalg.norm(o1), e) * radius
            up = slerp_dir(u0, u1, e)
            assert np.isfinite(pos).all() and np.isfinite(up).all(), (a, b, e)
            assert np.linalg.norm(pos) > HEART_RADIUS_MM + 150, f"eye enters the heart on {a}->{b}"
            assert sight_up_angle(pos, target, up) > 7.0, f"singular camera on {a}->{b} at e={e:.2f}"


def test_opposite_side_transition_does_not_cut_through_the_centre(planes):
    """A user who orbits to the far side and then picks a view: a straight
    chord would pass through the heart centre; the arc must not."""
    pos, target, _ = camera_pose(planes, "A2C", "anatomic")
    d1 = (pos - target) / np.linalg.norm(pos - target)
    d0 = -d1
    radius = np.linalg.norm(pos - target)
    for e in np.linspace(0, 1, 41):
        eye = target + slerp_dir(d0, d1, e) * radius
        assert np.linalg.norm(eye - target) == pytest.approx(radius, rel=1e-6)


def test_frontend_ships_the_camera_guards():
    js = (config.FRONTEND_DIR / "app.js").read_text()
    for needle in ("function sanitizePose", "function tweenPose", "function slerpDir", "cameraIsFinite()",
                   "webglcontextlost", "webglcontextrestored", "MIN_UP_SINE"):
        assert needle in js, needle
