"""Unit tests for the Simpson's-rule engine and the foreshortening model."""
import math

import numpy as np
import pytest

from backend.core.simpsons_math import (
    compute_simpsons_20_disks,
    foreshortened_length,
    foreshortening_sweep,
    lv_dimensions_from_volume,
    prolate_ellipsoid_volume_ml,
    simulate_apical_foreshortening,
    synth_ellipse_mask,
)

R_MM = 25.0
HALF_L_MM = 40.0
EXACT_ML = 104.72


def test_exact_reference_volume():
    assert prolate_ellipsoid_volume_ml(R_MM, HALF_L_MM) == pytest.approx(EXACT_ML, abs=0.005)


@pytest.mark.parametrize("mm_per_pixel", [0.15, 0.25, 0.4])
@pytest.mark.parametrize("angle_deg", [0, 17, 45, 90, 133])
def test_prolate_ellipsoid_under_1pct(mm_per_pixel, angle_deg):
    mask = synth_ellipse_mask(HALF_L_MM, R_MM, mm_per_pixel, angle_deg)
    res = compute_simpsons_20_disks(mask, mm_per_pixel)
    err = abs(res["volume_ml"] - EXACT_ML) / EXACT_ML * 100
    assert err < 1.0, f"{res['volume_ml']:.3f} mL vs {EXACT_ML} ({err:.3f} %)"
    assert res["length_mm"] == pytest.approx(2 * HALF_L_MM, rel=0.01)
    assert len(res["chords"]) == 20 and len(res["diameters_mm"]) == 20
    assert max(res["diameters_mm"]) == pytest.approx(2 * R_MM, rel=0.02)


def test_anisotropic_pixels():
    # Same ellipse sampled with 0.2 mm (x) by 0.4 mm (y) pixels.
    iso = synth_ellipse_mask(HALF_L_MM, R_MM, 0.2, 30)
    aniso = iso[::2, :]
    res = compute_simpsons_20_disks(aniso, (0.2, 0.4))
    assert abs(res["volume_ml"] - EXACT_ML) / EXACT_ML < 0.01


def test_long_axis_geometry_in_pixels():
    mmpp = 0.25
    mask = synth_ellipse_mask(HALF_L_MM, R_MM, mmpp, 0)
    res = compute_simpsons_20_disks(mask, mmpp)
    (x0, y0), (x1, y1) = res["long_axis"]
    assert y1 > y0, "first endpoint must be the top (apex side)"
    assert abs(x1 - x0) < 1.5, "vertical ellipse -> vertical axis"
    assert (y1 - y0) * mmpp == pytest.approx(2 * HALF_L_MM, rel=0.01)
    # chords are perpendicular to the long axis
    (xa, ya), (xb, yb) = res["chords"][10]
    assert abs(ya - yb) < 1.5 and abs(xb - xa) * mmpp == pytest.approx(2 * R_MM, rel=0.03)


def test_half_ellipsoid_lv_model():
    a, b = 88.9, 25.4  # ~120 mL
    exact = 2 / 3 * math.pi * b * b * a / 1000
    mask = synth_ellipse_mask(a, b, 0.3, 10, half=True)
    res = compute_simpsons_20_disks(mask, 0.3)
    assert abs(res["volume_ml"] - exact) / exact < 0.01


def test_empty_and_tiny_masks():
    assert compute_simpsons_20_disks(np.zeros((64, 64), bool), 0.3)["volume_ml"] == 0.0
    tiny = np.zeros((64, 64), bool)
    tiny[30:32, 30:32] = True
    assert compute_simpsons_20_disks(tiny, 0.3)["volume_ml"] == 0.0


# ------------------------------ foreshortening ------------------------------ #
def test_dimensions_reproduce_volume():
    for v in (120.0, 80.0, 48.0):
        a, b = lv_dimensions_from_volume(v, 120.0)
        assert 2 / 3 * math.pi * b * b * a / 1000 == pytest.approx(v, rel=1e-9)


def test_zero_tilt_is_identity():
    r = simulate_apical_foreshortening(120, 48, 0)
    assert r["foreshortened"]["edv_ml"] == pytest.approx(120)
    assert r["foreshortened"]["esv_ml"] == pytest.approx(48)
    assert r["foreshortened"]["lvef_pct"] == pytest.approx(60)
    assert r["errors"]["lvef_abs_pts"] == pytest.approx(0)


def test_volumes_fall_and_lvef_rises_with_tilt():
    sweep = foreshortening_sweep(120, 48, 30, 1)
    assert np.all(np.diff(sweep["edv_ml"]) < 0)
    assert np.all(np.diff(sweep["esv_ml"]) < 0)
    assert np.all(np.diff(sweep["lvef_pct"]) > 0)
    r15 = simulate_apical_foreshortening(120, 48, 15)
    assert -35 < r15["errors"]["edv_pct"] < -15
    assert 0 < r15["errors"]["lvef_abs_pts"] < 6


@pytest.mark.parametrize("edv,esv", [(120.0, 48.0), (210.0, 143.0)])
@pytest.mark.parametrize("tilt", [5, 10, 15])
def test_model_matches_numerical_plane_slice(edv, esv, tilt):
    """Slice the 3-D half ellipsoid with a tilted plane, run Simpson on the
    slice and compare with the closed-form model (<1 %)."""
    model = simulate_apical_foreshortening(edv, esv, tilt)
    th = math.radians(tilt)
    mmpp = 0.25
    for phase, vol in (("edv_ml", edv), ("esv_ml", esv)):
        a, b = lv_dimensions_from_volume(vol, edv)
        # plane coords: u along the mitral annulus (x), v along the tilted beam
        v_max = a + 5
        vs = np.arange(0, v_max, mmpp) + mmpp / 2
        us = np.arange(-b - 5, b + 5, mmpp) + mmpp / 2
        uu, vv = np.meshgrid(us, vs)
        x, y, z = uu, vv * math.sin(th), vv * math.cos(th)
        inside = (x / b) ** 2 + (y / b) ** 2 + (z / a) ** 2 <= 1.0
        num = compute_simpsons_20_disks(inside, mmpp)["volume_ml"]
        assert num == pytest.approx(model["foreshortened"][phase], rel=0.01)
        a_f = foreshortened_length(a, b, tilt)
        assert a_f < a or tilt == 0


@pytest.mark.parametrize("bad", [(0, 0, 0), (100, 120, 5), (100, 40, -1), (100, 40, 95), (100, 0, 5)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        simulate_apical_foreshortening(*bad)
