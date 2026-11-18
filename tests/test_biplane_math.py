"""Phase 4: single-plane vs biplane method of disks (Lang et al., ASE/EACVI 2015)."""
import math

import numpy as np
import pytest

from backend.core.simpsons_math import (
    biplane_volume_from_disks,
    compute_biplane_simpson,
    compute_simpsons_20_disks,
    synth_ellipse_mask,
)

MMPP = 0.25


def test_biplane_equals_single_plane_for_a_solid_of_revolution():
    mask = synth_ellipse_mask(40, 25, MMPP)
    single = compute_simpsons_20_disks(mask, MMPP)["volume_ml"]
    bi = compute_biplane_simpson(mask, mask, MMPP)
    assert bi["volume_ml"] == pytest.approx(single, rel=1e-9)
    assert bi["length_mismatch_pct"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("a,b", [(25.0, 20.0), (30.0, 18.0), (22.0, 27.0)])
def test_biplane_recovers_a_triaxial_ellipsoid_within_1pct(a, b):
    """Semi-axes a (seen in A4C), b (seen in A2C), half-length 40 mm: V = 4/3*pi*a*b*40."""
    exact = 4 / 3 * math.pi * a * b * 40 / 1000
    bi = compute_biplane_simpson(synth_ellipse_mask(40, a, MMPP), synth_ellipse_mask(40, b, MMPP, angle_deg=12), MMPP)
    assert abs(bi["volume_ml"] - exact) / exact < 0.01, (bi["volume_ml"], exact)
    # a single plane assumes circular disks, so it is biased by the factor a/b or b/a
    assert abs(bi["single_plane_a4c_ml"] / exact - a / b) < 0.02
    assert abs(bi["single_plane_a2c_ml"] / exact - b / a) < 0.02


def test_biplane_uses_the_longer_long_axis_and_reports_mismatch():
    full = synth_ellipse_mask(40, 25, MMPP)
    short = synth_ellipse_mask(34, 25, MMPP)            # a foreshortened second view
    bi = compute_biplane_simpson(full, short, MMPP)
    assert bi["length_mm"] == pytest.approx(max(bi["length_a4c_mm"], bi["length_a2c_mm"]))
    assert bi["length_mismatch_pct"] == pytest.approx(15.0, abs=1.0)


def test_formula_matches_hand_calculation():
    a = [10.0] * 20
    b = [20.0] * 20
    assert biplane_volume_from_disks(a, 80.0, b, 100.0) == pytest.approx(math.pi / 4 * 10 * 20 * 100 / 1000)
    with pytest.raises(ValueError):
        biplane_volume_from_disks(a, 80.0, b[:10], 80.0)


def test_empty_mask_gives_zero_volume():
    empty = np.zeros((64, 64), bool)
    assert compute_biplane_simpson(empty, synth_ellipse_mask(40, 25, MMPP), MMPP)["volume_ml"] == 0.0
