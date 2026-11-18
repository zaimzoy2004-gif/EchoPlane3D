"""Statistics used by scripts/run_paper_benchmarks.py, checked against known values."""
import numpy as np
import pytest

from scripts import run_paper_benchmarks as rb


def test_icc_matches_shrout_fleiss_1979_table():
    # Shrout & Fleiss (1979), Psychological Bulletin 86(2):420-428, Table 2: 6 targets x 4 judges -> ICC(2,1) = 0.29
    y = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]])
    assert rb.icc_2_1_matrix(y) == pytest.approx(0.29, abs=0.005)


def test_icc_is_one_for_identity_and_penalises_offset():
    ref = np.array([30.0, 42, 55, 61, 25, 48])
    assert rb.icc_2_1(ref, ref) == pytest.approx(1.0)
    shifted = rb.icc_2_1(ref + 10, ref)
    assert shifted < 0.8, "absolute agreement must penalise a constant bias"
    assert rb.agreement(ref + 10, ref)["pearson_r"] == pytest.approx(1.0), "while Pearson r ignores it"


def test_bland_altman_bias_and_limits():
    ref = np.arange(10, dtype=float)
    diff = np.array([1, -1, 2, -2, 1, -1, 2, -2, 0, 0], float) + 3.0
    a = rb.agreement(ref + diff, ref)
    sd = np.std(diff, ddof=1)
    assert a["bias"] == pytest.approx(3.0) and a["mae"] == pytest.approx(np.mean(np.abs(diff)))
    assert a["loa_low"] == pytest.approx(3.0 - 1.96 * sd) and a["loa_high"] == pytest.approx(3.0 + 1.96 * sd)
    assert a["n"] == 10 and a["n_excluded"] == 0


def test_agreement_excludes_non_finite_pairs_and_reports_them():
    a = rb.agreement([1, 2, float("nan"), 4, 5], [1, 2, 3, 4, float("inf")])
    assert a["n"] == 3 and a["n_excluded"] == 2


def test_dice_and_hd95_on_concentric_discs():
    yy, xx = np.mgrid[0:200, 0:200]
    r = np.hypot(xx - 100, yy - 100)
    a, b = r < 40, r < 46
    assert rb.dice(a, a) == 1.0
    assert rb.dice(a, b) == pytest.approx(2 * a.sum() / (a.sum() + b.sum()))
    assert rb.hd95(a, b, 0.5) == pytest.approx(6 * 0.5, abs=0.75), "boundary gap of 6 px at 0.5 mm/px"
    assert np.isnan(rb.hd95(a, np.zeros_like(a), 0.5))


def test_oracle_prompts_lie_where_they_claim():
    yy, xx = np.mgrid[0:200, 0:200]
    lv = ((xx - 100) / 25) ** 2 + ((yy - 100) / 60) ** 2 <= 1
    myo = (((xx - 100) / 33) ** 2 + ((yy - 100) / 68) ** 2 <= 1) & ~lv
    o = rb.oracle_prompts(lv, myo)
    x, y = (int(v) for v in o["interior"])
    assert lv[y, x]
    wx, wy = (int(v) for v in o["wall"])
    assert myo[wy, wx]
    for (px, py), lab in zip(o["multi_points"], o["multi_labels"]):
        assert (lv if lab == 1 else myo)[int(py), int(px)]
    ys = sorted(p[1] for p, l in zip(o["multi_points"], o["multi_labels"]) if l == 1)
    assert ys[-1] - ys[0] > 40, "positive points must spread along the long axis"
