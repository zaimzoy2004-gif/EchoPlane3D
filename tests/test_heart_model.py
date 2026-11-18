"""Phase 3: anatomical multi-structure heart model and LV contraction maths."""
import json

import numpy as np
import pytest

from backend.core import config
from scripts import build_heart_glb as hb

STRUCTURES = ["LV", "IVS", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC"]


@pytest.fixture(scope="module")
def meta():
    return json.loads(config.HEART_PLANES_PATH.read_text())


@pytest.fixture(scope="module")
def meshes():
    return hb.load_world_meshes()


def test_glb_has_nine_named_watertight_structures(meshes):
    assert sorted(meshes) == sorted(STRUCTURES)
    for name, m in meshes.items():
        assert m.is_watertight and m.volume > 0, name
        assert len(m.split(only_watertight=False)) == 1, f"{name} must be a single surface"
    assert config.HEART_GLB_PATH.stat().st_size < 6 * 2**20, "GLB must stay under 6 MB"


def test_metadata_describes_every_structure(meta):
    assert list(meta["colors"]) == STRUCTURES
    for name in STRUCTURES:
        info = meta["chambers"][name]
        assert info["volume_ml"] > 0 and len(info["label"]) == 3 and info["description"]
    assert "not patient-derived" in meta["model"]


def test_full_geometric_verification_passes(capsys):
    """Orientation, plane/structure intersections, label spacing and the 3-D
    foreshortening cross-check (re-runs the builder's own verification)."""
    hb.verify()
    assert "VERIFIED" in capsys.readouterr().out


@pytest.mark.parametrize("view,expected", [
    ("A4C", {"LV", "IVS", "RV", "LA", "RA"}), ("A2C", {"LV", "LA"}),
    ("PLAX", {"LV", "IVS", "RV", "LA", "Aorta"}), ("PSAX", {"LV", "IVS", "RV"}),
])
def test_each_plane_cuts_the_clinically_expected_structures(meta, meshes, view, expected):
    p = meta["planes"][view]
    got = hb.chambers_cut(meshes, np.array(p["origin"]), np.array(p["normal"]), np.array(p["beam"]),
                          p["depth_mm"], p["sector_deg"])
    assert set(got) == expected == set(p["expected_chambers"])


def test_septum_lies_between_the_ventricles(meshes):
    lv, rv, ivs = (meshes[n].center_mass for n in ("LV", "RV", "IVS"))
    axis = (rv - lv) / np.linalg.norm(rv - lv)
    t = float((ivs - lv) @ axis) / float(np.linalg.norm(rv - lv))
    assert 0.2 < t < 1.0, "IVS centroid must project between the LV and RV centroids"


@pytest.mark.parametrize("ratio", [1.0, 0.68, 0.40, 0.25])
def test_lv_contraction_scaling_preserves_the_volume_ratio(ratio):
    """Mirror of updateContraction() in app.js: r^0.25 along the long axis and
    r^0.375 radially must scale the mesh volume by exactly r."""
    longitudinal, radial = ratio ** 0.25, ratio ** 0.375
    assert radial * radial * longitudinal == pytest.approx(ratio, rel=1e-12)
    assert radial <= longitudinal <= 1.0, "radial shortening exceeds longitudinal shortening"


def test_frontend_drives_contraction_from_volume_curve():
    js = (config.FRONTEND_DIR / "app.js").read_text()
    for needle in ("function updateContraction", "Math.pow(r, 0.25)", "Math.pow(r, 0.375)", "meta.pivot", "tgBeat"):
        assert needle in js, needle
