"""Phase 2 regression tests: cavity snapping, candidate scoring, multi-point
prompting, offline front end and the layout guard."""
import re

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.core import config, sam2_tracker
from backend.main import app

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
WALL_CLICK = [358, 197]          # on the bright lateral wall of the HFrEF phantom (reported bug)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def hfref_frame0():
    meta = sam2_tracker.extract_frames(config.SAMPLES_DIR / "phantom_hfref.mp4")
    return cv2.imread(f"{meta['frames_dir']}/00000.jpg", cv2.IMREAD_GRAYSCALE)


# ------------------------------ snapping (CPU) ------------------------------ #
def test_wall_click_is_snapped_into_the_lv_cavity(hfref_frame0):
    from scripts.prepare_samples import load_gt_masks

    gt0 = load_gt_masks(config.SAMPLES_DIR / "phantom_hfref_gt.npz")[0]
    assert not gt0[WALL_CLICK[1], WALL_CLICK[0]], "the regression click must lie outside the cavity"
    snap = sam2_tracker.snap_to_cavity(hfref_frame0, WALL_CLICK)
    x, y = (int(round(v)) for v in snap["point"])
    assert snap["snapped"] is True
    assert gt0[y, x], "snapped prompt must land inside the reference LV cavity"
    assert 5 < snap["distance_px"] < 80


def test_click_inside_cavity_is_left_alone(hfref_frame0):
    snap = sam2_tracker.snap_to_cavity(hfref_frame0, [264, 230])
    assert snap["snapped"] is False and snap["point"] == [264.0, 230.0]


def test_click_outside_sector_is_not_snapped(hfref_frame0):
    snap = sam2_tracker.snap_to_cavity(hfref_frame0, [5, 5])
    assert snap["snapped"] is False and "outside" in snap["reason"]


def test_candidate_scoring_prefers_compact_dark_lumen_over_wall_and_sector():
    """Synthetic frame: a dark disc (cavity) inside a bright ring (wall) inside a grey sector."""
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - 128, yy - 128)
    img = np.zeros((h, w), np.uint8)
    img[r < 120] = 120                      # sector
    img[r < 60] = 230                       # wall
    img[r < 42] = 30                        # blood pool
    cavity, wall, sector = r < 42, (r < 60) & (r >= 42), r < 120
    horseshoe = wall & (yy < 150)
    cands = np.stack([sector, cavity, horseshoe | cavity]).astype(np.float32)
    mask, info = sam2_tracker.choose_cavity_mask(img, cands, [0.99, 0.30, 0.60], [[128, 128]])
    assert info["chosen"] == 1 and info["rule"].startswith("dark-lumen")
    assert mask.sum() == pytest.approx(cavity.sum(), rel=0.02)
    # a negative point inside the cavity makes that candidate ineligible
    _, info_neg = sam2_tracker.choose_cavity_mask(img, cands, [0.99, 0.30, 0.60], [[128, 128]], [[128, 140]])
    assert not info_neg["candidates"][1]["eligible"]


def test_prompt_normalisation_and_validation():
    assert sam2_tracker.normalise_prompt() == (None, None, "auto")
    assert sam2_tracker.normalise_prompt(click=[3, 4]) == ([[3.0, 4.0]], [1], "single")
    pts, lbl, mode = sam2_tracker.normalise_prompt(points=[[1, 2], [3, 4]], labels=[1, 0])
    assert mode == "multi" and lbl == [1, 0]
    for bad in (dict(points=[[1, 2]], labels=[0]), dict(points=[[1, 2], [3, 4]], labels=[1]),
                dict(points=[[1, 2, 3]]), dict(points=[[1, 2]], labels=[2]), dict(click=[1])):
        with pytest.raises(ValueError):
            sam2_tracker.normalise_prompt(**bad)


def test_quality_flags_volume_jump_and_wall_shaped_mask():
    yy, xx = np.mgrid[0:120, 0:120]
    disc = np.hypot(xx - 60, yy - 60) < 30
    ring = (np.hypot(xx - 60, yy - 60) < 40) & ~disc & (yy < 70)
    good = sam2_tracker.assess_quality([100, 90, 80, 90, 100], [disc] * 5, 0, 2, 20.0)
    assert good["level"] == "good" and good["flags"] == []
    bad = sam2_tracker.assess_quality([100, 40, 95, 90, 100], [ring] * 5, 0, 1, 60.0, np.stack([disc] * 5), 0)
    assert bad["level"] == "warning"
    assert {"volume_jump", "non_compact_mask"} <= set(bad["flags"])


# ------------------------------ API (GPU) ---------------------------------- #
@needs_gpu
def test_wall_click_regression_lvef_within_3_points(client):
    """Clicking the bright wall at (358, 197) used to return LVEF 12.8 %."""
    r = client.post("/api/segment", json={"sample_id": "phantom_hfref", "click": WALL_CLICK}).json()
    assert r["snap"]["snapped"] is True and r["click_original"] == [358.0, 197.0]
    assert abs(r["lvef_pct"] - 32.0) <= 3.0, r["lvef_pct"]
    assert r["quality"]["level"] == "good"
    assert r["prompt"]["rule"].startswith("dark-lumen")
    assert r["vram_mb"] < config.VRAM_BUDGET_MB


@needs_gpu
def test_raw_wall_click_is_flagged_when_snap_is_disabled(client):
    r = client.post("/api/segment", json={"sample_id": "phantom_hfref", "click": WALL_CLICK, "auto_snap": False}).json()
    assert r["snap"]["snapped"] is False and r["click"] == [358.0, 197.0]
    assert r["quality"]["level"] == "warning", "an unsnapped wall click must raise the quality warning"
    assert abs(r["lvef_pct"] - 32.0) > 3.0, "sanity: without snapping the wall is segmented"


@needs_gpu
def test_multi_point_prompt(client):
    body = {"sample_id": "phantom_hfref", "points": [[264, 150], [264, 230], [264, 310], [358, 197]], "labels": [1, 1, 1, 0]}
    r = client.post("/api/segment", json=body).json()
    assert r["prompt_mode"] == "multi" and r["labels"] == [1, 1, 1, 0] and len(r["points"]) == 4
    assert r["snap"]["snapped"] is False, "multi-point prompts are used exactly as given"
    assert abs(r["lvef_pct"] - 32.0) <= 3.0
    chosen = r["prompt"]["candidates"][r["prompt"]["chosen"]]
    assert chosen["honours_prompt"] is True


def test_multi_point_validation_errors(client):
    for bad in ({"points": [[1, 2]], "labels": [0]}, {"points": [[1, 2], [3, 4]], "labels": [1]},
                {"points": [[1, 2, 3]]}, {"points": [[99999, 2]], "labels": [1]}):
        assert client.post("/api/segment", json={"sample_id": "phantom_hfref", **bad}).status_code == 422, bad


# ------------------------------ front end ---------------------------------- #
def test_frontend_is_fully_offline(client):
    html = client.get("/").text
    refs = re.findall(r'(?:src|href)\s*=\s*"([^"]+)"', html)
    remote = [r for r in refs if r.startswith(("http://", "https://", "//"))]
    assert remote == [], f"remote assets referenced: {remote}"
    importmap = re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1)
    assert "http" not in importmap and "./vendor/three/build/three.module.js" in importmap
    for path, needle in [("/vendor/three/build/three.module.js", "REVISION"),
                         ("/vendor/three/examples/jsm/controls/OrbitControls.js", "OrbitControls"),
                         ("/vendor/three/examples/jsm/loaders/GLTFLoader.js", "GLTFLoader"),
                         ("/vendor/three/examples/jsm/utils/BufferGeometryUtils.js", "toTrianglesDrawMode"),
                         ("/vendor/plotly-2.35.2.min.js", "plotly.js v2.35.2"),
                         ("/vendor/tailwindcss-play-3.4.17.js", "tailwind")]:
        res = client.get(path)
        assert res.status_code == 200 and needle in res.text, path
    js = client.get("/app.js").text
    assert "https://" not in js and "http://" not in js


def test_layout_and_viewport_guards_are_present(client):
    html = client.get("/").text
    assert "grid-template-rows: minmax(0, 1fr)" in html, "the app shell row must not grow with its content"
    assert 'id="guideLog" class="flex-1 min-h-0 overflow-y-auto' in html, "chat history must scroll internally"
    js = client.get("/app.js").text
    for needle in ("setClearColor(CLEAR_COLOR, 1)", "function resizeThree", "w < 16 || h < 16", "tgMulti", "auto_snap"):
        assert needle in js, needle
