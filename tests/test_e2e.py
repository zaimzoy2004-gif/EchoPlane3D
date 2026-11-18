"""End-to-end tests: every API endpoint, the static front end, tracker
accuracy on the calibrated phantoms and VRAM stability over 3 consecutive
SAM 2.1 runs.

Run with:  pytest   (GPU tests skip automatically without CUDA)
"""
import json

import pytest
import torch
from fastapi.testclient import TestClient

from backend.core import config, sam2_tracker
from backend.main import app

VRAM_BUDGET_MB = 2500.0
BENCH_SAMPLE = "phantom_normal"          # 50 frames
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def samples(client):
    data = client.get("/api/samples").json()
    return {s["id"]: s for s in data["samples"]}


# ------------------------------- basics ------------------------------------ #
def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["sam2"]["checkpoint"] == str(config.SAM2_CHECKPOINT)
    assert h["sam2"]["checkpoint"].endswith("sam2.1_hiera_tiny.pt")
    assert h["sam2"]["config"] == "configs/sam2.1/sam2.1_hiera_t.yaml"
    if torch.cuda.is_available():            # the GPU machine must have the real assets; CPU CI does not
        assert h["sam2"]["checkpoint_exists"] and h["sam2"]["config_exists"]
    assert h["sam2"]["config"] == "configs/sam2.1/sam2.1_hiera_t.yaml"
    assert h["heart_model"] is True
    assert h["vram_budget_mb"] == VRAM_BUDGET_MB


def test_samples_listing(client, samples):
    assert {"phantom_normal", "phantom_hfref"} <= set(samples)
    for s in samples.values():
        assert s["mm_per_pixel"] > 0 and s["fps"] > 0 and s["n_frames"] > 1
        assert s["width"] % 2 == 0 and s["height"] % 2 == 0
        assert not any(k.startswith("_") for k in s), "internal fields must not leak"
        gt = s["ground_truth"]
        assert gt["edv_ml"] > gt["esv_ml"] > 0
    assert samples["phantom_normal"]["ground_truth"]["lvef_pct"] == pytest.approx(60.0, abs=0.01)
    assert samples["phantom_hfref"]["ground_truth"]["lvef_pct"] == pytest.approx(32.0, abs=0.01)
    assert samples["phantom_normal"]["n_frames"] == 50
    assert list(samples)[0] == "phantom_normal"


def test_sample_media(client, samples):
    s = samples[BENCH_SAMPLE]
    video = client.get(s["video_url"])
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert video.content[4:8] == b"ftyp"
    for idx in (0, s["n_frames"] - 1):
        frame = client.get(s["frame_url_template"].format(index=idx))
        assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"
    assert client.get(s["frame_url_template"].format(index=s["n_frames"])).status_code == 404
    assert client.get("/api/samples/does_not_exist/video").status_code == 404
    assert client.get("/api/samples/..%2F..%2Fmain/video").status_code == 404


def test_frontend_is_served(client):
    index = client.get("/")
    assert index.status_code == 200 and "text/html" in index.headers["content-type"]
    html = index.text
    for needle in ("EchoPlane", "vendor/tailwindcss", "plotly", "vendor/three", 'src="app.js"', "tiltSlider", 'max="30"'):
        assert needle in html, needle
    js = client.get("/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    for needle in ("localClippingEnabled = true", "THREE.Plane", "GLTFLoader", "/api/segment", "/api/foreshortening", "/api/guide"):
        assert needle in js.text, needle
    glb = client.get("/models/heart_chambers.glb")
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    planes = client.get("/models/heart_planes.json").json()
    assert set(planes["planes"]) == {"A4C", "A2C", "PLAX", "PSAX"}
    assert set(planes["colors"]) == {"LV", "IVS", "RV", "LA", "RA", "Aorta", "PA", "SVC", "IVC"}


def test_heart_glb_loads_with_all_structures():
    import trimesh

    scene = trimesh.load(config.HEART_GLB_PATH)
    assert sorted(scene.geometry) == ["Aorta", "IVC", "IVS", "LA", "LV", "PA", "RA", "RV", "SVC"]
    assert all(m.is_watertight and m.volume > 0 for m in scene.geometry.values())


# ---------------------------- foreshortening -------------------------------- #
def test_foreshortening_endpoint(client):
    zero = client.get("/api/foreshortening", params={"true_edv": 120, "true_esv": 48, "tilt_deg": 0}).json()
    assert zero["foreshortened"]["lvef_pct"] == pytest.approx(60.0)
    assert zero["foreshortened"]["edv_ml"] == pytest.approx(120.0)

    r = client.post("/api/foreshortening", json={"true_edv": 120, "true_esv": 48, "tilt_deg": 20,
                                                 "volumes_ml": [120, 90, 48]}).json()
    f, e = r["foreshortened"], r["errors"]
    assert f["edv_ml"] < 120 and f["esv_ml"] < 48
    assert e["esv_pct"] < e["edv_pct"] < 0, "ESV must be under-estimated more than EDV"
    assert f["lvef_pct"] > 60.0, "foreshortening over-estimates LVEF"
    assert r["curve_ml"][0] == pytest.approx(f["edv_ml"], abs=0.01)
    assert r["curve_ml"][2] == pytest.approx(f["esv_ml"], abs=0.01)
    assert r["sweep"]["tilt_deg"][0] == 0 and r["sweep"]["tilt_deg"][-1] == 30

    for bad in ({"true_edv": 100, "true_esv": 150, "tilt_deg": 5},
                {"true_edv": -1, "true_esv": 50, "tilt_deg": 5},
                {"true_edv": 100, "true_esv": 50, "tilt_deg": 95},
                {"true_edv": 100, "true_esv": 50}):
        assert client.post("/api/foreshortening", json=bad).status_code == 422


# -------------------------------- guide ------------------------------------ #
def test_guide_switches_views(client):
    info = client.get("/api/guide").json()
    assert len(info["presets"]) >= 6 and info["welcome"]
    for msg, view in [("show me the apical four chamber view", "A4C"), ("A2C please", "A2C"),
                      ("go to parasternal long axis", "PLAX"), ("short-axis view", "PSAX")]:
        r = client.post("/api/guide", json={"message": msg, "context": {"view": "A4C"}}).json()
        assert r["intent"] == "set_view"
        assert {"type": "set_view", "view": view} in r["actions"], msg
        assert r["reply"]


def test_guide_explains_foreshortening_with_numbers(client):
    ctx = {"view": "PLAX", "tilt_deg": 0, "edv_ml": 200, "esv_ml": 136}
    r = client.post("/api/guide", json={"message": "what is foreshortening?", "context": ctx}).json()
    assert r["intent"] == "explain_foreshortening"
    assert "over-estimated" in r["reply"]
    assert "200" in r["verified"], "numbers travel in the separate verified-calculation block"
    assert {"type": "set_view", "view": "A4C"} in r["actions"]

    demo = client.post("/api/guide", json={"preset": "demo_foreshortening", "context": ctx}).json()
    assert {"type": "set_tilt", "tilt_deg": 20.0} in demo["actions"]
    fix = client.post("/api/guide", json={"message": "how do I fix foreshortening", "context": ctx}).json()
    assert {"type": "set_tilt", "tilt_deg": 0} in fix["actions"]
    simpson = client.post("/api/guide", json={"message": "explain Simpson's method of disks"}).json()
    assert simpson["intent"] == "explain_simpson" and "20 disks" in simpson["reply"]


def test_guide_quiz_is_deterministic_and_graded(client):
    body = {"message": "quiz me on landmarks", "context": {"view": "PSAX", "quiz_index": 1}}
    a = client.post("/api/guide", json=body).json()
    b = client.post("/api/guide", json=body).json()
    assert a == b, "the tutor must be deterministic"
    quiz = a["quiz"]
    assert quiz["view"] == "PSAX" and len(quiz["options"]) == 4
    assert "answer" not in quiz, "the answer must not be sent with the question"

    results = [client.post("/api/guide", json={"quiz_answer": {"id": quiz["id"], "answer": i}}).json()["quiz_result"]
               for i in range(4)]
    assert sum(r["correct"] for r in results) == 1
    assert all(r["explanation"] for r in results)
    assert client.post("/api/guide", json={"quiz_answer": {"id": "nope:9", "answer": 0}}).json()["intent"] == "quiz_error"
    fallback = client.post("/api/guide", json={"message": "what's the weather"}).json()
    assert fallback["intent"] == "fallback" and fallback["actions"] == []


# ------------------------------- segment ----------------------------------- #
def test_segment_rejects_bad_requests(client):
    assert client.post("/api/segment", json={"sample_id": "does_not_exist"}).status_code == 404
    assert client.post("/api/segment", json={"sample_id": BENCH_SAMPLE, "click": [1, 2, 3]}).status_code == 422
    assert client.post("/api/segment", json={"sample_id": BENCH_SAMPLE, "click": [99999, 10]}).status_code == 422
    assert client.post("/api/segment", json={}).status_code == 422


@needs_gpu
def test_segment_vram_stable_over_three_runs(client, samples):
    """3 consecutive uncached SAM 2.1 runs on the 50-frame phantom: every run
    under 2.5 GB, no growth between runs, identical measurements."""
    gt = samples[BENCH_SAMPLE]["ground_truth"]
    runs, after = [], []
    for _ in range(3):
        res = client.post("/api/segment", json={"sample_id": BENCH_SAMPLE, "force": True})
        assert res.status_code == 200, res.text
        r = res.json()
        assert r["cached"] is False and r["n_frames"] == 50
        runs.append(r)
        after.append(torch.cuda.memory_allocated() / 2**20)

    for r in runs:
        for key in ("frames_data", "edv_ml", "esv_ml", "lvef_pct", "ed_frame_idx", "es_frame_idx", "vram_mb",
                    "inference_time_sec"):
            assert key in r, key
        assert r["vram_mb"] < VRAM_BUDGET_MB, f"peak VRAM {r['vram_mb']} MB over budget"
        assert r["vram_allocated_mb"] <= r["vram_mb"]
        assert r["autocast_dtype"] in ("bfloat16", "float16")
        # estimates and references are separately named and never mixed
        assert r["single_plane_lvef"] == r["lvef_pct"]
        assert r["reference_single_plane_lvef"] == gt["lvef_pct"]
        assert r["biplane_lvef"] is None and r["biplane"] is None, "the phantom has only one view"
        assert "dataset_reported_biplane_ef" in r and "reference_biplane_lvef" in r
        assert r["quality"]["level"] == "good" and r["calibrated"] is True and r["volume_unit"] == "mL"
        assert len(fd_first := r["frames_data"][0]["diameters_mm"]) == 20 and max(fd_first) > 0
        assert r["inference_time_sec"] > 0
        assert len(r["frames_data"]) == 50
        fd = r["frames_data"][r["ed_frame_idx"]]
        assert len(fd["chords"]) == 20 and len(fd["long_axis"]) == 2 and len(fd["contour"]) > 8
        assert r["edv_ml"] > r["esv_ml"] > 0
        # accuracy against the calibrated phantom
        assert abs(r["lvef_pct"] - gt["lvef_pct"]) < 3.0
        assert abs(r["edv_ml"] - gt["edv_ml"]) / gt["edv_ml"] < 0.10
        assert abs(r["esv_ml"] - gt["esv_ml"]) / gt["esv_ml"] < 0.10

    peaks = [r["vram_mb"] for r in runs]
    assert max(peaks) - min(peaks) < 128, f"peak VRAM drifts between runs: {peaks}"
    assert abs(after[2] - after[0]) < 16, f"VRAM leak between runs: {after}"
    assert max(after) < 600, f"too much VRAM left allocated after reset_state: {after}"
    assert torch.cuda.memory_reserved() / 2**20 < VRAM_BUDGET_MB
    assert runs[0]["volumes_ml"] == pytest.approx(runs[2]["volumes_ml"], abs=0.75)
    print("\nVRAM peaks (MB):", peaks, "| allocated after each run (MB):", [round(a, 1) for a in after],
          "| LVEF:", [r["lvef_pct"] for r in runs], "| time (s):", [r["inference_time_sec"] for r in runs])


@needs_gpu
def test_segment_cache_and_user_click(client, samples):
    cached = client.post("/api/segment", json={"sample_id": BENCH_SAMPLE}).json()
    assert cached["cached"] is True and cached["click_source"] == "auto-centroid"
    assert cached["ground_truth"]["edv_ml"] == 120.0
    cache_files = list(config.CACHE_DIR.glob(f"seg_{BENCH_SAMPLE}_*.json"))
    assert cache_files and "frames_data" in json.loads(cache_files[0].read_text())
    probe = client.post("/api/segment", json={"sample_id": BENCH_SAMPLE, "cached_only": True}).json()
    assert probe["available"] is True

    # explicit frame-0 click on the reduced-EF phantom
    s = samples["phantom_hfref"]
    r = client.post("/api/segment", json={"sample_id": "phantom_hfref", "click": s["lv_seed_point"], "force": True}).json()
    assert r["click_source"] == "user" and r["click"] == pytest.approx(s["lv_seed_point"], abs=0.11)
    assert r["vram_mb"] < VRAM_BUDGET_MB
    assert abs(r["lvef_pct"] - 32.0) < 3.0
    assert abs(r["edv_ml"] - 210.0) / 210.0 < 0.10


def test_auto_centroid_lands_in_the_lv(samples):
    """The automatic prompt must hit the LV on both phantoms and on at least
    85 % of the real clips (the heuristic was tuned on this demo cohort; the
    paper benchmark reports its hit rate on patients it was not tuned on)."""
    import cv2

    from scripts.prepare_samples import load_gt_masks

    hits = {}
    for sid, s in samples.items():
        if s["kind"] == "upload":
            continue
        meta = sam2_tracker.extract_frames(config.SAMPLES_DIR / s["video"])
        gray = cv2.imread(f"{meta['frames_dir']}/00000.jpg", cv2.IMREAD_GRAYSCALE)
        x, y = sam2_tracker.auto_centroid(gray)
        gt0 = load_gt_masks(config.SAMPLES_DIR / f"{sid}_gt.npz")[0]
        hits[sid] = bool(gt0[int(round(y)), int(round(x))])
    assert hits["phantom_normal"] and hits["phantom_hfref"]
    real = [v for k, v in hits.items() if k.startswith("camus_")]
    if real:
        assert sum(real) / len(real) >= 0.85, {k: v for k, v in hits.items() if not v}
