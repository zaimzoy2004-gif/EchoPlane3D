"""Phase 6: custom echo clip upload (validation, frame cap, calibration, tracking)."""
import io

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.core import config
from backend.core.video_io import read_video_gray
from backend.main import app

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.fixture(scope="module")
def client():
    created = []
    with TestClient(app) as c:
        c.created = created
        yield c
        for sid in set(created):
            c.delete(f"/api/upload/{sid}")


@pytest.fixture(scope="module")
def phantom_frames():
    return read_video_gray(config.SAMPLES_DIR / "phantom_normal.mp4")      # 50 frames = two beats


def encode(frames: np.ndarray, path, fourcc: str, fps: float = 25.0) -> bytes:
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    assert vw.isOpened(), f"cannot open a {fourcc} writer"
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    return path.read_bytes()


@pytest.fixture(scope="module")
def long_mp4(phantom_frames, tmp_path_factory):
    """150-frame clip (the phantom three times over) to exercise the 90-frame cap."""
    return encode(np.concatenate([phantom_frames] * 3), tmp_path_factory.mktemp("up") / "long.mp4", "mp4v")


def upload(client, data: bytes, name: str, **form):
    res = client.post("/api/upload", files={"file": (name, io.BytesIO(data), "application/octet-stream")},
                      data={k: str(v) for k, v in form.items()})
    if res.status_code == 200:
        client.created.append(res.json()["sample"]["id"])
    return res


def test_upload_is_capped_and_marked_uncalibrated(client, long_mp4):
    res = upload(client, long_mp4, "My Echo (patient X).mp4")
    assert res.status_code == 200, res.text
    s = res.json()["sample"]
    assert s["kind"] == "upload" and s["id"].startswith("upload_")
    assert s["n_frames"] <= config.MAX_UPLOAD_FRAMES == 90 and s["source_frames"] == 150 and s["subsampled"] is True
    assert s["calibrated"] is False and s["volume_unit"] == "index" and s["mm_per_pixel"] == 1.0
    assert "Uncalibrated" in s["description"] and "scale-invariant" in s["description"]
    assert s["width"] == 512 and s["height"] == 512 and s["ground_truth"] is None
    assert "(" not in s["name"], "file names are sanitised before being shown"
    # duration is preserved by lowering the frame rate
    assert s["n_frames"] / s["fps"] == pytest.approx(150 / 25.0, rel=0.02)

    listing = client.get("/api/samples").json()
    assert s["id"] in {x["id"] for x in listing["samples"]}
    assert any(subj["kind"] == "upload" and s["id"] in subj["views"].values() for subj in listing["subjects"])
    frame = client.get(s["frame_url_template"].format(index=0))
    assert frame.status_code == 200 and frame.content[:2] == b"\xff\xd8"
    assert client.get(s["video_url"]).content[4:8] == b"ftyp"


def test_avi_upload_and_calibration(client, phantom_frames, tmp_path):
    avi = encode(phantom_frames, tmp_path / "clip.avi", "MJPG")
    res = upload(client, avi, "clip.avi", mm_per_pixel=0.34, view="A2C")
    assert res.status_code == 200, res.text
    s = res.json()["sample"]
    assert s["calibrated"] is True and s["volume_unit"] == "mL" and s["view"] == "A2C"
    assert s["mm_per_pixel"] == pytest.approx(0.34) and s["n_frames"] == 50 and s["subsampled"] is False


@pytest.mark.parametrize("name,data,form", [
    ("notes.txt", b"hello", {}),
    ("fake.mp4", b"this is not a video" * 50, {}),
    ("empty.mp4", b"", {}),
    ("clip.mov", b"\x00" * 64, {}),
])
def test_bad_uploads_are_rejected(client, name, data, form):
    assert upload(client, data, name, **form).status_code == 422


def test_bad_calibration_and_view_are_rejected(client, long_mp4):
    assert upload(client, long_mp4, "a.mp4", mm_per_pixel=99).status_code == 422
    assert upload(client, long_mp4, "a.mp4", mm_per_pixel=0).status_code == 422
    assert upload(client, long_mp4, "a.mp4", view="PLAX").status_code == 422


def test_delete_only_touches_uploads(client, long_mp4):
    sid = upload(client, long_mp4, "to_delete.mp4").json()["sample"]["id"]
    assert client.delete(f"/api/upload/{sid}").status_code == 200
    assert sid not in {x["id"] for x in client.get("/api/samples").json()["samples"]}
    assert client.delete(f"/api/upload/{sid}").status_code == 404
    assert client.delete("/api/upload/phantom_normal").status_code == 404
    assert client.delete("/api/upload/..%2Fsamples%2Fphantom_normal").status_code in (404, 405)
    assert (config.SAMPLES_DIR / "phantom_normal.mp4").is_file()


@needs_gpu
def test_uploaded_clip_is_tracked_and_lvef_is_scale_invariant(client, long_mp4, phantom_frames, tmp_path):
    raw = upload(client, long_mp4, "uncal.mp4").json()["sample"]
    cal = upload(client, encode(phantom_frames, tmp_path / "cal.mp4", "mp4v"), "cal.mp4", mm_per_pixel=0.34).json()["sample"]
    r_raw = client.post("/api/segment", json={"sample_id": raw["id"]}).json()
    r_cal = client.post("/api/segment", json={"sample_id": cal["id"]}).json()
    for r in (r_raw, r_cal):
        assert r["vram_mb"] < config.VRAM_BUDGET_MB
        assert abs(r["lvef_pct"] - 60.0) < 3.0, "the uploaded clip is the 60 % phantom"
        assert r["reference_single_plane_lvef"] is None and r["biplane"] is None
    assert r_raw["calibrated"] is False and r_raw["volume_unit"] == "index"
    assert r_cal["calibrated"] is True and r_cal["volume_unit"] == "mL"
    assert abs(r_cal["edv_ml"] - 120.0) / 120.0 < 0.10, "calibrated upload must recover the phantom EDV in mL"
    assert r_raw["edv_ml"] > 10 * r_cal["edv_ml"], "uncalibrated volumes are a pixel-scale index, not mL"
    assert r_raw["lvef_pct"] == pytest.approx(r_cal["lvef_pct"], abs=1.0), "LVEF does not depend on calibration"
    # multi-point prompting works on uploads too
    r_multi = client.post("/api/segment", json={"sample_id": cal["id"], "points": [[264, 150], [264, 240]], "labels": [1, 1]}).json()
    assert r_multi["prompt_mode"] == "multi" and abs(r_multi["lvef_pct"] - 60.0) < 3.0
