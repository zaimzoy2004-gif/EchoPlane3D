"""Phase 5: multi-patient CAMUS cohort, subject pairing and biplane estimates.

CAMUS files are downloaded on demand and are not part of the repository, so
these tests skip when the cohort has not been prepared."""
import json

import pytest
import torch
from fastapi.testclient import TestClient

from backend.core import camus, config
from backend.main import app

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
HAS_CAMUS = len(list(config.SAMPLES_DIR.glob("camus_*_2ch.json"))) >= 8
needs_camus = pytest.mark.skipif(not HAS_CAMUS, reason="CAMUS cohort not prepared (run scripts/prepare_samples.py)")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def listing(client):
    return client.get("/api/samples").json()


def test_ef_bands():
    assert camus.ef_class(60).startswith("normal")
    assert camus.ef_class(52).startswith("low-normal")
    assert camus.ef_class(45).startswith("mildly reduced")
    assert camus.ef_class(40).startswith("reduced")
    assert camus.ef_class(None) == "unknown"


def test_every_sample_belongs_to_a_subject(listing):
    ids = {s["id"] for s in listing["samples"]}
    for subj in listing["subjects"]:
        assert subj["views"], subj
        assert set(subj["views"].values()) <= ids
        assert set(subj["views"]) <= {"A4C", "A2C"}
    assert sum(len(s["views"]) for s in listing["subjects"]) == listing["count"]


@needs_camus
def test_cohort_has_18_sequences_from_10_subjects(listing):
    subjects = listing["subjects"]
    camus_subjects = [s for s in subjects if s["kind"] == "camus"]
    assert listing["count"] >= 18 and len(camus_subjects) == 8
    assert sum(1 for s in subjects if s["kind"] == "phantom") == 2
    for s in camus_subjects:
        assert set(s["views"]) == {"A4C", "A2C"}, f"{s['id']} must have both apical views"


@needs_camus
def test_cohort_spans_ef_bands_and_image_qualities(listing):
    cam = [s for s in listing["samples"] if s["kind"] == "camus"]
    bands = {s["ef_class"].split(" (")[0] for s in cam}
    assert {"normal", "mildly reduced", "reduced"} <= bands
    assert {s["image_quality"] for s in cam} == {"Good", "Medium", "Poor"}
    for s in cam:
        gt = s["ground_truth"]
        assert gt["edv_ml"] > gt["esv_ml"] > 0
        assert 5 < gt["reference_biplane_lvef_pct"] < 90
        assert gt["camus_biplane_ef_pct"] is not None
        assert s["citation"].endswith("doi:10.1109/TMI.2019.2900516")
        assert s["calibrated"] is True and 0.05 < s["mm_per_pixel"] < 1.0


@needs_camus
def test_both_views_of_a_patient_share_the_reference_biplane_value(listing):
    by_id = {s["id"]: s for s in listing["samples"]}
    for subj in (s for s in listing["subjects"] if s["kind"] == "camus"):
        a, b = by_id[subj["views"]["A4C"]], by_id[subj["views"]["A2C"]]
        assert a["ground_truth"]["reference_biplane_lvef_pct"] == b["ground_truth"]["reference_biplane_lvef_pct"]
        assert a["subject_id"] == b["subject_id"] == subj["id"]


@needs_camus
@needs_gpu
def test_biplane_estimate_appears_once_both_views_are_segmented(client):
    r4 = client.post("/api/segment", json={"sample_id": "camus_patient0017_4ch"}).json()
    r2 = client.post("/api/segment", json={"sample_id": "camus_patient0017_2ch"}).json()
    assert r2["view"] == "A2C" and r4["view"] == "A4C"
    bp = r2["biplane"]
    assert bp is not None and r2["biplane_lvef"] == bp["lvef_pct"]
    assert bp["paired_sample_id"] == "camus_patient0017_4ch" and bp["paired_view"] == "A4C"
    assert bp["edv_ml"] > bp["esv_ml"] > 0
    assert isinstance(bp["length_warning"], bool) and bp["length_mismatch_pct_ed"] >= 0
    # estimates and references stay in separate fields
    assert r2["single_plane_lvef"] == r2["lvef_pct"]
    assert r2["reference_single_plane_lvef"] is not None and r2["reference_biplane_lvef"] is not None
    assert r2["dataset_reported_biplane_ef"] == 58.0
    for r in (r4, r2):
        assert r["vram_mb"] < config.VRAM_BUDGET_MB
    # asking for A4C again now also reports the pair
    again = client.post("/api/segment", json={"sample_id": "camus_patient0017_4ch"}).json()
    assert again["biplane"] is not None and again["biplane"]["paired_view"] == "A2C"
    assert again["biplane"]["lvef_pct"] == pytest.approx(bp["lvef_pct"], abs=0.01)


def test_dataset_documentation_exists_and_separates_value_kinds():
    doc = (config.PROJECT_ROOT / "documentation" / "DATASETS.md").read_text()
    for needle in ("10.1109/TMI.2019.2900516", "not redistributed", "Algorithmic estimate",
                   "Dataset reference annotation", "Dataset-reported value", "patient0017", "ImageQuality"):
        assert needle in doc, needle


def test_frontend_syncs_views_with_sequences():
    js = (config.FRONTEND_DIR / "app.js").read_text()
    for needle in ("function updateSyncBadge", "subj.views[view]", "seqButtons", "No ${state.view} clip in the library"):
        assert needle in js, needle
