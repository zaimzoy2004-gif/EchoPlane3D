# Datasets used by EchoPlane3D

EchoPlane3D is an educational and research simulation prototype. None of the data below, and none of the
numbers the software derives from them, are clinical measurements.

## 1. Calibrated phantoms (generated locally by `scripts/prepare_samples.py`; no download)

| Sample | Frames | Truth | How the truth is known |
|---|---|---|---|
| `phantom_normal` | 50 @ 25 fps, 512×512, 0.34 mm/px | EDV 120 mL, ESV 48.0 mL, LVEF 60 % | Analytic |
| `phantom_hfref` | 50 @ 25 fps, 512×512, 0.34 mm/px | EDV 210 mL, ESV 142.8 mL, LVEF 32 % | Analytic |

`scripts/prepare_samples.py` renders an apical-4-chamber-like layout with synthetic speckle. The LV cavity is the
exact long-axis section of a half prolate ellipsoid, so the volume at every frame is known in closed form
(V = 2/3·π·b²·a). The phantoms are synthetic images, not ultrasound physics simulations: there is no beam model,
attenuation model or realistic artefact.

## 2. CAMUS (real echocardiograms, downloaded on demand, **not redistributed**)

**Source.** CAMUS — *Cardiac Acquisitions for Multi-structure Ultrasound Segmentation* (University Hospital of
St Etienne, CREATIS). Files are fetched from the public Girder collection `CAMUS_public` at
<https://humanheart-project.creatis.insa-lyon.fr/database/>, which answers without authentication.

**Mandatory citation.** Every patient folder contains `MANDATORY_CITATION.md`, which requires citing:

> S. Leclerc, E. Smistad, J. Pedrosa, A. Østvik, et al. "Deep Learning for Segmentation using an Open Large-Scale
> Dataset in 2D Echocardiography", *IEEE Transactions on Medical Imaging* 38(9):2198–2210, 2019.
> doi:10.1109/TMI.2019.2900516

**Licence — check before you redistribute.** While preparing this document the dataset's web pages were fetched
and no machine-readable licence statement was found in them, so this repository makes no claim about the licence
beyond the mandatory citation above. Consequently:

* raw CAMUS files (`backend/data/camus_raw/`) and the clips derived from them (`backend/data/samples/camus_*`)
  are listed in `.gitignore` and are **not** part of the repository or of any release archive;
* anyone who wants the CAMUS samples runs `scripts/prepare_samples.py`, which downloads them from the original
  host under whatever terms that host states at the time;
* read the terms on the CAMUS site yourself before publishing figures or redistributing anything derived from it.

### 2.1 What is downloaded

Per patient and per apical view (`4CH` = A4C, `2CH` = A2C):

| File | Content |
|---|---|
| `Info_<view>.cfg` | `ED`, `ES` (1-based frame numbers), `NbFrame`, `Sex`, `Age`, `ImageQuality` (Good / Medium / Poor), `EF` (%), `FrameRate` |
| `<patient>_<view>_half_sequence.nii.gz` | Half cardiac cycle (ED→ES or ES→ED), grey-level frames |
| `<patient>_<view>_half_sequence_gt.nii.gz` | Per-frame labels: 1 = LV cavity, 2 = myocardium, 3 = left atrium |
| `<patient>_<view>_ED.nii.gz`, `_ES.nii.gz` and `_gt` | The two key frames with their labels (used by the paper benchmark) |

### 2.2 Demo cohort (8 patients × 2 views = 16 sequences)

Chosen to span ejection-fraction bands and image quality. `EF` and quality are copied from the dataset's own
`Info` files.

| Patient | Dataset-reported EF | Band | A4C quality | A2C quality |
|---|---|---|---|---|
| patient0017 | 58 % | normal (≥55 %) | Good | Good |
| patient0004 | 56 % | normal (≥55 %) | Medium | Good |
| patient0005 | 46 % | mildly reduced (41–49 %) | Good | Medium |
| patient0020 | 42 % | mildly reduced (41–49 %) | Medium | Good |
| patient0022 | 46 % | mildly reduced (41–49 %) | Poor | Medium |
| patient0038 | 21 % | reduced (≤40 %) | Good | Good |
| patient0019 | 33 % | reduced (≤40 %) | Medium | Good |
| patient0008 | 29 % | reduced (≤40 %) | Poor | Poor |

### 2.3 Three kinds of number — never mix them

| Kind | Where it comes from | Field names |
|---|---|---|
| **Algorithmic estimate** | SAM 2.1 mask → 20-disk Simpson, computed by this software | `single_plane_lvef`, `biplane_lvef`, `edv_ml`, `esv_ml` |
| **Dataset reference annotation** | The same Simpson code applied to the CAMUS LV masks | `reference_single_plane_lvef`, `reference_biplane_lvef`, `ground_truth.*` |
| **Dataset-reported value** | `EF` in `Info_<view>.cfg`, produced by the dataset's clinical readers | `dataset_reported_biplane_ef`, `ground_truth.camus_biplane_ef_pct` |

On the demo cohort the reference-mask biplane LVEF is 5–8 points higher than the dataset-reported EF for every
patient (for example 63.3 % vs 58 % for patient0017). The two are different measurements — different frames,
different tracing conventions, and the sequence masks are not the clinical tracings — so neither is treated as
ground truth for the other.

### 2.4 Processing applied to CAMUS

1. Frames are clipped to 0–255 and stored as 8-bit.
2. Anisotropic pixel spacing is resampled to square pixels (finer spacing kept); `mm_per_pixel` is recorded.
3. Frames are padded by at most one pixel to even dimensions (H.264 requirement); coordinates are unchanged.
4. Each sequence becomes `<id>.mp4` (H.264, CRF 12), `<id>.json` (metadata) and `<id>_gt.npz` (bit-packed LV masks).

### 2.5 Sample metadata (`<id>.json`)

`id`, `name`, `kind` (`phantom` | `camus` | `upload`), `view` (`A4C` | `A2C`), `subject_id`, `subject_name`,
`ef_class`, `image_quality`, `calibrated`, `volume_unit` (`mL` | `index`), `width`, `height`, `n_frames`, `fps`,
`mm_per_pixel`, `lv_seed_point`, `ground_truth` (reference EDV/ESV/LVEF, ED/ES frame indices, per-frame volumes,
reference biplane values, dataset-reported EF), `camus_info`, `source`, `citation`.

### 2.6 Benchmark cohort (`scripts/run_paper_benchmarks.py`)

A second, **held-out** CAMUS cohort is used only for validation: the first 10 patients per A4C `ImageQuality`
(Good / Medium / Poor) from `patient0041` onward, skipping the demo cohort. None of these patients was opened
while the prompting heuristics (auto-centroid, cavity snapping, candidate scoring) were developed; the exact list
is written to `results/benchmark_manifest.json`. For each patient the benchmark uses

* the expert-annotated ED and ES frames of both views (independent single-frame prompting, six strategies), and
* the full A4C and A2C half-cycle sequences (single frame-0 prompt + SAM 2.1 propagation, as in the workbench).

Prompts labelled "click" or "multi-point" are *simulated users*: they are derived from the reference mask
(deepest cavity point, points along the cavity's principal axis, nearest myocardium pixel). The "automatic"
strategy uses no reference information at all. Only derived metrics (Dice, HD95, volumes, LVEF) are written to
`results/`; no CAMUS image or mask is stored there.

## 3. User uploads

Clips uploaded through the UI are stored only in `backend/data/uploads/` on the local machine (git-ignored) and
can be deleted with `DELETE /api/upload/{id}`. They are sub-sampled to at most 90 frames. Without a `mm_per_pixel`
value they are *uncalibrated*: volumes are a pixel-scale index, not millilitres (LVEF is a ratio and is unaffected).
**Remove patient identifiers (burned-in names, dates, IDs) before uploading any clinical clip.**
