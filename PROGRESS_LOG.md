# EchoPlane3D — Progress Log

Persistent tracker for the publication-readiness upgrade (SoftwareX / JOSS / arXiv / Zenodo).
Every number in this file was measured in this repository on the date shown; nothing is carried over from earlier summaries.

**Overall project: 100%**

## Phase tracker

| Phase | Name | Progress | Status |
|---|---|---|---|
| 1 | Comprehensive audit | `[##########]` 100% | Done 2026-11-15 |
| 2 | Bug fixes (UI viewport, camera geometry, segmentation prompting) | `[##########]` 100% | Done 2026-11-15 (A2C white screen hardened but never reproduced) |
| 3 | 3D cardiac anatomy & dynamic contraction | `[##########]` 100% | Done 2026-11-16 (schematic model, not patient-derived) |
| 4 | Clinical echo mathematical engine | `[##########]` 100% | Done 2026-11-16 |
| 5 | Multi-patient CAMUS workflow & 2D/3D sync | `[##########]` 100% | Done 2026-11-16 |
| 6 | Custom echo video upload | `[##########]` 100% | Done 2026-11-16 |
| 7 | Offline intelligent tutor & action control | `[##########]` 100% | Done (LLM path tested with mocked HTTP only; Ollama not installed) |
| 8 | Reproducible scientific validation | `[##########]` 100% | Done (single run, exit 0; LLM co-residency not measured — Ollama absent) |
| 9 | Open-source repository hygiene | `[##########]` 100% | Done (nothing committed: no git identity; author placeholders) |
| 10 | Scientific manuscripts | `[##########]` 100% | Drafted (author details + AI-review statements required) |

> The brief that arrived truncated was later supplied in full as `UPGRADE_SPEC.md`; all ten phases are now specified.

## Phase 1 — Audit (100%)

- [x] 25% — Directory tree inspected
- [x] 50% — Interpreter, packages, GPU and Ollama checked
- [x] 75% — pytest baseline run, features and debt catalogued
- [x] 100% — This summary written before any application code was changed

### Environment (measured 2026-11-15)

| Item | Finding |
|---|---|
| `/home/burha/echo_env/bin/python3` | Present, Python 3.10.12 |
| `/home/burha/echo_env/bin/pytest` | Present, pytest 9.2.0 |
| SAM 2.1 checkpoint / config | Both present at the paths in `backend/core/config.py` |
| Key packages | torch 2.6.0+cu124, torchvision 0.21.0+cu124, numpy 2.2.6, scipy 1.15.3, fastapi 0.162.5, uvicorn 0.54.0, trimesh 5.2.1, SimpleITK 2.5.7, tqdm 4.70.1, httpx 0.28.1 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144 MiB, 0 MiB in use at idle, driver 610.91 |
| CPU / RAM visible to WSL2 | 12 logical CPUs, **7.6 GB RAM** + 2 GB swap (the host has 16 GB; WSL2 is capped at its default share) |
| Ollama | **Not installed**: no `ollama` binary in WSL, no `ollama.exe` on the Windows PATH, `localhost:11434` and the host gateway both unreachable |
| Version control | **Not a git repository** |
| Node / Linux browser | None in WSL (UI checks so far used the Windows Chrome headlessly) |

### Baseline tests

`/home/burha/echo_env/bin/pytest` → **47 passed in 65.97 s** (34 math unit tests, 13 end-to-end tests).

### Repository inventory

| Path | Lines | Role |
|---|---|---|
| `backend/core/config.py` | 74 | Paths, VRAM budget, SAM2 settings |
| `backend/core/simpsons_math.py` | 356 | 20-disk Simpson, foreshortening model, synthetic shapes |
| `backend/core/sam2_tracker.py` | 530 | SAM 2.1 video tracker, auto-centroid, candidate-mask selection, cache |
| `backend/core/guide.py` | 407 | Rule-based (deterministic) tutor and quiz bank |
| `backend/main.py` | 221 | FastAPI app: samples, segment, foreshortening, guide, static frontend |
| `frontend/index.html`, `frontend/app.js` | 231 / 963 | Zero-build split-screen UI (canvas, Three.js, Plotly) |
| `frontend/models/` | — | `heart_chambers.glb` (5 chambers) + `heart_planes.json` |
| `scripts/` | 919 | Heart builder, sample preparation, tracker benchmark |
| `tests/` | 369 | `test_simpsons_math.py`, `test_e2e.py` |
| `backend/data/samples/` | — | 2 calibrated phantoms + 2 CAMUS A4C sequences (mp4 + json + reference masks) |
| `backend/data/cache/` | — | 14 cached segmentation JSONs, 4 extracted-frame folders (21 MB total with samples) |

### What works today

- Steps 1–8 of `CLAUDE.md` are implemented and covered by the 47 tests.
- `./run_echoplane.sh` (`--check`, `--test`, `--rebuild`) starts the workbench on port 8000.
- Tracker benchmark on record in `results/benchmark_tracker.json` (to be re-run before any figure is quoted in a manuscript).

### Gaps against the new brief (technical debt)

| # | Gap | Evidence |
|---|---|---|
| G1 | Not offline-capable | `frontend/index.html` loads Tailwind, Plotly and Three.js from three CDNs (lines 10, 25, 29–30) |
| G2 | No local LLM | Ollama absent; `guide.py` is keyword rules only |
| G3 | No VRAM co-residency profiling | Only SAM 2.1 alone has been measured; nothing with Ollama or a live WebGL context |
| G4 | Not under version control | `git status` fails; no checkpoint to roll back to |
| G5 | No packaging artefacts | No README, LICENSE, CITATION.cff, requirements/pyproject, CHANGELOG |
| G6 | Hard-coded user paths | `/home/burha/...` in `config.py`, `run_echoplane.sh`, `tests/test_e2e.py` |
| G7 | Thin real-data validation | 2 CAMUS patients; EDV error on patient0038 was about −43 % in the last benchmark; no Hausdorff distance; no multi-patient statistics |
| G8 | Frontend has no automated tests | No JS runner; tests only grep the served files |
| G9 | Camera transition never observed live | Headless Chrome under a virtual clock does not tick `requestAnimationFrame` |
| G10 | No progress output in long scripts | `prepare_samples.py`, `build_heart_glb.py` print only summaries |
| G11 | WSL RAM cap | 7.6 GB visible; relevant if Ollama is to run inside WSL alongside SAM 2.1 |
| G12 | Cache grows without bound | One JSON per distinct click |

## Phase 2 — Bug fixes (100%)

- [x] 33% — A2C white-screen in the 3D Probe Navigator: investigated, **could not be reproduced**, defensive fixes verified (details below)
- [x] 66% — 1080p chatbox layout
- [x] 100% — Cavity snapping + multi-point prompting

### 2.1 A2C white screen — what was actually found

**Status: not reproduced. The root cause on the reporting machine is still unknown.**

Reproduction attempt (Windows Chrome, headless, real clock so `requestAnimationFrame` ticks — 5,125 frames;
WebGL renderer reported by the browser: ANGLE / Intel UHD Graphics, Direct3D 11; window 1920×1080, DPR 1):

| Scenario | Result |
|---|---|
| A2C via the guide chip, via the typed sentence "Show me the apical 2-chamber view", via the view button | Renders normally |
| A2C reached from A4C, PLAX and PSAX, in both Anatomic and Echo-aligned camera modes | Renders normally |
| Rapid view changes mid-transition, tilt 25°, cut-away off/on | Renders normally |
| All 17 steps: near-white pixel fraction ≤ 0.002, camera matrices finite, no context loss, no console errors | — |

Numerical check of the camera maths from `heart_planes.json`: no home pose has `camera.up` within 30° of the
line of sight (A2C: 67° anatomic, 90° echo-aligned), and no view-to-view transition comes near a singularity.
So the hypothesis named in the brief (up-vector collinearity / NaN in `lookAt`) is **not** what the current
code does for A2C.

Not covered by the reproduction: a non-headless browser, a display scale other than 100 %, the NVIDIA GPU as
the WebGL adapter, and Ollama or SAM 2.1 holding VRAM at the same time (Ollama is not installed here).
A dropped WebGL context under GPU-memory pressure would blank the canvas and is the leading *unconfirmed*
candidate.

### 2.1 Changes made (`frontend/app.js`, backup in `backups/20261115-205413/`)

| Change | Verified by fault injection in the browser |
|---|---|
| `sanitizePose()` — rejects non-finite poses, eye-on-target, and `up` within ~7° of the line of sight | Pose with `up` exactly parallel to the view → renders (fallback up). Pose containing NaN → renders |
| `tweenPose()` / `slerpDir()` — the eye travels on an arc around the target instead of a straight chord | Start on the far side of the heart, select A2C → eye never closer than 560 mm to the centre (a chord would pass through it) |
| Render-loop watchdog — a non-finite camera is reset to the view's home pose | `camera.position` set to NaN → 1 recovery, view renders again |
| `webglcontextlost` / `webglcontextrestored` handlers — overlay message, redraw on restore | `WEBGL_lose_context.loseContext()` → overlay shown; `restoreContext()` → overlay hidden, scene renders again |

New tests: `tests/test_camera_geometry.py` (16 tests) — mirrors the pose arithmetic in Python and fails if any
view or transition becomes singular or passes through the heart.

### 2.2 Offline front end, chat layout and viewport hardening

- **Vendored libraries** in `frontend/vendor/`: Three.js r160 (+ OrbitControls, GLTFLoader, BufferGeometryUtils), Plotly 2.35.2, Tailwind play build 3.4.17. `index.html` references no remote URL.
- **Layout root cause found:** the app shell's single grid row was `auto`, so appending a chat message grew the row, which resized the 3D pane (a `ResizeObserver` → `renderer.setSize`, which clears the drawing buffer) and pushed the chat input below the fold. The row is now `minmax(0, 1fr)`; the chat history scrolls internally.
- **Viewport hardening:** opaque clear colour `0x0b0f19` set on every resize and every frame, invalid sizes (<16 px, non-finite) ignored, scene redrawn in the same task as a resize.
- This layout coupling is the most plausible cause of the reported white screen (long chat message + view switch at the same moment), but the white frame itself was **never reproduced**, so that remains an inference.

Browser check (headless Chrome, real clock, network blocked with a host-resolver rule), measured:

| Window | Viewport | Chat input bottom (limit = viewport − 12) | 3D canvas resizes while 19 chat bubbles were appended | Non-local requests |
|---|---|---|---|---|
| 1920×1080 | 1902×984 | 959 (limit 972) | 0 | 0 |
| 1536×864 | 1518×712 | 690 (limit 700) | 0 | 0 |

### 2.3 Cavity snapping and multi-point prompting

| Prompt on `Phantom · HFrEF` (truth 32.0 %) | LVEF | Quality flag |
|---|---|---|
| Wall click (358, 197), auto-snap on (moved 47 px into the cavity) | 32.31 % | good |
| Same click, auto-snap off | 12.84 % (API) / 12.46–13.15 % (through the canvas) | warning: non-compact mask, under-segmentation |
| Multi-point: 3 cavity + 1 wall point | 32.34 % | good |

One of three browser runs timed out once at the multi-point step at 1536×864 and passed on re-run; cause not identified.

## Phase 3 — 3D cardiac anatomy & dynamic contraction (100%)

- [x] 40% — Nine-structure heart rebuilt as implicit surfaces (marching cubes, 1.5 mm voxels): LV, IVS, RV (+RVOT), LA (+LAA), RA, Aorta (root, arch, 3 branch stubs, descending), PA (trunk + bifurcation), SVC, IVC
- [x] 70% — Plane intersections verified; labels kept apart
- [x] 100% — 3D LV contracts with V(t)

**Open-mesh check:** the BodyParts3D archive answered HTTP 200, but it was not used: it is a whole-body, share-alike (CC BY-SA 2.1 JP) dataset of organ walls with no cavity meshes registered to echo planes, so the fallback named in the spec (implicit modelling) was taken. The result is a schematic teaching model, **not** patient-derived anatomy; the LV cavity is deliberately an exact half ellipsoid so the 3D cut matches the foreshortening formula.

| Check (measured) | Result |
|---|---|
| GLB size / structures | 1.80 MB, 9 watertight single-surface meshes (52,100 vertices in total) |
| A4C / A2C / PLAX / PSAX cut | exactly {LV, IVS, RV, LA, RA} / {LV, LA} / {LV, IVS, RV, LA, Aorta} / {LV, IVS, RV} inside the sector |
| 3D cut length vs formula at 0/10/20/30° tilt | 0.22 / 0.12 / 0.14 / 0.14 % difference |
| Smallest on-screen gap between two visible labels (8 view × camera combinations) | 35 px (PSAX, IVS/RV) |
| LV mesh volume scale at end-systole vs measured ESV/EDV (phantom normal) | 0.3935 vs 0.3935; mitral-annulus displacement 0 mm |

## Phase 4 — Clinical echo mathematical engine (100%)

- [x] 30% — Single-plane and true biplane method of disks (`compute_biplane_simpson`, `biplane_volume_from_disks`)
- [x] 70% — Separately named estimate / reference fields and tracking-quality flags
- [x] 100% — Reference ED/ES shown as star markers + horizontal dashed levels (no diagonal line)

| Check (measured) | Result |
|---|---|
| Biplane on three triaxial ellipsoids (exact V = 4/3·π·a·b·40) | all within 1 % (test threshold) |
| Fields returned by `/api/segment` | `single_plane_lvef`, `biplane_lvef`, `reference_single_plane_lvef`, `reference_biplane_lvef`, `dataset_reported_biplane_ef`, `quality` |
| Quality flags (heuristics, not a validated score) | `volume_jump` (>20 % of EDV between frames), `under_segmentation` (<50 % of reference area), `over_segmentation`, `non_compact_mask`, `empty_frames`, `implausible_lvef` |
| Plot on a CAMUS clip (browser) | traces: "Reference ED / ES" (2 stars) + measured curve; shapes: cursor + 2 dashed horizontals at 60.94 and 25.25 mL |

Biplane LVEF needs both apical views of one subject; that pairing is exercised in Phase 5 once A2C clips exist.

## Phase 5 — Multi-patient CAMUS workflow & 2D/3D sync (100%)

- [x] 40% — Demo cohort: 8 CAMUS patients × (A4C + A2C) + 2 phantoms = **18 sequences / 10 subjects**; raw files cached in `backend/data/camus_raw/` (70 MB) for offline re-runs
- [x] 80% — Bidirectional sync: 3D A4C/A2C ⇄ the subject's matching 2D clip; informational badge for PLAX/PSAX and for subjects with one view
- [x] 100% — `documentation/DATASETS.md`

| Check (measured) | Result |
|---|---|
| Cohort bands / qualities | normal, mildly reduced, reduced × Good, Medium, Poor (per-view quality from the dataset's Info files) |
| Sync in the browser (7 transitions: 3D button, 2D 4C/2C button, chat command, PLAX, PSAX) | all correct |
| Biplane pairing (patient0017, both views auto-segmented) | biplane LVEF 67.8 % returned with a long-axis-mismatch warning (29.5 % at ED) and a quality warning on the A2C clip |
| Reference-mask biplane vs dataset-reported EF | reference is 5–8 points higher for all 8 patients (different measurements; documented, not reconciled) |

**Finding and fix — automatic prompt.** The auto-centroid heuristic picked the left atrium on 2 of 16 real clips
(both A2C, LV wall drop-out). A fallback for un-enclosed cavities and a sharper depth prior were added; it now
hits the LV on 18/18 demo sequences. That is a tuned-on-these-clips number; the Phase 8 benchmark measures it on
patients it was not tuned on. The CAMUS licence could not be confirmed from the dataset's pages, so CAMUS-derived
files are kept out of the repository (see `documentation/DATASETS.md`).

## Phase 6 — Custom echo video upload (100%)

- [x] 50% — `POST /api/upload` (.mp4 / .avi, ≤200 MB, ≤90 frames after temporal sub-sampling, ≤800 px side), optional `mm_per_pixel`, `DELETE /api/upload/{id}`
- [x] 100% — `Upload Echo` button + form; uploads appear under "Your uploads" and support click, multi-point, tracking and 3D sync

| Check (measured) | Result |
|---|---|
| 150-frame clip | stored as 90 frames, duration preserved, flagged `subsampled` |
| Calibrated upload of the 32 % phantom (0.34 mm/px), click prompt, in the browser | LVEF 32.43 %, EDV 200.9 mL |
| Uncalibrated upload of the 60 % phantom, auto prompt, in the browser | LVEF 60.63 %, volume shown as an index (2907 idx), "LV volume index" axis and HUD label |
| Rejections | wrong extension, undecodable bytes, empty file, out-of-range calibration, non-apical view → HTTP 422 |
| Delete | removes only `upload_<hash>` files; bundled samples untouched |

Uploaded clips stay on the local disk (git-ignored). The form warns to remove patient identifiers first.

## Phase 7 — Offline intelligent tutor & structured action control (100%)

- [x] 50% — Hybrid engine in `backend/core/guide.py`: deterministic knowledge base (4 views, 9 structures, foreshortening geometry, single-plane vs biplane, LVEF interpretation, prompting help, probe orientation, 16 quiz questions with running score) + optional Ollama adapter (`/api/chat`, `ECHOPLANE_OLLAMA_MODEL`, default `qwen2.5:1.5b`, 4 s timeout, `keep_alive: 2m`)
- [x] 100% — Engine badge in the header and guide panel; chat commands control views, tilt, overlays and sample loading

**Separation enforced in code:** quizzes, grading, preset buttons and UI commands never reach the LLM. Every number is
produced by `simpsons_math` and shown in a separate *Verified calculation* block; LLM text is tagged
"local LLM · explanation only". LLM actions pass a Pydantic whitelist (`set_view`, `set_tilt`, `toggle_overlay`,
`start_quiz`, `select_sample` with an existing sample id only) — anything else is dropped.

| Check (measured) | Result |
|---|---|
| Mocked-Ollama tests: valid JSON, 6 malformed outputs, fenced JSON, read time-out, HTTP 500, disabled by config | used / all fall back / accepted / falls back once then marked unreachable / falls back / never called |
| Hostile LLM actions (`run_shell`, unknown sample, out-of-range tilt) | rejected |
| 13 typed commands in the browser (offline) | every view / tilt / overlay / sample change applied; 0 errors; 3D canvas resized 0 times during chat |
| Ollama on this machine | **not installed** — only the deterministic engine has run for real; the LLM path is verified with mocked HTTP only |

## Phase 8 — Reproducible scientific validation (100%)

One uninterrupted run of `scripts/run_paper_benchmarks.py` (exit 0, 4,164 s, 2026-11-18; provenance in
`results/benchmark_manifest.json`: SAM 2 commit `2b90b9f`, checkpoint SHA-256, torch 2.6.0+cu124, RTX 3050 6 GB).
"Reference" = CAMUS expert masks measured with the same Simpson code — not clinical ground truth.

- [x] 25% — **Experiment A.** Ellipsoid (exact 104.72 mL): worst |error| **0.400 %** over 25 spacing × rotation
  cases (reproduced independently). Foreshortening closed form vs numerical plane slice agree up to 25°;
  **finding:** at ≈30° the PCA long axis flips to the annulus (section wider than tall, height/half-width < ≈1.89),
  so the 20-disk value departs from the model — documented as a limitation. Phantom tracking: LVEF 60.65 % (truth 60)
  and 32.33 % (truth 32), EDV −4.6 %, mean Dice 0.984 / 0.986.
- [x] 65% — **Experiment B**, 30 held-out patients (10 Good / 10 Medium / 10 Poor by A4C quality, patient0041–0120).
  * ED/ES frames, 120 frames × 6 strategies — median Dice: naive click **0.275** (stock SAM picks the whole sector),
    snapped click **0.667**, multi-point **0.744**, automatic **0.646**; click on the wall: naive 0.275 → snapped 0.561.
    Independent ED and ES segmentation gives inconsistent pairs (ES > ED in 4–7/30 patients): biplane LVEF MAE
    22.9–64.9 points — **not usable for LVEF**.
  * **Full-cycle tracking (the workbench's real pipeline)**, 120 sequences: biplane LVEF vs reference —
    snapped click MAE **9.7** points, bias +4.7, 95 % LoA −21.5 to +30.9, ICC(2,1) 0.58;
    automatic MAE **12.9**, bias +3.6, LoA −28.4 to +35.6, ICC 0.48; all 60 values physically plausible.
    Single-plane A4C EDV is under-estimated (bias ≈ −44 mL).
  * Reference-mask biplane LVEF vs dataset-reported EF: bias +8.3 points, r 0.97 (the two references differ).
  * Demo cohort (tuned-on, reported separately): median Dice 0.707 (automatic) / 0.686 (snapped).
- [x] 85% — **Experiment C** (157 telemetry rows): peak VRAM reserved ≤ **866 MB** on all 154 tracking runs
  (budget 2,500 MB; `nvidia-smi` ≤ 979 MiB); median 173 ms/frame (5.8 fps) on CAMUS, 262 ms/frame on 512×512
  phantoms; image encoder 182 ms/frame, prompt decode + selection 45 ms; model load 4.2 s, 289 MB; peak process
  RSS 2.4 GB of the 7.6 GB visible to WSL2. Ollama was not reachable, so LLM co-residency was **not** measured.
- [x] 100% — Figures 1–4 (300 DPI PNG + PDF) in `results/figures/`; statistics validated against Shrout & Fleiss
  (ICC 0.29 reproduced) in `tests/test_benchmark_stats.py`.

## Phase 9 — Open-source repository hygiene (100%)

- [x] 50% — `git init` (branch `main`), `.gitignore`, MIT `LICENSE`, `THIRD_PARTY_NOTICES.md`, `CITATION.cff`
  (author placeholders), `CONTRIBUTING.md`, `CHANGELOG.md`, pinned `requirements.txt`, `environment.yml`, `README.md`
- [x] 100% — `.github/workflows/ci.yml` (CPU), `documentation/ARCHITECTURE.md`, `DATASETS.md`, `PUBLICATION_GUIDE.md`

| Check (measured) | Result |
|---|---|
| CI replay: clean virtualenv, CPU PyTorch 2.6.0, no SAM 2, no GPU, no CAMUS, workflow steps in order | 172 passed, 10 skipped (GPU / CAMUS only); CPU benchmark reproduces Table 1 byte-for-byte |
| Bug found by the replay | `--skip-gpu` crashed hashing a missing checkpoint — fixed |
| `.gitignore` bug found while auditing | trailing comments silently disabled the CAMUS / samples / cache patterns — fixed |
| Staged set | 98 files, ≈ 15 MB; no CAMUS file, clip, NIfTI or cache entry |
| Publication rules checked on 2026-11-18 | JOSS: ≥ 6 months public history (**blocker: repo not public yet**), AI-usage disclosure mandatory. SoftwareX: template, ≤ 6 figures, 3,000–4,000 words (sources disagree). Elsevier GenAI declaration above references |

**Not done, needs you:** no git identity is configured on this machine, so nothing was committed; author names,
affiliations, ORCIDs, repository URL and support e-mail are placeholders; decide whether `CLAUDE.md` /
`UPGRADE_SPEC.md` stay in the public repository.

## Phase 10 — Scientific manuscripts (100% drafted; author input required)

- [x] 50% — `paper/paper.md` (JOSS, 1,313 words before references) + `paper/paper.bib` (12 entries; 11 DOIs
  resolved on Crossref / DataCite, author lists taken from the registries — one memory error, a co-author's first
  name, was corrected that way)
- [x] 100% — `paper/softwarex_manuscript.md` (2,370 words before references, 5 figures cited in order, code
  metadata table C1–C9, ethics / data governance, generative-AI declaration, data availability)

Every number in the manuscripts and in the README's validation table is a `{{placeholder}}` filled from
`results/` by `scripts/build_paper_numbers.py` (363 values); `tests/test_paper_sync.py` fails on any drift or on a
hand-typed decimal inside a results section. Figure 5 (workbench screenshot) uses the synthetic phantom so no CAMUS
image is redistributed.

**Placeholders the authors must fill:** names, affiliations, corresponding author, repository URL (C2), Zenodo DOI
(C3), support e-mail (C9), funding, competing interests, and the human-review statement in both AI disclosures.

## Change log

| Date | Phase | Change | Verification |
|---|---|---|---|
| 2026-11-15 | 1 | Audit written; no application code changed | pytest 47/47 |
| 2026-11-15 | 2.1 | Camera pose guard, arc transitions, NaN watchdog, WebGL context-loss recovery; 16 camera-geometry tests | pytest 63/63 (was 47/47); 17-step real-time browser run + 5 fault injections, 0 errors |
| 2026-11-15 | 2 | Offline vendoring, layout fix, viewport hardening, cavity snap, multi-point prompts, candidate scoring, quality flags | pytest 75/75; browser checks at 1920x1080 and 1536x864 offline |
| 2026-11-16 | 3 | Implicit-surface 9-structure heart, plane verification, V(t)-driven LV contraction | pytest (see below); browser check of 8 view/camera combinations + contraction |
| 2026-11-16 | 4 | Biplane Simpson, estimate/reference field separation, quality flags, star + dashed reference markers | biplane unit tests; browser check of plot traces on a CAMUS clip |
| 2026-11-16 | 5 | 8-patient two-view CAMUS cohort, subject pairing, 2D/3D sync, DATASETS.md, auto-centroid fallback | pytest (see below); browser sync check; auto-centroid 18/18 on the demo cohort |
| 2026-11-16 | 6 | Upload endpoint and UI, frame cap, calibration handling, delete endpoint | 9 upload tests; browser upload of calibrated and uncalibrated clips |
| 2026-11-18 | 7 | Hybrid deterministic + Ollama tutor, Pydantic action whitelist, engine badge, chat-driven overlays and sample loading | 61 tutor tests; 13-command offline browser check |
| 2026-11-18 | 8 | run_paper_benchmarks.py: phantoms, 30 held-out CAMUS patients (frames x 6 strategies + full-cycle tracking), demo-cohort tracking, telemetry, 4 figures, manifest | benchmark exit 0 in 4,164 s; 6 statistics tests |
| 2026-11-18 | 9 | Licence, notices, CITATION, CONTRIBUTING, CHANGELOG, pinned deps, CI workflow, docs, git init + staging | CI replay in a clean CPU virtualenv: 172 passed, 10 skipped |
| 2026-11-18 | 10 | JOSS paper + bib, SoftwareX manuscript, README validation — all rendered from results/ | paper sync tests pass; 363 values extracted |
