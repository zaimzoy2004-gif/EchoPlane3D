# Changelog

All notable changes are listed here. Dates are when the change was made and verified on the reference machine
(RTX 3050 Laptop 6 GB, WSL2 Ubuntu 22.04, Python 3.10).

## [Unreleased]

### Added
- `scripts/run_paper_benchmarks.py`: one-pass reproducible validation — analytic phantoms and foreshortening,
  30 held-out CAMUS patients (ED/ES frames × 6 prompting strategies, plus full-cycle tracking of 120
  sequences), demo-cohort tracking, hardware telemetry, five figures and a provenance manifest.
- `scripts/build_paper_numbers.py`: renders `README.md`, `paper/paper.md` (JOSS) and
  `paper/softwarex_manuscript.md` from `results/`; `tests/test_paper_sync.py` fails if they drift.
- `scripts/record_test_summary.py`, `paper/paper.bib` (DOIs and author lists taken from Crossref/DataCite).
- Repository hygiene: MIT licence, third-party notices, CITATION.cff, CONTRIBUTING, CPU CI workflow
  (replayed in a clean virtualenv), `requirements.txt`, `environment.yml`, architecture / dataset /
  publication documentation.

### Changed
- `predict_frame_mask` can reuse an existing image embedding (benchmark speed; same code path as the app).
- UI: the two foreshortening lengths are labelled "3-D model LV length" and "fitted ED length".

### Fixed
- `run_paper_benchmarks.py --skip-gpu` crashed without a SAM 2 checkpoint (found by the CI replay); CPU-only
  runs now write `benchmark_manifest_cpu_only.json` instead of overwriting the GPU manifest.
- Tests no longer assume the reference machine's SAM 2 path.

### Known limitations (measured, see `paper/`)
- PCA long axis flips on sections wider than tall (≈30° foreshortening).
- Biplane LVEF limits of agreement vs CAMUS reference masks are wide (educational use only).

## [1.1.0] — 2026-11-18

### Added
- Fully offline front end: Three.js, Plotly and Tailwind vendored in `frontend/vendor/`.
- Cavity snapping, SAM candidate scoring and multi-point prompts (`points` / `labels`, Shift+click = negative).
- Biplane method of disks; estimates, reference values and dataset-reported values in separate fields;
  heuristic tracking-quality warnings.
- Nine-structure implicit-surface heart (adds IVS, PA, SVC, IVC, LAA, RVOT); LV contracts with V(t).
- Eight-patient CAMUS demo cohort with both apical views; 2-D/3-D view synchronisation.
- Clip upload (`POST /api/upload`) with frame cap and uncalibrated "volume index" mode.
- Hybrid tutor: expanded deterministic knowledge base, optional local Ollama adapter with action whitelist.

### Fixed
- Wall click on the HFrEF phantom segmented the myocardium (LVEF 12.8 % instead of 32 %).
- Appending chat messages resized the WebGL canvas and pushed the chat input off-screen at 1080p.
- Camera hardening against singular poses, NaN matrices and WebGL context loss (the reported A2C white
  screen was investigated but never reproduced).

## [1.0.0] — 2026-11-15

### Added
- Initial workbench: SAM 2.1 tracking, 20-disk Simpson, foreshortening model, five-chamber heart,
  two phantoms + two CAMUS clips, deterministic tutor, FastAPI backend, launcher, 47 tests.
