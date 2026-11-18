# EchoPlane3D — Autonomous Build & Verification Spec

## Environment & Hardware Guardrails
- OS: WSL2 Ubuntu 22.04.5 | GPU: NVIDIA RTX 3050 Laptop (6 GB VRAM) | Python 3.10 (`~/echo_env`).
- **Exact Local SAM2 Paths:**
  - Repo: `/home/burha/sam2`
  - Checkpoint: `/home/burha/sam2/checkpoints/sam2.1_hiera_tiny.pt`
  - Config: `configs/sam2.1/sam2.1_hiera_t.yaml`
- **Strict VRAM Budget:** Peak VRAM must stay under 2.5 GB. Always run `sam2.1_hiera_tiny.pt` in `torch.bfloat16` (or `float16`), and call `predictor.reset_state(inference_state)` + `torch.cuda.empty_cache()` after every video run.
- **Zero-Build Frontend:** Do NOT initialize npm/Webpack/Vite. Serve a sleek dark-mode single-page web app (`frontend/index.html` + `frontend/app.js`) directly from FastAPI using ES6 module CDNs (`Three.js`, `Plotly.js`, `TailwindCSS`).

## Verification-First Build Pipeline (Execute Sequentially & Verify Each Step)
1. **Configuration (`backend/core/config.py`):** Save the exact SAM2 checkpoint/config paths above and sample/cache directory paths.
2. **Mathematical Engine (`backend/core/simpsons_math.py`):** Implement 20-disk Modified Simpson's Rule integration (`compute_simpsons_20_disks`) using PCA long-axis eigenvector slicing + `simulate_apical_foreshortening(true_edv, true_esv, tilt_deg)`. Run a unit test on a synthetic prolate ellipsoid (R=25mm, L/2=40mm, exact V=104.72 mL) and assert <1.0% error.
3. **3D Multi-Chamber Heart Generator (`scripts/build_heart_glb.py`):** Procedurally generate an anatomically oriented 3D multi-chamber heart (`LV`, `LA`, `RV`, `RA`, `Aorta`) aligned to standard echo planes (`A4C`, `A2C`, `PLAX`, `PSAX`) using `trimesh`, and export to `frontend/models/heart_chambers.glb`. Run the script and verify the `.glb` loads cleanly.
4. **Sample Ultrasound Data (`scripts/prepare_samples.py`):** Generate 2 calibrated beating-ventricle phantom cine loops (Normal LVEF ~60% and Reduced HFrEF LVEF ~32% with known ground-truth EDV/ESV) in `backend/data/samples/` AND attempt to fetch 2 real open CAMUS ultrasound sequences if reachable without authentication; save `.mp4` + JSON `mm_per_pixel` metadata.
5. **SAM2.1 Video Tracker (`backend/core/sam2_tracker.py`):** Wrap `build_sam2_video_predictor` to accept a video path and Frame-0 click `[x, y]` (defaulting to auto-centroid if omitted), propagate masks across all frames, compute 20-disk Simpson volumes per frame, cache JSON in `backend/data/cache/`, and return `{frames_data, edv_ml, esv_ml, lvef_pct, ed_frame_idx, es_frame_idx, vram_mb, inference_time_sec}`. Benchmark on a 50-frame sample and verify VRAM < 2.5 GB.
6. **FastAPI Backend (`backend/main.py`):** Expose `/api/samples`, `/api/segment`, `/api/foreshortening`, and `/api/guide` (deterministic spatial tutor that switches probe views, explains foreshortening, and quizzes landmarks), and mount `frontend/` at `/`.
7. **Split-Screen 2D/3D Workbench (`frontend/index.html` & `frontend/app.js`):**
   - Left Pane: Interactive 2D Ultrasound canvas with click-to-segment prompt, playback controls, and toggleable overlays (SAM2 green mask, yellow long axis, 20 cyan Simpson chords).
   - Center Pane: `Three.js` 3D heart viewer (`heart_chambers.glb`) with `localClippingEnabled = true` and a live `THREE.Plane` + semi-transparent ultrasound sector fan slicing the heart along `A4C`, `A2C`, `PLAX`, `PSAX` plus an **Apical Foreshortening Tilt Slider (0°–30°)**.
   - Right Pane: Live `Plotly.js` Volume-Time curve $V(t)$, True vs. Foreshortened LVEF cards, and interactive Clinical Guide presets.
8. **End-to-End Test & Launcher:** Write `tests/test_e2e.py` to test all API endpoints + VRAM stability across 3 consecutive runs, run `pytest`, and create an executable `./run_echoplane.sh` launcher.
