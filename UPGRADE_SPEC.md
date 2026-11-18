# EchoPlane3D — Remaining Master Specification (Phase 2 [66%] through Phase 10)

## Context & Answers to Your Phase 1 Audit Questions
1. **Where the A2C White Screen Happened:**
   - Browser: Windows 11 Google Chrome (`localhost:8000` over WSL2 port forwarding) on a 1080p laptop display with Windows display scaling.
   - State when it happened: `Phantom - HFrEF (LVEF 32%)` was loaded after an uncached SAM2.1 segmentation (`peak VRAM 838 MB`), and the user clicked the chatbox action `"Show me the apical 2-chamber view"`, which simultaneously appended a long message to the right-hand chat log and switched the 3D view to `A2C`.
   - Additional hardening required: In addition to your pose guard, NaN watchdog, and WebGL context-loss restore, inspect whether appending chat messages in the right column alters flex/grid layout height and triggers a `ResizeObserver` or window resize callback on the Three.js canvas with transient invalid dimensions or clears the drawing buffer without re-rendering. Ensure `renderer.setClearColor(0x0b0f19, 1.0)` is enforced on every resize and frame render.
2. **WSL2 7.6 GB RAM Constraint & Offline Ollama Strategy:**
   - Because WSL2 allocates 7.6 GB total RAM and `ollama` is not currently installed (and installing it requires `sudo`), do NOT block on `sudo` installation of Ollama.
   - Instead:
     a) Build the complete HTTP client adapter in `backend/core/guide.py` supporting `http://localhost:11434/api/chat` (defaulting to lightweight `qwen2.5:1.5b` or `qwen2.5:3b` via environment variable `ECHOPLANE_OLLAMA_MODEL`, with a strict 4-second connect/read timeout and Pydantic action validation).
     b) Greatly expand the **Offline Deterministic Clinical Tutor (`backend/core/guide.py`)** into a rich, state-aware clinical knowledge base and natural-language intent matcher (covering A4C, A2C, PLAX, PSAX, apical foreshortening physics, single-plane vs. biplane Simpson's equations, EDV/ESV/LVEF interpretation, chamber/vessel anatomy, and interactive multi-question quizzes) so it works 100% offline with 0 MB extra RAM/VRAM when Ollama is absent.
     c) Test the Ollama adapter in `pytest` using mocked `httpx` responses (testing valid JSON action extraction, malformed LLM output rejection, and timeout fallback to the deterministic tutor).
3. **True 100% Offline Frontend (`frontend/vendor/`):**
   - Download the exact production JS/CSS bundles for `Three.js`, `Plotly.js`, and `Tailwind` (or compile/extract the required utility CSS) into `frontend/vendor/` and reference them locally in `frontend/index.html` so EchoPlane3D works with zero internet connectivity.

---

## Mandatory Progress Reporting Rule
Continue printing at every sub-step:
`[PROGRESS] Phase X (Name): Y% complete — <what finished & what is next> | Overall Project: Z%`
And update `/home/burha/echoplane3d/PROGRESS_LOG.md` continuously.

---

## Remaining Phases to Execute Sequentially

### PHASE 2 (Remaining Items: 66% -> 100%)
- **Item 2 (66%) — Fix 1080p Right-Sidebar Chatbox Layout Clipping (`frontend/index.html` & CSS):**
  - On a `1920x1080` viewport (and smaller laptop viewports like `1536x864` at 125% Windows scaling), the bottom-right `CLINICAL GUIDE` panel pushes the chat input field and `Send` button to the very bottom edge where the Windows taskbar clips it.
  - Constrain the main app shell to `height: 100vh; overflow: hidden;` (or compact internal panel heights: reduce Plotly chart height slightly, make the chat message history scroll internally with `flex: 1; min-height: 0; overflow-y: auto;`, and pin the chat input bar with a safe bottom margin/padding) so the input box and `Send` button are 100% visible at both `1920x1080` and `1536x864`. Verify via headless Chrome bounding-box assertions (`inputRect.bottom <= viewportHeight - 12`).
- **Item 3 (100%) — Fix Myocardial Wall vs. LV Blood Cavity Mis-Segmentation (`backend/core/sam2_tracker.py` & `frontend/app.js`):**
  - When a user clicks on the bright myocardial wall (such as `(358, 197)` on `Phantom - HFrEF`), SAM2 segments the horseshoe wall instead of the dark LV cavity (`12.8%` LVEF instead of `32.0%`).
  - Implement **Sector-Constrained Cavity Snapping + Multi-Point Prompting**:
    1. When a single positive click `(x, y)` arrives on a bright pixel inside the ultrasound sector, search a local neighborhood (e.g., radius 25–35 px inside the fan mask) or radial inward trajectory toward the sector centerline for the dark, homogeneous blood-pool region bounded by echogenic myocardium, and score SAM2's 3 multimask candidates using dark-lumen intensity contrast, solidity, and central sector containment.
    2. Add a regression test explicitly passing click `(358, 197)` on `Phantom - HFrEF` and asserting that the resulting LVEF lands within `32.0% ± 3.0%` (not `12.8%`).
    3. Support **Multi-Point Prompting** (`points: [[x1,y1], ...]`, `labels: [1, 1, 0, ...]`) in `/api/segment` and the frontend (e.g., Shift+Click or a `Multi-point` toggle for positive cavity points and negative wall points, plus an option to disable auto-snap for raw manual clicks).

### PHASE 3: Upgrade 3D Cardiac Anatomy & Dynamic Contraction (0% -> 100%)
1. **(40%) Anatomical 3D Heart + Great Vessels (`scripts/build_heart_glb.py`):**
   - Replace the simple geometric capsules/spheres with an anatomically credible, smooth, organic multi-structure 3D cardiac mesh (`frontend/models/heart_chambers.glb`, <6 MB).
   - Check if an open anatomical heart surface mesh (e.g., from BodyParts3D / MedShapeNet / open anatomical OBJ/STL/GLB archives) can be fetched cleanly; if not, use high-resolution anatomical parametric/implicit surface modeling (marching cubes over smooth signed distance fields / metaballs + anatomical splines with realistic wall thickness, chamber tapering, and sulci) to generate distinct, watertight named meshes for:
     - `LV` (conical tapered apex, mitral inlet, aortic outlet)
     - `RV` (crescentic cross-section wrapping anteromedially around the septum, infundibulum/RVOT)
     - `IVS` (interventricular septum between LV and RV)
     - `LA` (posterior superior chamber with Left Atrial Appendage `LAA` protrusion)
     - `RA` (right lateral atrium)
     - `Aorta` (aortic root, ascending aorta, arch with 3 supra-aortic branch stubs, descending thoracic aorta)
     - `SVC` and `IVC` (superior and inferior vena cava entering RA vertically)
     - `PA` (main pulmonary artery trunk arising from RVOT and bifurcating under the aortic arch)
2. **(70%) Plane Intersection Verification:**
   - Verify that `A4C`, `A2C`, `PLAX`, and `PSAX` planes intersect the clinically expected chambers and vessels, and update the legend/labels cleanly without overlapping text.
3. **(100%) Synchronized 3D LV Contraction Driven by $V(t)$ (`frontend/app.js`):**
   - On every playback frame $t$, deform/scale the 3D `LV` mesh smoothly between ED and ES in proportion to the active sample's instantaneous volume $V(t)$ (keeping the basal mitral plane anchored while contracting radially and longitudinally).

### PHASE 4: Clinical Echocardiography Mathematical Engine (0% -> 100%)
1. **(30%) ASE/EACVI Guideline Verification (`backend/core/simpsons_math.py`):**
   - Implement both **Single-Plane Method of Disks** ($V = \frac{\pi}{4}\sum_{i=1}^{20} D_i^2 \frac{L}{20}$) and **True Biplane Method of Disks** ($V = \frac{\pi}{4}\sum_{i=1}^{20} a_i^{(\text{A4C})} b_i^{(\text{A2C})} \frac{\max(L_{\text{A4C}}, L_{\text{A2C}})}{20}$) per Lang et al. (ASE/EACVI 2015).
2. **(70%) Explicit Distinction of Estimates vs. Reference Values:**
   - Clearly label and return: `single_plane_lvef`, `biplane_lvef` (when both A4C and A2C are segmented for the same patient), `reference_single_plane_lvef`, and `reference_biplane_lvef`.
   - Flag low-quality/noisy tracking (e.g., large frame-to-frame volume jumps or cavity area <50% of reference) with a clinical quality warning badge in the UI.
3. **(100%) Plotly ED/ES Benchmark Markers (`frontend/app.js`):**
   - Replace the diagonal line between CAMUS ED and ES reference volumes on the Plotly chart with distinct **horizontal dashed reference lines / star markers** at the exact ED and ES frames.

### PHASE 5: Multi-Patient CAMUS Dataset Workflow & 2D/3D Sync (0% -> 100%)
1. **(40%) Expand Curated Demo Cohort to 8 CAMUS Patients (`scripts/prepare_samples.py`):**
   - Import **8 real CAMUS patients** covering Normal LVEF ($\ge 55\%$), Mildly Reduced ($41\text{–}49\%$), and Reduced HFrEF ($\le 40\%$) across `Good`, `Medium`, and `Poor` image qualities, importing **both `A4C` (`4CH`) and `A2C` (`2CH`)** sequences + ground-truth masks + metadata for each patient (plus the 2 synthetic phantoms = 18 selectable sequences / 10 subjects).
2. **(80%) Bidirectional 2D/3D View Synchronization (`frontend/app.js`):**
   - Clicking `A4C` or `A2C` in the 3D Probe Navigator automatically switches the active CAMUS patient's 2D ultrasound video to their matching `A4C` or `A2C` sequence (and displays a clear informational badge when `PLAX` or `PSAX` is selected in 3D).
3. **(100%) Dataset Provenance Documentation:**
   - Document CAMUS dataset licensing and metadata structure in `documentation/DATASETS.md`.

### PHASE 6: Custom Echocardiography Video Upload (0% -> 100%)
1. **(50%) Backend Upload Endpoint (`POST /api/upload` in `backend/main.py`):**
   - Accept user-uploaded `.mp4` or `.avi` ultrasound clips (validating format, capping/downsampling frame count to $\le 90$ frames to protect 6 GB VRAM), extract frames and resolution, accept optional `mm_per_pixel` calibration (defaulting to clearly labeled **Uncalibrated Pixel-Scale Volume Index** while noting LVEF% scale-invariance).
2. **(100%) Frontend Upload Modal/Button (`frontend/index.html` & `frontend/app.js`):**
   - Add an `Upload Echo` button in the 2D Ultrasound header allowing instant upload, Frame-0 click/multi-point prompting, SAM2 tracking, and 3D synchronization. Test via `pytest` with a generated `.mp4` upload.

### PHASE 7: Offline Intelligent Tutor & Structured Action Control (0% -> 100%)
1. **(50%) Hybrid Offline Guide Engine (`backend/core/guide.py`):**
   - Implement the local Ollama adapter (`http://localhost:11434/api/chat`) + Pydantic command whitelist (`set_view`, `set_tilt`, `toggle_overlay`, `start_quiz`, `select_sample`) with automatic, instantaneous fallback to the expanded **Deterministic Spatial Tutor**.
2. **(100%) UI Engine Badge & Validation Tests:**
   - Display `Offline Deterministic Tutor` (or `Ollama Local LLM` when reachable) in the UI header, ensure chat commands smoothly control the 3D probe and overlays, and verify with unit tests.

### PHASE 8: Reproducible Scientific Validation (`scripts/run_paper_benchmarks.py`) (0% -> 100%)
1. **(25%) Experiment A — Analytical Phantom & Foreshortening Benchmark:**
   - Independently run and export `results/table1_phantom_verification.csv` (5 pixel spacings $\times$ 5 rotations + tilt sweep $0^\circ\text{–}30^\circ$).
2. **(65%) Experiment B — CAMUS Cohort Validation (30 Patients, A4C + A2C = 60 Sequences):**
   - Fetch/evaluate 30 CAMUS patients stratified by official `ImageQuality` (`Good`, `Medium`, `Poor`).
   - Compare (1) Naive Single-Point, (2) Cavity-Snapped Single-Point, and (3) Multi-Point (3 positive cavity + 1 negative wall) prompting on ED and ES frames (plus full-cycle tracking on the 8-patient showcase cohort).
   - Compute and export **DSC**, **$\text{HD}_{95}$ (mm)**, **Single-Plane & Biplane EDV/ESV/LVEF MAE**, **Pearson $r$**, **$\text{ICC}(2,1)$**, and **Bland-Altman Bias & 95% LoA** into `results/camus_patient_metrics.csv` and `results/table2_camus_validation.csv`.
3. **(85%) Experiment C — RTX 3050 6GB Hardware Telemetry:**
   - Log actual peak VRAM (MB), system RAM (MB), per-frame latency (ms), and FPS into `results/table3_hardware_telemetry.csv`.
4. **(100%) Publication Figures (300 DPI PNG + PDF in `results/figures/`):**
   - Generate `fig1_architecture_overview.png`, `fig2_bland_altman_lvef.png`, `fig3_prompt_ablation_by_quality.png`, and `fig4_foreshortening_sensitivity.png`.

### PHASE 9: Publication-Ready Open-Source Repository Hygiene (0% -> 100%)
1. **(50%) Initialize Git Repository & Standard Files:**
   - Initialize `git init` (if not initialized), create `.gitignore`, `LICENSE` (MIT), `CITATION.cff`, `CONTRIBUTING.md`, `CHANGELOG.md`, `requirements.txt`, `environment.yml`, and comprehensive `README.md`.
2. **(100%) CI Workflow & Documentation (`documentation/`):**
   - Create `.github/workflows/ci.yml` (CPU test suite), `documentation/ARCHITECTURE.md`, `documentation/DATASETS.md`, and `documentation/PUBLICATION_GUIDE.md` (covering SoftwareX, JOSS 6-month history & AI disclosure rules, arXiv/medRxiv, and Zenodo DOI archiving).

### PHASE 10: Scientific Manuscript Preparation (`paper/`) (0% -> 100%)
1. **(50%) JOSS Manuscript (`paper/paper.md` & `paper/paper.bib`):**
   - Write a complete 1,200–1,600 word JOSS paper with verified DOIs in `paper/paper.bib`, populated strictly with the real measured numbers from `results/`.
2. **(100%) SoftwareX Manuscript (`paper/softwarex_manuscript.md`):**
   - Write the full Elsevier SoftwareX structured manuscript (Abstract, Metadata Table, Motivation & Significance, Software Description & Architecture, Empirical Validation & Benchmarks, Illustrative Examples, Impact & Limitations, AI-Assisted Development Disclosure, and Ethics/Data Governance).
