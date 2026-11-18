# EchoPlane3D architecture

EchoPlane3D is a single-machine, offline workbench: a FastAPI process serves a zero-build web page and runs
all computation locally. Nothing is sent to any external service.

```
Browser (vendored Three.js r160, Plotly 2.35.2, Tailwind 3.4.17)
 ├─ 2-D pane     canvas playback, click / multi-point prompts, mask + long axis + 20 chords, upload form
 ├─ 3-D pane     GLB heart, THREE.Plane clipping + stencil caps, probe + sector fan, tilt slider, LV beat
 └─ right pane   Plotly V(t), estimate-vs-reference table, foreshortening cards, tutor chat
        │  JSON over http://127.0.0.1:8000
FastAPI (backend/main.py)
 ├─ GET  /api/health                 GPU, SAM 2.1 assets, tutor engine
 ├─ GET  /api/samples                sequences + subjects (A4C/A2C pairs)
 ├─ GET  /api/samples/{id}/video     mp4;  /frames/{i}.jpg  decoded frames
 ├─ POST /api/segment                prompt -> SAM 2.1 tracking -> Simpson -> biplane pairing
 ├─ POST /api/upload, DELETE /api/upload/{id}
 ├─ GET|POST /api/foreshortening     half-ellipsoid model + tilt sweep + foreshortened V(t)
 └─ GET|POST /api/guide              hybrid tutor turn
        │
backend/core
 ├─ sam2_tracker.py   prompt handling, cavity snapping, candidate scoring, propagation, quality flags, cache
 ├─ simpsons_math.py  single-plane + biplane method of disks, foreshortening model, synthetic shapes
 ├─ guide.py          deterministic knowledge base + optional Ollama adapter + Pydantic action whitelist
 ├─ camus.py          CAMUS download / cache / NIfTI loading
 ├─ uploads.py        clip validation, frame cap, calibration
 ├─ video_io.py       H.264 encoding / decoding helpers
 └─ config.py         paths, VRAM budget, model and tutor settings (env-overridable)
```

## Segmentation pipeline (`sam2_tracker.segment_video`)

1. **Decode** the clip once into `backend/data/cache/frames/<stem>_<hash>/00000.jpg …` (SAM 2's native input).
2. **Prompt** on frame 0 — one of
   * *automatic*: `auto_centroid()` finds the largest dark region enclosed by tissue in the upper-central sector,
     with a fallback to the deepest dark point when wall drop-out leaves the LV unenclosed;
   * *single click*: optionally moved by `snap_to_cavity()` from bright tissue into the nearest blood pool
     (within ~34 px on a 512-px frame, else along the scan line toward the sector centre line);
   * *multi-point*: positive (cavity) and negative (wall / atrium) points, used as given.
3. **Resolve SAM's ambiguity.** The image head returns three candidate masks. `choose_cavity_mask()` keeps
   candidates that honour every point and cover ≤ 50 % of the sector, and scores them by
   *(rim − interior intensity contrast) × (1 − bright fraction) × solidity × sector containment*.
   Stock SAM's choice (highest predicted IoU) is available as `select="naive"` for comparison.
4. **Propagate** the chosen mask with the SAM 2.1 video predictor (`add_new_mask` + `propagate_in_video`).
5. **Quantify** every frame: largest component, hole filling, 20-disk Simpson along the mask's PCA long axis,
   area-equivalent disk diameters with anti-aliased slab binning.
6. **Report** EDV (max), ESV (min), LVEF, per-frame geometry for the overlays, heuristic quality flags,
   and GPU telemetry; cache the JSON by video hash + prompt + calibration + tracker version.

**VRAM discipline.** hiera-tiny in bf16 autocast (float16 fallback); decoded frames stay in system RAM
(`offload_video_to_cpu=True`); `predictor.reset_state()` and `torch.cuda.empty_cache()` after every run;
a process-wide lock serialises GPU work. Budget: 2.5 GB peak reserved per run.

## Volumetry (`simpsons_math.py`)

* **Single plane**: `V = π/4 · Σ Dᵢ² · L/20`.
* **Biplane** (Lang et al., ASE/EACVI 2015): `V = π/4 · Σ aᵢ·bᵢ · max(L_A4C, L_A2C)/20`; the long-axis
  mismatch is reported (warning above 10 %). The API pairs the latest A4C and A2C results of one subject;
  ED and ES are taken per view because the two clips are different beats.
* **Long axis** = first principal axis of the mask. Known limitation: on a section that is wider than tall
  (height/half-width < ≈1.89, e.g. an apical plane tilted ≈30° off axis) the axis flips to the annular
  direction; a landmark-based apex-to-annulus axis would be needed there.
* **Foreshortening model**: the cavity is a half prolate ellipsoid; a plane through the mitral-annulus centre
  tilted by θ sees length `a′ = 1/√(cos²θ/a² + sin²θ/b²)`, so Simpson reports `V·a′/a`. Shape over the cycle:
  `a ∝ (V/EDV)^0.25`, `b ∝ (V/EDV)^0.375`.

## 3-D model (`scripts/build_heart_glb.py`)

Nine named, watertight meshes (LV, IVS, RV, LA, RA, Aorta, PA, SVC, IVC) built as smooth unions of ellipsoids
and spline tubes, polygonised with marching cubes (1.5 mm voxels). It is a schematic teaching model, not
patient anatomy; the LV cavity is kept an exact half ellipsoid so the 3-D cut agrees with the formula above.
`heart_planes.json` stores the probe origin, beam, screen-right and normal of each standard view in the same
frame. The builder verifies orientation, which structures each plane cuts, label spacing and the 3-D cut length
against the closed form.

## Tutor (`guide.py`)

* Deterministic engine: regex + fuzzy intent matching over a fixed knowledge base; every number it states is
  computed by `simpsons_math`, shown in a separate *verified calculation* block. Pure function of its inputs.
* Optional Ollama (`ECHOPLANE_OLLAMA_URL`, `ECHOPLANE_OLLAMA_MODEL`, default `qwen2.5:1.5b`; disable with
  `ECHOPLANE_OLLAMA=off`): wording only, 4 s timeout, `keep_alive` 2 min so it releases VRAM, JSON output whose
  actions must pass the whitelist. Quizzes, grading, presets and UI commands never reach the LLM. Any failure
  falls back to the deterministic reply.

## Front-end robustness

* Opaque clear colour re-applied on every frame and resize; invalid sizes ignored; redraw in the same task as
  a resize (a resize clears the WebGL buffer).
* App-shell grid row `minmax(0, 1fr)` so appending chat never resizes the 3-D canvas.
* Camera: pose sanitiser (finite, eye ≠ target, `up` ≥ 7° from the line of sight), arc interpolation around
  the target, a render-loop watchdog for non-finite matrices, and WebGL context-loss / restore handling.

## Tests

`pytest` covers the maths (synthetic ellipsoids, biplane, foreshortening vs numerical slicing), the heart model,
camera geometry, the API, prompting regressions, uploads, the tutor (with mocked Ollama HTTP) and the
benchmark statistics (ICC against Shrout & Fleiss 1979). GPU and CAMUS tests skip automatically when the
hardware or data are absent.
