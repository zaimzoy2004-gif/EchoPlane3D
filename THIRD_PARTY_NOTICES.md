# Third-party components

| Component | Version | Where | Licence |
|---|---|---|---|
| Three.js (+ OrbitControls, GLTFLoader, BufferGeometryUtils) | r160 | `frontend/vendor/three/` | MIT (`frontend/vendor/three/LICENSE`) |
| Plotly.js | 2.35.2 | `frontend/vendor/plotly-2.35.2.min.js` | MIT |
| Tailwind CSS (play/CDN build) | 3.4.17 | `frontend/vendor/tailwindcss-play-3.4.17.js` | MIT |
| SAM 2 / SAM 2.1 (code + `sam2.1_hiera_tiny.pt`) | commit `2b90b9f` | installed separately, **not** bundled | Apache 2.0 (Meta) |
| PyTorch, NumPy, SciPy, OpenCV, FastAPI, trimesh, scikit-image, SimpleITK, matplotlib … | see `requirements.txt` | installed with pip | their respective licences |

The vendored JavaScript files are unmodified copies of the upstream release files, kept in the repository so
the workbench runs with no network connection.

**Data.** The CAMUS dataset is *not* part of this repository. `scripts/prepare_samples.py` and
`scripts/run_paper_benchmarks.py` download it from its original host, whose terms apply; cite
Leclerc et al., IEEE TMI 38(9):2198–2210, 2019 (doi:10.1109/TMI.2019.2900516). See `documentation/DATASETS.md`.
