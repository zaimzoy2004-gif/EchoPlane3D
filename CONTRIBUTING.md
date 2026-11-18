# Contributing to EchoPlane3D

Thank you for helping. EchoPlane3D is an **educational and research simulation prototype**, not a medical
device. Contributions must keep it that way: no diagnostic claims, no patient data in the repository.

## Ground rules

* **Never fabricate results.** Any number in documentation or `paper/` must be produced by
  `scripts/run_paper_benchmarks.py` (or a test) and be traceable to a file in `results/`.
* **Keep three kinds of number apart** in code, UI and text: algorithmic estimates, dataset reference
  annotations, and values reported with a dataset. See `documentation/DATASETS.md`.
* **Offline first.** The runtime must not contact the network. New front-end libraries are vendored under
  `frontend/vendor/` with their licence; no build step (npm, Webpack, Vite).
* **6 GB GPU.** Peak VRAM for a tracking run must stay below 2.5 GB (`backend/core/config.py`).
* **No cloud AI services** in the application. The optional tutor LLM is a local Ollama model, and the
  deterministic tutor must keep working without it.

## Development setup

See `README.md` → *Installation*. Then:

```bash
python scripts/prepare_samples.py          # phantoms (+ CAMUS demo cohort if reachable)
python scripts/build_heart_glb.py          # rebuild + verify the 3-D model
pytest -q                                  # GPU tests skip without CUDA
./run_echoplane.sh                         # http://127.0.0.1:8000
```

## Pull requests

1. One logical change per PR, with tests. Front-end changes: describe how you checked them in a browser.
2. `pytest -q` passes (CPU) — and, if you touched `backend/core/sam2_tracker.py`, also on a CUDA machine.
3. Update `CHANGELOG.md` under *Unreleased*.
4. Disclose generative-AI assistance in the PR description (tool, what it did, what you verified).

## Reporting problems

Open an issue with: OS, browser, GPU, the sample or clip (never upload identifiable patient data),
what you did, what you expected, and what happened (console errors, `/api/health` output).
