"""EchoPlane3D central configuration.

All filesystem paths and hardware guardrails live here so every other module
(tracker, API, scripts, tests) reads a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = FRONTEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"
UPLOADS_DIR = DATA_DIR / "uploads"          # user-supplied clips (never committed)
CAMUS_RAW_DIR = DATA_DIR / "camus_raw"      # downloaded CAMUS NIfTI files, kept so re-runs work offline
FRAMES_CACHE_DIR = CACHE_DIR / "frames"
RESULTS_DIR = PROJECT_ROOT / "results"

HEART_GLB_PATH = MODELS_DIR / "heart_chambers.glb"
HEART_PLANES_PATH = MODELS_DIR / "heart_planes.json"

# --------------------------------------------------------------------------- #
# SAM 2.1 (exact local paths — do not change without updating CLAUDE.md)
# --------------------------------------------------------------------------- #
# Defaults match the reference machine; override with ECHOPLANE_SAM2_REPO / ECHOPLANE_SAM2_CHECKPOINT elsewhere.
SAM2_REPO = Path(os.environ.get("ECHOPLANE_SAM2_REPO", "/home/burha/sam2"))
SAM2_CHECKPOINT = Path(
    os.environ.get("ECHOPLANE_SAM2_CHECKPOINT", str(SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"))
)
# Hydra config name, resolved relative to the installed `sam2` package.
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# --------------------------------------------------------------------------- #
# Hardware guardrails (RTX 3050 Laptop, 6 GB)
# --------------------------------------------------------------------------- #
DEVICE = "cuda"
AUTOCAST_DTYPE = "bfloat16"  # "bfloat16" or "float16"
VRAM_BUDGET_MB = 2500.0  # hard ceiling for peak VRAM per video run
# Keep decoded frames in system RAM; only the frame being encoded goes to GPU.
OFFLOAD_VIDEO_TO_CPU = True
OFFLOAD_STATE_TO_CPU = False
MAX_VIDEO_FRAMES = 300  # refuse absurdly long clips rather than blow RAM

# --------------------------------------------------------------------------- #
# Quantification
# --------------------------------------------------------------------------- #
N_SIMPSON_DISKS = 20
DEFAULT_MM_PER_PIXEL = 0.30  # only used when a video has no calibration JSON

# --------------------------------------------------------------------------- #
# Uploads (user clips)
# --------------------------------------------------------------------------- #
UPLOAD_EXTENSIONS = (".mp4", ".avi")
MAX_UPLOAD_MB = 200
MAX_UPLOAD_FRAMES = 90          # longer clips are temporally sub-sampled to protect the 6 GB GPU
MAX_UPLOAD_SIDE_PX = 800        # larger frames are down-scaled (calibration is rescaled with them)

# --------------------------------------------------------------------------- #
# Optional local LLM (Ollama).  Never a cloud API; the tutor works without it.
# --------------------------------------------------------------------------- #
OLLAMA_URL = os.environ.get("ECHOPLANE_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("ECHOPLANE_OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_TIMEOUT_SEC = 4.0        # strict connect/read timeout; on expiry the deterministic tutor answers
OLLAMA_ENABLED = os.environ.get("ECHOPLANE_OLLAMA", "auto").lower() not in ("0", "off", "false", "no")

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
HOST = "127.0.0.1"
PORT = 8000


def ensure_dirs() -> None:
    """Create every runtime directory (idempotent)."""
    for d in (MODELS_DIR, SAMPLES_DIR, CACHE_DIR, FRAMES_CACHE_DIR, UPLOADS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate_sam2_paths() -> dict:
    """Return a status dict describing whether the SAM2 assets are present."""
    cfg_file = SAM2_REPO / "sam2" / SAM2_CONFIG
    return {
        "repo": str(SAM2_REPO),
        "repo_exists": SAM2_REPO.is_dir(),
        "checkpoint": str(SAM2_CHECKPOINT),
        "checkpoint_exists": SAM2_CHECKPOINT.is_file(),
        "config": SAM2_CONFIG,
        "config_exists": cfg_file.is_file(),
    }
