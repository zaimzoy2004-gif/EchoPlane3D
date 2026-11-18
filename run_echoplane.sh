#!/usr/bin/env bash
# EchoPlane3D launcher.
#
#   ./run_echoplane.sh             start the workbench on http://127.0.0.1:8000
#   ./run_echoplane.sh --test      run the full pytest suite instead
#   ./run_echoplane.sh --rebuild   regenerate the heart model and samples first
#   ./run_echoplane.sh --check     verify the environment and assets, then exit
#
# Environment overrides: ECHO_PYTHON, ECHO_HOST, ECHO_PORT
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PY="${ECHO_PYTHON:-/home/burha/echo_env/bin/python3}"
HOST="${ECHO_HOST:-127.0.0.1}"
PORT="${ECHO_PORT:-8000}"
MODE="serve"
REBUILD=0

for arg in "$@"; do
  case "$arg" in
    --test) MODE="test" ;;
    --check) MODE="check" ;;
    --rebuild) REBUILD=1 ;;
    -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;36m[echoplane]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[echoplane]\033[0m %s\n' "$*" >&2; exit 1; }

[ -x "$PY" ] || die "Python environment not found at $PY (set ECHO_PYTHON)"
command -v ffmpeg >/dev/null || die "ffmpeg is required (sudo apt install ffmpeg)"

# --- environment + SAM 2.1 assets ------------------------------------------
"$PY" - <<'PYCHECK' || die "environment check failed"
import sys
import torch
from backend.core import config
s = config.validate_sam2_paths()
missing = [k for k in ("repo_exists", "checkpoint_exists", "config_exists") if not s[k]]
if missing:
    sys.exit(f"SAM 2.1 assets missing: {missing} -> {s}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available to PyTorch")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)} ({total:.1f} GB) | torch {torch.__version__} | "
      f"checkpoint {config.SAM2_CHECKPOINT.name} | VRAM budget {config.VRAM_BUDGET_MB:.0f} MB")
PYCHECK

# --- generated assets -------------------------------------------------------
if [ "$REBUILD" = 1 ] || [ ! -f frontend/models/heart_chambers.glb ] || [ ! -f frontend/models/heart_planes.json ]; then
  say "building the 3D heart model"
  "$PY" scripts/build_heart_glb.py
fi
if [ "$REBUILD" = 1 ] || [ ! -f backend/data/samples/phantom_normal.mp4 ] || [ ! -f backend/data/samples/phantom_hfref.mp4 ]; then
  say "generating sample ultrasound loops (phantoms + open CAMUS sequences if reachable)"
  "$PY" scripts/prepare_samples.py
fi

case "$MODE" in
  check)
    say "environment and assets OK"
    ;;
  test)
    say "running the test suite"
    exec "$PY" -m pytest
    ;;
  serve)
    if (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; then
      die "port $PORT on $HOST is already in use (set ECHO_PORT to pick another)"
    fi
    say "EchoPlane3D is starting on http://$HOST:$PORT  (Ctrl+C to stop)"
    say "under WSL2, open http://localhost:$PORT in your Windows browser"
    exec "$PY" -m uvicorn backend.main:app --host "$HOST" --port "$PORT" --log-level info
    ;;
esac
