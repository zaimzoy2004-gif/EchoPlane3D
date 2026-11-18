#!/usr/bin/env python3
"""Run the test suite and record its outcome in results/test_summary.json (quoted by the manuscripts)."""
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
res = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
line = [l for l in res.stdout.splitlines() if re.search(r"\d+ (passed|failed)", l)][-1]
counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|skipped|errors?)", line)}
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    gpu = None
summary = {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0), "skipped": counts.get("skipped", 0),
           "errors": counts.get("error", 0) + counts.get("errors", 0),
           "total": sum(counts.values()), "summary_line": line.strip("= "), "gpu": gpu,
           "date": datetime.date.today().isoformat()}
(ROOT / "results" / "test_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
sys.exit(res.returncode)
