#!/usr/bin/env python3
"""Benchmark the SAM 2.1 tracker on every sample: accuracy vs ground truth,
speed and peak VRAM (PyTorch counters + nvidia-smi polled in the background).

Writes results/benchmark_tracker.json and exits non-zero if the VRAM budget
is exceeded on any run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.core import config  # noqa: E402
from backend.core import sam2_tracker as trk  # noqa: E402
from scripts.prepare_samples import load_gt_masks  # noqa: E402


class SmiPoller(threading.Thread):
    """Poll total GPU memory in use (MiB) via nvidia-smi."""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval, self.peak, self._stop_evt = interval, 0.0, threading.Event()

    @staticmethod
    def read() -> float:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()[0]
            return float(out)
        except Exception:
            return float("nan")

    def run(self):
        while not self._stop_evt.is_set():
            v = self.read()
            if v == v:
                self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / max(a.sum() + b.sum(), 1))


def main() -> int:
    config.ensure_dirs()
    smi_idle = SmiPoller.read()
    samples = sorted(p for p in config.SAMPLES_DIR.glob("*.json") if p.name != "manifest.json")
    rows, ok = [], True
    print(f"nvidia-smi before model load: {smi_idle:.0f} MiB in use")
    trk.get_predictor()
    print(f"model loaded: torch allocated {trk.gpu_memory_mb()['allocated_mb']:.0f} MiB")
    for side in samples:
        meta = json.loads(side.read_text())
        video = config.SAMPLES_DIR / meta["video"]
        poller = SmiPoller()
        poller.start()
        res = trk.segment_video(video, click=None, force=True, return_masks=True)
        poller.stop()
        gt = meta["ground_truth"]
        gt_masks = load_gt_masks(config.SAMPLES_DIR / meta["gt_masks"])
        d = [dice(res["_masks"][i], gt_masks[i]) for i in range(meta["n_frames"])]
        gx, gy = (int(round(v)) for v in res["click"])
        row = {
            "sample": meta["id"],
            "n_frames": meta["n_frames"],
            "click": res["click"],
            "click_inside_gt_lv": bool(gt_masks[0][gy, gx]),
            "edv_ml": res["edv_ml"], "gt_edv_ml": gt["edv_ml"],
            "esv_ml": res["esv_ml"], "gt_esv_ml": gt["esv_ml"],
            "lvef_pct": res["lvef_pct"], "gt_lvef_pct": gt["lvef_pct"],
            "edv_err_pct": round((res["edv_ml"] - gt["edv_ml"]) / gt["edv_ml"] * 100, 2),
            "esv_err_pct": round((res["esv_ml"] - gt["esv_ml"]) / gt["esv_ml"] * 100, 2),
            "lvef_err_pts": round(res["lvef_pct"] - gt["lvef_pct"], 2),
            "ed_frame_idx": res["ed_frame_idx"], "es_frame_idx": res["es_frame_idx"],
            "dice_mean": round(float(np.mean(d)), 4), "dice_min": round(float(np.min(d)), 4),
            "vram_mb": res["vram_mb"], "vram_allocated_mb": res["vram_allocated_mb"],
            "vram_after_mb": res["vram_after_mb"],
            "nvidia_smi_peak_mb": poller.peak,
            "inference_time_sec": res["inference_time_sec"], "fps_inference": res["fps_inference"],
            "autocast_dtype": res["autocast_dtype"],
        }
        rows.append(row)
        within = res["vram_mb"] < config.VRAM_BUDGET_MB and (poller.peak == 0 or poller.peak < config.VRAM_BUDGET_MB)
        ok &= within
        print(
            f"{row['sample']:24s} {row['n_frames']:3d} fr | click {row['click']} in-LV={row['click_inside_gt_lv']} | "
            f"EDV {row['edv_ml']:6.1f} (GT {row['gt_edv_ml']:6.1f}, {row['edv_err_pct']:+5.1f}%) "
            f"ESV {row['esv_ml']:6.1f} (GT {row['gt_esv_ml']:6.1f}, {row['esv_err_pct']:+5.1f}%) "
            f"LVEF {row['lvef_pct']:5.1f}% (GT {row['gt_lvef_pct']:5.1f}%, {row['lvef_err_pts']:+.1f} pts) | "
            f"Dice {row['dice_mean']:.3f} (min {row['dice_min']:.3f}) | "
            f"VRAM peak reserved {row['vram_mb']:.0f} MB / allocated {row['vram_allocated_mb']:.0f} MB / "
            f"nvidia-smi {row['nvidia_smi_peak_mb']:.0f} MiB | {row['inference_time_sec']:.1f} s "
            f"({row['fps_inference']:.1f} fps) {row['autocast_dtype']} | {'OK' if within else 'OVER BUDGET'}"
        )
    out = config.RESULTS_DIR / "benchmark_tracker.json"
    out.write_text(json.dumps({"vram_budget_mb": config.VRAM_BUDGET_MB, "nvidia_smi_idle_mb": smi_idle, "runs": rows}, indent=2))
    print(f"wrote {out}")
    print("VRAM budget:", "PASS (< %.0f MB on every run)" % config.VRAM_BUDGET_MB if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
