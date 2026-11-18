#!/usr/bin/env python3
"""Fill the manuscript templates with numbers read from ``results/``.

paper/templates/*.md contain ``{{key}}`` placeholders; this script computes every key
from the benchmark CSVs and writes paper/<name>.md plus paper/numbers.json.  No number
in the manuscripts is typed by hand, so they cannot drift from the data.

    python scripts/build_paper_numbers.py           # render
    python scripts/build_paper_numbers.py --check   # exit 1 if the rendered papers are stale
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
TEMPLATES = PAPER / "templates"
ROOT_FILES = {"README.md"}          # rendered to the repository root instead of paper/


def rows(name: str) -> list[dict]:
    with (RESULTS / name).open() as f:
        return list(csv.DictReader(f))


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def f1(v) -> str:
    return f"{num(v):.1f}"


def f2(v) -> str:
    return f"{num(v):.2f}"


def f3(v) -> str:
    return f"{num(v):.3f}"


def signed(v, d=1) -> str:
    x = num(v)
    return ("+" if x > 0 else "−" if x < 0 else "±") + f"{abs(x):.{d}f}"


def collect() -> dict[str, str]:
    k: dict[str, str] = {}

    # ---------------- Experiment A ----------------
    t1 = rows("table1_phantom_verification.csv")
    errs = [abs(num(r["error_pct"])) for r in t1]
    k["ellipsoid_n"] = str(len(t1))
    k["ellipsoid_exact_ml"] = f2(t1[0]["exact_volume_ml"])
    k["ellipsoid_worst_err_pct"] = f2(max(errs))
    k["ellipsoid_mean_err_pct"] = f2(statistics.mean(errs))
    sp = sorted({num(r["mm_per_pixel"]) for r in t1})
    k["ellipsoid_spacings"] = ", ".join(f"{s:.2f}" for s in sp)

    sw = rows("table1b_foreshortening_sweep.csv")

    def sweep(case, tilt, col):
        return next(r[col] for r in sw if r["case"] == case and int(r["tilt_deg"]) == tilt)

    for case, tag in (("phantom_normal", "n"), ("phantom_hfref", "h")):
        for tilt in (10, 15, 20, 30):
            k[f"fs_{tag}{tilt}_edv"] = signed(sweep(case, tilt, "edv_error_pct"), 0)
            k[f"fs_{tag}{tilt}_esv"] = signed(sweep(case, tilt, "esv_error_pct"), 0)
            k[f"fs_{tag}{tilt}_ef"] = signed(sweep(case, tilt, "lvef_error_pts"), 1)
    checked = [r for r in sw if r.get("model_vs_numerical_edv_pct") not in (None, "")]
    upto25 = [abs(num(r["model_vs_numerical_edv_pct"])) for r in checked if int(r["tilt_deg"]) <= 20]
    k["fs_model_vs_slice_worst_upto20_pct"] = f2(max(upto25))
    k["fs_model_vs_slice_n30_pct"] = signed(sweep("phantom_normal", 30, "model_vs_numerical_edv_pct"), 1)
    mesh = [r for r in sw if r.get("mesh_apparent_length_mm") not in (None, "")]
    k["mesh_worst_len_diff_pct"] = f2(max(abs(num(r["mesh_apparent_length_mm"]) - num(r["formula_length_mm"]))
                                          / num(r["formula_length_mm"]) * 100 for r in mesh))

    t1c = {r["sample"]: r for r in rows("table1c_phantom_tracking.csv")}
    for sid, tag in (("phantom_normal", "pn"), ("phantom_hfref", "ph")):
        r = t1c[sid]
        k[f"{tag}_est_lvef"] = f1(r["est_lvef_pct"])
        k[f"{tag}_true_lvef"] = f1(r["truth_lvef_pct"])
        k[f"{tag}_ef_err"] = signed(r["lvef_error_pts"], 1)
        k[f"{tag}_edv_err"] = signed(r["edv_error_pct"], 1)
        k[f"{tag}_esv_err"] = signed(r["esv_error_pct"], 1)
        k[f"{tag}_dice"] = f3(r["dice_mean"])

    # ---------------- Experiment B ----------------
    man = json.loads((RESULTS / "benchmark_manifest.json").read_text())
    cohort = man["camus_held_out_cohort"]
    k["camus_n_patients"] = str(sum(len(v) for v in cohort.values()))
    k["camus_n_per_quality"] = str(len(next(iter(cohort.values()))))
    allp = sorted(p for v in cohort.values() for p in v)
    k["camus_first_patient"], k["camus_last_patient"] = allp[0], allp[-1]
    frames = rows("camus_patient_metrics.csv")
    k["camus_n_frames"] = str(len({(r["patient"], r["view"], r["phase"]) for r in frames}))

    t2 = {(r["strategy"], r["sequence_quality"]): r for r in rows("table2_camus_validation.csv")}
    for s in ("naive_click", "snapped_click", "multi_point", "automatic", "naive_wall_click", "snapped_wall_click"):
        a = t2[(s, "All")]
        k[f"{s}_dice_med"] = f3(a["dice_median"])
        k[f"{s}_dice_iqr"] = f"{num(a['dice_q1']):.2f}–{num(a['dice_q3']):.2f}"
        k[f"{s}_dice_mean"] = f3(a["dice_mean"])
        k[f"{s}_hd95_med"] = f1(a["hd95_median_mm"])
        k[f"{s}_fail"] = str(int(num(a["failures_dice_below_0_5"])))
        k[f"{s}_hit_pct"] = f"{num(a['first_point_in_reference_lv_pct']):.0f}"
        for q in ("Good", "Medium", "Poor"):
            if (s, q) in t2:
                k[f"{s}_dice_med_{q.lower()}"] = f3(t2[(s, q)]["dice_median"])
                k[f"{s}_n_{q.lower()}"] = str(int(num(t2[(s, q)]["n_frames"])))

    t2b = rows("table2b_camus_agreement.csv")

    def agr(strategy, measure, comparator="reference masks"):
        return next(r for r in t2b if r["strategy"] == strategy and r["measure"] == measure
                    and r["comparator"].startswith(comparator))

    for s in ("naive_click", "snapped_click", "multi_point", "automatic"):
        for m in ("lvef_bp", "lvef_sp_a4c", "edv_bp", "esv_bp"):
            a = agr(s, m)
            k[f"{s}_{m}_mae"] = f1(a["mae"])
            k[f"{s}_{m}_bias"] = signed(a["bias"], 1)
            k[f"{s}_{m}_loa"] = f"{signed(a['loa_low'], 1)} to {signed(a['loa_high'], 1)}"
            k[f"{s}_{m}_icc"] = f2(a["icc_2_1"])
            k[f"{s}_{m}_r"] = f2(a["pearson_r"])
            k[f"{s}_{m}_n"] = str(int(num(a["n"])))
    rr = agr("reference_masks", "lvef_bp", "dataset-reported")
    k["ref_vs_reported_bias"] = signed(rr["bias"], 1)
    k["ref_vs_reported_mae"] = f1(rr["mae"])
    k["ref_vs_reported_icc"] = f2(rr["icc_2_1"])
    k["ref_vs_reported_r"] = f2(rr["pearson_r"])

    # held-out full-cycle tracking (the workbench's real pipeline)
    t2e = rows("table2e_heldout_tracking_agreement.csv")
    for p_ in ("automatic", "snapped_click"):
        for m in ("lvef_bp", "lvef_sp_a4c", "lvef_sp_a2c", "edv_bp", "esv_bp", "edv_sp_a4c", "esv_sp_a4c"):
            a = next(r for r in t2e if r["prompt"] == p_ and r["measure"] == m and r["comparator"].startswith("reference"))
            pre = f"trk_{p_}_{m}"
            k[f"{pre}_mae"] = f1(a["mae"])
            k[f"{pre}_bias"] = signed(a["bias"], 1)
            k[f"{pre}_loa"] = f"{signed(a['loa_low'], 1)} to {signed(a['loa_high'], 1)}"
            k[f"{pre}_icc"] = f2(a["icc_2_1"])
            k[f"{pre}_r"] = f2(a["pearson_r"])
            k[f"{pre}_n"] = str(int(num(a["n"])))
        a = next(r for r in t2e if r["prompt"] == p_ and r["comparator"].startswith("dataset-reported"))
        k[f"trk_{p_}_vs_reported_bias"] = signed(a["bias"], 1)
        k[f"trk_{p_}_vs_reported_mae"] = f1(a["mae"])
        k[f"trk_{p_}_plausible"] = str(int(num(a.get("n_physically_plausible_lvef"))))
    t2d = rows("table2d_heldout_tracking.csv")
    for p_ in ("automatic", "snapped_click"):
        rs = [r for r in t2d if r["prompt"] == p_]
        k[f"trk_{p_}_n_seq"] = str(len(rs))
        k[f"trk_{p_}_dice_med"] = f3(statistics.median(num(r["dice_mean"]) for r in rs))
        k[f"trk_{p_}_hd95_med"] = f1(statistics.median(num(r["hd95_median_mm"]) for r in rs))
        k[f"trk_{p_}_warnings"] = str(sum(r["quality_level"] == "warning" for r in rs))
        k[f"trk_{p_}_prompt_hit"] = str(sum(r["prompt_in_reference_lv"] == "True" for r in rs))
        for q in ("Good", "Medium", "Poor"):
            qs = [r for r in rs if r["sequence_quality"] == q]
            k[f"trk_{p_}_dice_med_{q.lower()}"] = f3(statistics.median(num(r["dice_mean"]) for r in qs))
            k[f"trk_{p_}_n_{q.lower()}"] = str(len(qs))
    frames_bad = {}
    for s_ in ("snapped_click", "multi_point", "automatic"):
        v = [r for r in rows("camus_patient_volumes.csv") if r["strategy"] == s_]
        frames_bad[s_] = sum(not (0 <= num(r["est_lvef_bp"]) <= 90) for r in v)
    k["frames_implausible_min"] = str(min(frames_bad.values()))
    k["frames_implausible_max"] = str(max(frames_bad.values()))

    show = RESULTS / "table2c_showcase_tracking.csv"
    if show.is_file() and show.stat().st_size:
        sc = rows("table2c_showcase_tracking.csv")
        for p in ("automatic", "snapped_click"):
            rs = [r for r in sc if r["prompt"] == p]
            k[f"show_{p}_n"] = str(len(rs))
            k[f"show_{p}_dice_med"] = f3(statistics.median(num(r["dice_mean"]) for r in rs))
            k[f"show_{p}_ef_mae"] = f1(statistics.mean(abs(num(r["est_lvef_sp"]) - num(r["ref_lvef_sp"])) for r in rs))
            k[f"show_{p}_dice_min"] = f3(min(num(r["dice_min"]) for r in rs))

    # ---------------- Experiment C ----------------
    t3 = rows("table3_hardware_telemetry.csv")
    track = [r for r in t3 if "full-cycle tracking" in r["run"]]
    k["tele_n_runs"] = str(len(track))
    k["tele_vram_peak_max"] = f"{max(num(r['vram_reserved_peak_mb']) for r in track):.0f}"
    k["tele_vram_alloc_max"] = f"{max(num(r['vram_allocated_peak_mb']) for r in track):.0f}"
    k["tele_smi_peak_max"] = f"{max(num(r['nvidia_smi_peak_mib']) for r in track):.0f}"
    k["tele_ms_per_frame_med"] = f"{statistics.median(num(r['ms_per_frame']) for r in track):.0f}"
    k["tele_fps_med"] = f1(statistics.median(num(r["fps"]) for r in track))
    k["tele_budget_ok"] = str(sum(r["within_2500mb_budget"] == "True" for r in track))
    k["tele_rss_peak_gb"] = f2(max(num(r["process_peak_rss_mb"]) for r in t3 if r.get("process_peak_rss_mb")) / 1024)
    load = next(r for r in t3 if r["run"] == "model load")
    k["tele_load_s"] = f1(load["inference_time_sec"])
    k["tele_load_vram"] = f"{num(load['vram_allocated_after_mb']):.0f}"
    k["tele_gpu_total"] = f"{num(load['gpu_total_mb']) / 1024:.1f}"
    emb = next(r for r in t3 if r["run"].startswith("single-frame image encoder"))
    dec = next(r for r in t3 if r["run"].startswith("single-frame prompt decode"))
    k["tele_embed_ms"] = f"{num(emb['ms_per_frame']):.0f}"
    k["tele_decode_ms"] = f"{num(dec['ms_per_frame']):.0f}"
    k["gpu_name"] = man.get("gpu", "n/a")
    k["torch_version"] = man.get("torch", "n/a")
    k["sam2_commit"] = (man.get("sam2_repo_commit") or "n/a")[:7]
    k["benchmark_total_min"] = f"{num(man['timings_sec']['total']) / 60:.0f}"
    k["benchmark_date"] = man["generated"][:10]

    # ---------------- tests ----------------
    tests = RESULTS / "test_summary.json"
    if tests.is_file():
        t = json.loads(tests.read_text())
        k["tests_passed"] = str(t["passed"])
        k["tests_total"] = str(t["total"])
    return k


def render(check: bool) -> int:
    nums = collect()
    stale = []
    for tpl in sorted(TEMPLATES.glob("*.md")):
        text = tpl.read_text()
        missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(nums))
        if missing:
            print(f"ERROR {tpl.name}: no value for {missing}", file=sys.stderr)
            return 2
        out = re.sub(r"\{\{(\w+)\}\}", lambda m: nums[m.group(1)], text)
        target = ROOT / tpl.name if tpl.name in ROOT_FILES else PAPER / tpl.name
        if check:
            if not target.is_file() or target.read_text() != out:
                stale.append(target.name)
        else:
            target.write_text(out)
            words = len(re.findall(r"\b\w[\w'’.-]*\b", re.split(r"^#+ References|^# References", out, flags=re.M)[0]))
            print(f"rendered {target.relative_to(ROOT)} ({words} words before references)")
    # figures referenced by the manuscripts are copied next to them (JOSS builds from paper/)
    (PAPER / "figures").mkdir(exist_ok=True)
    for fig in sorted((RESULTS / "figures").glob("fig*.png")):
        dst = PAPER / "figures" / fig.name
        if check:
            if not dst.is_file() or dst.read_bytes() != fig.read_bytes():
                stale.append(f"figures/{fig.name}")
        else:
            dst.write_bytes(fig.read_bytes())
    numbers = json.dumps(nums, indent=2, ensure_ascii=False, sort_keys=True)
    if check:
        if not (PAPER / "numbers.json").is_file() or (PAPER / "numbers.json").read_text() != numbers:
            stale.append("numbers.json")
        if stale:
            print("stale:", ", ".join(stale))
            return 1
        print("paper is in sync with results/")
        return 0
    (PAPER / "numbers.json").write_text(numbers)
    print(f"wrote paper/numbers.json ({len(nums)} values)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    raise SystemExit(render(ap.parse_args().check))
