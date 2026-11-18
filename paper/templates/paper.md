---
title: 'EchoPlane3D: an offline split-screen workbench coupling a video foundation model with 3-D cardiac plane slicing for echocardiography training'
tags:
  - Python
  - JavaScript
  - echocardiography
  - medical education
  - segmentation
  - Simpson's method of disks
  - apical foreshortening
authors:
  - name: Burhan Zaim
    affiliation: 1
affiliations:
  - name: Faculty of Medicine and Health, Al-Farabi Kazakh National University, Almaty, Kazakhstan
    index: 1
date: {{benchmark_date}}
bibliography: paper.bib
---

# Summary

EchoPlane3D is an open-source, single-machine workbench for teaching the quantitative side of
transthoracic echocardiography. One browser window shows three linked panes. The first plays an apical
clip in which the left-ventricular (LV) cavity is segmented and tracked through the cardiac cycle by the
SAM 2 video foundation model [@ravi2024sam2], and draws the long axis and the 20 disks of Simpson's
method of disks. The second shows a 3-D heart cut by the imaging plane of the four standard views — apical
four-chamber (A4C), apical two-chamber (A2C), parasternal long axis (PLAX) and parasternal short axis
(PSAX) — with a probe, a sector fan, and a slider that tilts the apical plane off the true apex. The third
plots the volume–time curve, compares single-plane and biplane ejection fraction with the dataset's
reference values, shows what a foreshortened plane would have reported, and hosts a tutor that switches
views, explains, and quizzes anatomical landmarks. Everything runs offline on a consumer laptop GPU. The
software is an educational and research prototype, not a medical device.

# Statement of need

Left-ventricular ejection fraction (LVEF) from the biplane method of disks is the guideline 2-D measure of
systolic function [@lang2015chamber], and its best-known error source is geometric: an apical plane that
misses the true apex shortens the long axis and under-estimates both volumes. Learners are usually told
this rather than shown it, because the 3-D relation between probe, plane and ventricle is invisible on the
2-D screen. Simulator-based teaching of transthoracic echocardiography is effective
[@neelankavil2012simulation], but simulators are costly, closed, and rarely expose the quantification
pipeline. Deep-learning research tools for automated LVEF [@ouyang2020echonet] and promptable segmentation
models for medical images [@kirillov2023sam; @ma2024medsam] are open, but they are not teaching
environments and give no spatial intuition.

EchoPlane3D fills that gap for educators and researchers: it makes the plane geometry, the disk
summation and the effect of foreshortening visible and manipulable side by side, on open data (CAMUS,
@leclerc2019camus) or on calibrated phantoms with analytic truth, without a network connection, cloud
service, or build tool chain.

# Functionality

* **Tracking.** A click (or a fully automatic prompt) on frame 0 yields three candidate masks from SAM 2's
  image head; EchoPlane3D keeps the candidate that best matches a blood pool (dark interior, bright rim,
  compact, inside the sector), optionally first moving a click that landed on bright myocardium into the
  adjacent cavity, then propagates the mask through the clip. Multi-point prompts with negative points are
  supported. Peak GPU memory is kept below a configurable budget by running in bfloat16 and resetting the
  video state after every run.
* **Volumetry.** Single-plane and biplane method of disks [@lang2015chamber] along the mask's principal
  axis, with per-frame overlays, heuristic tracking-quality warnings, and a closed-form foreshortening model
  (a half-ellipsoid cavity cut by a tilted plane) that also drives a foreshortened V(t) curve.
* **3-D navigator.** Nine named structures generated as implicit surfaces and polygonised with marching
  cubes [@lorensen1987marching], cut in real time by a clipping plane with stencil caps (three.js,
  @threejs). The LV mesh contracts with the measured V(t).
* **Data.** Two calibrated phantom cine loops with analytic volumes, an eight-patient CAMUS demo cohort
  with both apical views (downloaded on demand), and upload of the user's own clips.
* **Tutor.** A deterministic, state-aware knowledge base whose every number is computed by the volumetry
  engine; an optional local LLM (Ollama) may reword explanations, but its proposed UI actions must pass a
  schema whitelist and quizzes, grading and calculations never depend on it.

# Validation

<!-- numbers-from-results -->
All numbers come from one run of `scripts/run_paper_benchmarks.py` on an {{gpu_name}} ({{benchmark_date}},
{{benchmark_total_min}} min) and are regenerated into this paper by `scripts/build_paper_numbers.py`.
"Reference" means the CAMUS expert masks measured with the same Simpson code; it is not clinical ground truth.

*Analytic shapes.* On a prolate ellipsoid of exactly {{ellipsoid_exact_ml}} mL the 20-disk implementation
erred by at most {{ellipsoid_worst_err_pct}} % over {{ellipsoid_n}} pixel-spacing × rotation cases. The
foreshortening model matched an independent numerical plane slice to within {{fs_model_vs_slice_worst_upto20_pct}} %
(end-diastolic volume) up to 20° of tilt; at 30° the principal axis of the now wider-than-tall section flips to the annular
direction and the slice departs from the model by {{fs_model_vs_slice_n30_pct}} %, a limitation of a
landmark-free long axis. On the two calibrated phantoms the tracker reported LVEF {{pn_est_lvef}} % (truth
{{pn_true_lvef}} %) and {{ph_est_lvef}} % (truth {{ph_true_lvef}} %), mean Dice {{pn_dice}} and {{ph_dice}}.

*Real data.* {{camus_n_patients}} CAMUS patients that were not used to develop or tune the prompting
heuristics ({{camus_first_patient}}–{{camus_last_patient}}, {{camus_n_per_quality}} per A4C image-quality
grade) were evaluated. On the {{camus_n_frames}} expert-annotated end-diastolic and end-systolic frames, stock SAM
(highest predicted IoU) reached a median Dice of only {{naive_click_dice_med}} because it usually returns
the whole sector; EchoPlane3D's candidate selection raised this to {{snapped_click_dice_med}} for a single
click, {{multi_point_dice_med}} for three cavity points and one wall point, and {{automatic_dice_med}}
with no reference input at all. A click placed on the myocardium improved from {{naive_wall_click_dice_med}}
to {{snapped_wall_click_dice_med}} with snapping. Segmenting the two frames independently, however, gave
physically impossible LVEF in {{frames_implausible_min}}–{{frames_implausible_max}} of
{{camus_n_patients}} patients. Tracking the whole half-cycle from one frame-0 prompt — the workbench's real
pipeline — avoided this ({{trk_snapped_click_plausible}} of {{camus_n_patients}} plausible) and gave a
biplane LVEF mean absolute error of {{trk_snapped_click_lvef_bp_mae}} points (bias
{{trk_snapped_click_lvef_bp_bias}}, 95 % limits of agreement {{trk_snapped_click_lvef_bp_loa}}, ICC(2,1)
{{trk_snapped_click_lvef_bp_icc}} [@shrout1979icc; @bland1986agreement]) for a simulated click and
{{trk_automatic_lvef_bp_mae}} points (ICC {{trk_automatic_lvef_bp_icc}}) for the automatic prompt.
These limits are far wider than clinical tolerance; the tool is meant to show *why* measurements vary,
not to replace them.

*Hardware.* Across {{tele_n_runs}} tracking runs the peak GPU memory reserved by PyTorch never exceeded
{{tele_vram_peak_max}} MB (median {{tele_ms_per_frame_med}} ms per frame) on a {{tele_gpu_total}} GB laptop GPU.
<!-- /numbers-from-results -->

# Reproducibility and design constraints

The repository contains everything needed to regenerate the validation: `scripts/prepare_samples.py`
rebuilds the phantoms and downloads the CAMUS sequences it needs, `scripts/build_heart_glb.py` rebuilds and
verifies the 3-D model, and `scripts/run_paper_benchmarks.py` writes every table and figure together with a
provenance manifest (model commit, checkpoint hash, library versions, timings). The text of this paper is
rendered from those tables, and a test fails if the two disagree. Continuous integration runs the
CPU-only part of the test suite on every push; GPU tests run on the reference laptop. Two constraints
shaped the design throughout: the whole system must work without a network connection, and a tracking run
must stay within a fixed GPU-memory budget so that it fits on a 6 GB consumer laptop next to a browser.

# Limitations

The 3-D heart is a schematic teaching model, not patient anatomy. Volumes are single-plane or biplane
approximations whose accuracy depends on SAM 2's masks, which under-fill hazy cavities. The demo cohort
was used while tuning the prompting heuristics, so only the held-out cohort is used for the figures above.
The optional LLM path was tested with mocked HTTP only, and GPU co-residency with a local LLM was not
measured. The software has not been evaluated with learners.

# AI usage disclosure

Most of the source code, tests, documentation and the first draft of this paper were produced by an AI
coding assistant, Anthropic Claude (models Claude Fable 5.2 and Claude Opus 5.5, used through Claude
Code), working from written specifications kept in the repository (`CLAUDE.md`, `UPGRADE_SPEC.md`). The
assistant also ran the benchmarks and generated the figures by executing the repository's scripts; no
reported number was typed by hand. The author defined the clinical, anatomical, and pedagogical specifications, manually inspected the 3-D cardiac plane intersections and 2-D ultrasound segmentations across test cases, identified edge-case failure modes (such as myocardial wall prompt capture and apical foreshortening axis inversion), verified the mathematical formulations against ASE/EACVI guidelines, and takes full responsibility for the software and manuscript.

# Acknowledgements

We thank the creators of the CAMUS dataset [@leclerc2019camus] and of SAM 2 [@ravi2024sam2].
This research received no external grant funding.

# References
