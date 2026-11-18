# EchoPlane3D: an open-source, offline split-screen workbench coupling video foundation models with 3D cardiac plane slicing for echocardiography training

**Authors:** Burhan Zaim — **Affiliations:** Faculty of Medicine and Health, Al-Farabi Kazakh National University, Almaty, Kazakhstan — **Corresponding author:** Burhan Zaim (YOUR_EMAIL@example.com)

> Draft for the Elsevier SoftwareX *Original Software Publication* template. Move the text into the official
> Word/LaTeX template before submission (see `documentation/PUBLICATION_GUIDE.md`). Every number in this file
> is generated from `results/` by `scripts/build_paper_numbers.py`; do not edit numbers by hand.

## Abstract

<!-- numbers-from-results -->
Measuring left-ventricular ejection fraction (LVEF) with the method of disks depends on the 3-D relation
between the imaging plane and the ventricle, which learners cannot see on a 2-D echocardiogram. EchoPlane3D
is an open-source, fully offline workbench that links a 2-D clip, segmented and tracked by the SAM 2 video
foundation model and quantified with single-plane and biplane Simpson volumetry, to a 3-D cardiac model
sliced along the four standard views, with an interactive simulation of apical foreshortening and a
deterministic tutor. On {{camus_n_patients}} held-out CAMUS patients, full-cycle tracking from a single
simulated click gave biplane LVEF within a mean absolute {{trk_snapped_click_lvef_bp_mae}} points of the
dataset's reference masks (95 % limits of agreement {{trk_snapped_click_lvef_bp_loa}}), while using under
{{tele_vram_peak_max}} MB of GPU memory on a consumer laptop. It is an educational and research prototype,
not a medical device.
<!-- /numbers-from-results -->

**Keywords:** echocardiography; medical education; method of disks; apical foreshortening; segment anything; 3-D visualisation

## Code metadata

| Nr. | Code metadata description | Value |
|---|---|---|
| C1 | Current code version | v1.1.0 |
| C2 | Permanent link to code/repository used for this code version | https://github.com/YOUR_GITHUB_USERNAME/EchoPlane3D |
| C3 | Permanent link to a reproducible capsule | TO BE ADDED (Zenodo DOI of the release) |
| C4 | Legal code licence | MIT |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools and services used | Python 3.10, JavaScript (ES modules), FastAPI, PyTorch, SAM 2 (commit {{sam2_commit}}), three.js, Plotly |
| C7 | Compilation requirements, operating environments and dependencies | Linux (tested on WSL2 Ubuntu 22.04), NVIDIA GPU with ≥ 6 GB for tracking, ffmpeg; `requirements.txt`; no build step |
| C8 | Link to developer documentation/manual | `README.md`, `documentation/` in the repository |
| C9 | Support email for questions | TO BE ADDED |

*(Check the row list against the current SoftwareX template.)*

## 1. Motivation and significance

The biplane method of disks is the recommended 2-D measure of LV volumes and ejection fraction
[1]. Its accuracy depends on geometry the operator cannot see: the apical planes must pass through the
true apex, and a plane that slides one rib-space too high shortens the long axis, under-estimates both
volumes, and — because the systolic cavity is more elongated — tends to over-estimate LVEF. Teaching this
usually relies on static diagrams. Simulator-based echocardiography training is effective [2] but costly,
closed, and rarely exposes the quantification itself. Deep-learning research systems estimate LVEF
automatically [3], and promptable foundation models segment medical images [4–6], but none is a teaching
environment that couples the measurement to its spatial cause.

EchoPlane3D makes that coupling explicit. A learner clicks the LV cavity on a real or phantom clip, watches
the mask, long axis and 20 disks follow the beat, sees the same plane cut a 3-D heart, and drags a slider
that tilts the plane off the apex while the predicted foreshortened volumes and LVEF update. Researchers
get an open, reproducible pipeline for studying promptable segmentation on echocardiography (here on the
open CAMUS dataset [8]) under a strict GPU budget, with a benchmark script that regenerates every number in this paper.

## 2. Software description

### 2.1 Architecture

EchoPlane3D is a single FastAPI process serving a zero-build web page (Fig. 1). The browser loads only
vendored libraries (three.js, Plotly, Tailwind), so the workbench runs with no network connection. The
backend exposes REST endpoints for samples, segmentation, foreshortening, the tutor and clip upload, and
delegates to core modules: the SAM 2 tracker, the Simpson engine, the tutor, CAMUS data access and upload
handling. All settings that depend on the machine (model paths, GPU budget, optional local LLM) are
environment-overridable.

![Figure 1. Architecture.](figures/fig1_architecture_overview.png)

### 2.2 Functionalities

**Prompting and tracking.** Frame 0 is prompted automatically (largest dark region enclosed by tissue in the
upper-central sector, with a fallback when wall drop-out leaves the cavity open), by a click, or by several
positive and negative points. A click on bright tissue can be moved into the nearest blood pool. SAM 2's
image head returns three candidates; stock SAM keeps the one with the highest predicted IoU, which on
ultrasound is often the whole sector. EchoPlane3D instead keeps candidates that contain every positive point,
exclude every negative one and cover at most half of the sector, and scores them by rim-versus-interior
contrast, darkness, solidity and containment in the sector. The chosen mask seeds the SAM 2 video predictor.
Memory is bounded by bfloat16 autocast, decoded frames kept in system RAM, a state reset and cache release
after every run, and a lock that serialises GPU work.

**Volumetry.** Each frame's mask is reduced to its largest component and quantified with 20 disks along its
principal axis, using anti-aliased slab binning. Single-plane volume is π/4·ΣDᵢ²·L/20; biplane volume
pairs the A4C and A2C views of one subject as π/4·Σaᵢbᵢ·max(L)/20 and flags a long-axis mismatch above 10 %
[1]. Estimates, dataset reference values and values reported with the dataset are returned in separately
named fields and shown in separate rows. Heuristic warnings flag volume jumps, non-compact masks and large
area mismatches.

**Foreshortening model.** The cavity is modelled as a half prolate ellipsoid of length a and base radius b,
whose shape changes over the cycle as a ∝ (V/EDV)^¼ and b ∝ (V/EDV)^⅜. A plane through the mitral-annulus
centre tilted by θ sees a′ = 1/√(cos²θ/a² + sin²θ/b²), so Simpson's rule reports V·a′/a.

**3-D navigator.** Nine named structures (LV, septum, RV with outflow tract, LA with appendage, RA, aorta
with arch branches, pulmonary artery, venae cavae) are built as smooth implicit surfaces and polygonised
with marching cubes [7]. They are cut in real time by a three.js clipping plane with stencil caps. Plane
poses for the four views live in the same frame as the model; the builder verifies which structures each
plane cuts and that the 3-D cut length matches the formula above. The LV contracts with the measured V(t).

**Tutor.** A deterministic engine matches free text to intents (views, anatomy, foreshortening, single-plane
versus biplane, interpretation of the current study, prompting help, quizzes with scoring) and computes every
number with the volumetry engine. An optional local Ollama model may reword explanations; its proposed UI
actions must pass a Pydantic whitelist, and it never handles quizzes, grading or calculations.

## 3. Empirical validation

<!-- numbers-from-results -->
All values below were produced by one run of `scripts/run_paper_benchmarks.py` ({{benchmark_date}},
{{benchmark_total_min}} min, {{gpu_name}}, PyTorch {{torch_version}}, SAM 2 commit {{sam2_commit}}).
**Reference** means the CAMUS [8] expert LV masks measured with the same Simpson code. It is not clinical
ground truth: compared with the ejection fraction reported with the dataset, the reference biplane LVEF
has a bias of {{ref_vs_reported_bias}} points (r = {{ref_vs_reported_r}}).

**Held-out CAMUS cohort** [8]. {{camus_n_patients}} patients ({{camus_first_patient}}–{{camus_last_patient}},
{{camus_n_per_quality}} per A4C image-quality grade) were selected by a fixed rule and were not used to develop or
tune the prompting heuristics. After a first run showed that independent end-diastolic and end-systolic
prompting gives inconsistent LVEF, full-cycle tracking was added to the protocol; no algorithm parameter
was changed after any held-out result was seen. "Click" and "multi-point" prompts are simulated users derived from the reference mask;
"automatic" uses no reference information.

*Full-cycle tracking* (the workbench's pipeline: one frame-0 prompt, SAM 2 propagation;
{{trk_snapped_click_n_seq}} sequences per prompt; Fig. 2):

| Biplane measure vs reference | Snapped click | Automatic |
|---|---|---|
| LVEF MAE (points) | {{trk_snapped_click_lvef_bp_mae}} | {{trk_automatic_lvef_bp_mae}} |
| LVEF bias, 95 % limits of agreement [10] (points) | {{trk_snapped_click_lvef_bp_bias}}, {{trk_snapped_click_lvef_bp_loa}} | {{trk_automatic_lvef_bp_bias}}, {{trk_automatic_lvef_bp_loa}} |
| LVEF ICC(2,1) [9] / Pearson r | {{trk_snapped_click_lvef_bp_icc}} / {{trk_snapped_click_lvef_bp_r}} | {{trk_automatic_lvef_bp_icc}} / {{trk_automatic_lvef_bp_r}} |
| EDV bias (mL), ICC | {{trk_snapped_click_edv_bp_bias}}, {{trk_snapped_click_edv_bp_icc}} | {{trk_automatic_edv_bp_bias}}, {{trk_automatic_edv_bp_icc}} |
| ESV bias (mL), ICC | {{trk_snapped_click_esv_bp_bias}}, {{trk_snapped_click_esv_bp_icc}} | {{trk_automatic_esv_bp_bias}}, {{trk_automatic_esv_bp_icc}} |
| Median sequence Dice (Good / Medium / Poor) | {{trk_snapped_click_dice_med_good}} / {{trk_snapped_click_dice_med_medium}} / {{trk_snapped_click_dice_med_poor}} | {{trk_automatic_dice_med_good}} / {{trk_automatic_dice_med_medium}} / {{trk_automatic_dice_med_poor}} |
| Sequences with a quality warning | {{trk_snapped_click_warnings}} | {{trk_automatic_warnings}} |

All {{trk_snapped_click_plausible}} (snapped) and {{trk_automatic_plausible}} (automatic) biplane LVEF values
were physically plausible. The automatic prompt landed inside the reference cavity in
{{trk_automatic_prompt_hit}} of {{trk_automatic_n_seq}} held-out first frames. Single-plane A4C volumes were
systematically low (EDV bias {{trk_snapped_click_edv_sp_a4c_bias}} mL), consistent with SAM 2 under-filling
the cavity. On the eight-patient demo cohort — used while tuning the heuristics and reported only for
completeness — median sequence Dice was {{show_automatic_dice_med}} (automatic) and
{{show_snapped_click_dice_med}} (snapped click); one demo sequence lost the cavity entirely on some frames
(minimum frame Dice {{show_snapped_click_dice_min}}).

*Single frames* ({{camus_n_frames}} expert-annotated end-diastolic and end-systolic frames, Fig. 3):

| Prompting strategy | Median Dice (IQR) | Median HD95 (mm) | Frames with Dice below one half |
|---|---|---|---|
| Stock SAM, click | {{naive_click_dice_med}} ({{naive_click_dice_iqr}}) | {{naive_click_hd95_med}} | {{naive_click_fail}} |
| Snapped click | {{snapped_click_dice_med}} ({{snapped_click_dice_iqr}}) | {{snapped_click_hd95_med}} | {{snapped_click_fail}} |
| Multi-point (3 + 1) | {{multi_point_dice_med}} ({{multi_point_dice_iqr}}) | {{multi_point_hd95_med}} | {{multi_point_fail}} |
| Automatic | {{automatic_dice_med}} ({{automatic_dice_iqr}}) | {{automatic_hd95_med}} | {{automatic_fail}} |
| Click on the wall, stock SAM | {{naive_wall_click_dice_med}} ({{naive_wall_click_dice_iqr}}) | {{naive_wall_click_hd95_med}} | {{naive_wall_click_fail}} |
| Click on the wall, snapped | {{snapped_wall_click_dice_med}} ({{snapped_wall_click_dice_iqr}}) | {{snapped_wall_click_hd95_med}} | {{snapped_wall_click_fail}} |

Segmenting end-diastole and end-systole independently produced end-systolic masks larger than end-diastolic
ones for {{frames_implausible_min}}–{{frames_implausible_max}} of {{camus_n_patients}} patients,
so per-frame prompting is not a usable LVEF pipeline.

**Analytic verification and foreshortening.** On a prolate ellipsoid (exact volume {{ellipsoid_exact_ml}} mL) the 20-disk
engine erred by at most {{ellipsoid_worst_err_pct}} % (mean {{ellipsoid_mean_err_pct}} %) across pixel
spacings of {{ellipsoid_spacings}} mm and five rotations. The closed-form foreshortening model agreed with an
independent numerical slice of the 3-D half ellipsoid to within {{fs_model_vs_slice_worst_upto20_pct}} %
(end-diastolic volume) up to 20°, and the 3-D mesh cut length agreed with the formula to within {{mesh_worst_len_diff_pct}} %.
At 30°, the foreshortened section becomes wider than tall, the principal axis flips to the annular
direction, and the numerical value departs from the model by {{fs_model_vs_slice_n30_pct}} % (Fig. 4).
For the normal phantom, a 15° tilt under-estimated EDV by {{fs_n15_edv}} % and ESV by {{fs_n15_esv}} %
and raised LVEF by {{fs_n15_ef}} points. On the two calibrated phantom clips, tracking reported
LVEF {{pn_est_lvef}} % (truth {{pn_true_lvef}} %) and {{ph_est_lvef}} % (truth {{ph_true_lvef}} %), with
EDV errors of {{pn_edv_err}} % and {{ph_edv_err}} % and mean Dice {{pn_dice}} and {{ph_dice}}.

**Hardware.** Across {{tele_n_runs}} tracking runs, peak memory reserved by PyTorch was at most
{{tele_vram_peak_max}} MB (allocated {{tele_vram_alloc_max}} MB; whole-GPU usage reported by `nvidia-smi` at
most {{tele_smi_peak_max}} MiB), all {{tele_budget_ok}} runs stayed within the configured budget, and the
median tracking cost was {{tele_ms_per_frame_med}} ms per frame ({{tele_fps_med}} frames/s). Loading the
model took {{tele_load_s}} s and {{tele_load_vram}} MB. One image embedding took {{tele_embed_ms}} ms and
each prompt decode with candidate selection {{tele_decode_ms}} ms. Peak process memory was
{{tele_rss_peak_gb}} GB.
<!-- /numbers-from-results -->

![Figure 2. Biplane LVEF agreement: independent frames vs full-cycle tracking.](figures/fig2_bland_altman_lvef.png)

![Figure 3. Prompting strategy by image quality.](figures/fig3_prompt_ablation_by_quality.png)

![Figure 4. Foreshortening sensitivity.](figures/fig4_foreshortening_sensitivity.png)

## 4. Illustrative examples

Fig. 5 shows the workbench on the calibrated reduced-ejection-fraction phantom (truth LVEF 32 %). The left
pane shows the tracked mask, long axis and disks at end-systole. The centre pane shows the A4C plane tilted
20° off axis, with the probe, sector fan and the contracted LV. The right pane shows V(t) with the reference
end-diastolic and end-systolic levels and the foreshortened curve. Below it, the tutor reports, as a
separate verified calculation, what the tilted plane would measure. Typical sessions: (i) click on the wall
and watch snapping move the prompt into the cavity; (ii) segment both apical views of one CAMUS patient and
compare single-plane with biplane LVEF; (iii) ask the tutor to "demo a 20 degree tilt" and read the
predicted bias; (iv) upload an own clip without calibration and see LVEF reported while volumes become a
pixel-scale index.

![Figure 5. The workbench.](figures/fig5_workbench_screenshot.png)

## 5. Impact and limitations

EchoPlane3D lets a learner connect three things that are normally taught separately: where the probe is,
what the plane cuts, and what number comes out. It runs on a laptop without internet, which suits teaching
in low-connectivity settings and keeps clinical clips on the user's machine. For research it provides a
reproducible harness for promptable segmentation on echocardiography. Its main lesson so far is that
prompt handling matters more than the model: the same SAM 2 weights move from mostly unusable masks to
reasonable ones depending on candidate selection and temporal propagation.

Limitations are substantial. LVEF limits of agreement against reference masks are far wider than clinical
tolerance, so outputs must not be used for patient care. The reference itself is a set of expert masks, not
clinical truth. The 3-D heart is schematic. The principal-axis long axis fails on very round sections. The
demo cohort informed the heuristics; only the held-out cohort is used for validation, and it is small. The
optional LLM was tested only with mocked responses, and GPU co-residency with a local LLM was not measured.
No study with learners has been done yet.

## 6. Conclusions

EchoPlane3D is an open, offline workbench that makes the geometry of echocardiographic volumetry visible and
interactive, built on a video foundation model with explicit guards for a 6 GB GPU. Future work: a learner
study, landmark-based long axes, more views, and a patient-derived anatomical model.

## Ethics and data governance

The software processes only data the user loads. The CAMUS dataset is downloaded from its original host
under that host's terms and is never redistributed with the software; only derived metrics are stored in
`results/`. Figure 5 uses a synthetic phantom. Uploaded clips stay on the local machine. Users must remove
patient identifiers before uploading clinical clips. EchoPlane3D is not a medical device and is not
cleared by any regulator.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used Anthropic Claude (Claude Fable 5.2 and Claude Opus 5.5,
through Claude Code) to write most of the software, tests and documentation and to draft this manuscript,
and to run the benchmark scripts that produced every reported number. The author formulated the clinical and architectural requirements, tested the interactive 2-D/3-D workbench across phantom and CAMUS sequences, audited the statistical outputs and limitations, and edited all manuscript sections. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Declaration of competing interest

The author declares that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

Code, benchmark tables and figures: the repository (C2). CAMUS: <https://humanheart-project.creatis.insa-lyon.fr/database/>
(not redistributed). The phantoms are regenerated by `scripts/prepare_samples.py`.

## References

1. Lang RM, Badano LP, Mor-Avi V, et al. Recommendations for cardiac chamber quantification by echocardiography in adults: an update from the ASE and EACVI. *J Am Soc Echocardiogr* 2015;28(1):1–39.e14. doi:10.1016/j.echo.2014.10.003
2. Neelankavil J, Howard-Quijano K, Hsieh TC, et al. Transthoracic echocardiography simulation is an efficient method to train anesthesiologists in basic transthoracic echocardiography skills. *Anesth Analg* 2012;115(5):1042–1051. doi:10.1213/ANE.0b013e318265408f
3. Ouyang D, He B, Ghorbani A, et al. Video-based AI for beat-to-beat assessment of cardiac function. *Nature* 2020;580(7802):252–256. doi:10.1038/s41586-020-2145-8
4. Kirillov A, Mintun E, Ravi N, et al. Segment Anything. *ICCV* 2023:3992–4003. doi:10.1109/ICCV51070.2023.00371
5. Ravi N, Gabeur V, Hu Y-T, et al. SAM 2: Segment Anything in images and videos. arXiv:2408.00714, 2024. doi:10.48550/arXiv.2408.00714
6. Ma J, He Y, Li F, et al. Segment anything in medical images. *Nat Commun* 2024;15. doi:10.1038/s41467-024-44824-z
7. Lorensen WE, Cline HE. Marching cubes: a high resolution 3D surface construction algorithm. *ACM SIGGRAPH Comput Graph* 1987;21(4):163–169. doi:10.1145/37402.37422
8. Leclerc S, Smistad E, Pedrosa J, et al. Deep learning for segmentation using an open large-scale dataset in 2D echocardiography. *IEEE Trans Med Imaging* 2019;38(9):2198–2210. doi:10.1109/TMI.2019.2900516
9. Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater reliability. *Psychol Bull* 1979;86(2):420–428. doi:10.1037/0033-2909.86.2.420
10. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet* 1986;327(8476):307–310. doi:10.1016/S0140-6736(86)90837-8
