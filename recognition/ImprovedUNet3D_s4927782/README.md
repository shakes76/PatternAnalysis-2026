# HipMRI segmentation: is 3D spatial context worth its cost?

Author: Baoxu Wang (s4927782). COMP3710, 2026.

**Status: feasibility and environment audit stage. No HipMRI training or test
results have been produced yet.** The proposed final model is a **3D Improved
U-Net (Hard)** with residual units and deep supervision, evaluated against a
standard **2D U-Net** on the same patient-level splits of the HipMRI study.

The intended user is an imaging specialist reviewing draft anatomical contours.
The research question is whether additional 3D context reduces prostate
base/apex and adjacent-organ boundary errors enough to justify its memory and
runtime cost. This is a research prototype; operational suitability will be
discussed using measured failure cases.

## Current evidence and open checks

- A standard 2D U-Net implementation already exists in the author's own Demo 2.
  Its encoder/decoder and training experience are a starting point, not a new
  HipMRI experiment or evidence of 3D feasibility.
- The personal course fork and `topic-recognition` checkout have been created.
- Actual HipMRI filenames, patient identities, image/label pairing, label
  meanings, spatial alignment, GPU model and peak VRAM remain to be checked.
- No dataset-specific success, model accuracy or resource result is claimed.
- The author plans a tutor check-off on **28 October 2026**. The final deadline
  has not been confirmed; **30 October is only a planning assumption** and must
  be checked in Learn.UQ.

## Feasibility review

See [the one-page review draft](FEASIBILITY_REVIEW.md). It distinguishes existing
evidence from planned experiments. Update its evidence section after the first
successful cluster audit and model smoke test, before the tutor consultation.

## First reproducible check

The environment inventory uses only the Python standard library. It does not
load scan pixels, infer patient identities, submit jobs or train a model.

From this project directory on Rangpur:

```bash
python3 tools/audit_environment.py \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --output artifacts/environment_audit.json
```

Review the result before designing a dataset loader. A successful inventory is
not proof of patient independence, correct image/mask pairing or alignment.
Inventory output stays out of Git.

Local checks for this first tool:

```bash
python3 -m unittest discover -s tests -v
```

## Data and evaluation protocol

1. Audit the actual dataset layout, volume/mask correspondence, label meanings,
   shapes, spacing, orientation and affines. Do not guess a patient identifier
   from a filename before checking the dataset documentation.
2. Freeze disjoint train/validation/test patient lists. All visits, scans,
   slices and patches from one patient must remain in one split.
3. Train the 2D baseline on slices from the same patients used by the 3D model.
   Apply comparable normalization and spatial preprocessing. Resample masks
   with nearest-neighbor interpolation and check alignment visually.
4. Use training and validation data for smoke tests, tuning and checkpoint
   selection. Keep the test split for the final fixed comparison.
5. Report per-class Dice and IoU, per-patient results, and prostate base/mid/apex
   behavior. Investigate a spacing-aware boundary measure after geometry is
   verified. Define empty-label handling explicitly.

The course target for this Hard option is test Dice >= 0.70. The report must
state exactly how this is aggregated. Meeting the target alone does not
establish practical usefulness.

## Model and experiment plan

- **Baseline:** standard 2D U-Net, categorical multi-class output.
- **Final model:** 3D encoder/decoder, residual units and auxiliary segmentation
  heads for deep supervision. Document adaptations and their rationale.
- **First checks:** finite forward/loss/backward, ability to learn on a small
  training subset, checkpoint reload, and full validation-case reconstruction.
- **Initial budget:** one GPU, batch size 1 for the first 3D smoke test; adjust
  patch shape and channels from measured available memory and image spacing.
- **Final comparison:** same patient split, declared training budgets, fixed
  validation-based model selection and consistent whole-case evaluation.
- **Resources:** parameters, peak allocated GPU memory, training duration and
  synchronized per-case inference latency, all on recorded hardware.

The required model/data/training/inference files (`modules.py`, `dataset.py`,
`train.py`, `predict.py`) will be added as their implementations are developed
and verified. They are not represented by empty placeholder files.

## Results, failure analysis and engineering recommendation

Pending real experiments. The final report will include:

- Training/validation curves and a baseline/model metric and resource table.
- Three to five representative held-out failure cases, with image, ground
  truth, each model's prediction and class-specific FP/FN overlays.
- Anatomical explanations of improvement and remaining errors, including
  base/apex and nearby-organ interfaces where represented in the test data.
- A recommendation stating when 3D context is worth its resource cost and
  which cases still require careful human review.

No sample figures, synthetic smoke tests or previous Demo 2 scores will be
reported as final HipMRI test results.

## Environment and reproducibility

Exact training dependencies will be locked after inspecting the actual cluster
environment. The first audit records available package versions without
importing PyTorch. The final project will document seeds, patient-level split
generation, preprocessing, training/resume commands and inference commands.
Datasets and checkpoints must stay outside Git; final inference instructions
will state how to obtain the author's trained weights.

## Development history and submission

Make a commit when a real milestone works and has been checked. Planned
milestones are the audit and feasibility review, validated patient-level data
pipeline, a working baseline, a working 3D model, measured experiments, failure
analysis and the final report. Do not split an already completed project into
artificial commits or fabricate earlier development dates.

The final project belongs in this `recognition/` folder. Submit a working version
through a PR to upstream `topic-recognition` before the confirmed due date,
and export the completed README to PDF for Turnitin. Follow up on the teaching
team's later code review within the officially announced response window.

## Artificial Intelligence Usage Disclosure

Codex assisted with interpreting the specification, inspecting the author's
previous code, preparing this research plan, creating the course fork and
drafting the initial audit utility and its checks. Every later contribution
must be added here with its actual purpose and verification evidence.

Verification so far consists of checking the plan against the supplied 2026
assessment specification and reviewing the previous standard U-Net source.
The initial inventory utility passed seven local automated checks, covering
missing data, recursive file counts, bounded samples, JSON output, package
metadata and GPU-query limits. These are software checks, not real-data or
model-validation results. The author remains
responsible for reviewing and understanding all submitted code and results.

## References

1. COMP3710/COMP7710 Report, Version 2.5 (2026), supplied assessment specification:
   segmentation pp. 5-6, feasibility p. 12, structure/marking/submission pp. 13-16.
2. Isensee et al., *Brain Tumor Segmentation and Radiomics Survival Prediction:
   Contribution to the BRATS 2017 Challenge* (architectural starting reference):
   https://arxiv.org/abs/1802.10508.
3. The author's own Demo 2 standard U-Net:
   https://github.com/Jack-hithub/comp3710-demo2-part4.
