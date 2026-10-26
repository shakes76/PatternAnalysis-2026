# HipMRI segmentation: is 3D spatial context worth its cost?

Author: Baoxu Wang (s4927782). COMP3710, 2026.

**Status: all 211 cached volumes and 85 checks passed. A two-epoch full-data
pilot completed for both models on an A100; continuation to 30 epochs each is
queued. Final test accuracy remains unmeasured.** The final model is a **3D Improved
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
- The first real-data filename inventory found 211 MRI files and 211 label
  files. See [initial measured evidence](AUDIT_SUMMARY.md).
- The strict audit confirms 211 pairs, 38 patients and matching header grids.
  Patient-disjoint train/validation/test splits contain 26/6/6 patients and
  143/38/30 volumes. Reviewed sample contours align with their images.
- The [official CSIRO v2 label key](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034)
  confirms the mapping below. It is recorded with provenance in
  [label_schema.json](label_schema.json).
- The A100 synthetic forward/backward/checkpoint check passed. This is model
  feasibility evidence; final six-class model accuracy and resources remain
  unmeasured.
- The author plans a tutor check-off on **28 October 2026**. The final deadline
  has not been confirmed; **30 October is only a planning assumption** and must
  be checked in Learn.UQ.

| Raw ID / class index | Official label |
|---|---|
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

The original label-key file SHA-256 is
`dc3a99204684ced859872d79a75707af3c0ad31d00f5f3213063ce5e30727ee6`.
All visits (Week0 before treatment and weekly treatment scans) of one patient
remain in the same partition.

The complete cache contains 211 volumes at 64x128x128. Every image, mask and
spacing array was reloaded successfully and its mask counts matched the
manifest. Five deterministic TRAIN patients were visually reviewed in axial
and sagittal views; the sampled labels align with the intended structures.
This is a data-quality check rather than evidence of prediction accuracy.

![TRAIN B006 Week0: image and documented ground-truth contours in three planes](figures/training_alignment_qa.png)

The colors in this figure denote documented Body, Bones, Bladder, Rectum and
Prostate labels. Views use the shared downsampled RAS grid. Images are from the
HipMRI Study/CSIRO dataset credited below; no model prediction is shown.

## Feasibility review

The complete one-page feasibility review is reproduced below so it is included
in the README PDF. The separate [review source](FEASIBILITY_REVIEW.md) is the
check-off handout. Evidence snapshot: 26 October, after the two-epoch pilot.

<!-- FEASIBILITY_REVIEW_START -->
**Student:** s4927782 · **Check-off:** Wednesday, 28 October 2026 · **Difficulty:** Hard

### User need and scope

An imaging specialist needs draft prostate contours for review. This prototype investigates whether 3D spatial context improves difficult base/apex boundaries and interfaces with the bladder or rectum enough to justify additional memory and runtime. It compares a 3D Improved U-Net against a standard 2D U-Net on HipMRI, using identical patient-level train/validation/test assignments. Clinical deployment is outside scope.

### Measurable acceptance criteria

1. **Leakage:** assert zero patient overlap across splits, including repeated volumes and derived slices; verify image/mask pairing and spatial alignment before training.
2. **Accuracy:** target held-out prostate Dice **≥0.70** for the 3D model; report per-class Dice and IoU for both models. Use validation data for model selection.
3. **Boundary benefit:** compare patient-level and inferior/mid/superior results (approximate apex/base thirds), supplemented by grid-aware HD95 in millimetres, and identify where 3D context helps or fails relative to 2D.
4. **Failure analysis:** examine **3–5** representative test failures with input, ground truth, prediction and false-positive/false-negative overlays; give a justified recommendation on specialist review and computational trade-offs.
5. **Resources:** train on **one GPU**, using **volume batch size 1** for 3D. Peak allocated training VRAM **≤24 GiB** is the initial target. Report GPU, parameters, training time, peak VRAM and per-case inference time.

### Model choice and preliminary evidence

The Hard model uses volumetric convolutions, residual units, skip connections and deep supervision on downsampled HipMRI volumes. Relevant course concepts include convolutional spatial context, segmentation losses, generalisation and computational complexity. Isensee et al., [arXiv:1802.10508](https://arxiv.org/abs/1802.10508), is an architectural starting reference.

All **211 image/mask grids from 38 patients** passed cache checks, split **26/6/6** (143/38/30 scans). **75 checks passed**, including AMP overflow recovery and checkpoint resume. Five TRAIN patients were visually reviewed. [CSIRO's original key](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034) confirms **class 5 is prostate**. Two full-data A100 pilot epochs completed per model: validation foreground Dice reached **0.740 (2D)** and **0.535 (3D)**. These are early validation observations; **final test accuracy remains unmeasured**.

### Risks, budget and next experiment

Main risks are class imbalance, thin-boundary loss during downsampling and the short turnaround. Both models use class weights derived only from training voxels. Measured 3D pilot training peak was **2.23 GiB** on a 40-GB A100; training took about **51 s/epoch** versus **9 s/epoch** for warmed 2D. The next experiment resumes each model to **30 epochs**, selecting checkpoints only by validation. The initial 18-GPU-hour budget is revised to a **one-GPU-hour allocation** from the measured pilot cost; queue waiting is additional.

The preselected fallback reduces 3D patch size and channel counts while retaining residual units and deep supervision, preserving the approved Hard architecture. Moving instead to a Normal model would cap implementation at 15/20.

**Planning:** 30 October is an unconfirmed final-deadline assumption; verify the official date. Follow the course specification, pp. 5–6 and 12–16, including progressive commits, AI disclosure, PR and subsequent review response.
<!-- FEASIBILITY_REVIEW_END -->

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

Run checks in the project environment (the full suite also requires PyTorch,
NumPy, SciPy and Nibabel):

```bash
.venv/bin/python -m unittest discover -s tests -v
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

The required `modules.py`, `dataset.py`, `train.py` and `predict.py` are now
implemented. Integrated CPU checks passed; the two-scan smoke runs and resume
are development checks, not final experiments.

### Architecture and loss

![Implemented six-class 2D and 3D U-Net architectures with feature-map sizes and deep supervision](figures/model_architecture.png)

The diagram shows base width 16 and the actual cached input grid. Dashed lateral
arrows concatenate encoder features; orange heads supply auxiliary training
losses. Forward computes all 3D heads, while evaluation uses the main head's
argmax. The [editable SVG](figures/model_architecture.svg) can be regenerated
with `tools/plot_architecture.py`.

The 2D baseline has three pooling levels, double-convolution encoder/decoder
blocks, transposed-convolution upsampling and skip concatenation. The 3D model
has three volumetric pooling levels, residual blocks, GroupNorm, LeakyReLU,
resize-plus-convolution upsampling and two auxiliary heads. GroupNorm, pooling
and independently weighted auxiliary losses are explicit adaptations of the
cited architecture rather than an exact paper reproduction.

Every head predicts mutually exclusive classes. Its loss is weighted categorical
cross-entropy plus soft multiclass Dice, averaged over all classes including
background. Main/half/quarter losses have weights 1/0.5/0.25, divided by 1.75;
the baseline has only the main loss. Reported evaluation uses hard labels and a
separate foreground metric, not this training loss.

Both models derive CE weights from the same TRAIN volume counts: inverse square
root class frequencies, normalized to mean one, clipped to [0.05, 10], then
renormalized. Validation/test counts never determine these weights. Resolved
weights, seed, patient split and manifest hash are saved in each checkpoint.

### Shared preprocessing and metrics

Both models use the same canonical RAS image/mask grid, centre-aware linear
image/nearest-label resampling, and D,H,W array order. Nonzero image voxels are
z-normalized independently per case and clipped to [-5,5]; no split-level
statistics are fitted. Planned full resolution is D,H,W = 64,128,128. Metrics
refer to that downsampled grid; native-resolution accuracy is not claimed.

Dice and IoU use hard argmax predictions over entire volumes. A class absent
from both prediction and target is undefined and excluded; a false positive
for an absent target scores zero. We average foreground classes per case,
visits within each patient, then patients equally. Background is reported
separately. Validation selects the checkpoint; training never evaluates test.

Supplemental prostate HD95 uses six-connected surfaces and the shared grid's
D/H/W spacing in millimetres. Missing predictions retain Dice zero; HD95 is
undefined when either mask is empty. A paired boundary comparison uses only
common both-present visits and reports excluded-mask counts, so a missed organ
cannot appear as an artificially good distance. Inferior/mid/superior regions
are thirds of the ground-truth extent, approximate apex/base surrogates rather
than independent anatomical annotations.

The 2D loader caches unaugmented volumes up to 2 GiB per worker so neighboring
slices do not repeatedly decompress a full scan. Default training uses zero
worker subprocesses, keeping one such cache. Augmentations cannot modify it.

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

The verified direct versions are in `requirements.txt`; the active cluster
build is PyTorch 2.13.0+cu130 / CUDA 13.0, Python 3.11.15. Rangpur's existing
PyTorch environment is reused read-only through a project virtual environment;
only the extra packages in `requirements-extra.txt` were installed there.
On Rangpur, the one-time setup assumes a working PyTorch environment at
`~/miniconda3/envs/torch`. From the project directory, create the log folder and
submit the isolated environment setup, then wait for that job to succeed:

```bash
mkdir -p logs
/opt/slurm/bin/sbatch jobs/setup_cpu.slurm
```

The setup creates `.venv` using that Python with `--system-site-packages` and
installs the pinned extra dependencies. On a different machine, create a Python
3.11 environment and install `requirements.txt` with a PyTorch CUDA build
compatible with the GPU and driver. The actual experiments use an A100;
this CUDA 13 PyTorch build does not include Pascal/P100 support. Record the
resolved build rather than assuming every GPU is compatible.

Datasets and checkpoints stay outside Git. A final trained-weight location will
be supplied after the experiments finish.

For the supplied trained checkpoint, use the exact provided manifest alongside
the cache. Checkpoints verify its hash, patient assignments, labels and grid.
Retraining generates a new auditable manifest and new checkpoints.
The audit includes a creation timestamp, so regenerating it and the manifest
does not reproduce the exact digest expected by an existing checkpoint. Use the
frozen manifest supplied with those weights for inference; use a newly generated
manifest for a new training run. The fixed patient-split seed is **3710**;
model initialization/shuffling seed is **42**.
Python, NumPy and PyTorch are seeded; cuDNN benchmarking is disabled and
deterministic kernels are requested with warnings for unsupported operations.
This CUDA build warns that its categorical-CE kernel lacks a deterministic
implementation. These settings improve repeatability but do not guarantee
bit-identical retraining across runs, software builds or hardware.

The training source is frozen at commit
`69400a70b9b7bbed1971408e74785e926236410a`. [training_sources.json](training_sources.json)
records SHA-256 digests for the eleven training, preprocessing, inference,
environment and job files, plus the frozen manifest digest. Documentation and
analysis helpers may receive later commits without changing model behavior.
The actual remote files are checked before a provenance sidecar is recorded:

```bash
.venv/bin/python tools/record_provenance.py --expected training_sources.json \
  --manifest artifacts/cache64/manifest.json \
  --output artifacts/full_seed42/improved3d/source_provenance.json
```

Use the corresponding 2D directory for its sidecar. The tool can also hash a
finished checkpoint with `--checkpoint`; explicit `--replace` updates a sidecar.
It does not modify or deserialize weights and never changes the manifest or
training configuration. Final weight digests and selected epochs remain pending.
This digest book describes the original experiment. A new training experiment
must record its own source identities and new manifest digest rather than
claiming compatibility with the original checkpoint.

From this project directory, the preparation sequence on a CPU compute node is:

```bash
.venv/bin/python tools/audit_dataset.py \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --output artifacts/dataset_audit.json
.venv/bin/python tools/prepare_cache.py --audit artifacts/dataset_audit.json \
  --output-dir artifacts/cache64 --shape 64 128 128 --label-ids 0 1 2 3 4 5
.venv/bin/python tools/attach_label_schema.py \
  --manifest artifacts/cache64/manifest.json --schema label_schema.json
.venv/bin/python tools/verify_label_semantics.py \
  --manifest artifacts/cache64/manifest.json --count 5 \
  --output-dir artifacts/label_review_v1
```

Preparation refuses to overwrite an existing manifest/cache. Attach the official
label metadata **before** training, since it changes the manifest digest.
`jobs/label_review.slurm` also reads all 211 caches and verifies their stored
voxel counts. It opens labels for integrity checks, not model evaluation.

The full-data pilot uses two complete epochs per model and the following fixed
settings. Continuing uses `--resume` with the same output directory and a larger
total `--epochs`; the other settings must match exactly.

```bash
for model in unet2d improved3d; do
  batch_size=1
  if [ "$model" = unet2d ]; then batch_size=16; fi
  .venv/bin/python train.py --manifest artifacts/cache64/manifest.json \
    --model "$model" --output-dir "artifacts/full_seed42/$model" \
    --epochs 2 --batch-size "$batch_size" --base-channels 16 --lr 0.001 \
    --workers 0 --seed 42 --device cuda --amp --eval-batch-size 32
done
```

`jobs/train_pilot.slurm` requests one A100 for at most 20 minutes. The longer
`jobs/train_pair.slurm` requests the course GPU partition and resumes each
model's latest complete epoch. After the pilot completed, the formal 30-epoch
continuation was submitted with a one-hour limit, based on measured cost:

```bash
/opt/slurm/bin/sbatch --time=01:00:00 jobs/train_pair.slurm 30
```

Submit only one job writing these output directories at a time. The script skips
a model already at the requested total; otherwise it passes the model's
`--resume artifacts/full_seed42/<model>/latest.pt`. For an allocated GPU compute
node, the explicit equivalent for the 3D model is:

```bash
.venv/bin/python train.py --manifest artifacts/cache64/manifest.json \
  --model improved3d --output-dir artifacts/full_seed42/improved3d \
  --epochs 30 --batch-size 1 --base-channels 16 --lr 0.001 \
  --workers 0 --seed 42 --device cuda --amp --eval-batch-size 32 \
  --resume artifacts/full_seed42/improved3d/latest.pt
```

The direct command requires a checkpoint below epoch 30. For 2D, change the
model/output/resume paths to `unet2d` and batch size to 16; keep the remaining
settings. Interrupted epochs repeat from the last saved epoch. Equal epoch
counts cover the same training volumes, while batch units, optimizer-step
counts and elapsed compute differ and must be reported separately.

Validation inference uses `predict.py --split validation`. Only after model
decisions are fixed, the same command may use `--split test --final-test`:

```bash
.venv/bin/python predict.py --manifest artifacts/cache64/manifest.json \
  --checkpoint artifacts/full_seed42/improved3d/best.pt \
  --split validation --output-dir artifacts/validation/improved3d \
  --batch-size 32 --device cuda --target-class 5
```

Run the equivalent command for `unet2d/best.pt` with its own output directory.
Never overwrite an earlier evaluation. This section describes executable
commands, not evidence that a formal training or evaluation has finished.

After all decisions are frozen using validation, final test inference on a GPU
compute node uses both selected checkpoints under the same settings:

```bash
for model in unet2d improved3d; do
  .venv/bin/python predict.py --manifest artifacts/cache64/manifest.json \
    --checkpoint "artifacts/full_seed42/$model/best.pt" \
    --split test --final-test --output-dir "artifacts/test_final/$model" \
    --batch-size 32 --device cuda --target-class 5
done
```

Then on a CPU compute node, analyze paired predictions and produce the failure
gallery and prostate boundary summaries:

```bash
.venv/bin/python tools/analyze_pair.py --manifest artifacts/cache64/manifest.json \
  --baseline-eval artifacts/test_final/unet2d/evaluation.json \
  --model-eval artifacts/test_final/improved3d/evaluation.json \
  --output-dir artifacts/test_final/paired --target-class 5 \
  --bootstrap-samples 2000 --seed 3710
```

This seed controls paired patient bootstrap sampling, separately from training.
The held-out test contains six patients and 30 visits; visits from one patient
are correlated. A single training seed and six independent test patients limit
certainty about small improvements. Report the paired confidence interval and
case-level failures alongside the point estimates.

The failure gallery selects each patient's worst defined 3D prostate Dice visit,
then the five worst distinct patients. Zero Dice remains eligible. Each figure
shows GT extent thirds and each model's largest prostate FP/FN slice across the
full volume, including errors outside the GT extent; duplicate depths share a
row. The selection rule is fixed before final test evaluation.

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

## References

1. COMP3710/COMP7710 Report, Version 2.5 (2026), supplied assessment specification:
   segmentation pp. 5-6, feasibility p. 12, structure/marking/submission pp. 13-16.
2. Isensee et al., *Brain Tumor Segmentation and Radiomics Survival Prediction:
   Contribution to the BRATS 2017 Challenge* (architectural starting reference):
   https://arxiv.org/abs/1802.10508.
3. The author's own Demo 2 standard U-Net:
   https://github.com/Jack-hithub/comp3710-demo2-part4.
4. Dowling, J.; Greer, P. (2021), *Labelled weekly MR images of the male pelvis*,
   v2, CSIRO: https://doi.org/10.25919/45t8-p065. Official label key:
   https://data.csiro.au/dap/ws/v2/collections/52549/support/5034.
5. Dowling et al. (2015), *Automatic Substitute Computed Tomography Generation
   and Contouring for MRI-Alone External Beam Radiation Therapy From Standard
   MRI Sequences*: https://doi.org/10.1016/j.ijrobp.2015.08.045.
6. Ronneberger, O.; Fischer, P.; Brox, T. (2015), *U-Net: Convolutional Networks
   for Biomedical Image Segmentation*: https://arxiv.org/abs/1505.04597.

Data were provided by HipMRI Study and Calvary Mater Newcastle Hospital
(Professor Peter Greer). Use follows the dataset's non-commercial research
terms and no attempt is made to identify or contact participants.
