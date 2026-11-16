# HipMRI segmentation: midpoint feasibility review

**Student:** s4906920

## 1. Requirements and scope

For MRI researchers, build a reproducible 2D multiclass
segmentation prototype and determine whether an Improved U-Net reduces boundary
errors enough to justify its compute cost. This is not a clinical deployment.
**Proposed final acceptance criteria**, not achieved results:

1. Zero patient overlap across train/validation/test; verified image-mask pairing,
   one global label dictionary and a confirmed prostate class.
2. Mean patient-level prostate Dice >= 0.75 on the held-out test set; also report
   per-class Dice/IoU and variation across patients.
3. Improved U-Net gains >= 0.02 prostate boundary F1 over the baseline at a fixed
   two-pixel tolerance, with prostate Dice decreasing by no more than 0.01.
4. At the chosen real-data configuration, peak PyTorch allocated training memory
   <= 6 GiB and median batch-one model inference <= 100 ms on the local GPU.
5. Record seeds, splits, configuration and checkpoint reload agreement; explain
   3-5 genuine failure cases. An unmet accuracy target remains a reportable finding.

## 2. Model, data and rationale

Use course HipMRI slices at
`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`.
Baseline: Ronneberger et al.'s 2D U-Net [1]. Main model: the **Normal-difficulty**
2D Improved U-Net option, adapting Isensee et al. [2] with residual context blocks,
instance normalization, spatial dropout and deep supervision. This applies
multiscale CNN features, skip connections, regularization and gradient
propagation. It is a configurable 2D adaptation, not a complete reproduction of
the paper's 3D BraTS training or ensemble. Compare identical patient splits,
preprocessing and training budgets using cross-entropy plus foreground Dice loss.
Preserve all verified classes; do not merge every nonzero label into prostate.

## 3. Preliminary feasibility evidence

Both models completed six epochs on **synthetic geometric data**: 32 training and
8 validation images, 128 x 128, seed 3710. Saved/reloaded predictions agreed
exactly. Tests cover patient separation, label validation, model gradients and
metric edge cases. These are pipeline checks, not medical accuracy evidence.

Local RTX 4070 Laptop GPU (8 GB), float32, base width 16: a separate short tensor
probe used 256 x 256 inputs, training batch 4, **two output classes**:

| Measurement | U-Net | Improved U-Net |
|---|---:|---:|
| Parameters | 1,940,834 | 3,186,342 |
| Peak allocated / reserved memory (GiB) | 0.290 / 0.438 | 0.639 / 0.820 |
| Median model inference, batch 1 (ms) | 1.395 | 4.224 |

Sources: `results/resource_*.json` and synthetic summaries. CUDA timings are
synchronized; inference excludes preprocessing/I/O. Allocator memory excludes
driver/other processes. Actual HipMRI multiclass costs must be remeasured.
**Real data access, patient metadata and label meanings remain unverified.**

## 4. Risks, budget and next experiments

Provisional post-midpoint budget: 16 working hours and 12 GPU-hours, revised after
a real-data pilot. Next: verify Rangpur access; audit labels, patient IDs, spacing
and overlays; freeze patient splits; run a small U-Net pilot; then compare models
on validation data. Keep test data locked until model choices are finalized.
Risks are delayed access, mislabeled classes, leakage, limited memory and
poor apex/base boundaries. If access is delayed, present measured engineering
evidence and the outstanding dependency. If memory/time is insufficient, reduce
batch/width for both models and document the change; retain U-Net as the fallback.
Tutor discussion and personal check-off remain required.

**References:** [1] [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597).
[2] [Isensee et al., 2018](https://arxiv.org/abs/1802.10508).
Assessment: `COMP3710_Report_2026_draft_v2.2.pdf`, Sections 2.2 and 3.
