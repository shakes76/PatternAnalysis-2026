# Feasibility Review: HipMRI 3D Improved U-Net

**Student:** s4927782 · **Check-off:** Wednesday, 28 October 2026 · **Difficulty:** Hard

## User need and scope

An imaging specialist needs draft prostate contours for review. This prototype investigates whether 3D spatial context improves difficult base/apex boundaries and interfaces with the bladder or rectum enough to justify additional memory and runtime. It compares a 3D Improved U-Net against a standard 2D U-Net on HipMRI, using identical patient-level train/validation/test assignments. Clinical deployment is outside scope.

## Measurable acceptance criteria

1. **Leakage:** assert zero patient overlap across splits, including repeated volumes and derived slices; verify image/mask pairing and spatial alignment before training.
2. **Accuracy:** target held-out prostate Dice **≥0.70** for the 3D model; report per-class Dice and IoU for both models. Use validation data for model selection.
3. **Boundary benefit:** compare patient-level and base/mid/apex results, supplemented by HD95 in millimetres, and identify where 3D context helps or fails relative to 2D.
4. **Failure analysis:** examine **3–5** representative test failures with input, ground truth, prediction and false-positive/false-negative overlays; give a justified recommendation on specialist review and computational trade-offs.
5. **Resources:** train on **one GPU**, initially using **patch batch size 1**. Peak allocated training VRAM **≤24 GiB** is a provisional target, pending hardware confirmation and measurement. Report GPU, parameters, training time, peak VRAM and per-case inference time.

## Model choice and preliminary evidence

The Hard model uses volumetric convolutions, residual units, skip connections and deep supervision on downsampled HipMRI volumes. Relevant course concepts include convolutional spatial context, segmentation losses, generalisation and computational complexity. Isensee et al., [arXiv:1802.10508](https://arxiv.org/abs/1802.10508), is an architectural starting reference.

The audit verified **211 image/mask grids from 38 patients**, split **26/6/6** (143/38/30 scans). **56 checks passed**; tiny training-only runs verified preprocessing, models and CPU checkpoint resume. A100 synthetic checks passed; sampled image/label contours align. [CSIRO's original key](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034) confirms **class 5 is prostate**. **Final accuracy, full-training memory and patient-level generalisation remain untested.**

## Risks, budget and next experiment

Main risks are severe class imbalance, thin-boundary loss during downsampling, GPU memory limits and short turnaround. Next inspect more patients and profile complete real-data GPU epochs. Both models use class weights derived only from training voxels. Start with a provisional budget of **18 GPU-hours** (6 baseline, 12 3D), revising it after profiling. Complete the pilot before longer runs.

The preselected fallback reduces 3D patch size and channel counts while retaining residual units and deep supervision, preserving the approved Hard architecture. Moving instead to a Normal model would cap implementation at 15/20.

**Planning:** 30 October is an unconfirmed final-deadline assumption; verify the official date. Follow the course specification, pp. 5–6 and 12–16, including progressive commits, AI disclosure, PR and subsequent review response.
