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

The Hard model will use volumetric convolutions, residual units, skip connections and deep supervision on downsampled HipMRI volumes. Relevant course concepts include convolutional spatial context, segmentation losses, generalisation and computational complexity. Isensee et al., [arXiv:1802.10508](https://arxiv.org/abs/1802.10508), is an architectural starting reference.

An initial Rangpur inventory found **211 MRI filenames and 211 label filenames**. The existing Python 3.11.15/PyTorch 2.13.0 environment is accessible; the course queue lists A100 nodes. The read-only audit tool passed seven software checks. Demo2 supplies a standard 2D U-Net starting point. **Pairing, patient splits, geometry, the 3D model and real-data performance remain unvalidated.** Equal file counts do not establish alignment.

## Risks, budget and next experiment

Main risks are patient-identification mistakes, image/mask misalignment, class imbalance, anisotropic spacing, GPU memory limits and short turnaround. First audit sample pairs and patient identifiers, then run forward/loss/backward smoke tests and profile one real-data batch. Start with a provisional budget of **18 GPU-hours** (6 baseline, 12 3D), revising it after profiling. Produce an executable prototype before longer runs.

The preselected fallback reduces 3D patch size and channel counts while retaining residual units and deep supervision, preserving the approved Hard architecture. Moving instead to a Normal model would cap implementation at 15/20.

**Planning:** 30 October is an unconfirmed final-deadline assumption; verify the official date. Follow the course specification, pp. 5–6 and 12–16, including progressive commits, AI disclosure, PR and subsequent review response.
