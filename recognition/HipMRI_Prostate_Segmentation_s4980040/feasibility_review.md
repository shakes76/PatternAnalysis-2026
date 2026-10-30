# Feasibility Review: 2D vs 3D Context for Prostate Segmentation on HipMRI

**Hafizh Surya Mustafa Zen (s4980040)**

## 1. User Need, Scope and Acceptance Criteria

**User:** a radiotherapy planning team that needs first-draft contours of the prostate and nearby organs (bladder, rectum, bone) on pelvic MRI, to shorten manual contouring.

**Scope:** a research prototype that produces draft contours for clinician review only. It makes no treatment decisions and uses HipMRI data alone. It asks whether 3D volumetric context reduces errors at the prostate base and apex, or simply costs more compute. All criteria are tested on the held-out test patients (cases 040–042, 9 scans):

| ID | Criterion and test | Pass threshold |
|---|---|---|
| AC1 | Test prostate Dice at full native resolution | 3D ≥ 0.70; 2D baseline ≥ 0.75 |
| AC2 | Prostate Dice on boundary slices (first and last 20% of prostate-containing slices per scan) | 3D > 2D baseline |
| AC3 | Change in rectum and bladder Dice, 3D vs 2D (leakage at organ interfaces) | Drop ≤ 0.02 each |
| AC4 | Training time; peak GPU memory; inference per scan | ≤ 5× baseline; ≤ 16 GB; ≤ 2 s |

## 2. Model Choice and Course Concepts

**Data:** the HipMRI 2D slices for the baseline and the 3D volumes for the main model, using the course's patient-level split (31 / 4 / 3 patients). The 3D volumes are z-score normalised, downsampled ×2 from 256×256×128 to 128×128×64 (trilinear for images, nearest-neighbour for masks) and randomly flipped along each axis during training.

**Models:** a 2D U-Net baseline [1] (Easy, 31.0 M parameters) and the main model, a 3D Improved U-Net [2] (Hard, 9.5 M), with residual context modules, deep supervision and instance normalisation. Both are trained with the same combined cross-entropy and Dice loss and the Adam optimiser with cosine learning-rate decay, so the main difference between them is the spatial context.

**Course concepts:** encoder–decoder CNNs with skip connections to preserve boundary detail; residual learning for stable deep training; normalisation; augmentation; and a Dice-based loss to counter class imbalance, since the prostate occupies a very small fraction of each image. 3D convolutions [3] extend each filter to neighbouring slices, which is exactly the context the dilemma asks about.

## 3. Preliminary Feasibility Evidence

- **Data audit:** the official split has no patient overlap; images and masks pair one-to-one (11,460 / 660 / 540 slices); six labels (0–5); the 2D and 3D sets cover the same nine test scans.
- **Pipeline:** both models train and evaluate end to end. The 3D model is scored by upsampling its predictions to the original 256×256×128 and comparing them with the original masks.

| Model | Prostate Dice | Boundary | Middle | Rectum | Train time | Inference / scan |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | 0.863 | 0.707 | 0.898 | 0.851 | 4.1 h | 0.74 s |
| 3D Improved U-Net | 0.859 | 0.778 | 0.903 | 0.818 | 12.1 h | 0.84 s |

*Measured on an Apple M2 Pro (MPS), which cannot report peak GPU memory.*

AC1 and AC2 are met. AC3 narrowly fails (rectum −0.033), most likely because downsampling blurs thin structures. AC4 is met for time (about 3×), but peak memory has not yet been measured.

## 4. Risks, Budget and Fallback

- **Small test set** (3 patients, 7 of the 9 scans from one patient): report per-scan results and avoid claiming small differences.
- **Detail lost by downsampling** at the apex and the rectal wall: score at full resolution; try full-resolution patch training.
- **Compute access:** the Rangpur queue has been unavailable, and the Mac cannot measure VRAM. Use an RTX 5060 laptop or Rangpur when free.

**Budget:** about 24 GPU-hours for the remaining experiments.

**Next experiment:** measure peak VRAM on a CUDA GPU to complete AC4, then train the 3D model on 128-voxel patches at full resolution to target AC3.

**Fallback:** if patch training exceeds the budget, keep the current downsampled 3D model, which already meets AC1 and AC2. A plain 3D U-Net (the same network with residuals and deep supervision switched off) is already implemented as a Normal-difficulty alternative.

[1] Ronneberger et al., U-Net, MICCAI 2015. [2] Isensee et al., BrainLes 2017 (BraTS 2017). [3] Çiçek et al., 3D U-Net, MICCAI 2016.