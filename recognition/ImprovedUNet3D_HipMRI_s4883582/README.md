# 3D Improved U-Net for Prostate Segmentation on HipMRI

**Student:** s4883582 · **Task:** 3D Improved U-Net [Hard Difficulty] on downsampled HipMRI 3D data ·
**Baseline:** 2D U-Net in [`../UNet2D_HipMRI_s4883582`](../UNet2D_HipMRI_s4883582)

## Contents
1. [Problem](#1-problem)

2. [Algorithm](#2-algorithm)

3. [Feasibility Review](#3-feasibility-review)

4. [Data, Pre-processing and Splits](#4-data-pre-processing-and-splits)

5. [Usage](#5-usage)
6. [Dependencies and Reproducibility](#6-dependencies-and-reproducibility)

7. [Results](#7-results)

8. [Open Research Dilemma: 2D vs 3D Context](#8-open-research-dilemma-2d-vs-3d-context)

9. [Recommendation](#9-recommendation)

10. [Limitations and Next Steps](#10-limitations-and-next-steps)

11. [Artificial Intelligence Usage Disclosure](#11-artificial-intelligence-usage-disclosure)

12. [References](#12-references)

## 1. Problem

External-beam radiotherapy for different cancers is a millimetre precise therapy. For prostate cancer,
it needs the prostate contoured on MRI whilst avoiding other vital areas like the bladder and rectum.
Over-contouring will irradiate healthy organs and tissue potentially causing cystitis or proctitis.
Under-contouring leaves untreated tumour behind potentially leading to the cancer not regressing completely.
As contouring by hand is slow and tedious, automatic drafting with slight corrections can save time
as long as errors produced are predictable.

This prototype segments six classes in each image (background, body, bone, bladder, rectum, prostate)
in 3D pelvic MRI volumes from the HipMRI study [4]. As the prostate only occupies approximately 0.1-0.3%
of each volume, it is simultaneously the most difficult and most critical class to target. With 
this, the following can be answered:

**Does true 3D volumetric context fix the boundary errors at the prostate base and apex that a 2D 
slice-by-slice model makes, or does it only improve the easy central slices while costing more GPU 
memory and time?**

## 2. Algorithm

The model utilised is the 3D Improved U-Net of Isensee et al. [2], implemented from the paper in
PyTorch [13] ([`modules.py`](modules.py)). Like U-Net [1] and 3D U-Net [3], an encoder repeatedly halves the
input resolution to build up context and a decoder upsamples the resultant image back into full
resolution, with skip connections carrying fine details across. Isensee's changes to the original
model make it better in theory for small, imbalanced medical datasets.

Firstly, each encoder level below full resolution downsamples with a stride-2 3x3x3 convolution, then
applies a residual context module: a pre-activation residual block [5] (norm → LeakyReLU → conv →
dropout 0.3 [6] → norm → LeakyReLU → conv, added to its input). Instance normalisation [7] and
LeakyReLU (0.01) are used throughout, because batch statistics are unreliable with batches of 1-2 volumes. The decoder upsamples by repeating voxels (avoiding checkerboard artefacts [8]), concatenates
the skip connection, then recombines features with a 3x3x3 and 1x1x1 convolution. Segmentation layers
at 1/4, 1/2, and full resolution are upscaled and summed into the output (deep supervision [9]) so gradients reach lower decoder
layers directly.

The loss is cross-entropy plus soft multi-class Dice [10] (the same as the 2D baseline). Dice weighs every class
equally regardless of size, so the tiny prostate is not drowned out by background; cross-entropy keeps
voxel-wise gradients stable early in training. The model has **9.46M parameters**, a third of the 2D
U-Net's 31.0M.

## 3. Feasibility Review

The one-page review presented at the midpoint check-off is reproduced below, with typos fixed. It is
followed by a check of each acceptance criterion against the final results and the changes made since.

**Anatomical Contouring: 2D vs 3D Context in Prostate MRI**

**User needs, scope and acceptance criteria.** The intended users of this model are imaging specialists
planning and performing external-beam radiotherapy. The core domain dilemma is spatial context vs.
clinical boundary utility: over-contouring causes proctitis/cystitis, while under-contouring risks lethal
recurrence. The prototype will present automated anatomical contours for imaging review, while
investigating whether 3D context minimises boundary errors compared to restricted 2D context. This will be
done by training a 2D and a 3D U-Net.

| ID | Dimension | Target metric | Constraints | Verification |
|----|-----------|---------------|-------------|--------------|
| AC1 | Task performance | Dice coefficient > 0.7 | HipMRI downsampled test set | Global Sørensen-Dice script |
| AC2 | Clinical boundary utility | Base/apex slice IoU tracking | Avoid leakage into critical OARs | Visual multi-panel autopsy |
| AC3 | Resource footprint (VRAM) | Peak GPU memory <= 16 GB | Single Colab T4/L4 | PyTorch VRAM profiling |
| AC4 | Runtime / throughput | Inference fast enough for drafting | Acceptable per-case wait time | Inference time profiling |

**Model choice and course concepts.** Data will be sourced from the HipMRI dataset. Strict patient-level
splitting will be implemented to prevent data leakage. The 2D and 3D U-Net will be evaluated on the same
test data to determine whether the increased memory cost of the 3D U-Net is justified. As the prostate is
< 2% of the pelvic scan volume, Dice/Focal loss is needed, and the comparison checks whether the 3D network
reduces the loss enough to justify the extra memory.

**Preliminary feasibility evidence.** An initial data audit of the HipMRI dataset confirms the dataset is
structurally sound. With data integrity verified, operational feasibility depends on the GPU
constraints. The 2D U-Net is highly memory-efficient compared to the 3D, consuming approximately 2-4 GB of
VRAM and completing 100 epochs in about 2-3 hours at 1-2 minutes per epoch. In contrast, the 3D U-Net is
estimated to consume approximately 16 GB of VRAM and take roughly 16-25 hours for 100 epochs.

**Risks, budget and fallback.**

| Risk | Description | Warning sign | Mitigation |
|------|-------------|--------------|------------|
| Shared server limitations | The shared server may be in use, delaying testing and runs | `squeue` shows the job held for GPU resources | Run models at times of low activity and allow for delays |
| Missing/corrupted images | Images that do not fit the expected format may crash the model at runtime | Sudden, unexpected stop during training | Scan incoming images and reject those that do not fit the expected standard |
| Extreme class imbalance | The prostate is less than 2% of the volume | The model predicts only background and prostate Dice drops to zero | Use Dice/Focal loss to force foreground learning |

### Acceptance criteria against the final results

Each criterion is checked against the final full-resolution 3D run, with a link to the evidence.

| ID | Dimension | Target (from review) | Result | Status | Evidence |
|----|-----------|----------------------|--------|--------|----------|
| AC1 | Task performance | Prostate Dice > 0.7 on the HipMRI test set | Test prostate Dice **0.8556** (IoU 0.7476) | Met | [Results](#7-results) |
| AC2 | Clinical boundary utility | Track base/apex slice accuracy; avoid leakage into critical OARs | End-slice Dice **0.70** vs central **0.885**; base ends worse (0.63) than apex ends (0.77); prostate predicted on 1-11 slices beyond the true extent in 25/30 volumes; leakage into the bladder/rectum not measured separately | Tracked, boundary gap remains | [Dilemma](#8-open-research-dilemma-2d-vs-3d-context) |
| AC3 | Resource footprint | Peak GPU memory <= 16 GB | Training **7.83 GB**, inference **3.36 GB** | Met | [Results](#7-results) |
| AC4 | Runtime / throughput | Inference fast enough for drafting | **79.4 ms** per volume | Met | [Results](#7-results) |

### [Updated] Changes since the review
- **Hardware:** trained on Rangpur rather than a single Colab T4/L4. The 7.83 GB peak would still fit on a
  16 GB T4.
- **Cost estimate:** the review estimated ~16 GB of VRAM and 16-25 hours for 100 epochs of 3D training. The
  actual run used 7.83 GB and ~1.9 hours for 123 epochs (~57 s per epoch), with batch size 1 and mixed
  precision.
- **Loss:** cross-entropy + Dice was used rather than Dice/Focal. The prostate Dice never collapsed to zero,
  so Focal loss was not needed.
- **Same test data:** the review planned to evaluate 2D and 3D on the same test patients. The final 3D split
  is independent of the 2D one, so the comparison is on different test patients (see
  [Limitations](#10-limitations-and-next-steps)).
- **Risks:** the unexpected image risk did occur: one volume has 144 slices instead of 128. It is clipped
  to 128 in `load_nifti_volume` rather than rejected, and any 4D volume is reduced to 3D the same way
  (see [Data](#4-data-pre-processing-and-splits)). The class imbalance risk was handled by the Dice term in
  the loss.

## 4. Data, Pre-processing and Splits
**Data.** HipMRI Study 3D volumes [4] on Rangpur: images in `semantic_MRs`, labels in `semantic_labels_only`
(211 volumes, 38 patients, up to 8 weekly scans per patient). Files are paired by name
(`B006_Week0_LFOV.nii.gz` ↔ `B006_Week0_SEMANTIC.nii.gz`) rather than by sorted order.

**Pre-processing** ([`dataset.py`](dataset.py)):
1. Load with nibabel and drop any trailing 4th dimension (adapted from the spec's Appendix B loader).
2. Clip to the first 128 slices, so the one 144-slice volume has the same depth and spacing as the rest
   (as Appendix B does).
3. Z-score normalise each image volume (raw intensities range from 0 to ~600 and vary between scans).
4. Resize to `VOLUME_SIZE`: trilinear interpolation for images, **nearest-neighbour for masks** so class
   labels are never blended. The final model uses the native 256×256×128; the first experiment used
   128×128×64.
5. Keep masks as integer class indices; the loss one-hot encodes them internally.

All volumes are loaded and resized once at start-up, so the compressed files are not decoded every epoch.

**Augmentation** (training split only, applied identically to image and mask):
- Random left-right flip. Only left-right: the pelvis is roughly symmetric that way, while flipping the
  other axes would put the bladder below the prostate, which never occurs in a real scan.
- Random in-plane rotation (±10°) and isotropic scaling (0.9-1.1×), which also covers the 1.41-1.88 mm
  spread in voxel size between scans.
- Random brightness/contrast change and Gaussian noise (image only).

**Splits.** 
Patients are shuffled with a fixed seed (42) and split approximately 80/10/10. Splitting by volume would
put weekly scans of the same patient in different splits and leak test data into training, so every
weekly scan of a patient is kept in the same split. `check_patient_leakage` confirms that the splits are
patient-disjoint.

| Split | Patients | Volumes |
|-------|----------|---------|
| Train | 30 | 163 |
| Validation | 4 | 18 |
| Test | 4 (J026, M030, R039, S022) | 30 |

The best checkpoint is chosen on validation prostate Dice, and the test split is used only once, 
at the end.

## 5. Usage
1. Install the dependencies (section 6) and create a `.env` file in this folder pointing at the dataset
   (it is git-ignored, so each machine keeps its own):
   ```
   HIPMRI_ROOT=/home/groups/comp3710/HipMRI_Study_open
   ```
2. Train, then evaluate. As a Rangpur batch job:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=unet3d
   #SBATCH --partition=a100
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=48G
   #SBATCH --time=08:00:00
   #SBATCH --output=unet3d_%j.out
   #SBATCH --error=unet3d_%j.err

   conda activate torch
   cd ~/Unet_HipMRI_s4883582/recognition/ImprovedUNet3D_HipMRI_s4883582
   python -u train.py && python -u predict.py
   ```

3. Outputs in `outputs/`:

| Script | Prints | Saves |
|--------|--------|-------|
| [`train.py`](train.py) | Split sizes, per-epoch loss and validation prostate Dice, peak training VRAM, final test Dice/IoU | `unet3d_best.pt`, `unet3d_training_curves.png` |
| [`predict.py`](predict.py) | Test Dice/IoU per class, parameters, latency, peak inference VRAM, end vs central slice Dice, worst slices | `unet3d_per_slice_dice.png/.csv`, `unet3d_failure_1..5.png` |

| File | Purpose |
|------|---------|
| [`modules.py`](modules.py) | 3D Improved U-Net, Dice loss, combined CE + Dice loss (PyTorch only) |
| [`dataset.py`](dataset.py) | Nifti loading, pre-processing, patient-level split, 3D augmentation, leakage check |
| [`utils.py`](utils.py) | Per-class Dice/IoU accumulated over a split; per-slice Dice |
| [`train.py`](train.py) | Training with mixed precision, LR decay, early stopping, checkpointing, plots, test evaluation |
| [`predict.py`](predict.py) | Test evaluation, resource profiling, per-slice analysis, failure figures |
   
## 6. Dependencies and Reproducibility
| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.13.0 (CUDA 13.0) |
| NumPy | 2.4.6 |
| nibabel | 5.4.2 |
| matplotlib | 3.11.2 |
| python-dotenv | 1.2.4 |

[`requirements.txt`](requirements.txt) is the `pip freeze` of the conda environment used on Rangpur, without
the Linux-only CUDA packages (`nvidia-*`, `cuda-*`, `triton`), which `torch` installs itself on Linux.
Install with `pip install -r requirements.txt`.

Training ran on a single NVIDIA GPU on Rangpur (`a100` partition).

**Hyperparameters** ([`train.py`](train.py)): Adam [11], learning rate 5e-4 decayed ×0.985 per epoch (as in
[2]), weight decay 1e-5, batch size 1 (full resolution) or 2 (half resolution), up to 200 epochs, early
stopping after 40 epochs without a validation prostate Dice improvement, mixed precision [12] (float16
autocast with gradient scaling, Dice computed in float32).

**Reproducibility.** The patient split is deterministic (seeded `torch.randperm` over sorted patient IDs),
and `torch.manual_seed(42)` is set before training. cuDNN convolution algorithms and data-loader worker
timing are not fully deterministic, so retraining gives close but not bit-identical results. Evaluation is
deterministic: `predict.py` reproduced `train.py`'s test scores exactly from the saved checkpoint.

## 7. Results
![Per-slice prostate Dice on the test volumes](images/unet3d_per_slice_dice.png)

*Prostate Dice for every slice of every test volume, plotted against position within the ground-truth
prostate (0 = apex end, 1 = base end). The model is accurate through the body of the gland and fails at
the ends, especially the base.*

![Training curves](images/unet3d_training_curves.png)

Training and validation losses fall together and level off (at about 0.07 and 0.10) with no upturn in
validation loss. Validation Dice passes 0.7 by epoch 5 and plateaus at approximately 0.87 from
epoch 40. The best checkpoint came from epoch 83 and was stopped early at epoch 123. This contrasts
with the 2D baseline, in which the validation loss started rising after epoch 8.

### Test set (4 patients, 30 volumes)
| Class | 3D full res Dice | 3D full res IoU | 3D half res Dice | 3D half res IoU | 2D U-Net Dice* | 2D U-Net IoU* |
|-------|------|------|------|------|------|------|
| Background | 0.9963 | 0.9926 | 0.9927 | 0.9855 | 0.9970 | 0.9941 |
| Body | 0.9866 | 0.9736 | 0.9792 | 0.9593 | 0.9848 | 0.9700 |
| Bone | 0.9297 | 0.8687 | 0.9128 | 0.8395 | 0.9353 | 0.8785 |
| Bladder | 0.9617 | 0.9263 | 0.9452 | 0.8961 | 0.9667 | 0.9356 |
| Rectum | 0.8843 | 0.7926 | 0.8617 | 0.7570 | 0.8635 | 0.7598 |
| **Prostate** | **0.8556** | **0.7476** | 0.8355 | 0.7175 | 0.8823 | 0.7894 |

\*The 2D U-Net was evaluated on its own held-out split (3 patients, 540 slices), not on the same patients
as the 3D model; see [Limitations](#10-limitations-and-next-steps). Dice and IoU are accumulated over the
whole split rather than averaged per volume.

**AC1 (prostate Dice > 0.7): met**, at 0.8556.

### Resource profiling
| Model | Parameters | Peak training VRAM | Peak inference VRAM | Inference latency | Training time |
|-------|-----------|--------------------|---------------------|-------------------|---------------|
| 2D U-Net | 31.04M | not measured | 1,219.5 MB (batch 16) | 1.35 ms per slice | not measured (50 epochs) |
| 3D half res (128×128×64) | 9.46M | 2,054.8 MB (batch 2) | 468.7 MB (batch 1) | 11.2 ms per volume | ~13 min (107 epochs × ~7 s) |
| 3D full res (256×256×128) | 9.46M | 7,834.9 MB (batch 1) | 3,359.4 MB (batch 1) | 79.4 ms per volume (≈0.62 ms per slice) | ~1.9 h (123 epochs × ~57 s) |

Latency is measured on the GPU after one warm-up pass, synchronising before and after each forward pass.

**AC3 (≤ 16 GB): met** (7.8 GB peak). **AC4 (fast enough for drafting): met** (79 ms per volume).

## 8. Open Research Dilemma: 2D vs 3D Context
### Boundary vs central slices
`predict.py` measures prostate Dice on every axial slice and separates the slices at each end of the
ground-truth prostate (base and apex) from the central ones. The end zone is 4 slices at full resolution
and 2 at half resolution, both about 6 mm.

| Model | End slices (base + apex) | Apex end | Base end | Central slices |
|-------|--------------------------|----------|----------|----------------|
| 3D half res | 0.7072 (120 slices) | - | - | 0.8694 (242 slices) |
| 3D full res | 0.6996 (240 slices) | 0.769 | **0.630** | 0.8852 (492 slices) |

Doubling the resolution raised central-slice Dice by 0.016 and overall prostate Dice by 0.020, but did
nothing for the end slices (0.707 → 0.700). Even with full 3D context, the model loses about 0.19 Dice
at the ends of the gland, and the base is clearly the worst. **AC2 (base/apex tracking): the boundary
gap remains.** Extra context and resolution mostly improved the slices that were already easy.

### Failure case autopsy
The slice axis is the scanner's axial axis; the end with the bladder in view (high slice numbers) is
the base, and the opposite end is the apex. Across the 30 test volumes the errors fall into a few
recurring modes:

**1. Over-extension past the apex: J026_Week0, slice 44 (Dice 0).**
![Apex over-extension](images/failure_apex_overextension.png)
Two slices below the last annotated prostate slice, the model incorrectly predicts a small prostate 
blob directly in front of the rectum (in red). In 25/30 volumes, the model predicts the prostate on 1-11 slices
(mean of 3.1) beyond where the ground truth stops. As the prostate tapers smoothly into the surrounding
tissue, where the gland "ends" is partly an annotation decision, and the 3D model, having learned a
smooth gland shape, extrapolates it past that point. Four of the five lowest-scoring slices that
`predict.py` reported are over-extensions (three past the apex, one past the base); the fifth is the base
miss below.

**2. Missed base at the bladder neck: J026_Week0, slice 67 (Dice 0).**
![Base miss](images/failure_base_miss.png)
On the last annotated slice, the prostate sits between the bladder and the rectum, and the model misses
it completely (in blue), labelling the area as body tissue. In 11/30 volumes, the last base slice scores
Dice ≤ 0.1, and the base end averages 0.630. A likely cause is that, at the bladder neck, the prostate
and bladder wall have similar T2 intensity and the boundary between them is blurred by partial volume.

**3. Whole apex missed: R039_Week0, slices 33-36.** The first four annotated slices (about 6 mm) have
Dice 0, while the rest of the gland is segmented well (0.66-0.81 at the base end). This is the only
volume where the apex is missed over several slices, so it is rare but large when it happens.

**4. Both ends wrong at once: S022_Week6.** Prostate is predicted on 3 slices below the annotated apex,
while the last 3 annotated base slices have zero overlap. All 8 weekly scans of S022 over-extend by 2-4
slices, so the error is consistent within a patient.

**5. One patient dominates the worst cases.** All five lowest-ranked slices come from patient J026,
just as all five worst 2D slices came from patient 040. With 3-4 test patients, a single difficult
anatomy decides which failures are visible.

**Unsuitable for automated drafting:** the base slices next to the bladder neck, and the first and last
few slices of the predicted extent (where over-extension occurs). The central gland is reliable
(Dice 0.885 on central slices).

## 9. Recommendation
It is recommended to use the 3D Improved U-Net as a ***drafting tool*** for the body of the
prostate, but never as the final contour at the base or apex.

At full resolution, the 3D Improved U-Net gives accuracy comparable to the 2D baseline (prostate Dice
0.856 vs. 0.882) and whole-volume inference in 79 ms, with a model a third of the size of the 2D
baseline. Unfortunately, base and apex Dice were approximately 0.70 at both resolutions, so 3D context
alone does not remove the clinically dangerous errors at the ends of the gland. Over-extension and under-segmentation will lead
to separate risks that can be avoided by manual inspection.

Half resolution is the cheapest option tested (2.1 GB training / 0.47 GB inference VRAM, 11 ms per
volume, 13 min training). Full resolution costs about 3.8x the training memory, 7x the inference memory
and time, and about 9x the training time, for +0.02 prostate Dice, gained in slices that were already
performant. Both fit within the 16 GB budget, so full resolution is affordable but not justified:
**half resolution is the recommended configuration.**

A clinician must review and edit each end (approximately 6 mm) of the prostate prediction, with
particular care at the bladder neck, and check the slices beyond each end of the predicted extent for
spurious prostate (it over-extends by 1-11 slices, 3 on average).

## 10. Limitations and Next Steps

As 2D and 3D were tested independently, the baseline comparison is between two different splits.
The next experiment should use an identical split to more accurately evaluate models. A larger test
set would also improve results for the experiment. The 2D baseline's end-slice Dice was not measured,
so the boundary comparison rests on the 3D results. As each model was only trained once, run-to-run
variance was not measured.

Adding boundary-aware loss or extra weight on end slices would improve the next iteration of the model
as well as post-processing that keeps the largest connected prostate component and trims the
predicted extent.

## 11. Artificial Intelligence Usage Disclosure
- **Tool used:** Claude Opus 5.5.
- **Tasks it assisted with:**
  - Planning the 3D work from the spec and my 2D baseline.
  - Initial scaffolding of the 3D `dataset.py`, `modules.py`, `train.py`, `predict.py` and `utils.py`, adapted from my
    own 2D code.
  - Scaffolding the initial README design.
- **How I verified it:**
  - Ran every script on synthetic Nifti volumes before using real data:
    - checked output shapes;
    - checked Dice loss = 0 and Dice = 1 on a perfect prediction;
    - checked the masks stay aligned with the images after flips and rotations;
    - checked that the 144-slice volume is clipped correctly.
  - Checked the architecture against Isensee et al. [2] and the loader against spec Appendix B.
  - On real data:
    - confirmed the patient split has no leakage;
    - confirmed `predict.py` reproduces `train.py`'s test scores exactly.

## 12. References
1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in Proc. MICCAI, 2015, pp. 234-241. https://arxiv.org/abs/1505.04597
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, K. H. Maier-Hein, "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," in Brainlesion: Glioma, Multiple Sclerosis, Stroke and TBI Models, 2018, pp. 287-297. https://arxiv.org/abs/1802.10508
3. Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation from sparse annotation," in Proc. MICCAI, 2016, pp. 424-432. https://arxiv.org/abs/1606.06650
4. J. Dowling, P. Greer, "Labelled weekly MR images of the male pelvis," CSIRO Data Collection, v2, 2021. https://doi.org/10.25919/45t8-p065
5. K. He, X. Zhang, S. Ren, J. Sun, "Identity mappings in deep residual networks," ECCV, 2016. https://arxiv.org/abs/1603.05027
6. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," JMLR, vol. 15, pp. 1929-1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html
7. D. Ulyanov, A. Vedaldi, V. Lempitsky, "Instance normalization: The missing ingredient for fast stylization," arXiv:1607.08022, 2016. https://arxiv.org/abs/1607.08022
8. A. Odena, V. Dumoulin, C. Olah, "Deconvolution and checkerboard artifacts," Distill, 2016. https://doi.org/10.23915/distill.00003
9. C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, Z. Tu, "Deeply-supervised nets," AISTATS, 2015. https://proceedings.mlr.press/v38/lee15a.html
10. F. Milletari, N. Navab, S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," 3DV, 2016. https://doi.org/10.1109/3DV.2016.79
11. D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," ICLR, 2015. https://arxiv.org/abs/1412.6980
12. P. Micikevicius et al., "Mixed precision training," ICLR, 2018. https://arxiv.org/abs/1710.03740
13. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," arXiv:1912.01703, 2019. https://arxiv.org/abs/1912.01703
