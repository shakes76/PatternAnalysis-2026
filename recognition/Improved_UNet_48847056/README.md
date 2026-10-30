# 2D vs 3D U-Net for Prostate Segmentation in Radiotherapy MRI (HipMRI)

## 1. Algorithm and Problem Description
External-beam radiotherapy for prostate cancer requires the prostate to be contoured on all MRI slices with millimeter precision. Under-contouring poses a risk of tumor reoccurence, and over-contouring can irradiate neighboring organs and tissue potentially causing inflammation and damage. Manually contouring is a slow and tedious process, hence, this project tests a deep-learning drafting assistant that proposes contours for a specialist to review. The investigation is the dilemma of spatial context in this model. A 2D model segments each slice on its own and is relatively cheap to train, while a 3D model can use the spatial context of the whole 3D volume and in theory should produce more continuous contours. This project trains both models on the same patient-level split and measures the difference that 3D context makes.

**How it works:** Both models are the same U-Net, built from the same class in 'modules.py'. The only difference between the two models is the use of 2D or 3D convolution blocks. The encoder applies two 3x3 convolutions with BatchNorm and ReLU at each level of the U-Net, and then uses max-pooling to halve the resolution. This is repeated 4 times for the downsampling half, with channels doubling consecutively from 16 to 256. The decoder then upsamples with transposed (up) convolutions and concatenates the matching encoder features (skip connections). A 1x1 convolution then outputs one score per class for each pixel/voxel.

The loss used is cross entropy plus (1 - mean soft Dice).
- cross entropy gives stable early gradients, and Dice prevents the background from skewing results as the prostate only accounts for a small ratio of pixels/voxels. The 2D model is trained on individual axial slices, with the 3D model trained on the full volumes. For the testing, the 2D model predicts all slices of a given volume and the slices are stacked back together, <ins>so both models are scored on the same voxels.</ins>

![Original U-Net architecture: a contracting encoder (64 to 1024 channels) and an expanding decoder joined by copy-and-crop skip connections, producing a 2-class segmentation map.](images/Screenshot%202026-10-30%20123109.png)
*Figure 1: Basic U-Net architecture as provided in the lecture slides*

![MRI U-Net: a 1-channel input, a four-level encoder (16 to 128 channels), a 256-channel bottleneck, and a decoder with skip connections, ending in a 1×1 conv to 6 class logits](images/UNet%20architecture.drawio.png)
*Figure 2: U-Net architecture used for both the 2D and 3D U-Net models. Only difference is the convolution dimensions*


## 2. Feasibility Review
### User Need, Focus, and Acceptance Criteria
User is a radiotherapy imaging specialist that works with anatomical contours of the prostate gland
Prototype purpose: to investigate whether the true 3d volumetric continuity of a 3D U-Net model can eliminate the boundary and slice-transition errors of a 2D U-Net baseline in prostate radiotherapy planning, or if it only refines the easy central slices while consuming much more compute.
### Acceptance criteria
- Prostate label dice sim. >= 0.75 for 2d U-Net, global dice sim. >= 0.7 for 3d U-Net (test patients)
    - Global dice similarity may not capture the success of the model for the base and apex slices, but provides a good validity check
- Stratified Dice similarity by slice position (base, mid, apex) accompanied by HD95 to quantify the gap between predicted boundary and ground truth for each stratum.
    - This helps determine and quantify where 3D U-Net differs from 2D U-Net for each primary area of the prostate scans.
- Peak GPU VRAM and inference time per volume stays under 4 GB/5 seconds for 2D U-Net and 12 GB/15 seconds for 3D U-Net
### Model Choice & Course Concepts
- Dataset configuration:
    - HipMRI study: 38 patients, 211 3d volumes, labels: body, bone, bladder, rectum, prostate. 
    - Patient-level splitting: all slices from a given patient end up in the same train/validation/test split. Prevents leakage between sets. typical split ratio to be ## used: 70/15/15 -> 26/6/6 patients. 
- Preprocessing: 
    - Normalise image intensity per image using z-score; One-hot encoding for labels
- Model architecture:
    - 4 levels (3 downsamples)
    - Filters starting at 32/16, doubling each level.
    - Block: 2 x (3x3 conv, norm, ReLU) (3d convolution for 3D UNet)
    - Models kept similar in architecture aside from increase in convolution dimension – this should allow test results to better reflect the effectiveness of using a 3D UNet over a 2D one. 
### Preliminary Feasibility Evidence:
Initial Data Audit: scans are of 38 patients some over a period of up to 7 weeks of radiotherapy – for the 3d data it is difficult to tell what scans belong to a single patient.
Results of baseline smoke-test experiment/resource estimate for TinyUNet3d on cloud-hosted GPU: 
- (64, 64, 32) peak VRAM: 0.07 GiB
- (128, 128, 64) peak VRAM: 0.56 GiB
### Risks, Budget, and Fallback
Training 3D-UNet may exceed memory or time limitations – mitigation: train using sub-sections to begin with; mixed precision training, possibly lower input resolution if needed
Only 38 patients – stratified results may be noisy, and test set only contains 6 patients with the proposed splitting ratio. Treatment: report per-patient values for test set. Could possibly implement k-fold cross val
Primary computing budget: using the cloud hosted GPUs run by UQ’s GPU lab – hard quota of 26 hrs run time total.
Fallback – code should be compatible with Rangpur so that if GPU lab quota run’s out, it can be run on the Rangpur GPUs
2D Improved UNet as a back up model

### Changes following the review and evaluation of acceptance criteria
- **Compute:** the cloud GPU lab could not be accessed, so the fallback was used: all training ran on Rangpur (A100).
- **Architecture:** the final models use 4 downsampling steps (5 levels, 16→256 channels), since a 128×128×64 input divides evenly by 2⁴.
- **Split:** rounding 70/15/15 over 38 patients gave 27/6/5 patients rather than 26/6/6.
- **Labels:** stored as integer class indices and one-hot encoded inside the loss, which saves memory.
- **HD95** was not implemented due to time. Position along the gland is instead measured with Dice binned into 10 positions from base to apex (Section 5.2), plus visual failure overlays (Section 5.4).
- **Acceptance criteria outcome:**

| Criterion | Target | Result | Met? |
|---|---|---|---|
| 2D prostate Dice | ≥ 0.75 | 0.792 | ✅ |
| 3D prostate Dice | ≥ 0.70 | 0.847 | ✅ |
| Position-stratified Dice | reported | 10 bins, base to apex | ✅ |
| HD95 per stratum | reported | not implemented | ❌ |
| 2D VRAM / time per volume | < 4 GB / 5 s | 0.28 GB / 4.9 ms | ✅ |
| 3D VRAM / time per volume | < 12 GB / 15 s | 1.3 GB (train) / 8.8 ms | ✅ |

## 3. Data, Pre-processing and Splits
**Data.** HipMRI Study (`/home/groups/comp3710/HipMRI_Study_open` on Rangpur): 211 3D MRI volumes from 38 patients, many scanned weekly during treatment. Images are in `semantic_MRs`, labels in `semantic_labels_only`, with 6 classes (0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate).

**Pre-processing** (`dataset.py`):
1. Load each Nifti volume with nibabel; drop the extra 4th dimension that some files have.
2. Resample every volume to **128×128×64** (some scans were different sizes). Images use trilinear interpolation and labels use **nearest-neighbour** (because class samples shouldn't be interpolated). Resampling keeps all anatomy instead of clipping or padding it, while making 3D training fit comfortably in memory.
3. **Z-score normalise** each image volume (subtract mean, divide by standard deviation), since MRI intensities have no fixed scale between scans.
4. For the 2D baseline, the training volumes are cut into their 64 axial slices each (128×128).
5. **Augmentation** (training only): random intensity scaling (×0.9–1.1) and shift (±0.1) to imitate scanner variation. Left-right flipping is implemented but switched off, because flipping the wrong axis would swap anatomy (for example, putting the bladder behind the rectum).

**Splits.** Patients are shuffled with a fixed seed and split **70/15/15 by patient**: 27 training, 6 validation and 5 test patients. All weekly scans of a patient stay in the same split, because scans of one patient look almost identical, splitting by scan would leak test anatomy into training and inflate scores. `check_no_leakage()` checks this on every run and stops with an error if any patient appears in two splits. Both models use **the same split**, the 2D model being trained on the slices of the same 27 training patients. Validation is used only to pick the best checkpoint (highest validation prostate Dice); the test set is used once, at the end.

## 4. Training Setup

| Setting | 2D U-Net | 3D U-Net |
|---|---|---|
| Training sample | 128×128 axial slice | 128×128×64 volume |
| Batch size | 16 slices | 2 volumes |
| Epochs | 30 | 60 |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 | same |
| LR schedule | cosine annealing to 0 | same |
| Loss | cross-entropy + (1 − mean soft Dice) | same |
| Precision | mixed (float16 autocast + GradScaler) | same |
| Checkpoint | best validation prostate Dice | same |
 
The 3D model gets more epochs because each epoch is only ~74 optimiser steps (2 volumes per batch), against ~600 for 2D (16 of ~1,700 slices per batch); the 2D model still takes more steps in total. Both models select their checkpoint on validation Dice, so neither is judged on an arbitrary final epoch.
 
![2D training curves](figures/unet2d_curves.png)
![3D training curves](figures/unet3d_curves.png)

## 5. Results: Investigating the Spatial Context Dilemma

All numbers are on the same held-out test split (5 patients, whole volumes at 128×128×64), using each model's best validation checkpoint.

### 5.1 Segmentation accuracy

| Metric | 2D U-Net (baseline) | 3D U-Net |
|---|---|---|
| Body Dice / IoU | 0.966 / 0.934 | **0.972 / 0.945** |
| Bone Dice / IoU | 0.897 / 0.813 | **0.903 / 0.824** |
| Bladder Dice / IoU | **0.826 / 0.704** | 0.755 / 0.607 |
| Rectum Dice / IoU | **0.820 / 0.696** | 0.816 / 0.689 |
| **Prostate Dice / IoU** | 0.792 / 0.656 | **0.847 / 0.734** |
| Prostate Dice per volume, mean (worst) | 0.786 (0.702) | **0.838 (0.751)** |
 
Both models pass their respective prostate target. The 3D model improves prostate Dice by 0.055 and lifts the worst test volume from 0.702 to 0.751. Its bladder Dice drops by 0.07, however, so 3D context does not help every structure.

### 5.2 Where along the prostate the models fail (base to apex)

As mentioned in the task sheet, the global Dice score is dominated by the central slices of the prostate. To test this dilemma, each test slice which contained the prostate was binned by its position between the base (bladder end) and the apex, and per-slice Dice was averaged within each bin. To determine the base slice, the bladder position is determined by averaging the slices containing the bladder. 

![Prostate Dice from base to apex](figures/slice_position.png)
 
| Position bin | 0.0–0.1 (base) | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 (apex) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2D U-Net | 0.496 | 0.795 | 0.853 | 0.885 | 0.876 | 0.866 | 0.816 | 0.760 | 0.710 | 0.556 |
| 3D U-Net | 0.596 | 0.835 | 0.898 | 0.907 | 0.909 | 0.902 | 0.877 | 0.865 | 0.797 | 0.597 |
| Slices per bin | 31 | 20 | 24 | 23 | 18 | 25 | 24 | 23 | 21 | 35 |

- Both models score about 0.9 on central slices but drop to 0.5–0.6 at the two ends. The global Dice of 0.79–0.85 hides this as expected
- The 3D model is better in **every** bin. The largest gains are at the base (+0.10) and in the lower half towards the apex (+0.06 to +0.11 in bins 0.6–0.9), where slice-to-slice continuity helps most.
- At the extreme apex the gain is only +0.04, and both end bins remain near 0.6. This data implies that <ins>3D context **reduces** the end-slice failures but does **not eliminate** them.</ins>


### 5.3 Computational cost

| | 2D U-Net | 3D U-Net |
|---|---|---|
| Parameters | 1,942,374 | 5,646,470 (2.9×) |
| Peak VRAM, training (MB) | 284 | 1,297 (4.6×) |
| Peak VRAM, inference (MB) | 284 | 358 |
| Mean epoch time (s) | 8.9 | 5.0 |
| Total training time | ~4.5 min (30 epochs) | ~5 min (60 epochs) |
| Inference time per volume (ms) | 4.9 | 8.8 |


Inference time and VRAM come from the final test pass in `train.py`, where the model has already run many validation passes. A second timing in `predict.py` evaluated the 2D model first without a warm-up, so it included one-off CUDA initialisation (26.3 ms) and was discluded. Training VRAM is for a batch of 16 slices (2D) versus 2 whole volumes (3D) on an NVIDIA A100 (40 GB) (on rangpur)

At this downsampled resolution, the 3D model's extra cost is small in absolute terms: about 1.3 GB to train and under 10 ms per volume. Memory grows roughly with the voxel count, though, so 3D training at native resolution would cost several times more and might need patch-based training on smaller GPUs.

### 5.4 Failure case autopsy
The four cases below were selected automatically by `predict.py` as the test slices with the lowest 3D U-Net prostate Dice. Each figure shows the input, ground truth (orange = body, green = bone, red = bladder, purple = rectum, brown = prostate), and each model's prostate prediction (green = correct, red = false positive, blue = false negative).

**1. Missed apex: H007 Week 0, slice 24 (position 1.00)**
![failure 1](figures/failure_1.png)
This is the last slice of the prostate at the apex. Here the gland covers only a few voxels, just forward from the rectum. The 3D model predicts no prostate at all (Dice 0.00). The 2D model finds a fraction of it (Dice 0.16), but also places a larger false positive next to the rectum. At the apex the prostate tapers into the pelvic floor muscles, where contrast is low, and at 128×128 resolution it is only a few voxels. The 3D model's smooth inter-slice shape appears to end the gland one slice too early risking leaving tumour outside the treated volume due to under-contouring
 
**2. Prostate predicted above the base: K042 Week 0, slice 42 (no prostate in ground truth)**
![failure 2](figures/failure_2.png)
Both models predict a small prostate region at the edge of the bladder, between the bladder and the rectum, on a slice the annotator labelled as having no prostate. This is an example of over-contouring, extending the target into the bladder wall raises the risk of radiation cystitis.

**3. Missed apex: M015 Week 0, slice 28 (position 1.00)**
![failure 3](figures/failure_3.png)
This is similar to case 1, but here **both** models miss the prostate apex entirely (Dice 0.00). The labelled prostate is only a handful of voxels, and the rectum beside it is also very small at this level. Neither 2D appearance nor 3D context recovers it, so this case falls outside what either model can reliably do.
 
**4. 3D-specific overshoot above the base: M015 Week 1, slice 40 (no prostate in ground truth)**
![failure 4](figures/failure_4.png)
The 2D model correctly predicts no prostate here; its Dice is undefined (shown as "nan") because both the prediction and the ground truth are empty. The 3D model places small false-positive specks at the bladder–rectum interface. This shows the cost of 3D context: inter-slice continuity helps on hard slices inside the gland, but it can also carry the prostate shape one slice past where the gland ends. This is consistent with the 3D model's lower bladder Dice (0.755 vs 0.826), suggesting confusion at the bladder–prostate boundary.
 
**Summary of failures:** all four worst slices sit at the ends of the prostate (two at the apex, two just beyond the base), and none are in the mid-gland. Every one has Dice 0.00, because a structure only a few voxels across scores zero if it is missed or wrongly predicted. This matches the base-to-apex curve, where both end bins stay near 0.6 Dice for both models.

## 6. Recommendations for the Project Manager

**Adopt the 3D U-Net for prostate contour drafting with manual review of the first and last ~10% of slices of the gland (base and apex).**
 
- **Accuracy:** the 3D model improves prostate Dice from 0.792 to 0.847 and the worst patient from 0.702 to 0.751. It is better in every base-to-apex position bin, by up to about +0.1 at the base and in the lower half of the gland.
- **Cost:** at this resolution the extra cost is small (1.3 GB vs 0.3 GB of training VRAM, 8.8 vs 4.9 ms per volume, about 5 minutes to train). This is not the 5–10× burden the dilemma warns of, so the gain is worth it here.
- **What 3D context does not fix:** at both ends of the gland, Dice remains around 0.6 for both models. The worst failures are missed apex slices (under-contouring, recurrence risk) and predictions extending past the base into the bladder neck (over-contouring, cystitis risk). 3D context seems to be capable of causing the latter (failure case 4).
- **Other structures:** the 3D bladder contour is worse than the 2D one (0.755 vs 0.826) and should not be used for bladder dose constraints without review. A practical option is to use the 3D model for prostate and the 2D model for bladder.
- **Not suitable for automated drafting:** apex slices where the gland is only a few voxels across, and the bladder-neck slices just above the base. (evidenced by failure case 3)

**Limitations:** 
- prototype used a single training seed
- only 5 test patients, some contributing several weekly scans, so per-volume results are correlated
- evaluation at downsampled resolution (128×128×64), not native resolution
- base/apex orientation inferred from the bladder position
- no boundary-distance metric (HD95). 
Before clinical use, the model should be validated on more patients at native resolution.

## 7. Usage

**Files**
 
| File | Purpose |
|---|---|
| `modules.py` | U-Net (2D/3D in one class), Dice + cross-entropy loss, `predict_volume` |
| `dataset.py` | Nifti loading, resampling, patient-level split, leakage check, augmentation |
| `train.py` | Training, validation, checkpointing, training curves, test metrics and resource profiling |
| `predict.py` | Loads both checkpoints; comparison table, base-to-apex plot, failure overlays |
| `job.sh` | Slurm job script for Rangpur |
 
**On Rangpur** (the data path is the default `--root`; submit from this folder):
 
```bash
sbatch job.sh 2                                          # train the 2D U-Net (30 epochs)
sbatch job.sh 3                                          # train the 3D U-Net (60 epochs)
sbatch job.sh predict                                    # compare both on the test set
sbatch job.sh 3 --epochs 2 --early-stop --out smoke      # quick smoke test on 4 volumes
```
 
`job.sh` activates the `torch` conda environment and passes any extra arguments to the Python script. Only one job per user runs at a time on the `comp3710` partition, so predict can be queued to start after both training jobs: `sbatch --dependency=afterok:<2D job id>:<3D job id> job.sh predict`.
 
**Elsewhere:**
 
```bash
python train.py --root /path/to/HipMRI_Study_open --dims 2 --epochs 30
python train.py --root /path/to/HipMRI_Study_open --dims 3 --epochs 60
python predict.py --root /path/to/HipMRI_Study_open
```
 
**Outputs** (written to `outputs/`, which is git-ignored): `unet{2,3}d_best.pt` checkpoints, `unet{2,3}d_results.json` with all metrics and history, `unet{2,3}d_curves.png`, and from `predict.py`: `comparison.md`, `slice_position.png` and `failure_*.png`. The figures in this README were copied to `figures/`.

## 8. Dependencies and Reproducibility

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| nibabel | 5.4.2 |
| NumPy | 2.4.6 |
| Matplotlib | 3.11.2 |
| tqdm | 4.70.1 |
 
Hardware: NVIDIA A100-PCIE-40GB (Rangpur `comp3710` partition).
 
**Reproducibility:** the patient split uses a fixed seed of 42 and training uses seed 0, so the split is identical on every run.

## 9. Artificial Intelligence Disclosure

- **Tools used:** Anthropic Claude (chat).
- **What it assisted with:** 
    - scaffolding dataset loader
    - patient-level split
    - the command-line options in `train.py`
    - the evaluation and figure code in `predict.py`
    - Slurm/Rangpur job scripts and debugging (conda environment activation, job dependencies); and drafting parts of this README from my own experiment results.
    - Structuring JSON outputs and consolidating/comparing the results of both models to make it understandable
    - github markdown syntax for tables specifically 
- **My verification methods** 
    - smoke-tested every script on a small subset before the full runs
    - checked tensor shapes and that all six labels load
    - confirmed `check_no_leakage` reports no patient overlap (27/6/5 patients)
    - inspected training curves and per-slice results myself
    - checked the failure overlays against the raw images and ground truth
    - cross-checked timing numbers between `train.py` and `predict.py`, which exposed and corrected a warm-up bias in the latter.