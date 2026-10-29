# 3D Improved U-Net for Prostate Segmentation on HipMRI

**Author:** Harfateh Shergill (s50082083) · COMP3710 Pattern Recognition 2026 · Segmentation project, chosen model: **3D Improved U-Net (Hard)**

## Problem and algorithm

**Problem.** Before prostate radiotherapy, a planner contours the prostate (the treatment target) and the nearby organs at risk, the bladder and rectum, on MR images, slice by slice. Under-contouring risks leaving tumour untreated; over-contouring irradiates the rectal and bladder walls. This project segments six classes (background, body, bone, bladder, rectum, prostate) in 3D pelvic MR volumes to produce draft contours for a planner to review, and investigates the course's open question: does 3D context fix the clinically dangerous errors at the ends of the gland and at organ boundaries, or does it only improve easy central slices at extra cost?

**Algorithm.** The chosen model is a 3D Improved U-Net (Isensee et al., 2018). Like a U-Net, an encoder repeatedly halves the volume to learn *what* is present and a decoder enlarges it back to locate *where*, with skip connections carrying fine detail across. The Improved U-Net adds pre-activation residual *context modules* (instance normalisation, LeakyReLU, dropout) in the encoder, stride-2 convolutions for downsampling, *localisation modules* after each skip connection, and *deep supervision*: segmentation outputs from three decoder levels are upsampled and summed so the deeper layers get a direct training signal. All convolutions are 3x3x3, so every layer sees neighbouring slices. It is compared with two baselines working on single axial slices, a standard 2D U-Net (Ronneberger et al., 2015) and a 2D Improved U-Net, and with the same 3D model trained at full resolution.

Example outputs from every model are shown in the results and failure autopsy below (slice-by-slice Dice plots and a sagittal comparison), and `predict.py` saves a four-panel figure (input, ground truth, prediction, false-positive/false-negative overlay) for any test scan.

## Feasibility review

### 1. User need, scope and acceptance criteria

**User:** a radiotherapy planner who reviews and adjusts draft contours of the prostate and nearby organs at risk (bladder, rectum) on pelvic MR. The tool proposes a 3D draft contour to save contouring time. The planner always reviews it; it never goes to treatment unchecked.

**Scope:** automatic 6-class segmentation (background, body, bone, bladder, rectum, prostate) of the CSIRO HipMRI 3D volumes. The question is whether 3D context reduces the errors that matter clinically: base/apex slice failures and leakage into rectum or bladder. Out of scope: other organs, other scanners, clinical deployment.

**Acceptance criteria** (all on held-out test patients)

1. Mean prostate Dice ≥ 0.7, scored at the original 256×256×128 resolution.
2. Prostate Dice on base and apex slices reported separately for the 3D model and both 2D baselines, with the 3D–2D difference stated.
3. Prostate voxels predicted inside ground-truth rectum or bladder reported per test case, plus HD95 in mm.
4. 3D training fits a 16 GB GPU (Colab T4) at batch size ≥ 1; peak VRAM and per-volume inference time reported for all models.
5. Zero patient overlap between train, validation and test, enforced by assertions in `dataset.py`.

### 2. Model choice and course concepts

**Chosen model:** 3D Improved U-Net (Isensee et al., 2018), Hard difficulty, on downsampled HipMRI 3D volumes. **Baselines:** a 2D U-Net (Ronneberger et al., 2015) and a 2D Improved U-Net, both trained on full-resolution axial slices of the same training patients. Their predictions are stacked back into volumes and scored exactly like the 3D model.

**Why these models**

- **Skip connections** carry fine spatial detail from encoder to decoder, needed for millimetre-scale boundaries.
- **Residual context modules** with InstanceNorm and LeakyReLU make the deeper network stable to train with small batches.
- **Deep supervision** adds segmentation outputs at lower decoder levels, giving the tiny prostate class gradient signal earlier.
- **3D convolutions** see neighbouring slices. That should help where 2D fails: base and apex slices, where the prostate is small and its edges are ambiguous within a single slice.
- **Soft Dice + cross-entropy loss** counters the extreme class imbalance (prostate = 0.1% of voxels).

**The trade-off being tested.** Voxels are nearly the same size in every direction (about 1.68 × 1.68 × 1.56 mm), so the 2D models do not suffer from thick slices. The 3D model gains context but must work at half resolution to fit in memory; 2D keeps full resolution. The comparison asks which matters more at the clinically risky boundaries.

### 3. Preliminary feasibility evidence

A resolution test shows the task is feasible at ×2 downsampling: shrinking and re-enlarging the true labels alone limits prostate Dice to 0.918, well above the 0.7 target.

**Data audit** (CSIRO *Labelled weekly MR images of the male pelvis*)

| Item | Finding |
| --- | --- |
| Volumes | 211 image/label pairs, all paired, none unparsed |
| Patients | 38; scans per patient min 1, median 8, max 8 |
| Volume shape | 256×256×128 (210 volumes); one at 256×256×144, cropped to 128 slices |
| Voxel spacing | in-plane 1.41–1.88 mm (152 volumes at 1.68), slice 1.56 mm |
| Labels | 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate (confirmed visually) |
| Prostate | 0.1% of voxels; present in 100% of scans |
| Planned split | by patient, 26 / 6 / 6 (train / val / test) |

**Resolution ceiling.** Labels were downsampled with nearest-neighbour, upsampled back, and compared with the originals on 20 random scans. This is the best prostate Dice a perfect model could reach at each resolution:

| Downsample | Volume size | Mean Dice ceiling | Worst case |
| --- | --- | --- | --- |
| ×2 | 128×128×64 | 0.918 | 0.871 |
| ×4 | 64×64×32 | 0.790 | 0.708 |

×2 is chosen; ×4 would leave almost no margin above the 0.7 target.

### 4. Risks, budget and fallback

The main risk is compute: GPU time is limited to a 29 h EAIT GPU Lab quota, so development and smoke tests run on free Colab and only full training runs use the lab.

| Risk | Mitigation or fallback |
| --- | --- |
| Rangpur access delayed | Not blocking: full runs use the EAIT GPU Lab, development uses Colab; Rangpur becomes overflow once access arrives |
| 3D model doesn't fit in GPU memory | Batch size 1–2, mixed precision, 8–16 base filters, patch training with sliding-window inference |
| Prostate class ignored (0.1% of voxels) | Soft Dice + cross-entropy, prostate class weighting, sampling biased toward prostate regions |
| Patient leakage via repeat weekly scans | Split by patient ID only; assertions fail the run on any overlap |
| Small test set (6 patients) makes scores noisy | Report per-patient Dice with mean ± std; average each patient's scans before averaging across patients |
| 3D below Dice 0.7 by deadline | Submit the 2D Improved U-Net (Normal, capped at 15/20) as the chosen model, with the 2D U-Net as its baseline |
| Fallback needs the course 2D data (`keras_slices_data`), not yet obtained | Request it from the course now, before it is needed |

**Compute budget (estimates; replaced by measured times after each run):** EAIT Virtual GPU Lab, 29 h quota on L4 / A10G / L40S GPUs, used only for full training runs: 2D U-Net ≈ 1–2 h, 2D Improved U-Net ≈ 2 h, 3D Improved U-Net ≈ 10–15 h, leaving ≈ 10 h in reserve for reruns. Free Colab (T4) handles development, checks and smoke tests; Rangpur is overflow. Checkpoints are saved every epoch and results are backed up after each run, because lab instances may not keep files between sessions.

**Next experiment:** write `dataset.py` (patient-level split and loader), then train the 2D U-Net baseline on full-resolution axial slices of the 26 training patients, recording training loss, validation prostate Dice, peak VRAM and time per epoch to replace the estimates above.

## Dependencies and reproducibility

Exact versions used for the GPU Lab runs (also in `requirements.txt`):

| Package | Version |
| --- | --- |
| torch | 2.11.0+cu128 |
| nibabel | 5.4.2 |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| matplotlib | 3.10.9 |
| tqdm | 4.70.1 |

Hardware: Google Colab (NVIDIA T4) for the 2D U-Net and epochs 1–118 of the x2 3D model, with Colab's preinstalled PyTorch; the EAIT Virtual GPU Lab (NVIDIA L4) for everything else, in a virtual environment with the versions above (the lab's shared environment failed to load cuDNN for any convolution). All runs use seed 42; the patient split is regenerated identically from that seed and saved as `split.json` in each run folder. GPU training is not bit-for-bit deterministic, so re-runs can differ slightly in the last decimal places.

**Compute issues encountered.** Compute limits affected two runs: Colab's free GPU quota stopped the x2 3D run at epoch 118, which was resumed from its saved checkpoint on the GPU Lab (`--resume`), and the full-resolution run was limited to 80 epochs to fit the remaining GPU time. On the GPU Lab, the shared environment's PyTorch could not load a cuDNN library and every convolution failed; installing PyTorch 2.11.0+cu128 in a private virtual environment fixed this. Browser uploads to the lab sometimes produced incomplete or duplicated files, so every transferred checkpoint and archive was verified (file size in bytes, archive integrity test, and loading each checkpoint to confirm its stored model name and epoch) before use. A GPU Lab outage on the final day prevented downloading two evaluation figures; `regional_dice.png` and `per_patient_prostate.png` were redrawn with the same plotting code from the saved evaluation results, and two prediction figures were omitted.

## Data, preprocessing and splits

**Data.** CSIRO *Labelled weekly MR images of the male pelvis* (HipMRI): 211 MR volumes from 38 patients, each with a 6-class label volume (0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate). Most patients were scanned weekly, up to 8 times. The data is not included in this repository (licence CC BY-NC-ND 4.0).

**Preprocessing** (`dataset.py`, run once and cached as `.npy`):

1. Each image is paired with its label by patient and week, parsed from the file names (211 pairs, none unmatched).
2. The slice axis is fixed at 128 slices. One volume (`019_w1`, 144 slices) is centre-cropped by 8 slices at each end; it belongs to a training patient, so validation and test data are untouched.
3. Each image is z-score normalised per volume (mean 0, standard deviation 1) so intensities are comparable across scans.
4. Volumes are stored as (slice, height, width) so each axial slice can be read directly from disk.

**Splits.** Patients, not scans, are split 26 / 6 / 6 into train / validation / test with a fixed seed (42). Repeat weekly scans of one patient are near-identical, so a scan-level split would leak test anatomy into training. Assertions in `split_patients` stop the run if any patient appears in two splits.

| Split | Patients | Scans | 2D slices |
| --- | --- | --- | --- |
| Train | 26 | 150 | 19,200 |
| Validation | 6 | 27 | 3,456 |
| Test | 6 | 34 | 4,352 |

Only 20.6% of training slices contain prostate, so 2D training samples slices with weights that make about half of each batch prostate-containing. Training-only augmentation: random left-right flip (applied identically to image and label) and mild intensity scaling and shifting.

**3D input.** The chosen 3D model receives whole volumes downsampled x2 in every direction (128x256x256 to 64x128x128; trilinear for images, nearest-neighbour for labels). Its predictions are upsampled back and scored against the original full-resolution labels, exactly like the 2D models. Section 3 of the feasibility review shows x2 downsampling alone limits prostate Dice to about 0.92. A second 3D model with the identical architecture was trained on full-resolution volumes (`--scale 1`) to separate the effect of 3D context from the effect of resolution.

## Usage

```bash
pip install -r requirements.txt   # torch: pip install torch --index-url https://download.pytorch.org/whl/cu128

# Train. Each run saves best.pt, last.pt, history.json, curves.png, split.json and test_metrics.json;
# add --resume to continue an interrupted run from last.pt
python train.py --data-root DATA --cache-dir CACHE --out-dir RUNS/unet2d       --model unet2d  --epochs 30
python train.py --data-root DATA --cache-dir CACHE --out-dir RUNS/iunet2d      --model iunet2d --epochs 15  --lr 5e-4
python train.py --data-root DATA --cache-dir CACHE --out-dir RUNS/iunet3d      --model iunet3d --epochs 150 --batch-size 2 --lr 5e-4
python train.py --data-root DATA --cache-dir CACHE --out-dir RUNS/iunet3d_full --model iunet3d --epochs 80  --batch-size 2 --lr 5e-4 --scale 1

# Predict: per-class Dice and a 4-panel figure for chosen test scans
python predict.py --run-dir RUNS/iunet3d --cache-dir CACHE --case 039_w2

# Evaluate all models: choose check thresholds on validation, then report on test
M="unet2d iunet2d iunet3d iunet3d_full"
python evaluate.py --runs-dir RUNS --models $M --cache-dir CACHE --out-dir RUNS/eval_val --split val --calibrate
python evaluate.py --runs-dir RUNS --models $M --cache-dir CACHE --out-dir RUNS/evaluation --thresholds RUNS/eval_val/thresholds.json
```

`DATA` is the folder containing the HipMRI `.nii.gz` files (searched recursively); `CACHE` is created on the first run. `predict.py` prints per-class Dice for each scan in this format (illustrative values):

```
Model iunet3d (best epoch 97)
case       background       body       bone    bladder     rectum   prostate
039_w2          0.994      0.981      0.901      0.909      0.895      0.913
```

| File | Contents |
| --- | --- |
| `dataset.py` | file pairing, patient-level split, preprocessing cache, 2D slice and 3D volume datasets, augmentation |
| `modules.py` | U-Net, Improved U-Net (2D/3D), soft Dice + cross-entropy loss, model registry (PyTorch only) |
| `train.py` | training, validation, checkpointing, resuming, test evaluation, curves |
| `predict.py` | inference on chosen test scans with comparison figures |
| `utils.py` | volume prediction (2D slice-by-slice or 3D), Dice/IoU, per-patient summaries |
| `evaluate.py` | regional Dice, extent error, leakage, HD95, anatomy checks, resources, figures |

## Results

All four models are evaluated at full resolution on the 6 validation and 6 test patients. Metrics are averaged over each patient's scans first, then mean ± std across patients. The validation set was used to choose each model's best epoch, so validation scores are slightly optimistic for every model alike.

### Models trained

| Model | Input | Parameters | Epochs (best) | Optimiser | GPU |
| --- | --- | --- | --- | --- | --- |
| 2D U-Net (baseline) | 256x256 axial slices, batch 16 | 7,762,630 | 30 (14) | Adam 1e-3, cosine decay | T4 |
| 2D Improved U-Net (baseline) | 256x256 axial slices, batch 16 | 2,776,834 | 15 (6) | Adam 5e-4, cosine decay | L4 |
| **3D Improved U-Net, x2 (chosen)** | 64x128x128 volumes, batch 2 | 8,279,074 | 150 (97) | Adam 5e-4, cosine decay | T4, then L4 |
| 3D Improved U-Net, full resolution | 128x256x256 volumes, batch 2 | 8,279,074 | 80 (69) | Adam 5e-4, cosine decay | L4 |

All use soft Dice + cross-entropy loss and mixed precision. The full-resolution model was given 80 rather than 150 epochs to fit the available GPU time; its validation Dice had plateaued (0.808 over the final epochs; best checkpoint at epoch 69).

![2D U-Net training curves](images/curves.png)

### Overall prostate Dice

| Model | Validation | Test |
| --- | --- | --- |
| 2D U-Net | **0.823 ± 0.079** | 0.730 ± 0.113 |
| 2D Improved U-Net | 0.809 ± 0.100 | 0.745 ± 0.111 |
| 3D Improved U-Net (x2) | 0.780 ± 0.102 | **0.793 ± 0.083** |
| 3D Improved U-Net (full res) | 0.812 ± 0.092 | 0.751 ± 0.120 |

The chosen model meets the project target (test prostate Dice ≥ 0.7). The two sets rank the models differently, and the per-patient results show why.

![Prostate Dice per test patient](images/per_patient_prostate.png)

**Per patient** (prostate Dice, mean over each patient's scans):

| Patient | Set | 2D U-Net | 2D Improved | 3D x2 | 3D full res | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 008 | val | 0.82 | 0.84 | 0.78 | 0.85 | typical |
| 009 | val | 0.92 | 0.90 | 0.90 | 0.91 | typical |
| 016 | val | 0.87 | 0.86 | 0.82 | 0.84 | typical |
| 031 | val | 0.83 | 0.84 | 0.82 | 0.84 | typical |
| 035 | val | 0.84 | 0.82 | 0.79 | 0.82 | typical |
| 014 | test | 0.77 | 0.79 | 0.82 | 0.80 | typical |
| 033 | test | 0.87 | 0.87 | 0.88 | 0.89 | typical |
| 039 | test | 0.84 | 0.87 | 0.88 | 0.89 | typical |
| 029 | val | 0.66 | 0.59 | 0.57 | 0.62 | base fails for all models |
| 037 | test | 0.73 | 0.72 | 0.70 | 0.60 | base fails for all models |
| 038 | test | 0.59 | 0.61 | 0.67 | 0.61 | base fails for all models |
| 015 | test | 0.58 | 0.60 | **0.81** | 0.71 | unusual contrast, large gland |

For the 8 typical patients every model scores 0.77–0.92 and the models are usually within 0.03–0.05 of each other. The differences between the averages come from four hard patients: three whose base fails for every model, and patient 015, whom the x2 3D model rescues.

### Where along the gland: the core dilemma result

The true prostate's slice extent is split into thirds; the base is the end next to the bladder.

| Model | Apex (val) | Base (val) | Apex (test) | Mid (test) | Base (test) | HD95 test (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 2D U-Net | **0.817** | 0.759 | 0.800 | 0.833 | 0.524 ± 0.320 | 8.5 ± 2.7 |
| 2D Improved U-Net | 0.801 | 0.742 | 0.818 | 0.844 | 0.545 ± 0.307 | 11.9 ± 10.6 |
| 3D Improved U-Net (x2) | 0.732 | 0.758 | 0.835 | **0.866** | **0.627 ± 0.273** | **7.4 ± 4.1** |
| 3D Improved U-Net (full res) | 0.811 | 0.756 | **0.845** | 0.834 | 0.552 ± 0.344 | 14.6 ± 14.8 |

![Prostate Dice by region of the gland (test)](images/regional_dice.png)

- **The base is the weakest region and no model fixes it.** On the hard patients the base collapses for all four models (test base Dice 0.01–0.14 for patient 038 and 0.17–0.40 for 037; validation 0.39–0.46 for 029).
- **x2 downsampling costs apex detail; full resolution recovers it.** On validation the x2 model's apex is 0.732 against 0.811 at full resolution and 0.817 for the 2D U-Net (patient 008: 0.50 against 0.67 and 0.66). On test, the full-resolution model has the best apex of all (0.845).
- **Context at x2 helps the unusual patient.** Downsampling doubles the anatomy each layer sees, so the x2 model has about twice the physical field of view of the full-resolution model with the same depth. It recovers patient 015 best (0.81 against 0.71 at full resolution and 0.58–0.60 in 2D). This rests on one patient, so it is a hypothesis rather than a result.

| Model (test) | Apex extent (mm) | Base extent (mm) | Leak into bladder (mL) | Leak into rectum (mL) |
| --- | --- | --- | --- | --- |
| 2D U-Net | +4.7 ± 4.7 | 0.0 ± 5.1 | 0.63 ± 0.95 | 0.02 ± 0.03 |
| 2D Improved U-Net | +4.5 ± 3.7 | +5.7 ± 11.3 | 0.90 ± 1.34 | 0.02 ± 0.05 |
| 3D Improved U-Net (x2) | **+1.2 ± 2.7** | +1.0 ± 5.1 | **0.37 ± 0.27** | 0.02 ± 0.05 |
| 3D Improved U-Net (full res) | +5.2 ± 5.9 | +8.3 ± 18.6 | 1.73 ± 2.93 | 0.01 ± 0.01 |

Extent error is how far the predicted gland ends beyond (+) or short of (−) the true apex or base. Leakage into the rectum is negligible for every model.

### Other organs (Dice)

| Model | Bladder (val) | Bladder (test) | Rectum (test) | Bone (test) | Body (test) |
| --- | --- | --- | --- | --- | --- |
| 2D U-Net | **0.944** | **0.851 ± 0.167** | **0.883** | **0.913** | 0.972 |
| 2D Improved U-Net | 0.893 | 0.746 ± 0.243 | 0.865 | 0.910 | 0.971 |
| 3D Improved U-Net (x2) | 0.920 | 0.686 ± 0.356 | 0.868 | 0.883 | 0.962 |
| 3D Improved U-Net (full res) | 0.943 | 0.733 ± 0.319 | 0.852 | 0.911 | **0.973** |

The test bladder spread comes from patient 015, whose bladder only the 2D U-Net segments (bladder Dice 0.84, against 0.34, 0.01 and 0.11 for the other models).

### Resources

All inference measured on one NVIDIA L4; training per epoch on the GPU shown.

| Model | Inference per scan | Peak inference VRAM | Training per epoch | Peak training VRAM |
| --- | --- | --- | --- | --- |
| 2D U-Net | 0.20 s | 1.25 GB | 213 s (T4) | 1.73 GB |
| 2D Improved U-Net | 0.15 s | 0.94 GB | 85 s (L4) | 1.31 GB |
| 3D Improved U-Net (x2) | **0.07 s** | **0.48 GB** | 26 s (T4) / 12 s (L4) | 2.11 GB |
| 3D Improved U-Net (full res) | 0.16 s | 3.26 GB | 76 s (L4) | 15.80 GB |

At x2 the 3D model is the cheapest to run; at full resolution it needs 7.5x the training memory and 6.3x the training time per epoch of the x2 model on the same GPU, the "5–10x" cost the dilemma anticipates.

### Anatomy checks (no ground truth needed)

Three checks that need no ground truth could flag bad drafts in deployment: **A**, background enclosed inside the body on an axial slice (an impossible hole); **B**, the bladder missing or more than 10 mm from the prostate; **C**, the prostate in several pieces. The hole threshold for each model was set on the validation patients only (90th percentile of holes in that model's acceptable validation scans) and then applied once to test. A scan is *unacceptable* if prostate Dice < 0.7 or bladder Dice < 0.5.

| Model (test) | Hole threshold | Scans flagged (A / B / C) | Unacceptable caught | Acceptable but flagged |
| --- | --- | --- | --- | --- |
| 2D U-Net | 19.7 mL | 13 (13 / 0 / 2) of 34 | 6 of 9 | 7 of 25 |
| 2D Improved U-Net | 30.9 mL | 13 (3 / 0 / 11) of 34 | 8 of 9 | 5 of 25 |
| 3D Improved U-Net (x2) | 1.2 mL | 17 (17 / 4 / 2) of 34 | 8 of 10 | 9 of 24 |
| 3D Improved U-Net (full res) | 1.0 mL | 14 (14 / 1 / 5) of 34 | 8 of 10 | 6 of 24 |

On the ground-truth labels the checks flag 0 of 34 test scans. The 2D models leave large impossible holes even in good predictions (median 12–14 mL in acceptable validation scans), so the hole check needs a high threshold for them; 3D predictions are almost hole-free. The checks catch 67–89% of unacceptable scans, but a gland that simply stops short of the base is still anatomically plausible, so base under-contouring passes every check.

## Failure case autopsy

The slice-by-slice plots below compare the two 2D baselines with the x2 3D model; the full-resolution model is described from its metrics.

**1. The base of the gland stops short (patients 038, 037, 029; all models).** In `038_w0` all four models agree in the middle of the gland and all miss the last five base slices (about 7.8 mm). The 2D U-Net predicted those prostate voxels as body tissue (84%) and bladder (13%): the base meets the bladder neck, with similar intensity on these images. Neither 3D context nor full resolution helps. Clinically this is under-contouring of the target, a recurrence risk.

![Slice-wise prostate Dice, 038_w0](images/slice_dice_038_w0.png)

**2. Unusual contrast breaks each model differently (patient 015).** Patient 015's gland is large (28 slices against about 18 for others) and dark on MR. The 2D U-Net labels much of it as background, an impossible hole inside the pelvis (prostate Dice 0.35 on `015_w6`; 73% of the true base predicted as background), and the 2D Improved U-Net fails similarly (0.37 on `015_w6`). The x2 3D model follows the gland across slices and recovers it (0.73 on `015_w6`, 0.81 for the patient), but misplaces the bladder on every scan: its predicted bladder is 98% body tissue, centred about 105 mm anterior and 23 mm superior to the true bladder, in the abdominal wall. Only the 2D U-Net keeps this patient's bladder (0.84).

![Slice-wise prostate Dice, 015_w6](images/slice_dice_015_w6.png)
![Sagittal view of 015_w0: ground truth, 2D U-Net and 3D x2](images/sagittal_015_w0.png)

**3. Thin apex tips are lost at half resolution (`014_w1`, patient 008).** On the first apex slices of `014_w1` the full-resolution 2D U-Net scores 0.73–0.94 while the x2 3D model misses two slices and reaches 0.47 on the third; on validation patient 008 its apex Dice is 0.50. The same 3D model trained at full resolution recovers the apex (0.67 for patient 008; 0.811 on validation overall), confirming that downsampling, not 3D context, causes this failure.

![Slice-wise prostate Dice, 014_w1](images/slice_dice_014_w1.png)

**4. Impossible holes and fragments in 2D predictions.** Slice-by-slice prediction leaves background inside the body in nearly every 2D scan (median 12–14 mL even when the prostate Dice is acceptable) and the 2D Improved U-Net splits the prostate into several pieces on 11 test scans. The ground truth never contains either. 3D predictions are almost free of both.

## Recommendation to the project manager

1. **For typical anatomy the model choice matters little.** All four models reach 0.8–0.9 prostate Dice on 8 of the 12 evaluation patients.
2. **Always review the base of the gland.** No model, dimensionality or resolution fixes the patients whose base fails, and the anatomy checks cannot detect it.
3. **Draft the prostate with the 3D Improved U-Net at x2.** It is the most robust on the unusual patient, has the tightest test boundary (HD95 7.4 mm) and least bladder leakage, and is the cheapest to run (0.07 s and 0.5 GB per scan). Check the first apex slices, where x2 can lose a thin tip.
4. **Take the bladder from the 2D U-Net.** It is the most reliable bladder on both sets (0.944 validation, 0.851 test) and the only model that handled patient 015.
5. **Full-resolution 3D is not worth its cost here.** It recovers the apex but needs about 7x the training memory and time, and was less robust on the hard test patients.
6. **Run the anatomy checks on every draft as triage.** Send flagged scans for full manual contouring; they catch most gross failures (67–89%) but not plausible base under-contouring.
7. **More patients would help more than more architecture.** With 6 patients per set, single patients decide the ranking (validation and test disagree), and differences of 0.03–0.05 between models are within patient-to-patient variation.

**Limitations.** 12 evaluation patients; different training lengths and GPUs across runs (80 epochs for the full-resolution model); one random seed per model; the receptive-field explanation for patient 015 is supported by one patient only.

## Artificial Intelligence Usage Disclosure

**Tool used:** Claude (Anthropic; model Claude Opus 5.5), used through a chat interface throughout the project.

**What AI assisted with.** AI assistance was substantial, and I want to be clear about its extent:

- **Planning:** proposing the project pathway (2D U-Net baseline, 2D Improved U-Net, 3D Improved U-Net), the patient-level split and the commit plan.
- **Data exploration:** writing the Colab notebooks for the data audit and the x2/x4 downsampling resolution test.
- **Feasibility review:** drafting the one-page review from the numbers my data audit produced.
- **Code:** generating the code in `dataset.py`, `modules.py`, `utils.py`, `train.py`, `predict.py` and `evaluate.py`, and the Colab notebooks with a check for each commit. I committed the code in steps, each after running its check.
- **Debugging:** diagnosing Colab path errors, corrupted file uploads to the GPU Lab, and the GPU Lab's shared-environment cuDNN failure (fixed by installing PyTorch in my own virtual environment).
- **Analysis and writing:** suggesting the diagnostics (per-patient tables, slice-by-slice Dice, the sagittal comparison, what the missed prostate was predicted as), helping interpret the results, and drafting the README text, which I reviewed and edited.
- **Figures:** the final `regional_dice.png` and `per_patient_prostate.png` were redrawn by Claude with the same plotting code from my evaluation results, because the GPU Lab was unavailable when I needed to download the original files.
- **Learning:** explanations of Dice, soft versus hard Dice, U-Nets, 2D versus 3D convolutions, epochs, and the Improved U-Net's components.

**How I verified the AI-generated work.**

- **Every code commit was checked on the real data before the next step:** 211 image/label pairs and 38 patients found; a 26/6/6 patient split with assertions confirming no patient overlap; preprocessed volumes of shape (128, 256, 256) with mean 0 and standard deviation 1; label overlays inspected to confirm the prostate outline sits between bladder and rectum; model output shapes and parameter counts; the loss near 0 for a perfect prediction and Dice 0.75 for a hand-worked 4-square example.
- **Training was monitored** through loss and validation curves for every run, and a re-scored test scan reproduced the score saved during training exactly.
- **Evaluation metrics were unit-tested** on synthetic cases with known answers (extent error and its sign, leakage volume, HD95, holes, bladder distance, number of prostate pieces) before being used on real results.
- **Errors were caught by checks rather than trusted:** a checkpoint uploaded under the wrong model name was detected because its stored configuration said `iunet3d`, not `unet2d`; corrupted uploads were detected by comparing file sizes in bytes and testing archive integrity; the cuDNN failure was isolated with a small convolution test before any training.
- **No test-set peeking:** anatomy-check thresholds were chosen on validation patients only, and each model's test set was evaluated once with its best validation checkpoint.
- **Figures were inspected** for anatomical plausibility, and when results contradicted an earlier hypothesis (the 3D bladder failure, the validation and test rankings disagreeing), the README was corrected rather than the evidence selected to fit.

I ran all experiments, made the commits, and am responsible for every line of code and every claim in this report.

## References

1. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, "Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," 2018.
3. CSIRO, "Labelled weekly MR images of the male pelvis," doi:10.25919/45t8-p065.
