# ConvNeXt for Alzheimer's Disease Classification on ADNI Brain MRI

**Task 2.1 (Classification, Hard difficulty).** Classify T1-weighted ADNI brain MRI slices as Alzheimer's disease (**AD**) or cognitively normal (**NC**) with a ConvNeXt, reach at least 0.80 test accuracy, and investigate the *confidence and ambiguity dilemma*: does the model know when it is wrong, and can a reject rule safely route uncertain scans to a human?

<!--RESULTS_HEADLINE-->

---

## Contents
1. [Problem and algorithm](#1-problem-and-algorithm)
2. [Feasibility review](#2-feasibility-review)
3. [Data, pre-processing and splits](#3-data-pre-processing-and-splits)
4. [Training](#4-training)
5. [Results](#5-results)
6. [Open dilemma: confidence, calibration and referral](#6-open-dilemma-confidence-calibration-and-referral)
7. [Failure case autopsy](#7-failure-case-autopsy)
8. [Recommendation to the project manager](#8-recommendation-to-the-project-manager)
9. [Usage and reproducibility](#9-usage-and-reproducibility)
10. [References](#10-references)

---

## 1. Problem and algorithm

**Problem.** Alzheimer's disease causes progressive, irreversible neurodegeneration: the hippocampus and cortex atrophy and the ventricles enlarge. Structural MRI shows these changes, but they overlap with normal ageing and with natural differences in head shape. An automated AD vs NC classifier could pre-screen scans for a radiologist. In screening, though, a confident wrong answer is the dangerous failure. A missed AD case (a false negative) that the system reports as "certain" never reaches a human, so the classifier also needs to say when it is unsure.

**Algorithm.** ConvNeXt [1] is a pure convolutional network modernised step by step from ResNet-50 towards the design of Vision Transformers:

* a **patchify stem** (non-overlapping 4×4 convolution with stride 4) instead of a 7×7 convolution and max-pool;
* a **4-stage hierarchy** (3-3-9-3 blocks for ConvNeXt-T) with separate LayerNorm + 2×2 stride-2 down-sampling layers between stages;
* **inverted-bottleneck blocks**: a large-kernel **7×7 depthwise convolution** for spatial mixing, followed by a LayerNorm and an MLP of two 1×1 convolutions (C → 4C → C) with GELU for channel mixing. This mirrors a Transformer block, where the depthwise convolution plays the role of self-attention;
* fewer activations and normalisation layers, **LayerNorm** instead of BatchNorm, **layer scale** (a learnable per-channel scale initialised to 1e-6) and **stochastic depth** for regularisation.

The model is implemented from scratch in [`modules.py`](modules.py). It is initialised from torchvision's ImageNet-1k ConvNeXt-T weights, which are remapped onto this implementation; the mapping reproduces torchvision's features exactly (max abs. difference 0.0, checked by `python modules.py`). The RGB stem filters are summed into one channel so the network takes single-channel MRI slices directly. A global-average-pooled 768-d feature goes through LayerNorm and a 2-way linear head.

![ConvNeXt architecture](results/architecture.png)

**Baseline.** To find out what ConvNeXt's architecture and pretraining actually buy, it is compared with a plain **VGG-style CNN** (`BaselineCNN` in `modules.py`): four stages of two 3×3 conv-BN-ReLU layers with max-pooling (32-64-128-256 channels), global average pooling, dropout and a linear head. It has 1.2M parameters and is trained from scratch. Both models use the same data, splits, augmentation and evaluation code.

---

## 2. Feasibility review

*The one-page review presented at the midpoint check-off, reproduced as submitted. The changes made after it are listed at the end of this section.*

### 2.1 User need, scope and acceptance criteria

This prototype will classify ADNI structural brain MRI scans as Alzheimer's disease (AD) or cognitively normal (CN), supporting specialist review. The investigation is whether ConvNeXt produces more reliable confidence estimates than a simple CNN, or remains overconfident on incorrect predictions. It is a research prototype, not a diagnostic system.

Acceptance criteria:
1. target ≥0.80 patient-level test accuracy;
2. report per-class precision/recall, macro-F1 and AUROC for both models;
3. assess calibration and report error rate versus retained-case coverage under a validation-selected rejection threshold;
4. analyse five representative failures; and
5. record parameters, peak VRAM and per-patient inference latency, targeting <16 GB peak VRAM.

### 2.2 Model choice, data and experimental design

I propose a 2D ConvNeXt-Tiny architecture and a small CNN baseline using the course ADNI data at `/home/groups/comp3710/ADNI`. ConvNeXt applies convolution, residual learning, normalisation and hierarchical feature extraction. Both models will use identical patient splits, preprocessing and evaluation. Implementation and any pretrained-weight use will follow course requirements and be documented.

The supplied metadata contains three numeric labels without a diagnostic codebook, 0 being CN, 1 MDI, and 2 being AD. Subject IDs can be extracted consistently from the scan paths. I will initially select one eligible scan per subject using a fixed rule: after confirming the chosen image variant and applying predefined image-quality checks, retain the scan with the lowest numeric metadata image ID. This is a reproducible tie-breaker, not a claim that the scan is the earliest visit. All variants and slices from a subject will stay in one split, with patient separation established before slice extraction. I will average selected-slice probabilities within that single scan to produce one patient prediction.

Training-only augmentation and consistently defined anatomical slice selection will limit shortcuts. Validation data will select hyperparameters, temperature scaling and the reject threshold; the test set will remain untouched until final evaluation. Reliability diagrams and Brier scores will supplement classification metrics.

### 2.3 Preliminary feasibility evidence and next experiment

An initial audit of `meta_data_with_label.json` found 2,189 scan records covering 942 subjects; 567 subjects have multiple records. Codes 0, 1 and 2 have 810, 625 and 754 records respectively (459, 262 and 221 subjects). All records contain raw, masked and c1-c5 image paths and a label; subject IDs agree across the paths, and no subject has conflicting codes in this file. However, the diagnostic code mapping, image availability, spatial dimensions and quality remain unverified. Counts describe the complete metadata, not the final AD/CN cohort.

On 19 October 2026, a local torchvision ConvNeXt-Tiny benchmark using synthetic 224 × 224 three-channel inputs, batch size 8, FP16 mixed precision and AdamW completed 10 warm-up and 30 measured steps on an RTX 5070 Laptop GPU. Peak allocated/reserved VRAM was 0.949/1.015 GB, and mean measured step time was 0.0874 seconds, with all 30 measured optimiser updates completing. This supports local memory feasibility for the tested configuration; it excludes data loading and does not establish ADNI accuracy or Rangpur performance. The <16 GB target is a planning constraint, not a repository-specified minimum.

### 2.4 Risks, budget, fallback and recommendation

Risks include patient leakage, small cohorts, class imbalance, scanner-related shortcuts and misleading confidence. I will inspect five false-positive/false-negative or confidently wrong cases and document uncertainty in the results. A provisional 12 GPU-hour budget on GPU Lab's A10G or Rangpur will cover pilots and model comparisons, with checkpoints and time reserved for evaluation. This is a budget, not a runtime estimate.

If GPU memory is insufficient, I will reduce batch size and use gradient accumulation to retain a larger effective batch size. If training time exceeds the budget, I will reduce the shared slice count for both models while retaining patient separation. With fixed-size batches, reducing the total slice count mainly reduces training time rather than peak GPU memory. The final recommendation will weigh accuracy, calibration, coverage and compute cost, specifying which cases require specialist review; model confidence alone will not establish clinical safety.

*Reference:* [1] Liu et al. (2022), *A ConvNet for the 2020s*, CVPR. https://github.com/facebookresearch/ConvNeXt

### 2.5 Changes since the feasibility review

| Planned | Implemented | Reason |
|---|---|---|
| One scan per patient (lowest image ID) | **All scans** of each patient, kept in the same split, with patient- and class-balanced sampling | The provided AD_NC slice folders contain 1,526 scans from 680 patients. Using all of them gives ~2.2× more training data while keeping the patient-separation guarantee; balanced sampling stops patients with many visits from dominating (see §3). |
| Patient split built from the metadata | Same, but the official `train/` / `test/` folders were **pooled and re-split** by patient | The audit found that the official folders share 216 patients (§3.1). |
| Average slices of the single selected scan | Slice-, scan- and patient-level metrics are all reported; patient level averages every slice of every scan of that patient | Matches acceptance criterion 1 while keeping finer-grained results. |
| Slice selection | The 20 slices already supplied for each scan are used as given | The pre-processed release fixes the slices per scan; slice numbers are not aligned across scans, so no further anatomical selection was possible. |
| FP16 mixed precision | bf16 autocast | Same speed on the RTX 5070, with no loss scaling needed. |

---

## 3. Data, pre-processing and splits

**Dataset.** The pre-processed ADNI release used in COMP3710 (`AD_NC/{train,test}/{AD,NC}/<scan>_<slice>.jpeg`): 1,526 T1 MRI scans, each supplied as 20 consecutive axial slices (256×240, greyscale). The number in each file name is the ADNI **image (scan) ID**, not a patient ID. `meta_data_with_label.json` maps every scan to its patient (e.g. `068_S_0473`). The 1,526 scans come from **680 patients**: 221 AD patients with 743 scans (≈3.4 visits each) and 459 NC patients with 783 scans (≈1.7 each). No patient has scans under both labels. Slice numbers are not aligned across scans (each scan's 20 slices start at a different index), so position analyses use the slice's *rank* within its scan.

![Example input slices](results/data_examples.png)

### 3.1 Leakage audit: why the official split was not used

The first full ConvNeXt run used the official `test/` folder as the test set and split `train/` into train/validation by file-name ID. It reached 0.97 validation accuracy but only **0.748** test accuracy. Mapping scans to patients explained the gap:

* the validation split was really a *scan*-level split, so other visits of the same patients sat in both train and validation;
* the official `train/` and `test/` folders **share 216 patients** (5,560 of the 9,000 test slices).

Breaking that test result down by whether the patient had been seen during training shows what the model had learned:

| Official test patients | Patients | Slices | Accuracy | AD recall | NC recall |
|---|---|---|---|---|---|
| also in training (other visits) | 216 | 5,560 | 0.973 | 0.958 | 0.983 |
| **never seen** | 115 | 3,440 | **0.385** | **0.141** | 0.933 |

The network had largely learned to **recognise individual brains**, not Alzheimer's. AD patients appear many times in training (≈3.4 scans each), so "familiar anatomy" became a proxy for AD. On new AD patients it predicted NC 86% of the time, which is the most dangerous failure for a screening tool, and the leaky validation set would have hidden it. Everything below uses the patient-disjoint split instead.

### 3.2 Patient-disjoint splits ([`dataset.py`](dataset.py))

Both official folders are pooled, and **patients** (not scans or slices) are shuffled per class with a fixed seed and assigned 70 / 10 / 20 % to train / validation / test. Every slice of every scan of a patient lands in exactly one split. `python dataset.py` asserts that the three patient sets are disjoint.

| Split | Patients | Scans | Slices | AD slice fraction |
|---|---|---|---|---|
| Train | 476 | 1,068 | 21,360 | 0.478 |
| Validation | 68 | 155 | 3,100 | 0.581 |
| Test | 136 | 303 | 6,060 | 0.469 |

* The 20 % test share leaves 136 unseen patients (44 AD, 92 NC). That is enough for usable patient-level estimates while keeping most patients for training.
* Validation is used for model selection (best epoch), for fitting the calibration temperature and for choosing the referral threshold. The test split is never used to pick anything.
* **Balanced sampling.** Each training epoch draws 21,360 slices *with replacement*, with each slice weighted by 1 / (slices of its patient × patients in its class). Every patient, and both classes, then contribute equally, which removes the "many visits ⇒ AD" shortcut found above.

### 3.3 Pre-processing and augmentation

* All slices are decoded once into a cached `uint8` array (`cache/`, git-ignored), so training has no per-epoch disk I/O.
* Intensities are scaled to [0, 1] and standardised with the mean/std of the *training* patients (0.116 / 0.226). The statistics are stored in the checkpoint and reused at inference.
* Slices are resized to 224×224 (bilinear) to match ConvNeXt's pretraining resolution.
* **Augmentation** (training only, applied per sample on the GPU with `affine_grid` / `grid_sample`): rotation ±10°, scale 0.9–1.1, translation ±6%, horizontal flip (p = 0.5; AD atrophy is broadly bilateral), and brightness/contrast jitter of ±10%. These reflect plausible variation in head position and scanner intensity without changing anatomy.

---

## 4. Training

[`train.py`](train.py) trains either model with identical data handling:

| Setting | ConvNeXt-T | Baseline CNN |
|---|---|---|
| Initialisation | ImageNet-1k (torchvision), stem → 1 channel | random (PyTorch default Kaiming-uniform) |
| Optimiser | AdamW, lr 1e-4, weight decay 0.05 | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | 1 epoch linear warm-up, cosine decay to 0 | same |
| Epochs / batch | 30 / 64 | 30 / 64 |
| Regularisation | stochastic depth 0.2, augmentation | dropout 0.3, BatchNorm, augmentation |
| Loss | cross-entropy (no label smoothing, so calibration is measured on the raw objective) | same |
| Precision | bf16 autocast, channels-last, grad-clip 1.0 | same |
| Model selection | best validation accuracy | same |

Weight decay is not applied to norms, biases or layer-scale parameters. Each run writes per-epoch losses and metrics to `results/<tag>_history.json` and plots them to `results/<tag>_curves.png`.

<!--RESULTS_SECTIONS-->

---

## 9. Usage and reproducibility

### Dependencies

Tested with the following versions (Windows 11, NVIDIA RTX 5070 Laptop GPU 8 GB, CUDA 12.8):

| Package | Version |
|---|---|
| Python | 3.13.7 |
| torch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 (for ImageNet weights only) |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 (metrics) |
| matplotlib | 3.10.7 |
| Pillow | 11.3.0 |

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install numpy==2.2.6 scikit-learn==1.7.2 matplotlib==3.10.7 pillow==11.3.0
```

### Files

| File | Purpose |
|---|---|
| [`dataset.py`](dataset.py) | loading/caching, subject-level split, normalisation, GPU augmentation |
| [`modules.py`](modules.py) | ConvNeXt (from scratch) + ImageNet weight mapping, baseline CNN |
| [`train.py`](train.py) | training, validation, test evaluation, checkpointing, curves |
| [`predict.py`](predict.py) | example inference with the saved model + referral rule + visualisation |
| [`evaluate.py`](evaluate.py) | dilemma analysis: calibration, risk-coverage, profiling, Grad-CAM failure autopsy |
| [`utils.py`](utils.py) | metrics, ECE, temperature scaling, thresholds, subject aggregation |

### Running

Set the data location (defaults to `D:\ADNI\AD_NC`; on Rangpur use `/home/groups/comp3710/ADNI/AD_NC`):

```bash
export ADNI_ROOT=/home/groups/comp3710/ADNI/AD_NC
```

```bash
python train.py --arch convnext --epochs 30
```

```bash
python train.py --arch baseline --epochs 30 --lr 1e-3 --weight-decay 1e-4
```

```bash
python evaluate.py
```

```bash
python predict.py --ckpt checkpoints/convnext_best.pt
```

`predict.py` can also classify individual slices:

```bash
python predict.py --ckpt checkpoints/convnext_best.pt --images /path/to/AD_NC/test/AD/1003730_80.jpeg
```

**Reproducibility.** Seeds are fixed (`--seed 42`) for Python, NumPy and PyTorch. The subject split is deterministic given the seed. cuDNN autotuning and bf16 arithmetic make runs non-bit-exact, so expect run-to-run variation of roughly ±1% in test accuracy. Checkpoints and cached arrays are git-ignored. Only code and figures are committed.

---

## 10. References

1. Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie. *A ConvNet for the 2020s.* CVPR 2022. https://arxiv.org/abs/2201.03545
2. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017. (temperature scaling, ECE)
3. R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
4. Y. Geifman, R. El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017. (risk-coverage)
5. Alzheimer's Disease Neuroimaging Initiative (ADNI). https://adni.loni.usc.edu
6. torchvision ConvNeXt implementation and ImageNet-1k weights (used only as a weight source; architecture re-implemented). https://pytorch.org/vision/stable/models/convnext.html
