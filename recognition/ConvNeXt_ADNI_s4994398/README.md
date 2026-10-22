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

**User need and scope.** The intended user is a neuroradiology screening service. The prototype pre-reads axial T1 MRI slices of a subject, flags likely AD, and must *refer* cases it is unsure about rather than silently automating them. It is a decision-support prototype, not a diagnostic device.

**Acceptance criteria.**
1. Slice-level test accuracy ≥ 0.80 on the held-out ADNI test split (course requirement).
2. Strict subject-level separation: no subject's slices appear in more than one of train / validation / test.
3. Report per-class precision/recall, macro-F1, AUROC and expected calibration error (ECE), and compare against a from-scratch CNN baseline on identical splits.
4. A referral (reject-option) threshold chosen on *validation* data only, with its test accuracy, coverage and number of automated missed-AD cases reported.
5. Resource budget: training fits in an 8 GB laptop GPU in under 1 h per model; inference under 10 ms per slice.

**Model choice and course concepts.** ConvNeXt-T (Hard band) uses ideas from the course (residual CNNs, normalisation, depthwise-separable convolutions, transfer learning) and borrows the ViT design choices discussed in lectures. Fine-tuning ImageNet weights was chosen over training from scratch because only ~1,000 training subjects are available, and the 20 slices per subject are highly correlated. A VGG-style CNN is the baseline.

**Preliminary feasibility evidence.** Data audit: 21,520 training slices (1,076 subjects: 520 AD, 556 NC) and 9,000 test slices (450 subjects: 223 AD, 227 NC), 20 slices per subject, 256×240 greyscale, roughly balanced classes. A one-epoch smoke test of ConvNeXt-T took 88 s and reached 0.89 validation and 0.72 test accuracy. This showed the pipeline works, and also that validation (drawn from the training subjects) is noticeably easier than the official test split.

**Risks, budget and fallback.** *Risk:* the validation/test gap means early stopping on validation could overfit to the training distribution. *Mitigation:* strong augmentation, stochastic depth and weight decay, plus reporting both splits honestly. *Risk:* overconfident errors. *Mitigation:* temperature scaling and a referral rule. *Budget:* ~45 min per ConvNeXt run and ~15 min per baseline run on an RTX 5070 Laptop GPU (8 GB). *Fallback:* if slice-level accuracy fell short, aggregate the 20 slices per subject (subject-level voting), or move to ConvNeXt-S / higher resolution.

---

## 3. Data, pre-processing and splits

**Dataset.** The pre-processed ADNI release used in COMP3710 (`AD_NC/{train,test}/{AD,NC}/<subject>_<slice>.jpeg`). Each subject contributes 20 consecutive axial slices. Slice numbers are not aligned across subjects (each subject's 20 slices start at a different index), so position analyses use the slice's *rank* within its subject.

![Example input slices](results/data_examples.png)

**Splits and leakage prevention** ([`dataset.py`](dataset.py)).
* **Test:** the official `test/` folder (9,000 slices, 450 subjects), used only for final evaluation.
* **Train / validation:** the official `train/` folder is split **by subject**, stratified by class, with 10% of subjects held out for validation (108 subjects, 2,160 slices; 968 subjects, 19,360 slices remain for training). Splitting by slice would put near-identical neighbouring slices of the same brain on both sides and inflate validation accuracy. `python dataset.py` asserts that the train and validation subject sets are disjoint.
* Validation is used for model selection (best epoch), for fitting the calibration temperature and for choosing the referral threshold. The test set is never used to pick anything.

**Pre-processing.**
* All slices are decoded once into a cached `uint8` array (`cache/`, git-ignored), so training has no per-epoch disk I/O.
* Intensities are scaled to [0, 1] and standardised with the mean/std of the *training* subjects (0.115 / 0.225). The statistics are stored in the checkpoint and reused at inference.
* Slices are resized to 224×224 (bilinear) to match ConvNeXt's pretraining resolution.

**Augmentation** (training only, applied per sample on the GPU with `affine_grid` / `grid_sample`): rotation ±10°, scale 0.9–1.1, translation ±6%, horizontal flip (p = 0.5; AD atrophy is broadly bilateral), and brightness/contrast jitter of ±10%. These reflect plausible variation in head position and scanner intensity without changing anatomy.

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
