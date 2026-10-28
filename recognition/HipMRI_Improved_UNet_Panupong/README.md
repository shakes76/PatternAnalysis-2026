# HipMRI 2D Improved U-Net - Panupong

This COMP3710 project segments the prostate, together with the body, bones, bladder and rectum, in 2D slices of male pelvic MRI from the HipMRI study. It compares two networks implemented from scratch in PyTorch: a 2D adaptation of the Improved U-Net of Isensee et al. (2018), the chosen Normal-difficulty model, and a standard 2D U-Net (Ronneberger et al., 2015) as the baseline. Both models were trained under the same rules on the course's Rangpur cluster and tested once on the supplied test split. **Both exceed the target prostate test Dice of 0.75:** 0.8858 for the baseline and 0.8738 for the Improved U-Net. The report then investigates the HipMRI open research dilemma: where the remaining errors are, whether three-slice (2.5D) context removes them (in a single training run it did not fix the outermost slice of the gland), and which model to recommend.

| Result (test split, validation-selected checkpoints) | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Prostate Dice, per-series 3D mean (target >= 0.75) | **0.8858** | **0.8738** |
| Parameters | 31,036,806 | 3,189,282 |
| Peak training GPU memory (batch 16) | 2.32 GiB | 0.96 GiB |
| Inference time per slice, batch 16 / batch 1 | 0.63 / 3.06 ms | 0.46 / 7.46 ms |
| Prostate slices missed entirely (of 195) | 17 | 22 |

## Contents

1. [Problem and algorithm](#1-problem-and-algorithm)
2. [Feasibility review (compulsory check-off document)](#2-feasibility-review-compulsory-check-off-document)
3. [Data and preprocessing](#3-data-and-preprocessing)
4. [Training](#4-training)
5. [Test results](#5-test-results)
6. [Investigation of the open research dilemma](#6-investigation-of-the-open-research-dilemma)
7. [Reproducibility and usage](#7-reproducibility-and-usage)
8. [Files](#8-files)
9. [Development timeline](#9-development-timeline)
10. [Acknowledgements and references](#10-acknowledgements-and-references)
11. [Artificial Intelligence Usage Disclosure](#artificial-intelligence-usage-disclosure)

## 1. Problem and algorithm

### The problem

Radiotherapy planning for prostate cancer relies on outlines of the prostate and of the organs at risk next to it, the bladder and the rectum. The HipMRI data come from a study of MRI-alone radiotherapy planning with automatic contouring (Dowling et al., 2015). The task here is semantic segmentation: given one 256 x 128 MRI slice, assign each pixel to one of six classes (0 background, 1 body, 2 bones, 3 bladder, 4 rectum, 5 prostate). The prostate is the class of interest. It is small, low in contrast, and touches the bladder and rectum, so a model can score well on average while still missing the gland on some slices or leaking into a neighbouring organ.

### How the Improved U-Net works

Like every U-Net, the Improved U-Net has a contracting path that captures what is in the image and an expanding path that recovers where it is, joined by skip connections that copy fine detail across (Figure 1). The contracting path has five levels. Level 1 starts with a 3x3 convolution to 16 feature maps; at levels 2-5 a 3x3 convolution with stride 2 halves the image size and doubles the feature maps (32 up to 256). At every level a **context module** then refines the features. A context module is a pre-activation residual block: two 3x3 convolutions with dropout between them, whose output is added back to the block's input, which helps gradients flow through the deep network. The expanding path climbs back up with **upsampling modules** (repeat each pixel twice, then a 3x3 convolution) and **localization modules** (a 3x3 then a 1x1 convolution) that merge the upsampled features with the skip connection. **Deep supervision** adds segmentation outputs at three decoder levels: each lower-resolution map is upscaled and added to the next. The loss is computed on this summed output, so the coarser decoder levels contribute directly to the final segmentation and get a shorter gradient path from the loss. Instance normalisation and leaky ReLU are used throughout. A softmax turns the six output maps into class probabilities, and the most probable class gives the label map.

![Figure 1: the 2D Improved U-Net used in this project](figures/improved_unet_architecture.png)

*Figure 1. The 2D Improved U-Net used in this project, drawn from the layer definitions in `modules.py` for one 256 x 128 slice. It follows Fig. 1 of Isensee et al. (2018) with every 3D convolution made 2D.*

### Baseline: standard 2D U-Net

The baseline follows Ronneberger et al. (2015): two 3x3 convolutions per level, 2x2 max pooling, four downsampling steps with 64 to 1024 feature maps, 0.5 dropout at the bottleneck, 2x2 transposed convolutions for upsampling, concatenated skip connections and a final 1x1 convolution to the six classes. It has about 10 times more parameters than the Improved U-Net (31.0 M against 3.19 M).

### Loss

Both models are trained with the multiclass soft Dice loss of Isensee et al. (2018, Eq. 1), so only the architecture differs between them:

```text
L = -(2 / |K|) * sum_k [ sum_i u_ik v_ik / (sum_i u_ik + sum_i v_ik) ]
```

Here `u` is the softmax output, `v` the one-hot ground truth, `k` runs over the six classes K and `i` over every pixel of every slice in the batch. Summing over the whole batch keeps a useful gradient for the prostate even though most slices contain none. A Dice-based loss suits the class imbalance: the prostate is only about 0.4% of the training pixels.

### Deviations from the reference papers

**Baseline U-Net (Ronneberger et al., 2015):**

- `padding=1` replaces unpadded convolutions.
- Batch normalisation is added.
- Training uses the soft Dice loss with Adam instead of weighted cross-entropy with SGD.
- The 0.5 dropout at the end of the contracting path is kept.

**Improved U-Net (Isensee et al., 2018):**

- 3D convolutions become 2D, and the model takes one input channel and produces six outputs.
- The paper does not specify several details, so these are assumptions:
  - the normalisation/activation order outside the context modules;
  - the 1x1 segmentation layers;
  - the width of the localisation 3x3 convolution.
- The training schedule is adapted to full passes over the data (learning-rate decay rescaled).
- Elastic deformation, augmentation attenuation, test-time mirroring, MC-dropout averaging and the 5-fold ensemble are not used.
- Per-slice normalisation replaces per-patient normalisation within a brain mask.

## 2. Feasibility review (compulsory check-off document)

This is the one-page feasibility review presented to the tutor at the compulsory check-off. The check-off was completed by 11 October 2026. The text below is the 7 October 2026 version. It is reproduced unchanged except for four things: its title line ("COMP3710: HipMRI 2D Segmentation") is replaced by this heading, the student number is omitted from the header line, and the heading levels and numbering (1-4 became 2.1-2.4, References became 2.5) and the link formatting were adapted for this README. Reference [1] is the version of the assignment specification that was current at the time. It was written before any model was trained, so its open questions and plans are as they stood then. The outcome of each acceptance criterion is given after it.

**One-page feasibility review - DRAFT for tutor consultation**

Student: Panupong Pakwiset | Date: 7 October 2026 | Difficulty: Normal

### 2.1 User need, scope and acceptance criteria

The intended user is an imaging specialist reviewing proposed prostate contours. This educational prototype will compare standard 2D U-Net with 2D Improved U-Net on HipMRI. The engineering question is whether the improved 2D architecture reduces missed tissue and boundary leakage enough to justify its computational cost. Global Dice will be considered alongside boundary errors. Proposed acceptance criteria are:

1. **Data integrity:** preserve supplied splits; verify patient grouping across visits, readable image-mask pairs, spatial alignment and confirmed anatomical label IDs before modelling.
2. **Accuracy:** target prostate test Dice >= 0.75; report per-class Dice/IoU using an explicit aggregation and empty-mask policy fixed before final evaluation.
3. **Fair comparison:** evaluate both models on identical held-out data with matched preprocessing and a documented training budget; select settings using validation only.
4. **Failure analysis:** compare input, ground truth and both models' predictions for 3-5 low-performing cases. Quantify missed/extra prostate pixels, investigate causes such as small targets, unclear boundaries or annotation uncertainty, and distinguish recurring limitations from isolated errors. Inspect base/apex cases only when anatomical ordering is verified.
5. **Resources:** aim for peak training VRAM <= 12 GiB on one Colab T4; measure parameter count, peak VRAM and inference time for both models. This is a proposed limit, not measured usage.

### 2.2 Model choice and course concepts

Use HipMRI Nifti slices with the supplied train/validation/test allocation. Standard 2D U-Net [2] is the baseline. The Normal-level model will be a documented 2D adaptation of Isensee et al. [3], retaining residual context blocks, instance normalization and multi-scale segmentation outputs (deep supervision). Confirm this adaptation with the tutor because the reference architecture is 3D. Skip connections preserve spatial detail, residual connections support gradient flow, and Dice-based optimization addresses class imbalance. The paper's qualitative analysis links some errors to poorly defined labels, motivating inspection of annotation uncertainty in our own results rather than attributing every error to the architecture. Normalize image intensities and apply paired augmentation; class channels depend on the complete label audit.

### 2.3 Preliminary feasibility evidence

Local file counts are 11,460 training, 660 validation and 540 test image-mask pairs. The earlier filename audit recorded no missing counterparts or overlapping case IDs between splits; patient grouping across visits still requires verification. A saved inspection of one training pair shows matching 256 x 128 shapes and affines, finite values, and mask IDs 0-3; anatomical meanings and the complete label set remain unknown. Saved outputs from the 4 October VS Code/Colab check show successful PyTorch CUDA tensor arithmetic on a Tesla T4. This verifies GPU access, not model feasibility. No segmentation model, training run, benchmark or measured model memory exists yet.

### 2.4 Risks, budget, next experiment and fallback

Main risks are unknown prostate mapping, class imbalance, ambiguous boundaries, patient leakage and Colab interruptions. Confirm processed 2D label IDs with the tutor, audit grouping and labels, and store checkpoints/results persistently outside temporary runtime storage. A provisional budget is one T4 and at most 6 GPU-hours for the initial two-model comparison; revise this after measuring epoch time. The repository is set up. Next: complete the data audit, build the data loader and inspect a batch. After label confirmation, run a brief baseline smoke test before longer experiments. If memory is insufficient, reduce batch size first. If Improved U-Net remains infeasible, retain the standard 2D baseline and discuss an Easy-level scope with the tutor. Recommend the improved model only if measured boundary-error reduction justifies its additional memory and inference cost; otherwise recommend the baseline and explain the trade-off.

### 2.5 References

[1] COMP3710 Report draft v2.2, Sections 2.2 and 3 (pp. 5-6, 11-12).

[2] Ronneberger et al. (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*. <https://arxiv.org/abs/1505.04597>

[3] Isensee et al. (2018), *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*. <https://arxiv.org/abs/1802.10508>

### 2.6 Outcome of the review (updated 22 October 2026)

| Acceptance criterion | Outcome |
| --- | --- |
| 1. Data integrity | Met, with limits. The supplied splits were kept, and filename case IDs are disjoint across splits (31 / 4 / 3 cases). A full audit read all 12,660 image-mask pairs and found them readable, with matching shapes and affines. Label 5 = prostate was confirmed from the dataset publisher's documentation and a direct check of one series ([LABEL_MAPPING.md](LABEL_MAPPING.md)); this was found by independent research, not confirmed by the tutor. Patient identity beyond the filename case IDs cannot be verified. |
| 2. Accuracy | Met. Test prostate Dice (per-series 3D mean) is 0.8858 for the baseline and 0.8738 for the Improved U-Net. Per-class Dice/IoU are reported, and the aggregation and empty-mask rules were fixed before training. |
| 3. Fair comparison | Met. Both models used the same splits, preprocessing, loss, optimiser, 40-epoch budget, seed and GPU type. Checkpoints were chosen on validation only. Test metrics were computed once; the later `predict.py` runs only reproduce them, draw figures and add analysis, with nothing selected or tuned. |
| 4. Failure analysis | Largely met. Five failure cases are shown, and small targets at the ends of the gland are identified as the recurring limitation. The Improved U-Net predicts no prostate on these slices, so all their prostate pixels are missed. Missed and extra prostate pixels are quantified by class, with HD95 and results by distance from the ends of the gland, in Section 6.3. Annotation uncertainty was not assessed. The spacing has no known units and the physical orientation cannot be verified, so the condition for naming base or apex is not met; the cases are described as the two ends of the prostate's slice range instead. |
| 5. Resources | Met. Peak training memory was 2.32 GiB (baseline) and 0.96 GiB (Improved) on the A100, and 4.85 / 0.92 GiB in the Colab T4 timing runs, all below 12 GiB. Parameter counts and inference times were measured. |

Changes from the plan:

- **Platform:** final training moved from Colab T4 to the course's Rangpur cluster (A100). Training plus validation took 20.4 min (baseline) and 17.1 min (Improved) for 40 epochs. The job wall times were 29 and 19 min, well inside the 6 GPU-hour budget.
- **Risks resolved:** the prostate label mapping was resolved from the publisher's documentation. The pre-specified switch to a Dice plus cross-entropy loss was not needed, and the Improved U-Net was feasible, so the Easy-level fallback was not used.
- **Open question:** the review asks whether the Improved U-Net reduces missed tissue and boundary leakage enough to justify its cost. It does not reduce missed tissue: it misses the prostate on 22 slices against 17, all near the ends of the gland. Leakage into the rectum or bladder is below 1% for both models. The Improved U-Net is also cheaper, not more expensive, except for single-slice latency, so the expected trade-off does not arise. Details and the recommendation are in Section 6.

## 3. Data and preprocessing

### Dataset and labels

The data are the processed 2D slices of the HipMRI study supplied for COMP3710 (`keras_slices_data`): NIfTI image slices with matching segmentation masks, mostly 256 x 128 pixels, with 60 slices per case/week series. The dataset publisher's [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034), attached to the [CSIRO collection](https://doi.org/10.25919/45t8-p065), defines `0=Background`, `1=Body`, `2=Bones`, `3=Bladder`, `4=Rectum` and **`5=Prostate`**. This mapping came from independent online research, followed by a direct array comparison showing that all 60 processed masks of Case 006 / Week 0 keep the original numeric labels. It was not obtained from or confirmed by a tutor, and other series have not been individually matched to the original 3D masks. The source, verification method and remaining limitations are in [LABEL_MAPPING.md](LABEL_MAPPING.md). The NIfTI spacing is [1, 1] with no known units, so in-plane distances (HD95) are given in pixels and distances along the gland in slices.

### Splits

The supplied train/validation/test splits are kept unchanged: 11,460 / 660 / 540 slices from 31 / 4 / 3 cases. They are kept because the assignment asks for the supplied splits and because the filename case IDs are disjoint across them: all weekly scans of a case stay in one split, so the model is evaluated on cases it has never seen. Patient identity beyond the filename case IDs cannot be verified. Validation is used for checkpoint selection and the training-curve review, and the test split is used once at the end.

### Preprocessing and augmentation

These rules were set before any training:

- **Shape:** 60 training slices (Case 019 / Week 1) are 256x144. They are centre-cropped to 256x128, which keeps every bladder, rectum and prostate pixel. Nothing is resized, so labels are never interpolated.
- **Intensity:** per-slice z-score, as in the course loader (assignment Appendix B), clipped to [-5, 5] as in Isensee et al. (2018). No statistics are fitted on any split.
- **Labels:** labels 0-5 are kept as stored and one-hot encoded. Label 5 is the prostate.
- **Augmentation:** training slices only, following the augmentation types of Isensee et al. (2018). Rotation +/-15 degrees, scale 0.85-1.15, a flip along the 256-pixel axis, and gamma 0.7-1.5, each with p = 0.5. The same geometric transform is applied to the image and the mask (nearest-neighbour for the mask).
- **Loss:** both models use the multiclass soft Dice loss of Isensee et al. (2018, Eq. 1), summed over the whole batch. The batch size is at least 8.
- **Empty masks:** Dice is undefined (excluded from the mean and counted) when both the prediction and the ground truth are empty, and 0 when only the prediction is non-empty. Every validation and test series contains prostate.
- **Primary metric:** 3D prostate Dice per case/week series, averaged over series. The argmax label maps of the 60 slices of a series are stacked into one volume, and Dice = 2|P ∩ G| / (|P| + |G|) is computed on the prostate label of that volume; the soft Dice is used only as the training loss. It is also used for validation checkpoint selection. The global, case-weighted, per-class Dice/IoU and slice-level metrics are reported alongside it.
- **2.5D input (follow-up experiment only):** for the experiment in Section 6.4, the two neighbouring slices of the same series are added as extra input channels. Each channel is normalised as above, and all channels get the same augmentation. The main results use single 2D slices.

### Data pipeline check

`python dataset.py` was run on the course copy of the data on Rangpur (node `a100-6`, code at commit `92f51d4`, 95 s; console output supplied by the user):

| Split | Pairs | Cases | Series | Prostate-positive slices |
| --- | ---: | ---: | ---: | ---: |
| Train | 11,460 | 31 | 191 | 4,152 |
| Validation | 660 | 4 | 11 | 254 |
| Test | 540 | 3 | 9 | 195 |

- Every item of every split passed the shape, range and one-hot checks.
- The 60 training slices from Case 019 / Week 1 were cropped.
- Per-class pixel counts match the earlier full-dataset audit of the local copy, with one expected difference. The training Background, Body and Bones counts are lower by 245,760 pixels in total, which is exactly the cropped area (60 slices x 256 x 16 columns). Bladder, rectum and prostate pixels are unchanged, so the Rangpur copy and the audited copy agree on these summaries.

## 4. Training

### Setup

Both models were trained from scratch with identical settings, so the comparison changes only the architecture:

- **Optimiser:** Adam, learning rate 5e-4, weight decay 1e-5, as in Isensee et al. (2018).
- **Schedule:** the learning rate after epoch index e is 5e-4 x 0.985^(e x 300 / 40). The paper decays by 0.985 per "epoch" of 100 batches for 300 such epochs; rescaling the exponent keeps a similar overall decay over 40 full passes (the last-epoch rate is about 1.2% of the initial value, against 1.1% in the paper).
- **Budget:** 40 epochs of 716 batches, batch size 16, mixed precision (AMP), seed 42.
- **Checkpoints:** after every epoch, the per-series 3D prostate Dice on validation is computed. The best epoch is kept, along with the last epoch.
- **Fallback rule (fixed in advance):** switch to Dice plus cross-entropy only if training loss fell by less than 10% between epochs 1 and 5, or if validation prostate Dice was below 0.3 at epoch 5.
- **Test protocol:** the test split is evaluated only with `--test`, after the full epoch budget and after the training curves have been reviewed, once for the best and once for the last checkpoint.

Both runs were done on Rangpur node `a100-6` (NVIDIA A100-PCIE-40GB), with code at commit `92f51d4`. The values below come from each run's `history.json` and `config.json` and from the Slurm logs, which the user copied back from Rangpur.

### Training and validation results

**All Dice values in this table are on the validation split.**

| Measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Best validation prostate Dice, per-series mean (epoch) | **0.8686** (28) | **0.8564** (34) |
| At that epoch: case-weighted / global prostate Dice | 0.8331 / 0.8697 | 0.8157 / 0.8534 |
| At that epoch: foreground Dice (classes 1-5) | 0.9202 | 0.9200 |
| Last epoch (40): validation prostate Dice | 0.8626 | 0.8556 |
| Last 10 epochs: mean ± SD validation prostate Dice | 0.8628 ± 0.0018 | 0.8548 ± 0.0013 |

Training curves (loss and validation Dice per epoch): [baseline](figures/curves_unet.png) and [Improved U-Net](figures/curves_improved.png). Training time and memory are in the resource table in Section 6.2.

- **Accuracy:** on validation, the baseline's prostate Dice is slightly higher. The gap is about 0.012 at the best epoch and 0.008 over the last 10 epochs. The foreground Dice over classes 1-5 (body, bones, bladder, rectum and prostate) is nearly identical: 0.9202 against 0.9200 at the best epoch.
- **Training curves:** validation loss levels off after about 10-15 epochs, while training loss keeps decreasing (about -0.95 against -0.78 for validation). There is no late drop in validation Dice.
- **Fallback rule:** not triggered. Training loss fell by 18.7% for the baseline and 14.6% for Improved between epochs 1 and 5, and epoch-5 validation Dice was 0.853 and 0.838.
- **Caveats:**
  - Validation has only 11 series from 4 cases, and each model was trained with a single seed, so these gaps have not been tested for significance.
  - The best epoch was chosen on the same validation split, so the best-epoch scores are optimistic.

## 5. Test results

Each model was evaluated in a single test run, after 40 epochs of training and after its training curves had been reviewed. The run scored the two checkpoints fixed before training: the best-validation checkpoint and the last-epoch checkpoint. The runs used `train.py --resume --test` on Rangpur node `a100-5` (NVIDIA A100-PCIE-40GB), with code at commit `92f51d4`: job 683766 for the baseline and job 683765 for the Improved U-Net. The user ran the jobs and copied back the logs and `test_metrics_*.json` files, and the values below come from those files.

The test split has 540 slices from 9 case/week series of 3 cases (040-042). Prostate appears in 195 slices and in every series.

### Primary result

The checkpoint for each model was chosen beforehand: the epoch with the best validation prostate Dice. The primary metric is the 3D prostate Dice per series, averaged over the 9 series.

| Test measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Checkpoint epoch (best on validation) | 28 | 34 |
| **Prostate Dice, per-series mean (primary)** | **0.8858** | **0.8738** |
| Target prostate Dice >= 0.75 | met | met |
| Prostate Dice, case-weighted mean | 0.8762 | 0.8685 |
| Prostate Dice, global (all test pixels) | 0.8826 | 0.8730 |
| Prostate IoU, per-series mean | 0.7954 | 0.7765 |
| Mean slice-level prostate Dice (195 slices with prostate) | 0.7758 | 0.7446 |
| Slices with prostate scored 0 (of 195) | 17 | 22 |
| False-positive prostate on slices without prostate (of 345) | 2 | 4 |
| Foreground Dice, classes 1-5 (per-series mean) | 0.9279 | 0.9182 |

**Per-class Dice / IoU** (per-series mean, best checkpoint):

| Class | Baseline U-Net | Improved U-Net |
| --- | --- | --- |
| 0 Background | 0.9955 / 0.9911 | 0.9961 / 0.9921 |
| 1 Body | 0.9858 / 0.9720 | 0.9846 / 0.9697 |
| 2 Bones | 0.9428 / 0.8919 | 0.9385 / 0.8842 |
| 3 Bladder | 0.9635 / 0.9298 | 0.9588 / 0.9211 |
| 4 Rectum | 0.8616 / 0.7600 | 0.8351 / 0.7216 |
| 5 Prostate | 0.8858 / 0.7954 | 0.8738 / 0.7765 |

**Prostate Dice per test series** (best checkpoint):

| Series | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| 040 / week 0 | 0.8825 | 0.8649 |
| 040 / week 1 | 0.8809 | 0.8767 |
| 040 / week 2 | 0.9006 | 0.8778 |
| 040 / week 3 | 0.8885 | 0.8526 |
| 040 / week 4 | 0.8626 | 0.8504 |
| 040 / week 5 | 0.9132 | 0.9037 |
| 040 / week 7 | 0.9061 | 0.9088 |
| 041 / week 0 | 0.8657 | 0.8504 |
| 042 / week 0 | 0.8722 | 0.8788 |

The last-epoch checkpoints (epoch 40) were also tested, as planned before training. Their per-series prostate Dice was 0.8884 for the baseline and 0.8734 for the Improved U-Net. They are reported for completeness only. The primary result stays with the validation-selected checkpoints, because choosing between checkpoints by test score would use the test split for model selection.

### Observations

- **Target:** both models are above 0.75 on the per-series (primary), case-weighted and global 3D prostate Dice, for both checkpoints. The Improved U-Net's mean slice-level prostate Dice is just below 0.75 (0.7446 best, 0.7415 last), but the target applies to the 3D Dice. The baseline's slice-level value is above it (0.7758 best, 0.7833 last).
- **Baseline vs Improved:** the baseline is higher by 0.012 on the primary metric. This is about the same gap as on validation (0.0122 at the best epochs). The baseline is also higher for every foreground class (1-5), and the largest gap is on the rectum (0.026 Dice). Only on background is the Improved U-Net marginally higher (0.9961 against 0.9955). The baseline is higher in 7 of the 9 test series; the Improved U-Net is higher for 040 / week 7 and 042 / week 0.
- **Uncertainty:**
  - The test split has only 3 patients, and 7 of its 9 series come from case 040, so the series are not independent.
  - Each model was trained with a single seed.
  - No significance test was run, so the gap should not be read as a reliable difference between the architectures.
- **Cost:** the Improved U-Net reaches this accuracy with about 10 times fewer parameters (3.19 M against 31.0 M) and less training memory and time (Section 6.2).
- **Test vs validation:** test Dice is about 0.017 higher than the best validation Dice for both models. The two splits contain different cases and are both small (3 and 4 patients), so this difference may come from which cases are in each split. It should not be read as evidence of better generalisation.
- **Where the errors are:**
  - Slice-level prostate Dice (about 0.74-0.78) is lower than the per-series 3D Dice. The misses are concentrated on slices with very little prostate.
  - The Improved U-Net scores 0 on 22 prostate slices. Their median ground-truth prostate size is 54.5 pixels (12 of the 22 have 55 or fewer), against a median of 327 over all prostate slices.
  - The rectum is the hardest class for both models.
- Nothing was tuned or re-selected after the test results were seen.

### Example inputs and outputs

`predict.py` reloads the validation-selected checkpoints and the test split, predicts every test slice and draws the figures below. It was first run on Rangpur as job 685612 (node `a100-4`, NVIDIA A100-PCIE-40GB, code at commit `b7d4fba`, 1 min 7 s). The figures shown here come from a later run, job 690027 (node `a100-7`, code at commit `bb98a72`, 1 min 6 s), which added a difference overlay after each prediction; it selected the same slices, with the same slice Dice values, and reproduced the saved test Dice to four decimal places. The user ran both jobs and copied the outputs back.

**Consistency check.** The test prostate Dice recomputed by `predict.py` was 0.8858 for the baseline and 0.8738 for the Improved U-Net. These agree with the values saved by `train.py` when rounded to four decimal places. The absolute differences are about 0.00001: 8e-6 for the baseline and 1.5e-5 (job 685612) or 1.9e-5 (job 690027) for the Improved U-Net. This indicates that the reloaded models and the preprocessing match the test evaluation. The tiny difference is consistent with non-deterministic GPU arithmetic: `cudnn.benchmark` is on, mixed precision is used, and these runs were on different A100 nodes from the test evaluation (`a100-5`). This cause was not tested separately.

**Visual comparison** (Figure 2). For each of 4 test series, spread evenly over the sorted series, the figure shows the slice with the largest ground-truth prostate. The columns are the input MRI and the ground truth, then, for each model, its prediction followed by a prostate difference overlay. In the overlay, green marks prostate pixels found correctly (true positives), orange marks predicted prostate that is not prostate (false positives) and light blue marks missed prostate (false negatives); the panel title gives the false-positive and false-negative pixel counts. On the prediction panels, the thin bright-green (lime) line is the ground-truth prostate outline and the semi-transparent teal fill is the rectum; it is lighter than the solid green that marks correctly found prostate in the overlay panels. On these slices, both models outline the prostate, bladder and bones closely: prostate Dice is 0.92-0.96 for the baseline and 0.90-0.94 for the Improved U-Net.

![Figure 2: test predictions of both models](figures/test_comparison.png)

*Figure 2. Example test inputs, ground truth, and each model's prediction followed by its prostate difference overlay (green true positive, orange false positive, light blue false negative).*

## 6. Investigation of the open research dilemma

The HipMRI dilemma in the assignment asks whether a good global Dice hides failures at the base and apex slices, or leakage into the rectum or bladder. It also asks whether more spatial context removes these errors or only improves easy central slices. This section benchmarks the two models, profiles their cost, dissects their failures, tests whether three-slice (2.5D) context fixes them, gives a recommendation and closes with future work.

### 6.1 Quantitative benchmark

Both models were compared under identical evaluation splits and identical rules: same data, preprocessing, loss, optimiser, schedule, epoch budget, seed and GPU type, with checkpoints chosen on validation only. On the test split, the baseline scores 0.8858 and the Improved U-Net 0.8738 (per-series 3D prostate Dice, Section 5). The baseline is ahead on every foreground class and in 7 of 9 series, but the gap of 0.012 comes from one seed and 3 test patients and was not tested for significance.

A more elaborate U-Net is not guaranteed to beat a well-trained baseline. Rodrigues et al. (2023) compared 13 prostate MRI segmentation models under one pipeline and found no statistically significant Dice difference between them, except for nnU-Net, which was better than all others by about 2%. Their data and task (whole-gland segmentation on ProstateX, with external testing) differ from this project, so this is context, not a direct comparison.

### 6.2 Resource profiling

| Measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Parameters | 31,036,806 | 3,189,282 |
| Peak allocated GPU memory, training (batch 16, AMP) | 2.32 GiB | 0.96 GiB |
| Median training time per epoch (716 batches) | 29.0 s | 24.0 s |
| Median seconds per training batch (steady state) | 0.040 | 0.033 |
| Training + validation time, 40 epochs | 20.4 min | 17.1 min |
| Job time, script start to finish (from the log) | 28 min 57 s | 19 min 13 s |
| Inference, batch size 1: ms per slice | 3.06 | 7.46 |
| Inference, batch size 16: ms per slice | 0.63 | 0.46 |
| Inference, batch size 16: slices per second | 1,593 | 2,180 |
| Peak GPU memory, inference at batch size 16 | 0.62 GiB | 0.26 GiB |
| Derived: one 60-slice series at batch size 16 / 1 | about 38 / 184 ms | about 28 / 448 ms |

All measurements were made on an NVIDIA A100-PCIE-40GB on Rangpur. Training values come from the training runs (jobs 683616 and 683615). Inference values come from `predict.py` job 685612: forward passes only, mixed precision, test images already on the GPU, median of 5 passes over the 540 test slices after one warm-up pass. The derived row multiplies the per-slice time by 60 and was not timed directly. The inference peak memory includes the model weights and the 540 test images held on the GPU (about 0.07 GiB). Four later `predict.py` runs (jobs 685882, 690027, 696531 and 697982; 696531 on the validation split) re-measured inference times within about 8% of these values.

- The Improved U-Net uses about 10 times fewer parameters, about 17% less time per training batch and about 60% less peak training memory.
- At batch size 16, it is about 27% faster per slice and uses less than half the inference memory.
- At batch size 1, it is about 2.4 times slower. A likely reason is that it has more, but much smaller, layers: 29 convolutions and 91 registered layer modules in total, against 23 and 61 for the baseline. With one small slice at a time, the fixed cost of launching each GPU operation probably outweighs the arithmetic. This explanation was not measured separately.
- The job time also includes one-time dataset loading and the per-epoch checkpoint, history and plot saving. Most of the larger baseline overhead is probably from writing its larger checkpoints: about 0.5 GB per epoch, against about 0.05 GB for Improved. This overhead was not timed separately.

**Preliminary Colab T4 timing runs (21 October 2026).** Before the final training, both models were run for 100 training batches followed by one validation pass on a Colab Tesla T4 (15,360 MiB; Python 3.13.15, PyTorch 2.11.0+cu130), with batch size 16, AMP and seed 42. The user supplied the console outputs; the underlying run artifacts were not independently inspected. All 11,460 / 660 / 540 pairs also passed the pipeline checks there.

| Colab T4 timing measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Training time for 100 batches | 27 s | 16 s |
| Validation time (rounded) | 3 s | 2 s |
| Steady-state seconds per training batch | 0.187 | 0.082 |
| Estimated full epoch | 137 s | 61 s |
| Estimated 40-epoch training time | 1.5 h | 0.7 h |
| Peak allocated GPU memory | 4.85 GiB | 0.92 GiB |
| Validation prostate Dice after 100 batches only | 0.0000 | 0.5839 |

These are timing experiments only, not training results; the test split was not evaluated in them. The Dice row is the mean per-series 3D prostate Dice on validation. The epoch estimates exclude checkpoint/plot saving and other overhead and may vary across runs. Final training moved to Rangpur, where both models train faster.

### 6.3 Failure autopsy

**Failure cases** (Figure 3). The figure shows the prostate-containing test slices with the lowest Improved U-Net prostate Dice, at most one per series. Ties go to the larger ground-truth prostate. These slices were picked only for display, after all results were final.

![Figure 3: failure cases](figures/test_failures.png)

*Figure 3. Failure cases: five of the 22 test slices where the Improved U-Net scores 0 (at most one per series, larger ground-truth prostate first). As in Figure 2, each prediction is followed by its prostate difference overlay; a fully light-blue prostate (and "FP 0 px") means the prostate was missed entirely.*

| Series / slice | Ground-truth prostate (pixels) | Baseline Dice | Improved Dice | Position in the prostate's slice range |
| --- | ---: | ---: | ---: | --- |
| 040 / week 0 / 37 | 204 | 0.000 | 0.000 | 1 slice from the last prostate slice (16-38) |
| 042 / week 0 / 11 | 105 | 0.631 | 0.000 | 2 slices from the first prostate slice (9-40) |
| 040 / week 5 / 23 | 104 | 0.692 | 0.000 | 1 slice from the first prostate slice (22-41) |
| 040 / week 3 / 15 | 98 | 0.818 | 0.000 | 2 slices from the first prostate slice (13-32) |
| 041 / week 0 / 20 | 67 | 0.000 | 0.000 | 1 slice from the first prostate slice (19-40) |

What the failures show:

- **All failures are at the ends of the prostate.** All five shown slices, and all 22 test slices where the Improved U-Net scores 0, lie within 2 slices of the first or last slice that contains prostate. The prostate is small there (67-204 pixels in the figure, against a median of 327 over all prostate slices). No zero-Dice slice is in the middle of the gland.
- **Missed, not displaced.** On all five slices, the Improved U-Net predicts no prostate pixels at all, so these are missed detections, not shifted outlines.
- **The baseline finds some of them.** It detects the prostate on 3 of the 5 slices (Dice 0.63-0.82), which agrees with its higher slice-level Dice (0.7758 against 0.7446). Both models miss the other 2.
- **The neighbouring organs are still found.** On the largest miss (040 / week 0 / 37, Figure 3, first row), both models still segment the rectum and the bladder near the prostate, and neither predicts any prostate. Judged from the figure, the rectum prediction extends into the side of the prostate that touches the rectum (roughly a quarter to a third of it). The rest of the thin prostate strip is labelled as body. On this slice, the models find the anatomy around the prostate but do not separate the thin end of the gland from the rectum and the tissue next to it.
- **Why the 3D Dice stays high.** These edge slices contain few prostate pixels in total, so they lower the per-series 3D Dice only a little. This is why the 3D Dice stays above 0.85 while the slice-level mean is lower.

**Boundary analysis.** To test whether these failures are systematic, a boundary analysis was added to `predict.py`. It was run on Rangpur as job 685882 (node `a100-8`, NVIDIA A100-PCIE-40GB, code at commit `65b1b63`, 1 min 5 s) with the same validation-selected checkpoints; the user ran the job and copied the outputs back. The run reproduced the saved test Dice (0.8858 and 0.8738) and the same five failure cases. Its comparison and failure figures are byte-identical to those of job 685612, so the displayed slices were predicted identically. Elsewhere in the test split a few pixels differed: the recomputed baseline Dice differs from job 685612 in the sixth decimal place, which is consistent with non-deterministic GPU arithmetic. Job 690027, which drew Figures 2 and 3, printed the same boundary results as job 685882 at the printed precision; the saved full-precision values differ by less than 2e-5, again consistent with non-deterministic GPU arithmetic.

HD95 is in pixels and position in the gland is counted in slices, because the NIfTI spacing ([1, 1]) has no known units.

- **HD95 per slice:** the 95th-percentile Hausdorff distance between the predicted and ground-truth prostate boundaries. It is computed on every test slice where both contain prostate.
- **Position in the gland:** each prostate slice is grouped by its distance, in slices, from the nearest end of its series' prostate slice range: 0, 1, 2, or 3 and more. The two ends cannot be labelled base or apex, because the physical orientation of the slices is not verifiable.
- **Leakage:** the true classes of all predicted prostate pixels, and the predicted classes of all true prostate pixels, over the whole test split.

**Results by position in the gland** (195 test slices with prostate; 9 series, so 18 slices at each distance 0, 1 and 2):

| Slices from the nearest end | Slices | Mean slice Dice (baseline / Improved) | Slices with no prostate predicted (baseline / Improved) | Median HD95 in px (baseline / Improved) |
| --- | ---: | --- | --- | --- |
| 0 (end slice) | 18 | 0.154 / 0.156 | 12 / 12 | 4.00 / 4.24 |
| 1 | 18 | 0.457 / 0.308 | 5 / 8 | 3.00 / 4.33 |
| 2 | 18 | 0.754 / 0.655 | 0 / 2 | 2.02 / 3.00 |
| 3 or more (central) | 141 | 0.899 / 0.887 | 0 / 0 | 2.00 / 2.17 |
| All prostate slices | 195 | 0.776 / 0.745 | 17 / 22 | 2.06 / 2.24 |

HD95 is defined only on slices where a prostate was predicted. For example, at distance 0 it covers 6 of the 18 slices for each model.

**Leakage** (prostate pixels pooled over all 540 test slices; each block sums to 100%):

| Measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Predicted prostate pixels that are truly prostate | 88.8% | 88.1% |
| ... that are truly body tissue | 10.3% | 11.2% |
| ... that are truly bladder | 0.85% | 0.73% |
| ... that are truly rectum | 0.07% | 0.00% |
| True prostate pixels predicted as prostate | 87.7% | 86.5% |
| ... predicted as body tissue | 11.8% | 13.1% |
| ... predicted as rectum | 0.45% | 0.23% |
| ... predicted as bladder | 0.02% | 0.14% |

**Findings.**

- **Global Dice hides the end-slice failures.** On the 141 central slices, both models reach a mean slice Dice of about 0.89-0.90, never miss the prostate, and have a median HD95 of about 2 pixels. On the 18 end slices, the mean slice Dice is about 0.15, and both models miss the prostate entirely on 12 of them. Every slice where the prostate is missed lies within 2 slices of an end of the gland. These slices hold few prostate pixels, so they barely lower the per-series 3D Dice (0.87-0.89). This is the failure mode described in the dilemma. It is consistent with the PROMISE12 prostate MRI challenge (Litjens et al., 2014), which scored the base and apex thirds of the gland separately because they are the hardest parts to segment, and found that every algorithm performed worse there. PROMISE12 defines these regions as thirds of the gland, not by distance in slices from its ends as here.
- **Neither 2D model fixes the ends, and the Improved U-Net does not help there.** It is about equal to the baseline on the end slice and worse one and two slices in: mean Dice 0.31 against 0.46 and 0.66 against 0.75, with 8 against 5 and 2 against 0 missed slices. On central slices the two models are almost the same.
  - This is not a test of more spatial context. Both models see the same single slice and both downsample four times. The Improved U-Net changes several things at once: residual context modules, deep supervision, instance normalisation, leaky ReLU and about 10 times fewer parameters. So the result cannot say which change causes the near-end gap.
  - Each group has only 18 slices from 3 patients (14 of them from case 040), and each model was trained once. The near-end gap is an observation, not a tested difference.
- **Leakage into the rectum and bladder is very small overall.** Pooled over all test slices, less than 1% of predicted prostate pixels are truly bladder or rectum for either model. Less than 0.5% of true prostate pixels are given to the rectum or bladder.
  - Most misclassified prostate pixels involve the surrounding body tissue. About 10-11% of predicted prostate pixels are truly body, and 12-13% of true prostate pixels are predicted as body. The latter includes the slices where the prostate is missed entirely.
  - The central slices dominate these pooled figures, so leakage on the end slices is not measured separately. On the largest miss (Figure 3, first row), part of the gland is labelled rectum.
- **Physical scale is unknown.** With unknown spacing, a median HD95 of 2 pixels cannot be converted to millimetres or compared with clinical margins.

### 6.4 Follow-up experiment: does inter-slice context fix the ends of the gland?

The failure analysis suggests a cause: at the ends of the gland, a single 2D slice shows very little prostate. The smallest way to give the network more context is a 2.5D input (Zhang et al., 2022): the previous, current and next slices of the same series are given to the Improved U-Net as three input channels, and it predicts the middle slice. This is the first future-work direction of Section 6.6, tried with the time left before the deadline.

**Design, fixed before the run.**

- Only the input changes. The neighbouring slices come from the same series; at the first and last slice of a series the end slice is repeated. Each channel is z-scored on its own, as a 2D slice is, and all three channels get the same augmentation. Only the first convolution of the network grows (3,189,570 parameters, 288 more than the 2D model). Everything else is as in the 2D Improved U-Net run: seed 42, 40 epochs, the same optimiser, schedule and loss, and the best checkpoint chosen by validation prostate Dice.
- Three slices only. No other number of slices or setting was tried.
- The question is judged on the **validation** split, with the boundary analysis of Section 6.3. The test split is then evaluated once, whatever the validation result.
- The slice order was checked first. On validation, the mean absolute difference between normalised neighbouring slices is 0.176, against 0.460 for slices five apart, which is consistent with slices stored in anatomical order. With 2.5D input switched off, the data pipeline produces exactly the same samples as before (checked on all validation and test slices and 600 augmented training slices).
- This experiment was designed after the test failure analysis had been seen, so its test result is secondary and post hoc. The primary results in Section 5 are unchanged.

**Runs.** Training was job 690028 (node `a100-7`, code at commit `bb98a72`, 24 Oct 2026, 05:34-05:55). The validation analysis of all three models was `predict.py` job 696531 (node `a100-9`, 25 Oct 2026, 22:28-22:30), which recomputed each model's saved validation Dice to within 2e-5. The user ran the jobs and copied the outputs back.

**Validation results** (254 validation slices with prostate in 11 series, so 22 slices at each distance 0, 1 and 2; best checkpoints):

| Validation measurement | Baseline U-Net | Improved U-Net (2D) | Improved U-Net 2.5D |
| --- | ---: | ---: | ---: |
| Prostate Dice, per-series 3D mean (best epoch) | 0.8686 (28) | 0.8564 (34) | 0.8615 (11) |
| Mean ± SD over the last 10 epochs | 0.8628 ± 0.0018 | 0.8548 ± 0.0013 | 0.8534 ± 0.0018 |
| End slice (0): mean slice Dice / missed / median HD95 px | 0.256 / 10 / 3.52 | 0.300 / 8 / 4.64 | 0.265 / 10 / 4.13 |
| 1 slice from the end | 0.497 / 5 / 5.09 | 0.549 / 4 / 3.90 | 0.529 / 5 / 3.79 |
| 2 slices from the end | 0.655 / 3 / 4.00 | 0.702 / 3 / 3.27 | 0.676 / 3 / 4.00 |
| 3 or more (central, 188 slices) | 0.851 / 5 / 2.44 | 0.830 / 7 / 3.00 | 0.840 / 3 / 3.00 |
| All prostate slices: mean slice Dice / missed | 0.752 / 23 | 0.749 / 22 | 0.749 / 21 |
| Training + validation time, 40 epochs | 20.4 min | 17.1 min | 18.4 min |
| Peak training GPU memory (batch 16) | 2.32 GiB | 0.96 GiB | 0.97 GiB |
| Inference ms per slice, batch 1 / 16 (job 696531) | 3.01 / 0.62 | 7.47 / 0.47 | 7.42 / 0.47 |

"Missed" counts slices where no prostate was predicted at all. Unlike on test, a few central validation slices are also missed.

**Secondary test result.** After the validation analysis, the test split was evaluated once (job 697701, node `a100-9`, 26 Oct 2026, 04:48-04:49), and the same boundary analysis was run on test for all three models (`predict.py` job 697982, `a100-9`, 05:47-05:49; it recomputed the saved test Dice to within 3e-5). As explained above, this result is post hoc.

| Test measurement (secondary) | Baseline U-Net | Improved U-Net (2D) | Improved U-Net 2.5D |
| --- | ---: | ---: | ---: |
| Prostate Dice, per-series 3D mean (best checkpoint) | 0.8858 | 0.8738 | 0.8759 |
| Prostate Dice, last checkpoint (epoch 40) | 0.8884 | 0.8734 | 0.8774 |
| End slice (0): mean slice Dice / missed / median HD95 px | 0.154 / 12 / 4.00 | 0.156 / 12 / 4.24 | 0.138 / 13 / 6.32 |
| 1 slice from the end | 0.457 / 5 / 3.00 | 0.308 / 8 / 4.33 | 0.531 / 3 / 4.00 |
| 2 slices from the end | 0.754 / 0 / 2.02 | 0.655 / 2 / 3.00 | 0.716 / 1 / 3.00 |
| 3 or more (central, 141 slices) | 0.899 / 0 / 2.00 | 0.887 / 0 / 2.17 | 0.895 / 0 / 2.00 |
| All 195 prostate slices: mean slice Dice / missed / median HD95 px | 0.776 / 17 / 2.06 | 0.745 / 22 / 2.24 | 0.775 / 17 / 2.00 |
| Predicted prostate pixels that are truly rectum / bladder | 0.07% / 0.85% | 0.00% / 0.73% | 0.14% / 0.76% |

The baseline and 2D Improved U-Net values repeat Sections 5 and 6.3. At distance 0, HD95 covers only 5 of the 18 slices for the 2.5D model, because it predicts prostate on so few of them.

**Findings.**

- **The outermost slice is not fixed on either split.** With three-slice input, the end slice of the gland is missed slightly more often than with the 2D model: 10 against 8 slices on validation and 13 against 12 on test.
- **The two splits disagree one and two slices from the end.** On test, the 2.5D model misses fewer of these slices than the 2D Improved U-Net (3 against 8, and 1 against 2) and matches the baseline's 17 missed slices overall. On validation it does not help there (5 against 4, and 3 against 3). With 9 and 11 series and one training run, it cannot be said whether 2.5D input really helps next to the end.
- **The five failure slices of Figure 3.** On test, the 2.5D model predicts some prostate on all five slices where the 2D Improved U-Net predicted none (Dice 0.28-0.89). On validation, it does so on 3 of the 5 slices picked by the same rule (the 2D Improved U-Net's lowest-Dice slice in 5 series). These slices were chosen as the 2D model's worst cases, so a different model is likely to do better on some of them by chance alone; this is not evidence by itself.
- **On validation, the slices near the end do not improve.** On the end slice and the two slices next to it, the 2.5D model has a slightly lower mean slice Dice than the 2D Improved U-Net (0.265 against 0.300, 0.529 against 0.549, 0.676 against 0.702) and misses as many or more slices (10 against 8, 5 against 4, 3 against 3).
- **On validation, it may help in the middle of the gland.** It misses 3 central slices against 7, with a mean central Dice of 0.840 against 0.830. Over all prostate slices, the mean slice Dice is the same (0.749), with 21 against 22 missed slices. On test, no model misses a central slice (central Dice 0.895 against 0.887), and over all prostate slices the 2.5D model is higher (0.775 against 0.745; 17 against 22 missed), mostly because of the slices one and two from the end.
- **The overall Dice did not really change.** On validation, the best-epoch value is higher (0.8615 against 0.8564), but it comes from an early peak at epoch 11. Averaged over the last 10 epochs, the two runs are the same within their spread (0.8534 against 0.8548). On test, the 2.5D model scores 0.8759 against 0.8738 for the 2D Improved U-Net, still below the baseline's 0.8858.
- **The extra cost is negligible.** Training took about 8% longer (18.4 against 17.1 min), and peak training memory (0.97 against 0.96 GiB) and inference time were unchanged. Peak inference memory at batch 16 rose from 0.28 to 0.44 GiB on validation (0.26 to 0.39 GiB on test), almost entirely because the two extra input channels of the stored images are held on the GPU.
- **Limits.** Each model was trained once (one seed). The comparison rests on 11 validation series from 4 patients and 9 test series from 3 patients (7 of them from case 040). The larger first layer also changes how every later layer is randomly initialised, so the comparison includes the effect of a different starting point, much like a different seed. Validation was also used to choose each checkpoint, which makes all validation numbers slightly optimistic. These are observations, not tested differences.

### 6.5 Answer to the dilemma and recommendation for the project manager

- **Scope of the answer:** the baseline and the Improved U-Net were compared as 2D models, and a three-slice (2.5D) Improved U-Net was added as a follow-up (Section 6.4). True 3D models and dilated multi-scale networks were not tested. The evidence is that the failures are concentrated at the ends of the gland, where a single 2D slice shows very little prostate: on test every missed slice is within 2 slices of an end, and on validation 18 (baseline) and 15 (Improved) of the 23 and 22 missed slices are. Within 2D, neither model fixes them. On test the Improved U-Net is worse than the baseline one and two slices from the end; on validation it is slightly better there (mean slice Dice 0.300 / 0.549 / 0.702 against 0.256 / 0.497 / 0.655 at distances 0 / 1 / 2) and slightly worse in the centre (0.830 against 0.851), so the near-end difference between the two 2D models is not consistent across splits. Adding the two neighbouring slices did not fix the outermost slice either; one and two slices in, the validation and test splits disagree.
- **Accuracy:** on this test split, the baseline 2D U-Net is slightly better: test prostate Dice 0.8858 against 0.8738, and fewer misses near the ends of the gland (on validation the Improved U-Net misses fewer there, Section 6.4). Both gaps come from one seed and 3 test patients and were not tested for significance, so they favour the baseline without proving it is more accurate.
- **Resources:** the Improved U-Net is far cheaper.
  - It has about 10 times fewer parameters and uses about 60% less training memory.
  - At batch size 16 it takes about 27% less time per slice (0.46 against 0.63 ms).
  - On single slices it is about 2.4 times slower (7.5 against 3.1 ms per slice).
- **Recommendation:**
  - Where GPU memory allows, prefer the baseline 2D U-Net, which scored higher on the primary metric and on every foreground class.
  - Choose the Improved U-Net only when memory or batched throughput is the binding constraint.
  - With either model, a contouring pipeline should send the outermost predicted prostate slice at each end, and the two to three slices beyond it, for manual review. On test, both models miss the true outermost slice in 12 of 18 cases, and every missed slice is within 2 slices of a true end (within 1 for the baseline). On validation (Section 6.4), the outermost slice was missed in 10 (baseline) and 8 (Improved) of 22 cases, the baseline also missed 3 slices two from the end, and both models missed a few central slices (5 and 7 of 188), so this end-slice review would not catch every miss. How far the predicted range falls short of the true range was not measured, and the margin was first read from the test split, so it should be confirmed on new data before use.
- **Do not expect three-slice input to fix the ends.** In the 2.5D follow-up it cost almost nothing, but the outermost slice was missed slightly more often than before on both splits, and the evidence one and two slices in is contradictory (Section 6.4). On the present evidence it does not remove the need for the manual review of the end slices above.
- **Next experiments:** repeat the 2.5D comparison with several seeds and new held-out data to settle the slices next to the end, and test a true 3D model, which sees the whole gland, with the same distance-from-end breakdown, HD95 and measured memory cost. Section 6.6 lists this and two other directions.

### 6.6 Future work

These directions follow from the failure analysis. Apart from the 2.5D input of Section 6.4, none of them was tried in this project, so they are proposals, not results.

1. **Inter-slice context (first priority).** The misses are confined to the ends of the gland, where one slice shows little prostate. The smallest change is a 2.5D input: the previous, current and next slices as three input channels, predicting the middle slice. The larger one is a 3D U-Net (Çiçek et al., 2016), which applies the same encoder-decoder design to whole volumes. The 2.5D variant is not a method from that paper. Zhang et al. (2022) compared such designs on the Medical Segmentation Decathlon prostate data (32 MR volumes): a 3-slice input raised Dice from 0.7222 (2D) to 0.7460, against 0.7677 for a 3D U-Net, and 5 or 7 slices gave no further gain. Their data (multimodal T2 and ADC volumes) and label handling differ from HipMRI, so this motivated the experiment but did not predict its effect here. The 2.5D input was tried in Section 6.4: it did not fix the outermost slice, and its effect one and two slices in differed between validation and test. The 3D U-Net remains untested and costs more memory.
2. **Boundary-aware loss.** The boundary loss of Kervadec et al. (2019) is a distance measure on contours rather than a sum over regions, and it was proposed for highly unbalanced segmentation. Added to the Dice loss, it could target boundary errors on small prostate areas without adding parameters. It does not add inter-slice information. Its weight relative to the Dice loss, and its handling of slices without prostate, would have to be set on validation.
3. **Attention gates.** Attention U-Net (Oktay et al., 2018) adds gates to the skip connections that learn to suppress irrelevant regions and emphasise features useful for the task. It was shown on abdominal CT (pancreas), so its benefit on pelvic MRI is untested. Adding it would make a third model, not the Improved U-Net of Isensee et al.

Each variant should be compared under the same rules as in this report: settings chosen on validation only, and evaluation by prostate Dice, missed slices by distance from the ends of the gland, HD95 and measured memory cost. The test split of this project has already been used, so confirming an improvement needs new held-out data that has not informed any decision.

## 7. Reproducibility and usage

### Dependencies

These are the versions the code was checked with locally and the versions installed on Rangpur. Each training run also records its Python and PyTorch versions in `config.json`.

| Package | Local check | Rangpur |
| --- | --- | --- |
| Python | 3.12.13 | 3.12.13 |
| PyTorch | 2.13.0 (CUDA 13.0 build) | 2.13.0+cu130 |
| torchvision | 0.28.0 | 0.28.0+cu130 |
| NumPy | 2.5.2 | 2.5.1 |
| nibabel | 5.4.2 | 5.4.2 |
| matplotlib | 3.11.1 | 3.11.2 |

All Rangpur jobs ran on NVIDIA A100-PCIE-40GB GPUs. Node `a100-6` (interactive check, job 683570) reported driver 590.48.01 (CUDA 13.1), and a test GPU operation succeeded there.

### Usage

`--data-root` points to the supplied `keras_slices_data` folder.

```bash
pip install torch==2.13.0 torchvision==0.28.0 numpy matplotlib nibabel==5.4.2   # pick the CUDA build for your GPU
python dataset.py --data-root keras_slices_data                      # pipeline self-check
python train.py --model unet --epochs 1 --max-train-batches 100      # timing run, no test
python train.py --model improved --epochs 40 --out-dir runs/improved # full training; review the curves next
python train.py --model improved --out-dir runs/improved --resume --test   # one-time test after a completed 40-epoch run
python predict.py --runs-dir runs   # after both models are tested; writes runs/predict_best/
# 2.5D experiment: the previous, current and next slices as three input channels
python train.py --model improved --context-slices 1 --out-dir runs/improved_25d
python predict.py --runs-dir runs --split validation --models unet improved improved_25d
```

`--context-slices` defaults to 0, which is the plain 2D input used for the main results; with 0 the data pipeline produces exactly the same samples as before the option was added. `predict.py --split validation` runs the same figures and boundary analysis on the validation split, so the 2.5D run can be judged without the test split.

On Rangpur, `train.py` runs inside a Slurm job (`--partition=comp3710 --account=comp3710 --gres=gpu:1`) and reads the course copy of the data directly:

```bash
python train.py --model improved --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data --out-dir ~/hipmri_runs/improved
```

The test split is evaluated only with `--test`, after the full epoch budget. It is evaluated once for the best-validation checkpoint and once for the last checkpoint. Shortened runs refuse `--test`, and existing test results are never overwritten. After both tests, `predict.py` reloads the best-validation checkpoints once more to draw figures, recheck the saved test Dice and time inference. It does not select checkpoints or change any setting.

### Determinism

- The seed is 42. The shuffle order and augmentation of epoch e are seeded with 42 + e, so a resumed run sees the same data as an uninterrupted one.
- `cudnn.benchmark` is on and mixed precision is used, so reruns are not bit-for-bit identical. Reloading the checkpoints on other A100 nodes reproduced the test prostate Dice to four decimal places (Sections 5 and 6.3).
- Each model was trained once (one seed), so seed-to-seed variation was not measured.

### Run provenance

Every Rangpur result in this report comes from these runs. The user submitted all Slurm jobs and copied the outputs back. The other reported numbers come from the Colab T4 timing runs of 21 October 2026, run by the user (Section 6.2), from the local full-dataset audit of 21 October 2026, run by the AI (Sections 1 and 2.6), and from the local model-free checks of the 2.5D data option on 23 October 2026, run by the AI (slice order and unchanged 2D samples, Section 6.4). Times are in AEST (UTC+10).

| Run | Slurm job | Node | Code commit | Date and time |
| --- | --- | --- | --- | --- |
| Data pipeline check (interactive) | 683570 | `a100-6` | `92f51d4` | 22 Oct 2026, about 01:00 |
| Training, Improved U-Net | 683615 | `a100-6` | `92f51d4` | 22 Oct 2026, 01:11-01:31 |
| Training, baseline U-Net | 683616 | `a100-6` | `92f51d4` | 22 Oct 2026, 01:31-02:00 |
| Test, Improved U-Net | 683765 | `a100-5` | `92f51d4` | 22 Oct 2026, 03:36-03:38 |
| Test, baseline U-Net | 683766 | `a100-5` | `92f51d4` | 22 Oct 2026, 03:38-03:39 |
| `predict.py`: figures, consistency check, inference timing | 685612 | `a100-4` | `b7d4fba` | 22 Oct 2026, 16:46-16:47 |
| `predict.py`: boundary analysis | 685882 | `a100-8` | `65b1b63` | 22 Oct 2026, 17:49-17:50 |
| `predict.py`: figures with difference overlays (Figures 2-3) | 690027 | `a100-7` | `bb98a72` | 24 Oct 2026, 05:33-05:34 |
| Training, Improved U-Net 2.5D (Section 6.4) | 690028 | `a100-7` | `bb98a72` | 24 Oct 2026, 05:34-05:55 |
| `predict.py`: validation analysis of all three models | 696531 | `a100-9` | `bb98a72` | 25 Oct 2026, 22:28-22:30 |
| Test, Improved U-Net 2.5D (secondary) | 697701 | `a100-9` | `bb98a72` | 26 Oct 2026, 04:48-04:49 |
| `predict.py`: test analysis of all three models | 697982 | `a100-9` | `bb98a72` | 26 Oct 2026, 05:47-05:49 |

### Data and model storage

The HipMRI dataset and the saved model weights are kept outside Git version control. The checkpoints (`best.pt` and `last.pt` for each model) are stored on Rangpur and in a local copy outside this repository.

## 8. Files

- `dataset.py`: image-mask pairing, the case-ID split check, preprocessing, paired augmentation and DataLoaders, with an optional 2.5D input of neighbouring slices (`context_slices`).
- `modules.py`: the standard 2D U-Net baseline, the 2D Improved U-Net and the multiclass soft Dice loss.
- `metrics.py`: per-series 3D Dice and the other fixed evaluation rules.
- `train.py`: training, validation, checkpoint selection, saving, loss/metric curves and the one-time test evaluation; `--context-slices` selects the 2.5D input.
- `predict.py`: reloads the validation-selected checkpoints of both models after the test evaluation. It draws side-by-side test predictions and failure cases, each prediction followed by a prostate difference overlay (true positives, false positives and false negatives), rechecks the test prostate Dice, times inference and computes the boundary analysis (HD95, results by distance from the ends of the gland, and leakage into neighbouring classes). It does not choose a checkpoint or tune anything; the test predictions are used only to pick which slices to display. `--models` and `--split validation` compare other runs, such as the 2.5D experiment, on the validation split.
- `LABEL_MAPPING.md`: the source and verification of the label mapping (5 = prostate).
- `figures/`: the architecture diagram (Figure 1), training curves, test predictions and failure cases. The training curves come from `train.py` and the prediction figures from `predict.py`, both on Rangpur. The architecture diagram was drawn with a matplotlib script kept outside this repository, from the layer definitions in `modules.py`, without building or running a model.

## 9. Development timeline

The commit log records the work as it happened. The main milestones were:

| Date (2026) | Commit | Milestone |
| --- | --- | --- |
| 5 Oct | `3a9175c` | Project folder, documentation and ignore rules |
| 7-11 Oct | - | Feasibility review written (7 Oct) and check-off completed (by 11 Oct) |
| 21 Oct | `1c9f1e5`, `58b014a` | Prostate label mapping found in the publisher's documentation |
| 21 Oct | `9972ea0` | Data loader with the fixed preprocessing rules |
| 21 Oct | `688ea70` | Both models, the Dice loss, the metrics and the training script |
| 21 Oct | `92f51d4` | Colab T4 pipeline checks and timing runs |
| 22 Oct | `af1d8f7` | Rangpur environment and data pipeline check |
| 22 Oct | `8f1a8d5` | 40-epoch training and validation results |
| 22 Oct | `b7d4fba` | `predict.py` for figures, failure cases and inference timing |
| 22 Oct | `bb4ab23` | Test results, prediction figures, failure cases and inference timing |
| 22 Oct | `65b1b63` | Feasibility review added to the README; HD95 boundary analysis |
| 22 Oct | `33bd8be` | Boundary analysis, dilemma answer and recommendation |
| 23 Oct | `df8e3a0` | README reorganised into the final report; architecture figure |
| 23 Oct | `775d8c1` | Future-work section (now Section 6.6) with checked references |
| 23 Oct | `6552d29` | Related published work (PROMISE12, Rodrigues et al., Zhang et al.) and the exact metric definition |
| 23 Oct | `bb98a72` | Optional 2.5D input, difference overlays in `predict.py`, validation-split analysis |
| 26 Oct | `d940879` | Figures with difference overlays; 2.5D follow-up experiment with validation and secondary test results (Section 6.4) |

## 10. Acknowledgements and references

Acknowledgement: data were provided by the HipMRI Study ([CSIRO data collection](https://doi.org/10.25919/45t8-p065)), through the processed 2D slices supplied for COMP3710.

- Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox and O. Ronneberger, "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", in *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2016*, Lecture Notes in Computer Science, Springer, 2016, pp. 424-432. <https://doi.org/10.1007/978-3-319-46723-8_49>
- J. Dowling and P. Greer, "Labelled weekly MR images of the male pelvis", CSIRO Data Collection, 2021. <https://doi.org/10.25919/45t8-p065>
- J. A. Dowling, J. Sun, P. Pichler, D. Rivest-Hénault, S. Ghose, H. Richardson, C. Wratten, J. Martin, J. Arm, L. Best, S. S. Chandra, J. Fripp, F. W. Menk and P. B. Greer, "Automatic Substitute Computed Tomography Generation and Contouring for Magnetic Resonance Imaging (MRI)-Alone External Beam Radiation Therapy From Standard MRI Sequences", *International Journal of Radiation Oncology, Biology, Physics*, 93(5), 1144-1153, 2015. <https://doi.org/10.1016/j.ijrobp.2015.08.045>
- F. Isensee, P. Kickingereder, W. Wick, M. Bendszus and K. H. Maier-Hein, "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge", in *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries* (BrainLes 2017), Lecture Notes in Computer Science, Springer, 2018, pp. 287-297. <https://doi.org/10.1007/978-3-319-75238-9_25>
- H. Kervadec, J. Bouchtiba, C. Desrosiers, E. Granger, J. Dolz and I. Ben Ayed, "Boundary loss for highly unbalanced segmentation", in *Proceedings of the 2nd International Conference on Medical Imaging with Deep Learning* (MIDL 2019), Proceedings of Machine Learning Research, vol. 102, 2019, pp. 285-296. <https://proceedings.mlr.press/v102/kervadec19a.html>
- G. Litjens, R. Toth, W. van de Ven, C. Hoeks, S. Kerkstra, B. van Ginneken, G. Vincent, G. Guillard, N. Birbeck, J. Zhang, R. Strand, F. Malmberg, Y. Ou, C. Davatzikos, M. Kirschner, F. Jung, J. Yuan, W. Qiu, Q. Gao, P. E. Edwards, B. Maan, F. van der Heijden, S. Ghose, J. Mitra, J. Dowling, D. Barratt, H. Huisman and A. Madabhushi, "Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge", *Medical Image Analysis*, 18(2), 359-373, 2014. <https://doi.org/10.1016/j.media.2013.12.002>
- O. Oktay, J. Schlemper, L. Le Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, B. Glocker and D. Rueckert, "Attention U-Net: Learning Where to Look for the Pancreas", arXiv:1804.03999, 2018 (accepted to MIDL 2018). <https://arxiv.org/abs/1804.03999>
- N. M. Rodrigues, S. Silva, L. Vanneschi and N. Papanikolaou, "A Comparative Study of Automated Deep Learning Segmentation Models for Prostate MRI", *Cancers*, 15(5), 1467, 2023. <https://doi.org/10.3390/cancers15051467>
- O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", in *Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015*, Lecture Notes in Computer Science, Springer, 2015, pp. 234-241. <https://doi.org/10.1007/978-3-319-24574-4_28>
- Y. Zhang, Q. Liao, L. Ding and J. Zhang, "Bridging 2D and 3D segmentation networks for computation-efficient volumetric medical image segmentation: An empirical study of 2.5D solutions", *Computerized Medical Imaging and Graphics*, 99, 102088, 2022. <https://doi.org/10.1016/j.compmedimag.2022.102088>

## Artificial Intelligence Usage Disclosure

AI assistance (Anthropic Claude, through Claude Code) was used for the following:

- locating the publisher's online documentation;
- preparing and executing the mask-comparison and dataset-audit code;
- drafting the data/metric policy;
- drafting `dataset.py`, `modules.py`, `metrics.py`, `train.py` and `predict.py`;
- preparing the Rangpur environment-setup and Slurm job scripts (kept outside this repository);
- drafting this documentation, including its reorganisation into the final report structure (23 October 2026);
- suggesting, at the user's request for further experiments, the difference overlays and the 2.5D follow-up of Section 6.4, and drafting their code and pre-specified evaluation plan (23 October 2026; the user chose which to do);
- writing the matplotlib script that draws the architecture diagram (Figure 1), kept outside this repository.

The AI-suggested code was verified in four ways:

- running the data pipeline over every slice and comparing its counts with an independent audit;
- tracing the architectures against the papers' text and Fig. 1;
- comparing the metrics with an independent NumPy reference on synthetic data;
- running synthetic end-to-end training/resume/test checks on random data (on CPU, and for the 2.5D option also on the local laptop GPU);
- for the 2.5D option, checking on the local copy of the data, without any model, that the 2D samples are unchanged and that the neighbouring slices come from the right series.

OpenAI Codex also assisted with the Colab execution guidance and the Colab T4 timing update (21 October 2026), using the console outputs supplied by the user. OpenAI ChatGPT suggested the papers behind the future-work directions in Section 6.6, and Perplexity suggested the related papers cited in Sections 6.1, 6.3 and 6.6 (Litjens et al., Rodrigues et al., Zhang et al.) (23 October 2026). Claude Code checked those references and the cited numbers against the papers and Crossref, arXiv, PMC and PMLR, and drafted the text. Claude Code summarised the Rangpur run files supplied by the user for the training, test, prediction, boundary-analysis and 2.5D results, and described the figures. It also drafted the answer to the research dilemma and the engineering recommendation, which the user reviewed. The HD95 code was checked against a NumPy brute-force reference on synthetic masks, and the figure files of `predict.py` jobs 685612 and 685882 were byte-compared locally, as were the repository copies of Figures 2-3 with the job 690027 outputs. The architecture diagram was checked against the layer definitions in `modules.py`.

All training, validation, test and prediction results (the Colab timing runs and the Rangpur jobs) come from runs the user executed. The AI did not train, evaluate or run inference with the reported models. It ran only local checks: the mask comparison, the dataset audit, the model-free data checks of the 2.5D option, the synthetic checks on random data (including the HD95 reference check), a layer and parameter count of the untrained models, the byte-comparison of the `predict.py` figure files, and the script that draws Figure 1 (which builds no model). The AI derived summary figures such as gaps, counts and pixel medians from the result files the user supplied, and the derived per-series inference times in Section 6.2.
