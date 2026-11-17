# Alzheimer's classification with ConvNeXt

**COMP3710 — Semester 2, 2026**  
**Name:** Angus Walker  
**Student ID:** s4963173

## Overview

I compared a **ConvNeXt-Tiny trained from scratch** with a smaller convolutional neural network (CNN) to classify structural brain MRI images as Alzheimer's disease (AD) or cognitively normal (CN). I also tested whether uncertain *patient-level* predictions could be referred for human review. ADNI calls the normal class `NC`; the label `CN` is used throughout this README.

**The main result was not what I had hoped for.** ConvNeXt was more complex and expensive to run, but it did not beat the CNN on the required individual-image (slice) accuracy measure. Neither model reached the **80% target**.

| Held-out test result | ConvNeXt A1 | Regularized CNN |
|---|---:|---:|
| Correct MRI slices | **3,157 / 4,640** | **3,193 / 4,640** |
| **Slice accuracy (primary measure)** | **68.04%** | **68.81%** |
| Slice macro-F1 | 0.6769 | 0.6869 |
| Slice AD recall | 54.43% | 58.98% |
| Correct patients (supplementary) | 76 / 102 | 79 / 102 |
| Patient accuracy | 74.51% | 77.45% |

A1 (`scratch_augmented`, seed 42, epoch 13) remained my final ConvNeXt checkpoint because it performed best under my development selection rule. It correctly classified 3,157 test slices, **555 short** of the 3,712 needed for 80% accuracy. A1 detected one more AD patient after aggregation, but missed more AD *slices* and cost much more to run. Its referral rules also made confident mistakes, including labelling AD patients as CN. I therefore regard it as a **research prototype, not a diagnostic or screening system**.

I originally selected checkpoints using **patient-level validation**. The course later clarified that accuracy is assessed per **individual 2D slice**, but those earlier selections were not changed retrospectively. Version 1 had also evaluated the same test patients; although the revised models and rules were fixed before their later test run, this cannot be called an untouched cohort. I account for that exposure, repeated validation use and the small number of independent patients in my conclusions.

## 1. Feasibility and dataset

In my feasibility proposal, I planned to access the dataset through Rangpur, work within a 12 GiB GPU-memory cap, and start with a 12 GPU-hour budget. In practice, the experiments ran locally on an **NVIDIA RTX 4080 Laptop GPU**. During my prac, approval was received for the corrected data split described below.

The data audit found **216 patients in both supplied train and test folders**. This risked leakage, so a corrected split was used to keep every scan, visit and slice from each patient together. I matched JPEG prefixes to exact metadata keys and verified patient tokens across seven metadata path fields instead of assuming prefixes were patient IDs. Metadata codes 0 and 2 mapped to CN=0 and AD=1.

My local dataset contained **30,520 JPEG slices from 1,526 scans and 680 mapped patients**. A deterministic, class-stratified split of approximately 70% training, 15% validation and 15% testing was used, based on SHA-256 ordering with seed 42. I saved the assignments in manifests so the data loaders would follow the same partitions each time.

| Split | Patients (CN / AD) | Scans | Slices |
|---|---:|---:|---:|
| Train | 476 (321 / 155) | 1,061 | 21,220 |
| Validation | 102 (69 / 33) | 233 | 4,660 |
| Test | 102 (69 / 33) | 232 | 4,640 |

I verified that **no patients overlapped between splits**. The local audit found no unreadable images or exact duplicates in the inspected copy, but could not exclude near-duplicates, dataset bias or differences from an independently hashed Rangpur copy. Splitting by patient prevents direct patient leakage; slices from the same patient remain correlated.

### Image preparation

To prepare the images without distorting their shape, I resized each **256 × 240** greyscale JPEG to **224 × 210** using bilinear interpolation, then added seven black pixels above and below it. This produced a **1 × 224 × 224** input. I converted the pixels to float32 and normalized them using `(pixel / 255 - 0.5) / 0.5`, giving values in **[-1, 1]**. These are normalized JPEG pixel values, not physical MRI intensity measurements.

For A1, I added small random **±5° rotations during training** to introduce some variation and potentially reduce overfitting. I checked a private preview for possible cropping near the image edges. Validation and test images received no random augmentation, and no samples were removed for being difficult to classify.

## 2. Models and evaluation

### ConvNeXt-Tiny

I used a custom PyTorch **ConvNeXt-Tiny** with one greyscale input channel, two output classes and **no pretrained weights**. The stride-4 stem produces 96 channels. Four stages contain **3, 3, 9 and 3 blocks**, increasing channels **96 → 192 → 384 → 768** while reducing spatial resolution **56 → 28 → 14 → 7**. Each block combines a 7 × 7 depthwise convolution, LayerNorm, a fourfold expanding linear bottleneck, GELU, LayerScale and a residual connection. Stochastic depth provided regularization. Spatial averaging, LayerNorm and a two-logit head produce class scores. The model has **27,818,594 trainable parameters**.

### CNN baseline

A simpler baseline was also needed to judge whether ConvNeXt's extra complexity was worthwhile. My CNN baseline has four stride-2 convolution/BatchNorm/ReLU blocks with **16, 32, 64 and 128 channels**. It pools to a **4 × 4** feature map before a 128-unit dense layer, ReLU, dropout and a two-class output head. It has **359,922 parameters**. I compared both architectures using the same data splits, preprocessing and evaluation rules, while allowing their model-specific regularization to differ.

### What is being measured?

I measured **slice-level accuracy** because the assessment requires a decision for each individual image. Each slice gets a softmax AD score, classified at **0.5**. Supplementary scan and patient results were calculated by averaging slice probabilities within each scan, then averaging scans within each patient. This weights scans equally for a patient. I kept these measures separate because training on slices and judging aggregated patients answer different questions.

I used **accuracy** for overall correctness, **precision and recall** to see how well each class was predicted and detected, and **macro-F1** to give AD and CN equal importance. **AUROC** measures how well the scores separate the classes across different thresholds. I also tracked **negative log-likelihood (NLL)** and **Brier score** to assess the quality of predicted probabilities; for both, lower is better. A model's AD score was not treated as a clinically validated probability of disease.

## 3. Training and development experiments

### Selected training recipe

For my selected A1 training setup, unweighted cross-entropy was used, along with AdamW and float32 computation. Each physical batch contained **8 slices**, with gradients accumulated over **4 batches** before an optimizer update. I warmed up the learning rate for three epochs to a peak of **0.0003**, then used cosine decay to **0.000001 at epoch 30**. Weight decay was **0.05** for matrices/kernels and zero for biases and one-dimensional parameters, with gradient-norm clipping at **5**. Up to **30 epochs** were allowed, required at least **12**, and used early stopping with patience **8**. The checked deterministic/recovery setup used zero data-loader workers.

Checkpoints were initially selected by **validation-patient macro-F1**, breaking ties with lower NLL. In Version 2, the priority changed to **patient accuracy, then macro-F1, then ordinary patient NLL**, using the earliest checkpoint for an exact tie. I made these choices to guide development; they were **not prescribed by the course**. In hindsight, selecting by patient-level results was a limitation once the required slice-level accuracy measure was clarified.

### Main development results

I compared the following experiments using **the same 102 validation patients**. These results helped me decide what to investigate next; they are not independent test measurements.

| Experiment | Selected epoch | Patient accuracy | AD recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Initial CNN | 7 | 78.43% | 51.52% | 0.7292 |
| Regularized CNN | 14 | 78.43% | 63.64% | 0.7496 |
| Original ConvNeXt | 11 | 76.47% | 54.55% | 0.7167 |
| **A1, seed 42** | **13** | **83.33%** | **66.67%** | **0.8012** |
| A2, patient weighting | 12 | 80.39% | 63.64% | 0.7683 |
| A1, seed 43 | 8 | 78.43% | 63.64% | 0.7496 |
| A1, seed 44 | 7 | 78.43% | 69.70% | 0.7574 |
| S1, label smoothing | 9 | 78.43% | 57.58% | 0.7403 |
| S2, smoothing + schedule | 9 | 78.43% | 63.64% | 0.7496 |

**A1 (seed 42)** led under my patient-validation rule at **83.33%**, but seeds 43 and 44 did not reproduce it. All three A1 seeds misclassified the same **11 patients, including 9 with AD**. Nearly perfect training-slice accuracy alongside increasing validation-slice loss suggested overfitting. Patient weighting (A2) did not improve the selected result; smoothing reduced some confident errors without improving classification. Because development versions differed in more than rotation, I could not credit rotation alone for the changes.

### Follow-up studies

**Learning-rate schedule.** I tested whether faster learning-rate decay would improve generalization. My predeclared study changed only the cosine endpoint from epoch 30 to **12**, for seeds 42, 43 and 44. The runs classified **81, 76 and 81** validation patients correctly, versus **85, 80 and 80** for matched controls. Their combined **238** correct decisions missed the predefined **251** target; only one seed improved. Mean AD recall fell **66.67% → 60.61%**. Mean patient NLL improved (**0.5422 → 0.5074**), but that did not outweigh the classification loss, so I **retained A1**. Runs completed with bounded recovery where needed and passed replay checks. See the [schedule plan](SCHEDULE_ONLY_PLAN.md).

**Scan-level supervision and attention.** Another question was whether learning from complete scans would improve patient predictions. In **G0**, slice cross-entropy was kept but batches were grouped by full scan. **G1** trained on the log of the mean correct-class slice probability, while **G2** applied cross-entropy to averaged slice logits. **G3** added a gated-attention network over the slice-head inputs so the model could assign different weights to different slices. G2 served as the matched uniform-weight control for G3.

| Study (seed 42) | Validation patients correct / 102 | AD patients detected / 33 | Selected slice-head accuracy |
|---|---:|---:|---:|
| G0 | 75 | 21 | 69.33% |
| G1 | 69 | 0 | 55.79% |
| G2 | 81 | 27 | 66.01% |
| G3 | 73 | 24 | 59.46% |

None passed my original patient-based screen of **at least 86 correct patients, 22 AD detections and A1's macro-F1**. G3 learned nonuniform attention weights but still performed worse than G2 in this configuration. G1's lower training loss did not mean better accuracy because the loss functions differ. I ruled out these specific alternatives to A1, not all scan-based methods. Protocols are in the [G012 plan](g012/G012_PLAN.md), [G3 plan](g3/G3_PLAN.md) and [control comparison](g3/G3_CONTROL_COMPARABILITY.md). Recorded active-time charges for G0/G1/G2/G3 were **7,026.5 / 3,400.0 / 8,178.0 / 6,222.8 seconds** including replay/recovery. Different epoch counts and timing scopes limit direct comparisons.

### Retrospective slice-validation check

After the course clarified that **slice accuracy was the primary measure**, I reviewed the saved validation predictions to see whether an already-trained checkpoint offered a meaningful improvement. The checkpoints previously selected using patient-level results were **not** changed. A1 correctly classified **3,364/4,660 slices (72.19%)**, the highest result among the **16 selected checkpoints** reviewed. The regularized CNN achieved **68.33%**, S2 **71.03%**, schedule seed 42 **71.74%**, and G0/G1/G2/G3 **69.33% / 55.79% / 66.01% / 59.46%**.

I set a quick-improvement target of **3,597/4,660 correct validation slices (77.19%)**, without reducing macro-F1 or AD recall. No retained result passed. Some runs had a higher logged accuracy at a different epoch, but the weights or full metrics were unavailable. Many older histories recorded validation loss rather than per-epoch slice accuracy, so it was not possible to establish the best slice-accuracy epoch. I kept A1 rather than claim an unverified improvement. Details are in the [slice review](SLICE_ALIGNMENT_REVIEW.md), [CSV](slice_alignment_metrics.csv) and [JSON](slice_alignment_metrics.json).

![Selected validation slice accuracies](figures/slice_validation_alignment.png)

![A1 training and validation loss](figures/a1_training_loss.png)

![A1 patient-validation trajectory](figures/a1_validation_trajectory.png)

I have kept the original A1 learning plots. Any “test not evaluated” label describes when the figures were produced, before the revised test evaluation.

## 4. Confidence and referral policies

I also investigated whether uncertain **patient-level** predictions could be referred for review. Temperature scaling was fitted to validation-patient log odds, giving temperatures of about **1.61343 for CNN** and **1.85536 for A1**. This adjusts confidence without changing class decisions at 0.5. Five-fold cross-fitting tested referral-rule stability, although checkpoint selection had already used the full validation cohort. These are **not slice-level referral results**.

Before revised test inference, the following three referral policies were fixed:

| Policy | Decision rule |
|---|---|
| `reference` | Accept a prediction only when calibrated confidence reaches **0.85 (CNN)** or **0.75 (A1)**; otherwise refer. |
| `strict_cn` | Apply `reference`, but automatically propose CN only when calibrated **P(AD) ≤ 0.05**. |
| `no_auto_cn` | Refer every CN prediction; keep AD flags as in `reference`. |

AD flags were treated as requiring confirmation and referrals as requests for human review, **not diagnoses**. On validation, A1's full-fit `reference` rule accepted **48/102 patients with 4 errors**. With cross-fitting, it accepted only **20/102**, still with **4 errors**; the CNN accepted **27/102** with **4 errors**. This difference made me cautious about any claimed reduction in clinical workload. I didn't establish calibrated slice-level referral rules, and I didn't measure actual specialist-review outcomes.

## 5. Held-out test results

I report the figures from my **previously saved revised test evaluation** below; test set was not rerun for this README. The original Version 1 ConvNeXt had already been evaluated on these patients, reaching **72.55% test-patient accuracy**. Therefore, the later results were interpreted with the earlier test exposure in mind.

### Primary result: individual MRI slices

| Metric | ConvNeXt A1 | CNN |
|---|---:|---:|
| Correct / 4,640 | 3,157 | 3,193 |
| **Accuracy** | **68.04%** | **68.81%** |
| AD precision / recall | 78.72% / 54.43% | 76.81% / 58.98% |
| CN precision / recall | 61.86% / 83.39% | 63.32% / 79.91% |
| Macro-F1 | 0.676945 | 0.686915 |
| AUROC | 0.779813 | 0.776140 |
| Raw NLL | 1.928413 | 2.019384 |
| **TN / FP / FN / TP** | **1818 / 362 / 1121 / 1339** | **1742 / 438 / 1009 / 1451** |

I treated AD as the positive class and used a **0.5** threshold; TN/FP/FN/TP mean true negatives, false positives, false negatives and true positives. Compared with CNN, A1 made **76 fewer false AD predictions on CN slices** but **112 more missed AD slices**, leaving it correct on **36 fewer slices overall**. Its slightly higher AUROC and lower NLL did not outweigh weaker accuracy and macro-F1. The available summary did **not record test-slice Brier score**.

### Supplementary scan and patient results

| Measure | ConvNeXt A1 | CNN |
|---|---:|---:|
| Correct scans / 232 | 158 (68.10%) | 168 (72.41%) |
| Correct patients / 102 | 76 (74.51%) | 79 (77.45%) |
| Patient macro-F1 | 0.7040 | 0.7258 |
| Patient AUROC | 0.8142 | 0.8177 |
| Patient AD recall | 57.58% | 54.55% |
| Patient CN recall | 82.61% | 88.41% |
| Patient TN / FP / FN / TP | 57 / 12 / 14 / 19 | 61 / 8 / 15 / 18 |
| Raw → calibrated NLL | 0.6040 → 0.4979 | 0.5138 → 0.4839 |
| Raw → calibrated Brier | 0.1780 → 0.1661 | 0.1655 → 0.1607 |

At the patient level, A1 scored **2.94 percentage points lower** in accuracy than the CNN. The exploratory paired patient-bootstrap interval for that difference was **−10.78 to 4.90 percentage points**, which does not account for different training seeds, clinical sites or earlier test exposure. Temperature scaling improved the recorded patient NLL and Brier scores without changing any decisions. I report these supplementary patient results separately so they are not confused with the required slice accuracy.

### Patient referral trade-offs on the test cohort

| Model | Policy | Auto CN | AD flags | Referrals | AD patients incorrectly auto-CN | Error among accepted | Estimated clinical reviews* |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | `reference` | 28 | 4 | 70 | 0/33 | 3.12% | 74 |
| CNN | `strict_cn` | 15 | 4 | 83 | 0/33 | 5.26% | 87 |
| CNN | `no_auto_cn` | 0 | 4 | 98 | 0/33 | 25.00% | 102 |
| A1 | `reference` | 50 | 6 | 46 | **5/33** | 14.29% | 52 |
| A1 | `strict_cn` | 26 | 6 | 70 | **1/33** | 12.50% | 76 |
| A1 | `no_auto_cn` | 0 | 6 | 96 | 0/33 | 50.00% | 102 |

*I calculated estimated clinical reviews as referrals **plus** confirmation of every AD flag. These are workload estimates, not measurements from a clinical setting. Accepted error is calculated only among automatically accepted predictions.*

I found a clear trade-off in A1's referral rules. Switching from `reference` to `strict_cn` reduced the number of AD patients incorrectly given an automatic CN decision from **five to one**, but increased referrals from **46 to 70**. With `no_auto_cn`, no CN predictions were automatically cleared, but every one of the **102 patients** would need review once AD flags were included. I saw this as a reason to be cautious, not evidence that the model was clinically safe.

![Revised patient ROC](figures/a1_revised_test_roc.png)

![Reliability after fixed patient temperature scaling](figures/a1_revised_test_reliability.png)

## 6. Five illustrative errors

I examined five previously reviewed **patient-level errors** using saved results and derived image panels. Their displayed slice indices and rounded scores matched the records, but that does **not** establish anatomical causes. F01, F04 and F05 are different patients from the older worksheet with the same labels. Each panel shows three slices, whereas the patient score uses **all** recorded scans and slices. I did not reopen raw images for this rewrite, and medical-image publication permission has not been established.

| Case | True label | A1 patient prediction | A1 raw → calibrated P(AD) | CNN prediction; calibrated P(AD) | A1 `reference` → `strict_cn` → `no_auto_cn` |
|---|---|---|---|---|---|
| **F01** | AD | CN | 0.0019 → 0.0331 | CN; 0.2284 | auto-CN → auto-CN → refer |
| **F02** | CN | AD | 0.9962 → 0.9525 | AD; 0.9760 | AD flag → AD flag → AD flag |
| **F03** | CN | AD | 0.9532 → 0.8353 | CN; 0.3925 | AD flag → AD flag → AD flag |
| **F04** | CN | AD | 0.5295 → 0.5159 | CN; 0.4303 | refer → refer → refer |
| **F05** | AD | CN | 0.0193 → 0.1073 | CN; 0.2945 | auto-CN → refer → refer |

- **F01 — a confident false negative:** The three displayed A1 slice scores round to 0.000. Although the actual label is AD, A1 confidently predicts CN and even the stricter policy accepts it. CNN also predicts CN but sends the case for review. This showed me that a high-confidence negative prediction can still be dangerously wrong.
- **F02 — both models are confidently wrong:** The displayed A1 scores (**1.000, 0.999, 1.000**) favour AD even though the patient is labelled CN. Both models issue an AD flag. I could not treat agreement between them as clinical confirmation.
- **F03 — confidence does not mean accuracy:** A1's displayed scores (**0.984, 0.999, 0.993**) strongly favour AD for a CN patient. CNN predicts CN but refers the patient. This was another example of the larger model being confidently wrong.
- **F04 — slices can disagree:** A1's displayed scores (**0.990, 0.001, 0.004**) point in very different directions, while its overall patient score is only just above 0.5. Both models refer the case. This helped illustrate why I could not judge patient-level decisions from only a few selected slices.
- **F05 — a useful but incomplete safeguard:** All three displayed A1 slices (**0.002, 0.000, 0.176**) favour CN despite the AD label. Here, `strict_cn` refers a case that `reference` would automatically clear. I saw this as a useful improvement, but F01 shows that it did not catch every false negative.

I chose these cases to **illustrate different errors**, not to estimate how often they occur. Image appearance or contrast could be involved, but I do not have evidence to identify a particular anatomical cause, disease stage or biomarker. I have left the panels, private patient identifiers and raw data paths out of the public README.

## 7. Computational cost

| Recorded measure | ConvNeXt A1 | CNN |
|---|---:|---:|
| Trainable parameters | 27,818,594 | 359,922 |
| Selected-run training time | 6,437.32 s | 1,609.03 s |
| Training peak allocated GPU memory | 1.3108 GiB | 0.1170 GiB |
| Prior batch-1 forward median | 13.963 ms | 1.062 ms |
| Inference peak allocated GPU memory | 0.2364 GiB | 0.0461 GiB |
| Revised test-batch processing | 17.226 s | 14.943 s |

For me, the computational cost was a key result: ConvNeXt has **about 77 times more parameters** but worse test-slice accuracy. The timings have different scopes. CNN training time included in-loop checkpoint/history I/O; A1 excluded repeated interrupted epochs and some setup/I/O. Batch-1 latency excluded loading/aggregation, while test-batch processing included preprocessing/transfer but excluded setup, cache and recovery. Memory numbers are **process-allocated GPU memory**, not total VRAM. These are useful recorded costs, not fully controlled hardware benchmarks.

## 8. Reproducing the work

### Environment and verification

I carried out the recorded experiments on **Windows with Python 3.11.9, CUDA 12.6 and an RTX 4080 Laptop GPU**. My [requirements-pinned.txt](requirements-pinned.txt) lists `torch==2.7.1+cu126`, `torchvision==0.22.1+cu126`, `numpy==2.2.6`, `Pillow==11.2.1`, `scikit-learn==1.6.1` and `matplotlib==3.10.3`. I used the matching [PyTorch installation guidance](https://pytorch.org/get-started/previous-versions/#v271). These versions are top-level pins, **not a complete dependency lock**, and I did **not** verify installation in a freshly created environment.

For reproducibility, I used Codex-assisted clean-code verification. The recorded checks passed **15 synthetic interface tests**, regenerated approved manifests byte-for-byte, ran a small training-only smoke test and reproduced full validation predictions from the retained checkpoint. They did **not** retrain A1 or rerun test inference.

### Preparation and training

I included `dataset.py prepare` and `preparation_helpers/` so the saved split can be rebuilt from explicit data and metadata. The command checks patient assignments and hashes against `APPROVED_PARTITIONS.json` and refuses to overwrite existing outputs. It is **not** a new pixel audit or a new approval of the split. The commands below show the documented workflow with **my private local paths**. Anyone reproducing it should adjust the paths and choose a new, authorized output location.

```powershell
Set-Location '<checkout>/recognition/ConvNeXt_Alzheimers'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$out = 'C:/Users/Angus/COMP3710-private/submission-readiness/clean-verification-v2'

& $py -B dataset.py prepare --data-root $data --output-dir "$out/manifests"
& $py -B test_submission_interfaces.py
& $py -B train.py a1-smoke --data-root $data --splits-dir "$out/manifests" --output-dir "$out/training-smoke" --device cuda
```

I used the bounded smoke test to check the basic training path without running a full experiment. It uses **32 training slices and one optimizer update**, verifies finite loss, changing weights and checkpoint round-trip consistency, and reads no validation or test images. Its output is labelled `TRAINING_SMOKE_NOT_A_CANDIDATE`. The full-training command below is documented for reproduction; it was **not executed during submission-interface verification**:

```powershell
& $py -B train.py a1 --full-training --confirm-preprocessing-review --data-root $data --splits-dir "$out/manifests" --output-dir '<NEW_PRIVATE_A1_RUN>' --device cuda
# For recovery, use the same paths and add --resume.
```

The full route keeps my original V2 training recipe and **patient-based checkpoint-selection rule**. A different run might select another epoch or produce different scores; the earlier epoch-13 result is not guaranteed. The selected checkpoint is written to `<NEW_PRIVATE_A1_RUN>/run/PRIVATE_best_model.pt`. I kept the older commands and exact source requirements in [HISTORICAL_REPLAY.md](HISTORICAL_REPLAY.md).

### Independent predictions

My retained A1 weights are in my **private local workspace** and have not been made available for public download. The inference command below loads them without requiring the historical experiment/report folders:

```powershell
$checkpoint = 'C:/Users/Angus/COMP3710-private/v2-scratch_augmented-20261024-161044-951159/PRIVATE_best_model.pt'
& $py -B predict.py infer --checkpoint $checkpoint --checkpoint-sha256 efde4d21993085119d3ef1bc0efb88f837954c187c95b036373cd00e59d4d5ef --data-root $data --splits-dir "$out/manifests" --split validation --output-dir "$out/prediction" --device cuda
```

My `predict.py infer` route checks the checkpoint hash, compatible source, architecture/state, labels and manifests before inference. Optional saved patient rules use `--rules ... --rules-sha256 ...` and `--policy-rules ... --policy-rules-sha256 ...`; the tool **applies them without refitting**. Ordinary inference accepts authorized train/validation patients, with complete-patient input lists where used. Historical test evaluation is separately guarded.

The previous clean replay reproduced **all 4,660 slice, 233 scan and 102 patient validation probabilities** (within the existing **2e-5** tolerance, with no class-decision changes). It recovered **3,364/4,660 correct slices** and **85/102 correct patients**, along with these validation policy counts (auto-CN / AD flag / refer): `reference` **41/7/54**, `strict_cn` **21/7/74**, and `no_auto_cn` **0/7/95**. I used this as evidence that the saved model and interfaces could reproduce recorded outputs, **not** as a new round of model development.

Before making interface changes, I preserved the original **70 assignment files** and their SHA-256 hashes at revision `b67bb1327c744fdd9a30fea439efc4a5005fca27`. I kept checkpoint-relevant source in `historical_runtime/` and documented the earlier handoff in the [PR draft](PR_DESCRIPTION_DRAFT.md) and [local checklist](LOCAL_CLOSEOUT.md). These verification steps do **not** mean a final commit, publication or submission was completed.

## 9. Conclusions and limitations

Based on the required **test-slice results**, I concluded that the smaller CNN was the more practical choice. It reached **68.81% accuracy** and **0.6869 macro-F1**, compared with A1's **68.04%** and **0.6769**, while using far fewer resources. A1 had slightly better slice AUROC and fewer false AD predictions on CN slices, but missed more actual AD slices. At the patient level it detected one additional AD patient, at the cost of more CN false positives. I did not think these trade-offs justified saying ConvNeXt performed better overall.

The confidence experiments reinforced this conclusion. While there were improvements in patient-level probability scores after calibration, it **did not fix incorrect classifications**. Tightening the CN acceptance rule reduced A1's automatic clearance of AD-labelled patients from **five to one**, at the cost of more referrals. Referring every CN prediction avoided automatic negative decisions, but would mean reviewing everyone once AD flags were included. Therefore, it cannot be claimed that any of the policies are safe for clinical use.

My findings have important limits: just **102 independent test patients**; correlated slices/scans; historical patient-based selection despite slice-level assessment; repeated validation use and earlier test exposure; random-seed variation; and JPEG rather than quantitative MRI inputs. I did not establish the anatomical cause of errors, calibrate slice-level referral, or conduct prospective, cross-site or clinical validation. The schedule, grouped-loss and attention studies only tested specific configurations. I found no retained checkpoint meeting my quick five-percentage-point improvement screen, **but better classifiers may still exist**.

## Artificial intelligence usage disclosure

I used **ChatGPT** to assist with interpreting the assessment, planning experiments, writing and debugging Python and PowerShell, designing interfaces and synthetic tests, analysing confidence/referral policies, and identifying issues worth investigating. It also assisted with editing this README and ensuring everything mentioned is accurate. I reviewed the results and remain responsible for the project decisions and conclusions I present.

**Codex** was also used to audit source code and saved evidence, implement and debug parts of the training and submission workflow, and execute scoped verification tests. Codex helped with the schedule-only study, the grouped-supervision and attention implementations, and the evaluation/reporting checks; I ran the associated real-data experiments where recorded and reviewed the outputs. I have not presented AI-executed work as manual work I personally performed. Neither tool's involvement makes the project an independently validated clinical system.