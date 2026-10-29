# 2D Improved U-Net for HipMRI segmentation

COMP3710 Pattern Recognition Final Project | s4986405 | Normal difficulty

This project implements a basic 2D U-Net and a residual, multi-scale 2D Improved U-Net for the supplied HipMRI pelvic MRI slices. It investigates whether extra model complexity produces better overlap and fewer contour corrections at an acceptable memory and runtime cost. The intended output is an offline segmentation draft for human review. The course task targets prostate segmentation. MRI/reference/prediction analysis in this revision supports prostate = ID 5 as an image-based interpretation; original numeric IDs and measured results are preserved.

## Current evidence and scope

**Updated 29 October 2026 following the tutor discussion.** Both models completed matched 20-epoch training and three full validation and test passes. Image-based analysis identifies **ID 5 as the prostate** (an inferred mapping with illustrated evidence). Its pooled test Dice improves from **0.803655 to 0.843050 (+3.939 percentage points)**, with fewer false positives and false negatives. The six-ID test mean increases from **0.917867 to 0.924320 (+0.645 pp)**. IDs 3 and 4 decline, and Improved model-forward inference is 2.912x slower on the local RTX 4070.

| Evidence | Current status |
|---|---|
| Matched training and validation | Completed; both full Slurm tasks passed |
| Full local test evaluation | Completed; 540/540 slices, both fixed models, three passes each |
| Compulsory tutor feasibility check-off | Completed, as reported by the student on 29 October 2026 |
| Prostate label identification | ID 5, inferred from MRI/reference/prediction anatomy; illustrated below |
| Task-specific accuracy target | ID-5 pooled test Dice 0.843050 exceeds 0.75 under this interpretation |
| Whole-device peak VRAM | Unverified; allocator measurements are reported separately |

The result supports a measured task-specific accuracy/resource trade-off. It does not establish clinical suitability, uniform improvement across labels/slices or replication across training seeds.

## Prostate identification and direct baseline comparison

**Tutor follow-up, 29 October 2026.** The student reports completing the compulsory feasibility check-off. The tutor requested image-based identification of the prostate ID and a clearer account of the improvement over baseline. This revision addresses those points; the original training, checkpoints, full-test predictions and numeric-label metrics are unchanged.

### Why ID 5 is interpreted as the prostate

**The image-based interpretation is prostate = ID 5.** The review compared MRI appearance with the supplied reference masks and both fixed models' predictions, rather than choosing the ID with the highest Dice. In all three test patients, ID 5 forms a compact central region immediately below the bright bladder-like ID-3 region in the rotated display. Neighboring filename-index slices show its relationship to the more posterior rectum-like ID-4 region. ID 2 follows extensive pelvic/bony structures. These spatial and morphological relationships are consistent with prostate anatomy [4,5].

The figures and [anatomical analysis](ANATOMICAL_LABEL_ANALYSIS.md) document this inference. The course specification names the relevant structures but provides no numerical dictionary. Identity NIfTI affines and empty descriptions do not recover patient-space orientation, physical spacing or a definitive base/apex assignment. Accordingly, this is a reproducible image-based coursework interpretation, not an official dataset dictionary or a mapping confirmed by the tutor. All original numeric IDs are retained.

![MRI, reference labels and fixed-model ID-5 predictions](results/anatomy_review/anatomy_and_fixed_predictions.png)

The three displayed examples use week 0 and the largest reference ID-5 area per patient, with the lowest slice index breaking ties. Selection does not use either model's Dice or improvement. Patient 41's displayed slice is worse for Improved (0.929843 to 0.908316), even though that patient's pooled ID-5 Dice improves. These images illustrate anatomy and local behavior, not the frequency of success.

![Neighboring slices supporting the anatomical interpretation](results/anatomy_review/neighboring_reference_slices.png)

### How much better is the trained model?

The primary task-specific comparison is the **whole-test pooled ID-5 Dice: 0.803655 baseline versus 0.843050 Improved**, an absolute increase of **0.039394 Dice**, or **3.939 percentage points**. Relative to baseline Dice this is **4.90%**, a different quantity from percentage points. Under the image-based ID-5 interpretation, both models exceed the course's pooled test Dice target of 0.75; Improved exceeds it by 9.305 percentage points. This does not imply every slice exceeds 0.75.

| Measurement and scope | Baseline | Improved | Change |
|---|---:|---:|---:|
| ID-5 validation pooled Dice, 660 slices | 0.804001 | 0.819091 | +1.509 pp |
| ID-5 test pooled Dice, 540 slices | 0.803655 | 0.843050 | +3.939 pp |
| ID-5 test pooled IoU | 0.671759 | 0.728683 | +5.692 pp |
| Six-ID test mean of pooled Dice | 0.917867 | 0.924320 | +0.645 pp |
| ID-5 test false negatives, pixels | 7,357 | 5,394 | -1,963 |
| ID-5 test false positives, pixels | 18,383 | 14,951 | -3,432 |
| Parameters | 1,962,422 | 3,187,746 | 1.624x |
| Local RTX 4070 model forward, ms/slice | 1.605100 | 4.674550 | 2.912x |
| A100 peak allocated training memory, GiB | 0.119952 | 0.301517 | 2.514x |

**Where it is better:** fewer missed ID-5 target pixels and fewer extra ID-5 pixels across the full test cohort, consistent with its higher Dice and IoU. True positives increase from 52,678 to 54,641 over the same 60,035 reference ID-5 pixels. Pooled ID-5 Dice also improves for each of the three patients:

| Test patient | Baseline ID-5 Dice | Improved ID-5 Dice | Change (pp) |
|---|---:|---:|---:|
| 40 | 0.805732 | 0.843235 | +3.750 |
| 41 | 0.818287 | 0.820786 | +0.250 |
| 42 | 0.789606 | 0.853784 | +6.418 |

**Where it is worse and what it costs:** test ID-3 Dice decreases by 0.604 pp and ID-4 by 1.033 pp. Six-ID mean validation Dice decreases slightly, by 0.049 pp, while the ID-5 validation score increases by 1.509 pp. Improved uses more parameters and allocator memory and has a 2.912x longer local model-forward time. Residual context and multi-scale features are plausible contributors to the ID-5 gain, but the experiment compares complete architectures and cannot isolate their causal contributions. There is one training seed and only three test patients; no statistical or clinical superiority is claimed.

The comparison uses the same supplied patient splits, preprocessing, loss and 20-epoch training budget. Each model's checkpoint was selected by minimum validation loss before test use (baseline epoch 9; Improved epoch 20), then kept fixed for all three full test passes. These are repeats of inference, not three trained models. Anatomical interpretation after the tutor discussion did not change numerical labels, select a favorable checkpoint or trigger training on test results.

## Data and preprocessing

Data were provided by the HiPMRI Study through the COMP3710 teaching copy on UQ Rangpur at `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`. The supplied data-use agreement requests acknowledgement and an applicable dataset-related citation; see Liu et al. [3]. That publication supplies dataset context, not a verified mapping for these course label files. Raw NIfTI data and checkpoint weights remain outside Git.

| Supplied split | Slices | Patients | Patient IDs |
|---|---:|---:|---|
| Train | 11,460 | 31 | 4-33 and 35 |
| Validate | 660 | 4 | 36-39 |
| Test | 540 | 3 | 40-42 |

Patient IDs are disjoint. This respects the supplied patient-level partition and avoids putting slices from the same patient into training and evaluation. Within a split, slices and repeated weeks from a patient remain correlated; they are not independent subjects. Validation patient 39 contributes 480/660 slices, motivating reporting both pooled confusion counts and equal-patient means. The initial split audit inspected filenames; completed training and validation subsequently loaded their full splits. Test execution is tracked separately.

`dataset.py` sorts `case_*.nii.gz` filenames and pairs each with the corresponding `seg_*.nii.gz`. It rejects missing pairs, non-2D or mismatched grids, non-finite MRI values, and masks that are non-finite, fractional or outside 0-5. Images have shape `[1,H,W]`; hard masks have shape `[H,W]`. Training optionally returns six-channel one-hot targets `[6,H,W]` with exact recovery of the hard mask.

The frozen experiment uses per-slice z-score normalization over all image pixels with float64 summary statistics and a finite guard for constant images. This handles slice intensity scale without fitting statistics to validation or test cohorts. It may discard absolute intensity information; this choice was shared by both architectures and was not ablated. Training applies a paired last-axis flip with probability 0.5; validation and test use no augmentation. Image and mask receive the same transform.

Native dimensions are preserved. Training includes 11,400 slices at 256x128 and 60 at 256x144. `NativeShapeBatchSampler` groups equal shapes and retains incomplete batches, so every slice and pixel contributes without resizing or dropping samples. Prediction NIfTI exports copy the reference affine/header; qualitative analysis additionally checks selected image/mask affine agreement. Native array orientation is not assigned an anatomical plane, base or apex. Two sampled NIfTI headers had empty descriptive metadata; that initial metadata audit did not resolve anatomy. The later image-based ID-5 interpretation is documented above, without treating the identity affine as recovered physical orientation.

## Models, objective and metrics

The baseline adapts U-Net's encoder-decoder and skip-connection design [1]. Four downsampling levels use two-convolution/ReLU blocks and max pooling; bilinear upsampling restores resolution and combines matching encoder features. `UNet2D` returns raw six-class logits on the input grid.

`ImprovedUNet2D` adapts residual context blocks and multi-scale outputs from Isensee et al. [2] to 2D. Widths are b, 2b, 4b, 8b and 16b. Context blocks use affine InstanceNorm, LeakyReLU (0.01), two pre-activation 3x3 convolutions, an identity residual connection and Dropout2d (0.3). Stride-2 convolutions downsample. Decoder stages resize to the exact skip grid with nearest interpolation, reduce channels, concatenate skip features and apply localization convolutions. Quarter-, half- and full-resolution segmentation heads are aligned and summed as raw logits before the shared loss. They do not have separate auxiliary losses.

At b=16 the baseline has 1,962,422 parameters and Improved has 3,187,746. The Improved model requires each input dimension to be at least 32 for its deepest normalization layer; exact skip-size interpolation handles odd dimensions. Model components in `modules.py` depend on PyTorch, without NumPy. This implementation changes dimensionality, preprocessing, loss, augmentation and training schedule from the original 3D brain-MRI system; it is an adaptation, not an exact reproduction or a feature-by-feature ablation.

For six one-hot target channels, the objective is mean pixel cross-entropy plus one minus mean soft Dice. For channel c, soft Dice is `(2*sum(p_c*y_c)+1e-5)/(sum(p_c)+sum(y_c)+1e-5)`, summed over batch and spatial dimensions. All six channels, including ID 0, receive equal weight. No class weighting, anatomical selection or test-derived threshold is used.

Reporting uses hard argmax predictions and confusion matrices with ground truth on rows and predictions on columns. For each ID, `Dice=2TP/(2TP+FP+FN)` and `IoU=TP/(TP+FP+FN)`. Counts are summed before pooled overlap is calculated. When both reference and prediction are absent, overlap is undefined and exported as JSON null / an empty CSV cell. Equal-patient means are explicitly separate from pooled metrics. A lower soft training objective need not improve every hard-label score.

## Frozen experiment and validation results

Training source was fixed at `bc8a6a4688f50770f71de24a56dff8aad43204c7`, protocol `hipmri-2d-comparison-v1`. Both models used 20 full epochs, batch size 2, base width 16, Adam at constant learning rate 0.001, seed 3710, identical preprocessing, paired flips and loss. The minimum full-validation-loss checkpoint was selected, with the earliest epoch retained on ties. A fixed seed records the experiment; bitwise deterministic CUDA training is not claimed.

The detailed measurements below are the reviewed validation report. They are included directly so the README PDF contains the quantitative tables, figures and limitations as well as links to machine-readable records.

## Detailed validation measurements

Reviewed on 27 October 2026. Both models completed the frozen 20-epoch protocol and three full validation passes. Results are mixed: five numeric-label pooled Dice scores increased, but label 3 declined; the six-label mean is slightly lower for the Improved model. The Improved model also used more allocator memory and had a longer model-forward time. This experiment does not establish overall superiority or prostate-specific test accuracy.

### Protocol and evidence

- Protocol: `hipmri-2d-comparison-v1`; source commit `bc8a6a4688f50770f71de24a56dff8aad43204c7`.
- Reporting implementation: commit `6b63092`. Exported source/manifest/checkpoint hashes and reviewed result summaries are in [summary.json](results/comparison_v1_704715/summary.json).
- Models: UNet2D (1,962,422 parameters) and ImprovedUNet2D (3,187,746 parameters), base width 16, six output IDs. This compares whole architectures, not isolated feature ablations.
- Both use 20 epochs, batch size 2, Adam learning rate 0.001, seed 3710, the same paired flips, per-slice all-pixel z-score normalization, Dice plus cross-entropy, and native-shape batching.
- Each epoch covers 11,460 training slices / 31 patients and 660 validation slices / 4 patients. Patient separation passed. Evaluation covers 11 patient/week cases. Test images are not accessed by this comparison.
- Each model selects the minimum full-validation loss checkpoint (earliest tie). Selection epochs were baseline 9 and Improved 20; no checkpoint was selected by an individual label score.
- Both array tasks completed with Slurm exit code 0:0 and final task PASS markers. Baseline task 704715_0: a100-8, 49:05 elapsed. Improved task 704715_1: a100-1, 1:02:42 elapsed.
- The Improved task initially waited with `QOSMaxJobsPerUserLimit`; it started after the baseline finished.
- Saved-record audits passed for both downloaded runs. The local recomputation matched all downloaded comparison fields except relocated input paths; all four CSVs matched their JSON values. Three report figures were visually reviewed.

### Pooled numeric-label overlap

Each Dice/IoU value below is recomputed from the confusion matrix summed over all 660 validation slices. Changes are Improved minus baseline in percentage points (pp). The original numeric IDs are retained. ID 5 is interpreted as prostate in the illustrated 29 October analysis; other structures remain supporting visual interpretations.

| ID | Baseline Dice | Improved Dice | Dice change (pp) | Baseline IoU | Improved IoU | IoU change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.993030 | 0.994407 | +0.138 | 0.986156 | 0.988876 | +0.272 |
| 1 | 0.980363 | 0.981791 | +0.143 | 0.961482 | 0.964233 | +0.275 |
| 2 | 0.924195 | 0.934221 | +1.003 | 0.859073 | 0.876562 | +1.749 |
| 3 | 0.897304 | 0.861016 | -3.629 | 0.813737 | 0.755951 | -5.779 |
| 4 | 0.875168 | 0.880611 | +0.544 | 0.778044 | 0.786688 | +0.864 |
| 5 | 0.804001 | 0.819091 | +1.509 | 0.672242 | 0.693610 | +2.137 |

The unweighted mean of these six pooled-label Dice values is **0.912343 baseline versus 0.911856 Improved**. This descriptive mean includes every numeric ID; it is not a tutor-confirmed primary prostate metric. Label 3 loses 3.629 pp while label 5 gains 1.509 pp and label 2 gains 1.003 pp.

![Numeric-label validation overlap](results/comparison_v1_704715/validation_metrics_comparison.png)

Full precision: [per-class CSV](results/comparison_v1_704715/per_class_comparison.csv).

### Patient differences

Patient 39 contributes 480/660 slices (72.73%); each other patient contributes 60. The following table first computes a Dice value per patient and then averages the four patient values equally. It complements the pooled values above rather than replacing them.

| ID | Baseline patient-mean Dice | Improved patient-mean Dice | Change (pp) |
|---|---:|---:|---:|
| 0 | 0.993050 | 0.994321 | +0.127 |
| 1 | 0.972599 | 0.972500 | -0.010 |
| 2 | 0.922124 | 0.928116 | +0.599 |
| 3 | 0.901070 | 0.855549 | -4.552 |
| 4 | 0.866387 | 0.863876 | -0.251 |
| 5 | 0.765095 | 0.792361 | +2.727 |

- Patient 38, label 3: Dice **0.795200 -> 0.611632** (-18.357 pp). This is the largest patient/label decline and needs image-level error analysis.
- Patient 37, label 4: **0.856800 -> 0.786273** (-7.053 pp), despite the small pooled label-4 improvement.
- Patient 36, label 5: **0.772585 -> 0.894256** (+12.167 pp). The pooled label-5 gain is not an improvement for every patient; patients 38 and 39 slightly decline.
- These observations informed four actual failure examples and one improvement example, now documented with MRI, ground truth, predictions and TP/FP/FN overlays in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md). The subsequent image-based ID-5 interpretation and its limits are documented above.

All patient/label values: [patient metrics CSV](results/comparison_v1_704715/patient_metrics_comparison.csv).

### Loss histories and checkpoint choice

| Measurement | Baseline | Improved |
|---|---:|---:|
| Selected epoch | 9 | 20 |
| Minimum validation CE + soft-Dice loss | 0.394608332 | 0.242225716 |
| Final epoch validation loss | 0.445130924 | 0.242225716 |

The baseline improves early and then its validation loss fluctuates. The Improved loss decreases sharply around epochs 7-8 and reaches its minimum at the final budgeted epoch; convergence beyond this fixed budget is not established. The objective uses cross-entropy and soft Dice, while the reported overlap uses hard argmax predictions. A lower objective therefore does not guarantee higher hard Dice for every label or patient.

![Training and validation loss](results/comparison_v1_704715/training_loss_comparison.png)

### Runtime and memory

| Measurement | Baseline | Improved | Improved / baseline |
|---|---:|---:|---:|
| Peak allocated training GiB | 0.119952 | 0.301517 | 2.514x |
| Peak reserved training GiB | 0.167969 | 0.353516 | 2.105x |
| Training wall seconds including I/O and plots | 2904.286 | 3718.456 | 1.280x |
| Median of three model-forward pass medians, ms/slice | 2.134302 | 5.968241 | 2.796x |

Both nodes reported NVIDIA A100-PCIE-40GB, PyTorch 2.14.0+cu130 / CUDA build 13.0. Hardware nodes differ, and timing is observational rather than a controlled same-device benchmark. Each validation pass processes all 660 slices at batch size 1 after three shape-specific warmups, with CUDA synchronization around the model forward. Data loading, preprocessing, host-to-device transfer and metric computation are outside that forward timer.

- Baseline forward medians: 2.150752, 2.134302, 2.117505 ms.
- Improved forward medians: 5.886407, 5.979868, 5.968241 ms.
- These repeat measurements reuse the same weights; they are not three independent training seeds or confidence intervals.
- Both measured model-forward summaries are below the proposed 100 ms limit for this validation input shape. This does not establish complete application or held-out latency.
- Allocated/reserved peaks describe PyTorch training tensors and its allocator. They do not verify the proposed whole-device 12 GiB VRAM bound.
- Per-case times include loading, preprocessing, transfers, inference and metrics, excluding warmups and exports. Across 11 cases x 3 passes, median case processing time is 0.358803 s baseline and 0.577163 s Improved; each case contains 60 slices. These are not model-only per-slice times.

![Allocator memory and model-forward timing](results/comparison_v1_704715/resource_comparison.png)

Full precision: [model summary](results/comparison_v1_704715/model_summary.csv) and [case times](results/comparison_v1_704715/case_times.csv).

### Reproduction and remaining work

Run from the project checkout with the original completed experiment directories available:

```bash
python compare_results.py \
  --baseline-run "$HOME/hipmri_runs/comparison_v1_704715_0_UNet2D" \
  --improved-run "$HOME/hipmri_runs/comparison_v1_704715_1_ImprovedUNet2D" \
  --output-dir "$HOME/hipmri_runs/comparison_report_704715_reproduced"
```

Use a new output directory. The original committed runtime snapshot and protocol must be preserved. The committed summaries/CSVs/PNGs contain derived validation records; MRI data and checkpoint weights remain outside Git.

The test set remains untouched by the comparison runner. Tutor check-off has since been completed, according to the student on 29 October. The image-based anatomical interpretation is documented above; no official numerical dictionary was recovered. The fixed-model local held-out results are reported in the separate test section below. Four actual validation failures and one improvement example have since been rendered and reviewed; see [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) for the declared selection rules, evidence and limitations. Four validation patients and one training seed limit generalization claims; no statistical or clinical superiority is established. Source and filename/size hashes support record consistency but do not prove byte-identical data contents. The initial validation bundles excluded weights, so their local audit checked saved records rather than re-running validation. Both original checkpoints were later downloaded, verified and reloaded for the separate local held-out test.

AI assistance covered report-code scaffolding, command suggestions, numerical consistency checks, plot review and drafting this analysis. The student executed the commands and shared the outputs and files; the student remains responsible for reviewing, understanding and explaining the submitted work.

## Real validation error analysis

Reviewed on 27 October 2026. Five real validation slices show four failure examples and one improvement example. These historical validation figures retain their original numeric labels. The later image-based ID-5 interpretation is documented above. These are deliberately selected diagnostic examples, not a representative sample or held-out test evaluation. Overall scores remain in [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md).

### Method and provenance

- Source runtime: frozen commit `bc8a6a4688f50770f71de24a56dff8aad43204c7`; selected checkpoints: baseline epoch 9 and Improved epoch 20, unchanged from the formal comparison.
- `failure_analysis.py` audits both completed runs, verifies matching protocol/source/data records and uses the frozen runtime files. It verifies checkpoint SHA256 before and after inference, selected image/mask affine agreement, recorded filename/size consistency, and selected input SHA256 stability.
- Five CPU re-inferences preserve the native 256x128 grids and saved all-pixel z-score preprocessing. MRI 1st-99th percentile clipping is only a display window; it does not modify the model inputs. Each full MRI has a yellow box showing the common detail-view crop, defined by the union of ground truth and both selected-label predictions plus eight pixels of margin, clipped to the grid.
- Upper prediction panels use the six-ID color bar. Lower overlays use green=TP, red=FP and blue=FN for the selected ID; the ground-truth detail view uses yellow. These two palettes serve different purposes. Display orientation is native array orientation, without an anatomical plane or base/apex assignment.
- All eight tests passed on Rangpur CPU (5.664 s), including five synthetic PNG exports and hand-checked inference counts. The real five-slice run then printed `Exploratory validation overlays: PASS`. The downloaded manifest reports completed, and source/PNG/checkpoint checksums passed local checks. All five figures were visually inspected.
- Figure counts/Dice/IoU use fresh CPU inference. Selected-label metrics match the CUDA reference in all five examples. Full six-label confusion-matrix difference flags occur for baseline example 3 and Improved examples 4 and 5; these flags are preserved in the manifest. This analysis is not a timing benchmark.
- Full selection rules, dataset indices, filenames, reference and rendered counts, crop bounds, hashes and difference flags are in [analysis_manifest.json](results/failure_analysis_704715/analysis_manifest.json).

### Declared selection rules

The rules were chosen post hoc to investigate patient/label findings in the validation comparison. Each search is within the named patient, uses validation pass 1, and breaks ties by the lowest dataset index. The minimum target size prevents selecting a completely absent target; it is not a clinically meaningful size threshold.

| Example | Patient / ID | Rule | Minimum GT pixels | Eligible slices |
|---|---|---|---:|---:|
| 1 | 38 / 3 | Largest baseline-minus-Improved Dice; GT >=64 pixels | 64 | 54 |
| 2 | 37 / 4 | Largest Improved-minus-baseline FP count; GT >=32 pixels | 32 | 15 |
| 3 | 38 / 5 | Lowest Improved Dice; GT >=16 pixels | 16 | 20 |
| 4 | 39 / 5 | Lowest Improved Dice; GT >=32 pixels | 32 | 154 |
| 5 | 36 / 5 | Largest Improved-minus-baseline Dice; GT >=32 pixels; improvement example | 32 | 28 |

### Displayed-label measurements

| Example | Ground-truth pixels | Baseline TP / FP / FN | Improved TP / FP / FN | Baseline Dice | Improved Dice |
|---|---:|---|---|---:|---:|
| 1 | 4661 | 2554 / 80 / 2107 | 660 / 55 / 4001 | 0.700206 | 0.245536 |
| 2 | 78 | 76 / 59 / 2 | 78 / 408 / 0 | 0.713615 | 0.276596 |
| 3 | 35 | 0 / 0 / 35 | 0 / 0 / 35 | 0.000000 | 0.000000 |
| 4 | 79 | 79 / 251 / 0 | 79 / 480 / 0 | 0.386308 | 0.247649 |
| 5 | 152 | 55 / 81 / 97 | 115 / 68 / 37 | 0.381944 | 0.686567 |

### 1. Broad missed region: patient 38, ID 3

File `case_038_week_0_slice_0.nii.gz`; dataset index 120; week 0; slice ID 0.

The ground-truth detail view contains a large connected ID-3 region. Both models miss part of it, but the Improved prediction covers a substantially smaller portion. Its blue FN overlay covers most of the target, with only a relatively small green TP patch. FN rises from 2,107 to 4,001 pixels while FP decreases only from 80 to 55; Dice falls from 0.700206 to 0.245536. This is a broad regional omission, beyond a small contour offset. It illustrates the patient-38 ID-3 decline in the aggregate results, without establishing its anatomical cause.

Human review should check target completeness, including large omitted regions, before accepting a contour. A future experiment could examine adjacent slices and multi-scale head behavior on validation data; this example alone does not prove that residual blocks or a specific head caused the failure.

![1. Broad missed region: patient 38, ID 3](results/failure_analysis_704715/01_patient038_label3.png)

### 2. Region expansion: patient 37, ID 4

File `case_037_week_0_slice_46.nii.gz`; dataset index 101; week 0; slice ID 46.

The yellow ground truth contains 78 pixels. The Improved model covers all 78 but also predicts a large contiguous surrounding region: FP increases from 59 to 408, and predicted ID-4 area expands from 135 to 486 pixels. The red FP overlay makes this expansion visible. Reducing FN from 2 to 0 does not compensate for the added false positives: Dice falls from 0.713615 to 0.276596.

Human review must check outward spread as well as missed area. Potential future checks include confusion with neighboring numeric IDs and validation-only changes to sampling/loss design; any such change requires a fresh protocol and cannot be credited to this completed comparison.

![2. Region expansion: patient 37, ID 4](results/failure_analysis_704715/02_patient037_label4.png)

### 3. Complete small-region miss: patient 38, ID 5

File `case_038_week_0_slice_25.nii.gz`; dataset index 138; week 0; slice ID 25.

Both models produce no ID-5 pixels despite a 35-pixel ground-truth patch. Both have TP=0, FP=0, FN=35 and Dice=0. The blue overlay covers the entire yellow target region. Native-pixel enlargement makes this failure visible; no synthetic boundary smoothing or image resampling was used.

A high pooled score does not ensure recovery of every small visible region. Review should explicitly inspect small connected regions. Whether missing 3D context, low contrast or class imbalance contributes remains a hypothesis, not a conclusion from this image.

![3. Complete small-region miss: patient 38, ID 5](results/failure_analysis_704715/03_patient038_label5.png)

### 4. Excess predicted area despite complete coverage: patient 39, ID 5

File `case_039_week_5_slice_40.nii.gz`; dataset index 515; week 5; slice ID 40.

Both models cover all 79 target pixels (FN=0), but each extends beyond the ground truth. The Improved prediction has a larger elongated red FP region: FP rises from 251 to 480 and predicted area from 330 to 559 pixels. Dice drops from 0.386308 to 0.247649. Complete target coverage therefore coexists with a poor overlap score.

Review must compare the predicted extent with the reference boundary rather than checking coverage alone. A single postprocessing rule that trims all predictions could reduce FP here while worsening the omissions in examples 1 and 3; that tradeoff requires validation rather than an automatic global fix.

![4. Excess predicted area despite complete coverage: patient 39, ID 5](results/failure_analysis_704715/04_patient039_label5.png)

### 5. Improvement with remaining errors: patient 36, ID 5

File `case_036_week_0_slice_18.nii.gz`; dataset index 10; week 0; slice ID 18.

The Improved prediction covers more of the 152-pixel ground truth: TP increases from 55 to 115, FN falls from 97 to 37, and FP decreases from 81 to 68. Dice increases from 0.381944 to 0.686567. The green overlay expands into part of the region missed by the baseline, although residual red and blue regions remain.

This deliberately selected improvement example provides context for the pooled ID-5 gain. It does not establish consistent improvement across patients or remove the need to review the remaining boundary discrepancies.

![5. Improvement with remaining errors: patient 36, ID 5](results/failure_analysis_704715/05_patient036_label5.png)

### Engineering interpretation and limits

The engineering dilemma is whether the extra model complexity and resource cost produces contours that need less human correction. This validation run presents mixed evidence: larger omissions and expansions in some cases, a complete small-region miss in both models, and an improvement in another case. The model name, lower validation loss or high aggregate scores cannot replace image-level review. The image-based ID-5 interpretation does not establish clinical severity or prostate base/apex accuracy from these selected validation examples.

The five examples are selected extremes, include correlated slices from four validation patients, and reuse one training seed. They cannot estimate error prevalence, confidence intervals or held-out generalization. Explanations such as low contrast, class imbalance or missing 3D context remain candidate explanations requiring further evidence. The local held-out evaluation is now reported separately; the later illustrated ID-5 interpretation is image-based, and the student reports tutor check-off completed on 29 October.

### Reproduce

```bash
python -m unittest -v test_failure_analysis
python failure_analysis.py \
  --baseline-run "$HOME/hipmri_runs/comparison_v1_704715_0_UNet2D" \
  --improved-run "$HOME/hipmri_runs/comparison_v1_704715_1_ImprovedUNet2D" \
  --output-dir "$HOME/hipmri_runs/failure_analysis_704715_reproduced" \
  --device cpu
```

Use a new output directory and retain the frozen runtime snapshot, original checkpoints and supplied validation data. Outputs are five derived PNG figures and a JSON manifest; raw MRI/NIfTI data and checkpoint weights remain outside Git.

AI assistance covered deterministic example-selection and rendering code, test scaffolding, consistency checks, visual review and drafting this analysis. The student executed the commands and shared the results/files. The student remains responsible for checking and explaining the implementation, observations and limitations.

## Fixed-model held-out evaluation

Both validation-selected models completed three full test passes on all 540 supplied test slices, using the same fixed weights and original evaluator source. The local run printed `Local fixed-model full held-out test comparison: PASS`; completed provenance, grouped metrics, input/source/weight consistency, repeated predictions and every CSV cell passed the saved-record audit. Both figures were visually checked.

### Scope and provenance

- Test cohort: patients 40, 41 and 42; 540 slices, nine patient/week cases, all native 256x128. Patient 40 contributes 420 slices (77.78%); patients 41 and 42 contribute 60 each.
- Model source: `bc8a6a4688f50770f71de24a56dff8aad43204c7`; selected epochs baseline 9 / Improved 20, unchanged after validation. The checkpoint SHA256 values match the predeclared held-out protocol.
- Execution: Windows 11, Python 3.12.14, PyTorch 2.14.0+cu130 / CUDA build 13.0, RTX 4070 Laptop GPU (8188 MiB, driver 617.42). Local protocol `hipmri-heldout-numeric-local-v1` preserves the original full test procedure while documenting the user-requested hardware/path change.
- The original A100 held-out runner was committed as `bc601d92de5fb3da982b43d96de34afcd544a94c`. The local orchestration is identified by its exact source SHA256, not a fabricated Git commit. Actual execution logs/records remain in the local evidence directory.
- The downloaded accounting records show remote job 705243 already COMPLETED / 0:0 before the attempted cancellation. Its result files were not collected or reviewed. All numerical results here come from the separately executed local GPU run.
- All 1,080 local test image/mask files were content-hashed and rechecked around the evaluation stages. The transferred archive SHA256 matched the value printed on Rangpur. Patient separation from both training and validation passed.
- Numeric IDs 0-5 are preserved. This revision interprets ID 5 as prostate from image evidence and reports its task-specific comparison above; the student reports tutor feasibility check-off completed on 29 October.

### Pooled overlap

Confusion counts are summed over the full cohort before overlap is computed. Changes are Improved minus baseline in percentage points (pp).

| ID | Baseline Dice | Improved Dice | Dice change (pp) | Baseline IoU | Improved IoU |
|---|---:|---:|---:|---:|---:|
| 0 | 0.995656 | 0.995920 | +0.026 | 0.991349 | 0.991873 |
| 1 | 0.982035 | 0.984254 | +0.222 | 0.964704 | 0.968997 |
| 2 | 0.925427 | 0.938640 | +1.321 | 0.861204 | 0.884376 |
| 3 | 0.962184 | 0.956143 | -0.604 | 0.927123 | 0.915971 |
| 4 | 0.838244 | 0.827911 | -1.033 | 0.721531 | 0.706355 |
| 5 | 0.803655 | 0.843050 | +3.939 | 0.671759 | 0.728683 |

The descriptive mean over all six pooled-label Dice values is **0.917867 baseline versus 0.924320 Improved** (+0.645 pp). IDs 0, 1, 2 and 5 improve; IDs 3 and 4 decline. This all-ID mean is a secondary descriptive summary; the image-based ID-5 target comparison is reported separately. Neither establishes statistical/clinical superiority.

![Full-test pooled overlap](results/test_local_705243/heldout_metrics_comparison.png)

### Patient differences and aggregation

Each patient receives equal weight in the following per-label mean; all three patients have defined values. The cohort has only three independent patient IDs.

| ID | Baseline equal-patient Dice | Improved equal-patient Dice | Change (pp) |
|---|---:|---:|---:|
| 0 | 0.994015 | 0.995461 | +0.145 |
| 1 | 0.980949 | 0.984218 | +0.327 |
| 2 | 0.921972 | 0.939292 | +1.732 |
| 3 | 0.957475 | 0.950309 | -0.717 |
| 4 | 0.856776 | 0.868462 | +1.169 |
| 5 | 0.804542 | 0.839268 | +3.473 |

ID 3 declines for all three patients. ID 5 improves for all three, but the gain varies from about +0.250 pp for patient 41 to +6.418 pp for patient 42. For ID 4, pooled Dice falls while the equal-patient mean increases: patient 40 declines (0.829432 to 0.809963) and contributes most slices, while patients 41 and 42 improve. Pooled and patient-mean metrics answer different questions; neither should be chosen post hoc to hide a decline.

![Test patient changes and local GPU timing](results/test_local_705243/heldout_patients_and_timing.png)

### Local timing

| Measurement | Baseline | Improved |
|---|---:|---:|
| Median of three model-forward pass medians, ms/slice | 1.605100 | 4.674550 |
| Three pass medians, ms | 1.568750, 1.605100, 1.617400 | 4.674550, 4.640200, 4.947650 |
| Median case-processing seconds (27 case/pass records per model) | 0.335762 | 0.545410 |

Improved model-forward time is **2.912x** baseline on this local GPU. Batch size is 1 with three warmups per shape and CUDA synchronization. Forward time excludes loading, preprocessing, transfers, argmax and metric computation. Case processing includes loading, preprocessing, transfers, forward and metrics, while excluding warmups/exports; each case has 60 slices. The recorded evaluation/export wall time was 59.613 seconds, excluding dependency installation, transfer and initial preflight audits.

Both local forward summaries are below the proposed 100 ms per-slice target in this measured scope. They do not measure a complete interactive contouring workflow. A100 validation latency is a separate experiment and must not be combined with these RTX 4070 timings as a same-device benchmark. No local whole-device training-memory bound was measured.

### Interpretation and limits

The fixed-model test run shows a modest all-ID mean gain with specific regressions and higher forward cost. The ID-3 decline appears in both validation and test. The results do not support an unconditional recommendation to replace the baseline. The image-based ID-5 interpretation allows a task-specific coursework comparison, with its inference limits stated above. The existing five qualitative failure examples remain validation examples; new anatomy illustrations are explicitly post-hoc test images and are not used for model tuning.

The same weights were reused in all three passes. These are repeat timing checks, not three training seeds, confidence intervals or independent replications. Test results were not used to retrain, tune, choose thresholds or change checkpoints. A new model-development study would require a separate evaluation design.

### Machine-readable evidence

- [Pooled class CSV](results/test_local_705243/per_class_test.csv), [patient CSV](results/test_local_705243/patient_test_metrics.csv), [equal-patient means](results/test_local_705243/equal_patient_mean_test.csv).
- [Model timing CSV](results/test_local_705243/model_test_summary.csv), [case timings](results/test_local_705243/case_test_times.csv), [full summary](results/test_local_705243/summary.json).
- [Audit record](results/test_local_705243/audit.json), [execution provenance](results/test_local_705243/provenance.json), [environment](results/test_local_705243/environment.json), [protocol](results/test_local_705243/protocol.json), [original protocol](results/test_local_705243/parent_protocol.json).
- [Test file content hashes](results/test_local_705243/test_content_sha256.json) and [exact executed local wrapper](results/test_local_705243/run_local_heldout.py.txt). The wrapper snapshot is workspace evidence with recorded paths, not a standalone project CLI; portable evaluator commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

AI assistance covered local environment setup, transfer instructions, guarded extraction/orchestration, actual local command execution, numerical and source/data checks, figure review, and drafting this report. The student supplied the inputs from Rangpur and remains responsible for reviewing and explaining the submitted work.

## Engineering recommendation

Under the image-based prostate = ID 5 interpretation, Improved increases pooled test Dice from **0.803655 to 0.843050 (+3.939 pp)**, while reducing both FP and FN pixels. Its six-ID test mean also increases, but IDs 3 and 4 decline; the six-ID validation mean is slightly lower. Improved is therefore a task-specific accuracy/resource trade-off rather than a uniform replacement: it uses more parameters, 2.514x peak allocated training memory and 2.912x local model-forward time. Retain the baseline as the simpler fallback. The anatomical mapping remains an image-based inference, with no official dictionary recovered.

Human review must still inspect omitted regions, outward expansion and small targets. A high pooled score can hide complete slice-level misses. The selected examples do not establish clinical severity or base/apex accuracy. Adjacent-slice context, class-aware sampling and other losses are possible future validation studies, not implemented or proven causes of improvement here. Future development must not tune to these test results.

## Feasibility Review

**Project:** HipMRI 2D Improved U-Net | **Student:** s4986405 | **Initial review:** 26 October 2026. **Updated 29 October 2026:** the student reports completing the compulsory tutor consultation/check-off. The original proposal is retained below with subsequent evidence distinguished; no approval before the reported consultation is claimed.

**User need and dilemma.** An imaging specialist needs an offline contour draft for human correction. The proposed investigation asks whether a more complex 2D architecture reduces prostate/adjacent-organ boundary errors enough to justify added runtime and memory. Anatomical evaluation depends on confirming numeric label semantics independently.

**Choice and course concepts.** The Normal option uses the supplied HipMRI 2D patient split, U-Net as baseline [1], and residual context plus multi-scale outputs adapted from Isensee et al. [2]. Convolutions, skip/residual connections, normalization and overlap objectives connect the implementation to course concepts. Two-dimensional processing limits inter-slice continuity.

**Preliminary evidence.** Initial filename audits found 31/4/3 disjoint patients and 11,460/660/540 slices. One real MRI/mask pair loaded at 256x128. The untrained baseline produced finite six-channel logits; CPU shape/gradient, metric and small pipeline checks passed. This early evidence did not establish accuracy. Later full training, validation and five real diagnostic overlays are documented above; they do not constitute tutor approval.

**Five proposed acceptance criteria.** (1) Preserve patient separation and native image/mask alignment. (2) Target pooled prostate test Dice >=0.75, while reporting all-class overlap and patient means; the illustrated ID-5 interpretation gives pooled test Dice 0.843050, exceeding 0.75. The mapping is image-based and the pooled aggregation was not separately confirmed by the tutor. (3) Compare both models on the same held-out patients and inspect 3-5 actual boundary failures with FP/FN overlays. (4) Target whole-device peak VRAM <=12 GiB and median synchronized model forward <=100 ms per 256x128 slice at batch size 1, plus per-case timing. Allocator peaks do not verify the whole-device limit; validation forward summaries meet the latency target only in their stated scope. (5) Record dependencies, seeds, configurations and validation-only checkpoint selection before test use.

**Risks, budget and fallback.** Risks include small patient cohorts, imbalance, unverified anatomy/slice ordering, absent 3D context and imperfect reproducibility. The proposed budget was two one-hour GPU pilots and two training runs capped at four hours each. Both formal training tasks completed within that cap; the fixed-model test job requests 30 minutes on one GPU. The baseline remains the fallback for an unfavorable accuracy/resource trade-off. The student reports tutor check-off completed on 29 October. ID 5 is inferred as prostate from the image evidence above; the inference is not represented as an official numerical dictionary.

## Code structure and reproducibility

| File or group | Responsibility |
|---|---|
| `modules.py`, `model_factory.py` | Pure-PyTorch model components and architecture registry |
| `dataset.py`, `check_splits.py`, `shape_batching.py` | Paired loading, preprocessing, split checks and native-grid batches |
| `losses.py`, `metrics.py` | Shared differentiable objective and hard-label confusion metrics |
| `train.py`, `training_io.py` | Full training/validation loops, exclusive run outputs, atomic checkpoint files and histories |
| `gpu_profile.py`, `plot_history.py` | Allocator profiling and learning curves |
| `predict.py`, `evaluate.py`, `check_checkpoint.py` | Saved-model prediction, full split evaluation and checkpoint checks |
| `run_comparison.py`, `comparison_train.sbatch`, `comparison_protocol.json` | Frozen matched training and validation orchestration |
| `compare_results.py`, `failure_analysis.py` | Audited comparison tables/plots and paired real error overlays |
| `run_heldout.py`, `heldout_evaluate.sbatch`, `heldout_protocol.json` | Fixed-model full test orchestration and reporting |
| `test_*.py` | Regression checks, including synthetic fixtures and mocked orchestration |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact observed versions, representative commands, source snapshots and evidence locations. Dependencies are PyTorch, NumPy, NiBabel and Matplotlib; the existing environment reported Python 3.11.16, PyTorch runtime 2.14.0+cu130 (distribution metadata 2.14.0), CUDA build 13.0, NumPy 2.4.6, NiBabel 5.4.2 and Matplotlib 3.11.2. This is an observed working environment, not a validated clean-install lock file.

Meaningful checks observed on Rangpur include model shapes/gradients and shared checkpoint reloads; hand-calculated metric cases; paired transforms, losses and native-grid batching; real CPU/GPU smoke runs; 13 evaluation/metric tests; eight matched-run orchestration tests; eight reporting tests; eight qualitative-analysis tests plus five real renders; and nine held-out orchestration tests. The latter tests use mocked evaluation children and synthetic records, so passing them does not establish real test-set results. Full training/validation execution and real-image evidence are recorded separately above. Output directories are preserved rather than silently overwritten; interrupted runs are not automatically resumed.

Source and artifact hashes support consistency checks. The initial validation evidence bundles excluded weights. Both original fixed checkpoints were later downloaded, verified by SHA256 and reloaded for the complete local test evaluation; training was not independently repeated. There is one training seed, only four validation patients and three test patients, no controlled ablation, no whole-device VRAM trace and no claim of bitwise cross-device reproduction. Training/validation input records use filename/size manifests; all 1,080 local test image/mask files were additionally content-hashed and rechecked. CPU/GPU qualitative predictions had some full-matrix differences, retained in their manifest, even though the selected-label counts matched.

## Artificial Intelligence Usage Disclosure

OpenAI ChatGPT and Codex provided substantial assistance with model/data/training/evaluation scaffolding, regression tests, plotting and error-analysis code, frozen protocols, Slurm/Git commands, troubleshooting, literature and specification review, numerical consistency checks and report drafts. Assistance included proposing acceptance criteria and experiments, reviewing shared terminal output and downloaded artifacts, and interpreting measured trade-offs. No use of additional AI tools is asserted without confirmation.

The student executed the local WSL and Rangpur commands and supplied terminal outputs and downloaded result bundles. With AI assistance, verification covered input rejection and pairing tests, patient separation, hand-calculated overlap examples, finite gradients, checkpoint save/reload checks, native-shape coverage, real CPU/GPU smoke runs, matched full-training logs, JSON/CSV numerical recomputation, source/checkpoint/artifact consistency, and visual review of three validation comparison figures and five real error overlays. Nine original held-out tests plus nine local migration checks passed in the new Windows environment. Codex executed the authorized local setup, GPU feasibility checks and real six-pass held-out evaluation; all result/CSV audits passed and both test figures were visually reviewed. The student executed the remote packaging/download commands and provided the inputs. Local execution is recorded explicitly, rather than attributing those agent-run commands to the student.

These checks do not establish an independent manual audit of every line or independent replication. The student remains responsible for reviewing, understanding, correcting and explaining the code, formulas, references, results and disclosures. AI-supported debugging included correcting heterogeneous-slice batching, CSV line endings and an initially empty Git archive; earlier failed attempts are not counted as completed experiments. Current held-out measurements are from the completed local evaluation; remote held-out result files were not reviewed. The student reports tutor check-off completed on 29 October. Codex additionally assisted with image-based ID-5 interpretation, fixed-checkpoint illustrative predictions and the direct baseline comparison in this revision; the student remains responsible for validating and explaining the inference.

## References

[1] Ronneberger, O., Fischer, P. and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [Paper](https://arxiv.org/abs/1505.04597).

[2] Isensee, F. et al. (2018). *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*. [Paper](https://arxiv.org/abs/1802.10508). Architectural inspiration; implementation differences are stated above.

[3] Liu, S., Dowling, J. A., Engstrom, C., Greer, P. B., Crozier, S. and Chandra, S. S. (2020). *Manipulating Medical Image Translation with Manifold Disentanglement*. [Paper](https://arxiv.org/abs/2011.13615). Dataset-related context, not proof of the numerical label mapping.

[4] ACR, ESUR and AdMeTech Foundation (2015). *PI-RADS v2*, anatomy and Figure 1. [Consensus guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC6467207/). Anatomical context only; it does not assign IDs in the course dataset.

[5] *Normal anatomic relationship between urethral sphincter complex and zones of prostrate in young Chinese males on MRI* (2015). [MRI anatomy study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4659132/). Spatial anatomy context, not a course label dictionary.
