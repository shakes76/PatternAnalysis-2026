# Supplementary post-hoc analyses

This document records supplementary post-hoc controls, calibration experiments, and sensitivity analyses conducted after the pre-declared model architecture and evaluation pipeline were frozen. None of the analyses below were used to alter the primary model, its operating threshold, or the recommendations to the project manager.

## 1. Post-hoc comparison: Slice-level vs Patient-level Platt calibration

**Post-hoc comparison: Slice-level vs Patient-level Platt calibration (not in `PREDECLARATION.txt`).**
Because final referral decisions are made per patient, we directly test whether
`strong` SmallCNN's elevated patient ECE (`0.112`) is an artifact of slice-level
Platt fitting by fitting Platt scaling `(a_pat, b_pat)` and the 90%-accuracy
referral threshold `tau_pat` **directly on the 68 calibration patients'
aggregated scores** (each patient's mean slice logit, "mean-logit", or the
logit of each patient's mean uncalibrated slice probability, "mean-prob"), then
applying `(a_pat, b_pat, tau_pat)` unchanged to the 136 test patients. Both
aggregations were computed for pretrained ConvNeXt and `strong` SmallCNN; for
scratch ConvNeXt only mean-logit and for `scanner` SmallCNN only mean-prob were
computed in this post-hoc analysis, so the other two rows are missing rather
than hidden. Test FN / FP and AD recall are not separate measurements: with 44
AD and 92 NC test patients, the reported accuracy and balanced accuracy fix both
error counts uniquely.

ConvNeXt ensembles:

| Calibration | Fitted `a / b` | `tau` | Calibration cov. (auto. acc.) | Test patient acc. (bal. acc.) | Test FN / FP (AD recall) | Test patient ECE | Test cov. (auto. acc.) |
|---|---|---:|---|---|---|---|---|
| **Pretrained**, uncalibrated | `1.000 / 0.000` | — | — | 0.882 (0.854) | 10 / 6 (0.773) | 0.062 | — |
| Pretrained, slice-fitted Platt (pre-declared) | `0.246 / +0.193` | 0.626 | 0.941 (0.906) | **0.882 (0.866)** | **8 / 8 (0.818)** | 0.049 | 0.875 (`119/136`, **0.924**) |
| Pretrained, patient mean-logit Platt (post-hoc) | `0.309 / -0.519` | 0.611 | 0.956 (0.908) | 0.882 (0.842) | 12 / 4 (0.727) | **0.043** | 0.934 (`127/136`, 0.890) |
| Pretrained, patient mean-prob Platt (post-hoc) | `0.549 / -0.241` | 0.561 | 0.956 (0.908) | 0.875 (0.836) | 12 / 5 (0.727) | **0.041** | 0.941 (`128/136`, 0.898) |
| **Scratch**, uncalibrated | `1.000 / 0.000` | — | — | 0.809 (0.782) | 13 / 13 (0.705) | 0.076 | — |
| Scratch, slice-fitted Platt (pre-declared) | `0.354 / +0.500` | 0.836 | 0.397 (0.926) | 0.801 (0.806) | 8 / 19 (0.818) | 0.070 | 0.346 (`47/136`, 0.915) |
| Scratch, patient mean-logit Platt (post-hoc) | `0.449 / -0.324` | 0.682 | 0.735 (0.900) | 0.801 (0.752) | 17 / 10 (0.614) | **0.047** | 0.831 (`113/136`, 0.858) |

SmallCNN ensembles:

| Calibration | Fitted `a / b` | `tau` | Calibration cov. (auto. acc.) | Test patient acc. (bal. acc.) | Test FN / FP (AD recall) | Test patient ECE | Test cov. (auto. acc.) |
|---|---|---:|---|---|---|---|---|
| **`strong`**, uncalibrated | `1.000 / 0.000` | — | — | 0.824 (0.751) | 20 / 4 (0.545) | 0.073 | — |
| `strong`, slice-fitted Platt (pre-declared) | `0.957 / +0.813` | 0.563 | 0.897 (0.902) | 0.875 (0.860) | 8 / 9 (0.818) | 0.112 | 0.912 (`124/136`, 0.903) |
| `strong`, patient mean-logit Platt (post-hoc) | `1.083 / +0.215` | 0.638 | 0.897 (0.902) | 0.846 (0.785) | 17 / 4 (0.614) | **0.038** | 0.868 (`118/136`, 0.898) |
| `strong`, patient mean-prob Platt (post-hoc) | `1.212 / +0.174` | 0.632 | 0.853 (0.914) | 0.838 (0.774) | 18 / 4 (0.591) | **0.055** | 0.890 (`121/136`, 0.884) |
| **`scanner`** (post-hoc model), slice-fitted Platt | `1.071 / -0.063` | 0.508 | 1.000 (0.912) | 0.838 (0.821) | 10 / 12 (0.773) | 0.078 | 0.993 (`135/136`, 0.837) |
| `scanner`, patient mean-prob Platt (post-hoc) | `1.403 / -0.753` | 0.579 | 0.941 (0.906) | 0.838 (0.786) | 16 / 6 (0.636) | **0.057** | 0.926 (`126/136`, 0.865) |

Three findings emerge from this patient-level calibration control:
1. **Consistent with the ECE explanation:** Fitting Platt scaling at the
   patient level matches the `32.4%` patient AD prior and the variance of
   patient-averaged scores, reducing test patient ECE for every ensemble
   (`0.049 -> 0.043/0.041` for pretrained ConvNeXt, `0.070 -> 0.047` for scratch
   ConvNeXt, `0.078 -> 0.057` for `scanner` SmallCNN, and **`0.112 -> 0.038`**
   for `strong` SmallCNN). This supports the reading that `strong` SmallCNN's
   elevated post-Platt patient ECE (`0.112`) came from applying slice-fitted
   Platt parameters (`b = +0.813`) to slice logits before averaging the
   calibrated probabilities per patient, rather than from an
   inability of `SmallCNN` to produce calibrated patient probabilities.
2. **Better-calibrated confidence does not mean fewer missed AD patients:**
   Matching the lower patient AD prior moves every ensemble's decision boundary
   towards NC, so false positives fall and false negatives rise. For the
   pretrained ConvNeXt (mean-logit) total accuracy stays at **0.882**
   (`120/136`) only because 4 fewer NC patients called AD (`8 -> 4`) offset 4
   more missed AD patients (`8 -> 12`): AD recall drops from **0.818 to 0.727**.
   Unchanged total accuracy therefore shows a different trade-off between the
   two error types, not more stable decisions. The other ensembles move the
   same way, with larger losses in AD recall (`strong` SmallCNN `0.818 -> 0.614`,
   `8 -> 17` FN, total `0.875 -> 0.846`; scratch ConvNeXt `0.818 -> 0.614`;
   `scanner` SmallCNN `0.773 -> 0.636`).
3. **Finite-sample trade-off of fitting `(a, b, tau)` on the same 68 patients:**
   When all three parameters (`a_pat, b_pat, tau_pat`) are fitted on the same 68
   calibration patients (`22 AD, 46 NC`), `tau_pat` is slightly optimistic on
   calibration and accepts more test patients: for the pretrained ConvNeXt,
   coverage rises from `87.5%` to `93.4%` but automated test accuracy falls from
   **92.4% to 89.0%** (mean-logit; `89.8%` mean-prob), just below the 90%
   target. Only for `scanner` SmallCNN does patient-level Platt raise automated
   test accuracy (`83.7%` with `1/136` referred to `86.5%` with `10/136`).

These patient-level variants remain post-hoc and do not replace the
pre-declared operating point (slice-fitted Platt scaling, `tau = 0.626`). A larger
calibration cohort, and ideally an AD-recall constraint fitted alongside `tau`, would
be needed before adopting patient-level calibration.



## 2. Asymmetric referral threshold sensitivity analysis

**Post-hoc sensitivity analysis: asymmetric referral (`tau_NC > tau_AD`, not in `PREDECLARATION.txt`).**
The locked test evaluation above used the pre-declared symmetric rule `tau_AD = tau_NC = 0.626`.
Because automating a true AD patient as NC (false negative) is clinically more
harmful than automating an NC patient as AD (which triggers specialist follow-up),
we also inspect an exploratory **post-hoc sensitivity curve** for the asymmetric rule
`REFER if p(AD) < tau_AD and p(NC) < tau_NC` with `tau_AD = 0.626` fixed and four
candidate values of `tau_NC`:

| Exploratory rule on primary ensemble (`tau_AD = 0.626`) | `tau_NC` | Calibration coverage (accepted acc.) | Calibration auto-FN / auto-FP | Test coverage (accepted acc.) | Test auto-FN (`2017-21 AD` missed & automated) / auto-FP |
|---|---:|---|---|---|---:|
| Pre-declared symmetric rule (`tau_NC = tau_AD`) | 0.626 | 0.941 (64/68, acc. 0.906) | 2 / 4 | 0.875 (119/136, acc. 0.924) | 5 / 4 |
| Post-hoc sensitivity point 1 (above cal. FN `p(NC)=0.662`) | 0.750 | 0.824 (56/68, acc. 0.911) | 1 / 4 | 0.801 (109/136, acc. 0.936) | 3 / 4 |
| Post-hoc sensitivity point 2 | 0.800 | 0.794 (54/68, acc. 0.907) | 1 / 4 | 0.779 (106/136, acc. 0.943) | 2 / 4 |
| Post-hoc sensitivity point 3 (above cal. FN `p(NC)=0.910`) | 0.920 | 0.588 (40/68, acc. 0.900) | 0 / 4 | 0.566 (77/136, acc. 0.948) | 0 / 4 |

On the 68-patient calibration set there are only 22 AD patients and only 2 AD
false negatives—at `p(NC) = 0.662` (`p(AD) = 0.338`) and `p(NC) = 0.910`
(`p(AD) = 0.090`)—with zero calibration AD patients in between. Consequently,
calibration data alone cannot distinguish `tau_NC = 0.750` from `tau_NC = 0.800`
(in fact, `0.750` has slightly higher calibration accuracy, `0.911` vs `0.907`,
because moving from `0.750` to `0.800` only refers 2 true NC calibration
patients). We therefore report the table strictly as a post-hoc trade-off curve
rather than claiming a single calibration-selected `tau_NC`: raising `tau_NC`
into the `0.750-0.800` range reduces automated `2017-2021 AD` test false
negatives from `5` to `2-3` at `77.9-80.1%` test coverage (`93.6-94.3%`
accuracy), whereas eliminating all automated AD false negatives (`0/5`,
`tau_NC = 0.920`) cuts test coverage to `56.6%` (`77/136`). Mitigating the
acquisition shortcut more reliably therefore requires protocol-aware referral
and a larger calibration cohort rather than post-hoc confidence thresholding
alone.



## 3. Post-hoc control: SmallCNN with scanner augmentation

### Post-hoc control: SmallCNN with scanner augmentation (not in `PREDECLARATION.txt`)

Added on 2026-10-21, after the locked test run, in response to review. Because
`scanner` augmentation helped `SmallCNN` on folds 4-5 in Round 4 while the
pre-declared five-fold baseline kept `strong` augmentation, folds 1-3 of
`SmallCNN + scanner aug.` were trained with the identical Round-4 recipe (80
epochs, LR 1e-3, weight decay 1e-4, EMA 0.999, final-epoch weights, seed
`3710 + fold`; Slurm 681962, all five checkpoints replaying bit-for-bit) and
evaluated through the identical five-fold ensemble calibration and test pipeline
(Slurm 681963; `results/posthoc_cnn_scanner_20261021/`, stored on Rangpur outside Git). This control was not
used to select or alter the primary model or its threshold.

Development, five folds, same validation patients (mean ± sd):

| Model | Slice acc. | Scan acc. | Patient acc. | Slice / scan AUROC | Patient AUROC | 2017-21 scan AUROC | Patient FP / FN (all folds) |
|---|---:|---:|---:|---|---:|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k | 0.798 ± 0.028 | 0.828 ± 0.039 | 0.840 ± 0.019 | 0.885 ± 0.021 / 0.920 ± 0.014 | 0.905 ± 0.025 | 0.778 ± 0.047 | 24 / 52 |
| SmallCNN, `scanner` aug. (post-hoc) | 0.769 ± 0.016 | 0.808 ± 0.026 | 0.811 ± 0.042 | 0.850 ± 0.020 / 0.910 ± 0.015 | 0.893 ± 0.037 | 0.726 ± 0.090 | 41 / 49 |
| SmallCNN, `strong` aug. (pre-declared) | 0.690 ± 0.032 | 0.701 ± 0.054 | 0.750 ± 0.032 | 0.800 ± 0.068 / 0.853 ± 0.072 | 0.840 ± 0.068 | 0.606 ± 0.081 | 29 / 90 |

| Paired difference (t, 4 df; 2.78 = 5% threshold) | Slice acc. | Scan acc. | Patient acc. | 2017-21 scan AUROC |
|---|---|---|---|---|
| `scanner` - `strong` SmallCNN | +0.079 (t = 4.6) | +0.107 (t = 4.2) | +0.061 (t = 6.7) | +0.121 (t = 4.0) |
| pretrained - `scanner` SmallCNN | +0.029 (t = 3.0) | +0.021 (t = 2.0) | +0.029 (t = 1.5) | +0.052 (t = 2.1) |

Test, 136 patients, same pipeline (calibrated unless marked):

| Five-fold ensemble | Slice / scan / patient acc. (uncal → cal) | Patient bal. acc. (uncal → cal) | Patient AUROC | Platt a / b | τ | Test coverage (automated acc.) | Referred | AD 2017-21 patients automated as NC |
|---|---|---|---:|---|---:|---|---:|---:|
| ConvNeXt-Tiny, ImageNet-1k | 0.862 → 0.867 / 0.879 → 0.889 / 0.882 → 0.882 | 0.854 → 0.866 | 0.944 | 0.246 / 0.193 | 0.626 | 0.875 (0.924) | 17 | 5 |
| SmallCNN, `strong` aug. | 0.783 → 0.826 / 0.814 → 0.873 / 0.824 → 0.875 | 0.751 → 0.860 | 0.927 | 0.957 / 0.813 | 0.563 | 0.912 (0.903) | 12 | 5 |
| SmallCNN, `scanner` aug. (post-hoc) | 0.812 → 0.812 / 0.846 → 0.853 / 0.838 → 0.838 | 0.821 → 0.821 | 0.914 | 1.071 / -0.063 | 0.508 | 0.993 (0.837) | 1 | 9 |

* In five-fold development, `scanner` augmentation improves `SmallCNN` on all
  five folds and narrows its patient-accuracy gap to pretrained ConvNeXt from
  `+0.090 (t = 7.1)` to `+0.029 (t = 1.5, not significant, 3/5 folds)`, with
  pretrained ConvNeXt remaining ahead on all five folds at slice (`+0.029, t = 3.0`)
  and scan (`+0.021, t = 2.0`) levels.
* On the 136 test patients, however, `SmallCNN + scanner` reaches `0.838 [0.772, 0.897]`
  patient accuracy (`114/136`, 12 FP, 10 FN; AD precision / recall `0.739 / 0.773`,
  NC precision / recall `0.889 / 0.870`, macro-F1 `0.817`, confusion `[80 12; 10 34]`;
  slice / scan / patient AUROC `0.894 / 0.931 / 0.914 [0.859, 0.958]`; post-Platt
  slice / scan / patient ECE `0.022 / 0.064 / 0.078`, or `0.057` under
  patient-fitted Platt; 2017-2021 scan acc. / AUROC `0.804 / 0.805`, `14/27` AD
  scans missed; confidence AUROC `0.821`, AURC `0.050`), trailing pretrained
  ConvNeXt by `+0.044 [+0.007, +0.088]` (McNemar `7 / 1, p = 0.070`) and
  calibrated `strong` SmallCNN by `+0.037 [-0.015, +0.088]` (McNemar
  `9 / 4, p = 0.267`). Because all 68 calibration patients already reach `91.2%`
  accuracy (`62/68` at coverage `1.000`), τ drops to `0.508`, referring only
  `1/136` test patients and leaving automated test accuracy at `83.7%`
  (`113/135`, below the 90% target, with `9/16` `2017-2021 AD` patients
  automated as NC).
