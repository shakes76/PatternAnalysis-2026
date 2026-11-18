# Development experiments

All numbers below are copied from the `metrics.json` files of completed Rangpur
runs (NVIDIA A100-PCIE-40GB). Models in Rounds 1-4 and Round 7 were trained
from random initialization; Rounds 5-6 evaluate ConvNeXt-Tiny initialized from
official ImageNet-1k weights. None of the development runs below used the
calibration or test patients.

Two evaluation sets appear:

* **Outer validation (fold 1)**: 95 held-out development patients, 211 scans,
  4,220 slices. Each trained model is scored on it once.
* **Inner early-stop (fold 1)**: the 38 validation patients (12 AD) that also
  select the checkpoint epoch, 94 scans, 1,880 slices. These scores are optimistic and
  noisy; always predicting NC already gives 0.628 slice / 0.684 patient accuracy.

## Outer validation

| Model | Augmentation | Selected epoch | Slice acc. | Scan acc. | Slice AUROC |
|---|---|---:|---:|---:|---:|
| SmallCNN (LR 1e-3), seed 3710 | none | 14 | 0.809 | 0.834 | 0.882 |
| ConvNeXt-Tiny | none | 4 | 0.634 | 0.668 | 0.693 |
| ConvNeXt-Tiny | light (5 deg rotation, 3% shift) | 8 | 0.726 | 0.768 | 0.801 |

SmallCNN five-fold cross-validation, mean outer-validation slice accuracy over
the five folds: 0.735 (seed 3710), 0.753 (seed 4710), 0.735 (seed 5710); mean
scan accuracy 0.771, 0.774, 0.763.


### Earlier development screens (fold 1)

Development results before the current recipe (from `metrics.json`; full table
below), fold-1 outer validation, 95 patients:

| Model | Augmentation | Slice accuracy | Scan accuracy | Slice AUROC |
|---|---|---:|---:|---:|
| SmallCNN | none | 0.809 | 0.834 | 0.882 |
| ConvNeXt-Tiny | none | 0.634 | 0.668 | 0.693 |
| ConvNeXt-Tiny | light | 0.726 | 0.768 | 0.801 |

First run of the current recipe (`config/experiments_strong_aug.json`, two seeds,
same 95 patients, checkpoint chosen on the 38-patient inner `early_stop`
subset): ConvNeXt-Lite with three-slice context reached slice accuracy
0.717 / 0.740 and scan accuracy 0.754 / 0.810; details below. Slice accuracy is still below 0.80.

Second fold-1 screen (`config/experiments_round2.json`, 80 epochs, mean ± sd
over two seeds, same 95 patients):

| Model | Slice accuracy | Scan accuracy | Patient accuracy | Slice AUROC |
|---|---:|---:|---:|---:|
| SmallCNN, three-slice context | 0.735 ± 0.008 | 0.791 ± 0.020 | 0.779 ± 0.030 | 0.841 ± 0.048 |
| ConvNeXt-Lite, single slice | 0.714 ± 0.004 | 0.780 ± 0.010 | 0.753 ± 0.007 | 0.783 ± 0.004 |
| ConvNeXt-Lite, three-slice context | 0.728 ± 0.019 | 0.780 ± 0.030 | 0.742 ± 0.022 | 0.799 ± 0.022 |
| ConvNeXt-Tiny, three-slice context | 0.742 ± 0.001 | 0.799 ± 0.023 | 0.768 ± 0.030 | 0.812 ± 0.008 |

The scratch ConvNeXt memorised the training slices within a few epochs
(training accuracy 1.00) and optimizer/regularisation tweaks (including mixup [3])
stayed within seed noise; augmentation was the only clear improvement. Because the 38-patient
`early_stop` subset proved noisy as a checkpoint selector (scan AUROC 0.64-0.73
vs 0.84-0.87 on outer validation), subsequent rounds merged `early_stop` into
training and kept the final-epoch EMA weights.


## Inner early-stop screens (fold 1)

| Setting (two seeds where listed) | Slice acc. | Patient acc. | Train acc. at end |
|---|---|---|---|
| ConvNeXt-Lite, no augmentation | 0.628 | 0.789 | 0.99 |
| ConvNeXt-Lite, integer shift / gamma / both | 0.613 / 0.575 / 0.629 | 0.658 / 0.632 / 0.658 | 0.95-0.99 |
| ConvNeXt-Tiny reference, constant LR 1e-4 | 0.600 / 0.581 | 0.737 / 0.658 | 1.00 |
| + label smoothing 0.05 / 0.10 | 0.596 / 0.587 | 0.684 / 0.684 | 1.00 |
| + Mixup 0.1 / 0.2 | 0.597 / 0.616 | 0.763 / 0.763 | 1.00 |
| cosine LR | 0.564 / 0.580 | 0.632 / 0.658 | 1.00 |
| cosine, peak LR 3e-4 | 0.441 / 0.528 | 0.500 / 0.605 | 1.00 |
| cosine, weight decay 0.01 | 0.569 / 0.579 | 0.658 / 0.658 | 1.00 |
| cosine, DropPath 0.2 | 0.460 / 0.593 | 0.553 / 0.737 | 1.00 |
| SmallCNN, LR 1e-4 | 0.766 / 0.745 | 0.816 / 0.763 | 0.83 / 0.80 |

## What these runs show

* The scratch ConvNeXt memorises the training slices: training accuracy reaches
  about 0.93 by epoch 4 and 1.00 soon after, while early-stop log loss rises.
  The selected epoch is usually between 1 and 4.
* Optimizer, schedule, smoothing and DropPath changes stay within seed-to-seed
  noise on the 38 early-stop patients. A peak learning rate of 3e-4 makes
  training collapse.
* The only clear gain came from augmentation on outer validation (0.634 to 0.726).
* The baseline CNN under-fits rather than memorises, and outperforms ConvNeXt
  with about 1/285 of its parameters (97,521 vs 27,817,825).

These observations motivated the plan
[`config/experiments_strong_aug.json`](../config/experiments_strong_aug.json):
stronger training augmentation, three-slice context and an EMA of the weights,
compared on the 95 outer-validation patients.

## Strong augmentation screen (fold 1, outer validation, 95 patients)

All cases: warm-up + cosine LR over at most 60 epochs, patience 15, checkpoint
chosen by minimum early-stop scan log loss. Values are seed 3710 / seed 4710.

| Case | Selected / last epoch | Slice acc. | Slice AUROC | Scan acc. | Patient acc. | Patient AD recall |
|---|---|---|---|---|---|---|
| S1 Lite, light aug., Mixup 0.2, DropPath 0.2, no EMA | 12/27, 14/29 | 0.706 / 0.719 | 0.773 / 0.787 | 0.782 / 0.758 | 0.779 / 0.737 | 0.677 / 0.645 |
| S2 Lite, strong aug., single slice, EMA | 1/16, 1/16 | 0.570 / 0.594 | 0.598 / 0.620 | 0.592 / 0.640 | 0.579 / 0.621 | 0.645 / 0.677 |
| S3 Lite, strong aug., three-slice context, EMA | 23/38, 23/38 | 0.717 / 0.740 | 0.789 / 0.812 | 0.754 / 0.810 | 0.716 / 0.758 | 0.774 / 0.806 |
| S4 Tiny, as S3 | 16/31, 22/37 | 0.719 / 0.738 | 0.796 / 0.812 | 0.768 / 0.773 | 0.716 / 0.726 | 0.806 / 0.806 |
| S5 as S3 plus Mixup 0.2 | 30/45, 23/38 | 0.728 / 0.727 | 0.802 / 0.802 | 0.791 / 0.791 | 0.747 / 0.726 | 0.742 / 0.806 |
| B1 SmallCNN, as S3 (LR 1e-3) | 48/60, 8/23 | 0.751 / 0.673 | 0.857 / 0.792 | 0.806 / 0.711 | 0.811 / 0.768 | 0.484 / 0.323 |

* With strong augmentation ConvNeXt no longer memorises quickly: the selected
  epoch moved from 1-4 to 16-30 and training accuracy ended near 0.85.
* S2 selected epoch 1 for both seeds: an almost constant output near p = 0.5
  gives scan log loss ~0.693, which later, more decisive epochs did not beat
  within the patience window. S2 is therefore not a valid control.
* ConvNeXt errors are mostly NC patients called AD (S3, seed 4710: 17 false
  positives, 6 false negatives); the CNN is biased the other way.
* Averaging the predictions of several models of this fold changed slice
  accuracy by at most 0.01.

## Round 2: early-stop AUROC selection, full schedule (fold 1, 95 patients)

[`config/experiments_round2.json`](../config/experiments_round2.json) (Slurm
jobs 646976 and 646977): every case runs the full 80-epoch warm-up + cosine
schedule; the checkpoint is the epoch with the highest early-stop scan AUROC.
Values are seed 3710 / seed 4710; every checkpoint replayed exactly.

| Case | Selected epoch | Slice acc. | Slice AUROC | Scan acc. | Patient acc. | Patient FP / FN | Final train acc. |
|---|---|---|---|---|---|---|---|
| S2 Lite, single slice | 76, 76 | 0.716 / 0.711 | 0.780 / 0.786 | 0.787 / 0.773 | 0.758 / 0.747 | 14 / 9, 11 / 13 | 0.870 / 0.883 |
| S3 Lite, three-slice context | 63, 25 | 0.714 / 0.741 | 0.784 / 0.815 | 0.758 / 0.801 | 0.726 / 0.758 | 15 / 11, 17 / 6 | 0.914 / 0.902 |
| S4 Tiny, three-slice context | 78, 80 | 0.742 / 0.741 | 0.817 / 0.806 | 0.782 / 0.815 | 0.747 / 0.789 | 14 / 10, 14 / 6 | 0.957 / 0.959 |
| B1 SmallCNN, three-slice context (LR 1e-3) | 69, 48 | 0.740 / 0.729 | 0.875 / 0.807 | 0.806 / 0.777 | 0.800 / 0.758 | 2 / 17, 9 / 14 | 0.890 / 0.894 |

* No configuration exceeded 0.742 slice accuracy. ConvNeXt-Tiny with context
  is the most stable ConvNeXt (0.742 / 0.741) and its scan accuracy averages
  0.799; paired with the CNN on the same seed it differs by +0.007 slice and
  +0.007 scan accuracy on average.
* Three-slice context against single slice: +0.014 mean slice accuracy, smaller
  than the 0.027 difference between the two S3 seeds.
* The 38 early-stop patients were a weak selector: their best scan AUROC was
  0.64-0.73 for ConvNeXt against 0.84-0.87 on the 95 validation patients, and
  the chosen epoch was usually one of the last few (S3 seed 4710 picked 25).
* Tiny reaches 0.96 training accuracy against 0.74 validation slice accuracy.
* Training peak memory: Tiny 4,656 MiB, Lite about 1,580 MiB, CNN 727 MiB
  (A100, batch 32, 240x256, three channels).

## Round 3: five-fold comparison with final-epoch weights

Because the early-stop subset added noise but little information, it is merged
into training and each model keeps its weights after a fixed 80 epochs; the
validation patients are only monitored and used for comparison.
[`config/experiments_round3_cv.json`](../config/experiments_round3_cv.json)
trains ConvNeXt-Tiny (three-slice context) and the CNN baseline on all five
folds with the same pipeline (Slurm jobs 649955 and 649956; all ten
checkpoints replayed exactly). Each fold is scored on its own validation
patients (95-96), so the five folds cover all 476 development patients.

| Fold | Slice acc. (ConvNeXt / CNN) | Scan acc. | Patient acc. | Slice AUROC | Final train acc. |
|---:|---|---|---|---|---|---|
| 1 | 0.739 / 0.665 | 0.815 / 0.630 | 0.789 / 0.737 | 0.818 / 0.850 | 0.954 / 0.878 |
| 2 | 0.708 / 0.671 | 0.765 / 0.685 | 0.832 / 0.747 | 0.760 / 0.725 | 0.961 / 0.873 |
| 3 | 0.693 / 0.716 | 0.721 / 0.699 | 0.768 / 0.789 | 0.788 / 0.867 | 0.949 / 0.885 |
| 4 | 0.749 / 0.733 | 0.767 / 0.781 | 0.771 / 0.771 | 0.833 / 0.831 | 0.953 / 0.880 |
| 5 | 0.685 / 0.664 | 0.672 / 0.708 | 0.747 / 0.705 | 0.794 / 0.728 | 0.964 / 0.889 |
| mean ± sd | 0.715 ± 0.028 / 0.690 ± 0.032 | 0.748 ± 0.054 / 0.701 ± 0.054 | 0.782 ± 0.032 / 0.750 ± 0.032 | 0.799 ± 0.028 / 0.800 ± 0.068 | |

* Paired differences (ConvNeXt - CNN): slice +0.025 ± 0.035 (4/5 folds),
  scan +0.047 ± 0.088 (3/5), patient +0.032 ± 0.042 (3/5); paired t = 1.59,
  1.20 and 1.68 with 4 degrees of freedom, below the 2.78 needed at the 5% level.
  AUROC differences are within ±0.011.
* Patient errors over all folds: ConvNeXt 45 false positives / 59 false
  negatives, CNN 29 / 90. Same ranking ability, different operating point.

Per fold (ConvNeXt / CNN accuracy):

| Fold | Slice | Scan | Patient | Patient FP / FN (ConvNeXt; CNN) |
|---:|---|---|---|---|
| 1 | 0.739 / 0.665 | 0.815 / 0.630 | 0.789 / 0.737 | 11 / 9; 0 / 25 |
| 2 | 0.708 / 0.671 | 0.765 / 0.685 | 0.832 / 0.747 | 3 / 13; 6 / 18 |
| 3 | 0.693 / 0.716 | 0.721 / 0.699 | 0.768 / 0.789 | 12 / 10; 2 / 18 |
| 4 | 0.749 / 0.733 | 0.767 / 0.781 | 0.771 / 0.771 | 13 / 9; 5 / 17 |
| 5 | 0.685 / 0.664 | 0.672 / 0.708 | 0.747 / 0.705 | 6 / 18; 16 / 12 |



* Fold 1 slice accuracy (0.739) matches the round-2 Tiny runs (0.742 / 0.741),
  so merging the early-stop patients into training and keeping the final epoch
  neither helped nor hurt there.
* Variation between folds (slice sd 0.028) is larger than between the two
  round-2 seeds on fold 1 (0.001); for scan and patient accuracy the two are
  similar (seed sd 0.023 / 0.030 vs fold sd 0.054 / 0.032). With ~95 patients a
  patient accuracy near 0.78 has a binomial standard error of about 0.04.
* Most of the fold-to-fold spread and most errors trace back to the
  acquisition confound described in [ACQUISITION_CONFOUND.md](ACQUISITION_CONFOUND.md):
  AD scans acquired 2017-2021 (the same period as all NC scans) are missed far
  more often, and scan AUROC on the 2017-2021 subset is 0.729 (ConvNeXt) and
  0.654 (CNN) instead of 0.850 and 0.839.

## Rounds 4 and 5: scanner-style augmentation and ImageNet initialisation (folds 4 and 5)



Two attempts to weaken the acquisition shortcut were screened on folds 4 and 5
(one seed each; details below).
Validation accuracy slice / scan / patient, and scan AUROC on the 2017-2021
validation scans:

| Run | Fold 4 | Fold 5 | 2017-21 scan AUROC (f4, f5) |
|---|---|---|---|
| ConvNeXt-Tiny scratch, strong aug. (round 3) | 0.749 / 0.767 / 0.771 | 0.685 / 0.672 / 0.747 | 0.818, 0.696 |
| SmallCNN, strong aug. (round 3) | 0.733 / 0.781 / 0.771 | 0.664 / 0.708 / 0.705 | 0.718, 0.560 |
| ConvNeXt-Tiny scratch, scanner aug. | 0.738 / 0.763 / 0.729 | 0.720 / 0.774 / 0.789 | 0.779, 0.725 |
| SmallCNN, scanner aug. | 0.758 / 0.828 / 0.823 | 0.756 / 0.764 / 0.768 | 0.752, 0.710 |
| ConvNeXt-Tiny ImageNet-1k, strong aug. | 0.817 / 0.847 / 0.844 | 0.773 / 0.774 / 0.821 | 0.770, 0.740 |
| ConvNeXt-Tiny ImageNet-1k, scanner aug. | 0.801 / 0.814 / 0.802 | 0.773 / 0.785 / 0.789 | 0.745, 0.719 |

* ImageNet initialisation (cited weights, 40 epochs) gives the best overall
  accuracy so far, but discrimination within 2017-2021 stays at the level of
  the scratch model: the gain comes mostly from scans whose acquisition style
  already identifies the class.
* Scanner-style augmentation helps the CNN on plain accuracy and 2017-2021 scan
  AUROC (though on fold 5 it misses 26/26 `2017-2021 AD` scans, reducing fold-5
  2017-2021 scan balanced accuracy to 0.452), is mixed for scratch ConvNeXt and
  adds nothing to the pretrained model. We retained `strong` augmentation across
  all three pre-declared five-fold ensembles so they share the same split,
  augmentation and evaluation pipeline (learning rate, epochs and weight decay
  still differ, so this is not a single-variable comparison), and later ran a post-hoc five-fold `SmallCNN + scanner aug.`
  control after the locked test run (see *Post-hoc control: SmallCNN with
  scanner augmentation* below).



Two attempts to weaken the acquisition shortcut, screened on folds 4 and 5
(fold 5 was the weakest fold and has the most 2017-2021 AD validation scans).
All runs keep the round-3 pipeline (three-slice context, EMA 0.999,
final-epoch weights) and differ only as listed; every checkpoint replayed
exactly.

* Round 4, [`config/experiments_round4_scanner_aug.json`](../config/experiments_round4_scanner_aug.json)
  (Slurm 653053): the `scanner` augmentation profile (random bias field,
  noise inside the head, blur/sharpening, stronger gain/gamma), 80 epochs,
  ConvNeXt-Tiny from scratch and the CNN.
* Round 5, [`config/experiments_round5_pretrained.json`](../config/experiments_round5_pretrained.json)
  (Slurm 653854): ConvNeXt-Tiny initialised from the official ImageNet-1k
  weights (Liu et al. 2022; SHA-256 in the README), 40 epochs, learning rate
  5e-5, with the strong or the scanner profile.

Validation accuracy (slice / scan / patient) and the 2017-2021 subset (scan
AUROC, AD scans acquired 2017-2021 that were called NC):

| Run | Fold 4 | Fold 5 | 2017-21 scan AUROC (f4, f5) | 2017-21 AD missed (f4, f5) |
|---|---|---|---|---|
| Tiny scratch, strong (round 3) | 0.749 / 0.767 / 0.771 | 0.685 / 0.672 / 0.747 | 0.818, 0.696 | 10/24, 24/26 |
| CNN, strong (round 3) | 0.733 / 0.781 / 0.771 | 0.664 / 0.708 / 0.705 | 0.718, 0.560 | 22/24, 16/26 |
| Tiny scratch, scanner | 0.738 / 0.763 / 0.729 | 0.720 / 0.774 / 0.789 | 0.779, 0.725 | 11/24, 9/26 |
| CNN, scanner | 0.758 / 0.828 / 0.823 | 0.756 / 0.764 / 0.768 | 0.752, 0.710 | 17/24, 26/26 |
| Tiny ImageNet, strong | **0.817 / 0.847 / 0.844** | 0.773 / 0.774 / 0.821 | 0.770, 0.740 | 14/24, 22/26 |
| Tiny ImageNet, scanner | 0.801 / 0.814 / 0.802 | 0.773 / 0.785 / 0.789 | 0.745, 0.719 | 16/24, 20/26 |

(2017-2021 validation scans: 135 with 24 AD on fold 4, 131 with 26 AD on fold 5.)

* ImageNet initialisation gives the highest overall accuracy seen so far
  (fold 4: 0.817 / 0.847 / 0.844, above 0.80 at slice, scan and patient
  level; fold 5: slice 0.773) and the highest scan AUROC (0.900-0.920), in half
  the epochs. Training accuracy reaches 0.999-1.00.
* It does **not** improve discrimination inside 2017-2021 beyond the scratch
  model (AUROC 0.72-0.77 vs 0.70-0.82), and it still misses most 2017-2021 AD
  scans on fold 5. The gain is concentrated on scans whose acquisition style
  already identifies the class.
* The scanner profile helps the CNN clearly on both folds (+0.025 / +0.092
  slice accuracy, 2017-21 AUROC +0.03 / +0.15), gives mixed results for scratch
  ConvNeXt (fold 4 worse, fold 5 better, and on fold 5 it finds 17 of 26
  2017-2021 AD scans instead of 2) and no benefit for the pretrained model.
* Two folds and one seed per configuration: differences of a few points are
  within the fold/seed noise measured in rounds 2 and 3. Rounds 4 and 5 are
  screens, not a model selection; round 6 runs the pretrained model on all
  five folds for a like-for-like comparison with round 3.

## Round 6: five-fold ImageNet-initialised ConvNeXt-Tiny

[`config/experiments_round6_pretrained_cv.json`](../config/experiments_round6_pretrained_cv.json)
(Slurm 656071 tests+smoke, 656072): the round-5 recipe (official ImageNet-1k
weights, 40 epochs, peak LR 5e-5, strong augmentation, three-slice context,
EMA 0.999, final-epoch weights, seed 3710) on all five folds, compared with
round 3 on identical patients. All five checkpoints replayed exactly; folds 4
and 5 reproduce the round-5 numbers exactly.

| Fold | Slice acc. | Scan acc. | Patient acc. | Scan AUROC | Patient FP / FN | 2017-21 scan AUROC | 2017-21 AD missed |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.836 | 0.877 | 0.863 | 0.937 | 7 / 6 | 0.741 | 7/10 |
| 2 | 0.778 | 0.810 | 0.821 | 0.929 | 1 / 16 | 0.856 | 15/17 |
| 3 | 0.783 | 0.834 | 0.853 | 0.914 | 7 / 7 | 0.785 | 5/9 |
| 4 | 0.817 | 0.847 | 0.844 | 0.920 | 5 / 10 | 0.770 | 14/24 |
| 5 | 0.773 | 0.774 | 0.821 | 0.900 | 4 / 13 | 0.740 | 22/26 |
| mean ± sd | 0.798 ± 0.028 | 0.828 ± 0.039 | 0.840 ± 0.019 | 0.920 ± 0.014 | 24 / 52 total | 0.778 ± 0.047 | 63/86 |

Paired differences across the five folds (paired t, 4 df; 2.78 = 5% two-sided threshold):

| Comparison | Slice acc. | Scan acc. | Patient acc. | 2017-21 scan AUROC |
|---|---|---|---|---|
| pretrained - scratch ConvNeXt-Tiny | +0.083 (t = 14.3) | +0.080 (t = 6.4) | +0.059 (t = 3.4) | +0.042 (t = 1.5) |
| pretrained - SmallCNN | +0.108 (t = 6.1) | +0.128 (t = 3.9) | +0.090 (t = 7.1) | +0.172 (t = 5.1) |
| scratch ConvNeXt-Tiny - SmallCNN | +0.025 (t = 1.6) | +0.047 (t = 1.2) | +0.032 (t = 1.7) | +0.130 (t = 5.4) |

Out-of-fold scan error by acquisition group (all 1,050 development scans):

| Group | Pretrained Tiny | Scratch Tiny | SmallCNN |
|---|---:|---:|---:|
| AD acquired 2011-2016 (417) | 0.173 | 0.300 | 0.451 |
| AD acquired 2017-2021 (86) | 0.733 | 0.674 | 0.837 |
| NC (547) | 0.080 | 0.148 | 0.099 |

* Among the pre-declared configurations, ImageNet initialisation is the first
  that beats both Round-3 scratch models on every fold at slice and scan level,
  with mean slice accuracy 0.798 (3 of 5 folds >= 0.80) and patient accuracy
  0.840 (all folds >= 0.82).
* The gain is concentrated on scans that the acquisition style already separates
  (earlier AD and NC errors roughly halve). On AD scans acquired 2017-2021 the
  error is unchanged (0.73) and the 2017-21 AUROC gain over the scratch model is
  not significant (+0.042, t = 1.5). It separates those scans better than the
  Round-3 `strong`-augmented CNN (+0.172, t = 5.1).
* Training accuracy reaches 1.00; one seed per fold.

### Why the folds differ

**Why the folds differ.** Per-fold slice / scan / patient accuracy of the
pretrained model, and the number of 2017-2021 AD scans in each validation fold:

| Fold | Slice acc. | Scan acc. | Patient acc. | 2017-21 AD scans in fold (missed) |
|---:|---:|---:|---:|---:|
| 1 | 0.836 | 0.877 | 0.863 | 10 (7) |
| 2 | 0.778 | 0.810 | 0.821 | 17 (15) |
| 3 | 0.783 | 0.834 | 0.853 | 9 (5) |
| 4 | 0.817 | 0.847 | 0.844 | 24 (14) |
| 5 | 0.773 | 0.774 | 0.821 | 26 (22) |

Fold-to-fold variance stems from (1) ~95 validation patients per fold (±0.04
sampling error), (2) an uneven share of the hard 2017-2021 AD group (9 to 26
scans per fold), and (3) seed-to-seed training variance: in Round 2 (two seeds
on fold 1 for the scratch model), slice-level seed sd was tiny (0.001) but scan
(0.023) and patient (0.030) seed sd were comparable to across-fold patient sd
(0.019-0.032) because a few borderline patients flipping across the 0.5
threshold move patient accuracy by ~1 pp each (the five-fold comparisons use one
deterministic seed `3710 + fold` per fold). Pooled out-of-fold metrics across
all 476 development patients (`pooled_out_of_fold_round3_round6.json`) match the
fold means: pretrained ConvNeXt-Tiny `0.798` slice / `0.830` scan / `0.840`
patient accuracy, scratch ConvNeXt-Tiny `0.715 / 0.749 / 0.782`, and SmallCNN
`0.691 / 0.701 / 0.750`.



## Round 7 (post-hoc control): five-fold SmallCNN with scanner augmentation

Added on 2026-10-21 after the locked test evaluation (`Slurm 681962`), in
response to review. Because `scanner` augmentation improved `SmallCNN` on folds
4 and 5 in Round 4 while the pre-declared five-fold baseline retained `strong`
augmentation, folds 1-3 of `SmallCNN` were trained with the identical Round-4
recipe (80 epochs, LR 1e-3, weight decay 1e-4, three-slice context, `scanner`
augmentation, EMA 0.999, final-epoch weights, seed `3710 + fold`) and combined
with the Round-4 fold-4 and fold-5 checkpoints (`results/posthoc_cnn_scanner_20261021/`).
All five checkpoints replayed bit-for-bit.

| Fold | Slice acc. / AUROC | Scan acc. / AUROC | Patient acc. / AUROC | Patient FP / FN | Train acc. | 2017-21 scan AUROC | 2017-21 scan bal. acc. |
|---:|---|---|---|---|---:|---:|---:|
| 1 | 0.793 / 0.878 | 0.815 / 0.926 | 0.768 / 0.869 | 16 / 6 | 0.846 | 0.637 | 0.525 |
| 2 | 0.759 / 0.828 | 0.805 / 0.912 | 0.832 / 0.934 | 4 / 12 | 0.849 | 0.867 | 0.616 |
| 3 | 0.776 / 0.863 | 0.825 / 0.923 | 0.863 / 0.932 | 7 / 6 | 0.859 | 0.668 | 0.545 |
| 4 | 0.758 / 0.841 | 0.828 / 0.895 | 0.823 / 0.870 | 8 / 9 | 0.840 | 0.752 | 0.587 |
| 5 | 0.756 / 0.838 | 0.764 / 0.893 | 0.768 / 0.858 | 6 / 16 | 0.855 | 0.710 | 0.452 |
| mean ± sd | 0.769 ± 0.016 / 0.850 ± 0.020 | 0.808 ± 0.026 / 0.910 ± 0.015 | 0.811 ± 0.042 / 0.893 ± 0.037 | 41 / 49 total | 0.850 ± 0.007 | 0.726 ± 0.090 | 0.545 ± 0.063 |

Paired differences over the five folds (4 df; 2.78 = 5% two-sided threshold):

| Comparison | Slice acc. | Scan acc. | Patient acc. | 2017-21 scan AUROC |
|---|---|---|---|---|
| `SmallCNN (scanner)` - `SmallCNN (strong, R3)` | +0.079 (t = 4.6, 5/5 folds) | +0.107 (t = 4.2, 5/5 folds) | +0.061 (t = 6.7, 5/5 folds) | +0.121 (t = 4.0, 5/5 folds) |
| `Pretrained Tiny (R6)` - `SmallCNN (scanner)` | +0.029 (t = 3.0, 5/5 folds) | +0.021 (t = 2.0, 5/5 folds) | +0.029 (t = 1.5, 3/5 folds) | +0.052 (t = 2.1, 4/5 folds) |

* `Scanner` augmentation improves `SmallCNN` on every fold across slice, scan
  and patient accuracy as well as 2017-2021 scan AUROC, reducing total patient
  false negatives across the five folds from 90 (`strong`) to 49 (`scanner`,
  while false positives rise from 29 to 41).
* Against `SmallCNN + scanner`, the ImageNet-initialised ConvNeXt-Tiny stays
  ahead on all five folds at slice (+0.029, t = 3.0) and scan (+0.021, t = 2.0)
  levels, on 4 of 5 folds for 2017-2021 scan AUROC (+0.052, t = 2.1), and on 3
  of 5 folds at patient level (+0.029, t = 1.5, not significant). See Section
  9.4 of `REPORT.md` for the corresponding five-fold ensemble calibration and
  test evaluation (`Slurm 681963`).
