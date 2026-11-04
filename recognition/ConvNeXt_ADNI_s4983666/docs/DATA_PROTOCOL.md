# Data splitting and leakage prevention

The question is whether a model trained on some patients can classify MRI
slices of **patients it has never seen**. Therefore every scan and every slice
of a patient stays in one partition.

## Dataset

The course copy at `/home/groups/comp3710/ADNI` contains 680 patients, 1,526
scans and 30,520 grayscale 256 x 240 JPEG slices (20 consecutive sagittal slices
per scan). The filename prefix is the scan ID; the patient ID (`NNN_S_NNNN`) is
read from the metadata's `raw` path. The course's own `AD_NC/train` and
`AD_NC/test` folders share 216 patients (with different scans), so they are
used only as file locations: all images are indexed together and roles are
reassigned per patient.

## Partitions

`adni_splits.py prepare` (seed 3710) assigns patients once, stratified by
diagnosis history:

| Partition | Patients | Scans | Slices | Used for |
|---|---:|---:|---:|---|
| development | 476 | 1,050 | 21,000 | training and model comparison (validation) |
| calibration | 68 | 170 | 3,400 | Platt scaling and the referral threshold, after the model is frozen |
| test | 136 | 306 | 6,120 | one final evaluation of the frozen pipeline |

Development patients are divided into five folds. This gives the usual three
patient-disjoint sets: train, validation and test. For fold *k*:

| Role | Patients (fold 1) | Used for |
|---|---:|---|
| `train` | 381 | updating the weights (`train.csv` + `early_stop.csv` of the fold) |
| `val` | 95 | monitoring every epoch and comparing configurations |

Every model trains for a fixed number of epochs and the weights after the last
epoch are saved (`final.pt`), so no validation score chooses a checkpoint.
Validation scores are used only to compare configurations (for example
ConvNeXt against the CNN baseline on the same patients). Calibration patients
fit Platt scaling and the referral threshold of the frozen model; test
patients are only used by the single final evaluation in
`predict.py --role test`.

The frozen manifests still contain a small `early_stop.csv` per fold (38
patients in fold 1). Earlier screens used it to pick the checkpoint epoch;
with only 38 patients that choice was noisy, so it is now merged into
training when a fold is loaded. The manifests themselves are unchanged.

## Checks

`adni_splits.py verify` re-reads every source image and the metadata and
recomputes the assignment from the seed. It fails if any of the following holds:

* a patient, scan, file path, file hash or decoded-pixel hash appears in two
  roles of a fold, or in development and calibration/test;
* an outer-validation image is missing or appears twice across the five folds;
* a scan does not have exactly 20 slices, labels disagree with the metadata,
  or any manifest/source byte changed after preparation.

`train.py` and `predict.py` run this verification before loading any rows.

```bash
python3 adni_splits.py prepare --data-root /home/groups/comp3710/ADNI \
    --output "$HOME/comp3710/adni_splits_v1" --folds 5 --seed 3710
python3 adni_splits.py verify --data-root /home/groups/comp3710/ADNI \
    --output "$HOME/comp3710/adni_splits_v1"
```

## Limits

Hashes detect exact duplicates only, not near-duplicates. The checks rely on
the patient IDs in the supplied metadata and cannot reveal upstream
preprocessing that may have used all patients. The task is AD/NC
classification of current scans; it does not predict future conversion and
does not establish generalisation to other hospitals or scanners.

The patient split is stratified by diagnosis only. Acquisition period and
sequence, which can be read from the metadata file names, are strongly
associated with the diagnosis in this data (every NC scan was acquired
2017-2021; every scan acquired 2011-2016 is AD); see
[ACQUISITION_CONFOUND.md](ACQUISITION_CONFOUND.md).

## Manifest verification (2026-10-26 re-check)

An independent re-check of the frozen manifests (2026-10-26) found the
following:

* No patient ID, scan ID, file hash or decoded-pixel hash is shared between
  development, calibration and test.
* No patient appears in the validation set of two folds, and no fold's
  training or inner-early-stop patients appear in its own validation set.
* All 30,520 slices have distinct pixel hashes, so no image is duplicated
  under another patient.
* No patient has more than one label.

These checks have limits:

* Hashes detect exact copies only, not near-duplicates.
* The skull stripping and slice extraction were done upstream by the course
  and cannot be audited.

## Scanner-style augmentation profile

The `scanner` profile (screened in Round 4 and used in the post-hoc SmallCNN
control) is selected *instead of* `strong`, not stacked on top of it, but it
keeps every `strong` operation (same flip, crop, rotation and translation) and
changes or adds the following, applied in this order to each training image:
intensity gain widened from ±10% to ±20% and gamma from ±15% to ±30%; a smooth
multiplicative bias field (a random 4×4 grid of factors drawn from [0.7, 1],
bicubically upsampled to the image; applied to every image); with probability
0.3 an unsharp mask (radius 1 px, strength 50-150%), otherwise a Gaussian blur
with radius drawn from [0, 1] px; and zero-mean Gaussian noise added inside the
head only (standard deviation drawn from [0, 3%] of the intensity range; applied
to every image). The black background stays zero and all three context channels
receive the same draw (`adni_data/augmentation.py`).
