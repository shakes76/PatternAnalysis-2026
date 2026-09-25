# ADNI Data Splitting and Leakage Prevention Protocol

The research question is whether a model trained on MRI scans from one group of patients can classify scans from **previously unseen patients**. All scans and slices from the same patient must therefore remain in the same partition. A patient's images cannot appear on both sides of a training/evaluation boundary.

**Five-fold cross-validation rotates the held-out group of development patients across five runs.** It measures variation across patient groups while preserving patient separation. A separate group of patients is reserved for the final evaluation and does not participate in model development.

The following proportions are project choices, not course requirements. The patient counts below come from the user's reported successful server run on 680 patients.

| Partition | Proportion of all patients | Reported patients | Permitted use |
|---|---:|---:|---|
| `development` | 70% | 476 | Five-fold cross-validation, method and hyperparameter selection, and final model training |
| `calibration` | 10% | 68 | Confidence calibration and decision thresholds after the image classifier is fixed |
| `test` | 20% | 136 | Final evaluation after all decisions are fixed |

The calibration partition assesses and adjusts confidence under a predefined procedure. It is not used to train the image classifier or select its architecture. The test partition provides the final evaluation: its results must not be used to revise the method and then present a new result on the same patients as an independent evaluation.

## Reported Dataset and Corrected Split

The following information was supplied by the user after running checks on the course server. The user subsequently ran this repository's `prepare` and `verify` commands and supplied a `PASS` result. The real server dataset has not been accessed directly from the local development environment.

- 680 patients, 1,526 image IDs, and 30,520 grayscale JPEG images.
- Each image ID identifies one scan with 20 slices; every image is 256 × 240 pixels.
- The original training and test directories share 216 patients, although their image IDs do not overlap.
- The image filename prefix is an image ID. The patient ID is extracted from the metadata's `raw` path and has the general form `NNN_S_NNNN`.

The user reported the following new partitions:

| Partition | Patients | Scans | Images |
|---|---:|---:|---:|
| Development | 476 | 1,050 | 21,000 |
| Calibration | 68 | 170 | 3,400 |
| Final test | 136 | 306 | 6,120 |
| Total | 680 | 1,526 | 30,520 |

The server-side checks passed for patient, scan, path, and exact-duplicate separation, together with fold coverage. The complete server-generated manifests and `report.json` were subsequently copied into the ignored local `outputs/adni_splits_v1/` folder. Local review verified their sealed checksums, reconstructed assignments, recorded identity/hash boundaries and correspondence with the first five-fold baseline predictions. It did not reread the original MRI pixels or source metadata, which remain on the server. These private local artifacts are excluded from version control.

The original `AD_NC/train` and `AD_NC/test` directories are therefore treated **only as source image locations**. Their contents are indexed together, and new manifests assign roles by patient. The original training/test designation is not inherited. Source images are not moved, renamed, or modified.

The task is AD/NC classification of scans from unseen patients. It is not a prediction of whether a patient will develop the disease in the future, and it does not establish generalization to other hospitals.

## Patient and Label Rules

1. Each patient belongs to exactly one of `development`, `calibration`, or `test`.
2. All scans from a patient, and all 20 slices from each scan, follow that patient's assignment.
3. Development patients are assigned to five outer folds. Each patient appears in outer validation exactly once and cannot appear in both training and validation within the same fold.
4. Patient stratification uses `AD_only`, `NC_only`, and `mixed` histories. `mixed` means the patient's scans have different diagnoses; a single disease label is not imposed on that patient.
5. Each scan retains its own label. The program uses `AD=1` and `NC=0` while checking the source metadata's `AD=2` and `NC=0` encoding. These two encodings must not be confused.
6. Patient class distributions are balanced where possible, but patient separation takes priority. Rare strata cannot be guaranteed to appear in every fold. Report actual patient, scan, slice, and class counts; image counts are not a substitute for patient counts.
7. The random seed is fixed at `3710`. Do not try multiple seeds and select a split based on model performance, or resample the final test set to reach a target score.

The script implements a custom stratified assignment at the patient level. Holdout allocations are proportional to patient counts within strata, with rounding handled explicitly. Development fold assignment prioritizes balancing each stratum's patient counts and then total patient counts. It follows grouped cross-validation principles but does not call scikit-learn's `StratifiedGroupKFold`, and it does not weight stratification by slice count.

In grouped cross-validation, the groups represented in validation must be absent from the corresponding training data. Here, each group is a patient. See the [scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

## Five-Fold Cross-Validation Procedure

Cross-validation operates only within `development`. Neither calibration nor final test patients participate in any fold.

For each run, one development fold becomes outer validation (`val`). Approximately 10% of patients in the remaining four folds are assigned to `early_stop`; the rest are assigned to `train`:

```text
All patients
├── development: 70%
│   ├── Approximately 1/5 per run → val: evaluate this fold's model
│   └── Remaining approximately 4/5
│       ├── Approximately 90% → train: update model parameters
│       └── Approximately 10% → early_stop: select the stopping epoch
├── calibration: 10%, used only after the final image classifier is fixed
└── test: 20%, used only after all decisions are fixed
```

The 10% allocated to `early_stop` refers to **10% of patients in the remaining four folds**, not 10% of all 680 patients.

| Data | Update classifier parameters | Fit data statistics | Select stopping epoch | Evaluate outer fold |
|---|---|---|---|---|
| Current fold's `train` | Yes | Yes | Not as the held-out stopping criterion | No |
| Current fold's `early_stop` | No | No | Yes | No |
| Current fold's `val` | No | No | No | Yes |
| `calibration` and `test` | No | No | No | No |

Each fold must initialize its model, optimizer, learning-rate scheduler, early-stopping history, and other training state independently. If permitted external pretrained weights are used, every fold may start from the same external weights. Training must not continue from the previous fold's fitted model.

Outer `val` is not used for early stopping or checkpoint selection within a fold. Five-fold results may be used to compare model configurations specified in advance. Because they inform method selection, they must be described as **development cross-validation results**, not as a fully independent final test. The locked `test` partition supplies the independent evaluation.

The simple baseline, ConvNeXt, and ablation experiments use the same manifests and evaluation procedure. Report every fold's result, together with the mean and standard deviation, rather than selecting only the best fold.

## Additional Training Rules

- Fit all data-dependent means, standard deviations, feature transformations, class weights, and sampling rules using **only the current fold's `train` data**. Apply the fitted preprocessing unchanged to `early_stop` and `val`. Do not compute full-dataset statistics before splitting.
- Fixed constants from external pretraining and deterministic transformations applied independently to each image must have their sources and procedures documented. Distinguish them from statistics estimated across the dataset.
- Apply augmentation only after patient assignment and only during training. Do not generate augmented images first and then distribute an original image and its augmented versions across different roles.
- Class balancing and repeated sampling affect training data only. Do not oversample validation, calibration, or test data and report the resulting scores as performance on the natural distribution.
- Use evaluation mode for validation, calibration, and testing. Do not update BatchNorm or other training state. Determine preprocessing, slice aggregation, and input-size rules during development.
- Filenames, patient IDs, `AD`/`NC` directory names, and JSON labels are used for indexing and supervision only; they are not model input features.
- Identify caches, feature files, and checkpoints by fold and manifest version to prevent reuse of another fold's data or an earlier experiment's state.

Preprocessing steps that require fitting must learn only from training data; validation and test data receive the already fitted transformations. See the [scikit-learn guide to common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

## Final Training, Calibration, and Testing

1. **Complete five-fold development.** Choose the model architecture, hyperparameters, input processing, training-duration rule, and slice aggregation method. The final epoch count may be the median early-stopping epoch of the selected configuration's five folds. Record this rule in advance; do not adjust it based on calibration or test performance.
2. **Train a new final model.** Start from a fresh initialization or the agreed external pretrained weights, and train on all of `development` for the fixed number of epochs. Estimate final data statistics from `development` only. Do not use `calibration` for early stopping.
3. **Freeze the image classifier.** Save its weights and configuration. Then use `calibration`, under the predetermined procedure, to fit temperature scaling and set the classification and manual-review thresholds. Specify the calibration objective, parameterization, threshold targets, and fallback if a target cannot be reached before examining calibration results.
4. **Freeze the complete decision procedure.** Fix the model, preprocessing, calibration parameters, slice aggregation, and every threshold. Save version information.
5. **Evaluate the final test set.** Evaluate all prespecified final comparisons together. Report complete results, resource costs, and failure cases. Do not use final test results to select a model or adjust thresholds.

The following practices are prohibited under this protocol:

- Changing the network, training duration, augmentation, or input processing in response to calibration performance. This would make calibration an additional development set rather than an independent calibration partition.
- Adding calibration patients to classifier training after calibration and continuing to use the old calibrator and thresholds. Changing the classifier changes the procedure that was calibrated.
- Using the final test set to fit temperature scaling, choose classification thresholds, select the apparently safest manual-review rate, or choose the best training run.
- Revising the model after viewing test results and describing another score on the same test set as a first independent evaluation. Any material revision requires disclosure that the test results were already inspected and reconsideration of whether independent evaluation data remain available.

Performance on calibration data is information used to fit the calibration procedure, not final evidence of accuracy. A calibration set of 68 patients does not establish threshold stability; report uncertainty. Threshold optimization and classifier fitting should use separate data. See the [scikit-learn guide to decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).

This protocol uses **one final model** by default. Calibration from a single fold must not be applied directly to averaged predictions from five models. If an ensemble is introduced later, choose its construction during development, complete the ensemble, and then calibrate the entire ensemble using the independent calibration partition.

## Out-of-Fold Predictions and Statistical Reporting

An out-of-fold (OOF) prediction for a development patient is produced by the model corresponding to that patient's outer validation fold. That model used neither training nor early-stopping data from the patient. Retain only this outer-fold prediction for each development patient under each configuration.

**Do not predict the entire development set with all five fold models, average their outputs, and call the result OOF.** Some of those models have already used the patient's data, so the average does not represent evaluation on an unseen patient. This protocol uses an independent calibration partition; OOF predictions are not used for final calibration.

The 20 slices from one scan are not 20 independent patients. Specify the method for aggregating slice predictions into a scan prediction during development. Report scan-level results as the primary results, with optional slice-level results. If confidence intervals are computed, resample by patient and retain all scans associated with each sampled patient. Do not treat all 30,520 slices as independent observations.

A patient may have different diagnoses at different visits. Do not average all follow-up predictions and assign an arbitrary patient label. Any additional patient-level metric requires an interpretable selection rule specified in advance.

## Manifest Generation and Independent Verification

The audit tool is `adni_splits.py`, which requires Python 3.9 or newer and Pillow. It implements **data indexing, splitting, and integrity auditing**. Baseline development-fold training and checkpoint inference are implemented in `train.py` and `predict.py`; see the README for their dependencies and commands. The baseline uses fixed intensity scaling, training-only loss weights, an independent early-stop subset, and outer validation after checkpoint selection. Final refitting, confidence calibration, and final-test evaluation are not yet implemented. The full protocol remains a requirement for those future stages.

The tool indexes only files with `.jpg` or `.jpeg` extensions in the four source directories and verifies that their actual format is JPEG. Other files are excluded from the experiment manifests and listed under `source_audit.ignored_non_jpeg_files` in `report.json`. If an excluded file is an image that should be part of the experiment, investigate it first; exclusion does not mean the image passed the audit.

After copying the script to the server, run it from the directory containing the script. Store outputs in your own directory and keep source data read-only:

```bash
python3 adni_splits.py prepare \
  --data-root /home/groups/comp3710/ADNI \
  --output "$HOME/comp3710/adni_splits_v1" \
  --folds 5 \
  --seed 3710
```

This uses the default 70% development, 10% calibration, and 20% test allocation, with a 10% early-stopping holdout within each fold's training-side patients. After successful preparation, run the independent verification command:

```bash
python3 adni_splits.py verify \
  --data-root /home/groups/comp3710/ADNI \
  --output "$HOME/comp3710/adni_splits_v1"
```

Generated artifacts:

| File | Purpose |
|---|---|
| `all.csv` | Index of all included images |
| `patients.csv` | Patient partitions and strata |
| `development.csv` | Development data available for final model training |
| `calibration.csv` | Independent calibration data for the final model |
| `test.csv` | Locked final test data |
| `fold_01/train.csv` through `fold_05/train.csv` | Data used to update model parameters in each fold |
| `fold_01/early_stop.csv` through `fold_05/early_stop.csv` | Data used to select the stopping epoch in each fold |
| `fold_01/val.csv` through `fold_05/val.csv` | Outer validation data in each fold |
| `report.json` | Counts, overlap checks, duplicate checks, and integrity results |
| `COMPLETED.json` | Successful-generation marker and integrity information |

`verify` checks manifests and SHA digests against the actual source images and metadata, detecting changes to the data or manifests. The presence of CSV files or a completion marker does not replace this independent verification. Verify and record the result before formal experiments and after any change to the data or manifests.

## Audit Failures and Verification Limits

The following conditions must stop the creation of a usable split until their causes have been investigated and resolved:

- Missing metadata, unparseable patient/image IDs, or disagreement between directory labels and metadata labels.
- An image ID associated with multiple patients, a repeated image/slice identity, or an unexpected slice count for a scan.
- Corrupt or incompletely readable images, or manifests with missing files, duplicate rows, omissions, or extra samples.
- Patient, scan, or identical-content overlap between `train`, `early_stop`, and `val` within a fold, or between those roles and calibration/final test data.
- Exactly identical decoded images assigned to different patients. These require review rather than an automatic assumption that they are harmless. Blank images can also produce identical content and should be handled under predefined quality rules.
- Identical image content associated with different labels, even for the same patient.
- A verification mismatch involving the included source JPEGs, metadata, or recorded manifest digests.

Identical images with the same patient and label may be retained and reported. Patient grouping prevents them from crossing role boundaries, but duplicates can still affect sample weights. Any deduplication decision must follow a uniform rule established and documented before model development.

File and decoded-pixel digests can identify exact duplicates even after renaming, but **cannot establish the absence of approximate duplicates**. The tool also cannot independently establish that patient IDs are correct or that upstream course preprocessing did not fit statistics using all patients. Confirm upstream processing with the data provider and describe any unverified aspects as limitations in the report.

The purpose of this protocol is to establish leakage-prevention boundaries that can be inspected and reproduced. A split should only be described as having passed the implemented data checks after the real server run and review of its results. The user has reported a successful audit for the current split, as documented above. Baseline execution has been checked on synthetic CPU fixtures, and the first five real-data runs have undergone local log, checkpoint, prediction and manifest review. Original-image and metadata verification remains server-side; the future final-calibration/test workflow has not yet been implemented or verified.
