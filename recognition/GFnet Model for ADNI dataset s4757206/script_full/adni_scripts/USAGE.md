# ADNI: standalone scripts migrated from the four final notebooks

The four Python files work together without importing or executing notebooks.
Keep them in one directory. Google Drive mounting is a Colab setup action, not
part of the experiment code. Do not commit MRI data, ZIP archives, split/image
inventories or model weights to the course repository; keep generated artifacts
outside the source folder and retain only selected report figures as permitted.

## Contents and retained behaviour

- `dataset.py`: original inventory order (train/test, AD/NC, sorted filenames),
  JSON image-to-patient mapping, original patient-overlap audit, known duplicate
  removal, stratified 70/15/15 patient splits, checks, saved CSVs and figures.
  Includes ZIP extraction, both final augmentation pipelines and DataLoaders.
- `modules.py`: CNN and GFNet layers, preserving checkpoint parameter names.
  CNN has 92,930 parameters; final GFNet has 1,418,882.
- `train.py`: seed 42, batch 32, workers 8; CNN uses Adam and 10 epochs;
  GFNet uses AdamW, cosine annealing and 30 epochs. Saves the lowest-validation-
  loss checkpoint, configuration, history and curves. By default calls
  `predict.py` after training. `--skip-test` disables that final evaluation.
- `predict.py`: checkpoint loading, test metrics, prediction CSV, confusion
  matrix, ROC, confidence summaries, validation-only referral-threshold selection,
  retained test errors, four contrasting failure examples. `--benchmark` adds
  the CUDA forward-only profiling protocol. `--skip-analysis` requests only
  the core test metrics and figures.

Obsolete commented experiments and repeated notebook debug cells are not active
script functionality. Results are slice-level. Patient independence is checked;
patient-level aggregation, formal calibration and clinical validation are not
implemented or claimed.

## Environment

Original verified notebook/checkpoint environment: Python 3.13.16,
PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.1.3,
pandas 2.2.3, scikit-learn 1.6.1, Pillow 11.3.0, Matplotlib 3.10.0,
CUDA build 13.0, NVIDIA A100-SXM4-40GB.

Use the matching PyTorch/torchvision CUDA build for your environment. The scripts
record the actual environment; no automatic package installation is performed.
Full retraining may differ numerically from notebook runs despite the same seed.

## Colab setup

Upload the four scripts into a new `project/scripts_full` directory, retaining
exact filenames. In a Python cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

In another cell:

```python
%cd /content/drive/MyDrive/COMP3710/project/scripts_full
```

The following `%%bash` blocks each go in their own Colab cell. In a terminal,
remove the `%%bash` first line. Replace the project root as needed.

## 1. Prepare from the original ZIPs and metadata

Expected archive layout beneath `AD_NC`: `train/AD.zip`, `train/NC.zip`,
`test/AD.zip`, `test/NC.zip`. As in the notebooks, each ZIP contains flat image
filenames. Metadata is `ADNI/meta_data_with_label.json` with image IDs as keys and
patient identifiers in each entry's `raw` path. No old inventory CSV is needed.

```bash
%%bash
set -e
PROJECT=/content/drive/MyDrive/COMP3710/project
python dataset.py \
  --archive-root "$PROJECT/dataset/raw/ADNI/AD_NC" \
  --image-root /content/mri_reproduction_cache \
  --metadata "$PROJECT/dataset/raw/ADNI/meta_data_with_label.json" \
  --output-dir "$PROJECT/reproduction_run_01/splits" \
  --reference-split-dir "$PROJECT/dataset/splits"
```

`--reference-split-dir` verifies every image's assigned split against your old
CSVs, ignoring machine-specific path roots. For a first-time user with no old
splits, omit this optional argument. A reference mismatch aborts before saving
new CSVs. Original notebook image counts are 21,620 / 4,460 / 4,440, with
476 / 102 / 102 patients. Counts alone do not prove identical membership.

The cache and output directories must be new or empty. If images are already
extracted, omit `--archive-root` and point `--image-root` at that existing image
root. The program will scan it without extracting again.

Generated CSV paths point to the image root used for preparation. Therefore the
following commands use the cache as both `--source-root` and `--image-root`.
If a later Colab session loses the cache, extract the four ZIPs again using
`extract_image_archives` from `dataset.py`, or rerun preparation into a new output
directory. If images move to a different root, keep `--source-root` equal to the
old root recorded in the CSV and set `--image-root` to their new location.

## 2. Train each model from scratch

Each command saves its own best weights and, after training, runs test evaluation
and the additional confidence/referral/failure analyses in `evaluation/`.

```bash
%%bash
set -e
PROJECT=/content/drive/MyDrive/COMP3710/project
python train.py \
  --model cnn \
  --split-dir "$PROJECT/reproduction_run_01/splits" \
  --source-root /content/mri_reproduction_cache \
  --image-root /content/mri_reproduction_cache \
  --output-dir "$PROJECT/reproduction_run_01/cnn" \
  --device cuda
```

Repeat with `--model gfnet` and output directory
`$PROJECT/reproduction_run_01/gfnet`. Default budgets are 10 and 30 epochs,
respectively. For a pipeline smoke test use `--epochs 1` and a separate directory;
that run is not a reproduction of the final training budget. `--num-workers`
can be lowered on smaller Colab runtimes. Keep your final report weights intact.

## 3. Evaluate a checkpoint independently, including GPU profiling

```bash
%%bash
set -e
PROJECT=/content/drive/MyDrive/COMP3710/project
python predict.py \
  --model gfnet \
  --checkpoint "$PROJECT/reproduction_run_01/gfnet/gfnet_best.pth" \
  --split-dir "$PROJECT/reproduction_run_01/splits" \
  --source-root /content/mri_reproduction_cache \
  --image-root /content/mri_reproduction_cache \
  --output-dir "$PROJECT/reproduction_run_01/gfnet_profile" \
  --device cuda --benchmark
```

For CNN change the model, checkpoint to `cnn/cnn_best.pth`, and output folder.
You may instead supply your old compatible report checkpoint to check its
metrics on the reconstructed splits without retraining. All output directories
must be fresh; evaluation never overwrites old metrics.

## Outputs and interpretation

Preparation writes `image_records.csv`, `train.csv`, `val.csv`, `test.csv`,
`preparation_summary.json`, `split_summary.csv`, `class_distribution.png`,
`adni_examples.png` and representative-image properties.

Training writes `<model>_best.pth`, `config.json`, `history.json`,
`best_metrics.json`, `training_curves.png`, plus `evaluation/` by default.
Weights alone do not support exact optimizer/scheduler resumption.

Evaluation writes `metrics.json`, `classification_report.csv`,
`test_predictions.csv`, `confusion_matrix.png`, `roc_curve.png`, actual environment
and argument JSONs. Extended analysis adds `validation_predictions.csv`,
`confidence_summary.csv`, `high_confidence_diagnostic.json`,
`validation_thresholds.csv`, `referral_results.json`, `failure_cases.csv` and
`failure_cases.png` (when eligible errors exist).

The 0.90 confidence diagnostic is descriptive. Referral selection uses the fixed
0.50–0.99 grid, validation accuracy >=90% and coverage >=10%. If no threshold
qualifies, the script records that outcome without evaluating a test referral
rule or relaxing the criteria. No test metric participates in selecting it.

CUDA profiling writes `benchmark.csv` and `benchmark_timings.csv`: batch one,
30 warmups, 200 timed forwards. Absolute memory allocation in a standalone
inference process differs from a notebook retaining optimizer states. These
measurements are forward-only allocated memory, not training peak VRAM or total
GPU memory. Do not replace report numbers silently with different-protocol runs.

## Verification status of this migration

Checked locally: all four files compile; model forward operations match the
notebook source; simulated raw images/metadata and four ZIPs pass the preparation
CLI, reference split comparison and figure generation; reference mismatches and
missing patient IDs are rejected; confidence and both referral-selection outcomes
are checked on synthetic predictions. No PyTorch runtime or ADNI data is available
in the migration environment. Full PyTorch import, training, checkpoint loading,
failure-image rendering and CUDA benchmarking must be verified in Colab before
claiming end-to-end reproduction of this revised package.
