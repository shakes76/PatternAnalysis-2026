# Running and checking the models

Run commands below from the repository root. Use Python 3.12 with
`recognition/Siamese_Nandos/requirements.txt` to install the pinned experiment
dependencies in a virtual environment. The older NumPy and pandas pins do not
support Python 3.14; do not silently upgrade them when reproducing experiments.

```powershell
python -m pip install -r recognition/Siamese_Nandos/requirements.txt
python -m unittest discover -s recognition/Siamese_Nandos -v
```

The regression suite uses temporary synthetic JPEGs and CPU execution. It checks
both trainers, Siamese resume, strict checkpoint loading, evaluation CSVs and
plots, cached preprocessing equivalence, and epoch updates in persistent workers.
It does not download pretrained weights or require the ISIC dataset.

The original frozen split CSVs are included in `recognition/Siamese_Nandos/splits`.
The original duplicate-pair metadata is also included and found automatically.
Supply the JPEGs separately:

```powershell
python recognition/Siamese_Nandos/train.py --model baseline --splits recognition/Siamese_Nandos/splits --images DATA/jpeg --output runs/baseline
python recognition/Siamese_Nandos/train.py --model siamese_aux --splits recognition/Siamese_Nandos/splits --images DATA/jpeg --output runs/siamese_aux
python recognition/Siamese_Nandos/predict.py --model baseline --checkpoint runs/baseline/best.pt --splits recognition/Siamese_Nandos/splits --images DATA/jpeg --output runs/baseline/evaluation --threshold-policy val_specificity80_max_sensitivity
```

Change the prediction model and checkpoint path for the Siamese model. All three split
CSVs require `image_name`, binary `target`, `patient_id` and `lesion_id`. Every
entry point audits `train.csv`, `val.csv` and `test.csv` before model use. Only
train/validation images are loaded during training; only validation/test images
are loaded during evaluation. Test labels are checked for metadata integrity
but are never used by training or validation-threshold selection. Image IDs may include the JPEG extension. Use `--workers 0` for
single-process loading and `--no-pretrained` only when intentionally training
without ImageNet initialization. The default requirements install CUDA 13.0 PyTorch. All experiment commands default
to `--device cuda` and fail if CUDA is unavailable. CPU smoke tests use
`requirements-cpu.txt` and explicitly pass `--device cpu`; the regression suite
already supplies that flag. The local test environment is separate from the CUDA
experiment environment.

The repairs restore missing code from Git history while retaining the original
model definitions, losses, augmentation, and current defaults. Pair workers now
observe the epoch seed as intended, so multi-worker training generates fresh
pairs each epoch; runs affected by the stale-epoch bug will differ after epoch 1.

The local repair checks use Python 3.14 with PyTorch 2.11 / torchvision 0.26 and
newer compatible supporting libraries in `.venv`. The supporting-library pins remain unchanged; the default PyTorch wheel builds
now select CUDA instead of CPU. Full experiment reproduction requires the original environment,
data and checkpoints.

## Notebook references

The repair was checked against `COMP3710_Baseline_Training_Clean.ipynb`,
`COMP3710_Siamese_Training_A100.ipynb`, and
`Copy_of_COMP3710_Siamese_Auxiliary_Classifier_Head.ipynb` supplied as references.
With identical weights and CPU inputs, the auxiliary notebook and repaired model
produce exactly equal embeddings, logits, combined losses and parameter gradients.
The notebook's bilinear cached resize also matches the repaired preprocessing.
The baseline retains torchvision ResNet-18 checkpoint keys and a single linear
output. The earlier A100 notebook uses an embedding-only model with a reference
bank; it is a separate experiment, not the auxiliary classifier implemented here.

The baseline notebook's old `train.py` commands need `--model baseline` and
`--learning-rate` in place of `--lr`. Its `--smoke` option belongs to an older
trainer and is not supported here; use the synthetic regression suite above for
a short pipeline check. Both notebook threshold policies remain available;
the current default stays `val_specificity80_max_sensitivity`.


## Frozen split preparation and leakage audit

The development notebook uses **GroupShuffleSplit**, not StratifiedGroupKFold.
For seeds 0 through 99, it reserves 15% of patient groups for test and 0.15/0.85 of the
remaining groups for validation. It minimizes the sum of image-proportion errors
against 70/15/15 plus twice each split's prevalence difference from the whole
cohort, retaining the first candidate in a tie. Use the original ground-truth
CSV in its original row order and the pinned scikit-learn version:

```powershell
python recognition/Siamese_Nandos/prepare_splits.py --metadata DATA/ISIC_2020_Training_GroundTruth_v2.csv --duplicates recognition/Siamese_Nandos/ISIC_2020_Training_Duplicates.csv --output DATA/splits
```

This command refuses to overwrite frozen splits. It validates the chosen
candidate before writing files and saves `split_definition.json` with the seed,
score, counts, library version and input/output SHA-256 hashes. Existing original
manifests should be kept; generate into a separate directory for comparison.
The supplied frozen manifests are now included and their counts match the report.
Image, patient and lesion separation passes across all three splits. All 425 known
duplicate pairs have endpoints present, consistent labels and no cross-split leakage. These manifests have not
been regenerated from the original full metadata. See `splits/manifest_audit.json`
for byte-level hashes and verification status.

Both trainers and prediction require the duplicate-pair CSV, defaulting to
`ISIC_2020_Training_Duplicates.csv` in the parent of `--splits`. Supply
`--duplicates PATH` for another location. Missing audit files are errors, not a
reason to skip checks. Checks reject missing IDs, invalid labels, normalized
image-ID duplication, image/patient/lesion overlap across any pair of splits,
conflicting lesion labels, and known duplicate pairs crossing partitions or
having conflicting labels. Every duplicate-pair endpoint must appear in the
manifests. This checks the supplied duplicate list, not undiscovered visual
near-duplicates. Clean manifests retain their row order and membership.

CUDA placement does not change the existing precision policy: baseline training
uses FP32; auxiliary training prefers BF16 on supported CUDA hardware and uses
FP16 otherwise; prediction retains FP16 CUDA autocast. These settings must be
recorded when comparing saved experiment scores.
