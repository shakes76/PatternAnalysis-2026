# Running the code

[&larr; Back to the main README](../README.md)

---

## Repository contents

```
PatternAnalysis-2026/
└── recognition/
    └── GFNet_s4967283/
        ├── README.md                 Main report page, links to the rest
        ├── progress_log.md           Dated record of each working day
        │
        ├── docs/
        │   ├── feasibility-review.md     Midpoint check-off review
        │   ├── running-the-code.md       This page
        │   ├── data-and-splits.md        Dataset, splits, preprocessing
        │   ├── dilemma-investigation.md  Calibration and the trade-off study
        │   ├── error-autopsy.md          Four named failure patients
        │   ├── scan-id-finding.md        The dataset ID problem
        │   └── experiment-ledger.md      All 17 experiments
        │
        ├── modules.py                GFNet and BaselineCNN, plus six self-tests
        ├── dataset.py                Patient-level splits, augmentation, loaders
        ├── train.py                  Training loop, validation, checkpoints
        ├── predict.py                Inference, confusion matrix, report figures
        │
        ├── audit_adni.py             Dataset audit, written before the loader
        ├── visualise_data.py         Raw data figures and a leakage measurement
        ├── brightness_baseline.py    One-feature classifier, audits for an artifact
        ├── evaluate_calibration.py   ECE, reliability, reject option, temperature
        ├── inspect_patient.py        Per-slice predictions for a named patient
        ├── try_ensemble.py           Combines the two models and measures the gain
        ├── audit_split_difference.py Compares the shipped train and test folders
        ├── find_duplicate_subjects.py Finds repeat scans of one person across splits
        ├── measure_leakage_impact.py Counts the leaked patients and scores without them
        │
        ├── run_smoke.sh              SLURM, two epochs, run before any full job
        ├── run_train.sh              SLURM, full run. Args: model seed epochs aug
        ├── run_figures.sh            SLURM, rebuilds every figure from checkpoints
        ├── run_recipe_figures.sh     SLURM, same but for a depth-12 patch-8 checkpoint
        ├── run_tta.sh                SLURM, measures what test-time augmentation buys
        │
        └── figures/
            ├── fig1_class_comparison.png        Example slices, AD against NC
            ├── fig2_one_patient_all_slices.png  All 20 slices of one patient
            ├── fig3_class_averages.png          Mean image per class
            ├── fig4_intensity_histograms.png    Pixel intensity by class
            ├── predictions_baseline.png         Example predictions
            ├── predictions_gfnet.png
            ├── failures_baseline.png            Most confident mistakes
            ├── failures_gfnet.png
            ├── filters_gfnet.png                Learned frequency filters
            ├── calibration_reliability.png      Reliability diagram
            ├── calibration_confidence.png       Confidence, right against wrong
            ├── reject_option_curves.png         Retained accuracy by referral rate
            ├── calibration_results.json         Every calibration number
            ├── test_predictions_*.npz           Saved test probabilities
            └── validation_predictions_*.npz     Saved validation probabilities
```

Model checkpoints and training logs are not committed. The spec forbids it and
they are regenerable from `run_train.sh`.

| File | Purpose |
|------|---------|
| `modules.py` | `BaselineCNN` and `GFNet` definitions, plus six self-tests |
| `dataset.py` | Data loader with patient-level splits and augmentation |
| `train.py` | Training loop, validation, checkpoints, metric plots |
| `predict.py` | Loads a checkpoint, runs inference, draws the report figures |
| `audit_adni.py` | Dataset inspection, written before the loader |
| `visualise_data.py` | Four figures of the raw data, plus a leakage measurement |
| `brightness_baseline.py` | A one-feature classifier, used to audit the result |
| `evaluate_calibration.py` | ECE, reliability, reject-option sweep, temperature scaling |
| `inspect_patient.py` | Per-slice predictions for one patient, used for the error autopsy |
| `try_ensemble.py` | Combines the two models and measures what that buys |
| `audit_split_difference.py` | Compares the shipped folders on eight measures, with effect sizes |
| `find_duplicate_subjects.py` | Detects repeat scans of one person, using test as a control |
| `measure_leakage_impact.py` | Counts leaked patients and recomputes accuracy without them |
| `run_smoke.sh` | SLURM smoke test, two epochs, run before any full job |
| `run_train.sh` | SLURM training run. Takes model, seed, epochs and augmentation |
| `run_figures.sh` | SLURM job that rebuilds every figure from a pair of checkpoints |

---

## Dependencies and reproducibility

I measured these on the Rangpur HPC cluster.

| Package | Version |
|---------|---------|
| Python | 3.11 |
| torch | 2.14.0+cu130 |
| torchvision | matching cu130 build |
| CUDA | 13.0 |
| numpy | used in `train.py`, `predict.py` and the analysis scripts only |
| matplotlib | for all figures |
| Pillow | image loading |

Hardware: one NVIDIA A100-PCIE-40GB.

**Note on `modules.py`.** The assignment says the model code must depend only on
PyTorch. `modules.py` imports nothing else. NumPy appears in `train.py`,
`predict.py` and the analysis scripts, where the assignment allows it.

### What is fixed, and what is not

`set_all_seeds` fixes the seed for Python `random`, for NumPy and for torch.
It fixes the CUDA seed as well. The train and validation split uses the same
seed, so the same patients go to the same split on every run.

An honest limit: `cudnn.benchmark` stays on. It picks the fastest convolution
algorithm for the hardware, and that choice is not guaranteed to be the same
between runs. So the results repeat to within a small margin. They are not
bit-exact. `deterministic=True` would fix that, at a clear cost in speed.

---

## How to run

Every command below shows the output you should expect. The output is real and
abridged, with `...` where lines are cut. If what you see differs, something is
wrong and the note under each block says what to check.

Run the whole pipeline in this order. Each step depends on the one before it.

### 1. Audit the dataset

```bash
python audit_adni.py
```

Expected output:

```
Auditing ADNI dataset at: /home/groups/comp3710/ADNI

1. FOLDER STRUCTURE (depth limited to 3)
ADNI/
    AD_NC/
        test/
            NC/   <- 4540 image files
            AD/   <- 4460 image files
        train/
            NC/   <- 11120 image files
            AD/   <- 10400 image files

TOTAL IMAGES FOUND: 30520

3. SAMPLE IMAGE PROPERTIES
  size (width, height): (256, 240)
  mode: L   (L means greyscale, RGB means colour)
  pixel min: 0
  pixel max: 255

4. PATIENT ID PATTERN (the important one)
Folder: .../AD_NC/train/AD
  files matching the <patient>_<slice> pattern: 10400 of 10400
  unique patient IDs: 520
  first few patient IDs: ['218391', '218393', '228875', '228879', '236920']
  slices per patient: min 20, max 20, average 20.0
...

5. DO ANY PATIENTS APPEAR IN MORE THAN ONE FOLDER?
No patient appears in two different folders. The shipped split is clean.
```

**Check:** every file must match the `<patient>_<slice>` pattern, and every
patient must have exactly 20 slices. The loader depends on both.

**A note on section 5.** That conclusion is true by ID and false by subject.
The section "The filename ID is a scan ID, not a person ID" below shows that
consecutive IDs are the same person. The first few training IDs in the output above hint at it already. 218391
and 218393 are two apart. 228875 and 228879 are four apart. I did not notice that until a validation accuracy
looked wrong three days later.

Run this first. Every later step assumes the layout it reports.

### 2. Produce the data figures

```bash
python visualise_data.py --data-root /home/groups/comp3710/ADNI/AD_NC \
  --output-dir figures
```

Expected output:

```
Reading train split from /home/groups/comp3710/ADNI/AD_NC
Found 21520 images from 1076 patients
  AD patients: 520
  NC patients: 556

wrote figures/fig1_class_comparison.png
wrote figures/fig2_one_patient_all_slices.png
wrote figures/fig3_class_averages.png
wrote figures/fig4_intensity_histograms.png

SLICE SIMILARITY (plain correlation between flattened images)
  same patient, adjacent slices:  mean 0.9612  (min 0.9041, max 0.9891, n=30)
  different patients, same class: mean 0.6769  (min 0.2880, max 0.8626, n=30)
  gap: 0.2843
```

**Check:** the similarity gap is the measurement that justifies patient-level
splitting. Two slices of one patient correlate at 0.96, two slices of different
patients at 0.68. A random image-level split would scatter one person across
train and validation and hand the model that 0.28 gap as a shortcut.

### 3. Check the splits and the loader

```bash
python dataset.py
```

Expected output:

```
Building splits...

SPLIT SUMMARY
train         18300 images    915 patients   NC: 9460   AD: 8840
validation     3220 images    161 patients   NC: 1660   AD: 1560
test           9000 images    450 patients   NC: 4540   AD: 4460

No patient appears in two splits (the assertion in build_splits passed).

Loading one batch to check shapes...
  image batch shape: (8, 1, 224, 224)
  label batch shape: (8,)
  pixel range after normalising: -0.518 to 3.779
```

**Check:** the patient counts must read 915, 161 and 450, and the assertion
line must appear. If `build_splits` raises an `AssertionError` instead, a
patient is in two splits and nothing after this is worth running.

### 4. Check the models before spending GPU time

```bash
python modules.py
```

Expected output:

```
GFNet and baseline self-tests

test: filter with unit weights returns its input unchanged
   largest absolute difference: 4.768e-07
   passed

test: filter shapes
   grid  7x7  -> rfft width 4, weight (7, 4, 4, 2)  OK
   grid 14x14 -> rfft width 8, weight (14, 8, 4, 2)  OK
   grid 16x16 -> rfft width 9, weight (16, 9, 4, 2)  OK
   passed

...

test: model output shapes and parameter counts
   BaselineCNN  output (4, 2)   parameters    421474
   GFNet        output (4, 2)   parameters   4788482
   GFNet is 11.4x the size of the baseline
   passed

all checks passed
```

**Check:** the first test is the important one. It sets every filter weight to
1 + 0i, so the block does FFT, multiply by one, inverse FFT, and must return
its input unchanged. A difference above 1e-4 means the transform plumbing is
wrong and no amount of training will fix it.

This runs on CPU in a few seconds. On the login node it prints a CUDA driver
warning, which is expected there and harmless.

### 5. Train

```bash
# Arguments, in order:
#   MODEL SEED EPOCHS AUG PATCH SLICES DEPTH DROPPATH WARMUP OPTIM MIXUP SMOOTH

# The baseline
sbatch run_train.sh baseline 42 80 strong

# The two GFNet builds this report compares
sbatch run_train.sh gfnet 42 80 normal 8 1 12 0.1 5 adamw
sbatch run_train.sh gfnet 42 80 strong 8 1 12 0.3 5 adamw 0.8 0.1

# Other configurations from the ledger
sbatch run_train.sh gfnet 42 80 normal 16 3
sbatch run_train.sh gfnet 42 150 normal 8 1 12 0.1 5 adamw
```

Expected output from `sbatch`:

```
Submitted batch job 692552
```

Then watch it:

```bash
squeue -u $USER
```

While the job runs, the log fills with one line per epoch:

```
epoch 148/150  train loss 0.2617 acc 0.8879  |  val loss 0.5767 acc 0.7724  |  13.6s
epoch 149/150  train loss 0.2649 acc 0.8862  |  val loss 0.5776 acc 0.7720  |  14.0s
epoch 150/150  train loss 0.2730 acc 0.8843  |  val loss 0.5779 acc 0.7714  |  13.7s
------------------------------------------------------------------------
Training finished in 35.2 minutes
Best validation accuracy 0.7907 at epoch 60

Loading best checkpoint and evaluating on the test set

TEST RESULTS
  accuracy: 0.6837
  macro F1: 0.6777
  NC: precision 0.6488  recall 0.8128  f1 0.7216
  AD: precision 0.7434  recall 0.5522  f1 0.6337

RESOURCE PROFILE
  trainable parameters: 4788482
  peak GPU memory: 1.58 GB
  total training time: 35.2 minutes
  mean seconds per epoch: 14.0
  inference: 5589 images/second (0.179 ms per image)

Wrote:
  outputs_gfnet_seed42_normal/best_gfnet.pt
  outputs_gfnet_seed42_normal/history_gfnet.json
  outputs_gfnet_seed42_normal/test_predictions_gfnet.npz
  outputs_gfnet_seed42_normal/summary_gfnet.json
```

**Check:** seconds per epoch tells you the job is healthy.

| Configuration | Seconds per epoch |
|---|---|
| Either model at patch 16, depth 8 | about 14 |
| Patch 8, depth 8 | about 42 |
| Patch 16, three slice channels | about 34 |
| Patch 8, depth 12, the builds this report uses | about 62 |

A figure far above these means the data loader is starving the GPU.

**The arguments that change the model's shape are `--patch-size` and
`--depth`.** Both must be repeated to `predict.py` when loading a checkpoint,
or the weights will not fit and the load fails. That failure is deliberate. A
silent fallback would fill the missing blocks with random weights and produce a
plausible-looking figure set.

The arguments that change only training are `--augmentation`, `--drop-path`,
`--warmup-epochs`, `--optimiser`, `--mixup` and `--label-smoothing`. None of
them need repeating at inference.

The output folder is named from the arguments. Patch 16 with one slice keeps
the plain name, and anything else gets a `_p<size>` or `_s<slices>` suffix, so
runs never overwrite each other.

### 6. Run a trained model and draw the figures

```bash
python predict.py --model gfnet \
  --checkpoint outputs_gfnet_seed42_strong_p8_recipe_mix/best_gfnet.pt \
  --patch-size 8 --depth 12 --output-dir figures
```

Expected output:

```
Using GPU: NVIDIA A100-PCIE-40GB

Loading data from /home/groups/comp3710/ADNI/AD_NC
Running on the test split: 141 batches

Loading gfnet from outputs_gfnet_seed42_strong/best_gfnet.pt
  4788482 trainable parameters

RESULTS
  slices: 9000
  accuracy: 0.7122

  confusion matrix
                 predicted NC   predicted AD
  actual NC            3814            726
  actual AD            1864           2596

  NC: precision 0.6717  recall 0.8401  (missed 726)
  AD: precision 0.7815  recall 0.5821  (missed 1864)

  patient level: 0.7444  (335 of 450 patients)

  FAILURE CASES FOR THE AUTOPSY
  Most confident false negatives (has AD, called normal):
    patient 985197  confidence 0.9990
    ...
```

**Check:** the confusion matrix rows must sum to 4,540 and 4,460. If they do
not, the loader returned the wrong split.

Add `--patch-size 8` or `--slice-context 3` when the checkpoint was trained
that way. A mismatch fails loudly when the weights do not fit the model, which
is the behaviour you want.

### 7. Validation predictions for temperature scaling

```bash
python predict.py --model gfnet \
  --checkpoint outputs_gfnet_seed42_strong/best_gfnet.pt \
  --split validation --no-figures --output-dir figures
```

Expected output ends with:

```
  patient level: 0.8199  (132 of 161 patients)

  wrote figures/validation_predictions_gfnet.npz
```

**Check:** 161 patients, not 450. The temperature must be fitted on validation.
Fitting it on test and then reporting test calibration would be tuning on the
test set.

### 8. The dilemma investigation

```bash
python evaluate_calibration.py \
  --predictions figures/test_predictions_baseline.npz figures/test_predictions_gfnet.npz \
  --val-predictions figures/validation_predictions_baseline.npz figures/validation_predictions_gfnet.npz \
  --names baseline gfnet --output-dir figures
```

Expected output, per model:

```
MODEL: gfnet
Slice level:   9000 slices, accuracy 0.7122
Patient level: 450 patients, accuracy 0.7444  (20-20 slices each)

Calibration:
  expected calibration error: 0.0934
  worst single bin gap:       0.1594
  mean confidence when right: 0.8281
  mean confidence when wrong: 0.7499
  separation:                 0.0782

Temperature scaling (fitted on validation):
  temperature: 1.11 (>1 means it was overconfident)
  ECE 0.0934 -> 0.0787
  accuracy 0.7122 -> 0.7122 (should be unchanged)

Recommended operating point:
  NOT ACHIEVABLE. No threshold reaches 0.95 accuracy on automated cases
  while referring no more than 30% of cases.
```

**Check:** accuracy before and after temperature scaling must be identical.
Temperature scaling divides the logits by one number, which cannot reorder
them. If accuracy moves, the implementation is wrong.

### Rebuild every figure at once

```bash
sbatch run_figures.sh
```

Runs steps 6, 7 and 8 for both models inside one GPU allocation. Use it after
a new pair of checkpoints, so that the figures and the tables beside them
always agree.

---

## Analysis scripts

These read the saved `.npz` files or the images directly. None needs a GPU, so
the login node is fine.

### The brightness control

```bash
python brightness_baseline.py --data-root /home/groups/comp3710/ADNI/AD_NC
```

Expected output ends with:

```
APPLIED TO THE TEST SPLIT
  slice-level accuracy:   0.5850
```

**Check:** this is a deliberately stupid classifier that uses one number per
slice. A result near 0.5 would mean brightness carries no signal. A result near
0.72 would mean the CNN is close to a brightness meter. At 0.585 it sits
between, so the network learns real structure and part of the signal is
brightness.

### Per-slice predictions for one patient

```bash
python inspect_patient.py --patient 985197 --worst 10
```

Expected output:

```
PATIENT 985197
------------------------------------------------------------------------
  baseline  true class AD   correct on 0 of 20 slices
             mean P(AD) 0.0000   min 0.0000   max 0.0000
  gfnet     true class AD   correct on 0 of 20 slices
             mean P(AD) 0.0086   min 0.0010   max 0.0290
```

**Check:** this is the single most important measurement in the error autopsy.
The baseline returns a probability below 0.00005 on all 20 slices of a patient
who has Alzheimer's.

### The ensemble

```bash
python try_ensemble.py
```

Expected output ends with:

```
THE EVEN SPLIT, WHICH NEEDS NO FITTING
  slice accuracy:   0.7494
  patient accuracy: 0.7600  (450 patients)
  patients wrong on all 20 slices: 42
```

**Check:** the script sweeps the mixing weight and then warns you not to report
the best one. That weight was chosen from the test split. The
even split needs no fitting, so that is the number the report uses.

### Compare the two shipped folders

```bash
python audit_split_difference.py
```

Expected output ends with:

```
IMAGE MEASURES, TRAIN AGAINST TEST
  measure                train mean   test mean    Cohen d   size
  --------------------------------------------------------------------
  tissue_mean            113.621      113.853      -0.014    negligible
  head_area_fraction     0.258        0.263        -0.215    small
  ...
```

**Check:** every measure should read negligible or small. Anything marked
LARGE would be a candidate explanation for the validation-to-test gap.

### Detect repeat scans of one person

```bash
python find_duplicate_subjects.py
python measure_leakage_impact.py
```

Expected output from the second:

```
ID DIFFERENCE FOR HIGH-CORRELATION PAIRS
  validation
    correlation >= 0.995 (n=26): median ID gap 2, 26 of them within 20
    correlation <  0.995 (n=135): median ID gap 47026, 31 of them within 20
```

**Check:** the gap between those two median values is the evidence that the
filename ID identifies a scan and not a person. See the section on that finding
below.

---

## Where to run what

| Needs a GPU, so use SLURM | Login node is fine |
|---|---|
| `train.py`, through `run_train.sh` | `modules.py`, `dataset.py`, `audit_adni.py` |
| `predict.py`, through `run_figures.sh` | `brightness_baseline.py`, `inspect_patient.py` |
| | `try_ensemble.py`, `evaluate_calibration.py` |
| | `audit_split_difference.py` and the two leakage scripts |

The rule I keep: a GPU, any `num_workers` above zero, or anything over a few
minutes goes to SLURM. I broke it once and ran `predict.py` on the login node
with four worker processes on a shared machine.

---
