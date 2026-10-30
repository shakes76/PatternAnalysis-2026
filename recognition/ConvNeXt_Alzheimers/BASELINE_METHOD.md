# Initial baseline experiment — predeclared method

This is the **simple CNN comparison**, not the main Hard-Difficulty ConvNeXt model.
These are proposed experiment settings implemented by this package, not additional
requirements imposed by the assessment. Do not interpret synthetic test outcomes as
ADNI classification results. Read and understand the code before submitting it.

## Architecture and inputs

The existing `dataset.py` is retained unchanged: one greyscale channel, 224x224,
aspect-preserving resize and black padding, fixed scaling to [-1,1]. There are no
fitted intensity statistics and no augmentation in this initial experiment.

`BaselineCNN` has four Conv2d/BatchNorm2d/ReLU blocks with channels 16, 32, 64, 128.
Each convolution uses a 3x3 kernel, stride 2, padding 1, and no bias. The resulting
14x14 feature map is reduced to 4x4 with fixed 5x5 average pooling, stride 3. The
classifier is Flatten -> Linear(2048,128) -> ReLU -> Dropout(0.25) -> Linear(128,2).
Outputs are raw logits, with NC=0 and AD=1. The network uses only PyTorch, is
initialised from scratch, and downloads no weights.

Fixed pooling avoids the CUDA adaptive-pooling backward operation that may reject
strict deterministic execution. The loss is the usual mean cross-entropy,
implemented as the negative mean true-class log-softmax; tests compare its value
and gradient with PyTorch cross-entropy. No alternative statistical objective is
introduced by this expression.

## Fixed initial training settings

- AdamW: learning rate 0.001, weight decay 0.0001; fixed learning rate.
- Batch size 32; zero worker subprocesses; keep the last incomplete batch.
- Seed 42. Deterministic algorithms requested; TF32 and mixed precision disabled.
  Reproducibility is scoped to the same software, device and settings, not all
  machines or library versions.
- At most 20 epochs, with early stopping after five consecutive epochs that fail
  to improve the selected validation criterion. No minimum epoch count.
- Gradient norm clipping at 5.0. No class weighting or patient-balanced sampling.
- All training images are uniformly shuffled once per epoch. Patients with more
  scans therefore contribute more slice-level loss terms; this is recorded as a
  limitation and must not be confused with patient-equal evaluation.

## Evaluation and selection

The three CSVs are read for integrity/group checks. Only training/validation images
are opened. There is no `--test` switch in this stage.

Average p(AD) across slices of each scan. Then average scan probabilities within
each patient. Each scan gets equal weight in its patient's final prediction, and
each patient gets equal weight in patient metrics. Predictions use p(AD) >= 0.5.
Probabilities are **not calibrated** and are not asserted to measure diagnostic
certainty. Calibration/referral analysis is a later stage.

Each epoch reports validation slice, scan and patient metrics (accuracy,
per-class precision/recall/F1, macro-F1, balanced accuracy, AUROC, and log loss).
Select the checkpoint with highest patient macro-F1. On an EXACT tie, prefer lower
patient log loss; on a second exact tie retain the earlier checkpoint. The same
criterion controls early stopping. Log loss uses probability clipping at 1e-7.
The selected validation results are development results, not an unbiased test
estimate. The held-out test partition remains for the later fixed-model comparison.

A constant-class sanity comparator chooses the majority diagnosis using TRAINING
patient labels and reports its validation patient score. It does not replace the
trained baseline. In the provided real split, NC-only validation accuracy is 69/102
(~67.65%), so accuracy alone can conceal failure to identify any AD patients.

The baseline is not required to reach the main ConvNeXt model's stated 0.80 target
before it is a valid simpler comparison. Weak first-run results are evidence to
analyse, not permission to reshuffle patients or inspect the final test set.

## Debugging and checkpoints

The smoke check verifies a real batch's logits, finite gradients, parameter update,
and saved-checkpoint inference. It then tries to fit up to eight training images,
one per patient and up to four per class, for at most 200 updates. For THIS
DEBUGGING CHECK ONLY, dropout and weight decay are disabled. The criterion is
training-subset accuracy >=0.95 and loss <0.10 and below its starting loss. That is
a diagnostic criterion, not an assessment performance requirement.

Actual baseline training starts from new weights and a new optimizer. It does not
reuse smoke/tiny-fit weights. The best and last completed-epoch checkpoints are
saved; the latter includes optimizer state. This package does NOT implement
automatic resume. If interrupted, preserve the folder and examine it before
restarting. Never silently append to an existing run or change the experiment.

## Resource measurements and privacy

Record train-loop wall time, validation end-to-end throughput, parameter count,
peak PyTorch allocated/reserved GPU memory, and model-only batch-size-1 latency
(10 warm-ups, 50 synchronized forward passes). Model-only latency excludes image
loading, preprocessing, CPU-to-GPU transfer and scan/patient aggregation. Allocator
peaks are not total Windows/device memory. Laptop thermals and other GPU workloads
can affect timings; keep conditions comparable for the later model comparison.

Use `C:\Users\Angus\COMP3710-private` for runs. Checkpoints, patient-linked
prediction JSON and any private error traces stay there. Only aggregate summaries,
learning curves and documentation belong in the public report. No dataset source
files or split manifests are changed by these scripts.

## Academic integrity and references

ChatGPT assisted with this draft, its synthetic tests and instructions. Review the
architecture, loss, train/validation separation and aggregation. Run the tests and
actual hardware checks yourself and document what you verified under
`Artificial Intelligence Usage Disclosure` in the final README.

Sources for APIs and definitions (not claims about latest software versions):
- Supplied COMP3710 assessment brief, Sections 2.1, 4–6.
- PyTorch neural-network API: https://docs.pytorch.org/docs/stable/nn.html
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html

The installed version in the user's report is PyTorch 2.7.1+cu126. The package's
container tests use another recorded CPU version; they do not prove Windows/CUDA
execution. The required local smoke check supplies that evidence.
