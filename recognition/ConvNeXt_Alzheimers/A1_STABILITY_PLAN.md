# A1 stability check: seeds 42, 43 and 44

## Scope declared before new runs

A1 (`scratch_augmented`) is the provisional ConvNeXt candidate. Seed 42 has
already completed and helped motivate this follow-up. Run exactly two additional
replications, seeds 43 and 44, on the existing training/validation patients.
There is no test inference, pretrained download, ensemble, patient re-splitting,
new threshold selection or architecture change in this study.

The historical seed-42 folder, its checkpoint/predictions/history and the old V2
pointer files remain unchanged. `stability_a1.py` calls the original
`train_convnext_v2.train_run`; it copies seed 42's recorded configuration and
changes only `seed`. No V2 source file is replaced or monkey-patched.

## Fixed recipe

| Setting | Value |
|---|---|
| Architecture | Existing one-channel ConvNeXt-Tiny, trained from random initial weights |
| Training augmentation | Uniform random rotation within ±5 degrees; existing A1 implementation |
| Patient weighting | Disabled |
| Input | Existing fixed greyscale resize/padding/normalization to 224 × 224 |
| Batch / accumulation | 8 / 4 (effective batch 32 apart from the last partial group) |
| Optimizer | AdamW; peak LR 0.0003; minimum LR 0.000001; weight decay 0.05 with existing exclusions |
| Schedule | Existing three-epoch warmup and cosine schedule |
| Maximum epochs / minimum epochs / patience | 30 / 12 / 8 |
| Gradient clipping | 5.0 |
| Precision / workers | Float32; TF32 and AMP disabled; workers=0 |
| Selection within each run | Patient accuracy, then macro-F1, then lower patient log loss; earliest exact tie |
| Class threshold | 0.5 |
| New seeds | 43 and 44 only |

Changing the seed affects initialization, training order, augmentation draws and
stochastic-depth masks. All other training settings remain fixed. Different
seeds may finish after different epoch counts under the same early-stopping rule.

## Candidate policy and interpretation

Keep seed 42 as the provisional candidate. Do not automatically select the
highest of the three peaks, tune the stopping rule, extend a disappointing run,
swap patients, choose an ensemble or add more seeds to chase 80%. Retain every
result. The purpose is to assess repeatability before deciding whether to stop
model development or investigate a specific remaining problem.

Report accuracy, macro-F1, AUROC, AD/NC recall, error counts, selected/completed
epochs, and how many epochs crossed 80% validation accuracy. Summarize the three
selected-checkpoint scores with mean, sample standard deviation and range. Also
summarize seeds 43/44 separately because seed 42 already influenced recipe choice.
These statistics are descriptive, not confidence intervals. All runs reuse the
same validation patients, and validation selected every reported checkpoint.
Three runs on 102 patients are NOT 306 independent patients. Crossing 80% here
does not establish minimum 0.80 test accuracy or clinical suitability.

No old calibration temperature or referral rule is applied to A1. Calibration
for a revised candidate and the revised evaluation protocol are subsequent
stages. The Version 1 test set has already been exposed. It must not be described
as untouched again or used to pick among these seeds.

## Recovery and provenance

Each new seed has its own private folder. The same `run --seed` command starts a
new run or resumes the existing one; it never creates a duplicate attempt for
that seed. The unchanged V2 trainer saves full model/optimizer/RNG/history state
at completed-epoch boundaries. An unfinished epoch normally repeats. Its fallback
to the preceding readable checkpoint is retained. A completed run is verified
and skipped. OS locks prevent concurrent stability commands, and release on
process death. Source, configuration, environment and manifests must match before
continuation. Do not change packages, code or settings during a run.

Runtime records exclude repeated partial work, downtime and some file I/O. No
cross-machine bitwise reproducibility claim is made. The model and statistics
remain research-only, not a diagnostic service.

## Assistance

ChatGPT assisted with the additive runner, tests, this plan and instructions.
Document the actual code review and checks performed in the assessment's
Artificial Intelligence Usage Disclosure. Automated tests are not medical
validation and do not replace understanding the implementation.
