# Verified CAN3D pilot - 29 October 2026

Downloaded config, three-epoch history, full validation report, two previews
and the earlier GPU-smoke report were independently inspected. Current five
training-source hashes, exact split SHA256 and all 48 validation case names
match. The smoke's earlier sources match preserved commit 50e8ee7. All six
patient-macro Dice/IoU values below were independently recomputed from cases.

Recipe: methods-text one-downsampling CAN3D, 170,342 parameters, shape
64x128x128, one volume per update, Adam 0.001, seed 42, unit Dice+CE.
Training-only paired left/right flips have probability 0.5. Each epoch uses
136 training volumes/updates; validation uses all 48 scans from six groups.

| Epoch | Training objective | Validation prostate Dice | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 1.002114 | 0.245025 | 285.2 |
| 2 | 0.532697 | 0.492840 | 283.1 |
| 3 | 0.394109 | **0.674381** | 288.0 |

The selected best is epoch 3. These are validation results, not test results.

| Class | Patient-macro Dice | Patient-macro IoU |
|---|---:|---:|
| Background | 0.989365 | 0.979035 |
| Body | 0.943950 | 0.894193 |
| Bones | 0.667870 | 0.503715 |
| Bladder | 0.613564 | 0.462792 |
| Rectum | 0.579086 | 0.414836 |
| Prostate | 0.674381 | 0.512289 |

The prostate prediction totals 68,446 voxels against 49,917 reference voxels.
Two inspected previews use checkpoint SHA256
`cbfdf5777972d4e01531c3c73e911fd69ca00efe9cc12a8f99c8ffb216892309`:

- B006_Week0 (training): prostate Dice 0.660796, IoU 0.493424; true/predicted
  voxels 1,287/2,006, TP/FP/FN 1,088/918/199. The first displayed labelled
  slice has excess prediction; larger displayed slices have both excess and
  missed boundary regions. No definitive clinical cause is inferred.
- C032_Week0 (validation): Dice 0.559899, IoU 0.388792; true/predicted voxels
  683/1,696, TP/FP/FN 666/1,030/17. The displayed first/last labelled slices
  are over-segmented, and the central slice also shows excess prediction.
  No claim is made that all patients share this failure mode.

Independently loaded prediction NIfTIs are 128x128x64 on RAS grids; label
counts agree with the JSONs. Preview inference was 0.2103/0.2071 seconds and
peak inference allocation 196.663 MiB. Peak training allocation was
909.998 MiB in each epoch. These are descriptive observations under this
recipe, not a controlled final resource benchmark. Epoch times include
validation/loading; the separate 0.05695-second repeated-case GPU smoke step
must not be substituted for an epoch time.

A parity check found the C032 preview and fit-time validation have identical
prostate metrics, but predicted counts for codes 0..5 differ by
`[5,-8,2,0,1,0]`. The preview counts above belong to its own exported mask.
Independent prediction now applies the same seed/deterministic helper as
fit/validation. GPU confirmation of that alignment is pending; the cause of
the small differences is not proven, and no checkpoint weights were changed.

The 2D baseline's selected epoch-7 validation Dice is 0.694518. The two models
have different update budgets, augmentation and sampling: the old 2D model
used balanced slices with no flips. The current results do not prove that
CAN3D improves accuracy, nor do they establish the final test target.

The prepared can3d_resume.slurm restores epoch-3 weights and Adam state and
extends the same recipe through epoch 8 in a new output directory. It then
rechecks the best checkpoint on the complete validation split and generates
low-scoring examples from three different validation patients, plus one
train/validation preview. No test cases are opened. Its 40-minute allocation
is a limit; the expected extra five epochs are roughly 24 minutes based on
this run, excluding queue time and subsequent diagnostics. Results are pending.

Update 29 October: continuation 709852 is now verified; all-class preview/validation agreement is confirmed. See [continuation evidence](can3d_resume_709852.md) for epoch-7 selection and the fixed final-test plan. The earlier pending statements describe the pilot audit stage.
