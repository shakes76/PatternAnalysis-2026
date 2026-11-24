# Frozen 2D evaluation and paired 3D comparison

Completed job 716656: native test Dice **0.825359**, shared 2D/3D Dice
**0.823198 / 0.773216**. Results and paired figures are in
[the archived result record](../saved_results/test_results/README.md).

Selected checkpoint: formal run 716367, epoch 5, mean validation prostate
Dice 0.8341295702 across 39 scans. It is not a test result. SHA256:

`3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b`.

Preserve a read-only copy at `results/frozen_epoch5_716367/best_model.pt`
using `evaluate.freeze_checkpoint`; the submitted evaluation script reads
that copy, verifies the hash again and saves its own read-only copy under
the new evaluation output. No optimizer updates occur.

The Slurm script `support/submit_jobs/evaluate_final.sbatch` uses the same comp3710 partition,
account, GPU and 01:30:00 limit, without extra CPU/memory requests. Synthetic
alignment/metric checks run before submission; the job evaluates once and
renders the saved arrays.
Remote frozen source folder: `evaluation_2d_final_v1/`.

## Outputs: `results/final_test_JOBID/`

- `summary.json`: native 2D / shared 2D / shared 3D metrics in separate keys,
  target check, checkpoint/source hashes, geometry checks and precision.
- `native_2d_per_case_metrics.csv`: six-class scores on all 60 original
  unpadded slices per scan, including native-band 3D surface distances using
  original MRI affine spacing. The per-slice identity affine is not used.
- `shared_2d_per_case_metrics.csv`: scores after flip-before-nearest mapping
  to the 64×128×30 shared grid. Shared GT must equal the reference exactly.
- `reference_3d_shared_metrics.csv`: new shared-region HD95/confusion metrics
  from the frozen 3D model. Its recomputed full/shared overlap counts must
  reproduce the archived original 3D result for every scan and class.
- `paired_cases.csv`: all 27 paired scan scores, 2D-minus-3D Dice differences,
  zero GT-mismatch counts, and GT/prediction contact with shared W boundaries.
- `prediction_inputs/`: private cached MRI/labels/predictions for the same
  five cases reviewed for 3D. Outside the 2D observed band is marked 255.
- `figures/`: five paired figures (MRI, GT, 3D, 2D, 2D TP/FP/FN); gray shade
  marks excluded regions. Extent rows are proxies, not clinical landmarks.
- `native_predictions/`: private per-scan native predictions/GT for traceability;
  no raw images, weights or prediction caches belong in Git.
- `evaluation_complete.txt` and `figures/plots_complete.txt`: stage completion.

The native 2D threshold check uses mean per-scan prostate Dice >=0.75.
Comparison uses the shared-grid values, not native 2D versus full-volume 3D.
Both-empty classes remain undefined; invalid HD95 cases are counted, not zeroed.
Shared cropping can introduce surface boundaries; inspect the recorded contact
flags when interpreting HD95. 2D inference is FP32 to match its validation,
and 3D remains BF16 to match the frozen reference. This is not a matched-
precision efficiency benchmark. No claim of isolated architectural causality
is made: normalization, capacity, preprocessing and training differ.

`python predict.py evaluate <evaluate.py arguments>` supports direct loading,
inference and visualization. The submitted workflow runs `evaluate.py`, then
`predict.py render --input-dir .../prediction_inputs --output-dir .../figures`,
which uses cached outputs without a second model inference.

Do not adjust training/checkpoint/thresholds against these held-out outcomes.
After submission, stop and hand the user the job ID and query commands.
