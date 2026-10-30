# Frozen 2D test evaluation: job 716656

Submitted on 2026-10-30 (Australia/Brisbane), once. No new training.
Completed: final log timestamp 04:00:05 AEST, both completion markers present,
five paired figures saved, and empty error output. Native test Dice 0.825359
meets the 0.75 target; shared 2D/3D Dice 0.823198 / 0.773216. All 27 GT grids
matched exactly and reference counts reproduced the archive. Actual metrics
and reviewed figures are archived at `support/saved_results/test_results/`.

Frozen epoch-5 weights:
`results/frozen_epoch5_716367/best_model.pt`, read-only exact copy; SHA256
`3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b`.

Remote directory:
`~/PatternAnalysis-2026/recognition/UNet2D_s4984160`.
Source snapshot: `evaluation_2d_final_v1/`.
Script: `evaluate_final_v1.sbatch` (local source `evaluate_final.sbatch`).
Seven synthetic checks passed remotely, including the two original Z sizes,
flip-before-resize mapping, isolated 2D/3D imports, freeze/no-overwrite and HD95.
A synthetic paired figure was rendered and visually inspected before submission;
this is not a patient result.

The job reads the frozen 2D model once across 1620 test slices, verifies exact
shared GT alignment, reinfers the frozen 3D model to fill shared HD95/confusion
counts, and verifies all reference overlap counts against the archived 3D run.
Native 2D and shared-grid scores remain separate. It generates five fixed-case
paired figures from saved arrays. Account/partition/GPU/1.5h resources unchanged;
no extra CPU/memory request. Runtime one thread, zero extra loading workers.

User query commands:

```bash
cd ~/PatternAnalysis-2026/recognition/UNet2D_s4984160
squeue -j 716656
tail -n 25 unet2d_eval_716656.out
cat unet2d_eval_716656.err
cat results/final_test_716656/summary.json
```

Outputs under `results/final_test_716656/` include `paired_cases.csv`, separate
native/shared per-case metrics, a frozen checkpoint copy, and `figures/`.
`evaluation_complete.txt` and `figures/plots_complete.txt` mark stage completion.
While queued or incomplete, summary/figures may not exist. See
`EVALUATION_USAGE.md` for exact fields and limitations. Stop after submission;
do not automatically monitor, tune on test outcomes or submit another job.
