# 2D baseline submission: 716367

## Shared terminal requirement

All subsequent cluster operations must be visible in the user's existing
tmux session `codex-714278`, not an independent SSH shell. Current shared
directory is `~/PatternAnalysis-2026/recognition/UNet2D_s4984160`.
Both sides may type; leave a prompt for the user and avoid simultaneous input.
If attaching from a terminal with TERM=dumb, use
`TERM=xterm-256color tmux attach-session -t codex-714278`.
The session was resynchronized after evaluation submission 716656; a visible
query showed it pending for Priority. No background polling was started.

Submitted to Rangpur on 2026-10-30 (Australia/Brisbane). Job **716367** completed
successfully (exit 0:0) in 00:17:31 on a100-8. It stopped at epoch 13 after
eight epochs without improvement. Best epoch 5: mean validation prostate Dice
0.8341295702 over 39 scans; precision 0.8490039442, recall 0.8047404765,
pooled predicted/GT volume ratio 0.9478642379. Subsequent frozen evaluation 716656 completed with native test Dice 0.825359;
shared-grid 2D/3D Dice were 0.823198 / 0.773216. See the completed result record.

Selected checkpoint SHA256:
`3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b`.
Model configuration: one input channel, six output classes, base width 16.
The selected checkpoint was checked to be formal (`smoke_only=false`) and
selected by `mean_scan_prostate_dice`, matching the training history.
Do not treat the synthetic test output of 0.2 as an experimental score.

Remote directory:
`~/PatternAnalysis-2026/recognition/UNet2D_s4984160`

Frozen sources: `baseline2d_v1/`; submitted script: `train_baseline_v1.sbatch`
(local equivalent: `support/submit_jobs/train_baseline.sbatch`). The original shared 3D sources
and completed results were not changed.

Resources: partition/account comp3710, GPU, 01:30:00 walltime; no explicit
CPU/memory requests. Runtime threads 1, loading workers 0. Standard 2D U-Net,
base width 16, batch size 16, Adam LR 0.001, equal CE/foreground Dice,
30 maximum epochs, minimum 10, early-stop patience 8. All 14 synthetic tests
passed on the cluster before submission. The job repeats those tests, runs
two real-data train/validation batches, then starts a fresh full training run.
Smoke scores are not performance evidence.

Expected full split: 8,700 training / 2,340 validation / 1,620 test slices,
with the same patient groups as 3D. Test data are not iterated during training.
Selection metric is mean scan prostate Dice across supplied 60-slice bands;
this is not full-volume 3D Dice or a held-out test result. The course's 2D
test target is prostate Dice >=0.75; reaching a validation threshold alone
does not confirm the held-out test requirement.

Query from a normal cluster login terminal:

```bash
cd ~/PatternAnalysis-2026/recognition/UNet2D_s4984160
squeue -j 716367
tail -n 25 unet2d_716367.out
cat unet2d_716367.err
tail -n 8 results/baseline2d_716367/history.csv
cat results/baseline2d_716367/training_summary.json
```

The history/summary may not exist while queued or while executing tests/smoke
checks. The summary is written after full training finishes. Formal weights:
`results/baseline2d_716367/best_model.pt` and `last_model.pt`; partial smoke
outputs are separated under `results/smoke_716367/`. No automatic held-out
evaluation, further job submission or monitoring was started after submission.

The first non-interactive SSH attempt passed tests but lacked `sbatch` in
PATH and submitted no job. Normal interactive login then successfully
submitted 716367 once. Do not resubmit merely because it is queued.
