# Frozen evaluation plan

Recorded before any model-based test evaluation.

- Basic run: job 679897; Improved run: job 679896.
- Same supplied patient splits, per-image normalization, native grids and masked padding.
- Same six classes, width 16, Adam lr 0.001, weight decay 1e-5, batch 16, seed 2026,
  CE plus foreground soft Dice; no augmentation. Twenty epochs, safe stop after a
  complete epoch at 1,000 seconds of training if the time budget is reached.
- Model differences include parameter count, normalization, residuals, decoder and
  summed scale heads; this comparison measures the complete architectural package.
- Choose each checkpoint by maximal pooled validation label-5 Dice; first wins a tie.
- No threshold tuning: exclusive argmax. Test once after both runs succeed; report
  all six Dice/IoU values and each patient's metrics, retaining null for both-empty.
- ID 5 is a prostate candidate inferred from anatomy; missing authoritative mapping
  remains explicitly unresolved. Do not silently rename numeric metrics as confirmed
  prostate scores. The target criterion is >=0.75 after anatomical confirmation.
- Analyse five actual worst-positive validation slices with numeric label-5 FP/FN
  overlays, component counts and symmetric pooled-surface HD95 in array pixels.
  Both-empty and one-empty distances are undefined, with reasons recorded separately.
- GPU profile: peak allocated/reserved training memory, total parameters, five
  warmups plus 30 synchronized batch-one forwards, excluding disk I/O. Record training
  loop time separately from cache/preprocessing and total Slurm elapsed time.
- Test scores must not guide additional tuning. Any later experiments use validation;
  disclose that the held-out test has already been inspected when reporting them.
- No 3D spatial-continuity or clinical-safety benefit can be inferred from a 2D score.
