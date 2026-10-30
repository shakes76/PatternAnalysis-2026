# Evaluation protocol for the corrected ADNI split (fixed before any test evaluation)

Data: corrected subject-disjoint split released by the teaching team (Ed, 22 Oct 2026);
478 train / 202 test subjects, 0 overlap; content hash 93db60db7559c7eaae872f62c572a661.

Models (recipe unchanged from the original official-split runs):
- ConvNeXt, context3 input, seeds 42/43/44, 3-model ensemble (Rangpur A100, jobs 686006-686008).
- MatchedCNN (1,834,122 parameters, similar to ConvNeXt's 1,900,898), context3 input,
  seeds 42/43/44, 3-model ensemble, same recipe (EAIT GPU Lab, NVIDIA L4).
- Recipe: AdamW lr 3e-4, wd 0.03, batch 8, 30 epochs, 1-epoch warm-up + cosine, EMA 0.999,
  label smoothing 0.1, training-only augmentation.

Decisions fixed now, on validation only:
- Checkpoint per run: best validation slice macro-F1 at AD threshold 0.5.
- Primary metric: slice-level test accuracy at threshold 0.5; AUROC, per-class metrics secondary.
- Reject rule: lowest confidence cut-off giving 90% accepted accuracy on validation.
- The test split is evaluated once per ensemble after all training has finished;
  nothing is changed after seeing test results.
