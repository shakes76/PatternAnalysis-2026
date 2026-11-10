# E3 protocol: E2's recipe with E0's small adapter (capacity ablation)

Approved by the student 2026-10-14 (Australia/Brisbane), before any E3 job ran.
This file is not edited afterwards; results go in `experiments.md`.

## Why

E1 changed four things at once relative to E0, and E1's protocol listed that
as a known limitation: the improvement from SSIM 0.377 (E0) to 0.705 (E2)
cannot be split between the **training recipe** (prompt dropout, offset noise,
EMA, ten epochs at batch 16, defective-scan exclusion) and the **capacity**
change (797,184 adapter parameters to 860 M UNet parameters). The split matters
for both sides of the section 2.4 dilemma:

- Fidelity versus cost: a rank-4 adapter is a 3 MB file and trains fewer
  parameters; the full UNet is a 1.7 GB file per checkpoint.
- Fidelity versus memorisation: more trainable parameters give a model more room
  to memorise. E1 amplified one defective scan; whether a small adapter would
  have done the same is untested.

E3 trains E0's adapter with E2's recipe, so the three runs decompose the gain:
E0 to E3 is the recipe at fixed capacity, E3 to E2 is capacity at fixed recipe.

## Hypotheses

- **H1 (recipe).** At fixed capacity, E2's recipe alone removes E0's grey
  background and colour artefacts: E3 border <= 0.01 and colour < 0.0208.
- **H2 (capacity).** Capacity accounts for a substantial part of the fidelity
  gain: E3 feature-nearest SSIM falls between E0 (0.377) and E2 (0.705), at
  least 0.05 below E2.
- **H3 (memorisation).** The small adapter shows a weaker memorisation signal than
  the full UNet, by the same measures (nearest-training RMSE and the final
  protocol's membership test), reported without a pass threshold.

If E3 reaches SSIM >= 0.6, the engineering recommendation changes: the cheap
adapter would be adequate, and that is reported as such.

## The change

`train.py --trainable lora --lora-rank 4 --lr 1e-4`: the E0 adapter (rank 4
on `to_q`, `to_k`, `to_v`, `to_out.0`, Gaussian initialisation, 797,184
parameters). Everything else is E2's command: defective-scan exclusion with the
expected count of 2 scans and 40 slices, 10 epochs, batch 16, 500 warm-up
steps, 10% empty-prompt dropout, offset noise 0.1, EMA 0.999, seed 3710, the same
validation slices and sample seeds.

The learning rate is the one deliberate second difference. LoRA starts from a
zero update and trains far fewer parameters; 1e-5, E2's rate for full
fine-tuning, would leave it nearly untrained and confound the capacity
comparison with under-training. 1e-4 is E0's own rate for this adapter. The
difference is reported with the results.

## Stages and budget

1. **S1 probe**, `a100-test`: 200 steps with the filter, adapter save and reload.
2. **S2 main run**, `comp3710`, 1.5 h request, 75 min budget; EMA adapter weights at
   epochs 5 and 10 (about 3 MB each, so no storage concern).
3. **S3 evaluation**, identical to E1 and E2: 64-image cohorts at epochs 10
   (primary) and 5 (secondary), features over all cohorts, SSIM, appearance,
   pixel nearest neighbours, closest-pair panels.

About 1.2 A100-hours. One rerun at most, for infrastructure failure only.

## Reported for E3 (validation data)

| Measure | E0 | E2 epoch 10 | Used for |
| --- | ---: | ---: | --- |
| Feature-nearest SSIM, mean | 0.377 | 0.705 | H2 |
| Border intensity, mean | 0.0592 | 0.00007 | H1 (<= 0.01) |
| Colour range, mean | 0.0208 | 0.0059 | H1 (< 0.0208) |
| Near-empty samples per class | 0 / 0 | 0 / 0 | check |
| Nearest-training RMSE, mean; closest pairs | 0.1122 | 0.0673 | H3 |
| Training time, peak memory, weight file size | — | 45 min, 23.8 GiB, 1.7 GB | cost |

## What this does not show

- Rank 4 on attention layers is one point on the capacity axis, chosen because it
  is E0's adapter; it does not establish the best adapter size.
- The comparison is single-seed, like E1 and E2.
- The test split stays sealed; E3 enters the final test protocol alongside the
  other models.
