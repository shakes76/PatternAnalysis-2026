# R1 protocol: second training seed for E2 and E3

Approved by the student 2026-10-17 (Australia/Brisbane), before any R1 job ran.
This file is not edited afterwards; results go in `experiments.md`.

## Why

Two of the project's main conclusions rest on single training runs:

1. **Capacity adds little fidelity.** E2 (full UNet) reached feature-nearest
   SSIM 0.705 and E3 (rank-4 adapter, same recipe) 0.696, a gap of 0.009.
2. **The adapter memorises less.** E3's mean nearest-training RMSE (0.0846)
   meets the 0.0731 criterion; E2's (0.0673) does not.

Without knowing how much a run varies with its random seed, neither the
smallness of the first gap nor the size of the second can be judged. R1 repeats
E2 and E3 once each with a different training seed and otherwise identical
settings.

## What changes

- Training seed **4710** instead of 3710. It drives batch order, diffusion noise
  and timesteps, prompt dropout, and E3's adapter initialisation.
- Everything that defines the evaluation stays fixed: the patient split and
  defective-scan exclusion, the 1,024 validation slices **and their fixed noise
  and timesteps** (seed 3710), the per-epoch sample seeds, and the 64 cohort seeds
  (37200-37231 per class).

This needs a small code change: `train.py` currently uses one seed for both
training randomness and validation selection. A new `--val-seed` (default 3710)
takes over the validation role, so every earlier run is reproduced unchanged.

Commands are E2's and E3's exactly, plus `--seed 4710`. To fit the home quota
(3.2 GB free; an E2 checkpoint is 1.7 GB), the replicates save only epoch-10
weights; the evaluation uses epoch 10 only, the primary checkpoint throughout.

## Stages and budget

1. Code change and unit test; no probe, because the training path is unchanged
   apart from the seed and was probed for both models.
2. R1-E2 and R1-E3 on `comp3710`, 1.5 h request each, after B3 and B2b.
3. 64-image cohorts at epoch 10; the same features, SSIM, appearance,
   nearest-neighbour and closest-pair audits.

About 1.6 A100-hours. One rerun at most, for infrastructure failure only.

## Decision rules, fixed now (validation data)

For each model, report both seeds, their mean and their range (difference).

- **Fidelity gap.** If the difference between the E2 and E3 seed means is smaller
  than the larger of the two within-model ranges, the capacity effect on SSIM is
  reported as **within seed variation**. Otherwise it is reported as a
  difference, with its size.
- **Memorisation.** "The adapter memorises less by this measure" is reported as
  **robust** only if both E3 seeds have mean nearest-training RMSE >= 0.0731 and
  both E2 seeds are below it. Any other pattern is reported as **not robust**,
  with the four values.
- **Failure modes.** Near-empty samples are counted per class; any non-zero count
  is reported, since E2's data cleaning should hold across seeds.

Separately, and at no GPU cost, each cohort's mean SSIM gets a bootstrap 95%
interval over its 64 images (10,000 resamples, seed 3710). This measures
evaluation noise from having only 64 samples, which is different from training
seed variation; both are reported.

## What this does not show

- Two seeds per model estimate seed variation only roughly; R1 can show that a
  difference is fragile, but not establish a precise interval.
- E1, B2b and B3 are not replicated.
