# E1 protocol: full-UNet fine-tuning

Recorded 2026-10-12 (Australia/Brisbane), before any E1 job ran. The student
approved full-UNet fine-tuning on the same day. This file fixes the hypotheses,
configuration, budget and acceptance criteria in advance; results go in
`experiments.md` and do not change this file.

## Why another run

E0 (job 629931) trained rank-4 attention LoRA for 20,000 steps at batch 1,
which is 0.94 epochs over 21,280 training slices. Its fixed-seed cohort reaches a
feature-nearest SSIM of 0.377 (fixed-reference 0.350), against a course target
of 0.6 and a real-validation-to-training value of 0.710. The distribution is
bimodal: 23 of 64 samples already exceed 0.6, while grey-background and colour
failures sit near zero. That pattern points to specific, addressable failures
rather than an unworkable approach.

## Hypotheses tied to observed failures

| Observation in E0 | Hypothesis | E1 change |
| --- | --- | --- |
| Grey background where MRI is black (F1; mean border 0.059 vs real about 0) | SD1.5's schedule keeps signal at its last timestep, so zero-mean noise training cannot learn to move global brightness (Guttenberg 2023; Lin et al., WACV 2024) | Offset noise 0.1 |
| Colour artefacts that grow with guidance (F2; CFG 5 to 7.5 raises colour in 45/64 samples) | The empty-prompt branch used by guidance was never trained on MRI, so guidance extrapolates from a natural-image prior (Ho & Salimans 2022) | Replace the prompt with the empty prompt for 10% of samples |
| Low-level texture and intensity statistics stay photographic | These live in the convolutional ResNet blocks, which attention-only LoRA cannot change | Train every UNet weight |
| Under 1 epoch at batch 1 | Under-training | Batch 16, 10 epochs (13,300 steps) |

## Configuration

| Setting | E0 | E1 |
| --- | --- | --- |
| Trainable | Attention LoRA r4, 797,184 parameters | Full UNet, about 860 M parameters |
| Frozen | VAE, text encoder, base UNet | VAE, text encoder |
| Steps and batch | 20,000 x 1 | 13,300 x 16 (10 epochs) |
| Learning rate | 1e-4 constant | 1e-5, linear warm-up over 500 steps, AdamW, weight decay 0.01 |
| Prompt dropout | none | 10% empty prompt |
| Offset noise | none | 0.1 |
| Weight averaging | none | EMA, decay 0.999 with (1+n)/(10+n) warm-up; samples use EMA |
| Precision | fp32 weights, bf16 autocast | same |
| Determinism | bitwise | seeded; TF32 and cuDNN benchmarking enabled for throughput |
| Unchanged | SD1.5 revision 451f4fe, 256 px, class prompts, patient split `manifest-seed3710.csv`, evaluation protocol | same |

Code: `modules.py`, `dataset.py`, `train.py`, `predict.py`. Slurm wrappers:
`tools/e1_probe.sbatch`, `tools/e1_train.sbatch`, `tools/e1_predict.sbatch`.

## Stages and budget

1. **S1 probe** on `a100-test` (20 min cap): 200 steps on the first 3,200
   training slices, weight save and reload, four generated images. Measures
   throughput and peak memory; fixes the S2 step count if the estimate is wrong.
2. **S2 main run** on `comp3710`, 4 h request, `--time-budget-minutes 200`.
   Validation loss on 1,024 fixed validation slices after each epoch, with live
   and EMA weights; fixed-seed sample grid per epoch; EMA weights saved at
   epochs 5 and 10 (about 1.7 GB each).
3. **S3 evaluation**: `predict.py` with seeds 37200-37231 for AD and NC, DDIM 30,
   eta 0, guidance 7.5, the deterministic settings of cohort 632557. Then the
   existing SSIM, feature, nearest-neighbour and appearance audits, extended to
   accept the E1 cohort.

Expected GPU use is about 3 A100-hours; the ceiling is 5. One rerun at most, and
only for an infrastructure failure (node fault, time limit), not to tune.

## Acceptance criteria, validation data only

| Criterion | E0 value | E1 passes if |
| --- | --- | --- |
| Feature-nearest SSIM, mean over 64 | 0.377 | >= 0.6 |
| Fixed-reference SSIM, mean over 64 | 0.350 | reported; no threshold |
| Mean outer-border intensity (grey background) | 0.059 | <= 0.01 |
| Mean colour range before grayscale conversion | 0.021 | below 0.021 |
| Generated-to-training nearest-neighbour RMSE, mean | 0.1122 | >= 0.0731 (real validation to training) and no near-copy among the 5 closest pairs on visual review |

If fidelity passes and the memorisation audit fails, that is the central
finding of section 2.4 and is reported as such. The test split stays sealed
until the final protocol is frozen; this run does not read it.

## Stop conditions

- Non-finite loss or gradient: the run raises and stops.
- EMA validation loss rising for two consecutive epochs: evaluate the epoch-5
  weights as well as the final ones and report both.
- Time budget: `train.py` stops after the epoch that would overrun 200 minutes
  and saves the EMA weights.

## Known limitations

The four changes are bundled, so their individual contributions cannot be
separated. If time allows, one ablation reruns the E1 recipe with
`--trainable lora --lora-rank 4` to isolate capacity. Fine-tuning every weight
also raises memorisation risk, which the nearest-neighbour audit tests rather
than assumes.

## References

- Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, CVPR 2022.
- Ho & Salimans, Classifier-Free Diffusion Guidance, NeurIPS Workshop 2021 / arXiv 2207.12598.
- Lin et al., Common Diffusion Noise Schedules and Sample Steps are Flawed, WACV 2024.
- Guttenberg, Diffusion with Offset Noise, Crosslabs blog, 2023.
- Song et al., Denoising Diffusion Implicit Models, ICLR 2021.
