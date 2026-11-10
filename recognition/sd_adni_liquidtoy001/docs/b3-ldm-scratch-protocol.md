# B3 protocol: a latent diffusion model trained from scratch (DDPM objective, DDIM sampling)

Approved by the student 2026-10-15 (Australia/Brisbane), before any B3 job ran.
This file is not edited afterwards; results go in `experiments.md`.

## Why

E2 starts from Stable Diffusion's UNet, pretrained on billions of natural
images. Whether that pretraining helps on 21,240 grayscale brain slices, or
merely carries a natural-image prior that had to be trained out (E0's colours
and grey background), is untested. The tutor suggested comparing against other
diffusion models. B3 answers the pretraining question with one change: the same
latent diffusion pipeline, with a smaller UNet trained from random
initialisation.

It also bears on the memorisation corner. Diffusion models trained from scratch
on small datasets are known to replicate training images (Somepalli et al.,
CVPR 2023; Carlini et al., USENIX Security 2023). B3 tests whether that happens
here.

## What is the same as E2

Frozen SD1.5 VAE (so the same 4x32x32 latent space at 256 px), frozen CLIP text
encoder and the same AD/NC prompts, the noise-prediction objective, 10%
empty-prompt dropout, offset noise 0.1, EMA 0.999, seed 3710, the defective-scan
exclusion (2 scans, 40 slices), the 1,024 validation slices, and the evaluation
protocol (DDIM 30 steps, guidance 7.5, the 64 fixed seeds).

## What differs, and why

| Setting | E2 | B3 | Reason |
| --- | --- | --- | --- |
| UNet initialisation | pretrained SD1.5 | random | the question being tested |
| UNet size | 860 M parameters | 92.2 M: same block layout as SD, channels (128, 256, 384, 384), 2 layers per block, cross-attention width 768 | training 860 M parameters from scratch on 21 k slices is not a reasonable design; a model a tenth the size is |
| Learning rate | 1e-5 | 1e-4, 1,000 warm-up steps | standard for training from scratch; 1e-5 is a fine-tuning rate |
| Batch and length | 16, 10 epochs (13,270 steps) | 64, 121 epochs (about 40,000 steps) | a model trained from scratch needs far more updates; the smaller UNet makes each one cheaper |

The comparison is between two reasonable designs, not a single-variable
ablation; the table lists every difference.

## Budget and the step-count rule

`comp3710`, 3 h request, 160 min budget. The S1 probe measures throughput.
**Rule fixed now:** if the measured rate cannot complete 121 epochs within 150
minutes, the main run uses the largest whole number of epochs that fits, decided
from the probe before submission and recorded. EMA weights are saved at the
halfway epoch and at the end (about 0.2 GB each).

## Stages

1. **S1 probe**, `a100-test`: 300 steps from scratch, throughput, save and reload.
2. **S2 main run**, `comp3710`.
3. **S3 evaluation**, identical to E2, at the final weights (primary) and the
   halfway weights (secondary). Guidance 7.5 is primary, for comparability.
   Guidance 3.0 is reported as secondary, because models trained from scratch
   often prefer lower guidance; neither is chosen on test data.

About 3 A100-hours including evaluation. One rerun at most, for infrastructure
failure only. If the queue leaves no time for B3 before 21 October, B3 is dropped
and reported as not run; B2 has priority.

## Hypotheses, reported without pass thresholds

- **H1 (pretraining helps fidelity).** B3's feature-nearest SSIM is below E2's
  (0.705), and its samples show more anatomical errors on visual review.
- **H2 (memorisation).** B3 shows a stronger memorisation signal than E2: lower
  nearest-training distances, near-copies among the closest pairs, or a higher
  membership-test score in the final protocol.
- If either hypothesis fails (for example, B3 matches E2 at a ninth of the size),
  that is reported as the finding.

## What this does not show

- One small architecture and one seed; a larger from-scratch model or longer
  training could do better.
- The frozen VAE and text encoder are still pretrained; B3 isolates the
  denoising UNet's pretraining only.
