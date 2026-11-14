# B2 protocol: a conditional DCGAN-style baseline

Approved by the student 2026-10-15 (Australia/Brisbane), before any B2 job ran.
This file is not edited afterwards; results go in `experiments.md`.

## Why

Section 2.4 asks for "a simpler generative baseline (e.g., standard DCGAN or
basic autoencoder)" and frames the problem as a trilemma: blurry or collapsed
samples, hallucinated anatomy, and memorised patients. The project already has
the autoencoder baseline (B1, the conditional VAE of job 632570): it scores the
highest SSIM (0.727) while being visibly blurry. A GAN is the textbook case of
the other half of the first corner, mode collapse. With B2, each corner of the
trilemma has a model family that is prone to it: VAE (blur), GAN (collapse),
fine-tuned diffusion (memorisation, observed in E1). The tutor also suggested
comparing against other simple models on the same data.

## Design: reuse of the student's demonstration-2 GAN

The architecture and training recipe come from the student's own OASIS GAN in
COMP3710 demonstration 2 (repository COMP3710Rangpurfordemo2, configuration "d"
with minibatch standard deviation). The assessment allows reusing one's own
elements from a previous demonstration; the submitted model (Stable Diffusion)
is unrelated to it. On OASIS, that work found that hinge loss with spectral
normalisation and DiffAugment collapsed, and that the configuration below did
not.

- Generator: 128-dimensional Gaussian noise plus a learned AD/NC label
  embedding, through a 4x4 map and upsample-then-convolve stages to 128x128,
  one grayscale channel, tanh output; EMA of weights (0.999) for sampling.
- Discriminator: convolutional, no BatchNorm, no spectral normalisation,
  minibatch standard deviation, and a projection term for the label
  (Miyato & Koyama, 2018).
- Loss: non-saturating logistic with an R1 penalty on real images, gamma 10
  (Mescheder et al., 2018); Adam, learning rate 2e-4 for both networks,
  betas (0, 0.99); no augmentation.
- New relative to demonstration 2: class conditioning, and ADNI data.

"DCGAN-style" means an all-convolutional generator and discriminator in the
DCGAN tradition (Radford et al., 2016) with these published stabilisers; it is
not the 2016 recipe unchanged, which is known to be unstable.

## Data and budget

E2's training data: subject-level split, the two defective NC scans excluded
(21,240 slices), at 128 px like the VAE baseline (demonstration 2 found 256 px
unstable). 30,000 generator updates at batch 64 (about 90 epochs), on
`comp3710` with a 2 h request. Compute is roughly matched to E2 (one A100 for
about one to one and a half hours), not image exposures.

**Collapse rule, fixed now (from demonstration 2).** Every 1,000 steps, the
diversity ratio is the mean distance from each of 256 EMA samples to its
nearest other sample, divided by the same for 256 random real training slices.
From step 5,000: whenever the ratio is at least 0.8, the EMA generator is saved
as the snapshot; if it is below 0.6 at two consecutive checks, training stops
and the last snapshot is the result. Whether the rule fired, and at which step,
is reported. There is no other checkpoint selection.

## Stages

1. **S1 probe**, `a100-test`: 500 steps, throughput, snapshot save and reload,
   one generated cohort of 4 images.
2. **S2 main run**, `comp3710`.
3. **S3 evaluation**, identical to the other models: the 64 fixed seeds
   (37200-37231 for AD and NC) as the generator's noise, the same SSIM,
   appearance, feature, nearest-neighbour and closest-pair audits. The final
   test protocol adds precision, recall and coverage for every model.

About 1.5 A100-hours. One rerun at most, for infrastructure failure only.

## Hypotheses, reported without pass thresholds

- **H1 (sharpness).** The GAN's edge measure is closer to real slices (0.0176)
  than the VAE's (0.0076).
- **H2 (coverage).** The GAN covers less of the real distribution than E2:
  lower diversity ratio and lower recall or coverage in the final protocol. If
  it collapses, that is the expected failure and is reported as a result.
- **H3 (fidelity).** SSIM and the appearance measures are reported beside E2
  and the VAE; no target is set, because the GAN's 128 px output and the
  course's 0.6 target are not a like-for-like comparison with E2.

## What this does not show

- One GAN configuration with one seed; GANs vary from run to run.
- 128 px against E2's 256 px; both are compared at the audit resolutions.
- Label conditioning (AD/NC) rather than text prompts.

## Erratum, 2026-10-15, before any B2 job ran

The collapse rule above misdescribes demonstration 2's diversity ratio. Its
thresholds (0.8 healthy, 0.6 collapse) were calibrated on this measure, which B2
therefore uses: every 1,000 steps, the **mean pairwise Euclidean distance** among
64 EMA-generator samples, divided by the same for 64 random real training slices,
both at 64x64 in [0, 1]. Not the nearest-other-sample distance over 256 images,
as written above. Nothing else in the protocol changes. Found while porting the
demonstration-2 code, and corrected before implementation.
