# Final evaluation protocol: the sealed test split

Approved by the student 2026-10-18 (Australia/Brisbane), including the optional
blinded sort and the deletion of the E1 epoch-10 weights from Rangpur, before
any code read the test split. This file is not edited afterwards; deviations go
in `experiments.md` as dated errata.

## Purpose

Every result so far is on the validation split, which was also used to choose
checkpoints, guidance and the defective-scan rule. This protocol does three
things, once, on the 102 test subjects (4,700 slices; AD 2,440 from 33
subjects, NC 2,260 from 69):

1. answers Section 2.4's question for every model with the same measures:
   structural fidelity (SSIM > 0.6), distributional coverage, memorisation,
   anatomical plausibility, cost;
2. picks the recommended model (E2 or E3) by a rule fixed here;
3. delivers the engineering verdict: is the recommended model's synthetic
   cohort safe (non-memorised), plausible, and fit for downstream use?

Each measure is tied to one corner of the 2.4 trilemma. Earlier results showed
why one measure is not enough: B2 collapsed and still scored SSIM 0.58; B3 matched
E2's SSIM while replicating training layouts.

## Models, all fixed on validation

| Name | Model | Settings |
| --- | --- | --- |
| SD | Stable Diffusion 1.5, zero-shot | DDIM 30, guidance 7.5 |
| E0 | Codex's rank-4 adapter (run 629931) | same |
| **E2** | full UNet, epoch 10, seed 3710 | same |
| R1-E2 | E2 recipe, seed 4710 | same |
| **E3** | rank-4 adapter with E2's recipe, epoch 10, seed 3710 | same |
| R1-E3 | E3 recipe, seed 4710 | same |
| B3 | 92 M UNet from scratch, epoch 98 | same |
| B1 | conditional VAE (632570), 128 px | as trained |
| B2 | conditional GAN, step 29,000, 128 px | as trained |
| B2b | per-class GANs (rerun) | included only for the classes that produce a result |

No other checkpoint, guidance value or variant is evaluated on test. B3's epoch
49 and guidance 3.0 stay validation-only.

## Stage 0: code first, then a rehearsal on validation

New code, each part with CPU tests on tiny inputs:

- batched generation in `predict.py` (seeds stay per image);
- Inception-v3 feature extraction (torchvision ImageNet weights, pool layer,
  2,048-d);
- FID, KID and precision/recall/density/coverage (local, NumPy/SciPy, tested
  against hand-computed cases);
- the training-preference ratio and the membership test (below);
- sampling-speed profiling;
- DDIM inversion for interpolation, tested on a linear noise predictor where
  inversion must reconstruct its input.

The full pipeline then runs with **validation standing in for test**. It must
reproduce the existing validation SSIM of every 64-image cohort (for example,
E2 0.705). Batched generation must reproduce the existing 64 images for seeds
37200-37231 (identical hashes, or a maximum pixel difference of 1/255, which is
recorded). The code is then frozen; its commit is recorded in the ledger before
the test split is read.

If a bug is found after the test split is read, it is fixed, recorded as an
erratum, and every affected measure is recomputed for every model, never
selectively.

## Stage 1: test cohorts

1,024 images per model: seeds 37200-37711 for each class (512 AD, 512 NC). The
first 32 seeds per class are the existing cohorts, so the uncurated 64-image
grids stay comparable with everything reported so far.

## Stage 2: measures

### Corner 1: blurry or collapsed (fidelity and coverage)

| Measure | How | Why |
| --- | --- | --- |
| **Feature-nearest SSIM** (course criterion) | each sample against its nearest test slice of the same class in frozen-VAE feature space; mean and share above 0.6. Fixed-reference SSIM as before. Real floor: validation slices against test | the course's structural measure; insensitive to coverage (B2) |
| **KID** (primary) and FID | Inception-v3 features; 1,024 generated against the class-balanced test set (all 2,260 NC plus 2,260 random AD). KID: polynomial kernel, 100 subsets of 500. Also per class | distribution distance; KID is unbiased at 1,024 samples, FID is not and is reported for familiarity |
| **Coverage** (primary), precision, recall, density | Naeem et al. 2020, k = 5, Inception features; 1,024 generated against 1,024 class-balanced test slices, 10 draws of the real set; mean ± sd | separates "realistic" (precision, density) from "covers the data" (recall, coverage), the distinction SSIM misses |
| Same distances in VAE feature space | secondary | the spec allows latent distances; Inception is trained on ImageNet, not MRI |
| Within-class pairwise RMSE | as before | continuity with B2 and B2b |

**Real floor:** 1,024 class-balanced validation slices treated as a generator.
They give the scores a perfect generator would get at this sample size; every
model is read against it. B1, B2 and B2b are 128 px images enlarged to 256 px,
which costs them on sharpness-sensitive measures. This is noted beside their
numbers.

### Corner 2: hallucinated anatomy (plausibility)

- **Uncurated grids:** the first 32 seeds per class, every model.
- **Failure autopsy:** the recommended model's 5 lowest-SSIM samples out of
  1,024, plus every near-empty sample. Each is classified (checkerboard, distorted
  tissue margins, impossible junctions, partial or missing anatomy, intensity
  artefact). The selection rule is fixed here, so there is no cherry-picking.
  Synthetic only, so the figures can be public.
- **t-SNE** of VAE features: training and test slices coloured by true class,
  plus the first 64 samples of each model coloured by requested class.
- **A blinded real/synthetic sort by the student (non-expert).** 32 test slices
  and 32 recommended-model samples, shuffled with seed 3710; the student marks each
  real or synthetic without seeing the key. Accuracy near 50% means a non-expert
  cannot tell them apart, which is not clinical validation. Private (real slices).
  About 10 minutes.

### Corner 3: memorisation

| Measure | How | Why |
| --- | --- | --- |
| M1 nearest-training RMSE (continuity) | grayscale 64 px RMSE to the nearest of the 21,280 training slices; mean and minimum, against test slices' own nearest-training distances as the new real reference | the criterion used since E1; confounded, because smoother samples are closer to every real slice |
| **M2 training preference** (primary, distance) | per sample, log(nearest-training distance / nearest-test distance); model mean minus the same quantity for real validation slices (also held out). Pixel space primary, VAE features secondary. A negative difference whose 95% CI excludes 0 means samples sit closer to training data than held-out data would | removes the smoothness confound: a smooth sample is equally close to both sets, a replicated one is closer to training. The calibration absorbs the different set sizes (21,280 against 4,700) |
| **M3 membership inference** (primary, model-based) | diffusion models only. 1,024 training slices that E2, E3 and B3 trained on (512 per class, seed 3710) against 1,024 test slices. Score: noise-prediction loss with the true-class prompt, on the VAE posterior mean, averaged over t = 100, 200, 300 with 4 fixed noise draws each (the same draws for every model). AUC and true-positive rate at 1% false-positive rate. SD (never trained on ADNI) is the control for any train/test difference unrelated to training. Also reported: AUC at each t from 50 to 950, without selection | does the model treat training patients differently from unseen ones? This is the confidentiality question itself, and it needs no reference subsets (Carlini et al. 2023; Duan et al. 2023) |
| M4 closest pairs | the 5 closest generated-training pairs per model, visual review | the only check that can tell a copy from a near neighbour; private |

E0 is left out of M3: it trained for less than one epoch, so many "members"
were never seen.

### Interpolation (the spec's engineering recommendation)

- **I1, synthetic to synthetic:** E2 and E3; 4 seed pairs (2 AD, 2 NC); 9
  spherical-interpolation frames between the initial noises. Per frame:
  nearest-training and nearest-test distance and the distance to the previous
  frame. A path that jumps towards a training slice (an interior frame below the
  closest real-to-real distance) is flagged. Public figure.
- **I2, training scan to synthetic:** 4 training slices (2 per class, seed 3710)
  inverted to initial noise by DDIM inversion (30 steps, guidance 1.0, since
  inversion is accurate only without guidance), then interpolated towards a
  synthetic seed; same per-frame measures. Inversion quality is checked by
  regenerating the endpoint (SSIM to the original). Panels private (they contain
  a training slice); numbers public.

### Cost

- **Sampling:** seconds per image and peak memory at batch 1 and 16 (3 warm-up,
  20 timed batches, CUDA events, median) for SD-architecture models (E2, plus
  E3's unmerged adapter), B3, B1 and B2. GPU model recorded.
- **Training:** from the existing `metrics.json` files: wall time, images per
  second, peak memory, A100-hours, trainable parameters, weight-file size.

### Uncertainty

- Bootstrap 95% CIs (10,000 resamples, seed 3710) for SSIM, M2 and M3 (members
  and non-members resampled together for the M3 difference from SD). KID gives a
  subset sd; FID has no CI.
- Seed spread: R1-E2 and R1-E3 get every measure. A difference between E2 and E3
  is claimed only if it exceeds both the bootstrap CI and the seed-to-seed
  difference.

## Stage 3: recommended model, a rule fixed now

The candidates are E2 and E3; both are "latent diffusion fine-tuning" in the 2.4
table.

1. A candidate is excluded if its test mean feature-nearest SSIM is at most
   0.6, or if M4 shows a copy.
2. A candidate without a memorisation signal (M2 and M3 both clear) is preferred
   over one with a signal.
3. If both or neither are clear, the one with higher coverage is preferred,
   then lower KID, if the difference exceeds the seed spread.
4. Otherwise E3: a 1.6 MB adapter instead of a 1.7 GB UNet, and the weaker
   memorisation signal on validation.

## Verdict for the project manager

For the recommended model's cohort:

- **Structural fidelity:** passes if mean feature-nearest SSIM > 0.6 (course
  criterion).
- **Non-memorised:** no M2 or M3 signal and no copies in M4. If there is a
  signal, its size is stated (for example, "samples are x% closer to training
  slices than held-out slices would be").
- **Coverage:** comparative, with no pass threshold, since no prior exists for
  this data: position between the real floor and the baselines on KID and
  coverage.
- **Plausibility:** the autopsy, the grids and the blinded sort. The
  verdict says "plausible to a non-expert", never "clinically validated".
- **Downstream training:** not measured. A train-on-synthetic, test-on-real
  study is the next step before any use, and the verdict says so.

## Resources

- About 1.5 A100-hours on `a100-test` (20-minute jobs). Generation is batched;
  membership scoring is about 25,000 forward passes per model.
- Inception-v3 weights (torchvision, about 104 MB) downloaded once on the login
  node into `models/`.
- About 0.8 GB on Rangpur (cohorts about 54 MB per model). Home has 1.1 GB free
  before the R1-E3 and B2b outputs arrive. Proposed: delete the E1 epoch-10
  weights from Rangpur (1.7 GB; local backup with verified SHA-256). This needs
  the student's separate approval.
- Features of real slices stay private (`private-*` files, not committed); only
  aggregate numbers enter the repository.

## Schedule

19-20 October code and rehearsal; 21 October test run; 21-24 October README;
25 October pull request; deadline 26 October.

## What this does not show

- Clinical validity: no radiologist review.
- Downstream utility: no classifier trained on synthetic data.
- Generality: one dataset, one seed per baseline, two seeds for E2 and E3.
- FID on Inception features of MRI slices is a convention, not a validated
  measure for this domain; it is reported beside the domain-independent checks,
  not instead of them.
