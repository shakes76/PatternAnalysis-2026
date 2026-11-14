# E2 protocol: the E1 recipe without two defective training scans

Approved by the student 2026-10-13 (Australia/Brisbane), before any E2 job ran,
together with deleting the E1 epoch-5 weights from Rangpur after a local backup.
This file is not edited afterwards; results go in `experiments.md`.

## Why

E1 met the fidelity criteria but failed the memorisation criterion. Seven of
its 32 epoch-10 NC samples are near-empty (under 5% foreground), and the nearest
training slice of every one belongs to the same NC scan. A census of real
slices found near-empty slices only in NC training data (0.26%), none in AD or
validation.

The exclusion rule below, applied to the training split (job 656452), flags
**2 NC scans from 2 subjects, 40 of 21,280 training slices (0.19%)**. One scan
has 19 of 20 slices near-empty, the other 11. Private panels of all their
slices, next to a typical NC scan, show failed acquisition or preprocessing, not
unusual anatomy: one is blank apart from scattered fragments, the other is dark
and grainy with only a ghost outline. These two appearances match E1's two
NC failure types (blank, and dim grainy ghost).

## Hypotheses

- **H1 (primary).** E1's near-empty and ghosted NC samples are learned from
  these two scans. Training without them removes those samples.
- **H2.** With that failure mode gone, the unchanged memorisation criterion is
  re-tested. Stated in advance: it may still fall just short, because E1's
  non-empty samples already averaged 0.0712 against the 0.0731 reference.
- **H3 (no regression).** Removing 0.19% of training slices does not weaken the
  fidelity results.

## The one change

`train.py --exclude-defective-scans` drops every training scan in which any
slice has under 5% of its pixels above 10% intensity, measured on the 256 px
loader output (`dataset.defective_scans`). The rule is computed from the
training split at run time; the run records the number of scans and slices
removed, and keeps scan IDs in a private file. Expected: 2 scans, 40 slices. If
the run removes a different number, it stops before training.

Validation is unaffected (0 flagged). The test split stays sealed. Because the
rule is fixed now, from training data only, the final test protocol may apply
it to test scans without making a test-informed choice; that decision belongs
to the final protocol.

Everything else is E1's configuration: full UNet, 10 epochs, batch 16, AdamW
lr 1e-5 with 500 warm-up steps, 10% empty-prompt dropout, offset noise 0.1, EMA
0.999, seed 3710, SD1.5 revision 451f4fe, 256 px, the same prompts, the same
1,024 validation slices and the same sample seeds. With 21,240 slices an epoch
is 1,327 steps (13,270 in total, E1 13,300). The same seed over a shorter list
also changes the batch order, so E1 and E2 differ in data order as well as in
the 40 slices; this cannot be avoided and is reported.

## Stages and budget

1. **S1 probe**, `a100-test`, 20 min cap: full training split with the filter,
   200 steps, weight save and reload. Confirms 2 scans and 40 slices removed.
2. **S2 main run**, `comp3710`, 1.5 h request, 75 min budget (E1 took 45 min).
   EMA weights at epochs 5 and 10.
3. **S3 evaluation**, identical to E1: 64-image cohorts from epoch 10
   (primary) and epoch 5 (secondary), frozen-VAE features, SSIM, pixel nearest
   neighbours, appearance, closest-pair panels.

About 1.2 A100-hours in total. One rerun at most, for infrastructure failure only.

**Storage.** Home has 5.0 GB free and E2 saves 3.4 GB of weights, while the
training script requires 5 GiB free before starting. Plan: copy both E1 weight
files to the local D: drive, verify their SHA-256, then delete E1's epoch-5
file on Rangpur (1.7 GB). Its cohort is already generated and evaluated.

## Acceptance criteria, validation data only

| Criterion | E1 epoch 10 | E2 epoch 10 passes if |
| --- | ---: | --- |
| **NC samples under 5% foreground (H1)** | 7/32 | **<= 1/32** |
| AD samples under 5% foreground | 0/32 | 0/32 |
| Feature-nearest SSIM, mean | 0.688 | >= 0.6 |
| Mean outer-border intensity | 0.0016 | <= 0.01 |
| Mean colour range | 0.0061 | < 0.0208 (E0) |
| Generated-to-training nearest RMSE, mean (H2) | 0.0655 | >= 0.0731, and no near-copy among the 5 closest pairs on visual review |

The fidelity and memorisation targets are E1's, unchanged. The 1/32 allowance
in H1 tolerates one unexplained outlier; the real NC rate of 0.26% predicts 0.1
of 32. Epoch 5 is evaluated the same way and reported as secondary. E1 versus
E2 differences on every row are reported whether or not they pass.

## What this does not show

- Removing two scans is data cleaning on a disclosed, pre-stated rule; it is
  not tuning on the evaluation. The counts are reported; scan IDs stay private.
- Fewer blank outputs would show that the model reproduced those scans; it
  would not prove the absence of anatomical memorisation elsewhere. Pixel
  distance cannot do that.
- If H1 fails, the blank outputs have another cause (for example, an empty
  mode of the prompt-dropout branch), and that is reported instead.
