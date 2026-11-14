# B2b protocol: one GAN per class, after B2's conditional collapse

Approved by the student 2026-10-17 (Australia/Brisbane), as the rerun option
described after B2's evaluation, before any B2b job ran. This file is not edited
afterwards; results go in `experiments.md`.

## Why

B2's conditional GAN ignored its noise from the first check and drew one image
per class (conditional mode collapse; Mao et al., CVPR 2019). Its collapse rule
did not fire because the diversity ratio pooled both classes. B2 is kept and
reported as a result. B2b asks whether a GAN baseline that does not collapse
can be obtained with the student's demonstration-2 recipe as validated there:
an unconditional GAN, here one per class, with every collapse check made within
a class.

## The changes from B2

| Setting | B2 | B2b |
| --- | --- | --- |
| Conditioning | one GAN, AD/NC label embedding and projection | two GANs, one trained only on AD slices and one only on NC slices; each sees a single label, so its label input is constant and it behaves as an unconditional GAN, as in demonstration 2 |
| Minibatch standard deviation | groups mixed both classes | every batch is one class, so the feature measures within-class spread |
| Collapse rule's diversity ratio | 64 generated with alternating labels, against 64 random training slices | per GAN: 64 generated, against 64 random training slices of that class |
| Steps | 30,000 planned, 29,000 run | 20,000 per GAN, at batch 64 (about 126 epochs of AD, 115 of NC) |
| Time limit | 2 h request, 105 min budget | 1.5 h request and 95 min budget per GAN |

Unchanged: architecture, non-saturating logistic loss with R1 gamma 10, no
spectral normalisation, no augmentation, Adam 2e-4 with betas (0, 0.99), EMA
0.999, 128 px, E2's defective-scan exclusion (2 scans / 40 slices, applied to the
whole training split before splitting by class), and the rule's thresholds:
from step 5,000, snapshot when the ratio is at least 0.8; stop and keep the last
snapshot after two consecutive ratios below 0.6.

## Stages

1. **S1 probe**, `a100-test`: 500 steps of the AD GAN, collapse checks from step 0.
2. **S2**, `comp3710`: the AD GAN, then the NC GAN, as two jobs.
3. **S3**: one 64-image cohort, seeds 37200-37231 from each class's result
   generator, then the same SSIM, appearance, feature, nearest-neighbour and
   closest-pair audits as every other model.

About 3 A100-hours. One rerun at most, for infrastructure failure only.

## Reported, without pass thresholds

- Within-class pairwise RMSE against real slices' (about 0.15), and visual review
  of the uncurated grid: did the per-class GANs avoid collapse?
- SSIM, edge, border and nearest-training measures beside B1, B2, E2 and E3.
- Whether either collapse rule fired, and at which step the result comes from.

## What this does not show

- Two GANs at roughly twice B2's compute; the comparison with other models is
  between reasonable designs, not matched budgets.
- One seed per class.
