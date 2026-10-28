## Failure autopsy and engineering recommendation

The five Improved examples above were visually inspected. Counts are observations;
appearance-based explanations below are hypotheses, not confirmed anatomical causes.
The last three panels were rendered on CPU from the same saved checkpoint. Aggregate
test tables retain the original frozen GPU pass. Corrected validation analysis matches
checkpoint Dice within 0.00002; batch-one versus batch-16 floating-point kernels can
change a few boundary pixels. The earlier failed image export was excluded.

| Actual validation case | Paired baseline / Improved ID5 Dice | Observed error and likely trigger | Practical response |
|---|---:|---|---|
| 036, week 0, slice 45 | 0.4976 / 0.0000 | Improved omits all 64 target pixels while baseline retains 52. A small local region is suppressed among surrounding tissue; added complexity does not consistently preserve small targets. | Review empty predictions near mask transitions; do not infer safety from overall Dice. |
| 037, week 0, slice 49 | 0.1120 / 0.0000 | All 315 target pixels are missed by Improved. In slices 49-52 the target shrinks from 315 to 155, 35 and 5 pixels, and all are omitted. This is a transition failure, not merely a single one-pixel error. | Inspect neighbouring slices together. Call this a mask-defined transition until the original plane and anatomical direction are established. |
| 038, week 0, slice 35 | 0.7778 / 0.6823 | Improved includes all 248 target pixels but adds 231 FP pixels, versus 137 FP for baseline. The red ring shows boundary expansion around a poorly separated region. | Require local contour review; improved global Dice does not guarantee better local specificity. |
| 038, week 0, slice 27 | 0.0000 / 0.6057 | Improved recovers 53 target pixels, misses 53 and adds 16 FP pixels; its prediction has two 4-connected components. The image shows a split small region. | Treat disconnected target proposals as review candidates. Any component filtering must be validated rather than assumed harmless. |
| 038, week 0, slice 42 | 0.0000 / 0.0000 | Improved misses all 56 target pixels and predicts a separate 15-pixel island. HD95 is 23.59 pixels. Baseline misses the target without this extra island. | A plausible-looking isolated contour can be misplaced; a single-component rule alone would not detect this failure. |

Across all 660 validation slices, complete positive-target misses fall from 17 to 14
and false-positive-only slices from 17 to 13. Median finite HD95 falls from 4.00 to
3.61 pixels, but finite-only summaries exclude one-empty failures and use slightly
different slice populations. These are descriptive validation findings, not proof of
clinical boundary benefit or millimetre accuracy.

On the frozen test split, Improved increases pooled ID5 Dice by 0.0161 and IoU by
0.0245. It improves two of three patients, while patient 041 decreases from 0.8295 to
0.8259. Peak allocated VRAM is about 3.06 times baseline, forward latency 2.84 times,
and training-loop time 1.30 times. Only three test patients and one training seed
support these observations; slice-wise significance claims would be unjustified.

**Recommendation:** retain Improved as the coursework Normal model and the basic
U-Net as the implemented comparison/fallback. Both fit comfortably in the measured
resource budget. The data exploration in README.md supports prostate (ID5) as the coursework target. Independent expert review of labels and original geometry is needed for practical use; these 2D headers do not establish anatomical base/apex. The modest pooled gain and persisting regressions do
not justify claiming that 2D architectural changes solve boundary safety. Defer Hard
3D work until these source questions and additional time/data requirements are clear.
