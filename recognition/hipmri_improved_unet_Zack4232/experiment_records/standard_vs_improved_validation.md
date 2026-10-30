# Standard and improved U-Net validation comparison

Both models used CE+Dice, seed 42, batch size 8, learning rate 0.001, and a 20-epoch budget. The selected checkpoints were standard U-Net run 711775, epoch 10, and Improved U-Net run 712829, epoch 3.

Sources: `ce_dice_712115_slice_metrics.csv` and `improved_712829_slice_metrics.csv`. All 660 case/week/slice keys matched uniquely, with equal image shapes and ground-truth prostate areas. Dice and IoU were independently recomputed from pixel counts and matched the CSV values.

## Positive-slice results

The primary metric is the equal mean of prostate Dice across 254 ground-truth-positive slices. Empty predictions score zero. The remaining 406 slices are excluded from this metric and assessed separately for false positives.

| Ground-truth area (pixels) | Slices | Standard Dice | Improved Dice | Standard empty predictions | Improved empty predictions |
|---|---:|---:|---:|---:|---:|
| 1-100 | 30 | 0.318449 | 0.397617 | 4 | 3 |
| 101-500 | 126 | 0.786899 | 0.786413 | 0 | 0 |
| Over 500 | 98 | 0.865939 | 0.859939 | 0 | 0 |
| All positive slices | 254 | 0.762066 | 0.768861 | 4 | 3 |

These area groups are exploratory pixel-based groups, not clinical size thresholds. The small-target group accounts for the net mean Dice gain; the other two group means decreased. Across paired slices, 120 improved, 130 worsened, and 4 were unchanged. Zero-overlap predictions decreased from 8 to 4. Unlike empty predictions, zero-overlap predictions include cases where a prediction exists but misses the target entirely. The improved model has one zero-overlap result in the 101-500 pixel group.

## Negative-slice results

| Measure | Standard | Improved |
|---|---:|---:|
| Slices with any prostate false positive | 40 | 52 |
| False-positive pixels in total | 2367 | 3396 |
| Median false-positive pixels among affected slices | 22 | 43.5 |
| Maximum false-positive pixels on one slice | 530 | 389 |
| Affected slices with over 100 false-positive pixels | 6 | 10 |

There were 23 newly affected slices and 11 resolved slices. Pixel counts cannot establish whether errors are connected or scattered.

## Paired examples

| Case / week / slice | Standard Dice | Improved Dice | Observation from pixel counts |
|---|---:|---:|---|
| 37 / 0 / 9 | 0.000000 | 0.086957 | Only 2 of 19 target pixels overlap the new prediction |
| 37 / 0 / 10 | 0.000000 | 0.444444 | Partial recovery, with 91 false-positive pixels |
| 37 / 0 / 11 | 0.628931 | 0.385057 | False negatives decrease from 34 to 17, but false positives increase from 25 to 197 |
| 38 / 0 / 27 | 0.393939 | 0.417910 | Slight gain, substantial under-segmentation remains |
| 38 / 0 / 36 | 0.761578 | 0.870175 | Both false positives and false negatives decrease |
| 37 / 0 / 37 | 0.781250 | 0.690909 | False negatives increase from 342 to 454 |

## Provisional engineering conclusion

The improved architecture raises mean validation Dice by 0.006795, mainly through the smallest targets, but increases false positives and does not improve all target sizes. Existing training logs show about 1.64 times as many parameters and 3.20 times peak allocated GPU memory. These results do not yet justify a general recommendation to replace the standard model.

This is one seed and four validation cases. Slices from a case are related. No statistical significance or test-set performance is established. Preserve both selected checkpoints and freeze the comparison settings before final test evaluation. Report per-class metrics, per-case results, inference costs, and representative failures in the final evaluation.
