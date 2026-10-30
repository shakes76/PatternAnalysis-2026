# CE and CE+Dice validation comparison

Sources: `baseline_710213_slice_metrics.csv` and `ce_dice_712115_slice_metrics.csv`.
The CE checkpoint is from epoch 16 of run 710034. The CE+Dice checkpoint is from epoch 10 of run 711775. Both were selected by validation prostate Dice.

## Checks and metric protocol

660 unique matching keys; equal shape and GT area; Dice and IoU recomputed from counts: PASS
The main Dice is the equal mean across 254 slices with ground-truth prostate pixels. The other 406 slices are excluded and their false positives are reported separately. Empty predictions on eligible slices score zero.

## Target area groups

Area groups are exploratory and measured in pixels. They are not clinical size thresholds.

| GT prostate pixels | Slices | CE Dice | CE+Dice Dice | CE empty predictions | CE+Dice empty predictions |
|---|---:|---:|---:|---:|---:|
| All | 254 | 0.730262 | 0.762066 | 18 | 4 |
| 1-100 | 30 | 0.193860 | 0.318449 | 14 | 4 |
| 101-500 | 126 | 0.761362 | 0.786899 | 4 | 0 |
| 501+ | 98 | 0.854479 | 0.865939 | 0 | 0 |

## Previously inspected slices

| Case / week / slice | GT pixels | CE Dice | CE+Dice Dice | CE+Dice predicted pixels |
|---|---:|---:|---:|---:|
| 37 / 0 / 9 | 19 | 0.000000 | 0.000000 | 0 |
| 37 / 0 / 10 | 56 | 0.000000 | 0.000000 | 0 |
| 37 / 0 / 11 | 84 | 0.000000 | 0.628931 | 75 |
| 38 / 0 / 27 | 106 | 0.000000 | 0.393939 | 26 |
| 38 / 0 / 36 | 262 | 0.850365 | 0.761578 | 321 |
| 37 / 0 / 37 | 967 | 0.499612 | 0.781250 | 633 |

## False positives on GT-negative slices

| Measure | CE | CE+Dice |
|---|---:|---:|
| fp_slices | 13 | 40 |
| fp_pixels | 1317 | 2367 |
| median_fp_when_present | 66 | 22.0 |
| max_fp | 346 | 530 |
| 1-10 | 2 | 15 |
| 11-100 | 6 | 19 |
| 101+ | 5 | 6 |

New FP slices: 29. Resolved FP slices: 2.

## Interpretation and limits

Use the paired counts above to assess the balance between missed targets and false positives. Pixel counts do not show whether errors form connected regions or scattered pixels. Spatial claims require prediction images.
This is one seed on four validation cases. Slices from the same case are related. The results do not establish statistical significance or test performance. The main metric remains slice-weighted, not an equal mean of case means.
Keep both loss experiments as evidence. Use the same loss and training settings for the standard and improved U-Net architecture comparison. Keep the test split unused until the experiment settings are fixed.
