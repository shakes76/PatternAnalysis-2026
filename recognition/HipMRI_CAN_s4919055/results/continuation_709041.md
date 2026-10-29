# Verified 2D continuation update - 28 October 2026

Later verification: task 709282 reproduced best epoch-7 prostate Dice 0.694518 and IoU 0.535584 on all 48 validation scans. The current README includes its full class table and three inspected validation error examples. The notes below preserve the distinction from the epoch-8 report.

This update supplements the report snapshot written after the three-epoch
Dice+CE pilot. Task 709041 resumed task 708618 through epoch 8. Downloaded
config/history/latest-validation reports, both PNGs and both prediction
NIfTIs were inspected. Four training source hashes and the exact split match
the pilot; the first three history entries are unchanged. All epoch-8
patient-macro class metrics were independently recomputed from 48 case records.

| Epoch | Training Dice+CE objective | Validation patient-macro prostate Dice |
|---:|---:|---:|
| 4 | 0.281913 | 0.645411 |
| 5 | 0.236005 | 0.593976 |
| 6 | 0.205426 | 0.662736 |
| 7 | 0.176910 | **0.694518** |
| 8 | 0.170520 | 0.606007 |

The best history entry is epoch 7, not the last epoch. A falling training
objective does not guarantee improving validation overlap. This short history
alone does not establish the cause of the fluctuations.

The complete downloaded validation report is **epoch 8**:

| Class | Patient-macro Dice | Patient-macro IoU |
|---|---:|---:|
| Background | 0.994840 | 0.989788 |
| Body | 0.972516 | 0.946702 |
| Bones | 0.819696 | 0.698185 |
| Bladder | 0.796919 | 0.686400 |
| Rectum | 0.643449 | 0.491698 |
| Prostate | 0.606007 | 0.444322 |

These class values must not be combined with epoch 7's prostate score and
presented as one checkpoint's results. A validation-only recheck of best.pt is
prepared in validate_best.py and baseline_validate_best.slurm; its real output
was independently inspected after task 709282. It checks the known preview checkpoint hash, records the
checkpoint epoch, recomputes the full validation metrics, verifies the saved
score, and saves low-scoring examples from three different patient groups.
It performs no optimization and does not open the final test split.

The existing best.pt previews share SHA256
`094deac975b46ecd9c0fa497d9581fd188d120bd34c318b9897e6bf3239a303c`:

- B006_Week0 (training): Dice 0.617063, 1,287 true and 1,280 predicted prostate
  voxels. The first labelled slice is missed; the larger displayed slices have
  both false-positive and false-negative boundary regions. Similar total
  volumes do not imply correct locations.
- C032_Week0 (validation): Dice 0.532072, 683 true and 1,749 predicted prostate
  voxels. The displayed small first/last slices have excess prediction, and the
  largest slice has both excess and missed regions. This is visible
  over-segmentation in one case, not evidence that every validation case fails.

Both saved masks are 128 x 128 x 64 on an RAS grid; independently counted label
voxels match the JSON reports. No clinical base/apex designation or definitive
root cause is inferred from these panels. Raw reports, medical-volume outputs
and these private preview files remain outside Git.

This is still an implemented **2D U-Net** baseline. A local CAN3D prototype is now implemented and synthetic-tested; real GPU training and final held-out evaluation are pending. The two current cases do not complete the required
3-5 representative final-model failure analyses or a controlled cost benchmark.
