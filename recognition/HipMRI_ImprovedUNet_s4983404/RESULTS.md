## Frozen 20-epoch experiment results

Numeric IDs are retained for traceability; ID5 is interpreted as prostate using the anatomical exploration documented in README.md.
Checkpoints were selected using validation only; test argmax was evaluated once per model.

| Model / job | Epochs / selected epoch | Best val ID5 Dice | Test ID5 Dice | Test ID5 IoU |
|---|---:|---:|---:|---:|
| basic / 679897 | 20 / 19 | 0.8258 | 0.8453 | 0.7321 |
| improved / 679896 | 20 / 9 | 0.8466 | 0.8614 | 0.7566 |

| ID | Basic Dice | Improved Dice | Basic IoU | Improved IoU |
|---|---:|---:|---:|---:|
| 0 | 0.9965 | 0.9968 | 0.9931 | 0.9936 |
| 1 | 0.9838 | 0.9850 | 0.9682 | 0.9704 |
| 2 | 0.9344 | 0.9376 | 0.8768 | 0.8826 |
| 3 | 0.9652 | 0.9646 | 0.9328 | 0.9317 |
| 4 | 0.8314 | 0.8635 | 0.7115 | 0.7599 |
| 5 | 0.8453 | 0.8614 | 0.7321 | 0.7566 |

| Patient | Basic ID5 Dice | Improved ID5 Dice |
|---|---:|---:|
| 040 | 0.8521 | 0.8662 |
| 041 | 0.8295 | 0.8259 |
| 042 | 0.8341 | 0.8664 |

| Model | Parameters | Peak allocated / reserved MiB | Train loop s | Batch-one latency ms |
|---|---:|---:|---:|---:|
| basic | 1,940,902 | 661.6 / 974.0 | 706.0 | 2.06 |
| improved | 3,189,714 | 2026.6 / 3026.0 | 916.7 | 5.84 |

Training-loop time excludes cache loading. Slurm wall time is recorded in the retained logs.
Latency excludes I/O; five warmups, 30 synchronized batch-one forwards on A100-PCIE-40GB.
Both models use the frozen shared recipe in [evaluation plan](EVALUATION_PLAN.md).

![Basic learning curves](figures/basic_learning_curves.png)

![Improved learning curves](figures/improved_learning_curves.png)

## Validation boundary failures

Basic: five worst positive-label slices. Improved: two worst misses plus three explicit error modes.
Selection is deliberately biased toward failures; these panels are not test averages.
ID5 is interpreted as prostate as documented in README.md. HD95 uses symmetric pooled 4-connected surface distances in pixels.
Both-empty and one-empty distances are undefined, with reasons retained in JSON.

**basic:** median finite slice HD95 4.00 px (237 finite; 389 both-empty; 34 one-empty). Finite-only median omits total misses.

| Slice | Dice | FP / FN pixels | HD95 px | Predicted components |
|---|---:|---:|---:|---:|
| case_036_week_0_slice_17.nii.gz | 0.0000 | 0 / 17 | undefined | 0 |
| case_037_week_0_slice_10.nii.gz | 0.0000 | 0 / 56 | undefined | 0 |
| case_037_week_0_slice_11.nii.gz | 0.0000 | 0 / 84 | undefined | 0 |
| case_037_week_0_slice_50.nii.gz | 0.0000 | 0 / 155 | undefined | 0 |
| case_037_week_0_slice_51.nii.gz | 0.0000 | 0 / 35 | undefined | 0 |

![basic failure 1: input, GT, prediction, red FP and blue FN](figures/basic_failure_1.png)

![basic failure 2: input, GT, prediction, red FP and blue FN](figures/basic_failure_2.png)

![basic failure 3: input, GT, prediction, red FP and blue FN](figures/basic_failure_3.png)

![basic failure 4: input, GT, prediction, red FP and blue FN](figures/basic_failure_4.png)

![basic failure 5: input, GT, prediction, red FP and blue FN](figures/basic_failure_5.png)

**improved:** median finite slice HD95 3.61 px (240 finite; 393 both-empty; 27 one-empty). Finite-only median omits total misses.

| Slice | Dice | FP / FN pixels | HD95 px | Predicted components |
|---|---:|---:|---:|---:|
| case_036_week_0_slice_45.nii.gz | 0.0000 | 0 / 64 | undefined | 0 |
| case_037_week_0_slice_49.nii.gz | 0.0000 | 0 / 315 | undefined | 0 |
| case_038_week_0_slice_35.nii.gz | 0.6823 | 231 / 0 | 5.00 | 1 |
| case_038_week_0_slice_27.nii.gz | 0.6057 | 16 / 53 | 4.00 | 2 |
| case_038_week_0_slice_42.nii.gz | 0.0000 | 15 / 56 | 23.59 | 1 |

![improved failure 1: input, GT, prediction, red FP and blue FN](figures/improved_failure_1.png)

![improved failure 2: input, GT, prediction, red FP and blue FN](figures/improved_failure_2.png)

![improved failure 3: input, GT, prediction, red FP and blue FN](figures/improved_failure_3.png)

![improved failure 4: input, GT, prediction, red FP and blue FN](figures/improved_failure_4.png)

![improved failure 5: input, GT, prediction, red FP and blue FN](figures/improved_failure_5.png)
