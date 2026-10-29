# Experiment evidence index

## Frozen final test: job 710105, 29 October 2026

The fixed CAN3D and U-Net epoch-7 checkpoints were evaluated on the same27 test scans/six patient groups in one A100 allocation. Exact plan bytes, all seven evaluation-source hashes, checkpoint identities and the original split were independently checked. All-class patient/scan means, repeated timing summaries and peak-memory summaries were recomputed. All54 exported predictions have matching RAS grids and class counts. Five shared case pairs (ten full-field PNGs) were visually inspected. Display zooms do not change model inputs or predictions. Six-neighbour prostate component sizes are descriptive; no test-driven filtering or retraining was applied.

- test_summary_710105.csv: full precision patient/scan-macro Dice and IoU for all six classes.
- test_cases_710105.csv: per-class case metrics and prediction counts for both models.
- test_resources_710105.csv: parameters, mean case-median time, allocated/reserved memory and device/job provenance.
- final_test_710105.json: selected checkpoint identities, protocol, runtime and aggregate results.
- test_components_710105.json: descriptive disconnected-prediction component sizes.

## Historical development evidence

| Job | Evidence and interpretation |
|---|---|
| 696947 | Tiny CUDA diagnostic; not model accuracy or CAN3D memory needs |
| 699944 | Three training-volume loader checks; duplicate699945 cancelled |
| 700021 | Selected-training-slice overfit/debug, not held-out validation |
| 703884 | All-slice Dice-only baseline;25 completed epochs then TIMEOUT; downloaded history/latest validation |
| 704488 | Matched three-epoch balanced-slice Dice-only baseline; downloaded reports |
| 704533 | Two historical failed-model prediction panels/NIfTIs; CPU diagnostic |
| 708234 | Dice-only continuation through epoch8; screenshot evidence only, detailed JSON unavailable |
| 708618 | Fresh matched three-epoch balanced Dice+CE pilot; source hashes checked against preserved50e8ee7 revision |
| 709041 | U-Net continuation through epoch8; retained first3epoch history checked |
| 709282 | Complete48-scan revalidation of U-Net bestepoch7; three distinct validation-case panels |
| 709556 | Five-step CAN3D training-only smoke; raw JSON checked against preserved source revision |
| 709710 | Three-epoch CAN3D whole-volume pilot, exact manifest/source and full validation checked; see can3d_pilot_709710.md |
| 709852 | CAN3D continuation through epoch8, bestepoch7 full validation and three inspected low-case panels; see can3d_resume_709852.md |

Validation values in validation_summary.csv and learning_history.csv are historical, and must not be relabelled as test metrics. Exact downloaded MRI/prediction/weight files and audit notes remain outside Git. Small result figures/CSV/JSON are report evidence. The original one-page feasibility text remains unchanged in feasibility_review_original.md.
