# Frozen standard 3D U-Net: held-out test result

For a complete source-to-method explanation, training trajectory, six-class
result table and report-ready interpretation, see
[完整实验记录（中文）](../../how_to_run/methods.md).

Evaluation job 715670 completed on 2026-10-30 (Australia/Brisbane). The
checkpoint was selected using validation data, then frozen before testing.
This directory preserves the cluster's original JSON/CSV outputs, refinement
history/configuration/split and five prediction figures. No weights, raw MRI
volumes or cached inference arrays are included.

## Main result

| Prostate metric | Held-out test result |
| --- | ---: |
| Mean scan Dice | 0.7732160094 |
| Standard deviation of scan Dice | 0.0557164621 |
| Mean patient Dice | 0.7678182947 |
| Pooled Dice | 0.7729904799 |
| Mean HD95 | 6.4995133 mm |
| Pooled precision | 0.7147264873 |
| Pooled recall | 0.8415968542 |
| Pooled predicted / GT volume ratio | 1.1775089759 |
| Predicted prostate voxels inside GT bladder | 89 |
| Predicted prostate voxels inside GT rectum | 21 |

All 27 test scans (six patients) had a valid prostate HD95. Distances and
voxel counts refer to the resized 64x128x128 grid. HD95 millimetres are
approximate shape-ratio-adjusted distances, not original-resolution clinical
measurements. The model tends to oversegment: pooled predicted volume is
17.8% larger than ground truth. Full per-case and per-class results are in
the CSV files and `summary.json`; do not confuse pooled, scan and patient
averages.

## Model and training provenance

Standard 3D U-Net with base width 8, InstanceNorm3d, 351,206 parameters,
without residual blocks or deep supervision. Patient-disjoint split:
145 training / 39 validation / 27 test scans, or 26 / 6 / 6 patients.

Scratch run 714988 supplied its last checkpoint at epoch 15. Refinement
run 715380 started a new optimizer with learning rate 0.0005 and training-only
class weights; its selected checkpoint is refinement epoch 11 (26 cumulative
epochs), with mean validation prostate Dice 0.7559718763. This is a two-phase
training package, not evidence isolating the effect of class weights alone.

Frozen checkpoint SHA256:
`77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156`.
The actual checkpoint remains on the cluster at
`results/refine_v2/best_model.pt`.

Duplicate source folders have been consolidated into canonical root code,
`support/train_phase1/scratch.py` and root `train.py`. Original submitted sources are
recoverable through [the provenance manifest](../../old_code_and_notes/source_snapshots.json).
Current Slurm scripts are under `../../scripts/`; their resource requests are
unchanged. Earlier training code is under `../../history/`.

To retrain, the scratch stage must finish before running refinement, and the
dataset must be available at the configured course path. Evaluation requires
the private checkpoint and saved split; uploading these sources does not
include the checkpoint. `../../docs/evaluation.md` explains evaluation.

## Figures and comparison limits

The five PNG examples were chosen deterministically to cover difficult
Dice/boundary/organ-confusion/missed-voxel cases and a middle-Dice case, with
different patients preferred. They are diagnostic examples, not a random
sample. Each figure shows MRI, GT, prediction and prostate TP/FP/FN.
Superior/inferior extent rows are GT-based proxies, not independently
verified clinical apex/base annotations. Figure contents have been generated;
the displayed views have now been reviewed against the numerical records in
`../../docs/case_analysis.md` and the main project README. This does not constitute
an independent clinical annotation audit or inspection of all slices.

`shared_2d_region_metrics.csv` restricts evaluation to the supplied 2D slice
band on the resized grid. It is not the full-volume test score and is not
automatically comparable with native-resolution 2D scores. Do not tune the
model against this held-out test set.

BF16 forward-only median latency was 17.55 ms per scan on an A100 40GB,
excluding loading, transfer, argmax, metrics and plotting. Peak evaluation
GPU tensor allocation was 0.1364 GiB, not training memory. Exact environment
versions and metric definitions are stored in `summary.json`.

Implementation, evaluation adaptation and documentation used AI assistance;
the final report should retain the full project AI-use disclosure.

## Checked case review

See [the five-case analysis](../../how_to_run/case_analysis.md) and
[`case_review_metrics.csv`](case_review_metrics.csv) for reviewed observations,
boundary-region counts, selection reasons and figure hashes.

## Comparison scopes

The [comparison tables](../../../README.md) keep this complete-volume
result separate from the measured shared-band overlap scores. Shared-band HD95
and organ-confusion counts were not measured in the archived band CSV and
remain blank. Native 2D test scores also remain blank until a frozen 2D
checkpoint is independently evaluated. The
[protocol](../../how_to_run/comparison_protocol.md) fixes the patient and grid alignment rules.
