# Final test protocol — before test evaluation

Recorded 2026-10-29, after validation exploration and before any test evaluation reported in this chat.

## Fixed comparison

- Weights: baseline640/weights/best.pt and compare960/weights/best.pt, each selected by validation fitness during its 50-epoch training.
- Input sizes: 640 and 960 respectively; same test image inventory, original class order, no test-time augmentation.
- AP evaluation: Ultralytics 8.4.191, low confidence floor 0.001, class-aware NMS IoU 0.7. Retain the framework's AP protocol and explicitly distinguish it from the single-class-per-candidate inference operating analysis.
- Operating comparison: fixed confidence 0.15 for both, class-aware one-to-one matching at IoU 0.5 and 0.8; report per-class TP/FP/FN/P/R and all-class FP per image.
- The 0.15 choice is post-hoc exploratory validation selection, fixed before test scoring; no claim that this is a clinically adequate policy.
- Do not retune weights, threshold or resolution based on test results. Report both models even if neither meets the assignment's localization target.

Report-finalization clarification (2026-10-29): the assignment specifies test localization IoU >= 0.8, suitable classification performance, detection metrics and a sensitivity/false-alarm investigation. It specifies no numerical recall pass criterion. An earlier optional recall gate remains in archived threshold JSONs for provenance and is not used to judge assignment compliance. This clarification changes no pre-test choice, weights, predictions, thresholds or test metrics.

## Leakage and storage checks before scoring

Use official ISIC 2017 test images, segmentation masks and diagnosis labels. Match IDs and dimensions. Check decoded exact duplicates across train, validation and test; investigate near duplicates. Membership checks do not independently establish patient identities.

The previously excluded training duplicate remains excluded. If test overlaps with any image used in training or validation, do not simply call that image an independent test case. Since weights/selection are already fixed, document and exclude overlapping test cases from a clearly labelled decontaminated test subset before scoring, or retrain/reselect under a revised protocol. Preserve original memberships and exclusion reasons. Conflicting labels or uncertain matches require investigation, not silent removal. Do not use test performance to decide exclusions.

Use hard links or another verified nonduplicating export method on the same filesystem where supported; otherwise check available space before making a full copied export. Do not delete raw files or weights to resolve storage pressure. Previously verified, fully extracted ZIP downloads can be removed because they are reproducible download caches.

## Validation context (student-reported terminal evidence)

| Fixed conf 0.15 | 640 | 960 |
|---|---:|---:|
| melanoma TP, IoU >= 0.5 | 11 | 16 |
| melanoma FP, IoU >= 0.5 | 7 | 10 |
| melanoma FN, IoU >= 0.5 | 19 | 14 |
| melanoma Recall, IoU >= 0.5 | 0.367 | 0.533 |
| melanoma TP, IoU >= 0.8 | 8 | 5 |
| melanoma FP, IoU >= 0.8 | 10 | 21 |
| melanoma FN, IoU >= 0.8 | 22 | 25 |
| melanoma Recall, IoU >= 0.8 | 0.267 | 0.167 |

This supports a resolution trade-off, not universal superiority of 960. Figures are not test results. Import the actual school-side JSONs before final report publication.
