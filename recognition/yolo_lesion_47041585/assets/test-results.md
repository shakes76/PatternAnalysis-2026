# Test results

Split: test, 189 images (36 malignant, 153 benign) from 93 patients. Thresholds were frozen on validation (2026-11-05T04:13:17+00:00, final: True) and applied unchanged. Intervals are patient-level 95% (2000 of 2000 resamples valid).

## Frozen settings

| | candidate | baseline |
|---|---|---|
| run | A0-s47041585 | B0-s47041585 |
| checkpoint sha256 | e39a71cd2451 | 8e2dddce7c7a |
| tau_m | 0.07 | 0.01 |
| tau_b | 0.001 | 0.001 |
| clear guard tau_c | None | None |

## Operating-point results

| quantity | A0 | B0 | A0 minus B0 |
|---|---:|---:|---:|
| malignant detected and alerted | 66.7% [45%, 86%] | 61.1% [41%, 82%] | +5.6 points [-14, +27] |
| benign routed urgent (biopsy burden) | 30.7% [22%, 39%] | 41.8% [34%, 51%] | -11.1 points [-20, -2] |
| benign routed monitor | 0.0% [0%, 0%] | 0.0% [0%, 0%] | +0.0 points [+0, +0] |
| benign routed clear | 69.3% [61%, 78%] | 58.2% [49%, 66%] | +11.1 points [+2, +20] |
| benign escalation (urgent + monitor) | 30.7% [22%, 39%] | 41.8% [34%, 51%] | -11.1 points [-20, -2] |
| malignant routed monitor | 0.0% [0%, 0%] | 5.6% [0%, 17%] | -5.6 points [-17, +0] |
| malignant routed clear (safety cost) | 33.3% [14%, 55%] | 30.6% [14%, 50%] | +2.8 points [-12, +18] |
| benign targets matched (IoU >= 0.5) | 99.3% [98%, 100%] | 94.1% [89%, 98%] | +5.2 points [+2, +10] |
| malignant targets matched (IoU >= 0.5) | 66.7% [45%, 86%] | 61.1% [41%, 82%] | +5.6 points [-14, +27] |
| mean matched IoU, benign | 0.890 [0.875, 0.903] | 0.857 [0.838, 0.873] | +0.034 [+0.021, +0.048] |
| mean matched IoU, malignant | 0.872 [0.833, 0.909] | 0.847 [0.792, 0.900] | +0.024 [-0.025, +0.080] |

## Diagnosis by action (image counts)

**A0**

| true class | clear | monitor | urgent | total |
|---|---:|---:|---:|---:|
| Benign | 106 | 0 | 47 | 153 |
| Malignant | 12 | 0 | 24 | 36 |

**B0**

| true class | clear | monitor | urgent | total |
|---|---:|---:|---:|---:|
| Benign | 89 | 0 | 64 | 153 |
| Malignant | 11 | 2 | 23 | 36 |

## Detector average precision (Ultralytics validator, same split)

| class | A0 AP50 | A0 AP50:95 | B0 AP50 | B0 AP50:95 |
|---|---:|---:|---:|---:|
| Benign | 0.904 | 0.709 | 0.838 | 0.585 |
| Malignant | 0.535 | 0.357 | 0.266 | 0.182 |

## Targets for A0 (feasibility review)

- Malignant sensitivity at least 90%: 66.7%, NOT met.
- Benign escalation at most 30% at the same point: 30.7%, NOT met.
- Mean matched IoU at least 0.8: benign 0.890 (met), malignant 0.872 (met), over matched targets only; coverage is in the table above.
- Latency target: see artifacts/latency.json (measured separately on the validation split).

Per 1,000 images at this split's malignant prevalence (19.0%):
- 127 of 190 malignant lesions are caught.
- 249 of 810 benign lesions are sent to urgent review (2.0 benign per malignant caught).
- 63 malignant lesions are cleared with no follow-up.
