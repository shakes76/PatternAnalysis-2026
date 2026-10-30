# Fixed brightness sensitivity experiment

Validation-only paired diagnostic; coefficients and threshold fixed before inference.

| Class | Factor | N | Accuracy | AD recall | NC recall | AUROC | Mean AD probability change | Flip fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1.0 | 280 | 0.7750 | 0.7639 | 0.7788 | 0.8509 | 0.0000 | 0.0000 |
| all | 0.9 | 280 | 0.7786 | 0.7222 | 0.7981 | 0.8484 | -0.0036 | 0.0893 |
| all | 1.1 | 280 | 0.7357 | 0.7500 | 0.7308 | 0.8405 | 0.0013 | 0.0607 |
| AD | 1.0 | 72 | 0.7639 | 0.7639 | NA | NA | 0.0000 | 0.0000 |
| AD | 0.9 | 72 | 0.7222 | 0.7222 | NA | NA | -0.0186 | 0.1250 |
| AD | 1.1 | 72 | 0.7500 | 0.7500 | NA | NA | 0.0013 | 0.0694 |
| NC | 1.0 | 208 | 0.7788 | NA | 0.7788 | NA | 0.0000 | 0.0000 |
| NC | 0.9 | 208 | 0.7981 | NA | 0.7981 | NA | 0.0016 | 0.0769 |
| NC | 1.1 | 208 | 0.7308 | NA | 0.7308 | NA | 0.0013 | 0.0577 |

## Limits

- This measures the fixed model response to +/-10% foreground scaling on a fixed validation subsample.
- Paired bootstrap resamples entire patients, retaining the same images across conditions.
- Class-only AUROC and recall for the absent class are undefined, not zero.
- Image scaling can alter contrast/anatomical visibility; saturation is explicitly reported.
- Zero sensitivity at these two factors does not exclude other intensity dependencies.
- Validation already influenced historical model/threshold selection; this is exploratory.
- Subsample metrics are not the full 2620-slice validation metrics and need not match 78.82%.
- Relative positions are not verified anatomical coordinates.
- No causal explanation of test accuracy, patient memorization, clinical robustness or preprocessing improvement follows.
- Do not choose the best factor, update the official threshold, or claim the highest accuracy as an improved model.
- local_paired_predictions.csv and the HMAC key are local-only; aggregate reports have no raw identities or paths.
