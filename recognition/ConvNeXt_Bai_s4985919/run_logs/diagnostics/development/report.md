# Fixed-model development diagnostics

Threshold: 0.67. No test inference, training or threshold search.

| Split | Slice accuracy | AD recall | NC recall | AUROC |
|---|---:|---:|---:|---:|
| train | 0.8708 | 0.8309 | 0.9025 | 0.9408 |
| validation | 0.7882 | 0.7888 | 0.7877 | 0.8628 |

## Interpretation limits

- Training scores measure fitting to training patients, not generalization.
- Validation was already used for historical checkpoint/threshold selection.
- Position is within-scan numeric rank 0-19, NOT anatomical coordinates. Original indices remain in the local cache.
- Correlations are exploratory associations, not evidence of causal brightness reliance.
- Within-scan centering does not remove slice-position/anatomy confounding.
- Scan counts describe visits available in this dataset, not lifetime follow-up.
- Bootstrap resamples whole patients, without class stratification; undefined replicates are reported.
- Patient-equal slice accuracy is not patient probability-aggregation accuracy.
- Foreground thresholds 0/5/10 are descriptive masks, not brain segmentation.
- Raw slice cache and local_patient_metrics.csv are pseudonymous local research data; do not publish them.
- Legacy weights have no cryptographic link to historical config.
- No test-cohort conclusions or new operating threshold can be derived here.
