# R5c: AD-recall supplemental thresholds for patient ensembles

Use the accepted R5b calibrated validation probabilities, with all temperatures
and nine underlying best checkpoints unchanged. Main classification stays at 0.5.
This stage does not retrain, recalibrate, read images or evaluate final test.

```bash
.venv/bin/python run_r5_threshold.py \
  --source /path/to/final_report/adni_manifest_local/r5_temperature_validation \
  --output /path/to/final_report/adni_manifest_local/r5_ensemble_thresholds

.venv/bin/python -m unittest discover -s tests -p test_r5_threshold.py
.venv/bin/python -m unittest discover -s tests
```

Use a new private output directory outside Git; existing outputs are refused.
No cluster job is needed. Numerical analysis uses the standard library and
reuses the tested R3f threshold selector without changing historical R3f files.

## Frozen selection rule

- Each model family contains 136 validation patients: 44 AD, 92 NC.
- Thresholds are all unique calibrated patient probabilities plus zero.
- Predict AD when calibrated p_ad >= threshold; tied probabilities stay together.
- Require at least ceil(0.8 * 44) = 36 AD detections.
- Among eligible candidates maximize NC specificity (equivalently TN count).
- For equal specificity choose the highest candidate threshold.

The three thresholds are separately selected from three ensemble probability
tables, never averaged from earlier seed-specific thresholds. Store the exact
JSON/CSV float; rounded values in a report are for display only.

## Provenance and acceptance

The pinned R5b summary SHA binds its fitted-temperature file and three patient
CSVs. Require passed/non-boundary/converged fits, matching temperatures, original
raw seed means, calibrated p = sigmoid(logit(clip(raw p))/T), the same patient
identities and class counts, and validation-only rows. Verify main classifications
remain identical and rehash every input before writing outputs.

Outputs:

- `threshold_candidates.csv`: all candidates, confusion counts, metrics,
  eligibility and selected markers (137 candidates per family on the real data).
- `threshold_comparison.csv`: 0.5 and supplemental results, six rows total.
- `threshold_costs.csv`: fewer AD misses and additional NC false alarms.
- Three `{model}_patient_decisions.csv`: unchanged probabilities and both decisions.
- `threshold_details.json`: full selection evidence.
- `threshold_summary.json`: exact thresholds, temperatures, source/code/output
  hashes, read counts and selection configuration.

Costs are signed differences relative to 0.5, not assumptions that every selected
threshold is lower than 0.5. Unit tests include a selected threshold above 0.5.
No gain in sensitivity is assumed free: report specificity and accuracy too.

Independent acceptance counts every candidate again and verifies the maximum
feasible threshold via the 36th highest AD probability. Seven numerical output
files must reproduce exactly. No patient table goes into Git.

## Accepted development-validation results

| Ensemble | Threshold (display) | AD recall | NC specificity | Accuracy | FN: main → supplemental | FP: main → supplemental |
|---|---:|---:|---:|---:|---:|---:|
| CNN-A | 0.288932 | 81.82% | 66.30% | 71.32% | 23 → 8 | 12 → 31 |
| CNN-B | 0.333587 | 81.82% | 70.65% | 74.26% | 19 → 8 | 13 → 27 |
| ConvNeXt-F | 0.303211 | 81.82% | 77.17% | 78.68% | 16 → 8 | 9 → 21 |

These validation thresholds do not guarantee >=80% AD recall on final test.
Keep 0.5 as the main comparison; this is the prespecified sensitivity analysis.

Next R5d separately evaluates batch referral at 0/5/10/20%, with 10% primary.
It ranks original ensemble |p-0.5| ascending, patient ID as tie-breaker, without
using labels to choose referrals. Do not combine this with the supplemental
threshold to choose a more favorable reporting subset. Final test stays sealed
until R5e freezes the complete protocol.

AI assistance: Codex implemented, tested, independently audited the thresholds,
and updated documentation. Patient-level evidence remains private and outside Git.
