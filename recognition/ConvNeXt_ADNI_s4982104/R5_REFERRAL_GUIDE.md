# R5d: prespecified batch referral simulation

At the main threshold 0.5, refer patients with the smallest raw ensemble
abs(p_ad - 0.5). Break exact ties by patient_id lexicographically. The selection
function accesses only patient_id and raw probability; labels are used afterward
to evaluate the selected and retained subsets. Temperature scaling supplies
displayed confidence, not a new uncertainty ranking. R5c thresholds are not used.

## Local execution

```bash
.venv/bin/python run_r5_referral.py \
  --ensemble-source /path/to/final_report/adni_manifest_local/r5_ensemble_validation \
  --temperature-source /path/to/final_report/adni_manifest_local/r5_temperature_validation \
  --output /path/to/final_report/adni_manifest_local/r5_referral_validation

.venv/bin/python -m unittest discover -s tests -p test_r5_referral.py
.venv/bin/python -m unittest discover -s tests
```

Use a new private output directory outside Git. No GPU, image access, retraining,
recalibration, threshold fitting or actual human review occurs. Existing outputs
are refused. Pinned R5a/R5b summaries and their CSV/fit hashes bind both source
sets; patient identities and raw probabilities must match exactly across them.
The loaders validate calibration, counts and validation-only rows.

## Fixed rules and denominators

Rates are 0/5/10/20%, with 10% primary. The batch count is ceil(rate * N), giving
0/7/14/28 of 136 patients. The primary actual rate is 14/136 = 10.29%, leaving
122/136 = 89.71% for automatic handling.

- Error capture = referred errors / all 136 patients' errors.
- AD miss capture = referred false negatives / all false negatives.
- Retained risk = retained errors / retained patient count.
- Retained AD recall = retained true positives / retained AD patients.
- Coverage = retained patient count / full patient count.

Full, referred and retained confusion counts and denominators are saved for every
rate. Counts conserve the full cohort. A zero denominator yields null (blank in
CSV), not zero performance. Full-cohort accuracy stays the primary classification
result. Referred errors remain observed model errors: no correction is assumed.

## Primary development-validation result (14 referred per family)

| Model | Errors in referred group / all errors | Error capture | Referred AD misses / all AD misses | Errors among 122 retained | AD misses among retained |
|---|---:|---:|---:|---:|---:|
| CNN-A | 6 / 35 | 17.14% | 2 / 23 | 29 (23.77%) | 21 |
| CNN-B | 6 / 32 | 18.75% | 4 / 19 | 26 (21.31%) | 15 |
| ConvNeXt-F | 6 / 25 | 24.00% | 2 / 16 | 19 (15.57%) | 14 |

Each referred group also contains eight correctly classified patients. ConvNeXt
referral catches 24% of errors but only 12.5% of AD misses. Thus the rule captures
some errors while most AD misses stay with automatic handling. This is not proof
of safe deployment or of human correction. No statistical superiority is claimed.

At the prespecified 20% supplement (28 referred), remaining errors are 22/21/12
and remaining AD misses 17/12/9 for CNN-A/CNN-B/ConvNeXt-F. These do not replace
the primary 10% analysis or trigger an unplanned referral-rate search.

## Artifacts and acceptance

- `referral_metrics.csv`: 12 model/rate rows, full/referred/retained metrics.
- `primary_10_percent.csv`: three primary rows with every denominator.
- Three `{model}_referral_flags.csv` tables preserve probabilities and contain
  ranks, confidence, observed errors and four referral flags for all patients.
- `referral_analysis.svg/.pdf/.png`: risk–coverage and remaining AD misses;
  four prespecified points only, lines guide the eye. Rings mark 10% primary.
- `referral_summary.json`: protocol, source/code/output hashes and zero read counts.

Tests cover label-inaccessible selection, counts, ties, reordering, no mutation,
conservation, invalid inputs, empty denominators, tampering and the full pipeline.
Independent acceptance reranks from original R5a probabilities and recomputes all
12 operating points, 408 patient ranks and 1,632 flags. Five numerical CSV files
and metric records must reproduce exactly. Plots receive visual inspection.

Next R5e freezes the nine checkpoint hashes, preprocessing/inference settings,
three ensembles, three temperatures, three supplemental thresholds, main 0.5
rule, all referral rules and evaluation/bootstrap configuration. Final test is
still sealed until the protocol and synthetic end-to-end checks are complete.

AI assistance: Codex implemented/tests/audited this simulation and updated the
local progress record. Private patient tables and figures stay outside Git.
