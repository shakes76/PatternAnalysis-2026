# R5a: validation-only equal-weight ensembles

Run locally with Python 3.11 (standard library only). No GPU, image decoding,
checkpoint loading, temperature fitting or threshold search is performed.

```bash
python run_r5_ensemble.py \
  --source /path/to/final_report/adni_manifest_local/r4_efficiency_v2/r4_efficiency_v2_706895/run \
  --split-file /path/to/final_report/adni_manifest_local/split_seed42_60_20_20/split_samples.csv \
  --output /path/to/final_report/adni_manifest_local/r5_ensemble_validation
```

The output directory must not already exist. For a verification rerun, use a
new directory; do not overwrite accepted evidence.

## Inputs and acceptance

The accepted job 706895 summary SHA and frozen nine-best manifest SHA are pinned
in the script. They anchor the hashes of the nine run summaries, nine unified
patient CSVs, nine unified slice CSVs, and the frozen split CSV. The script checks
model/seed/epoch/checkpoint identity, passed R4 gates, validation-only inference,
and FP32/batch4/TF32-off settings. Historical predictions are not substituted.

Every source must have the same 136 validation patients (44 AD, 92 NC), one scan
and exactly 20 frozen slices per patient. Paths in the split CSV are treated as
identifiers, never opened as images. Each patient probability is checked against
the slice mean within 1e-12, an aggregation check separate from R4 tolerances.
Patient IDs, not CSV row order, join seed 42/43/44. Their probabilities receive
weights 1/3 each; a probability >= 0.5 predicts AD. Inputs are rehashed before
outputs are written. Malformed identities, probabilities, metadata or hashes
fail before output creation.

## Outputs (private; do not commit patient CSVs)

- Three `{model}_patient_predictions.csv` tables, including all three input
  probabilities and the ensemble probability.
- `ensemble_metrics.csv`: patient accuracy, AD/NC recall, TN/FP/FN/TP, macro F1,
  AUROC, NLL and Brier at 0.5.
- `nine_unified_metrics.csv` and `seed_metrics_mean_sd.csv`: individual unified
  predictions and seed stability; these are distinct from ensemble metrics.
- `ensemble_summary.json`: source identities, 30 input hashes, six output hashes,
  code hash, configuration and zero image reads. NLL uses clipping at 1e-7.

Unit and synthetic full-pipeline tests:

```bash
python -m unittest discover -s tests -p test_r5_ensemble.py
python -m unittest discover -s tests
```

Independent acceptance should reconstruct each ensemble from the 60 slice
probabilities per patient across seeds, use a separate metric implementation,
and verify all input/output hashes. These are development-validation results,
not final-test evidence. No accuracy target is an engineering acceptance gate.
Next: R5b fits one positive patient-level temperature per family on validation
probabilities; final test remains sealed until R5e protocol freeze.

AI assistance: Codex implemented the aggregation/audit script, tests and guide,
and cross-checked local evidence. Patient-level evidence stays outside Git.
