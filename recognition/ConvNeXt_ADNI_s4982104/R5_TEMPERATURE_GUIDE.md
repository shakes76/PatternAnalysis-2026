# R5b: patient-level temperature scaling

Fit one scalar temperature for each CNN-A/CNN-B/ConvNeXt-F equal-weight ensemble.
Use the accepted R5a validation tables only: 136 patients, 44 AD and 92 NC.
No training, model loading, image reads, threshold search or final test occurs.

## Local execution

Install the project requirements in the existing virtual environment if needed.
The numerical routines use the Python standard library; plots use the pinned
Matplotlib dependency. Training dependencies and settings are unchanged.

```bash
.venv/bin/python run_r5_temperature.py \
  --source /path/to/final_report/adni_manifest_local/r5_ensemble_validation \
  --output /path/to/final_report/adni_manifest_local/r5_temperature_validation

.venv/bin/python -m unittest discover -s tests -p test_r5_temperature.py
.venv/bin/python -m unittest discover -s tests
```

Output must be a new private directory outside Git. Existing evidence is never
overwritten. The R5a summary SHA is pinned in the script and binds the three
patient CSV hashes. Every fitting row must explicitly be `validation`.
Cross-model patient/scan/label identities, counts, seed means, finite probabilities
and 0.5 predictions are checked. Pure `scale(p,T)` is also usable later with a
frozen T; it never fits or reads labels.

## Fixed mathematical protocol

1. Let z = logit(clip(p, 1e-7, 1-1e-7)) for the three-seed patient probability p.
2. Fit T in [0.05,20] by deterministic golden-section minimization over log(T),
   tolerance 1e-10, at most 256 iterations. Patients have equal weight.
3. Use stable Bernoulli NLL (softplus of the signed z/T). No class weights,
   intercept, per-slice or per-seed calibrators are introduced.
4. Explicitly compare both endpoints and T=1. T=1 wins exact objective ties.
   Values within 1e-6 log-distance of a boundary, a flat objective (range <=1e-12
   across endpoints and baseline), failed convergence, or an interior
   inverse-temperature gradient >1e-6 in magnitude need review.
5. Save p_cal = sigmoid(z/T). Preserve original p and all seed columns. Reject
   any numerical change in the 0.5 patient classifications.

The convex NLL in inverse temperature makes the positive-temperature objective
unimodal. Boundary/nonconverged fits are still saved with `needs_review` and CLI
exit code 2. Do not automatically expand bounds or adopt these temperatures.

Primary reported NLL uses the same stable logit objective as fitting. The extra
`nll_probability_clipped_1e7` field retains the previous probability-clipped
reporting convention for comparison: it can differ at saturated probabilities.
No original R5a files or historical metrics are changed.

## Confidence calibration and output

Confidence is max(p,1-p); correctness means the 0.5 prediction matches the label.
ECE is sum over bins of (bin count / N) times abs(mean confidence - accuracy).
Use ten equal-width bins on [0,1], left-inclusive/right-exclusive except the last
bin includes 1. Empty bins have null values (blank in CSV), not zero accuracy.
This measures overall confidence calibration, not specifically AD sensitivity.

- Three patient CSVs preserve raw and calibrated probabilities/confidences.
- `temperature_fits.json` records baseline, endpoints, all objective evaluations,
  final interval, convergence, stationarity, boundary flags and review reasons.
- `calibration_metrics.csv` contains six rows: raw/calibrated for each model.
- `reliability_bins.csv` contains all 60 bins, including empty ones.
- `reliability_diagram.svg`, `.pdf`, `.png` show reliability and bin support.
- `temperature_summary.json` binds four inputs, three code files and nine output
  artifacts by SHA; records configuration, scope and read counts.

The plots include all patients. Lines connect occupied adjacent bins; empty bins
are missing. SVG/PDF text is editable; PNG is a 600-dpi preview, not a journal
TIFF submission. No intervals or significance tests are shown. The same patients
fit T and assess validation calibration, so improvement is a development result.

## Acceptance and next stage

Check analytic synthetic optima, extremes, unidentifiable/boundary/nonconverged
fits, split and source rejection, class preservation, ECE bins and complete
outputs. Independently solve the inverse-temperature derivative, reconstruct
probabilities and metrics, and verify hashes. Repeated fits and numerical output
files must reproduce. Figure exports need visual inspection in addition to tests.

Next is R5c: use each calibrated ensemble's validation probabilities to choose
its supplemental AD-recall >=80% threshold. Keep the main threshold at 0.5.
R5c/R5d/R5e must finish before final-test evaluation.

Method reference: Guo et al., *On Calibration of Modern Neural Networks*, ICML
2017, https://proceedings.mlr.press/v70/guo17a.html . Here the scalar calibrator
acts on the logit of an averaged patient probability rather than a single
network's original logits.

AI assistance: Codex implemented and tested fitting, provenance checks and plots,
independently recomputed results and updated the progress record. Private patient
predictions, detailed audits and figures remain in the local evidence directory.
