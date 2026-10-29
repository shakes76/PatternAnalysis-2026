# R5e — Frozen final evaluation protocol

R5e freezes evaluation decisions and rehearses post-prediction evaluation using
synthetic patients. It does not read MRI images or run the real final test.

## Frozen artifacts

- `r5_final_protocol.json`: nine patient-NLL checkpoints, data hashes, preprocessing,
  unified FP32 batch-4 inference, three temperatures, three supplemental thresholds,
  ensemble/referral/metric/bootstrap rules and 17 required source/dependency hashes.
- `r5_final_protocol.sha256`: raw file hash. The trusted version is the committed
  file; do not regenerate a changed protocol to bypass a failed check.
- `final_evaluation.py`: applies frozen parameters to nine prediction bundles.
- `run_r5_freeze.py`: verifies accepted validation evidence and creates a synthetic
  rehearsal. All simulated identifiers are prefixed `synthetic`; its metrics are
  software checks, not performance claims.

The primary application is the ConvNeXt-F three-seed ensemble. CNN-A/B are controls.
First average 20 slices per patient per seed; then average seed42/43/44 probabilities.
Apply the frozen positive temperature after averaging. Main threshold is 0.5;
supplemental thresholds were selected on validation for AD recall >=80%. They do
not guarantee 80% recall on the final test. Never refit on test patients.

Referral sorts by absolute raw ensemble probability distance from 0.5, then patient
ID lexicographically. Only IDs/probabilities enter selection. Refer ceil(rate*N),
with main rate10% and fixed supplements0/5/20%. Referral is not assumed correction.
Report full-cohort scores and retained errors/AD misses separately; zero denominators
are null. ECE uses 10 equal-width confidence bins; the last includes1.

Report nine individual models, three seed means/sample SDs, three raw/calibrated
ensembles, supplemental-threshold results and12 referral rows. Seven scalar metrics
receive stratified patient percentile95% intervals, 2000 draws, PCG64 seed42,
NC then AD, shared indices for all models and paired differences. These intervals
condition on fixed trained models/calibration/thresholds and class counts.
ECE and referral are descriptive; seed SD is a separate variability measure.
Raw NLL clips probability at1e-7; calibrated NLL uses stable scaled logit loss.

## Verify and rehearse locally

Run from this project folder, with the pinned requirements installed:

```bash
python run_r5_freeze.py --verify-protocol r5_final_protocol.json \
  --expected-sha256 96b492906327273f8379d2e532434f57fe1b77a99ebc8b9151918ec9bc521fa8 \
  --output /path/to/new/r5e_rehearsal
python -m unittest tests.test_final_evaluation tests.test_r5_freeze
```

Output must be new. The CLI checks protocol bytes/code, runs the synthetic pipeline
twice, compares all results, and publishes only on success. It emits a summary plus
standalone protocol and SHA file. Hash failure requires investigation; do not relax
checks. Real prediction bundles must carry the canonical JSON digest of the protocol
(`final_evaluation.digest`), which is distinct from the raw protocol-file SHA above.

## Next: R6a implementation, then cluster execution

The GPU prediction producer and Slurm wrapper are the next task, not supplied by
this phase. Before reading images they must verify this committed protocol, split
metadata and actual checkpoint bytes, enforce inference/preprocessing settings,
and save complete slice predictions plus identity/environment/command/log evidence.
The nine checkpoint identities here derive from accepted R4 records; physical
checkpoint files remain on the cluster and were not rehashed locally.

Do not reuse a validation-only runner by editing its split guard. Build a final-test
entry point that loads the frozen contract, performs inference once, then applies
`evaluate`. Test the entry point and failure handling with synthetic inputs first.
Return all nine slice tables, per-run manifests, protocol/hash, environment and
job logs, patient/ensemble tables, metrics/intervals/referral tables and report hashes.
A technical failure may be investigated and reproduced; no test-driven tuning.

AI assisted implementation, tests, synthetic audit and this documentation. Actual
model performance remains the previously accepted validation evidence until R6.
