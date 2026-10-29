# Frozen fixed-model held-out test protocol

**29 October update:** the student reports compulsory tutor check-off completed. The original numeric evidence below is retained as a historical record; [README](README.md) and [anatomical analysis](ANATOMICAL_LABEL_ANALYSIS.md) document the subsequent image-based prostate = ID 5 interpretation and direct baseline comparison.

Protocol `hipmri-heldout-numeric-v1`, dated 27 October 2026, is specified in
`heldout_protocol.json` before the supplied test split is evaluated. It fixes
both validation-selected models; it performs evaluation only.

| Setting | Frozen value |
|---|---|
| Runtime/model source | bc8a6a4688f50770f71de24a56dff8aad43204c7 |
| UNet2D selected epoch | 9 |
| ImprovedUNet2D selected epoch | 20 |
| Test coverage | All 540 slices; patients 40, 41, 42 |
| Evaluation batch / device | 1 / CUDA |
| Repeats / warmup | 3 full passes per model / 3 forwards per native shape |
| Execution order | Baseline then Improved, repeated three times |
| GPU allocation | One A100, one CPU, 30-minute limit, one Slurm job |
| Reporting | All numeric IDs 0-5; pooled Dice/IoU/counts, per-patient metrics and equal-patient means |

Checkpoint SHA256 values are fixed independently of test performance:

- UNet2D: `c27975f00f0357fac9a35bea3d765a56bb45621a92c36df71342890bfd92b1a5`
- ImprovedUNet2D: `64eef4f620355a38c0624ff21833e27c0e03728321f020146731ca0ab7513c0b`

No numeric ID is designated prostate or background. Anatomical attribution must
come from an independent documented source rather than from observed scores.
The proposed prostate-specific acceptance target therefore remains unverified.
Tutor feasibility check-off is also still pending. Three repeated passes reuse
the same weights and are not independent training replications.

`run_heldout.py` first audits both completed training/validation runs, frozen
runtime sources and checkpoint bytes. Test preflight checks all image/mask
filenames, byte sizes, expected counts and patient separation from both training
and validation. It saves the exact protocol, a filename/size manifest, source
hashes, checkpoint hashes, runner Git revision, node/job identity and commands.
A filename/size digest is not a content hash of every test image.

All six evaluation children use the existing frozen `evaluate.py` with
`--split test --max-samples 0 --device cuda --warmup-steps 3`. Source, weights
and filename/size records are checked around every child. The final audit checks
full coverage, native grids, per-slice/patient/case confusion sums, metrics,
repeated predictions and timing scopes. Partial or inconsistent results cannot
produce a completed comparison. Models are not retrained, tuned or selected
from test outcomes.

On 27 October 2026, all nine held-out-runner tests passed on Rangpur (4.002 s),
including both synthetic plot exports; `bash -n heldout_evaluate.sbatch` also
passed. The tests use temporary synthetic records and mock evaluation child
processes. They verify orchestration and checks, not actual test-set accuracy.
The protocol was committed before submission. Downloaded accounting later showed job 705243 COMPLETED / 0:0 before an attempted cancellation; its result files were not collected. At the user's request, an explicitly recorded local execution adaptation preserved the frozen weights, model/evaluator code, complete split and repeat order. The local evaluation completed and is reviewed in [TEST_RESULTS.md](TEST_RESULTS.md).

Historical A100 submission procedure (already performed; retain for audit):
commit the protocol and runner, export that exact project tree with `git archive`
and extract it into a new `$HOME/hipmri_heldout/FULL_RUNNER_SHA` directory.
Preserve the earlier training runtime snapshot and both run directories.
Export `HIPMRI_HELDOUT_COMMIT` as the full runner commit. From the new snapshot,
run shell syntax validation and a `--dry-run` first; dry-run reads no supplied
test data, starts no children and creates no experiment outputs. Then submit
`heldout_evaluate.sbatch` once. Its log is `heldout-evaluate-JOB_ID.log` and its
new output directory is `$HOME/hipmri_runs/heldout_v1_JOB_ID`.

A successful job must have Slurm COMPLETED / exit 0:0, completed provenance and
`Frozen-model full held-out test comparison: PASS`. A queued job with exit 0:0
or a partial set of output files does not establish success.

Output includes the six per-pass evaluation directories, `test_comparison.json`,
`test_provenance.json`, `test_data_manifest.json`, the exact protocol,
`TEST_RESULTS.md`, five LF CSVs, `heldout_metrics_comparison.png`, and
`heldout_patients_and_timing.png`. CSV/JSON preserve undefined overlap as empty
cells/null rather than zero. Timings distinguish synchronized model-only
forward passes from case processing including I/O and transfers. Both models
run in the same allocation; timing variation and the small test cohort remain
limitations. Real local result audits and visual review have since passed; the A100 result artifacts were not reviewed.

AI assistance covered protocol drafting, runner/test/script scaffolding and
reviewing the user-executed test results. The student remains responsible for
reviewing and explaining the code, methodology and final results.
