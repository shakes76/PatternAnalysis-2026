# Fixed revised evaluation: selected ConvNeXt A1

This is a revised evaluation on the **previously exposed** Version 1 test cohort,
not an independent new holdout. The original result and source/history are retained.
No accuracy or clinical-safety outcome is promised. No pretrained weights are used.

## Frozen inputs and decisions

- A1: `scratch_augmented`, seed 42, selected epoch 13. Baseline: regularized CNN,
  selected epoch 14. Checkpoint identities, strict tensor shapes and all manifests
  are checked by the corrected `selected_a1.load_selected_checkpoint` adapter.
- Input remains the checked one-channel 224x224 JPEG pipeline, no test augmentation.
- Patient probability is mean(scan means(slice P(AD))); diagnosis cutoff 0.5.
- Temperatures and original referral cutoffs are reused, not fitted on the test.
- Report all three policies: `reference`, `strict_cn` (additional calibrated
  P(AD)<=0.05 for existing automatic CN decisions), `no_auto_cn` (no automatic CN).
  AD flag logic is unchanged. No policy is chosen by its revised test score.
- Primary evaluation unit: patient. Scan and slice metrics are also reported.
  The course brief does not explicitly specify the unit for the 0.80 threshold.
  Observed threshold flags are not a guarantee of marks or clinical suitability.
- Inference: float32, batch 8, no workers, deterministic settings, no TF32/AMP.

## Interface and preservation

`predict.py revised-freeze --private-root PATH --device cuda` records a separate
`A1_REVISED_EVALUATION_PLAN.json` before inference. Review and commit the four new
or changed source/method/test files and this plan before evaluation.

`predict.py revised-evaluate --private-root PATH --confirm-reused-test` opens the
revised test under that exact plan. The evaluator never trains a model, fits a
calibrator or searches cutoffs. Do not use the results to tune or change selection.

`predict.py revised-publish --private-root PATH` backs up the current README and
appends the aggregate results and two statistical plots. It does not publish MRI
panels, identifiers, individual predictions or checkpoints.

Only a small dispatch addition is made to `predict.py`. Old command definitions
remain, but do NOT use them for A1. Old source-pinned Version 1 plans may require
checking out their recorded historical Git commit to reproduce their exact code.
Historical results, models, datasets, rules and pointers are not overwritten.
The original entry point is backed up privately by the installer.

## Recovery and checks

New results use a dedicated private folder `a1-revised-evaluation` and a dedicated
pointer. A cross-platform OS lock prevents concurrent evaluators; the lock is
released when a process ends or the computer restarts. Do not delete the lock.
Each completed inference batch is saved atomically with a plan/environment binding
and probability digest. On recovery, previously used image bytes are rechecked
against the original local audit. Rerun the SAME revised-evaluate command. At most
an unfinished batch normally repeats; ordinary reruns after completion reuse results.
An unreadable cache or mismatched evidence stops rather than silently starting over.
Power-loss/storage corruption cannot be guaranteed recoverable; preserve evidence.

The saved local image hash checks do not establish source-versus-Rangpur equality.
The frozen plan binds the corrected model adapter, policies, old metric utilities,
new evaluator and test sources; no historical source guards are disabled.

## Metrics and uncertainty

Report all-patient accuracy, macro-F1, AUROC, per-class precision/recall, confusion
matrices, raw/fixed-temperature NLL and Brier scores, confidence-error diagnostics
and all three policies. Wilson intervals are descriptive. A fixed-seed (2026),
5,000-replicate paired patient bootstrap describes the accuracy difference for the
fixed models on this cohort. It does not account for training-seed/site variability,
previous test exposure, model search or prior policy selection.

Report both false-negative denominators: AD auto-CN / all AD, and AD / auto-CN.
Referrals are not diagnoses. Clinical review workload assumes every AD flag also
requires review; no clinician outcome/accuracy is measured. No-auto-CN has zero
automated negative errors by construction, undefined negative predictive value,
and does not demonstrate perfect sensitivity or safety.

## Resources

New test-batch timing covers image hash verification, JPEG decode/preprocessing,
transfer, model forward and conversion of results. It excludes cache writes, model
loading, recovery verification and work lost within interrupted batches. GPU
memory is the PyTorch process allocator peak, not Windows total or training use.
The prior batch-1 forward benchmark for these exact checkpoints is retained with
its original validation measurement scope; it is not claimed as a new test timing.
Earlier training-resource reports remain the source for training costs.

## Failure analysis

Select up to five distinct patients: confident A1 FN, confident A1 FP, A1 error
corrected by CNN, low-confidence remaining A1 error, then remaining confident
errors and CNN errors if needed. Sort confidence first and private subject ID for
ties. Choose the scan nearest the patient probability and ordered slices at the
25/50/75% positions. These are representative input panels, not saliency maps or
causal explanations. A private worksheet records both models and all policy actions.
Its manual text is never overwritten on recovery. Fewer than five errors are
reported honestly, not manufactured. Generating it is not completing the analysis.

Complete observations, hypotheses, operational consequences and limitations
separately. Do not publish patient identities or MRI images without permission.
The automatic summary retains manual_review=false; do not manually edit it.

## Assistance and references

ChatGPT assisted with this integration, synthetic tests and documentation. The
student must review, understand, validate and disclose the actual assistance and
checks performed. Reused model/calibration references are recorded in modules.py
and confidence_analysis.py. This extension introduces no new learned model or
calibration method. Exact software/source/plan/checkpoint/manifest hashes are kept
in the aggregate outputs. No network requests or additional dependencies are used.
