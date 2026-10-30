# Step 10: frozen final-test evaluation and manual failure review

## Why this stage follows Step 9

The supplied Step 9 result is not evidence of safe automation. CNN cross-fitted
accepted error was 4/27 (14.81%), exceeding the 10% development criterion, despite
an apparent full-fit error of 1/27 (3.70%). ConvNeXt's full-validation rule is
**refer all**. Its cross-fitted policies' pooled 3/22 error (13.64%) are different
fold-specific rules, NOT an operating threshold to substitute for the saved rule.
CNN cross-fitted Brier score improved slightly; ConvNeXt's worsened. Both had
four incorrect patients at raw confidence >=0.90. These development findings
must remain in the report; do not relax constraints to manufacture success.

Proceed with the existing two selected checkpoints and saved Step 9 rules rather
than another open-ended search. This is a research evaluation, not deployment.
The main model's 0.80 all-patient accuracy target remains unverified. A final
result below it must be reported as below target, with possible assessment
consequences; no software pass marker guarantees full marks.

## Frozen choices, before any test inference

- Preferred baseline: the CNN already chosen in Step 8 (currently regularized
  CNN, epoch 14). Main model: existing ConvNeXt-Tiny, epoch 11.
- Same finalized manifests, unchanged greyscale preprocessing, no augmentation.
- Raw slice class probabilities, averaged within scans, then equally over scans
  within a patient. Primary unit patient; secondary scan and slice metrics.
- Class cutoff remains 0.5 (>=0.5 predicts AD). No ROC-based cutoff search.
- Apply the saved temperature once to the clipped PATIENT aggregate log-odds.
  Do not apply it to slices first or fit another temperature on test labels.
- CNN T=1.613427788322983 with calibrated confidence cutoff 0.85. ConvNeXt
  T=1.7048542769320942 with **refer-all**. Actual values/hashes are read and
  cross-checked from the user's saved Step 9 artefacts, not silently recomputed.
- One inference pass per model, float32, batch 8, no workers, no AMP/TF32.
- Primary metrics: all-patient accuracy, per-class precision/recall/F1, macro-F1,
  AUROC and confusion matrices. Report descriptive Wilson 95% accuracy/recall
  intervals. Provide raw/calibrated NLL, Brier, five-bin ECE and uncertainty/error
  ranking. Apply only the frozen referral rules.
- Paired fixed-model accuracy difference (ConvNeXt minus CNN) with a predeclared
  5,000-resample, seed-2026 patient bootstrap interval. Exploratory, not a statement
  about seed/site variability. No architecture selection on final-test scores.
- No diagnostic performance is credited to referrals. When all are referred,
  coverage=0 and accepted accuracy/error are undefined, not 100% or zero error.
- Scope: one site-distribution dataset split and one seed. Clinical safety,
  clinical utility and deployment readiness are not established.

The freeze command reads cached validation evidence, metadata and old image
hashes only. It does not decode or evaluate test images. It creates the aggregate
FINAL_EVALUATION_PLAN.json in the project for Git and a private identical copy.
Commit the method, new code/tests and that plan BEFORE running evaluate. The
interpreter versions and source/checkpoint/manifest hashes are checked thereafter.
Existing model/training/data/calibration files are not modified. The test access
record binds subsequent invocations to this one plan.

## Inference recovery and integrity

Inference uses the original selected weights in eval/inference_mode; no optimiser
or backward pass is created. Each decoded test image's exact bytes must match the
pre-modelling local audit. This is not a claim of byte verification against the
Rangpur source, which remains unverified. Predictions are saved atomically after
each batch. Resuming on the same software/device/settings reuses completed batches
and rechecks their image bytes. Only the interrupted, uncommitted batch may be
recomputed; no completed prediction is selected by its test performance. A changed
GPU/software/frozen source causes a stop rather than mixing results silently.
Use the same evaluate command after an ordinary interruption; after power loss,
clear only a demonstrably stale process lock using recover-lock. Never run two
evaluators concurrently. Do not edit PRIVATE progress JSON or delete the test
access record. Failed checkpoints/data integrity require diagnosis, not a restart
with a new model. Completed evaluation calls do not repeat inference.

Resource records distinguish batch-forward time from image loading, hashing,
transfer and prediction collection. Result-write time, idle time and resume
verification are excluded. Earlier training-memory/runtime and batch-1 latency
measurements remain the main controlled resource comparison. Allocator peaks are
not total GPU usage. Inference recovery is not resumption of model TRAINING.

## Error cases: qualitative work remains mandatory

Select up to five DISTINCT patients deterministically after scoring: highest-
confidence ConvNeXt false negative; highest-confidence false positive; remaining
ConvNeXt error corrected by CNN; lowest-confidence remaining ConvNeXt error;
remaining highest-confidence errors; then CNN errors if too few ConvNeXt cases.
Keep the actual number if fewer exist. These are diagnostic examples, not an
unbiased estimate of error prevalence.

For each patient, choose the scan whose mean P(AD) is closest to the patient mean,
with scan-ID tie-break; show slices at ordered positions 25%,50%,75%. Panels show
processed inputs, NOT attention/saliency and NOT a causal explanation. Predictions
for both models and their fixed referral actions are provided in a private
worksheet. The student must add: observed properties, plausible explanations with
uncertainty, consequences of each error, comparison with baseline, and operational
safeguards. Never invent evidence of atrophy or disease mechanism from a few
slices. Generating templates is not completing the assessment's 3–5 case analyses.

## Privacy, report and AI disclosure

Only explicit aggregate reports, the frozen plan and two aggregate plots enter
the results ZIP. MRI panels, identities, predictions, checkpoints, locations,
audit inventory and tracebacks stay in COMP3710-private, outside Git/OneDrive.
Do not publish medical images without applicable authorisation. The final report
must include suitable permitted visual evidence and genuine failure analysis;
resolve sharing permissions before publishing examples.

ChatGPT assisted with this design, implementation, synthetic tests and report
scaffolding. Review and test the code and accurately describe the work personally
performed in Artificial Intelligence Usage Disclosure. Synthetic test passes are
not Windows/CUDA/ADNI results. Report negative clinical and accuracy results
honestly. References underlying inherited methods: Guo et al., On Calibration of
Modern Neural Networks, ICML 2017; Liu et al., A ConvNet for the 2020s, CVPR 2022;
the supplied assessment specification and the existing project method documents.
