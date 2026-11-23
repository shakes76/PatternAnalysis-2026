# Selected A1: revised Step 9, validation only

## Selection fixed before calibration

The main model is the existing `scratch_augmented` ConvNeXt-Tiny, seed 42,
epoch 13, trained without pretrained weights. The retained comparator is the
regularized CNN, epoch 14. Neither S1 nor S2 met the predeclared promotion screen.
No further training or threshold search for classification is performed here.
The stored labels remain NC=0 and AD=1; the class cutoff remains 0.5.

## What is checked

`selected_a1.py prepare` restores private pointers, verifies finalized manifests
and historical source hashes, reaggregates saved validation predictions, and
creates `A1_SELECTED_MODEL.json`. It reads all three manifests only for integrity;
it does not decode any image. The selection plan includes checkpoint hashes,
configurations, source and manifest hashes, and validation metrics without IDs or
absolute paths. It is a development selection record, NOT a new final-test plan.

Commit the three additions and this generated plan before `run`.

`run` loads each exact, locally produced checkpoint using `weights_only=True`
and strict state-dictionary loading. The A1 adapter checks the V2 format,
scratch initialization, selected epoch, labels, configuration, and manifests.
The baseline adapter checks its original Step 8 checkpoint format.

The script replays ALL validation images for both models at batch size 8,
no augmentation and deterministic float32 inference. Each slice probability
must agree with the saved value within absolute tolerance 2e-5; the predicted
class must be unchanged at slice, scan, and patient levels. A failure is reported,
not silently converted into a new evaluation result. Small numerical differences
and maximum deviations are recorded. Original, verified cached probabilities
are used for calibration, not a new checkpoint or a rounded probability list.

Patient aggregation remains the mean of scan means of slice P(AD). A1's expected
validation count is 85/102 correct, including 22/33 AD patients. This is selected
validation performance, not test accuracy.

## Calibration and referral

Reuse the existing `confidence_analysis.py` without edits: bounded temperature
scaling of clipped patient log-odds, T in [0.5,10], five subject-stratified
hash-defined folds, and five-bin reliability plots. The full-fit temperature and
rule are candidates for the later revised evaluation. Cross-fitted scores are
development diagnostics because the entire cohort already selected checkpoints
and influenced multiple experimental decisions. They are NOT nested-independent
estimates of the final selected model.

The previous finite referral grid is unchanged. Choose maximal empirical
coverage, at least max(20,ceil(0.25*N)) accepted, <=10% accepted error, and <=10%
of all AD patients automatically cleared as NC. Otherwise refer all. These are
project criteria, not medical safety standards. A referral is not a correct
diagnosis, and refer-all has no accepted accuracy or automation benefit.

A positive temperature cannot change class decisions at 0.5. It does not create
discriminative information or make the 0.80 test target satisfied. Temperature
and referral thresholds must be bound to the exact selected checkpoint, not
copied from the previous ConvNeXt. The CNN is unchanged, so the script verifies
its recomputed temperature and full-fit referral rule reproduce the previous ones.

## Output, interruption and scope

All results are under `COMP3710-private/a1-selected-confidence`, outside Git and
OneDrive. Completed model replay receipts are reused on a rerun; if power is lost
mid-replay, that model's validation pass repeats. No training work is repeated.
An OS lock prevents parallel analyses. Calibration/report generation is repeatable.
An identical completed run is checked and skipped. Changed source, model inputs,
plan or environment cause a stop rather than implicit replacement.

The ZIP is an allowlist of aggregate reports, the selection plan, operating rules
and three statistical plots. It excludes checkpoints, medical images, individual
predictions and private paths. Inspect aggregate output before sharing.

Historical `predict.py`, Version 1 results, final-test plan, and old pointers stay
unchanged. `load_selected_checkpoint` supplies the verified adapter to be used in
the next revised prediction/evaluation integration; this stage does not yet enable
A1 test inference through the old `predict.py`. Do not rerun the old evaluate command.

## Resources and limits

No packages are installed and no network/weight downloads occur. The real replay
uses the existing GPU environment. It records parameter count, validation-pass
time, allocator peaks and batch-1 forward latency (10 warmup, 50 measured passes).
GPU allocator peaks are not total device memory. Timing with a preloaded tensor
is not JPEG-to-patient latency. No source-to-Rangpur byte comparison is claimed.

The Version 1 test was previously exposed; any later reuse must explicitly report
that limitation and must not be described as a fresh untouched evaluation.
No clinical validity or course permission for a revised test protocol is implied.

## Authorship and references

ChatGPT assisted with implementation, tests and report drafting. The student must
review the code, perform and describe the actual local checks, and include the
required Artificial Intelligence Usage Disclosure. Temperature-scaling reference:
Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017,
https://arxiv.org/abs/1706.04599 . Network reference remains Liu et al.,
*A ConvNet for the 2020s*, CVPR 2022, as cited in `modules.py`.
