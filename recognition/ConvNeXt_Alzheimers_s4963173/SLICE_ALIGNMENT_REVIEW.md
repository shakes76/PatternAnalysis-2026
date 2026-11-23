# Slice-accuracy alignment and bounded retained-checkpoint review

29 October 2026. **Disposition: NO_SUPPORTED_QUICK_GAIN_PROCEED_TO_CLOSEOUT. Retain A1, scratch_augmented, seed 42, epoch 13.**

## Authority, chronology and decision

The supplied COMP3710 Version 2.5 specification requires minimum 0.80 test accuracy for the AD/CN ConvNeXt project (§2.1). The **user-supplied Ed clarification** establishes the assessed unit as an individual 2D test image/slice. No author, date, permalink, waiver or marking penalty is asserted for that clarification. Patient-independent splitting remains required (§2.1); scoring slices does not permit patient overlap. Explicit scan/patient aggregation remains mathematically well-defined and useful engineering analysis, but does not replace assessed image accuracy.

Original development selected checkpoints by patient metrics: Version 1 patient macro-F1 then lower patient NLL; V2 and later patient accuracy, then patient macro-F1, then lower ordinary patient NLL, earliest exact tie. Those were project choices, not assessment requirements. The original plans, selected epochs, promotion decisions and scientific outputs remain unchanged. This is a new retrospective reporting review, not a rewritten experiment plan.

The resource-prioritisation screen proposed for this review is at least 3,597/4,660 correct validation slices (five percentage points over A1), slice macro-F1 at least 0.7130464976654525 and AD slice recall at least 0.6179611650485437, plus exact usable weights and a valid standalone slice output. It is neither a course rule nor a significance threshold. No retained selected ConvNeXt checkpoint even exceeds A1's 3,364 correct; none passes the count gate. No logged maximum with available slice-accuracy history reaches it either. No candidate verification command is justified; no broad inference sweep or new training is proposed.

## Primary recorded test result (aggregate summaries only)

| Model | Test slices correct / N | Test slice accuracy | AD precision / recall | CN precision / recall | Slice macro-F1 | Slice AUROC | Raw slice NLL | Raw slice Brier | TN / FP / FN / TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 (epoch 13) | 3157/4640 | 68.04% | 78.72% / 54.43% | 61.86% / 83.39% | 0.676945 | 0.779813 | 1.928413 | not recorded | 1818 / 362 / 1121 / 1339 |
| Regularized CNN (epoch 14) | 3193/4640 | 68.81% | 76.81% / 58.98% | 63.32% / 79.91% | 0.686915 | 0.776140 | 2.019384 | not recorded | 1742 / 438 / 1009 / 1451 |


A1 needs 3,712/4,640 correct for 80%; its recorded 3,157 is 555 short. This is count arithmetic, not a request to identify or target errors. A1 has 1,121 AD slice false negatives and 362 CN slice false positives, versus CNN 1,009 and 438. A1 reduces CN false positives by 76 but loses 112 AD detections, for 36 fewer correct slices (-0.775862 percentage points). Its slightly higher AUROC and lower raw NLL do not meet the accuracy target. CNN remains the required comparator, not a replacement ConvNeXt candidate. No candidate decision used these test comparisons.

The source is the already published revised aggregate summary, not newly opened individual test predictions. Precision/recall/F1 and correct counts were checked against its integer matrices; AUROC/NLL are taken from its saved probability-derived aggregates. Test slice Brier is not present in that summary and is unavailable here. No test predictions were opened to reconstruct it. No independent-slice confidence intervals are calculated: 4,640 correlated images come from 102 patients, not 4,640 independent subjects. Prior test exposure remains disclosed.

## A. Historically selected completed checkpoints

All rows below use exactly the approved 4,660 validation slices (2,600 CN / 2,060 AD), 233 scans and 102 patients. Each row is its **original patient-selected checkpoint**, not a retrospectively selected slice epoch. All sixteen completed runs finish under their recorded patience rule; incomplete work is excluded below. All selected checkpoint files exist and their stored epoch matches the original selection. Full-precision metrics, class counts and checkpoint hashes are in [aggregate JSON](slice_alignment_metrics.json) and [CSV](slice_alignment_metrics.csv).

| Run / seed | Selected / completed epoch | Validation slices correct / N | Validation slice accuracy | AD precision / recall | CN precision / recall | Slice macro-F1 | Slice AUROC | Slice NLL | Slice Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial CNN / 42 | 7 / 12 | 3220 / 4660 | 69.10% | 69.92% / 52.82% | 68.69% / 82.00% | 0.674658 | 0.778987 | 1.081850 | 0.253251 |
| CNN / 42 | 14 / 22 | 3184 / 4660 | 68.33% | 66.67% / 56.70% | 69.33% / 77.54% | 0.672414 | 0.744020 | 2.023849 | 0.287649 |
| V1 ConvNeXt / 42 | 11 / 19 | 3076 / 4660 | 66.01% | 62.93% / 56.21% | 68.01% / 73.77% | 0.650798 | 0.720507 | 2.245499 | 0.310535 |
| A1 / 42 | 13 / 21 | 3364 / 4660 | 72.19% | 71.44% / 61.80% | 72.65% / 80.42% | 0.713046 | 0.788117 | 1.559219 | 0.248017 |
| A2 / 42 | 12 / 20 | 3125 / 4660 | 67.06% | 65.16% / 54.76% | 68.18% / 76.81% | 0.658733 | 0.727529 | 1.555233 | 0.284952 |
| A1 seed 43 / 43 | 8 / 16 | 3232 / 4660 | 69.36% | 67.27% / 59.76% | 70.71% / 76.96% | 0.684961 | 0.757438 | 1.283005 | 0.262683 |
| A1 seed 44 / 44 | 7 / 15 | 3277 / 4660 | 70.32% | 67.40% / 63.64% | 72.41% / 75.62% | 0.697237 | 0.773052 | 0.982948 | 0.238141 |
| S1 / 42 | 9 / 17 | 3195 / 4660 | 68.56% | 67.15% / 56.55% | 69.40% / 78.08% | 0.674404 | 0.737702 | 0.770366 | 0.247006 |
| S2 / 42 | 9 / 17 | 3310 / 4660 | 71.03% | 68.32% / 64.27% | 72.96% / 76.38% | 0.704334 | 0.758241 | 0.722501 | 0.226722 |
| Schedule seed 42 / 42 | 7 / 15 | 3343 / 4660 | 71.74% | 72.61% / 57.91% | 71.26% / 82.69% | 0.704938 | 0.784579 | 1.159064 | 0.240361 |
| Schedule seed 43 / 43 | 5 / 13 | 3237 / 4660 | 69.46% | 67.41% / 59.85% | 70.79% / 77.08% | 0.686041 | 0.765954 | 0.748905 | 0.224151 |
| Schedule seed 44 / 44 | 9 / 17 | 3330 / 4660 | 71.46% | 67.27% / 69.03% | 74.94% / 73.38% | 0.711454 | 0.794759 | 1.366919 | 0.245013 |
| G0 / 42 | 16 / 24 | 3231 / 4660 | 69.33% | 65.89% / 63.50% | 71.89% / 73.96% | 0.687912 | 0.746815 | 0.708483 | 0.226408 |
| G1 / 42 | 7 / 15 | 2600 / 4660 | 55.79% | 0.00% / 0.00% | 55.79% / 100.00% | 0.358127 | 0.571028 | 0.684136 | 0.245525 |
| G2 / 42 | 20 / 28 | 3076 / 4660 | 66.01% | 59.02% / 75.58% | 75.12% / 58.42% | 0.660063 | 0.697176 | 1.618217 | 0.327343 |
| G3 / 42 | 14 / 22 | 2771 / 4660 | 59.46% | 52.41% / 90.24% | 81.94% / 35.08% | 0.577173 | 0.642805 | 1.079570 | 0.339880 |


| Run | Original selected checkpoint / current retained epochs | Original selection | Native output / limitation |
| --- | --- | --- | --- |
| Initial CNN | selected 7; PRIVATE_best_baseline.pt e7, PRIVATE_last_baseline.pt e12 | highest validation patient macro-F1; then lower patient log loss; then earliest | standalone two-logit slice model |
| CNN | selected 14; PRIVATE_best_model.pt e14, PRIVATE_last_model.pt e22 | validation patient macro-F1; lower patient log loss; earliest exact tie | standalone two-logit slice model |
| V1 ConvNeXt | selected 11; PRIVATE_best_model.pt e11, PRIVATE_last_model.pt e19 | validation patient macro-F1; lower patient log loss; earliest exact tie | standalone two-logit slice model |
| A1 | selected 13; PRIVATE_best_model.pt e13, PRIVATE_previous.pt e20, PRIVATE_resume.pt e21 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| A2 | selected 12; PRIVATE_best_model.pt e12, PRIVATE_previous.pt e19, PRIVATE_resume.pt e20 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| A1 seed 43 | selected 8; PRIVATE_best_model.pt e8, PRIVATE_previous.pt e15, PRIVATE_resume.pt e16 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| A1 seed 44 | selected 7; PRIVATE_best_model.pt e7, PRIVATE_previous.pt e14, PRIVATE_resume.pt e15 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| S1 | selected 9; PRIVATE_best_model.pt e9, PRIVATE_previous.pt e16, PRIVATE_resume.pt e17 | validation patient accuracy, then macro-F1, then lower ordinary log loss; earliest exact tie | standalone two-logit slice model |
| S2 | selected 9; PRIVATE_best_model.pt e9, PRIVATE_previous.pt e16, PRIVATE_resume.pt e17 | validation patient accuracy, then macro-F1, then lower ordinary log loss; earliest exact tie | standalone two-logit slice model |
| Schedule seed 42 | selected 7; PRIVATE_best_model.pt e7, PRIVATE_previous.pt e14, PRIVATE_resume.pt e15 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| Schedule seed 43 | selected 5; PRIVATE_best_model.pt e5, PRIVATE_previous.pt e12, PRIVATE_resume.pt e13 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| Schedule seed 44 | selected 9; PRIVATE_best_model.pt e9, PRIVATE_previous.pt e16, PRIVATE_resume.pt e17 | validation patient accuracy; then macro-F1; then lower log loss; earliest exact tie | standalone two-logit slice model |
| G0 | selected 16; PRIVATE_previous.pt e23, PRIVATE_resume.pt e24, PRIVATE_selected.pt e16 | patient accuracy, macro-F1, lower NLL; earliest tie | slice CE, complete scan batching; raw slice head valid; scan mean probabilities |
| G1 | selected 7; PRIVATE_previous.pt e14, PRIVATE_resume.pt e15, PRIVATE_selected.pt e7 | patient accuracy, macro-F1, lower NLL; earliest tie | scan-supervised raw slice-head diagnostic; native mean probabilities |
| G2 | selected 20; PRIVATE_previous.pt e27, PRIVATE_resume.pt e28, PRIVATE_selected.pt e20 | patient accuracy, macro-F1, lower NLL; earliest tie | scan-supervised raw slice-head diagnostic; native softmax mean logits |
| G3 | selected 14; PRIVATE_previous.pt e21, PRIVATE_resume.pt e22, PRIVATE_selected.pt e14 | patient accuracy, macro-F1, lower NLL; earliest tie | attention scan-bag output; slice head projection diagnostic only |


G0 uses a slice loss and original standalone head despite grouped batching. G1/G2 use scan losses with raw per-slice head scores saved as diagnostics; native G1 means probabilities, native G2 softmaxes mean logits. G3's native attention-bag output is distinct from its per-slice head projection. No pooled probability was copied across images. The G3 diagnostic is not a verified standalone slice candidate. A new adapter/single-image execution path was neither needed nor built for these below-screen checkpoints. All three G012 variants and G3 failed their original declared patient-based screens; those findings are retained, rather than relabelled as prospective slice-screen outcomes.

A1 is the highest observed slice-validation accuracy among this set of historically selected checkpoints, including the CNN comparators. That limited fact does **not** establish that it is the best possible slice classifier or the best epoch from histories that did not save slice accuracy. S2 and some schedule variants improve raw slice NLL/Brier, and other runs trade higher AD recall for lower CN recall; these are reported trade-offs, not a substantial image-accuracy gain. In particular lower bag training loss is not evidence of better slice classification.

### Substantial-gain screen of historically selected alternatives

| ConvNeXt run | Selected correct count >=3597 | Slice macro-F1 preserved | AD slice recall preserved | Substantial gain supported |
|---|---|---|---|---|
| V1 ConvNeXt | FAIL (3076) | FAIL | FAIL | NO |
| A2 | FAIL (3125) | FAIL | FAIL | NO |
| A1 seed 43 | FAIL (3232) | FAIL | FAIL | NO |
| A1 seed 44 | FAIL (3277) | FAIL | PASS | NO |
| S1 | FAIL (3195) | FAIL | FAIL | NO |
| S2 | FAIL (3310) | FAIL | PASS | NO |
| Schedule seed 42 | FAIL (3343) | FAIL | FAIL | NO |
| Schedule seed 43 | FAIL (3237) | FAIL | FAIL | NO |
| Schedule seed 44 | FAIL (3330) | FAIL | PASS | NO |
| G0 | FAIL (3231) | FAIL | PASS | NO |
| G1 | FAIL (2600) | FAIL | FAIL | NO |
| G2 | FAIL (3076) | FAIL | PASS | NO |
| G3 | FAIL (2771) | FAIL | PASS | NO |

All alternatives fail the correct-count condition. G1/G2/G3 values remain raw slice-head diagnostics and cannot turn a bag output into a verified standalone image path. This retrospective prioritisation check does not rewrite their original patient screens.

## B. Retrospective logged slice opportunities, kept separate

| Run | Best logged slice-validation epoch / correct | Slice accuracy | Ordinary slice-validation NLL | Exact weights / selection | Other slice metrics at that epoch |
|---|---|---|---|---|---|
| Initial CNN | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| CNN | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| V1 ConvNeXt | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| A1 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| A2 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| A1 seed 43 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| A1 seed 44 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| S1 | 14: 3275/4660 | 70.2790% | 0.789551113 | exact epoch weights unavailable; not originally selected | precision/recall/F1/AUROC/Brier unavailable; NLL is recorded |
| S2 | 13: 3348/4660 | 71.8455% | 0.730976731 | exact epoch weights unavailable; not originally selected | precision/recall/F1/AUROC/Brier unavailable; NLL is recorded |
| Schedule seed 42 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| Schedule seed 43 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| Schedule seed 44 | Unrecorded maximum | unavailable per epoch | loss history exists; cannot locate a slice-accuracy maximum | Original selected and current/previous states retained | Only selected checkpoint has complete cached slice metrics |
| G0 | 14: 3273/4660 | 70.2361% | 0.668274429 | exact epoch weights unavailable; not originally selected | precision/recall/F1/AUROC/Brier unavailable; NLL is recorded |
| G1 | 14: 2633/4660 | 56.5021% | 0.974921493 | PRIVATE_previous.pt; not originally selected | precision/recall/F1/AUROC/Brier unavailable; NLL is recorded |
| G2 | 20: 3076/4660 | 66.0086% | 1.618216888 | PRIVATE_selected.pt; originally selected | complete cached selected metrics |
| G3 | 16: 2825/4660 | 60.6223% | 1.089040419 | exact epoch weights unavailable; not originally selected | precision/recall/F1/AUROC/Brier unavailable; NLL is recorded |


For S1/S2/G0/G3, the logged maximum's exact weights are no longer in their retained selected/current/previous checkpoint sets. All of those logged accuracies are nevertheless below A1. G1 **does** retain epoch 14 as the current model in PRIVATE_previous.pt (distinct from its embedded original best epoch 7), but it achieves only 2,633/4,660; ordinary slice NLL is recorded, but the full slice metric set for that epoch is not recorded. G2's maximum is its original selected epoch, with metrics and exact selected weights. Stable retrospective tie handling here retains the earliest maximum accuracy; no tied accuracy winner needs an invented F1/NLL. No new selection is made.

The A1/V2/stability/schedule and V1/CNN histories do not record per-epoch slice-validation accuracy. Fields named train_slice_accuracy are training measurements; validation_log_loss / validation_slice_loss are losses. Their final and previous model tensors cannot be assigned missing slice metrics by inference from those columns, nor can missing older weights be recreated from a later state. Embedded best states remain the original patient-selected epoch. No log-loss minimum is renamed a best-slice epoch. Metrics missing for unselected current/previous states remain unavailable without model execution; that absence does not justify a last-minute sweep.

## Verification scope and limitations

One finite opportunity inventory was completed within the 30-minute active-work cap; documentation followed the decision. CPU-only arithmetic recomputed all sixteen selected validation slice metric sets from saved scores and checked exact manifest membership/labels and original selection keys across complete histories. All present recorded accuracy/F1/AUROC/NLL/Brier fields agreed within 1e-12. Trusted saved checkpoint dictionaries were loaded on CPU only to inspect epochs/metadata and bind actual retained files by SHA-256; no model was constructed or called. The 94 input files in the local review ledger were hashed before and after and stayed unchanged. The calculation script blocks PIL image opening. Checkpoint/source hashes support identity, not proof of chronological execution or exact retraining reproducibility. No new replay was run; previous cached replay checks and clean-interface verification remain their historical observations, with existing 2e-5 probability tolerance unchanged.

Source anchors: train_experiment.py:312–332 (Version 1 selection/history, no per-epoch validation slice accuracy); train_convnext_v2.py:174–176 and 380–395 (V2 selection/history); train_smoothing.py:349–364 (separate training and validation slice accuracy); g012/core.py:57–74 (loss/native pooling); g3/model.py:39–61 (native weighted logits versus returned slice projections). Checkpoint inventory reflects actual saved dictionaries, not filenames alone. Source at frozen G3 revision daf136cee3159f7308c11ce0c3d438448664d90c and completed submission revision 585152470540e98fca97d0b96e876a9cad8014cb was inspected. The original submission interfaces and frozen historical runtime are unchanged in this documentation branch.

Excluded: the interrupted first Version 1 ConvNeXt attempt (summary still RUNNING, experiment_completed=false, ten recorded epochs); smoke/preview/diagnostic outputs; original overlapping folder partitions; unrelated models. No U3 was run: G3 used REUSE_G2 and no U3 run directory exists. The retained completed Version 1 run is distinct from the interrupted attempt. No missing file was repeatedly searched for or assumed to exist elsewhere.

Confidence temperatures and referral rules were fitted/evaluated at the **patient** level. They are not calibrated slice policies and are not applied to raw slice scores. Raw validation NLL/Brier are supplied here; calibrated primary-unit uncertainty/referral evaluation remains a gap. The five previously reviewed patient cases remain documentation evidence, with labels distinguishing aggregate mistakes from displayed slice outputs. No new case/panel was opened, no MRI published and no clinical cause inferred.

The reused validation cohort, single G012/G3 seed, prior test exposure and unmet slice-test target limit all conclusions. No performance guarantee, exhaustive search or clinical readiness is claimed. No real-data training, fine-tuning, smoke, inference, replay, test evaluation, calibration, threshold search or ensemble was executed in this closeout.

## Unsent Ed question

> Thank you for clarifying that the 0.80 requirement is slice-level test accuracy. Could you clarify how falling below it affects the marking: is it an absolute hurdle or a condition for particular implementation/difficulty marks, or is performance assessed with partial credit alongside the other criteria? Our retained model records 68.04% slice-level test accuracy, and we will report this explicitly. We have retained the baseline, patient-independent splits, uncertainty investigation and negative experiments.

This draft was not sent. Report corrections do not await a reply; no waiver or mark deduction is assumed.
