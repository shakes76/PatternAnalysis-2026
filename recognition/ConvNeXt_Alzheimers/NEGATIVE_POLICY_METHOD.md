# Fixed negative-side referral comparison

## Purpose and assessment link

Investigate false-negative consequences and the workload cost of more cautious
negative decisions. The COMP3710/7710 Version 2.5 brief, Section 2.1, asks whether
confidence identifies mistakes and whether ambiguous cases should be referred.
Its minimum 0.80 test-accuracy requirement remains separate. This comparison
cannot increase all-patient classification accuracy because neither the weights
nor the 0.5 class cutoff changes.

## Fixed scope

Retain ConvNeXt A1 (scratch_augmented, seed 42, epoch 13) and the regularized CNN
(epoch 14). Use their completed A1-confidence selection, temperatures, original
referral rules and saved validation probabilities. No training, model inference,
image opening, new temperature fitting, cutoff optimization or new test inference.
Existing manifests may be read by the reused integrity checks, including test
manifest metadata; no test predictions or test images are read.

## Three policies declared before this comparison

Let q be the existing temperature-scaled patient P(AD), and tau the saved
symmetric confidence cutoff (A1 0.75; CNN 0.85 in the current selection).
The binary classifier predicts AD for q >= 0.5 and NC/CN otherwise.

1. **reference**: use the current rule, accepting a research class label when
   max(q,1-q) >= tau; otherwise refer.
2. **strict_cn**: retain the exact original AD-flag branch. Retain an original
   automatic CN proposal ONLY when q <= 0.05. Otherwise refer it. This is a fixed
   illustration of stronger negative-side evidence, equivalent here to model
   confidence at least 0.95 for CN. It is NOT a verified 95% probability of
   correctness, not a clinically justified risk tolerance, and not specified by
   the assessment.
3. **no_auto_cn**: retain the exact original AD-flag branch and refer every CN
   proposal, even q=0. If the original policy is refer-all, all three variants
   refer everyone; none creates acceptance that the reference disallowed.

A1 actions (full-validation rule): reference accepts CN at q<=0.25 and flags AD
at q>=0.75; strict_cn accepts CN only at q<=0.05 and still flags at q>=0.75.
CNN reference thresholds are q<=0.15 and q>=0.85; strict_cn uses q<=0.05 and
q>=0.85. The implementation evaluates the original confidence expression,
then removes CN acceptance; this also avoids changing floating-point boundary
behaviour in the original policy.

The secondary variants may fall below the original minimum-coverage constraint;
that reduction in automation is an explicit measured trade-off, not hidden or
treated as a reason to loosen the rule.

The new 0.05 setting is fixed before computing this comparison, not chosen by
searching the current patients. It is a secondary exploratory analysis motivated
by previously observed errors, not an independent confirmatory study. Report all
three policies; do not select a winner by inspecting subsequent test results.

## Cross-fitting and uncertainty

Reuse the original deterministic five-fold allocation, fold temperatures and
fold-specific reference thresholds from the completed confidence analysis.
Reconstruct/verify held-fold probabilities and actions against private rows and
the original aggregate report. The same fixed 0.05 cap is applied to each fold.
No temperature or threshold is refitted in this extension. If a fold had no
feasible original policy, do not manufacture one.

These diagnostic folds were already used during model and policy development.
They do not create an independent model-selection estimate. Pooled results use
fold-specific rules, not the single full-validation rule on an unseen cohort.
Wilson 95% intervals are descriptive; they do not account for model/policy
selection, repeated validation use or cross-fold dependence.

## Required denominators and trade-offs

Report ordinary all-patient class metrics separately and unchanged. For each
research referral policy report:

- AD patients assigned automatic CN / ALL AD patients.
- AD patients among automatic CN proposals / ALL automatic CN proposals.
- CN/AD/referral counts separately by actual class; AD flags that are false
  positives; AD patients referred rather than diagnosed.
- Number of formerly automatic CN proposals now referred, by true class.
- Accepted-label coverage and error with explicit denominators; no accepted
  cases means error and accuracy are not estimable, never perfect.
- Workload if positive flags also require specialist assessment. Under that
  assumption no-auto-CN sends every patient to a clinician; it may prioritize
  review but does not eliminate review workload.

For identical scores, stricter CN acceptance can only reduce or preserve the
COUNT of AD automatic-CN errors, since its accepted-negative set is a subset.
The negative predictive value and error FRACTION among remaining accepted cases
need not improve. Refer-all/zero automatic CN does not demonstrate perfect AD
recall or clinical safety. No human-review accuracy is assumed or measured.

## Provenance, privacy and continuation

Add this extension without editing previous source files, results, selection
records or operating rules. Bind the new plan to current inputs/code, commit it,
then calculate the comparison. Keep patient-linked diagnostic rows private;
export only aggregate results and the fixed policy definitions.

This script does not update historical predict.py or authorize the revised test.
The next integration must reuse the already verified selected-A1 loader, bind the
frozen three-policy definitions, disclose Version 1 test exposure and report the
primary reference plus both secondary variants without test-driven selection.
Do not recommend autonomous clinical clearance on the present evidence.

## Assistance

ChatGPT assisted with this implementation, tests, plan and report draft. Review
and understand the code, run the local tests, check actual outputs and record
those activities accurately in Artificial Intelligence Usage Disclosure.
