## Documentation and Ed clarification - 29 October 2026

The student supplied Ed screenshots containing two teaching-team replies. Gayan Kulatilake first asks whether the GNN actually predicts confidence, then replies: "This seems fine. Though the use of T is a bit unusual. Where did you get that from?" The screenshot establishes a methodological exchange, not the original posting date, formal PR approval or an assessed mark.

The final report now identifies logits as the GNN output and maximum temperature-scaled softmax as a derived confidence proxy. It cites Guo et al. (ICML 2017), Section 4.2, Equation 9; temperature fitting minimises calibration cross-entropy with fixed network weights. Temperature T and the acceptance threshold have different roles. Positive scalar scaling preserves class labels, can sharpen when T<1, and cannot eliminate confident mistakes. A source-citing English reply is prepared; posting and any subsequent answer are not confirmed.

At 18:06 Brisbane time on 29 October, PR #9 remained open with zero formal reviews, inline comments and discussion comments. This documentation revision changes no experiment, policy, score, source algorithm or Colab. Earlier dated records below retain the status known at their own dates, including statements that no Ed reply was then available. Turnitin submission and future formal review responses remain separate user actions.

## Earlier dated records

## Latest confidence investigation - 26 October

A fixed six-model confidence study and five-case edge-isolation diagnosis are now complete; [full record](consensus/README.md). Mean routing reduces accepted errors 32 to 29 and missed-politician errors 7 to 5 against prespecified seed42, with lower coverage and 6.23x measured CPU latency. Unanimity adds no final-case benefit; politician ECE worsens and a shared >=.99 politician error remains. This is post-hoc, with 26 targeted tests and two negative integration checks; no external validation or human correction is claimed. Original results and Colab remain unchanged.

# Latest exact-search and weighted-loss update - 26 October 2026

Exact score-boundary search restores nonzero politician prediction coverage without changing unweighted model labels. Three weighted GCNs increase mean politician recall to 92.34% while precision decreases to 88.22%; accepted errors are not uniformly reduced. The post-hoc status, seed-43 shortfall, random-quality-review workload and unresolved external validation are recorded in [refinement/README.md](refinement/README.md). Eighteen checks and both 2254-case reloads pass. Original studies and Colab remain preserved.

## Earlier records

# Latest robustness and engineering update - 26 October 2026

Six new CPU runs on a precommitted five-role within-graph split separate temperature and policy fitting. All policies precede final-role evaluation. Class-aware thresholding is implemented and compared, including its politician auto-coverage failure, 7/5/5 wrongly accepted true politicians, and 32/33/29 total accepted errors. [Full record](five_role/README.md). Twelve targeted checks pass; the first 97% runner also gains dataset/identity/temperature/partial-output guards.

The student confirms no extra data or Ed reply is available. External validation, universal 97% reliability, zero errors and human-review effectiveness remain unestablished. Colab stays unchanged. Original benchmark and prior post-hoc experiment remain historical records.

## Earlier consultation and first policy update

# Consultation and policy experiment record

## Current update - 26 October 2026

The completed feasibility check-off remains student-confirmed. The original 90% threshold-zero outcome and all training evidence are retained. A separate frozen 97% empirical calibration acceptance objective has now been implemented without retraining or changing temperatures. It yields thresholds 0.73/0.76/0.74. Mean referral on the historically inspected original test is 10.74%, and accepted-case accuracy is 97.07%; full-population model accuracy remains 93.84%. Seed 44 reaches 96.99%, below target, and is not retuned.

All policies were fitted on calibration labels and frozen before this follow-up calculated reused-test results. Choosing the new objective after historical test inspection makes this a post-hoc extension, not independent confirmation. High-confidence errors and class differences remain. [Method, results, reproduction and limitations](policy97/README.md).

The student's last Ed status was no reply; no teaching-team approval is inferred. The formal report and existing PR now include the extension; Colab remains the original baseline snapshot. Turnitin receipt and later formal PR-review response are separate external states.

## Historical documentation record - 22 October 2026

The following records the earlier decision before the additional experiment. Its then-unimplemented statements are historical, not current policy status.

# Post-consultation documentation and threshold decision record

This record documents the completed feasibility check-off and tutor feedback as reported by the student, alongside observed results and the pending Ed clarification.

## Consultation and document history

- The feasibility check-off is completed. No exact completion date is assigned in this record.
- The tutor suggested including failure analysis in the experimental record; the student described the other consultation matters as satisfactory.
- The 15 October preliminary review and the 21 October updated review remain dated snapshots. Their then-pending status statements are not retrospectively rewritten. The final README now explains the later reported completion and the completed report/PR work.
- The initial Ed review/check-off request was reported posted on 21 October. A separate follow-up question about the unresolved final requirement interpretation has been prepared; sending is not confirmed and no reply has been recorded. The two communications must not be conflated.

## Locked experimental finding

The unchanged threshold rule scans 0, 0.01, ..., 1.00 on calibration nodes. It requires at least `max(1, floor(0.05 * N_calibration))` accepted nodes and at least 90% empirical accepted-case accuracy, maximises coverage, and breaks ties towards the smaller threshold. If no candidate qualifies, 1.01 rejects all predictions.

All three GCN runs select threshold 0 because the full calibration set already meets the 90% target. Test coverage is 100%; the five reported failures, including a low-confidence near-tie, are accepted. This satisfies the fitting objective while failing the intended ambiguity-routing purpose. Completing the analysis does not establish that the operational goal has been met.

## Decision and next actions

- Retain the original code, model weights, calibration settings, result files and figures. This documentation update neither retrains a model nor replaces an inconvenient result.
- Do not retune on the already inspected test labels or selected errors. The cases support evidence-led diagnosis, not a new independent validation claim.
- Preserve the confidence analysis and five failure autopsies while requirement clarification is unresolved.
- Possible future risk bounds, review budgets, ambiguity margins, content audits and graph ablations are proposals only. They require a new validation protocol and independent evaluation data before any efficacy claim.
- The Turnitin receipt has not been provided, and the later formal PR review response remains outstanding.

## Scope of this update

Only report documentation and the current status record are changed. Reported numerical results, nine existing figures, the six software checks and the original dated reviews are preserved. Failure discussions now distinguish evidence, hypothesis, limits and future action. These changes improve traceability and interpretation without asserting new experimental evidence.
