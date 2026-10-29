**Historical 0.01-grid study.** Subsequent exact-search/weighted refinement is documented in [../refinement/README.md](../refinement/README.md). The original numbers below remain unchanged.

# Five-role robustness study and class-aware triage

Yuhan Liao - 49839991 - 26 October 2026

## Question and precommitted design

This study investigates whether a cleaner separation of calibration roles and predicted-class thresholding can improve the original all-accept outcome. It is a new training experiment on the same previously studied graph, not new external data. The old test and models have been inspected; repeating a split does not restore project-level independence.

Before the study's final-role evaluation, `validation_study.py` fixed split seed 20261026, initialisation seeds 42/43/44, both MLP/GCN architectures and original training hyperparameters. The five stratified disjoint roles are train 13,481, selection 2,245, temperature 2,245, policy 2,245, final 2,254. Feature scaling uses training nodes only; selection Macro-F1 chooses checkpoints. Temperature uses only its own role; thresholds use a different role. All six models and policies are saved and hash-recorded before any final-role metrics. No split, target or threshold is adjusted after those results.

## Policies

- All-accept: reference classification accuracy across the new final role.
- Global97: maximum coverage with at least 97% empirical accuracy on the policy-fitting role.
- Class98: for each predicted class, maximum coverage with at least 98% empirical accepted policy accuracy, a one-percentage-point fitting buffer above the desired 97%. Require at least `max(50, ceil(0.05 * predicted_class_policy_count))` accepted cases. Use 0.00-1.00 in 0.01 steps, lower-threshold ties. An infeasible class gets 1.01 and is fully referred, not silently dropped. This is an empirical buffer, not a statistical guarantee.

Route selection uses probabilities and the predicted class, never true labels or a list of previously identified error IDs. The original 90% runs and the first 97% post-hoc follow-up remain separate, unchanged historical experiments.

## Once-evaluated GCN results

| Seed | Whole accuracy | Global97 referral | Global97 accepted accuracy | Class98 referral | Class98 accepted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 93.66% | 18.54% | 97.82% | 30.97% | 97.94% |
| 43 | 93.57% | 15.66% | 97.53% | 30.79% | 97.88% |
| 44 | 93.35% | 17.57% | 97.74% | 32.17% | 98.10% |

Means: whole classification accuracy 93.52%, Global97 referral 17.26% and accepted accuracy 97.70%, Class98 referral 31.31% and accepted accuracy 97.98%. Across the same initialisations, accepted-accuracy sample SD is 0.15 percentage points for Global97 and 0.11 for Class98. These are optimisation variations on one new split, not population confidence intervals. All three observed accepted accuracies exceed the desired 97%, but the 98% fitting buffer is not achieved by every final run/group.

These new checkpoints and 2,254 final cases differ from the original checkpoints and 4,496 test cases. Do not replace the original seed-44 96.99% outcome, claim it has been corrected, or compare percentage changes as matched improvements. The full six-run results, per-class precision/recall/F1, confusion matrices, AUROC and calibration metrics are retained in `STUDY_RESULTS.json`; the control MLP is never omitted because its score is lower.

## Critical class and error limitations

Class98 could not fit an eligible politician-prediction rule with at least 50 cases in any GCN seed; every predicted politician is therefore referred. Each nonempty accepted predicted-class group has observed accuracy above 97% on this final role, but this does not establish reliability for every true class.

In particular, **7, 5 and 5 true politicians are still wrongly accepted as other categories** for seeds 42/43/44. Among accepted cases whose true class is politician, accuracy is therefore 0% here: no prediction of politician is automatically accepted. This is a serious coverage/recall trade-off, not a universal class-safety solution. Other confidently incorrect predictions also remain; accepted-error counts are **32, 33 and 29** out of 1,556/1,560/1,529 accepted cases. The rule is a conservative research alternative, not an unconditional replacement deployment policy. Global97 offers broader coverage and remains a separately reported control.

Neither policy guarantees future accuracy or perfect error detection. No human correction effectiveness was measured. The original source data are globally supplied SVD features with the same graph reused throughout; no untouched external or temporal dataset has been provided. External validation remains outstanding. Do not route using true labels, hand-exclude known error IDs, or tune against these now-observed final results.

## Run and reload

From `recognition/Facebook_GCN`, after installing the documented packages and obtaining the pinned dataset:

```bash
python validation_study.py --output runs/five_role_study
python predict_study_policy.py --study-dir runs/five_role_study \
  --run gcn_seed42 --policy class98 --output runs/study_routes.csv
```

The first command actually retrains six CPU models; it refuses existing folders, including partial studies. The second reloads saved weights and checks dataset, checkpoint, frozen policy and split/scaling bindings. Its CSV contains predicted class, confidence, acceptance, route and reason, without true labels or calculated accuracy. The actual reload matched every one of the 2,254 seed-42 decisions; 1,556 were accepted and 698 referred. `--policy global97` reproduces the control route. `--subset all` exports the full-graph candidate routes, without treating training-node outputs as held-out evaluation.

Data, weights, split IDs, per-node labelled predictions and private replay artifacts stay outside Git. Public JSON/CSV contains aggregate metrics and hashes. Current checkpoints and split/scaling files are required for prediction and are retained in the private evidence ZIP. The original GPU measurements apply to the original experiments; this new study used CPU and does not invent GPU or end-to-end deployment figures.

## Engineering checks and disclosure

Twelve tests passed: the original six data/model checks plus six policy controls covering malformed probabilities, threshold equality/all-refer semantics, infeasible predicted classes, label-free class routing, disjoint reproducible roles, wrong checkpoint bindings and partial-output protection. The first 97% runner now also verifies the actual dataset checksum and checkpoint phase/model/seed/split/calibration-role/temperature, preserving original output rather than overwriting partial studies.

ChatGPT/Codex assisted with design, code, CPU training, checks, analysis and reporting. See `FROZEN_PROTOCOL.json`, `ALL_POLICIES_FROZEN.json`, `SPLIT_MANIFEST.json`, `VALIDATION.json`, `TEST_RESULTS.txt`, `ENVIRONMENT.json` and all six policy/evaluation records. The student must review and understand the changes. Colab remains unchanged. Teaching-team approval and Turnitin success are not inferred from this experiment.
