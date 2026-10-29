# Exact-threshold repair and class-weighted GCN ablation

Yuhan Liao - 49839991 - 26 October 2026

## Scope and chronology

The original 90% GPU experiment, first 97% policy extension, and five-role coarse-grid study are preserved. This refinement follows inspection of the five-role final outcomes, so **all new final-role comparisons are post-hoc**, not independent confirmation. The same train/selection/temperature/policy/final roles and split seed 20261026 are reused. All six new policies are saved before this comparison, without adjusting weights, targets or thresholds afterwards. No external data has been provided.

## Exact search fixes a coarse-grid limitation

The old 0.01 grid jumps from 0.99 to 1.00. It can miss a useful interval immediately below one. `exact_policy.py` evaluates every distinct confidence boundary using whole tie blocks. It preserves the 98% empirical policy-role target, minimum `max(50, ceil(5% of predicted-class policy cases))`, maximum-coverage objective, threshold-zero all-accept tie rule and 1.01 all-refer fallback. It never splits equal-confidence samples according to their labels.

For predicted politician on the policy-fitting role, exact thresholds are 0.9996114373207092, 0.9996351003646851 and 0.9997503161430359 for seeds 42/43/44. They accept 65/67/64 fitting cases at 98.46%/98.51%/98.44% empirical accuracy, whereas the old grid refers all. JSON retains full precision; do not copy rounded PDF thresholds into code. This is a search-resolution fix, not a new statistical guarantee; more exhaustive fitting also does not remove fitting optimism.

## Controlled weighted-loss experiment

For three new CPU GCNs, the only training-setting change is weighted cross-entropy with `weight_c = N_train / (4 * n_train,c)`. Class counts and weights are calculated only from training labels and saved before training. The weights, in tvshow/government/company/politician order, are approximately 0.864832/0.816437/0.974061/1.688502. Architecture (128-64-4), dropout .5, Adam .01, weight decay .0005, max 200 epochs, patience 30, selection Macro-F1 and seeds 42/43/44 are unchanged. New temperatures are fitted using unweighted NLL on the temperature role; exact thresholds use the separate policy role.

The unweighted comparator reuses the five-role GCN checkpoints and original temperatures, without modifying them. Three variants are reported on the same 2,254 previously inspected final cases:

| Variant | Whole accuracy | Politician recall | Politician precision | Coverage | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unweighted / grid | 93.52% | 87.26% | 91.18% | 68.69% | 97.98% |
| Unweighted / exact | 93.52% | 87.26% | 91.18% | 72.29% | 98.08% |
| Weighted / exact | 93.88% | 92.34% | 88.22% | 72.38% | 98.02% |

These are means across the same three initialisations, not independent data replicates. Exact search changes routing, not class predictions. Weighted training increases politician recall by about 5.07 percentage points while decreasing precision by about 2.96 points; it does not establish class imbalance as the unique cause of errors or a universally superior classifier. The original 93.84% result on a different split/test size is not replaced by this comparison.

## Every seed and remaining errors

Unweighted exact policies accept 72/73/71 predicted politicians on final cases; weighted exact accepts 68/64/71. No mistakes are observed within those small accepted predicted-politician subsets, but this is not future 100% accuracy, complete politician coverage, or safety for the true politician class.

True politicians wrongly accepted as other classes change from **7/5/5 to 5/6/5** with weighting; seed 43 worsens. Total accepted mistakes are **32/33/29 versus 31/34/32**. Accepted errors with calibrated confidence at least .99 are **11/8/9 versus 7/8/8**. Thus high-confidence mistakes remain, and weighted training does not consistently reduce selective errors. Weighted seed 43 accepted accuracy is 97.93%, below the 98% fitting target; this is retained without further tuning. The simpler unweighted-exact seed-42 replay remains the default; weighted replay is an explicitly labelled comparison option.

## Label-free reload and optional random quality review

```bash
python refinement_study.py --base-study runs/five_role_study \
  --output runs/exact_weighted_study
python predict_refinement.py --study-dir runs/exact_weighted_study \
  --run unweighted_exact_seed42 --audit-fraction 0.05 \
  --output runs/exact_routes_with_audit.csv
```

The refinement command needs the original five-role weights, policies and split/scaling files, and actually trains three additional CPU GCNs. It refuses existing output folders, including partial runs. Prediction reloads saved models only. Use `--run weighted_exact_seed42` for the weighted comparison. Private data/weights are excluded from Git; use the archived private evidence or reproduce them.

The new CSV records node ID, predicted class, confidence, full-precision threshold, policy acceptance, audit flag, route, reason and policy ID without reference labels. Frozen policy/checkpoint/protocol/dataset/split bindings are verified. With no audit, the two seed-42 reloads match every one of their 2,254 stored predictions and policy decisions. In the tested unweighted example, 1,630 cases qualify and 624 are referred by threshold. A predeclared label-free random sample of 5% of eligible cases (rounded up) adds 82 quality reviews, leaving 1,548 automatic candidates. This additional workload is separate from the table's pre-audit coverage. No actual human review or correction effectiveness is measured, and a random sample cannot catch every high-confidence error.

## Verification and next boundary

All 18 targeted tests pass: original data/model/policy controls plus narrow feasible intervals, indivisible confidence ties, all-accept/infeasible groups, invalid inputs, training-only class weights and reproducible label-free audits. Per-node count accounting, exact-policy hashes, source hashes and both reloads are checked. Full six-run results and nine-row variant summary retain the lower-performing outcomes. ChatGPT/Codex assisted with design, implementation, CPU execution, testing and reporting; the student remains responsible for comprehension.

Further changes must not be selected by repeatedly optimising these now-inspected final outcomes. Truly independent external/temporal validation requires compatible new data and feature/graph construction, which are unavailable. This remains Normal GNN research, not a Hard upgrade or zero-error deployment claim. Colab remains unchanged.
