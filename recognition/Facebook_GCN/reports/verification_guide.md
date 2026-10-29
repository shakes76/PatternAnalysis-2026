# What the verification establishes

Yuhan Liao - 49839991 - 26 October 2026

The full suite currently passes **26 targeted unit tests**. These small fixtures check defined implementation properties. Their number and success do not demonstrate generalisation, absence of all leakage, originality, clinical safety or future accuracy. Real runs, saved-model reloads and error analysis provide separate evidence.

Run from the project directory: `python -B -m unittest discover -v`.

## Each test and its purpose

| Test (prefix `test_` omitted) | Specific property checked |
| --- | --- |
| split_reproducibility_class_coverage_and_no_overlap | Fixed seed reproduces the split; classes remain represented and duplicated split IDs are rejected. |
| scaling_does_not_fit_on_held_out_nodes | An extreme held-out feature cannot change training mean/scale; constant training dimensions use scale one. |
| duplicate_self_loops_are_removed_before_gcn_normalization | Duplicate links and original self-loops do not gain repeated graph weight. |
| invalid_endpoints_and_nonfinite_features_are_rejected | Out-of-range graph endpoints and NaN features fail validation. |
| binary_graph_normalization_with_duplicate_loops | Sparse normalised adjacency agrees with a small explicit graph calculation. |
| permuting_nodes_permutes_predictions_and_gradients_are_finite | Relabelling a synthetic graph permutes outputs consistently, and backpropagation yields finite gradients. |
| equality_all_refer_and_invalid_probabilities | Threshold equality accepts; 1.01 rejects all; malformed probabilities are refused. |
| unqualified_predicted_class_refers_all | A class with no qualifying fitting group receives the all-refer fallback. |
| routes_depend_on_predicted_class | Class-specific routing uses the predicted label, without needing the true label. |
| five_label_roles_are_disjoint_and_reproducible | Training/selection/temperature/policy/final IDs are exhaustive, disjoint and repeatable. |
| wrong_checkpoint_binding_is_rejected | A policy cannot silently be applied to an unrelated checkpoint. |
| partial_experiment_folder_is_preserved | An existing partial run is refused rather than overwritten. |
| finds_feasible_interval_above_point99 | Exact score search finds a legitimate group between .99 and 1.00 missed by the coarse grid. |
| confidence_ties_cannot_be_partially_accepted | Equal-confidence cases are accepted as a whole block, never selected using their correctness. |
| all_accept_tie_uses_zero_and_small_groups_refer | The all-accept tie uses zero, while an undersized class cannot qualify. |
| malformed_values_and_labels_are_rejected | Wrong label range and invalid probability/threshold arrays fail validation. |
| class_weights_use_training_labels_only | Changing held-out labels leaves class weights unchanged; missing training classes fail. |
| audits_are_reproducible_subsets_without_labels | Random quality review samples only eligible cases, reproducibly and without labels. |
| confident_disagreement_is_referred | Two individually certain but conflicting outputs cannot pass unanimity routing. |
| shared_confident_error_is_not_detected_by_unanimity | A deliberately shared wrong prediction still passes: this test demonstrates a method limit, not successful error detection. |
| averaging_is_member_permutation_invariant | Reordering members preserves averaged probabilities and agreement; sums remain one. |
| invalid_members_and_temperatures_are_rejected | Too few members, NaNs and invalid ensemble temperatures fail. |
| temperature_preserves_mean_class_predictions | Positive scalar temperature changes sharpness while preserving the class on the fixture. |
| fitting_excludes_ineligible_cases | Ineligible policy cases do not enter gated threshold fitting or gated acceptance. |
| empty_eligible_group_rejects_all | No eligible policy cases produces a fail-closed all-refer rule. |
| true_and_predicted_politician_counts_are_separate | A true politician missed into another class and a false-positive politician are counted in the correct groups. |

See the corresponding five `test_*.py` files and [saved test output](consensus/TEST_RESULTS.txt). Synthetic examples do not establish their incidence in the real graph. Test labels used in fixtures are not labels supplied to deployment routing.

## Real execution evidence

- Six original A100 training runs, source/data checksums, training histories, metrics and GPU resource measurements remain the original benchmark.
- Six five-role CPU runs and three weighted CPU runs are separately recorded, with explicit same-graph and post-hoc limits.
- Both new consensus policies are saved before the new comparison, but final labels were already inspected earlier; freezing does not restore project-level independence.
- Two actual consensus replays each match all 2,254 saved class/confidence predictions and policy decisions. Decision CSVs contain no reference labels; the optional 5% random sample adds 81 reviews.
- Two negative integration checks corrupt a copied policy and a copied checkpoint. The production replay rejects each before creating an output. Original artifacts are preserved.
- Five selected missed-politician cases receive same-split MLP, training-neighbour, nearest-training-feature and node-edge-isolation diagnosis. Edge isolation measures model sensitivity under changed graph inputs, not a population causal effect or an annotation audit.
- One full-graph CPU forward and the six-member pipeline are profiled separately from original GPU results; neither timing measures loading, training or human review.

Complete accounting and negative checks are in [consensus validation](consensus/VALIDATION.json). Unanimity has no observed incremental final-case benefit; a shared >=.99 true-politician error remains. Verification must not be presented as having solved these empirical limits.
