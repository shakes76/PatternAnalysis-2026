# Frozen 97% policy follow-up - 26 October 2026

This is a post-hoc extension of the original Normal Facebook GCN experiment. It preserves the trained models and original 90% rule. The old test had already been inspected; these results are not independent confirmation. See [the report](../../README.md#post-hoc-policy-improvement---26-october-2026) for the comparison, seed-44 shortfall and interpretation.

## Method and provenance

Using the original calibration subset and original temperatures, choose the maximum-coverage threshold with at least 97% empirical accepted-case accuracy and at least `max(1, floor(0.05 * N_calibration))` accepted cases. The grid remains 0.00-1.00 in 0.01 steps; ties prefer the smaller threshold. If no threshold qualifies, 1.01 refers all cases and accepted accuracy is undefined. All three policies were saved before this follow-up calculated reused-test metrics. 97% is exploratory, not a course requirement, population risk bound, or per-class guarantee. No human correction accuracy is assumed.

Seeds 42/43/44 yield thresholds 0.73/0.76/0.74 and reused-test accepted accuracy 97.02%/97.20%/96.99%. Mean referral is 10.74% and accepted accuracy 97.07%; whole-test model accuracy stays 93.84%. Seed 44 falls below 97%; no threshold was changed after seeing this follow-up. Seed 42 remains the prespecified main example. Politician accepted accuracy is approximately 94.80-95.42%, and highly confident mistakes can still pass.

`FROZEN_PROTOCOL.json`, `POLICIES_FROZEN_BEFORE_FOLLOWUP.json`, `FOLLOWUP_RESULTS.json`, `FOLLOWUP_SUMMARY.csv` and `PROVENANCE.json` retain the protocol, pre-evaluation hashes and all aggregate/per-class results. The three portable policy files are in [../../policies](../../policies). Per-node evidence, checkpoints and data remain in private backups, outside Git.

## Run from recognition/Facebook_GCN

Install `requirements.txt` with an appropriate CPU/CUDA PyTorch wheel and obtain the pinned data as in the main README. Restore the original three GCN checkpoints and `summary.json` files in `runs/final/gcn_seed42`, `gcn_seed43`, `gcn_seed44`. Saved policies are bound to those precise checkpoint hashes; newly trained checkpoints need their own calibration-fitted policies.

```bash
python policy97_experiment.py --output runs/policy97_replay
python predict_with_policy.py --policy policies/gcn_seed42_policy97.json --checkpoint runs/final/gcn_seed42/model.pt --output runs/policy97_seed42_predictions.csv
```

Use a new output path to preserve existing evidence. `--data` and `--runs-dir` allow private paths. CPU is sufficient for this saved-model replay; no new GPU training is needed. The experiment replay is still post-hoc reuse, even when executed again.

`predict_with_policy.py` outputs `node_id,predicted_label,predicted_class,confidence,accepted,route,policy_id`; routes are `automatic_candidate` or `human_review`. It neither exports ground-truth labels nor calculates accuracy. The policy verifies checkpoint, dataset, seed, temperature and model configuration. The original `predict.py` still applies the original checkpoint's policy for reproducibility.

## Boundaries

Do not present selective accuracy as whole-population accuracy, assume human review corrects every error, or retune using these reused-test outcomes. A future untouched evaluation is required for independent policy confirmation. The original GPU profiling measures the unchanged model forward pass, not the added end-to-end review workflow. Colab is the unchanged original baseline snapshot and does not display this follow-up.

ChatGPT/Codex assisted with the policy proposal, implementation, execution, result checks and documentation; see the main disclosure. The experiment follows the coverage/error trade-off of [selective classification](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html); no theorem-based risk guarantee is implemented.
