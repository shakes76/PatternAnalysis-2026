# Six-model confidence/disagreement investigation

Yuhan Liao - 49839991 - 26 October 2026

## Protocol and source attribution

All six pre-existing unweighted/weighted GCNs are included equally. Member temperatures are retained; their probability mean receives one new positive scalar fitted only on the temperature role. Mean-only and unanimity-gated class thresholds are fitted on separate policy cases at 98% empirical accuracy with minimum max(50, ceil(5% eligible predicted-class count)). No member or rule is chosen by final score. The final cases have already been inspected, making this post-hoc research, not independent validation. Model weights and all earlier outcomes remain unchanged.

Probability averaging is inspired by [Lakshminarayanan et al., NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html); the mixed-loss pool and unanimity gate are project-specific, not the paper's full method. Temperature scaling follows [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a). No statistical guarantee is implemented.

## Results, including negative findings

| Candidate | Whole accuracy | Coverage | Accepted accuracy | Accepted mistakes | Missed politicians accepted | Accepted errors with confidence >= .99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prespecified unweighted seed 42 | 93.66% | 72.32% | 98.04% | 32 | 7 | 11 |
| Six-model mean + thresholds | 93.74% | 71.74% | 98.21% | 29 | 5 | 8 |
| Mean + thresholds + unanimity | 93.74% | 71.74% | 98.21% | 29 | 5 | 8 |

Same 2,254 previously inspected cases; one ensemble, not three new seed replicates. The 94 disagreements are all already threshold-rejected, so the gate provides no added final-case benefit. Of seven original seed-42 accepted missed politicians, two receive lower ensemble probabilities and are referred; none changes to the correct class. All seven have unanimous wrong votes. Five are still wrong and accepted. In particular, one true politician at >= .99 confidence remains wrongly accepted; two >= .99 false-positive politician predictions also remain. Sixty-nine predicted politicians are accepted with no observed error, which is a small conditional sample, not future 100% safety.

The calibrated mean uses T = 0.971649706363678; politician cutoff is 0.9998544454574585. Other thresholds and full precision are in the frozen policy JSON. The high threshold is not a measure of model accuracy or evidence that all class predictions are overconfident.

| Diagnostic | Prespecified seed 42 | Six-model mean |
| --- | ---: | ---: |
| NLL | 0.23100 | 0.22217 |
| Top-label ECE | 0.02507 | 0.02334 |
| Politician one-vs-rest Brier | 0.02418 | 0.02275 |
| Politician one-vs-rest ECE | 0.01191 | 0.01298 |

Politician ECE worsens despite other improvements. Binary ECE includes many non-politicians and depends on bins; graph cases are correlated. Predicted-politician mean confidence 89.51% is close to precision 89.58%, but that aggregate comparison does not validate the tail.

## Five selected missed-politician cases

These are the first five node IDs among the seed-42 baseline's accepted missed politicians, not a representative random sample. Every true label is politician and every original/ensemble prediction is tvshow. Votes are unanimous. Diagnostic labels and IDs never enter routing.

| Node | Original confidence | Ensemble confidence / route | Degree / training-neighbour evidence | Feature-only MLP | Edge-isolated seed-42 GCN |
| --- | ---: | --- | --- | --- | --- |
| 909 | 99.90% | 99.45% / accept | 141 / 53 tvshow, 20 politician | tvshow / 52.26% | tvshow / 99.93% |
| 1958 | 90.84% | 85.79% / review | 6 / 2 tvshow, 1 politician | tvshow / 97.28% | politician / 99.55% |
| 5152 | 89.68% | 71.02% / review | 1 / 1 tvshow, 0 politician | tvshow / 56.67% | tvshow / 80.08% |
| 5395 | 98.65% | 98.48% / accept | 6 / 1 tvshow, 1 politician | tvshow / 68.57% | politician / 68.22% |
| 7042 | 94.43% | 91.85% / accept | 7 / 2 tvshow, 2 politician | tvshow / 68.11% | tvshow / 69.25% |

- **909:** Many tvshow neighbours coexist with three nearest training feature vectors labelled politician. MLP also errs at 52.26%, all GCNs agree, and node isolation still errs at 99.93%. The model-family agreement and neighbour majority are insufficient evidence of truth. Content/annotation review and a new data protocol are needed to investigate shared systematic error; label corruption is not established.
- **1958:** Mixed training neighbourhood and feature-neighbour labels. Isolation flips the fixed GCN to politician at 99.55%, demonstrating response to graph context under this intervention. Averaging alone refers the original case. This is neither a verified human correction nor proof that deleting edges would improve population performance.
- **5152:** One nonself neighbour, a training tvshow, while two of the three closest training feature vectors are politicians. MLP and isolated GCN still predict tvshow. Sparse context is an observed property; it is not proved to be the cause. Averaging refers this case, with every vote still wrong.
- **5395:** A small, mixed known-label neighbourhood; isolation flips to politician at 68.22%, while original and ensemble stay wrong and accepted. This reveals sensitivity, not a deployable label-free edge-deletion rule.
- **7042:** MLP, original, ensemble and isolated GCN all remain tvshow, with confidence reduced after isolation. Nearest-feature labels include government/company. Shared representation or boundary limitations are hypotheses; no annotation/content audit has confirmed them.

The intervention removes each selected node's incident nonself edges, renormalises the graph and retains trained seed-42 weights. It changes both graph information and degree normalisation. It is selected-error model sensitivity, not a retraining ablation, population causal effect or independent validation. Full counts and source hash are in `FAILURE_AUTOPSY_GRAPH.json`.

## Cost, replay and recommendation

The pool has 51,096 parameters versus 8,516. Four-thread CPU full-graph inference, including probability calculations, takes median 486.365 ms versus 78.108 ms (6.23x; ten repeats, three warmups). This excludes loading, graph construction, training and human review. Ensemble GPU cost was not measured; the original A100 benchmark remains separate.

```bash
python consensus_study.py --base-study runs/exact_weighted_study --output runs/consensus_study
python predict_consensus.py --study-dir runs/consensus_study --policy unanimous --audit-fraction 0.05 --output runs/consensus_routes.csv
```

The first command reloads six models and fits the two fixed policies; it does not train. The second only replays. Private archived `members/` weights, policies and split/scaling are required; raw numeric data remains separate. New paths are mandatory. No reference labels are exported or used in routes. Replays match all 2,254 predictions and policy decisions. Optional 5% random review adds 81 reviews after 637 policy referrals, leaving 1,536 automatic candidates. No human correction is measured.

All 26 targeted unit checks pass; two negative integration checks reject changed policies/weights before creating exports. [Verification guide](../verification_guide.md) explains their limits. Retain the simpler single-GCN research replay by default: the modest post-hoc count improvement and no gate increment do not establish that sixfold cost is justified. Preserve the ensemble as an evaluated option; do not tune this final set repeatedly or claim new external data. The original zero threshold, 96.99% shortfall and weighted trade-offs remain.

ChatGPT/Codex assisted with implementation, CPU execution, diagnosis, tests, analysis and documentation. The student remains responsible for understanding the operations, loss, metrics and limitations. Colab is unchanged.
