# Feasibility Review - Normal GNN

**Yuhan Liao | Student ID 49839991 | COMP3710, Semester 2, 2026 | 21 October 2026**

**Consultation status:** awaiting tutor confirmation; this is not a check-off completion record.

## 1. User need, scope and course concepts

A social-network reviewer needs four-class Facebook page predictions and a way to identify uncertain cases for manual review. The prototype compares a two-layer GCN with a feature-only MLP on FacebookPagePage: 22,470 nodes, 128 supplied features, and four classes (tvshow, government, company, politician). Both models are 128-64-4 with 8,516 parameters. Native PyTorch implements the GCN; PyG Data stores the graph. Normalised sparse neighbour aggregation, self-loops and nonlinear layers connect the design to graph matrices, representation learning and semi-supervised classification.

## 2. Five testable acceptance criteria

1. Data integrity: valid finite features and edge endpoints; reproducible, exhaustive, disjoint node splits; no held-out labels in training loss or message inputs.
2. Matched benchmark: run GCN and implemented MLP on the same split; report per-class precision/recall/F1, Macro-F1 and AUROC across the specified seeds, irrespective of which model wins.
3. Confidence analysis: fit temperature and an acceptance threshold using calibration data only; report test coverage, accepted-case error and reliability, including any failure to reject ambiguous cases.
4. Interpretability: show ground-truth-coloured t-SNE plots and examine 3-5 real errors, distinguishing observed evidence from causal hypotheses.
5. Resources and reproducibility: record parameters, synchronised full-graph inference latency and peak PyTorch allocated GPU memory against the original 4 GiB planning target; retain versions, commands and saved-model reload evidence.

## 3. Feasibility evidence and current findings

The 15 October audit and five-epoch CPU checks established execution feasibility; all six software checks passed. Subsequent formal GPU runs on 21 October used seeds 42, 43 and 44 on one fixed split. GCN mean test accuracy is 93.84% (sample SD 0.32 percentage points) and Macro-F1 is 0.9333; MLP accuracy is 80.77% and Macro-F1 is 0.7884. Five real GCN errors, embeddings and saved-weight inference are recorded. Crucially, the original 90%-accuracy/max-coverage calibration rule selects threshold 0 for GCN: coverage is 100%, so it does not provide useful ambiguity routing. This limitation remains unresolved.

## 4. Leakage controls, resource budget, risks and fallback

The setting is transductive: all node features and edges are visible, but labels never enter messages; only training labels enter optimisation. The split has 13,481 training / 4,493 validation / 4,496 test nodes; validation is separated into 2,245 selection and 2,248 calibration nodes. Standardisation is fitted on training nodes; supplied embeddings were precomputed upstream. A shared A100-PCIE-40GB allocation was limited to 10 minutes. GCN peak allocated memory was 82.98 MiB and mean per-run median whole-graph forward latency was 0.780 ms, excluding loading and adjacency construction; these are allocator/shared-device measurements. Risks include overconfident errors, misleading neighbourhoods and split-specific conclusions. CPU debugging and the simpler MLP are technical fallbacks; neither is a validated replacement for human review.

## 5. Next actions and consultation request

Please discuss this review and confirm the required check-off or any permitted catch-up procedure. Formal experiments have already been completed; this current review does not claim that an earlier tutor check-off occurred. Remaining work is the final README/report, repository/PR and Turnitin submission. Any stricter review policy needs a separately validated protocol; the inspected test set will not be used to retune this submission. No additional GPU training is currently planned.

References: course assessment draft, Sections 2.1 and 3; Kipf & Welling (2017), arXiv:1609.02907; course-linked gayanku/facebook_page-page_dataset (pinned ec01d02).

AI assistance: ChatGPT/Codex supported planning, implementation, verification and drafting. Evidence was checked against code, tests and recorded experiments; student review and understanding remain essential.
