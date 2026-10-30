# AI assistance disclosure

I used OpenAI Codex throughout this project for code generation,
refinement, debugging, experiment execution, analysis and documentation. I
reviewed and approved the project milestones, selected the project scope and
approved the dataset source and final reports. The automated checks described
below were executed through Codex; my review is distinct from independently
writing, running or testing the implementation.

## Repository and environment

Codex assisted with the staged plan, repository structure, dependency pins,
Python environment and Git exclusions. Compatibility checks exercised PyTorch
and PyG on CPU and CUDA, finite gradients, metric imports and plotting;
`pip check` passed. Small artificial tensors were software checks, not
experimental data.

## Dataset loading and integrity

Codex investigated the prescribed source, implemented the pinned numeric-array
adapter to PyG Data and deterministic stratified label partitions, and documented
provenance. I approved using the specification-linked Gayanku artifact, now
explicitly prescribed by specification v2.5. Automated checks verified the SHA-256,
22,470 nodes, 128 features, four classes, graph integrity and disjoint splits.
An independent UCI comparison matched labels by node ID and unique edges.
Checks also covered duplicate-edge/self-loop normalization and isolation of
training, selection, calibration, policy and test labels. No replacement
features were fitted locally.

## MLP and GCN implementation and training

Codex generated and refined the two-layer MLP, PyG GCN, shared training loop,
early stopping, checkpoint handling and learning-curve exports. Codex ran
real-data smoke tests and seed-42 training, then checked checkpoint restoration
and repeated training. Reproductions matched histories, selected weights and
plots locally. Reserved-label poisoning tests checked that training and model
selection used their designated targets. I approved the simple architectures
and comparable training settings; these were not independently authored or
executed by me.

## Evaluation and visualisations

Codex implemented classification metrics, confusion matrices, hidden-embedding
t-SNE and resource profiling, and ran the trained models on selection nodes.
Analytic fixtures checked metric calculations; real-data repeat runs matched
metrics, predictions, embeddings and plots. CUDA inference measurements used
warm-up and synchronization. Timing variability and the single-seed/split
limitation are reported rather than treated as exact repeatability.

## Calibration, referral and failure analysis

Codex implemented temperature scaling, ECE/NLL/Brier metrics, confidence plots,
threshold search and failure diagnostics. I requested the coverage objective
with at most 5% observed accepted-case error; Codex implemented the minimum
accepted count and deterministic search. Automated tests checked fitting/search
partition isolation, numerical cases and threshold selection. Repeated analyses
matched temperatures, thresholds and plots. Codex investigated genuine node
errors using training-labelled neighborhoods, distinguishing observations from
hypotheses. Mixed calibration outcomes and highly confident accepted mistakes
were retained in the report without tuning them away.

## Inference and final held-out verification

Codex generated label-free inference, artifact-identity checks, error handling
and the guarded final-evaluation driver. Fresh-process tests exercised both
models, checkpoint/policy pairing and missing-file errors. The final methodology
and artifact manifest were frozen before explicit held-out label release.
Codex evaluated both checkpoints with unchanged temperatures and thresholds,
then analysed genuine test failures. No test-driven retraining or retuning was
performed. All 62 tests passed together. A clean-source snapshot with copied
frozen artifacts reproduced test logits, metrics, selected cases and PNG bytes;
it reused the existing pinned environment and was not fresh-environment
installation or full retraining.

## Report and feasibility review

Codex drafted and revised the README, feasibility review, provenance notes,
reproduction instructions and assessment checklist, and checked the v2.5
specification against the implementation. I reviewed and approved the reports
and supplied my identity and consultation-completion information. Codex checked
reported values against recorded real experiments and preserved their technical
results during wording revisions. Codex also implemented the local PDF export
and inspection helpers. Checks confirmed a 21-page report with 15 embedded
figures and a matching one-page, footer-free feasibility review, with no blank
pages or out-of-bounds text; rendered pages were visually inspected through
Codex. This record supplements the mandatory disclosure in README.md.
