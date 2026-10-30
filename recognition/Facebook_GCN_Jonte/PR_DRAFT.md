# Proposed pull request

Base: `shakes76/PatternAnalysis-2026:topic-recognition`

Head: `LittleEmperor99/PatternAnalysis-2026:topic-recognition`

**Title:** Classify Facebook pages with a two-layer GCN and calibrated referral policy

## Description

This Normal-difficulty COMP3710 project compares a two-layer PyG GCN
(128 -> 64 -> 4) with a feature-only MLP for transductive Facebook page
classification. It evaluates whether graph context improves classification,
confidence reliability and automation coverage enough to justify its cost.

The implementation uses the modified Gayanku Facebook Page-Page dataset
prescribed by specification v2.5, pinned by revision and checksum. It includes
integrity audits, deterministic train/selection/calibration/policy/test roles,
training and checkpoint restoration, label-free inference, and a frozen final
evaluation procedure. Reproduction instructions recreate model checkpoints;
datasets, weights and generated experiment outputs are excluded from the
current repository tree.

On the same 4,494 held-out nodes, GCN achieves 92.48% accuracy and 0.920009
macro-F1, versus 78.95% and 0.772748 for MLP. The unchanged referral policies
achieve 92.68% coverage at 4.80% observed accepted-case error for GCN, versus
47.86% coverage at 4.18% error for MLP. Highly confident incorrect predictions
remain, and the report recommends supervised analyst assistance rather than
guaranteed autonomous safety. Results are limited to one seed and split.

The README includes per-class metrics, OvR AUROC, confusion matrices, t-SNE,
resource measurements, calibration comparisons, five genuine test failure
investigations, the one-page feasibility review and AI usage disclosure.
Validation covers analytic metric fixtures, real-data graph integrity and
normalization, label-role isolation, checkpoint reloads, frozen-artifact
reproducibility and fresh-process inference.

References: [GCN](https://arxiv.org/abs/1609.02907),
[temperature scaling](https://proceedings.mlr.press/v70/guo17a.html),
[t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html),
[UCI dataset](https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network),
[modified dataset](https://github.com/gayanku/facebook_page-page_dataset).
