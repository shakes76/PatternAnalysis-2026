# One-page feasibility review
## Alzheimer’s classification with ConvNeXt
**Status:** Initial planning draft based on supplied directory listings; experiments and resource measurements pending.

### User need and research question
Develop a research-only assistant for medical-image reviewers: classify ADNI brain MRI images as Alzheimer’s disease (AD) versus cognitively normal (CN), and flag uncertain predictions. Does ConvNeXt improve classification and confidence-based referral over a simple CNN enough to justify its computational cost? This is not a clinically validated diagnostic system.

### Model choice and course concepts
Proposed main model: ConvNeXt-Tiny in PyTorch, within the approved Hard Difficulty ConvNeXt family; baseline: a small CNN using the same preprocessing and evaluation split [1, 2]. Relevant concepts are convolutional feature learning, cross-entropy optimisation, regularisation, generalisation and confidence calibration. Input dimensions and channel handling will follow image inspection.

### Preliminary feasibility evidence
Read-only listings confirm access to /home/groups/comp3710/ADNI, AD_NC/{train,test}/{AD,NC}, sampled .jpeg files under test/NC, and meta_data_with_label.json. This is directory-level evidence only: image decoding, patient counts/mapping, split independence and compute performance remain unverified. The disk label NC must be reconciled with the brief’s CN terminology.

### Five testable acceptance criteria
| Criterion | Target / evidence |
|---|---|
| 1. Data integrity | Verify patient identifiers and zero patient overlap across training, validation and test sets; keep all visits and slices of a patient together. |
| 2. Classification | Target ≥0.80 held-out test accuracy for ConvNeXt [1]. Report per-class precision/recall, macro-F1 and AUROC; predeclare the evaluation unit and any slice aggregation. |
| 3. Fair comparison | Train a reproducible simple CNN baseline and ConvNeXt, record learning curves, and compare them on identical evaluation data without test-set tuning. |
| 4. Confidence and errors | Select a referral threshold on validation data only. Report held-out coverage, error among accepted predictions and AD recall; analyse 3–5 representative failures and compare both models. |
| 5. Resources | Proposed cap: ≤12 GiB peak allocated GPU memory at the chosen training settings. Record both models’ parameter counts, inference latency and training time; confirm the cap after a feasibility smoke test, before full training. |

### Data splitting and next experiment
Use metadata to establish patient/scan/visit relationships; do not treat a filename prefix as a verified patient ID. Retain the supplied test partition only if patient-independent, and derive validation from training patients. If overlap or missing identifiers prevent this, pause and agree a corrected split with the tutor. Next: inspect metadata, audit counts/overlap, decode training samples, then run a small baseline smoke test in an allocated GPU session.

### Budget, risks and fallback
Provisional planning budget: one GPU and 12 GPU-hours for initial experiments; availability, quota and runtime are not yet confirmed. Risks include leakage, label ambiguity, imbalance, overfitting, overconfidence and memory limits. Reduce batch size/resolution or use supported mixed precision if necessary, while retaining ConvNeXt as the Hard main model. A baseline-only fallback does not preserve full-mark eligibility. Revise the budget transparently using measured evidence.

**Sources and status:** [1] Supplied COMP3710 assessment brief, §§2.1, 3–4. [2] Liu et al., “A ConvNet for the 2020s,” CVPR 2022 (brief reference [7]). All budgets/configurations above are proposals, not measured results.
