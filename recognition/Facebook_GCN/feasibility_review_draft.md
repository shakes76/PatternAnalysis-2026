# Feasibility Review

**Project:** Confidence-aware Facebook Page Classification - Normal GNN

**Working draft - 15 October 2026.** Data and short CPU checks are complete. Tutor check-off and formal experiments remain outstanding.

**Scope and user need.** Build a Normal-difficulty node-classification prototype that helps a reviewer categorize Facebook pages and refer uncertain predictions for manual review. Investigate whether a multi-layer GCN provides more useful and reliable predictions than a feature-only MLP at an acceptable resource cost.

**Proposed data and model.** Use the course-linked FacebookPagePage mirror, with 128-dimensional SVD features and four classes, converted to native PyG Data. Compare a two-layer GCN with a feature-only MLP; both use hidden width 64 and have 8,516 parameters. GCN adds normalized neighbour aggregation, linking the project to graph structure, matrix operations and semi-supervised learning. Use fixed stratified train/validation/test splits. Split validation further into model-selection and calibration roles. The transductive model uses all node features and edges, but only training labels enter optimization; labels are never messages or input features.

**Acceptance criteria.**

1. Verify dataset dimensions and edge validity, and demonstrate that the saved split is reproducible, exhaustive and disjoint.
2. Train both models and report per-class precision/recall/F1, macro-F1 and AUROC on the same held-out split. Diagnose the outcome whether or not GCN improves on MLP.
3. Select a confidence threshold using validation data and report held-out coverage and accepted-case error. Compare confidence reliability between the models.
4. Provide an interpreted t-SNE/UMAP visualization and explain 3-5 actual failure cases.
5. Measure parameter count, inference time and GPU memory. The planning budget is one GPU with a 4 GiB peak-memory target; this remains unmeasured and does not assume particular hardware availability.

**Preliminary evidence.** Data audit: 22,470 nodes, 128 finite float32 features, four classes and 342,004 raw edge entries. Seed 42 gives 13,481/4,493/4,496 training/validation/test nodes without overlap; validation contains 2,245 selection and 2,248 calibration nodes. All six software tests passed. Five CPU epochs reduced MLP training loss from 1.421 to 1.082 and GCN loss from 1.393 to 1.069. Both saved models were reloaded successfully. These checks establish execution feasibility, not final quality. The 4,496-node test set remains unevaluated. Source checksums, splits, versions and short-run evidence are saved with the project.

**Risks, budget and fallback.** Main risks are leakage, oversmoothing and overconfident errors. Keep labels out of message passing, use training-only feature scaling, and use MLP as the operational fallback if graph context adds insufficient value. Canonicalize duplicate/self edges before GCN normalization. The new Virtual GPU Lab has a finite quota; Rangpur is another option. CPU feasibility checks used no school GPU time. Check the actual GPU and package environment before longer jobs.

**Next steps and schedule.** Discuss this review with the tutor and record check-off, then run matched experiments across three initialization seeds, inspect confidence and failures, and complete the report. The course profile lists 19-30 October 2026; confirm the individual lab slot, midpoint requirement and precise submission time. Include this review in the final README.

**References:** Kipf and Welling (2017), GCN; course-linked data mirror `gayanku/facebook_page-page_dataset` (Appendix A.6).
