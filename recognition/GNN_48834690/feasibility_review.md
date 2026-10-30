# Feasibility Review: Multi-layer GNN for Facebook Page-Page Node Classification

**Student:** Nasser Kessas (48834690) · **Project:** 2.1 Classification, Multi-layer GNN · **Difficulty:** Normal

## 1. User need, scope and acceptance criteria

**User:** a content-moderation or analytics team that has to sort Facebook pages into four categories (company, government, politician, TV show) for review or reporting.
**Prototype purpose:** classify every page automatically. Handle the confident predictions with no human involved, and send the uncertain ones to a human reviewer. This addresses the *Confidence and Ambiguity Dilemma*: high overall accuracy is not enough if the model is confidently wrong on ambiguous pages.

**Acceptance criteria** (measured on a held-out, stratified test split; models are compared with an MLP baseline on the same split):

1. **Accuracy:** GCN test accuracy ≥ 0.90 and macro-F1 ≥ 0.90, and both higher than the feature-only MLP baseline.
2. **Calibration:** expected calibration error (ECE) ≤ 0.05, and the max-softmax confidence separates correct from wrong predictions with AUROC ≥ 0.80.
3. **Reject option:** a confidence threshold, chosen on the validation split only, gives ≥ 99% accuracy on the pages it handles automatically and handles more of them than the baseline does at the same accuracy.
4. **Resources:** trains in under 10 min on a laptop CPU, with no GPU required. Inference over the whole graph takes under 1 s, and the model has under 50k parameters.
5. **No leakage:** transductive setting. Only training-node labels enter the loss. Feature standardisation uses training nodes only. The threshold is never tuned on the test set.

## 2. Model choice and course concepts

**Data:** the Facebook Large Page-Page Network [Rozemberczki et al., 2021], as the course-provided `facebook.npz` (gayanku): 22,470 pages, 171,002 mutual-like edges, and 128-dimensional SVD features from page descriptions.
**Split:** a 70/15/15 split of the nodes, stratified by class with a fixed seed. The setting is transductive: all edges are visible, and validation/test labels are masked.
**Models:**
- **Baseline:** a 3-layer MLP that sees only each page's own features.
- **Proposed:** a 3-layer GCN [Kipf & Welling, ICLR 2017]. Each layer computes H' = D^-1/2 (A+I) D^-1/2 H W, a degree-normalised average over the page and its neighbours.

Both models have the same hidden width, so any difference comes from the graph alone.
**Course concepts:**
- MLPs, and convolution generalised from image grids to graphs
- semi-supervised learning
- softmax and cross-entropy
- regularisation (dropout, weight decay) and early stopping on validation
- t-SNE for inspecting the embeddings

## 3. Preliminary feasibility evidence

- **Data audit:** the classes are mildly imbalanced (company 6,495; government 6,880; politician 5,768; TV show 3,327).
- **Edge homophily is 0.885:** 88.5% of edges join pages of the same class, which strongly suggests that aggregating over neighbours will help.
- **Label check:** the dataset README's label order is wrong. The real order is alphabetical, which I confirmed against the raw MUSAE CSV (22,470/22,470 labels match). Without this check, every per-class result would have been reported under the wrong class name.
- **Baseline smoke test:** the MLP reaches 82.1% test accuracy (macro-F1 0.80). It trains in about 30 s on CPU with 12,676 parameters. TV-show recall is the weakest at 0.64, which shows that page features alone are not enough.

## 4. Risks, budget and fallback

| Risk | Mitigation / fallback |
|---|---|
| Over-smoothing as the GCN gets deeper | Keep 2–3 layers. Use dropout and early stopping on validation |
| Pages whose neighbours mostly belong to another class get "pulled" to the wrong class | Compare with the MLP in the failure-case analysis. Send pages to review when the GCN and MLP disagree |
| The GCN is confidently wrong, so a threshold cannot catch it | Measure ECE and the reliability diagram. Temperature scaling on the validation set as a fallback |
| Results depend on one random split | Fixed seeds. Report the sampling error on test metrics |
| GCN underperforms | Fall back to GraphSAGE (mean aggregation), using the same pipeline |

**Compute budget:** a laptop CPU, about 5 min per full training run of both models. Rangpur is only needed if GPU memory has to be profiled.
**Next experiment:** train the 3-layer GCN under the same split. Then measure calibration, choose the reject threshold on validation, and analyse the 5 most confident errors.
