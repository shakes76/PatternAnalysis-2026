# Current assessment alignment and evidence boundaries

Yuhan Liao - 49839991 - 29 October 2026

Checked against the user-supplied **Final Report Assessment v2.5**, Sections 2.1, 3-6 and Appendix A.6, including the later supplied PDF with the same relevant body. Classroom tips are applied as engineering guidance; they are not substituted for the official rubric. Feasibility completion is student-confirmed. This is an evidence map, not a prediction of marks or approval by the teaching team.

## Formal marking criteria

| Criterion | Marks | Current evidence and remaining boundary |
| --- | ---: | --- |
| 1.1 Appropriate solution | 5 | Native two-layer GCN solves the four-class transductive task; matched MLP, calibration and runnable review rules are evaluated. Original zero-threshold routing failure, class trade-offs and residual high-confidence errors remain disclosed. No zero-error claim. |
| 1.2 PyTorch/TF implementation works | 3 | Explicit PyTorch training and reloads; real GPU/CPU experiments, 26 targeted tests, two new complete saved-model replays and two tamper-rejection checks. These are particular checks, not exhaustive correctness. |
| 1.3 Design quality | 1 | Separate model/data/training/evaluation/prediction; training-only scaling, sparse graph normalisation, bound saved policies, protected output paths. |
| 1.4 Comments/readability | 1 | English docstrings and protocol boundaries; source roles in README; [each-test explanation](verification_guide.md). New model computations use Torch. |
| 1.5 Normal or above | 5 | The specified FacebookPagePage task with actual multi-layer neighbour aggregation and 128-dimensional features is implemented. |
| 1.6 Hard | 5 | Not applicable to this Normal GNN. Additional analyses or an ensemble do not change its difficulty classification. |
| 2.1 Meaningful commit messages/originality | 2 | Real progressive source/experiment/document commits, cited methods and broad AI disclosure. Commit existence does not independently prove authorship; the student remains responsible for understanding and permitted use. |
| 2.2 Progressive logical history | 3 | Original planning/data/prototype/GPU/report work and later evidence-based refinements have separate actual milestones. History is preserved, not backdated. |
| 3.1 Principles and problem | 5 | README explains GCN equations, data, transductive scope, fairness, all required metrics, embedding interpretation, original five failures, additional five politician cases, calibration diagnostics, measured costs and engineering choices. |
| 3.2 Usage and script comments | 3 | Exact executed versions, reproducible commands, expected inputs/outputs, saved-model requirements and data/weight exclusions. Each added policy has a runnable bound replay; new paths protect prior runs. |
| 3.3 GitHub Markdown | 2 | Headings, tables, relative images/links and source references. Formal PDF is exported from the same current README and visually checked. |
| 4.1 Correct working PR by deadline | 2 | Existing PR #9 targets upstream `topic-recognition`; final remote file identities and source ZIP are checked after sync. The exact personal deadline/time is governed by Blackboard. |
| 4.2 Incorporate/respond to review | 2 | Current read-only API check found no reviews, review comments or discussion comments. This is a future phase, not completed in advance. Actual feedback must be addressed during the announced window. |
| 4.3 Professional PR description | 1 | Description states task, implementation, benchmark, reproducibility, negative findings, evaluation limits and AI assistance; updated to final scope. |

Normal implementation is capped at 15/20; the overall structural maximum is 35/40 if all other criteria receive full marks. This is a cap, not an awarded or predicted grade.

## Required GNN investigation and delivery

| Requirement | Evidence |
| --- | --- |
| Correct dataset and PyG container | Pinned modified course FacebookPagePage; SHA256, numeric validation, 22,470 x 128 features, PyG Data. Upstream supplied SVD construction is explicitly not claimed as train-only. |
| Leakage boundary | Transductive features/edges visible; labels never enter forward or messages. Scaling uses training nodes; selection, temperature, policy and final roles are separated in the follow-up. Previously inspected graph/final outcomes remain post-hoc. |
| Implemented baseline/same splits | Matched feature-only MLP and GCN, same original evaluation masks, matched width/parameter count and three initialisations. Later comparisons identify their different split/test sizes. |
| Quantitative metrics | Accuracy, per-class precision/recall/F1, Macro-F1, macro OVR AUROC, calibration and coverage/error counts. No class metric is replaced by a whole-population score. |
| Embedding | Ground-truth-coloured t-SNE for MLP/GCN, with critical interpretation and stated projection limits. |
| Resource profile | Original A100 parameter/peak VRAM/latency table; additional consensus CPU cost is scoped separately, without claiming an unmeasured ensemble GPU profile. |
| 3-5 failure autopsies | Original five real cases remain; additional selected politician cases distinguish observations, hypotheses and fixed-weight graph sensitivity. Shared confident errors and no-incremental gate result are retained. |
| Actionable recommendation | Retain the simpler single-GCN research replay. Six-member pooling has modest post-hoc gains, 6.23x CPU latency and no additional gate benefit; not automatically adopted from inspected scores. New data/content review is needed before reliability claims. |
| Mandatory architecture | `modules.py`, `dataset.py`, `train.py`, `predict.py`, `README.md` in `recognition/Facebook_GCN/`; model components use only Torch. Helpers do not replace these required files. |
| Feasibility review/hurdle | Historical one-page review included in README; check-off confirmed by student, without fabricating an institutional receipt. |
| AI disclosure | Dedicated exact required heading states tools, broad tasks, verification and student responsibility; includes new CPU analysis/diagnosis. |
| Git exclusions | No raw data, trained weights, split IDs or full per-node labelled exports in repository or source ZIP. Private evidence archived separately. |
| Turnitin | Current PDF is prepared from README. A successful institutional receipt has not been provided; this workflow does not claim submission or infer non-submission. |
| Prior assessed work | If code is reused from an earlier assessed demo, the specification requires a substantially different final model. No prior assessed codebase was supplied for independent comparison; a feasibility prototype alone is not evidence of an earlier graded demo. |

## Classroom tips addressed

- A concrete protocol precedes each new experiment; roles and constraints are fixed and outcomes preserved.
- Each test has a purpose and boundary; a passing suite is not a substitute for real error analysis.
- Negative attempts are part of the record: zero threshold, below-target seeds, precision loss, worsened politician ECE, shared errors and no observed unanimity gain.
- Actual staged commits, modular structure, README/PDF consistency and a professional PR preserve reviewability.
- Cost and fallback choices are measured; no additional school GPU quota is required for the saved-model refinement.
- AI assistance is disclosed broadly, and no automated workflow can substitute for the student's understanding.

External data, page-content/annotation investigation, human-review effectiveness, Turnitin receipt and future formal reviewer responses cannot be manufactured by code or report edits. Ed screenshots supplied on 29 October show a methodological clarification and a request for the source of temperature T; the report now identifies Guo et al. (2017), Section 4.2, Equation 9. No experimental target or replacement threshold is described as formally approved, and no PR review was present at the 29 October check. Colab stays unchanged under the student's explicit instruction.
