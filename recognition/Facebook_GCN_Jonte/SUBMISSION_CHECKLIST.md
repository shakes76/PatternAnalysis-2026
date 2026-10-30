# Assessment compliance and submission checklist

Initial audit: 18 October 2026; verification repeated 28 October 2026; feasibility
review wording refreshed 29 October 2026. Source: supplied 21-page Version 2.5
assessment PDF, especially Sections 2.1, 3, 4.1, 5 and 6. Its cover includes DRAFT;
administrative dates must follow the operative course profile. These are evidence
checks, not predicted marks or a claim that external hurdles are complete.

| Requirement | Where addressed | Status |
| --- | --- | --- |
| Appropriate classification solution (5 implementation marks) | README working principles/final held-out evaluation; `modules.py`, `final_evaluate.py` | Implemented and genuinely evaluated |
| Working PyTorch implementation/demonstration (3) | MLP/GCN, `predict.py`, real-data and fresh-process tests | Verified |
| Modular design (1) | Model/data/train/inference files and focused evaluation helpers | Complete |
| Readable/commented source (1) | Docstrings, role guards, method/source references | Present; student review still required |
| Normal-or-above model (5) | Two-layer 128–64–4 PyG GCN | Normal tier; implementation ceiling 15/20 |
| Hard-only implementation criterion (5) | Not selected | Not pursued by design; not eligible under Normal tier |
| Progressive technical commits (5, pass hurdle) | Eight reviewed milestones including the approved Stage 8 commit | Evidence present; originality/understanding remains student's responsibility |
| Title, problem, principles and visuals (documentation 5) | README introduction, working principles, example/embedding/confusion figures | Complete |
| Usage and script comments (documentation 3) | README reproduction/inference/final-evaluation/PDF commands | Complete |
| GitHub Markdown formatting (documentation 2) | README tables, fences, references and relative assets | Structural checks passed; PDF tables/figures inspected |
| Five required files in dedicated recognition folder | `modules.py`, `dataset.py`, `train.py`, `predict.py`, `README.md` | Present |
| Model implementation in approved framework | PyTorch and PyG only in model components | Verified; NumPy/sklearn used for utilities only |
| Prescribed graph and 128-feature provenance | README data section; `DATA_PROVENANCE.md`; pinned checksum | Specification-linked reconstruction explicitly disclosed; not original PyG-equivalent features |
| Transductive setting and leakage prevention | Feature/edge-only graph; separate masks/target access guards; tests | Verified |
| Implemented baseline on identical splits | MLP and GCN selection/test tables | Complete |
| Per-class precision/recall, macro-F1, AUROC | Final held-out tables/confusions, raw and scaled scores | Complete |
| Ground-truth-coloured t-SNE and interpretation | README selection embedding figure; fixed projection settings | Complete |
| Parameter count, peak VRAM and latency | Development/final resource tables; 20 warm-up/100 synchronized calls | Complete; scopes and variability disclosed |
| Confidence reliability and empirical referral | ECE/NLL/Brier, diagrams, frozen temperatures/thresholds, actual coverage/error | Complete; no future-risk guarantee claimed |
| 3–5 genuine failure autopsies | Five held-out IDs, predictions/confidences, degrees and training-only neighbors | Complete; hypotheses separated from observations |
| Engineering recommendation | README recommendation: supervised analyst assistance, fallback and limitations | Complete |
| One-page written feasibility review | README dedicated section; PDF isolated page 2 | Verified; footer-free one-page standalone copy generated from the same source |
| Compulsory tutor-recorded feasibility consultation (pass hurdle) | Student confirmed completion on 30 October 2026; consultation date not supplied | **Completed (student-reported)** |
| Exact dependencies/reproducibility | Requirements/lock files; README; frozen protocol; clean source snapshot | Verified as far as practical; existing pinned environment reused, no clean-environment retraining |
| Mandatory AI usage disclosure | README exact named subsection; tools/tasks/verification | Present and accurate; student final review pending |
| Attribution in README/source/PR | References, source docstrings and local `PR_DRAFT.md` | Prepared; actual PR not opened |
| Current-tree repository restrictions | `.gitignore`; datasets/weights/caches/generated reports absent from Git | Verified |
| History restrictions | `514067a` contains three small CSV/JSON aggregate reports, untracked by `ecfce37` | **Decision needed** on literal Section 4.1 compliance; no history rewrite authorized |
| PR to upstream `topic-recognition` before deadline (2 PR marks, pass hurdle) | Local draft and correct remotes/branch | **Pending explicit push/PR authorization**; no push/PR |
| Professional PR title/description (1 PR mark) | `PR_DRAFT.md` | Draft prepared, not posted |
| Turnitin PDF export | Local `outputs/submission/COMP3710_Facebook_GCN_Report.pdf` | Exported and inspected; submission not performed |
| Later review response (2 PR marks) | Published review window after Phase 1 | Future task; cannot complete before teaching-team feedback |

## Verification record

- Stage 7's five tests passed again before its seven-file commit.
- Final driver passed three new tests on selection labels before final release.
- Freeze manifest precedes label release and binds model/config/data/mask/source
  hashes; before/after checks reject any mismatch.
- First held-out results are retained; no retraining/refitting/threshold selection
  used test labels. A clean-source rerun checks reproducibility with copied frozen
  artifacts and the same environment, not a claim of new-environment retraining.
- **All 62 automated tests passed together**, repeated on 28 October
  (98.709 seconds; original Stage 8 run 111.158 seconds). `pip check` found
  no broken requirements. The existing PyG typing deprecation warning remains.
- Clean-source reproduction matched every test logit, classification/calibration
  metric, referral decision, failure selection and PNG byte exactly. The pinned
  ML environment and existing artifacts were reused; no full retraining occurred.
- README local/reference links, tables, fences, source structure and figure paths
  passed checks. The 21-page PDF includes all 15 images, has no blank pages or
  text outside page bounds, and isolates the feasibility review on page 2.
  Every page was rendered to a contact sheet; feasibility and key result/failure
  tables were also inspected at full resolution. Generated audits stay ignored.

## Remaining submission actions

1. Student reviews final code, report, disclosure and historical-report issue.
   Do not rewrite history without explicit authorization; clarify acceptance
   with teaching staff if keeping those historical aggregate outputs.
2. Tutor consultation is completed, as confirmed by the student on 30 October
   2026; retain the course check-off record for submission verification.
3. Stage 8 commit approval was received on 28 October. After commit verification,
   separately authorize push and the PR to upstream `topic-recognition`.
   Nothing has been pushed.
4. Inspect the exported PDF personally and submit it to Turnitin; retain the
   receipt and PR URL. No external submission has been performed.
5. Respond to later teaching-team review within its published window.
