# Pre-pull-request audit

Audit date: 29 October 2026; updated 30 October 2026. Source: `recognition/task.md`, especially Sections
2.6, 3, 4.1, 5.2, 5.3, 5.4 and 6.1. This is an evidence checklist, not a grade
prediction or a claim of clinical validation.

## Requirements and evidence

| Requirement | Evidence/status | Remaining action |
|---|---|---|
| One approved Hard project | FLAN-T5 on BioLaySumm LaymanRRG; original model, LoRA and full fine-tuning | No new training required for this audit |
| Model components in `modules.py` | PyTorch wrapper, pretrained loading, LoRA, forward and generation | Uses Transformers/PEFT rather than reimplementing T5; explain and cite this honestly |
| Dataset module and split handling | Root `dataset.py`; identifier filtering, connected grouping, masked targets, prompt | Patient-level separation unverified and documented; no claim of patient-independent performance |
| Sequence handling | Complete sequences retained; dynamic padding; separate source/target padding; `-100` labels | This is an intentional no-truncation policy, not a claim that sequences were limited to 512 |
| Train/validate/test/save driver | Root `train.py`; recovery, development loss and best-model saving; test-only mode now delegates to evaluator | New test-only CLI checked locally; reported final results came from `evaluate.py` |
| Saved-model prediction | Root `predict.py` now accepts full/LoRA checkpoints, text/file or held-out examples and CUDA/MPS/CPU | Verified from job 714343 output: both final checkpoints, three CUDA examples each, all EOS |
| Baseline under identical evaluation | Untuned instruction-prompted FLAN-T5, same 10,000 reserved examples and decoding | Implemented via pretrained generation API, not an independently trained baseline |
| Required metrics | All four ROUGE metrics, total/adaptation parameter counts, memory, training time, tokens/s and latency | No confidence intervals or repeated training seeds; limitations stated |
| 3–5 safety autopsies | Four final cap-hit cases plus separate low-overlap/reference-quality review | Student must verify and defend interpretations; not expert clinical assessment |
| Recommendation | Authored README recommends restricted human-reviewed drafting, describes trade-offs | Proposed safeguards are not implemented or validated |
| Figures and example usage | Workflow, final results/learning curves, development input/output example | Visually check final README PDF; excerpt terms remain to be confirmed by the student |
| One-page feasibility review and check-off | Full review in README and docs; student reported tutor check-off 23 October | Retain original one-page version/confirmation; Markdown does not certify printed page count |
| Dependencies and reproducibility | Separate Mac, Rangpur and reporting requirements; seeds and settings described | Historical model revision not pinned; full clean-room retraining has not been performed |
| AI disclosure | Required exact subsection present and detailed | Student must confirm accuracy and personal verification before submission |
| References and attribution | T5/FLAN/LoRA, dataset/shared-task, ROUGE, Transformers/PEFT implementation references | Student should read relevant sources and confirm attribution is sufficient |
| Progressive Git history | Numerous milestone commits from setup through training/profiling | Current organisation/report edits are uncommitted; review then commit them |
| Correct fork/branch | Local `topic-recognition`; origin is wxrlp/PatternAnalysis-2026 | Actual PR must target shakes76/PatternAnalysis-2026 `topic-recognition` |
| No committed data/weights | No tracked datasets, result dumps, checkpoints or virtual environment found in project | Recheck staged files before committing; ignored artifacts remain local |
| Slurm/distributed-launch wording | A100 jobs provided; reorganised training scripts use one-process `torchrun` | Historical runs used Python launcher; do not claim multi-GPU training |
| Turnitin PDF | Not generated/checked by this audit | Export the final README to PDF, inspect it, then submit to Turnitin |
| PR and later review response | Not performed by this audit | User submits PR before deadline and responds in teaching-team review window |

## Directory organisation

The five required files remain directly under the student project:
`modules.py`, `dataset.py`, `train.py`, `predict.py`, `README.md`.
Root `config.py` and `evaluate.py` remain available to their importers.

- `scripts/inspection/`: data/token/adapter/prediction inspection.
- `scripts/diagnostics/`: original GPU, forward/backward and tiny-overfit checks.
- `scripts/analysis/`: failure candidate selection and inference profiling.
- `scripts/setup/`: cache preparation.
- `scripts/reporting/`: results figure generation.
- `jobs/train/`, `jobs/evaluate/`, `jobs/profile/`, `jobs/setup/`, `jobs/predict/`:
  Slurm entry points.
- `docs/`: supporting plans, reviews and this checklist.
- `requirements/`: dependency snapshots for separate environments.
- `figures/` and `report_data/`: report assets and numerical measurements.

Supporting Python modules run as `python -m scripts.inspection.inspect_data`
from the project root. Local checkpoints, predictions, environments and caches
have not been deleted or moved. No upload, remote job, staging, commit, push or
pull request was performed during this cleanup.

## Report author text and final prediction verification

The student approved both method paragraphs, which are now included under
Models and training procedure. The opening draft notice and writing prompts
have been removed from the README. The submission checklist is retained below
rather than included in the assessed report text.

The supplied log from Rangpur job 714343 shows the final LoRA and full
checkpoints each loading on CUDA and producing three held-out summaries. All
six outputs ended with EOS; the supplied error log contains warnings but no
traceback. This establishes functional prediction, not clinical accuracy.

## Checks completed during this audit

- Parsed all 24 project Python files (excluding environments and runtime artifacts).
- Checked every Slurm script with `bash -n`.
- Resolved local links from the README and supporting documents after the moves.
- Exercised the `predict.py`, `train.py`, and `evaluate.py` CLI help paths.
- Loaded the existing tiny-overfit full checkpoint on CPU and generated the
  expected 18-token explanation; no new training occurred.
- Loaded the existing LoRA pilot checkpoint on CPU and generated a seven-token
  explanation. This verifies adapter-aware prediction loading locally, not the
  final checkpoints that remain on Rangpur.
- Checked indirect identifier grouping and split separation on a synthetic
  six-row dataset without downloading clinical data.
- Checked the training driver's test-only delegation with a mocked subprocess;
  it did not run training or recompute final metrics.
- Simulated evaluation interruption after 25 predictions, recovered only the
  remaining two predictions, checked saved stopping diagnostics, and rejected
  changed generation settings. No actual model was loaded for this simulation.
- Ran the moved prediction-inspection module against downloaded results and
  regenerated the report figure with the reorganised plotting module.
- Checked tracked project files for environments, raw results, datasets and
  checkpoint extensions; none were found.

These checks do not establish a clean-environment installation, validate the
historical measurements independently, or replace student review. Numerical
training/evaluation settings for the completed experiments were not changed.

## Submission checklist

- [x] Insert all approved author sections and remove unfinished prompts.
- [x] Insert the complete feasibility review and record the completed tutor check-off.
- [x] Update and verify the trained-model prediction interface (Rangpur job 714343).
- [x] Include cluster dependency snapshot and document revision limitations.
- [ ] Verify training transcription against downloaded raw cluster logs.
- [ ] Review clinical interpretations and permissions for example excerpts.
- [ ] Complete AI disclosure and citations honestly.
- [ ] Check figure rendering, links, commands, and README PDF export.
- [ ] Check Git excludes data/weights and has meaningful progressive commits.
- [ ] Submit working PR to upstream topic-recognition and README PDF to Turnitin.
- [ ] Respond to later teaching-team review within the published window.
