# FLAN-T5 radiology summarisation implementation plan

Status: planning. No training or evaluation results have been collected.

This is the working checklist for the Hard difficulty radiology report summarisation project in `recognition/task.md`, Section 2.6. Update checkboxes, decisions, and experiment records as work progresses. Development is on `topic-recognition`.

## Objective

Adapt FLAN-T5 to translate technical radiology reports into clear layperson summaries using BioLaySumm 2025 Subtask 2.1 LaymanRRG. Investigate whether adaptation improves readability and clinical factual fidelity compared with the untuned model, rather than simply increasing ROUGE.

Compare three configurations of the same checkpoint on identical evaluation examples:

1. Untuned, instruction-prompted FLAN-T5 baseline.
2. FLAN-T5 with LoRA adaptation.
3. FLAN-T5 with full fine-tuning.

Use PyTorch and Hugging Face tooling, subject to verification of current APIs and course expectations. Begin with FLAN-T5-small for pipeline development. Choose the final checkpoint after measuring cluster resources; keep it identical across the main comparison. Pretrained components must be attributed, and their use must satisfy the course's model implementation requirements.

## Acceptance criteria

- [ ] A reproducible command loads saved weights and produces summaries for unseen reports.
- [ ] The baseline, LoRA, and full fine-tuning runs use the same audited splits, checkpoint, input prompt, and evaluation procedure.
- [ ] Report ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum, plus total/trainable parameters, peak GPU VRAM, training time, and generation throughput.
- [ ] Audit 3–5 representative held-out cases for hallucinations, critical omissions, altered certainty, and readability, including comparisons with the baseline.
- [ ] Document a measured resource budget and provide a recommendation about human review and possible guardrails, supported by actual results.

Set numeric quality and resource targets after the data audit and smoke test, before the main experiments. There is no specified minimum ROUGE score in the task brief. Do not invent a clinical safety threshold or treat automated metrics as proof of safety.

## Planned files

| File | Responsibility |
|---|---|
| `modules.py` | PyTorch model wrapper, generation interface, and LoRA/full-training configuration; no NumPy dependence in model components |
| `dataset.py` | Dataset parsing, split audit, tokenisation, prompts, truncation diagnostics, and target padding masked with `-100` |
| `train.py` | Training, validation, final evaluation, checkpoint saving, and loss/metric plots |
| `predict.py` | Load a trained checkpoint/adapter, summarise held-out reports, and display input/output examples |
| `evaluate.py` | Shared generation, ROUGE, profiling, and structured case exports |
| `configs/` | Explicit experiment settings and seeds |
| `scripts/` | Rangpur Slurm jobs and any required distributed launch commands |
| `requirements.txt` | Exact tested dependency versions |
| `.gitignore` | Exclude datasets, weights, caches, and raw clinical outputs |
| `README.md` | Final report, feasibility review, instructions, results, citations, and AI disclosure |

Keep reusable model logic in `modules.py`; avoid putting the entire implementation into a notebook. Add helpers when they make responsibilities clearer.

## Phase 1 — Confirm data, resources, and scope

- [ ] Verify the exact LaymanRRG dataset source, version, licence, access requirements, fields, and official splits.
- [ ] Inspect example pairs and count records, missing/empty values, duplicates, and input/target token lengths.
- [ ] Identify patient or study identifiers where available. Check overlap across splits; document limitations if identifiers are unavailable. Resolve exact duplicate leakage without silently redefining official benchmark splits.
- [ ] Confirm whether test references are available. If they are hidden, use official validation as the final labelled evaluation set and reserve a separate portion of training data for development. Document this choice before training.
- [ ] Confirm Rangpur GPU type, VRAM, allocation limits, storage locations, and available software.
- [ ] Run a small baseline generation and forward/backward smoke test; measure memory and runtime.
- [ ] Check tutor expectations for using pretrained FLAN-T5 through a modular wrapper.
- [ ] Record the midpoint check-off, submission deadline, and later review response window from official course information.

Deliverable: data audit and measured feasibility evidence.

## Phase 2 — Feasibility review and compulsory check-off

- [ ] Write a one-page review covering intended users, scope, 3–5 testable acceptance criteria, model/course concepts, preliminary evidence, risks, budget, next experiment, and fallback.
- [ ] Include a resource target informed by Phase 1 measurements.
- [ ] Present the review at the compulsory tutor consultation and record completion.
- [ ] Include the review in the final README.

Deliverable: completed check-off. This is a pass hurdle; a trained final model is not needed for it.

## Phase 3 — Build and verify the baseline pipeline

- [ ] Implement loaders and deterministic split handling in `dataset.py`.
- [ ] Fix an instruction prompt asking for plain language while preserving findings, negation, and uncertainty. Select it using development data only.
- [ ] Choose source and target lengths from the audit; log the fraction of examples truncated. Inspect whether truncation removes important findings.
- [ ] Mask target padding with `-100`; verify it does not contribute to the loss.
- [ ] Implement the model wrapper and a common generation configuration.
- [ ] Evaluate the untuned baseline and save local predictions and metrics.
- [ ] Verify ROUGE tokenisation and sentence handling, especially ROUGE-Lsum. Record package versions and settings.
- [ ] Test meaningful failure points: split overlap, target padding, checkpoint reload, and adapter reload.

Deliverable: a complete baseline with reproducible evaluation and inference commands.

## Phase 4 — LoRA and full fine-tuning

- [ ] Verify adapter target module names for the selected FLAN-T5 checkpoint; log trainable parameters.
- [ ] Implement separate LoRA and full fine-tuning configurations.
- [ ] First overfit a tiny training subset to check that the optimisation pipeline works; keep this diagnostic separate from reported experiments.
- [ ] Run pilot experiments to set learning rates, effective batch size, epochs, and generation limits using development data.
- [ ] Use gradient accumulation and mixed precision when supported. Measure their actual resource effects.
- [ ] Save checkpoints, tokeniser settings, model/data revisions, adapter configuration, seeds, and resolved experiment settings.
- [ ] Plot training and validation loss and validation metrics. Define checkpoint selection using development data before final evaluation.
- [ ] Create Slurm scripts for longer runs. If distributed execution is required, verify the working `torchrun` launch command rather than copying the brief's `torch.run` spelling.
- [ ] Run the main comparison under a declared tuning and compute budget; document any differences in training settings.

Fallback: reduce sequence lengths only after checking information loss, reduce batch size with accumulation, or use a smaller checkpoint consistently across all three configurations. If full fine-tuning remains infeasible, consult the tutor about the required comparison and report the measured limitation rather than claiming it was completed.

## Phase 5 — Final benchmarking and safety analysis

- [ ] Freeze configurations before generating results on the final held-out evaluation set. Do not tune on that set.
- [ ] Evaluate all three configurations on the same examples and report all four required ROUGE metrics.
- [ ] Profile total/trainable parameters, peak allocated GPU VRAM, training wall time, and generation tokens/second. State GPU, precision, batch size, input/output lengths, warm-up, and timing boundaries.
- [ ] Inspect output lengths, empty/repetitive generations, and truncation-related failures.
- [ ] Select 3–5 cases using a documented rule that includes successes and failures; include negation, uncertainty, and important findings where present.
- [ ] Compare source report, reference, baseline, LoRA, and full fine-tuning outputs for each case.
- [ ] Annotate unsupported claims, omitted important findings, changed certainty/negation, and readability. Tie each judgment to text in the source; state the limits of student clinical expertise.
- [ ] Explain why ROUGE alone cannot establish clinical safety. Readability scores may supplement the audit but do not prove patient comprehension.
- [ ] Recommend whether the extra training cost is justified and what human review or guardrails should be investigated. Distinguish proposed guardrails from implemented and tested ones.

Deliverable: benchmark table, resource table, learning curves, case analysis, and evidence-based recommendation.

## Phase 6 — Documentation and submission

- [ ] Complete `predict.py` with checkpoint loading and clear example usage.
- [ ] Verify documented commands from a clean environment with pinned dependencies.
- [ ] Write the README: purpose, architecture figure, working principles, feasibility review, preprocessing, split justification, training settings, results, resource costs, case analyses, limitations, recommendation, and references.
- [ ] Include a subsection titled **Artificial Intelligence Usage Disclosure** identifying tools, assisted tasks, and how their suggestions were audited and tested.
- [ ] Cite adapted code and pretrained components in relevant docstrings, README, and PR description.
- [ ] Check that datasets, model weights, and raw clinical records are excluded from Git. Confirm permissions before including any example text in public documentation.
- [ ] Open a professional PR against upstream `topic-recognition` with a working version before the deadline.
- [ ] Export the README to PDF and submit it to Turnitin.
- [ ] After teaching-team review, address requested changes and reply within the published response window.

## Development and experiment records

Make genuine milestone commits as work is completed: data audit, baseline, LoRA, full fine-tuning, evaluation, and documentation. Do not manufacture a retrospective commit history or submit everything in one commit.

For each experiment, record:

| Field | Required information |
|---|---|
| Identity | Run ID, date, Git commit, configuration path |
| Data | Dataset revision, split manifest, counts, preprocessing |
| Model | Checkpoint revision, adaptation method, LoRA settings |
| Training | Seed, precision, batch size, accumulation, optimiser, learning rate, epochs |
| Generation | Decoding settings, input/output limits |
| Environment | Dependency versions, GPU model |
| Evidence | Metrics, resource measurements, local checkpoint/output paths, failures |

## Decisions to resolve

| Decision | Current position |
|---|---|
| Final checkpoint | FLAN-T5-small for smoke tests; final size pending resource measurements |
| Dataset access and exact schema | Pending verification |
| Development and final evaluation splits | Pending official split/reference audit |
| Training budget and numeric acceptance targets | Pending pilot measurements |
| Public example text | Pending dataset licence/access review |
| Course dates and pretrained implementation expectations | Pending course/tutor confirmation |

## Immediate next step

Verify dataset access and Rangpur resources, then run the data audit and a small FLAN-T5 baseline smoke test. Use those measurements to complete the feasibility review before committing to expensive training.
