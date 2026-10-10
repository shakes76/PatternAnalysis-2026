# Radiology Report Summarisation with FLAN-T5

**Author:** Phan Anh Minh Le  
**Course:** COMP3710 / COMP7710  
**Project:** Radiology Report Summarisation: Clinical Language Models

## Project status

Repository scaffold only. The intended task is to translate technical radiology
reports into accessible layperson summaries while investigating factual fidelity,
critical omissions, and readability. Model and dataset selection, downloads,
baseline experiments, fine-tuning, and evaluation have not started.

Consult the author before further code edits and before each commit.

## Repository setup

- Personal fork: <https://github.com/Mikelele311/PatternAnalysis-2026>
- Upstream: <https://github.com/shakes76/PatternAnalysis-2026>
- Working and submission target branch: `topic-recognition`
- Project directory: `recognition/FLAN_T5_PhanAnhMinhLe/`
- Use the existing Python environment; no separate environment is required.
- Dependencies and their exact versions will be documented once the implementation
  and training platform are agreed. No packages were installed during setup.

The existing environment inspected during setup has Python 3.12.12,
PyTorch 2.4.1+cpu, and Transformers 5.8.0. PyTorch currently reports no CUDA or
HIP support. These are observed versions, not a validated dependency specification.
The training platform is still to be agreed with the author.

## File responsibilities

| File | Intended responsibility |
| --- | --- |
| `modules.py` | Model components and adaptation configuration |
| `dataset.py` | Data loading, split auditing, preprocessing, and tokenisation |
| `train.py` | Training, validation, evaluation, checkpoint saving, and learning curves |
| `predict.py` | Loading trained weights and demonstrating inference |
| `README.md` | Setup instructions, feasibility review, experiments, and final report |

The Python files currently contain documentation placeholders only and do not
provide a runnable algorithm.

## Data and generated files

Keep datasets, model weights, caches, and training outputs out of Git.
The root `.gitignore` excludes conventional directories such as `data/`,
`datasets/`, `checkpoints/`, `models/`, `outputs/`, and `.cache/`, along with
common checkpoint formats. Review `git status` before each approved commit.
Report figures may be stored in a separate `figures/` directory when available.

## Feasibility review

To be written after consultation: intended user, scope, testable acceptance
criteria, model rationale, initial feasibility evidence, compute budget,
technical risks, and fallback.

## Experimental methodology

To be designed after consultation. The assessment draft calls for an untuned
baseline, an investigation of adaptation methods, leakage-free evaluation,
ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum, and resource measurements.
No experiments or performance results are available yet.

## Clinical safety and error analysis

To be completed using 3-5 representative held-out cases, examining hallucinations,
critical omissions, changes in diagnostic certainty, and layperson readability.
Include an evidence-based recommendation and explain the limits of ROUGE as a
measure of clinical safety.

## Reproducibility and usage

Exact package versions, dataset revision and split audit, random seeds,
preprocessing choices, training commands, inference commands, and hardware
measurements will be added after the implementation is agreed and verified.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with inspecting the assessment draft and local environment,
setting up the GitHub fork and local checkout, and creating this repository
scaffold and Git exclusions. The author authorised scaffold creation and asked
to defer model and dataset work. No model implementation, experiments, or
performance claims have been generated. Record subsequent assistance and the
author's review and verification as development progresses.

## References

- Course assessment specification supplied by the author:
  `COMP3710_Report_2026_draft_v2.2 (1).pdf`, particularly Sections 2.6 and 4.1.
- Course repository: <https://github.com/shakes76/PatternAnalysis-2026>
