# FLAN-T5 for Layperson Radiology Report Generation

COMP3710 / COMP7710 Pattern Recognition Project — s4983489

**Status: data loading implemented; model implementation pending.** No training, benchmark results, or clinical safety conclusions are available yet. This document records the intended study and will be updated with measured evidence.

## Problem and Research Question

The project aims to translate technical radiology reports into plain English that a patient can understand, while preserving the original findings, negation, anatomical location, and diagnostic uncertainty. The intended reading level is approximately sixth grade. Explaining medical terminology may require a longer output than the source report.

The central question is: **Does fine-tuning FLAN-T5 improve factual faithfulness and readability compared with a zero-shot baseline, and how does LoRA compare with full fine-tuning in quality and computational cost?**

This is the Hard-difficulty radiology report summarisation project described in Section 2.6 of the supplied assessment specification. The intended deliverable is a research prototype; suitability for patient-facing use must be assessed from the eventual evidence.

## Model Principles

FLAN-T5 is an instruction-tuned encoder-decoder language model. The encoder processes a task instruction and the technical report; the decoder generates the layperson explanation. Full fine-tuning updates all model parameters. LoRA trains low-rank adaptations while keeping the original model parameters frozen.

The exact checkpoint size will be selected after measuring the resources required for both fine-tuning methods. All three experimental conditions will start from the same pretrained checkpoint.

## Dataset and Task Definition

Dataset: [BioLaySumm2025-LaymanRRG-opensource-track](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track).

| Field | Intended use |
| --- | --- |
| `radiology_report` | Source text supplied to the model |
| `layman_report` | Reference layperson explanation used as the training target and, where available, for evaluation |
| `source` | Dataset origin, retained for auditing and checking sampling balance |
| `images_path` | Image filename metadata; images are not inputs to this text-only study |

The [published dataset metadata](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track/blob/main/README.md) lists 150,454 training examples, 10,000 validation examples, and 10,537 test examples. These counts were confirmed by running `dataset.py`. The availability of non-empty reference answers throughout the test split remains to be checked before using it for scoring.

The implementation uses `load_dataset()` to load the official splits directly. It does not merge, re-split, or modify the original reports. The training split is used for training; the official validation split is used for validation and comparison as required by the assessment. The test split can support additional evaluation if its reference answers are available. Using official splits does not by itself establish patient-level separation.

At this stage, running `dataset.py` prints the size of each split and the source/target texts of the first training example. Tokenization, sequence truncation, and target padding masks will be added in the next implementation step.

### Run the Data Loader

The script was successfully run in the existing `ICLR3.0` Conda environment with Python 3.12.13 and `datasets` 5.0.0. No dependency changes were needed in that environment.

Run these commands from the repository root:

```bash
conda activate ICLR3.0
python recognition/FLAN_T5_s4983489/dataset.py
```

Hugging Face downloads the dataset on first use and caches it outside the repository. For this verification, the data was downloaded to a temporary cache and then loaded from that cache using `ICLR3.0`.

Expected split counts:

```text
train: 150454 samples
validation: 10000 samples
test: 10537 samples
```

## Planned Experiments

| Condition | Parameter updates | Purpose |
| --- | --- | --- |
| Zero-shot FLAN-T5 | None | Establish performance before task-specific adaptation |
| Full fine-tuning | All model parameters | Measure the benefit and cost of full adaptation |
| LoRA fine-tuning | Adapter parameters | Compare adaptation quality and resource requirements |

The conditions will use the same source formatting, task instruction, evaluation examples, and generation settings. The two fine-tuned conditions will use the same training examples and a documented, comparable training budget. Any model or hyperparameter selection using validation results will be reported explicitly.

If a subset is needed for preliminary experiments, its selection will be reproducible and its source distribution recorded. Preliminary subset results will be labelled separately from the final benchmark.

## Evaluation Plan

- **Text overlap:** ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum, calculated from generated outputs against complete reference texts with documented scoring settings.
- **Computational cost:** total and trainable parameters, GPU model, peak GPU memory, training time, and inference latency and generated tokens per second under stated measurement conditions.
- **Training behaviour:** training and validation loss, plus generation-based validation metrics at documented intervals.
- **Qualitative audit:** 3–5 representative cases comparing the original report, reference explanation, zero-shot output, and fine-tuned outputs. The analysis will examine unsupported claims, important omissions, changes to negation or certainty, and layperson readability.

ROUGE measures textual overlap and does not establish clinical correctness. A small case audit can illustrate failure mechanisms but cannot establish an overall clinical safety rate. The final recommendation will describe the limitations supported by the actual results.

## Implementation Plan

| File | Responsibility | Status |
| --- | --- | --- |
| `dataset.py` | Official data loading, tokenization, and target padding masks | Loading implemented; tokenization pending |
| `modules.py` | Model construction and selection of full fine-tuning or LoRA | Not implemented |
| `train.py` | Training, validation, measurements, plots, and checkpoint saving | Not implemented |
| `predict.py` | Reload the saved model or adapter and generate outputs for new reports | Not implemented |

The current implementation loads the official dataset and displays an example pair. A subsequent small training experiment must check generated outputs on held-out examples and verify that saving and reloading preserves predictions; a decreasing training loss alone is insufficient.

## Feasibility Review and Reproducibility

The compulsory one-page feasibility review is pending. It will record 3–5 testable acceptance criteria, the selected checkpoint, measured preliminary performance and resource use, the compute budget, technical risks, and a fallback. Completion of the tutor check-off will be recorded only after it takes place.

The verified data-loading environment and command are documented above. Random seeds, checkpoint and dataset revisions, training settings, additional dependency versions, and training commands will be recorded as those parts are implemented. The interpretation of the assessment's model-dependency rules should be confirmed with the tutor before relying on Transformers and PEFT in the submitted implementation.

Datasets, downloaded base models, and trained checkpoints must remain outside Git. The final report will contain the measured comparisons, example outputs, plots, and evidence-based recommendations, and will be exported to PDF for submission.

## Artificial Intelligence Usage Disclosure

OpenAI Codex has assisted with inspecting the repository, checking the supplied assessment requirements and official technical sources, moving the project into the required directory, drafting this project plan, and implementing the initial data-loading script.

The planning statements and data-loading API were checked against the supplied assessment specification and the linked official documentation. The loading script was executed against the official dataset and rerun successfully in `ICLR3.0`, confirming the split counts and displaying a source/target pair. No model code has been implemented or empirically validated at this stage. This disclosure will be updated to describe subsequent AI assistance and the specific checks performed on any suggested code. The student remains responsible for reviewing and defending the submitted work.

## References

- Supplied COMP3710 / COMP7710 assessment specification, particularly Sections 2.6 and 3–6. The supplied document is marked Draft; course announcements determine the applicable dates and any updated requirements.
- [BioLaySumm LaymanRRG dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track).
- [Hugging Face Datasets loading guide](https://huggingface.co/docs/datasets/loading).
- [FLAN-T5 model documentation](https://huggingface.co/google/flan-t5-base).
- [Hugging Face Transformers summarisation guide](https://huggingface.co/docs/transformers/tasks/summarization).
- [Hugging Face PEFT overview](https://huggingface.co/docs/peft/main/en/quicktour).
