# Radiology Report Summarisation with FLAN-T5

**Author:** Phan Anh Minh Le  
**Course:** COMP3710 / COMP7710

## Status

A zero-shot FLAN-T5-base baseline is implemented. Fine-tuning has not started;
`train.py` remains a placeholder. Consult the author before further edits or commits.
Use the existing Python environment. Model weights, data, and outputs are ignored by Git.

## Files and usage

- `dataset.py`: `load_reports`, `make_prompt`, and `get_tokenized_splits`.
- `modules.py`: `load_baseline` and `count_params`.
- `predict.py`: `predict`, `evaluate_baseline`, and a command-line entry point.

Run these commands from `recognition/FLAN_T5_PhanAnhMinhLe/`:

```powershell
python -m pip install -r requirements.txt
python predict.py --report "No pleural effusion or pneumothorax."
python predict.py --limit 20 --output-dir outputs/baseline_smoke
python predict.py
```

The last command evaluates all 10,000 validation reports; it can take substantially
longer on CPU. `--limit` selects a reproducible random sample, with seed 42 by default.
The evaluator writes `metrics.json`, `predictions.json`, and `safety_cases.json`.
Safety cases require qualitative review; the program does not assign clinical safety scores.

```python
from dataset import get_tokenized_splits
from modules import load_baseline

tokenizer, model = load_baseline()
splits = get_tokenized_splits(tokenizer)
splits.save_to_disk("outputs/tokenized_splits")
```

The current local model is preferred; otherwise `load_baseline` downloads
`google/flan-t5-base`. `count_params(model)` returns total, trainable, and non-trainable
parameter counts, plus the names of layers containing trainable parameters.
Evaluation mode disables dropout; `torch.inference_mode()` disables gradient tracking.
Neither changes `requires_grad`, and this baseline performs no parameter updates.

## Data and preprocessing

Dataset: `BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track`, revision
`6783ab38f9de28c4372799346f51d3927ae7eb9a`, stored under `data/`.
The source text is `radiology_report`; the target is `layman_report`.

| Published split | Rows | Reference summaries |
| --- | ---: | --- |
| train | 150,454 | Available |
| validation | 10,000 | Available |
| test | 10,537 | All blank |

The published splits are preserved. Train and validation tokens have `labels`;
blank test targets are omitted, so test tokens are for inference only. Inputs are
truncated and padded to 512 tokens; targets to 256. Target padding becomes -100
so it contributes nothing to the loss. Attention masks preserve input padding.

**Split audit:** train/validation share 138 unique `(source, images_path)` identifiers.
They also share 2,164 distinct normalised source/report pairs; train/test share 3,
and validation/test share 1. Repeated text alone need not identify the same patient,
but overlapping image identifiers need investigation. No patient ID column is supplied.
The loader warns about shared image identifiers. Resolve leakage with the author before
fine-tuning; these published splits cannot currently be claimed to be leakage-free.
The untuned baseline has not learned from this training split.

## Baseline protocol and smoke results

Model revision: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`.
The prompt requests sixth-grade plain English while preserving findings, negations,
and uncertainty. Decoding is greedy, with a maximum of 256 new tokens.

The initial run evaluated **20 randomly selected validation reports**, seed 42.
These are smoke-test measurements, not full validation results.

| Measurement | Value |
| --- | ---: |
| ROUGE-1 F1 | 0.1522 |
| ROUGE-2 F1 | 0.0440 |
| ROUGE-L F1 | 0.1359 |
| ROUGE-Lsum F1 | 0.1391 |
| Total parameters | 247,577,856 |
| Trainable parameters (`requires_grad`) | 247,577,856 |
| Non-trainable parameters | 0 |
| Training time | 0 seconds; zero-shot |
| Training GPU resources | None |
| Peak GPU VRAM | Not applicable; CPU run |
| Generation throughput | 47.26 tokens/second |
| Timed generation | 9.16 seconds, 433 output tokens |
| Truncated sample inputs | 0 |

Throughput measures batch-size-one `generate` calls after warmup, excluding
prompt tokenisation, decoding, scoring, model loading, and safety cases. Token counts
exclude decoder-start, padding, and EOS tokens. On CUDA, the evaluator synchronises
before timing and reports peak allocated VRAM including resident model weights.
ROUGE uses the mean per-report F1 on a 0-1 scale with stemming. ROUGE-Lsum uses
sentence newlines from punctuation splitting, a simple convention with limitations
for abbreviations. Keep this convention fixed for later comparisons.

Hardware: CPU, Intel64 Family 6 Model 198, four PyTorch threads.
Python 3.12.12, PyTorch 2.4.1+cpu, Transformers 4.51.3; other direct versions are
in `requirements.txt`. Transformers was changed from 5.8.0 because model loading
failed with the installed PyTorch build. The shared environment's `pip check` also
reports an unrelated existing pandas-ta/NumPy version mismatch.

## Preliminary clinical safety analysis

Five test cases were randomly selected using seed 42. Test references are absent;
this review compares the generated summary directly with its source report.
This is a preliminary Codex-assisted textual audit for the author to verify.

| Test index | Failure observed | Faithfulness, omissions, and readability |
| --- | --- | --- |
| 409 | Output: "A radiographic report of a pulmonary fibrosis." | Omits osteoporosis, rib callus, honeycombing, and increased left-lung density. Drops the source's "probable" qualification. Pulmonary fibrosis remains unexplained jargon. |
| 1824 | Output repeats the requested reading level | Omits the entire finding that the chest has no significant radiological abnormalities. Fluent words provide no useful explanation. |
| 4012 | Output repeats the rewriting instruction | Omits the interstitial pattern, possible poor inspiration, and the source's recommendation to repeat imaging. Fails the requested transformation. |
| 4506 | Output falsely denies a calcified lymph node | The source explicitly describes a calcified left-iliac lymph node; the output says none are present. It also invents a negative statement about gastric fatty content, omits the conclusion about no evidence of residual/recurrent tumour, and retains dense CT terminology. |
| 10476 | Output repeats the report task | Omits right-lung opacities, right hilar thickening, the left-lung band, and absence of pleural effusion. The output is short but not a summary. |

These samples expose omissions, contradictions, uncertainty changes, and instruction
copying. They do not establish a failure rate or whether an omitted finding is acute.
ROUGE measures lexical overlap and can miss clinically meaningful changes in negation.
The current baseline is unsuitable for patient-facing use. Next investigations should
compare adaptation methods on agreed leakage-controlled splits, verify entities and
negations, check retention of important findings, and route uncertain outputs to human
review. A specialist must validate any clinical guardrail.

## Feasibility review

Full tutor consultation review remains to be written. Initial feasibility evidence:
local inference works, padding/masking checks pass, and a seq2seq forward pass produces
a finite loss. CPU execution supports initial experiments; GPU training resources and
budget remain to be agreed. Split overlap and factual errors are current priorities.

## Artificial Intelligence Usage Disclosure

OpenAI Codex assisted with repository setup, dataset/model downloads, implementing
and checking the baseline, dependency compatibility fixes, and preliminary error
analysis. Checks included actual split sizes, missing test references, identifier
and report overlaps, target padding masks, a finite model loss, and a 20-report
inference evaluation. The author must review the code and audit conclusions.
No fine-tuning results or full-validation metrics are claimed.

## References

- Supplied COMP3710 assessment draft, Sections 2.6 and 4.1.
- [Course repository](https://github.com/shakes76/PatternAnalysis-2026).
- [FLAN-T5 model card](https://huggingface.co/google/flan-t5-base).
- [Local dataset source](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track).
- [T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5).
- [Google ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge).
