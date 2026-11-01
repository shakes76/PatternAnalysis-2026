<!-- Maintained submission README. Edit this file directly and export it to README.pdf. -->

# Fine-tuning FLAN-T5 for Plain-Language Radiology Report Summarisation

Weiming Liang — 46375489  
COMP3710, Task 2.6

[Problem](#sec-problem) · [Algorithm](#sec-algorithm) · [Feasibility review](#sec-feasibility) · [Example inputs and outputs](#sec-examples) · [Data](#sec-dataset) · [Implementation and usage](#sec-usage) · [Reproducibility](#sec-experiments) · [Results](#sec-results) · [Error analysis](#sec-safety) · [Recommendation](#sec-recommendation) · [AI disclosure](#sec-ai-disclosure) · [References](#sec-references)

<a id="sec-problem"></a>

## 1. Problem Description and Project Overview

Radiology reports often use technical language that patients find difficult to understand. This prototype adapts FLAN-T5-base to produce plain-language summaries for the general public while preserving the findings, their location, whether they are present or absent, and any uncertainty. Given a radiology report, the model generates a summary that is easier for non-specialists to read while retaining the essential clinical information.

Using BioLaySumm LaymanRRG report–summary pairs, the project compares LoRA and full fine-tuning against the pretrained baseline on summary quality and computational cost in terms of summary quality and computing costs. The key investigation is whether fine-tuning can improve the summary quality both in readability and factual accuracy. The outputs of the models are evaluated for readability, factual accuracy, and overlap with reference summaries.

<a id="sec-algorithm"></a>

## 2. Algorithm / Architecture

As [Figure 1](#fig-model-workflow) shows, FLAN-T5-base reads an instruction followed by a radiology report and generates a summary ([Raffel et al. 2020](#ref-t5); [Chung et al. 2024](#ref-flan)). The encoder uses self-attention to relate parts of the input. The decoder uses causal self-attention over earlier summary tokens and cross-attention to the encoded input. In these experiments, greedy decoding selects the most likely next token and feeds it back to the decoder until an end marker or the length limit is reached.

During training, the decoder receives earlier reference-summary tokens and learns to predict the next token; cross-entropy loss measures prediction error and guides weight updates. Full fine-tuning updates all 247,577,856 base parameters. LoRA keeps these weights frozen and trains 884,736 parameters in pairs of small matrices. Their products form updates of rank at most 8 to the query and value projections in all encoder and decoder attention modules, including cross-attention ([Hu et al. 2022](#ref-lora)).

The [pinned checkpoint configuration](https://huggingface.co/google/flan-t5-base/blob/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/config.json) is summarised below. Its gated-GELU feed-forward layers and separate output weights follow the [T5 v1.1 architecture](https://huggingface.co/google/t5-v1_1-base).

| Component | FLAN-T5-base configuration |
|:---|:---|
| Layers | 12 encoder layers and 12 decoder layers |
| Hidden size and attention | 768 hidden units; 12 heads, each of size 64 |
| Feed-forward layers | 2,048 units; gated-GELU activation |
| Embeddings and output | Shared encoder/decoder input embeddings; separate output weights |
| Position information | Relative-position biases with 32 buckets |
| Model vocabulary | 32,128 embedding/output entries, [padded to a multiple of 128](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin) |

<a id="fig-model-workflow"></a>

![Encoder–decoder workflow with self-attention, cross-attention, LoRA query/value targets, a generated-token feedback loop, and separate reference-token paths for training and loss evaluation.](figures/model-workflow.png)

**Figure 1.** Generated tokens feed back into the decoder during generation. Dashed paths use reference summaries for teacher forcing and loss evaluation. In LoRA mode, q/v labels mark the adapted query and value projections.

<a id="sec-feasibility"></a>

## 3. Feasibility Review

This records the initial pilot and plan; Sections 7–10 report the final experiment.

**User Need, Scope & Acceptance Criteria**

The prototype helps clinicians explain radiology reports to patients. The challenge is simpler language without losing or inventing findings. Proposed criteria:

1.  Improve all four validation ROUGE scores over the pretrained baseline.

2.  Reduce hallucination and critical-omission rates and improve readability over baseline, using a fixed rubric on 100 validation examples sampled uniformly without replacement (seed 42).

3.  Complete each epoch, including validation, within 60 minutes and 32 GB peak reserved GPU memory on an A100 (decimal GB; approximately 29.80 GiB).

**Model Choice & Course Concepts**

BioLaySumm LaymanRRG provides 150,454 training, 10,000 validation and 10,537 unlabelled test examples. I chose [FLAN-T5-base](https://arxiv.org/abs/2210.11416) (**Hard** difficulty) for instruction-based rewriting, applying encoder–decoder attention and transfer learning. The experiment compares [LoRA](https://arxiv.org/abs/2106.09685) (884,736 trainable parameters) with full fine-tuning (247,577,856).

**Preliminary Feasibility Evidence**

The pilot trained each method for one epoch on the full training split using Seq2SeqTrainer and an A100 80GB, then evaluated both and the baseline on all 10,000 validation examples.

The final experiment used two custom-loop epochs, an A100 40GB, and different prompts, batches and learning-rate schedules (Sections [7](#sec-experiments)–[8](#sec-results)). Both used greedy decoding, but their scores are not directly comparable.

Pilot ROUGE uses 0–100 and memory uses decimal GB; final results use 0–1 and GiB. Pilot timing includes training, validation loss and local checkpoints, excluding generation/ROUGE, initial loading and Drive export. Final evaluation includes generation/ROUGE (Section [7.2](#scoring-and-measurement)).

| Method      | Val. loss |   R-1 |   R-2 |   R-L | R-Lsum | Time (min) | VRAM (GB) |
|:------------|----------:|------:|------:|------:|-------:|-----------:|----------:|
| Baseline    |     2.307 | 21.46 |  7.43 | 18.56 |  19.41 |          — |         — |
| LoRA        |     0.903 | 68.51 | 47.96 | 62.84 |  65.26 |       59.9 |     12.72 |
| Full tuning |     0.710 | 71.80 | 52.69 | 66.47 |  68.66 |       50.3 |     16.50 |

<a id="fig-feasibility-pilot"></a>

![Training loss for LoRA and full fine-tuning (left); validation ROUGE for all three methods (right).](figures/comparison_1x2_2026-10-29.png)

**Figure 2.** Training loss for LoRA and full fine-tuning (left); validation ROUGE for all three methods (right).

Both methods improve ROUGE and meet pilot resource limits under this timing scope. Full fine-tuning scores higher; LoRA uses less memory. Fidelity and readability require separate review.

**Risks, Budget & Fallback**

**Risk:** Report text in 4,506 validation rows (45.06%) matches training after ignoring case and whitespace, risking inflated scores. I retained official splits following staff guidance. Section [9.3](#sec-sentence-overlap)'s 23.80% instead requires matching report–summary pairs, also applying Unicode NFKC normalisation.

**Budget and next step:** I proposed **10 additional A100 GPU-hours** for two epochs per method with custom training/evaluation, generation and subset scoring. Controlled overlap-removal retraining remains future work (Section [10.1](#next-engineering-steps)).

**Fallback:** If reliability remained unproven, report performance, errors and resource use without claiming readiness for patient use.

<a id="sec-examples"></a>

## 4. Example Inputs and Outputs

Validation row 4333 (zero-based) is a selected illustration from the [100-report assessment sample](analysis/validation_sample_100_seed42.json) (seed 42). The overlap audit found no exact normalised sentence matches for this report in the training reports. Both trained outputs use the weights saved after epoch 2.

**Original radiology report:** Bilateral alveolar involvement, predominantly in the left, with no significant changes compared to the previous study dated 22 03 2011, related to bilateral bronchopneumonia.

**Layman report (reference):** Both lungs have some involvement, especially the left one, and it hasn't changed much compared to the previous study from March 22, 2011. This is related to an infection in both lungs called bronchopneumonia.

| Checkpoint | Prediction |
|:---|:---|
| Baseline | This report summarizes the present findings of a prospective study of bilateral bronchopneumonia. |
| LoRA, epoch 2 | Both lungs show some involvement, mainly on the left side, and there are no major differences compared to the previous study from March 22, 2011, related to a lung infection called bronchopneumonia. |
| Full, epoch 2 | Both lungs show signs of involvement, mainly on the left side, and there are no significant changes compared to the previous study from March 22, 2011. This is related to a type of pneumonia called bronchopneumonia affecting both lungs. |

Both trained models preserve involvement of both lungs, greater involvement on the left and no significant change compared with 22 March 2011. The baseline retains the diagnosis but invents a “prospective study” and omits the left predominance and dated comparison.

All three AI assessors flagged no hallucinations or critical omissions in either trained output. Section [9.1](#sec-quality-review) describes the assessment and its limitations; Section [9.2](#sec-error-autopsies) presents failure cases.

<a id="sec-dataset"></a>

## 5. Dataset and Preprocessing

<a id="data-source-and-split-protocol"></a>

### 5.1. Data source and split protocol

The project uses the BioLaySumm 2025 LaymanRRG open-source track ([BioLaySumm 2025](#ref-dataset); [Xiao et al. 2025](#ref-biolaysumm)). Training comprises 77.66% PadChest, 20.85% BIMCV-COVID19 and 1.49% Open-i. Reports from the first two were translated into English, accounting for 98.51% of training rows; performance on originally English reports needs separate evaluation.

The references were generated using [Zhao et al. (2025)](#ref-laymanrrg)'s GPT-4o method with semantic checks and self-refinement. The paper reports human checks of 500 sentence pairs, which do not establish comprehensive clinical validation of this dataset's references.

Dataset version `6783ab38f9de28c4372799346f51d3927ae7eb9a` uses the official splits, retained following staff guidance.

| Split | Rows | Role |
|:---|---:|:---|
| Training | 150,454 | All pairs used to train both methods. |
| Validation | 10,000 | Evaluate the baseline and each saved checkpoint. |
| Test | 10,537 | Inputs only; no references, not scored. |

Example: training row 2 (zero-based).

| Field | Text |
|:---|:---|
| `radiology_report` | The cardiomediastinal silhouette and pulmonary parenchyma show no significant pathological findings. |
| `layman_report` | The heart and the area around it, as well as the lung tissue, do not show any significant problems or abnormalities. |

<a id="prompting-and-tokenisation"></a>

### 5.2. Prompting and tokenisation

The pinned FLAN-T5 tokenizer converts text to tokens. Each input uses this instruction, a newline, then the report:

> `Simplify this radiology report for a layperson:`  
> `{report}`

Inputs, including the instruction and special tokens, are capped at 512 tokens; reference targets for loss calculation, including the end-of-sequence token, at 256. Longer texts are truncated. No additional text cleaning or data augmentation is applied, so formatting such as ambiguous dates (`12 02 2015`) is retained.

Word counts split on whitespace; truncation counts use the final prompt and pinned tokenizer.

| Split | Mean words (report / reference) | Inputs truncated | Loss targets truncated |
|:---|---:|---:|---:|
| Training | 23.28 / 38.56 | 324 (0.22%) | 2,042 (1.36%) |
| Validation | 23.32 / 38.60 | 11 (0.11%) | 127 (1.27%) |

Each batch pads reports and references separately to their longest sequence. An attention mask identifies input padding; reference padding is set to $-100$ so loss ignores it. ROUGE uses the complete references.

<a id="sec-usage"></a>

## 6. Implementation and Usage

### 6.1. Code organisation

The implementation uses explicit PyTorch training and evaluation loops. Hugging Face supplies the pretrained weights, tokenizer, padding tools and LoRA adapters; the final experiment does not use the Hugging Face `Trainer` API ([PyTorch](#ref-pytorch), [Transformers](#ref-transformers), [PEFT](#ref-peft)).

| File | Responsibility |
|---|---|
| [modules.py](modules.py) | Load FLAN-T5, attach LoRA and identify trainable parameters. |
| [dataset.py](dataset.py) | Load the official splits, tokenise reports/targets and collate batches. |
| [train.py](train.py) | Train, evaluate, save checkpoints and record losses, metrics and resources. |
| [predict.py](predict.py) | Load saved weights and generate single, batch or file predictions. |
| [utils.py](utils.py) | Handle seeds, device selection, timing, memory measurements and logging. |

### 6.2. Requirements and installation

The recorded experiment used Python **3.13.16**, PyTorch **2.11.0+cu130**, CUDA **13.0** and an **NVIDIA A100-SXM4-40GB**. The exact package pins in [requirements.txt](requirements.txt) are:

| Group | Package | Version |
|:---|:---|:---|
| **Model training and inference** | `torch` | 2.11.0 |
| | `transformers` | 4.56.2 |
| | `peft` | 0.17.1 |
| | `accelerate` | 1.10.1 |
| **Tokenisation and file handling** | `tokenizers` | 0.22.2 |
| | `sentencepiece` | 0.2.1 |
| | `huggingface-hub` | 0.36.2 |
| | `safetensors` | 0.8.0 |
| **Data loading and analysis** | `datasets` | 4.1.1 |
| | `pyarrow` | 23.0.1 |
| | `pandas` | 2.2.3 |
| | `numpy` | 2.1.3 |
| **Evaluation and plotting** | `rouge-score` | 0.1.2 |
| | `matplotlib` | 3.10.0 |
| **Configuration and monitoring** | `PyYAML` | 6.0.3 |
| | `psutil` | 5.9.5 |

Run the following from the repository root to enter the implementation directory. All later commands and Python examples use that directory as their working directory.

```bash
cd recognition/FLAN-T5_46375489
```

Use a fresh Python 3.13 environment. For the recorded Linux/CUDA 13.0 setup, install PyTorch first, then the remaining packages:

```bash
python -m pip install torch==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
```

For other hardware, use the corresponding [PyTorch 2.11.0 build](https://pytorch.org/get-started/previous-versions/). Initial model and dataset loading requires internet access unless the files are already cached.

### 6.3. Train the model

**Full fine-tuning is the first choice.** This command uses the main experiment's training and generation hyperparameters:

```bash
python train.py --config configs/full.yaml \
  --epochs 2 --batch-size 16 --eval-batch-size 32 \
  --num-beams 1 --bf16 --device cuda \
  --save-every-epoch --output-dir outputs/full
```

Use LoRA only if full fine-tuning cannot meet the resource limits and a LoRA trial can. Replace `configs/full.yaml` with `configs/lora.yaml` and `outputs/full` with `outputs/lora`; keep the other overrides. The YAML files alone default to five epochs, batch size four and four beams, so these overrides matter. Similar measured training times mean LoRA should mainly be considered for its memory and checkpoint-storage savings.

The configurable options below use the shared experiment settings; the LoRA-specific values apply only when LoRA is selected. Command-line arguments override YAML values. YAML keys use underscores instead of hyphens and omit the leading `--`.

| Option | Example value | Controls |
|:---|:---|:---|
| `--epochs` | 2 | Passes through the training data. |
| `--batch-size` | 16 | Reports per training batch. |
| `--learning-rate` | `2e-4` | Learning rate for weight updates. |
| `--weight-decay` | 0.01 | Weight shrinkage during updates. |
| `--gradient-accumulation-steps` | 1 | Batches combined before a weight update. |
| `--max-grad-norm` | 1.0 | Gradient clipping limit. |
| `--lora-r` | 8 | Rank of the LoRA adapter matrices. |
| `--lora-alpha` | 16 | LoRA scaling, applied as $\alpha/r$. |
| `--lora-dropout` | 0.05 | LoRA dropout probability. |
| `--max-input-length` | 512 | Maximum input tokens, including the prompt. |
| `--max-target-length` | 256 | Maximum reference tokens used for loss. |
| `--max-new-tokens` | 256 | Maximum generated tokens. |
| `--num-beams` | 1 | Decoding beams; 1 gives greedy generation. |

Related controls are `--eval-batch-size 32`, `--seed 42` and `--bf16`; `--no-bf16` disables BF16. For a small pipeline check before a full run:

```bash
python train.py --config configs/smoke.yaml
```

This runs one epoch on 16 training and eight validation examples, using batch size two and shorter token limits. It checks the pipeline; resource use on this tiny run does not establish that the full workload will fit.

With `--save-every-epoch`, the two-epoch run writes `outputs/full/epoch-01`, `epoch-02` and `best_checkpoint.json`. The JSON file identifies the checkpoint with the lowest validation loss. Checkpoints include weights or adapters, the tokenizer, `config_run.json` and `training_state.pt`. Logs and plots include `history.json`, `step_history.csv`, `resources.json`, `plots/losses.png` and `plots/metrics.png`. Checkpoints can be loaded for prediction; automatic training resume is not implemented.

Keep datasets and saved model weights out of Git; retain code, configuration and report figures in the submission.

This command demonstrates ordinary usage. Reproducing the exact recorded experiment also requires the revision pins, controlled batch order and evaluation procedure in Section [7](#sec-experiments).

### 6.4. Generate summaries from saved weights

`predict.py` exposes a Python API. For a full epoch-2 checkpoint:

```python
import torch
from predict import RadiologyPredictor

predictor = RadiologyPredictor.from_checkpoint(
    "outputs/full/epoch-02",
    fine_tuning_mode="full",
    device="cuda",
    batch_size=32,
    max_input_length=512,
    max_new_tokens=256,
    num_beams=1,
)
with torch.autocast("cuda", dtype=torch.bfloat16):
    print(predictor.predict("Chronic pulmonary changes"))
```

For LoRA, use `outputs/lora/epoch-02`, `fine_tuning_mode="lora"` and the same base-model revision used during training. `from_checkpoint()` does not automatically restore inference settings from `config_run.json`, so pass the intended settings explicitly.

`predict_batch()` accepts a list of report strings. `predict_file()` reads a JSON list such as:

```json
[{"radiology_report": "Chronic pulmonary changes"}]
```

It writes corresponding `layman_report` summaries and a separate `<output-stem>_metrics.json` measurements file. Section [4](#sec-examples) shows saved inputs and outputs from the recorded experiment.

### 6.5. Evaluate a checkpoint or the pretrained baseline

Load the checkpoint configuration, prepare the data and call `evaluate()`:

```python
import json
from pathlib import Path
from dataset import RadiologyData
from predict import RadiologyPredictor
from train import RadiologyTrainer

checkpoint = Path("outputs/full/epoch-02")
config = json.loads((checkpoint / "config_run.json").read_text())
config["training"]["device"] = "cuda"

predictor = RadiologyPredictor.from_checkpoint(
    checkpoint,
    model_name=config["model"]["model_name"],
    fine_tuning_mode=config["model"]["fine_tuning_mode"],
    model_revision=config["model"].get("model_revision"),
    device="cuda",
)
data = RadiologyData(predictor.tokenizer, config["data"])
data.load()
metrics = RadiologyTrainer(predictor.model, data, config).evaluate()
print(metrics)
```

`evaluate()` scores the validation split and returns `loss`, four ROUGE scores and generation measurements. The ordinary data loader also prepares the official test sources without target labels; those rows are not used for scored evaluation. This API does not itself save all prediction texts or perform the experiment harness's independent-checkpoint resource profiling.

The **implemented baseline** uses pretrained `google/flan-t5-base` without adapters or task-specific weight updates. Load it in full mode, prepare the same validation data and call `evaluate()` **before any `fit()` call**. This separates the effect of fine-tuning from the pretrained model's existing ability.

<a id="sec-experiments"></a>

## 7. Experimental Design and Reproducibility

<a id="controlled-comparison"></a>

### 7.1. Controlled comparison

Pretrained FLAN-T5-base, LoRA and full fine-tuning were compared in separate sessions using the same GPU model (Table [1](#tab-experiment-settings)). Both training methods completed two passes (epochs) through 150,454 training reports. Each epoch had 9,404 updates, including a final batch of six reports, giving 18,808 updates per method overall. The baseline and four saved models were evaluated on the same 10,000 validation reports in their original order.

All runs used the model and dataset revisions in Table [1](#tab-experiment-settings), with identical prompts and generation settings. Seed 42 controlled randomness. A separate random generator, seeded with $42+\text{epoch}$, gave both training methods the same shuffled batches in each epoch.

Before checkpoint evaluation, the experiment scripts moved the training model and optimiser state to CPU memory, then reloaded the saved weights. LoRA adapters remained separate from the base weights. The checkpoint with the lowest validation loss was selected as best.

<a id="scoring-and-measurement"></a>

### 7.2. Scoring and measurement

**Scores.** Validation loss and ROUGE were calculated on all 10,000 validation reports, once for the pretrained baseline and after each LoRA/full-tuning epoch, with weights fixed.

- **Validation loss:** Total cross-entropy divided by the number of non-padding reference tokens across the split (up to 256 per summary), using the report and earlier reference tokens. Lower is better.
- **ROUGE:** Mean per-report ROUGE-1/2/L/Lsum F1 scores (0–1), measuring wording overlap between summaries generated without references and the complete references; higher is better ([Lin 2004](#ref-rouge)). Scoring uses `rouge-score` 0.1.2 with lowercasing, punctuation removal and word stemming. For ROUGE-Lsum, sentences are split at existing newlines and after `.`, `!` or `?` followed by whitespace.

**Time.** Training time excludes initial loading, evaluation and checkpoint operations. Total evaluation time includes generation, text processing, data transfer, loss calculation, scoring and output writing, but excludes model and dataset loading.

**Generation speed.** Tokens per second are measured during generation only. Counts include the end-of-sequence token (EOS), but exclude the decoder-start token and later padding. Training and generation timers wait for GPU work to finish before starting and stopping.

**Memory.** Peak GPU memory is reported in GiB: allocated memory holds active tensors; reserved memory also includes PyTorch’s cached space.

### 7.3. Experiment settings

<a id="tab-experiment-settings"></a>

**Table 1.** Experiment settings. LoRA settings apply only to the adapter-based runs.

| Setting | Value |
|:---|:---|
| Model revision | `google/flan-t5-base`: `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2` |
| Dataset revision | `6783ab38f9de28c4372799346f51d3927ae7eb9a` |
| GPU | NVIDIA A100-SXM4-40GB |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| Training | 2 epochs; batch size 16; gradient accumulation 1 |
| Optimisation | AdamW ([Loshchilov and Hutter 2019](#ref-adamw)); constant learning rate $2\times10^{-4}$ with no warmup or schedule; weight decay 0.01; gradient-norm limit 1.0 |
| LoRA | Rank 8; $\alpha=16$; dropout 0.05; attention query/value projections |
| Lengths | Prompted input at most 512 tokens; loss targets at most 256 tokens |
| Generation | At most 256 new tokens; greedy decoding; one beam; no sampling |
| Evaluation | Batch size 32 |
| Random seed | 42 |
| Precision | BF16 autocast |

<a id="sec-results"></a>

## 8. Quantitative Results and Resource Profiling

<a id="validation-performance"></a>

### 8.1. Validation performance

Both training methods improve all four ROUGE scores over the baseline (Figure [3](#fig-rouge-comparison); Table [2](#tab-validation-results)). At epoch 2, full fine-tuning reaches ROUGE-Lsum 0.7241, compared with 0.6788 for LoRA and 0.1625 for the baseline. Full fine-tuning scores 0.0454 higher than LoRA on the 0–1 scale, calculated from unrounded scores. Each evaluation uses the same 10,000 validation examples in the same order.

<a id="tab-validation-results"></a>

**Table 2.** ROUGE F1 scores averaged across examples, and validation loss averaged across reference tokens. All five evaluations use the complete official validation split.

| Model/checkpoint |    R-1 |    R-2 |    R-L | R-Lsum | Val. loss |
|:-----------------|-------:|-------:|-------:|-------:|----------:|
| Baseline         | 0.1779 | 0.0520 | 0.1535 | 0.1625 |    2.3083 |
| LoRA, epoch 1    | 0.6962 | 0.4945 | 0.6418 | 0.6645 |    0.8492 |
| LoRA, epoch 2    | 0.7099 | 0.5164 | 0.6566 | 0.6788 |    0.7676 |
| Full, epoch 1    | 0.7470 | 0.5711 | 0.6998 | 0.7188 |    0.5150 |
| Full, epoch 2    | 0.7520 | 0.5760 | 0.7054 | 0.7241 |    0.4729 |

<a id="fig-rouge-comparison"></a>

![Figure 3. All four ROUGE scores for the baseline and four saved models, evaluated on the same 10,000 validation reports.](figures/rouge_comparison.png)

**Figure 3.** All four ROUGE scores for the baseline and four saved models, evaluated on the same 10,000 validation reports.

The second epoch increases ROUGE-Lsum by 0.0143 for LoRA and 0.0053 for full fine-tuning. Both validation losses also fall. The final comparison uses epoch 2.

**Uncertainty.** Paired percentile bootstrap intervals use 10,000 resamples of the same 10,000 validation rows across checkpoints (seed 42).

| ROUGE-Lsum comparison | Mean gain | 95% interval |
|:---|---:|---:|
| Full epoch 2 − LoRA epoch 2 | 0.04535 | 0.04274–0.04802 |
| Full epoch 2 − Full epoch 1 | 0.00533 | 0.00310–0.00753 |

Both intervals exclude zero. They assume independent rows and fixed checkpoints; they do not account for repeated-report or patient clusters, training-seed variation, or bias from validation reuse and training overlap.

<a id="training-behaviour-and-cost"></a>

### 8.2. Training behaviour and cost

Average training loss falls from 1.2317 to 1.0012 for LoRA and from 0.7045 to 0.5393 for full fine-tuning. Loss falls most quickly early in training (Figure [4](#fig-training-history)). Training loss is measured while dropout is active and the model’s weights are still being updated. Validation loss is measured with dropout turned off and the saved weights held fixed. These different conditions help explain why the two losses are at different levels. Training overlap (Section [9.3](#sec-sentence-overlap)) may also affect validation loss; its effect was not measured.

<a id="fig-training-history"></a>

![Training and validation loss for LoRA and full fine-tuning, with the shared pretrained baseline at epoch zero.](figures/training_history.png)

**Figure 4.** Training loss at each update, its token-weighted trailing mean over up to 200 updates, and validation loss at each saved epoch. The epoch-zero markers show the separately evaluated pretrained baseline (2.3083), shared as a reference for both methods.

LoRA updates 884,736 parameters, 99.64% fewer than full fine-tuning, and reduces peak allocated training memory by 18.6%. Freezing weights reduces gradient and optimiser storage, but both methods still require the base model and training activations. Their training times are similar: 58.18 minutes for LoRA and 58.01 minutes for full fine-tuning over two epochs.

<a id="tab-training-resources"></a>

**Table 3.** Parameter counts, total training time over two epochs, and the highest allocated and reserved GPU memory recorded during training. Allocated memory holds tensors in use; reserved memory also includes cached blocks. All timings were recorded on an A100-SXM4-40GB.

| Model    | Total (M) | Updated (M) | Train min | Alloc. GiB | Reserv. GiB |
|:---------|----------:|------------:|----------:|-----------:|------------:|
| Baseline |   247.578 |       0.000 |         – |          – |           – |
| LoRA     |   248.463 |       0.885 |     58.18 |      18.71 |       23.37 |
| Full     |   247.578 |     247.578 |     58.01 |      22.98 |       26.52 |

**Resource criterion.** Recorded training plus validation time is below 60 minutes in every epoch:

| Method | Epoch 1 (min) | Epoch 2 (min) |
|:---|---:|---:|
| LoRA | 57.79 | 59.10 |
| Full | 53.01 | 51.80 |

Peak reserved memory is at most 26.52 GiB (28.48 GB), below the 32 GB limit. Both methods meet criterion 3 under this measured timing scope, with only 0.90 minutes of headroom for LoRA's second epoch. These sums exclude loading, checkpoint operations and other experiment-harness overhead.

**Checkpoint storage.** Saved epoch-2 training checkpoints, including optimiser state, occupy 14.03 MB for LoRA and 2.97 GB for full fine-tuning. Adapter/model weights alone occupy 3.56 MB and 990.35 MB, respectively (decimal units). LoRA's checkpoint excludes the frozen base weights, which are still required for inference.

<a id="inference"></a>

### 8.3. Inference

The final fully fine-tuned model generates 374.26 tokens/s, compared with 279.69 for LoRA with its adapters kept separate from the base model (unmerged LoRA) (Table [4](#tab-inference-resources)). The baseline generates fewer tokens per report and finishes its evaluation sooner. Total evaluation time averages 106.07, 179.30 and 138.19 ms per report for the baseline, LoRA and full fine-tuning. These averages come from evaluating reports in batches of 32 and dividing the total time by 10,000 reports.

Tokens/s also depends on output lengths and their variation within batches, as well as unmerged adapter computation. The individual effects were not isolated in these measurements.

<a id="tab-inference-resources"></a>

**Table 4.** Summary generation by the final saved models over 10,000 validation reports. Generation speed uses timing that waits for GPU work to finish. Total evaluation time also includes calculating loss, scoring summaries and writing outputs. Peak memory lists allocated memory followed by reserved memory, in GiB.

| Model | Generate s | Evaluate s | Average evaluation ms/report | Tokens/s | Mean tokens | Peak GiB (allocated / reserved) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1010.39 | 1060.72 | 106.07 | 271.89 | 27.47 | 3.81 / 5.62 |
| LoRA | 1729.58 | 1793.04 | 179.30 | 279.69 | 48.38 | 3.82 / 5.68 |
| Full | 1320.07 | 1381.87 | 138.19 | 374.26 | 49.40 | 3.82 / 5.63 |

The average milliseconds per report are amortised over batches of 32 and include the full evaluation workload. They are not single-request latency measurements.

<a id="sec-safety"></a>

## 9. Clinical Fidelity, Readability and Data Overlap

<a id="sec-quality-review"></a>

### 9.1. Assessment of the second acceptance criterion

The planned review randomly selected 100 of the 10,000 validation reports using seed 42. Every report had the same chance of selection, and none was selected twice. Three AI assessors reviewed the predictions from all five checkpoints for these reports, giving 500 evaluations per assessor. These were ChatGPT (Pro effort) ([ChatGPT 2026](#ref-qualityreview)), Gemini 4 Argon ([Gemini 2026](#ref-geminiqualityreview)), and Claude Opus 5.5 (High effort) ([Claude 2026](#ref-claudequalityreview)). Each gave scores and brief reasons for every complete prediction. Their recorded rating criteria are summarised below. The original radiology report is the basis for checking the facts; the reference provides additional guidance on plain wording. If they disagree, the original report takes priority.

- **Hallucination (H, 0/1).** Score 0 when the original report supports all factual claims; score 1 when at least one clinical claim is unsupported, made up or contradicts the report. Checks cover diagnoses, procedures, medical history, location, side of the body, measurements, whether a finding is present or absent, and how certain a diagnosis is. Accurate rewording and generic or repetitive wording alone are not hallucinations.

- **Critical omission (O, 0/1).** Score 0 when all medically essential information is kept; score 1 when an important finding, detail that qualifies a finding, or recommendation is missing or not properly communicated. Checks cover abnormalities, important findings reported as absent, location and side of the body, changes over time, uncertainty, follow-up, measurements and device position. Repeated information, writing style and medically insignificant details are excluded.

The two error flags are scored separately: both may be 1, but one does not automatically imply the other. **Readability (R, 1–10)** is a whole-number score for how easily an adult without medical training can understand the text. It is scored separately from accuracy and completeness. It considers whether medical terms are explained, how clear and well organised the text is, grammar, repetition, cut-off text and whether the length is appropriate.

| Readability score | Interpretation |
|:---|:---|
| 1–2 | Incoherent, severely repetitive or largely unintelligible. |
| 3–4 | Difficult to understand; substantial jargon or confusing language. |
| 5–6 | Moderately understandable; some unexplained terminology or awkward phrasing. |
| 7–8 | Generally clear and accessible, with minor readability issues. |
| 9–10 | Exceptionally clear, natural, concise and understandable without medical knowledge. |

<a id="tab-quality-review"></a>

**Table 5.** Each AI assessor reviewed the same 100 validation reports for each checkpoint. Error rates are the percentage of outputs with at least one flagged error. Readability is the mean $\pm$ sample standard deviation on a 1–10 scale. These statistics were recalculated from each assessor’s scores.

| AI assessor | Model/checkpoint | Hallucinations | Critical omissions | Readability |
|:---|:---|---:|---:|---:|
| ChatGPT | Baseline | 38% | 71% | $5.14 \pm 2.16$ |
| ChatGPT | LoRA, epoch 1 | 17% | 15% | $8.06 \pm 1.92$ |
| ChatGPT | LoRA, epoch 2 | 17% | 13% | $8.27 \pm 1.72$ |
| ChatGPT | Full, epoch 1 | 16% | 7% | $8.67 \pm 1.52$ |
| ChatGPT | Full, epoch 2 | 9% | 9% | $8.83 \pm 1.30$ |
| Gemini 4 Argon | Baseline | 39% | 76% | $4.52 \pm 1.42$ |
| Gemini 4 Argon | LoRA, epoch 1 | 16% | 24% | $8.51 \pm 1.54$ |
| Gemini 4 Argon | LoRA, epoch 2 | 13% | 25% | $8.76 \pm 1.30$ |
| Gemini 4 Argon | Full, epoch 1 | 9% | 11% | $9.19 \pm 1.07$ |
| Gemini 4 Argon | Full, epoch 2 | 4% | 14% | $9.36 \pm 0.88$ |
| Claude Opus 5.5 | Baseline | 39% | 69% | $3.93 \pm 1.35$ |
| Claude Opus 5.5 | LoRA, epoch 1 | 19% | 6% | $7.63 \pm 1.93$ |
| Claude Opus 5.5 | LoRA, epoch 2 | 14% | 4% | $7.79 \pm 1.79$ |
| Claude Opus 5.5 | Full, epoch 1 | 11% | 4% | $8.30 \pm 1.46$ |
| Claude Opus 5.5 | Full, epoch 2 | 7% | 5% | $8.58 \pm 1.16$ |

All three assessors give all four fine-tuned checkpoints lower error rates and higher average readability than the baseline. Each AI assessment therefore supports the second acceptance criterion. Full tuning at epoch 2 has the lowest observed hallucination rate and highest mean readability for every assessor. Full epoch 1 has the fewest observed omissions for ChatGPT and Gemini, and ties with LoRA epoch 2 for Claude.

**Assessor agreement.** Pairwise Cohen's $\kappa$ (agreement adjusted for chance) ranges from 0.65–0.84 for hallucinations and 0.34–0.53 for omissions. Omission judgments are less consistent: LoRA epoch 2 rates range from 4% to 25%. These agreement statistics cover 400 outputs from the same 100 reports across four fine-tuned checkpoints; they do not establish clinical accuracy.

**Sampling uncertainty.** With 100 reports, Wilson 95% intervals are 4.81–16.23% for ChatGPT's hallucination rate of 9% at full epoch 2 and 17.55–34.30% for Gemini's omission rate of 25% at LoRA epoch 2. These intervals describe sampling uncertainty conditional on the assessor's judgments. Exploratory exact McNemar tests found no significant difference between full epochs 1 and 2 in either error flag for any assessor (all unadjusted $p>0.05$); this does not establish equivalence.

**Voting patterns.** For full epoch 2, counts out of the same 100 outputs are:

| Flag | Any assessor | Majority (at least two) | All three |
|:---|---:|---:|---:|
| Hallucination | 11 | 6 | 3 |
| Critical omission | 17 | 6 | 5 |

These counts summarise disagreement, not bounds on the true error rate: assessors may share missed errors or incorrect flags. Their ratings are not independent samples.

**Assessment limitations.** Assessors knew which checkpoint produced each output. ChatGPT's model version was not recorded, limiting reproducibility. AI assessor judgments are unvalidated and require case-by-case checking against the source reports by clinicians; readability also needs checking with readers without medical training.

<a id="sec-error-autopsies"></a>

### 9.2. Qualitative error autopsies

These four cases from the 100-report assessment illustrate hallucinations, critical omissions and readability trade-offs. Each shows the original report, reference and complete epoch-2 predictions. H/O/R records ChatGPT's hallucination flag, critical omission flag and readability score; 1 flags an error, and readability ranges from 1 to 10. Agreement and disagreement with the other assessors are stated below each case. Row indices start at zero.

<a id="hallucinations"></a>

#### 9.2.1. Hallucinations

**Row 6065: Normal findings are changed during simplification.**

**Original radiology report:** No consolidations are identified in the lung parenchyma. The cardiomediastinal silhouette and **pulmonary hila** are within normal limits. No other significant findings are noted.

**Layman report (reference):** No signs of pneumonia are found in the lung tissue. The heart and the area around it, as well as the **lung’s main structures**, are all normal. There are no other important issues observed.

| Checkpoint | H/O/R (ChatGPT) | Prediction |
|:---|:--:|:---|
| LoRA, epoch 2 | 1/0/9 | No signs of lung infections are found in the lung tissue. The heart and middle chest area and the **lung bases** are normal. There are no other important issues found. |
| Full, epoch 2 | 1/0/9 | No signs of pneumonia are found in the lung tissue. The heart and the area around it, as well as the **lung bases**, are all normal. There are no other important issues noted. |

Both outputs replace normal pulmonary hila with normal lung bases. ChatGPT rated both H=1 despite R=9, because the source does not support this change of location. Claude also flagged hallucinations in both outputs; Gemini did not.

**Row 7573: Fluent wording changes a finding and adds reassurance.**

**Original radiology report:** **Lytic image** in the left clavicle **without radiological criteria for aggressiveness**. A complete CT study is recommended.

**Layman report (reference):** There's a lytic, or broken down, appearance in the left collarbone without any signs of aggressiveness on the x-ray. A full CT scan is recommended for further evaluation.

| Checkpoint | H/O/R (ChatGPT) | Prediction |
|:---|:--:|:---|
| LoRA, epoch 2 | 0/0/4 | There is a **lyte image** in the left collarbone, but it doesn't show any signs of aggressiveness. It's suggested to have a full CT scan to check it out. |
| Full, epoch 2 | 1/1/9 | There is a **swollen area** in the left collarbone that doesn't look like it's **causing any serious problems**. It's suggested to do a full CT scan. |

The full model replaces “lytic image” with “swollen area” and broadens a qualified imaging statement into reassurance about serious problems. All three assessors flagged hallucination in this output. ChatGPT also flagged omission; Gemini and Claude did not. LoRA's unclear term “lyte image” shows a separate readability failure.

<a id="omission-of-important-information"></a>

#### 9.2.2. Omission of important information

**Row 5155: Readable summaries omit redistribution of blood flow.**

**Original radiology report:** Cardiomegaly. Prominent hila of vascular origin with **vascular redistribution** and a mild interstitial pattern, without significant changes compared to the study from yesterday.

**Layman report (reference):** The heart is enlarged. The blood vessels near the lungs are prominent, indicating a **redistribution of blood flow** and a mild pattern of fluid in the lungs, which hasn’t changed much compared to the previous day’s examination.

| Checkpoint | H/O/R (ChatGPT) | Prediction |
|:---|:--:|:---|
| LoRA, epoch 2 | 0/1/8 | The heart is enlarged. The areas where the blood vessels enter the lungs are prominent, likely due to blood vessel issues. There is a mild pattern in the lung tissue, but it hasn’t changed much since the study done yesterday. |
| Full, epoch 2 | 1/1/8 | The heart is enlarged. The areas where the blood vessels enter the lungs are prominent, likely due to blood vessel issues, and there is a mild pattern of fluid in the lung tissues. This is similar to what was seen in the study from yesterday. |

Both outputs retain the enlarged heart and comparison with yesterday's study, but omit the reported redistribution of blood flow. All three assessors flagged omissions in both outputs. ChatGPT gave both R=8 and also flagged the full model's interpretation of the interstitial pattern as fluid (H=1); Gemini and Claude did not flag that hallucination.

**Row 916: An unchanged finding loses its defining detail.**

**Original radiology report:** Single-chamber pacemaker with sensor. Left pleural effusion is stable compared to the previous control dated 19 06 2012. **Superior mediastinal widening** shows no significant variation compared to the previous study dated 15 06 2012.

**Layman report (reference):** The patient has a single-chamber pacemaker with a sensor. The fluid buildup on the left side of the lungs has not changed much since the last check on June 19, 2012. The area around the heart and lungs also looks the same as it did in the study from June 15, 2012.

| Checkpoint | H/O/R (ChatGPT) | Prediction |
|:---|:--:|:---|
| LoRA, epoch 2 | 0/0/7 | A single-chamber pacemaker with a sensor is present. The fluid buildup around the left lung has remained the same since the last check on June 19, 2012. The area around the heart, called the superior mediastinal, is **wider than usual**, and there are no major differences compared to the previous study from June 15, 2012. |
| Full, epoch 2 | 0/1/9 | A single-chamber pacemaker with a sensor is present. The fluid buildup in the left chest cavity has not changed since the last check on June 19, 2012. The area around the heart and lungs **has not changed significantly** since the last study on June 15, 2012. |

The full model preserves the date and lack of change but omits the widening itself. All three assessors flagged this omission; none flagged either error type in LoRA's output, which retains “wider than usual”. ChatGPT rated the full output more readable (9 versus 7). The reference also omits the widening, illustrating why fidelity checks must use the source report.

<a id="sec-sentence-overlap"></a>

### 9.3. Training–validation text overlap

Data leakage can occur when training and validation contain the same examples, reworded copies, or related reports from the same patient or study. The saved text-overlap checks found:

| Matching condition | Validation reports (of 10,000) |
|:---|---:|
| Any shared source sentence | 8,940 (89.40%) |
| Shared source sentence of at least five words | 6,097 (60.97%) |
| Every distinct source sentence seen in training | 5,608 (56.08%) |
| Exact report–summary pair | 2,380 (23.80%) |

The categories overlap and must not be added. Sentence matching normalises Unicode (NFKC), case and spacing, and ignores terminal `.?!`; word counts use whitespace. Sentences may match different training reports, without matching references. Common radiology wording can produce sentence overlap without duplicate clinical cases.

For the exact-pair comparison, the complete source report and target summary must match the same training row after standardising character forms (Unicode NFKC), capitalisation and spacing. All wording, numbers and punctuation are kept. Table [6](#tab-exact-pair-overlap) compares the 2,380 matching pairs with the remaining 7,620 reports.

<a id="tab-exact-pair-overlap"></a>

**Table 6.** Mean ROUGE F1 (0–1) for **matching / remaining** validation reports. Scores use the original prediction and reference texts.

| Checkpoint   |             R-1 |             R-2 |             R-L |          R-Lsum |
|:-------------|----------------:|----------------:|----------------:|----------------:|
| Baseline     | 0.1352 / 0.1912 | 0.0303 / 0.0588 | 0.1263 / 0.1620 | 0.1265 / 0.1738 |
| LoRA epoch 1 | 0.8143 / 0.6593 | 0.6778 / 0.4372 | 0.7997 / 0.5925 | 0.8000 / 0.6222 |
| LoRA epoch 2 | 0.8271 / 0.6733 | 0.7036 / 0.4579 | 0.8139 / 0.6075 | 0.8142 / 0.6365 |
| Full epoch 1 | 0.8442 / 0.7166 | 0.7330 / 0.5206 | 0.8346 / 0.6578 | 0.8347 / 0.6826 |
| Full epoch 2 | 0.8425 / 0.7238 | 0.7165 / 0.5321 | 0.8329 / 0.6656 | 0.8330 / 0.6901 |

**Effect on scores.** Fine-tuned models score higher on exact pairs; the baseline scores lower. For full epoch 2, ROUGE-Lsum is 0.8330 on matches and 0.6901 on the remaining reports, giving an overall mean of 0.7241 (+0.0340 above the remaining reports). Matching reports are also much shorter: 4.7 versus 29.1 source words on average. This comparison cannot separate repetition from report length; controlled retraining (Section [10](#sec-recommendation)) is needed to distinguish their effects. Absence of an exact pair match does not establish an independent clinical case.

**Three matching examples.** Each pair has identical complete reports and reference summaries. Tables show all five checkpoints’ complete predictions. Row numbers start at zero.

1.  **Training row 3,677; validation row 247.**  
    **Radiology report (both):** Cardiomegaly.  
    **Layman reference (both):** The heart is enlarged.

    | Checkpoint   | Prediction                                                |
    |:-------------|:----------------------------------------------------------|
    | Baseline     | This radiology report is for the following: Cardiomegaly. |
    | LoRA epoch 1 | The heart is enlarged.                                    |
    | LoRA epoch 2 | The heart is enlarged.                                    |
    | Full epoch 1 | The heart appears enlarged.                               |
    | Full epoch 2 | The heart is enlarged.                                    |

2.  **Training row 48,706; validation row 3,032.**  
    **Radiology report (both):** Left pleural effusion.  
    **Layman reference (both):** There is fluid buildup on the left side of the chest.

    | Checkpoint   | Prediction                                                 |
    |:-------------|:-----------------------------------------------------------|
    | Baseline     | This radiology report describes the left pleural effusion. |
    | LoRA epoch 1 | There is fluid buildup on the left side of the chest.      |
    | LoRA epoch 2 | There is fluid buildup on the left side of the chest.      |
    | Full epoch 1 | There is fluid buildup on the left side of the chest.      |
    | Full epoch 2 | There is fluid buildup on the left side of the chest.      |

3.  **Training row 85,059; validation row 424.**  
    **Radiology report (both):** Scoliotic posture. No significant findings.  
    **Layman reference (both):** The person has a curved spine posture, but there are no significant issues found.

    | Checkpoint | Prediction |
    |:---|:---|
    | Baseline | This report summarizes the results of a scolioscopy examination of a scolioscopy patient. |
    | LoRA epoch 1 | The person is in a curved posture. There are no important issues found. |
    | LoRA epoch 2 | The person is bending forward. There are no important issues found. |
    | Full epoch 1 | The person has a curved spine posture. There are no important issues found. |
    | Full epoch 2 | The person has a curved spine posture. There are no important issues found. |

    **Reference variation for this report.** Across training and validation, this exact radiology report occurs in three rows with two distinct layman references. Training row 85,059 and validation row 424 share the reference shown above. Training row 34,971 instead has: “The person has a curved spine posture, also known as scoliosis. There are no significant issues found.” These are all matching locations in the two splits.

<a id="sec-recommendation"></a>

## 10. Engineering Recommendation and Trade-off Synthesis

<a id="recommendation"></a>

### 10.1. Engineering recommendation and required actions

Use full fine-tuning for further development, with full epoch 2 as the current prototype and epoch 1 retained for clinician comparison. Use LoRA only if full fine-tuning exceeds resource limits and a LoRA trial meets them.

The prototype is not ready for patient use. Full epoch 2 has 4–9% AI-flagged hallucinations and 5–14% critical omissions across assessors. These judgments require clinician validation before deployment.

<a id="next-engineering-steps"></a>

Prioritise the following work:

1.  **Check resources.** Define training and deployment hardware and resource budgets. Measure peak memory, complete runtime and storage with long inputs and targets; adjust training batch size and gradient accumulation if needed.

2.  **Audit overlap in stages.** First remove training pairs with exact validation matches and record removals. Retrain from pretrained weights for two epochs with fixed validation and settings, comparing against a control that removes the same number of training pairs at random. Then assess near-duplicates and longer shared passages before choosing broader filters.

3.  **Extend training.** After the controlled comparison, continue full fine-tuning for more epochs on the filtered data. Evaluate each epoch and select checkpoints using validation loss and summary fidelity to the source. Set a budget and early-stopping rule beforehand.

4.  **Validate clinical checks.** Have clinicians assess outputs against the same 100 source reports, with checkpoint labels hidden and output order randomised. Validate automated checks against their judgments and withhold flagged summaries from patient display.

5.  **Check patient understanding.** Test clinician-reviewed summaries with readers without medical training. Show source and summary together, flag truncated inputs and require clinician approval before patient use.

<a id="trade-off-synthesis"></a>

### 10.2. Trade-off synthesis

<a id="tab-trade-off-synthesis"></a>

**Table 7.** Measured trade-offs and their implications, based on Sections [8](#sec-results) and [9](#sec-safety).

| Trade-off | Engineering implication |
|:---|:---|
| Full fine-tuning achieves higher ROUGE, including on pairs without exact training matches, but uses more training memory. | Allow for higher peak memory before scaling batch size or input length. |
| LoRA training checkpoints occupy 14 MB versus 2.97 GB for full fine-tuning; training times are similar. | Useful when storing many checkpoints. Sizes include training state; LoRA still needs the frozen base weights. |
| Unmerged LoRA generates 280 versus 374 tokens/s for full fine-tuning in the recorded batched workload. | Benchmark merged adapters on the chosen target; merging was not evaluated. |
| Higher ROUGE and readability coexist with AI-flagged factual errors. | Include source fidelity in selection and require clinician approval before patient use. |
| Full epoch 2 improves ROUGE and observed hallucination rates; epoch 1 has 1–3 fewer omission flags per 100 reports. | Compare both with clinicians; these small gaps do not establish epoch 1 as superior. |

<a id="sec-ai-disclosure"></a>

## 11. Artificial Intelligence Usage Disclosure

I used AI assistants through OpenAI Codex, ChatGPT, Gemini, Claude, Claude Code and the Antigravity IDE to support the project. I set the scope, selected comparisons and directed revisions. I manually reviewed AI-generated outputs, sought clarification where needed, directed testing appropriate to each component and inspected the results. I also used AI from different providers to cross-check one another’s outputs.

- **Concepts and planning.** AI supported conceptual clarification, literature interpretation and experimental planning, including model selection, evaluation criteria and resource constraints.

- **Implementation and testing.** AI assisted with code development, refactoring and debugging. AI checked changes through separate-agent reviews, controlled CPU tests, before-and-after comparisons, checkpoint checks and small GPU test runs.

- **Experiments and analysis.** AI executed the training and evaluation runs, verified metrics and examples against saved results and dataset records, and analysed dataset overlap and model errors. ChatGPT, Gemini and Claude provided clinical-fidelity and readability scores (Section [9.1](#sec-quality-review)); automated checks verified score completeness and the reported summary calculations.

- **Writing and presentation.** AI assisted with report drafting and revision, including figures, tables, plain-language explanations, usage instructions and source attribution. I directed the organisation and presentation and requested corrections supported by project evidence.

<a id="sec-references"></a>

## 12. References and Attribution

Google provides the pretrained model weights and tokenizer ([Google 2022](#ref-flanmodel)). BioLaySumm provides the reports and reference summaries ([BioLaySumm 2025](#ref-dataset); [Xiao et al. 2025](#ref-biolaysumm)). PyTorch, Transformers and PEFT provide the underlying software libraries ([Paszke et al. 2019](#ref-pytorch); [Wolf et al. 2020](#ref-transformers); [Mangrulkar et al. 2022](#ref-peft)). The project adds the task settings, custom training and evaluation loops, measurement code and analysis. AI assistance is described in Section [11](#sec-ai-disclosure).

<a id="refs"></a>

<a id="ref-dataset"></a>

BioLaySumm. 2025. “BioLaySumm2025-LaymanRRG-opensource-track.” Hugging Face dataset repository. <https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track>.

<a id="ref-qualityreview"></a>

ChatGPT. 2026. “Evaluation Summary: Clinical Fidelity and Readability Across Five Checkpoints.” Shared AI rubric assessment. <https://chatgpt.com/share/6ae3bef4-1b18-83ec-abcd-f24c58895884>.

<a id="ref-flan"></a>

Chung, Hyung Won, Le Hou, Shayne Longpre, et al. 2024. “Scaling Instruction-Finetuned Language Models.” *Journal of Machine Learning Research* 25 (70): 1–53. <https://www.jmlr.org/papers/v25/23-0870.html>.

<a id="ref-claudequalityreview"></a>

Claude. 2026. “Claude Opus 5.5: Clinical Fidelity and Readability Assessment Across Five Checkpoints.” Shared AI assessment chat. <https://claude.ai/share/3afe55b7-b9a0-43e2-8a61-a37bea190f04>.

<a id="ref-geminiqualityreview"></a>

Gemini. 2026. “Gemini 4 Argon: Clinical Fidelity and Readability Assessment Across Five Checkpoints.” Shared AI assessment chat. <https://share.gemini.google/oiyo6aFaEsVp>.

<a id="ref-flanmodel"></a>

Google. 2022. “FLAN-T5-base: Pretrained Checkpoint and Model Card.” Hugging Face model repository. <https://huggingface.co/google/flan-t5-base>.

<a id="ref-lora"></a>

Hu, Edward J., Yelong Shen, Phillip Wallis, et al. 2022. “LoRA: Low-Rank Adaptation of Large Language Models.” In *International Conference on Learning Representations*. <https://arxiv.org/abs/2106.09685>.

<a id="ref-rouge"></a>

Lin, Chin-Yew. 2004. “ROUGE: A Package for Automatic Evaluation of Summaries.” In *Text Summarization Branches Out*, 74–81. Association for Computational Linguistics. <https://aclanthology.org/W04-1013/>.

<a id="ref-adamw"></a>

Loshchilov, Ilya, and Frank Hutter. 2019. “Decoupled Weight Decay Regularization.” In *International Conference on Learning Representations*. <https://arxiv.org/abs/1711.05101>.

<a id="ref-peft"></a>

Mangrulkar, Sourab, Sylvain Gugger, Lysandre Debut, et al. 2022. “PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning Methods.” Hugging Face software repository. <https://github.com/huggingface/peft>.

<a id="ref-pytorch"></a>

Paszke, Adam, Sam Gross, Francisco Massa, et al. 2019. “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” In *Advances in Neural Information Processing Systems*. Vol. 32. <https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html>.

<a id="ref-t5"></a>

Raffel, Colin, Noam Shazeer, Adam Roberts, et al. 2020. “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.” *Journal of Machine Learning Research* 21 (140): 1–67. <https://jmlr.org/papers/v21/20-074.html>.

<a id="ref-transformers"></a>

Wolf, Thomas, Lysandre Debut, Victor Sanh, et al. 2020. “Transformers: State-of-the-Art Natural Language Processing.” In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45. Association for Computational Linguistics. <https://doi.org/10.18653/v1/2020.emnlp-demos.6>.

<a id="ref-biolaysumm"></a>

Xiao, Chenghao, Kun Zhao, Xiao Wang, et al. 2025. “Overview of the BioLaySumm 2025 Shared Task on Lay Summarization of Biomedical Research Articles and Radiology Reports.” In *Proceedings of the 24th Workshop on Biomedical Language Processing*, 365–77. Association for Computational Linguistics. <https://doi.org/10.18653/v1/2025.bionlp-1.31>.

<a id="ref-laymanrrg"></a>

Zhao, Kun, Chenghao Xiao, Sixing Yan, et al. 2025. “X-ray Made Simple: Lay Radiology Report Generation and Robust Evaluation.” arXiv:2406.17911v6, revised 19 May 2025. <https://arxiv.org/abs/2406.17911v6>.
