# FLAN-T5 for Layperson Radiology Report Generation

COMP3710 (Hard difficulty), Junrong Tang (s4983489)

## Problem and Approach

This project uses FLAN-T5 to rewrite radiology reports in plain English for patients. The aim is roughly a sixth-grade reading level without losing clinical findings, anatomical details, negation or uncertainty. I compare a zero-shot baseline with full fine-tuning and LoRA fine-tuning using ROUGE scores, reading-grade estimates, resource measurements and a four-case review of generated reports. The model receives report text only, not medical images.

`google/flan-t5-base` is an instruction-tuned encoder-decoder transformer. The encoder processes the instruction and report using self-attention. The decoder generates the explanation using causal self-attention and cross-attention to the encoder output. Fine-tuning uses teacher forcing with token-level cross-entropy loss. The zero-shot baseline leaves the model unchanged. Full fine-tuning updates all weights, while LoRA fine-tuning freezes the base model and trains low-rank updates in the attention layers. The implementation uses PyTorch, Transformers and PEFT.

[Chung et al.](https://arxiv.org/abs/2210.11416) report better held-out benchmark performance after instruction tuning, although this does not directly measure performance on radiology reports. [Hu et al.](https://arxiv.org/abs/2106.09685) describe low-rank updates as `W = W₀ + (α/r)BA` and discuss limitations of small ranks. These updates can be merged into the base weights for inference, but the adapter in this project is kept separate.

![FLAN-T5 pipeline comparing zero-shot generation, full fine-tuning and LoRA fine-tuning](figures/flan-t5-pipeline.svg)

## Dataset and Preprocessing

The official [BioLaySumm LaymanRRG dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) contains 150,454 training rows, 10,000 validation rows and 10,537 test rows. I kept the entire official validation set for the final ROUGE comparison and formed a development set from the training data.

| Split | Reports | Use |
| --- | ---: | --- |
| Training | 102,733 | Parameter updates for both fine-tuning methods |
| Development | 5,488 | Loss and ROUGE after each training epoch |
| Official validation | 10,000 | Final loss and ROUGE on the same reports for all three methods |
| Official test | 10,537 | A common sample of 100 reports for generation, timing and case review |

Before filtering, **4,504 validation rows** shared their full report text with the official training split. Of those, **2,370** also had a matching target summary and **2,134** had a different target. Within the original training data, 12,150 distinct report texts had multiple target summaries. Repeated report–summary pairs can inflate ROUGE if a model sees them during training.

I removed all **42,233 training rows** containing any of the 2,148 report texts shared with validation. I then grouped the remaining rows by complete report text and assigned 5% of the groups to development with seed 42. This kept all target variants of each report together. Training, development and final validation have zero exact report texts in common. Matching used the original strings before tokenization. The test set has no public reference summaries, so its outputs are assessed against their source reports.

`radiology_report` is the input and `layman_report` the target. Source and image identifiers are retained for auditing, but images are not loaded. Each report follows this prompt and a newline:

```text
Explain this radiology report in plain English for a patient. Preserve the findings, negation, and uncertainty:
```

Inputs and targets are tokenized separately and truncated to 512 tokens each, including special tokens and, for inputs, the prompt. Dynamic batch padding uses attention masks for inputs and `-100` for target padding, which is ignored by the loss. References never enter source inputs, and complete original references are retained for ROUGE.

## Feasibility Review

**User need and scope.** Technical radiology language can be difficult for patients to understand. The prototype aims to explain findings clearly while preserving the original qualifications. It does not interpret images or make new diagnoses.

**Model choice and course concepts.** I selected FLAN-T5-base for text-to-text generation and compare its zero-shot outputs with full fine-tuning and rank-8 LoRA fine-tuning on BioLaySumm LaymanRRG. This Hard-difficulty design uses encoder-decoder attention, transfer learning and supervised sequence prediction to investigate whether better ROUGE scores also mean more faithful explanations.

**Five testable acceptance criteria.**

1. Keep training and validation separate, exclude reference answers from model inputs, and evaluate all methods on identical validation examples.
2. Improve ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum over the zero-shot baseline.
3. Preserve findings, negation, uncertainty and examination limitations without unsupported claims or important omissions in a four-case audit.
4. Explain medical terms clearly without unexplained jargon or repeated wording in the audited outputs.
5. Complete each formal job within 12 hours on one 40 GB A100 and save weights that can be reloaded for inference.

**Initial feasibility evidence.** Data checks confirmed split sizes and the availability of validation references.

**Budget, risks and fallback.** The budget was one A100 GPU with a 12-hour limit per job. The main risks were slow validation, 512-token truncation and clinically inaccurate summaries. I planned to save a checkpoint after each epoch and reduce the number of epochs for both fine-tuning methods if runtime became a problem. The validation set would remain unchanged. A source-based review was needed because ROUGE cannot detect every factual error.

**Experimental plan.** Train both adapted models for two epochs with seed 42, compare them against the zero-shot baseline using the same prompt and decoding method, profile compute use, and inspect four test reports for clinical errors.

## Experimental Design

I trained both adapted models with the PyTorch loop in `train.py`, starting from the same FLAN-T5-base checkpoint. The zero-shot comparison uses saved outputs from the original baseline run with the same checkpoint, prompt, decoding settings and official validation reports. Each fine-tuning job used one NVIDIA A100-PCIE-40GB and two CPU cores on Rangpur.

| Setting | Full fine-tuning | LoRA fine-tuning |
| --- | --- | --- |
| Starting checkpoint | `google/flan-t5-base` | Same |
| Epochs / optimizer updates | 2 / 12,842 | 2 / 12,842 |
| Batch / gradient accumulation | 2 / 8 (effective batch 16) | Same |
| Optimizer / learning rate | AdamW / `5e-5` | AdamW / `3e-4` |
| Schedule / warm-up / weight decay | Linear decay / 0 / 0 | Same |
| Precision | FP32 (no mixed precision) | Same |
| Input / target limits | 512 / 512 tokens | Same |
| Seed | 42 | Same |
| Generation | Greedy, at most 512 new tokens | Same |
| Model selection | Final epoch | Final epoch |
| Adaptation | All weights | `q`, `v` (rank 8, alpha 16, dropout 0.05) |

I fixed the two-epoch schedule and evaluated the final model once on official validation. Only development scores were monitored during training. LoRA fine-tuning used no trainable biases. The learning rates differed, so the comparison measures the two configurations as run. Both saved models reproduced their first batch of final validation predictions after reloading.

## Results

### Validation Quality

Scores are mean per-example ROUGE F1 on a **0 to 100 scale**, with Porter stemming and NLTK sentence splitting for ROUGE-Lsum, evaluated on all 10,000 validation reports.

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot baseline | 12.19 | 4.13 | 10.84 | 11.32 | 2.4202 |
| Full fine-tuning, 2 epochs | **73.25** | **54.81** | **68.31** | **70.27** | **0.5421** |
| LoRA fine-tuning, 2 epochs | 68.53 | 48.29 | 63.15 | 65.42 | 0.7635 |

![Baseline, full fine-tuning and LoRA fine-tuning compared on four validation ROUGE metrics, using a shared 0 to 100 scale](figures/rouge_comparison.png)

Full fine-tuning led on all four metrics, finishing **5.16 ROUGE-L points** ahead of LoRA fine-tuning. On development, ROUGE-L rose from **64.22 to 64.98** between epochs for full fine-tuning and from **59.46 to 60.17** for LoRA fine-tuning.

### Validation by Original Training-Set Overlap

Following the Ed clarification, I also scored the 5,496 validation reports whose source text was absent from the *original* training split. The other 4,504 had a match before filtering. All 10,000 were held out of parameter updates and development in these runs.

| Method | Match in original training split | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
| --- | --- | ---: | ---: | ---: | ---: |
| Zero-shot baseline | Absent | 14.28 | 5.16 | 12.49 | 13.20 |
| Full fine-tuning | Absent | 71.01 | 50.44 | 64.63 | 67.56 |
| LoRA fine-tuning | Absent | 66.91 | 44.48 | 59.85 | 63.25 |
| Zero-shot baseline | Present | 9.65 | 2.88 | 8.83 | 9.02 |
| Full fine-tuning | Present | 75.99 | 60.13 | 72.79 | 73.59 |
| LoRA fine-tuning | Present | 70.52 | 52.94 | 67.17 | 68.08 |

Both adapted models scored higher on the previously matched group even after its training copies were removed. The groups also differ in length. Source reports averaged 10.84 words in the matched group and 33.54 words in the group without a match.

### Readability

Flesch–Kincaid grade estimates reading difficulty from sentence length and syllables per word. I calculated it separately for each report using `textstat==0.7.13` with US English settings. The table covers all 10,000 validation reports and their saved outputs.

| Text | Mean words | Mean grade | Grade ≤6 (%) |
| --- | ---: | ---: | ---: |
| Original report | 23.32 | 13.79 | 2.69 |
| Reference summary | 38.60 | 7.45 | 31.13 |
| Zero-shot baseline | 10.61 | 13.64 | 12.53 |
| Full fine-tuning | 36.17 | 6.81 | 40.78 |
| LoRA fine-tuning | 35.82 | 6.42 | 45.41 |

The baseline produced 13 outputs with no countable words. These are included in mean word count but excluded from grade statistics; all other rows have 10,000 grade scores.

Both methods produced summaries with a lower mean reading grade than the references. Fewer than half reached grade six or below. The baseline gave the shortest answers yet had the highest estimated reading grade among the models. Case 1 illustrates why a short answer can still be inadequate for a patient.

### Resource Use and Inference

| Method | Total parameters | Trainable parameters | Peak allocated VRAM (MiB) | Training + development (s) | Final validation (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot baseline | 247,577,856 | 0 | 1,428.48 | — | 3,256.38 |
| Full fine-tuning | 247,577,856 | 247,577,856 | 8,679.37 | 18,465.92 | 5,762.14 |
| LoRA fine-tuning | 248,462,592 | 884,736 | 5,370.20 | 21,510.96 | 7,363.04 |

Training time includes both development passes and checkpoint saves. Peak VRAM is the highest memory allocated by PyTorch across training and evaluation. LoRA fine-tuning updated **0.356%** of its parameters and used **38.1%** less peak VRAM. End-to-end job time was **6 h 49 min 20 s** for full fine-tuning and **8 h 05 min 15 s** for LoRA fine-tuning, including final validation and test inference.

Inference used the same **100 test reports sampled with seed 42**, batch size one, on the same GPU model:

| Condition | Generated tokens | Generation time (s) | Mean seconds/report | Tokens/second | Inference peak allocated VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,794 | 61.52 | 0.615 | 61.67 | 1,043.04 |
| Full fine-tuning | 6,878 | 111.89 | 1.119 | 61.47 | 1,043.04 |
| LoRA fine-tuning | 6,655 | 135.96 | 1.360 | 48.95 | 1,046.41 |

Timing includes only CUDA-synchronized `generate()` calls. It excludes loading, tokenization, input transfer, decoding and scoring. There was no warm-up. Token counts exclude decoder start and padding tokens but include EOS. The baseline generated shorter outputs, so seconds per report cannot be compared without considering output length. Timing was measured once per method.

## Clinical Safety Autopsy: Four Test Cases

Cases 1–4 are reports 14, 15, 19 and 20 in the same 100-report test sample (seed 42, counting from 1). I kept these four reports fixed across runs and compared each output with the source text. The public test split has no reference summaries.

### Case 1: Technical terms left unexplained

Source: **PadChest**. Identifier: `216840111366964013590140476722013044085724775_02-062-040.png`.

**Original report:**

> Images of the lingular and right basal zones are consistent with bronchiectasis and COPD. A small nodular image of small size is noted in the left parahilar region. There is an increase in apical pleural thickening on the left. A complete study with a chest CT is recommended.

**Baseline:**

> A complete chest CT is recommended.

**Full fine-tuning:**

> The images of the lingular and right basal areas are consistent with bronchiectasis and COPD. A small nodule of small size is noted in the left parahilar region. There is an increase in apical pleural thickening on the left. A complete study with a chest CT is recommended.

**LoRA fine-tuning:**

> The images of the lingular and right basal zones are consistent with bronchiectasis and COPD. A small nodule of small size is seen in the left parahilar region. There is an increase in thickening of the pleural pleura on the left. It is suggested to have a full study with a chest CT scan.

**Analysis.** The baseline gives only the CT recommendation, omitting the nodule and increased pleural thickening. Full fine-tuning keeps those details but copies terms such as “bronchiectasis” and “parahilar” into the patient summary. Even “a small nodule of small size” closely follows the source. LoRA fine-tuning introduces the awkward phrase “pleural pleura” and drops “apical,” so the reader loses the location of the thickening as well as an explanation of the term.

### Case 2: An opacity interpreted as fluid

Source: **BIMCV-COVID19**. Identifier: `ses-E67984`.

**Original report:**

> Posteroanterior chest radiograph. Impression: Right basal and lingular consolidation, consider an inflammatory or infectious origin. Chronic parenchymal lung changes. Right apical thickening. Radiologic signs of EPO (extrapleural opacity). Skin fold projected over the right hemithorax.

**Baseline:**

> A chest radiograph.

**Full fine-tuning:**

> A chest x-ray taken from the back to the front shows a solid area in the lower right part of the lung and a part called the lingular, which could be due to inflammation or infection. There are also long-term changes in the lung tissue. The top part of the right lung is thicker than normal. There are signs of fluid buildup outside the lungs. A skin fold is seen over the right side of the chest.

**LoRA fine-tuning:**

> The chest x-ray taken from the front shows signs of lung tissue thickening at the bottom right and the lingular area, which could be due to inflammation or infection. There are long-term changes in the lung tissue. The top right part of the lung has thickening. There are signs of fluid buildup around the lungs. A skin fold is seen over the right side of the chest.

**Analysis.** The report expands EPO as “extrapleural opacity.” Both fine-tuned models tell the reader there is fluid, although the report never identifies what caused the opacity. This is easy to overlook because both models preserve the *separate* suggestion of inflammation or infection as a possible cause of the consolidation. The unsupported fluid claim concerns a different finding. The baseline leaves out both findings and says only that a chest radiograph was taken.

### Case 3: A lost negative and an uncertain finding

Source: **BIMCV-COVID19**. Identifier: `ses-E46467`.

**Original report:**

> No clear pulmonary infiltrates are observed, with a questionable right basal opacity that should be evaluated over time.

**Baseline:**

> This report is a radiology report for a patient with a pulmonary infiltrate.

**Full fine-tuning:**

> There are no clear signs of lung infections, but there might be a possible cloudiness at the bottom right part of the lung that needs to be checked over time.

**LoRA fine-tuning:**

> There are no clear signs of lung infections, but there might be a possible cloudiness at the bottom right part of the lung that should be checked over time.

**Analysis.** The report makes two distinct observations. It finds no clear infiltrates, while a questionable opacity on the lower right needs follow-up. The baseline drops “No clear,” tells the reader there is an infiltrate and omits the follow-up. Both fine-tuned models keep the uncertainty and follow-up and explain the opacity as “cloudiness.” But they change the first observation to “no clear signs of lung infections.” This narrows an imaging finding to a particular clinical interpretation that the source did not make.

### Case 4: Which parts of the lungs were examined?

Source: **BIMCV-COVID19**. Identifier: `ses-E28411`.

**Original report:**

> Findings: 22-year-old woman with acute appendicitis. Positive PCR for coronavirus. Requested examination performed. Suboptimal study as both lung apices are not included. In the visualized lung parenchyma, no consolidations or other intraparenchymal pulmonary infiltrates are observed. No pleural effusion is evident. Cardiac silhouette and pulmonary hila are of normal morphology and size.

**Baseline:**

> A 22-year-old woman with acute appendicitis. Positive PCR for coronavirus.

**Full fine-tuning:**

> A 22-year-old woman has acute appendicitis, which is a type of inflammation that affects the lining of the lungs. She tested positive for coronavirus, and was asked to have an examination done. The study was not perfect because both the top parts of the lungs weren't included. In the lung tissue that can be seen, there are no signs of pneumonia or other lung infections. There is no fluid around the lungs. The heart and the areas where the lungs connect to the heart are normal in shape and size.

**LoRA fine-tuning:**

> The patient is a 22-year-old woman with acute appendicitis. She has a positive PCR for coronavirus. She was asked to have an examination. The examination was not very good because both the lungs are not included. The lung tissue looks normal, but there are no signs of pneumonia or other lung infections. There is no fluid around the lungs. The heart and the areas where the lungs connect to the heart are normal in shape and size.

**Analysis.** Only the tops of both lungs were outside the image. LoRA fine-tuning broadens that limitation to the lungs themselves, telling the reader that “both the lungs are not included,” then saying “The lung tissue looks normal.” A reader cannot tell what was actually examined. Full fine-tuning keeps the boundary in “the lung tissue that can be seen,” but introduces a different error when it explains appendicitis as inflammation of the lung lining. Appendicitis affects the appendix. The baseline retains the presenting history and omits both the examination limitation and the chest findings.

### What the Cases Show

The baseline omits findings or follow-up in all four cases. The longer fine-tuned summaries retain more information, but both models make the same unsupported substitutions in Cases 2 and 3. In Case 4, full fine-tuning explains the imaging limitation more accurately, while adding a medical definition that is wrong. ROUGE rewards words shared with a reference. These errors depend on which finding a word describes, where it is located and how certain the report is.

## Recommendation and Limitations

I would choose full fine-tuning when reference agreement and inference speed are the priorities. It led on all four ROUGE measures and generated 61.47 tokens per second, compared with 48.95 for LoRA fine-tuning. When GPU memory is limited, LoRA fine-tuning cut peak VRAM by 38.1% and saved a 3.56 MB adapter. Full-model weights took 990.35 MB. The adapter requires the base model at inference. Both configurations still produced unsupported findings in the case review.

| Feasibility criterion | Outcome | Evidence |
| --- | --- | --- |
| 1. Data and evaluation separation | Met | Training, development and final validation shared zero exact report texts. All methods were scored on the same 10,000 final reports. |
| 2. Higher ROUGE than zero-shot | Met | Both adapted models improved all four metrics. |
| 3. Clinical faithfulness in four cases | Not met | Both models described the opacity as fluid in Case 2. In Case 3 they changed “no clear pulmonary infiltrates” to “no clear signs of lung infections.” |
| 4. Clear, non-repetitive explanations | Not met | In Case 1, “parahilar” remained unexplained and the LoRA fine-tuning output used “pleural pleura.” |
| 5. Resource budget and saved models | Met | Both jobs finished within 12 hours on one A100 and reproduced predictions after reloading. |

**Engineering recommendation.** Neither model should generate summaries for patients without review. A safer workflow would compare each draft with its source report before release, checking anatomical location, findings, negation, uncertainty and severity. It should also flag missing findings, follow-up advice and examination limits, as well as repeated or unexplained technical terms. A summary with a possible clinical contradiction or omission should be sent to a qualified reviewer. These checks are proposed safeguards, not features implemented or validated in this prototype.

**Limitations.** The comparison uses one base checkpoint, seed and prompt, with one learning rate per method. Longer reports can be truncated at 512 input tokens. Exact text matching can miss near-duplicates and reports from the same patient. The official validation set was also used in earlier experiments. Reading grades estimate sentence and word complexity. The four selected cases show specific errors, while estimating their frequency or measuring patient comprehension would require a separate study.

## Implementation and Reproduction

| File | Purpose |
| --- | --- |
| `dataset.py` | Audit the official splits, remove shared report texts, form a grouped development set and tokenize pairs |
| `modules.py` | Load FLAN-T5, configure LoRA fine-tuning and count parameters |
| `train.py` | Run explicit training and evaluation loops; save predictions, metrics, weights and curves |
| `predict.py` | Reload a run, generate explanations and measure inference |
| `evaluate.py` | Re-score saved validation predictions for overlap subsets and readability |

The training loop explicitly runs forward passes, backpropagation, gradient accumulation, gradient clipping, AdamW updates and linear learning-rate decay. It averages the losses over the actual number of microbatches in each update, including the final partial group. Evaluation uses `model.eval()` and `torch.inference_mode()`. References are used for loss and ROUGE, while generation receives only source tokens and their attention masks. The final epoch is saved without validation-based checkpoint selection.

### Environment

Formal runs used the following environment on Rangpur:

| Component | Recorded version |
| --- | --- |
| Python | 3.11.15 |
| PyTorch / CUDA build | 2.13.0 / 13.0 |
| NVIDIA driver / GPU | 590.48.01 / A100-PCIE-40GB |
| Transformers / PEFT | 4.57.6 / 0.19.1 |
| Datasets / Accelerate | 5.0.0 / 1.14.0 |
| SentencePiece / rouge-score | 0.2.2 / 0.1.2 |
| NLTK | 3.10.0 (`punkt_tab` installed) |
| NumPy / Matplotlib | 2.4.6 / 3.11.1 |

To install the direct dependencies into an environment with compatible Python and PyTorch:

```bash
python -m pip install transformers==4.57.6 peft==0.19.1 \
    datasets==5.0.0 accelerate==1.14.0 sentencepiece==0.2.2 \
    rouge-score==0.1.2 nltk==3.10.0 numpy==2.4.6 matplotlib==3.11.1
python -m nltk.downloader punkt_tab
```

### Training and Prediction

Run these commands from the **repository root**, using new output directories. Formal experiments require an allocated CUDA GPU. The default prediction device is CPU.

```bash
python recognition/FLAN_T5_s4983489/train.py \
    --mode baseline --output-dir ../runs/baseline

python recognition/FLAN_T5_s4983489/train.py \
    --mode full --output-dir ../runs/full --epochs 2 --learning-rate 0.00005

python recognition/FLAN_T5_s4983489/train.py \
    --mode lora --output-dir ../runs/lora --epochs 2 --learning-rate 0.0003

python recognition/FLAN_T5_s4983489/predict.py \
    --run-dir ../runs/full \
    --text "No clear pulmonary infiltrates are observed, with a questionable right basal opacity that should be evaluated over time."

python recognition/FLAN_T5_s4983489/predict.py \
    --run-dir ../runs/full --split test --samples 100 --device cuda \
    --output ../runs/full/inference-test100.json

python recognition/FLAN_T5_s4983489/dataset.py \
    --audit-output ../runs/overlap-audit.json
```

The commands above use seed 42 and assign 5% of the remaining report groups to development. Each epoch saves its weights in `model/` and optimizer and scheduler states in `checkpoint.pt` before development evaluation. `training_history.json` and `training_curves.png` record training loss and development metrics. The run directory also contains `dataset_audit.json` with partition counts, `metrics.json` with final validation results and `predictions.jsonl` with one output per validation row.

Prediction requires the run configuration and, for adapted models, its saved `model/`. The custom input above corresponds to Case 3. Use `--split validation` to generate and score validation reports. Test and custom text have no reference-based ROUGE. Sampling uses the saved seed. `--output` needs a new file in an existing directory. Without it, the result is printed. `--help` lists the remaining options.

### Saved-Output Evaluation

`evaluate.py` checks saved predictions against official validation sources, references and identifiers in order. Its `unseen` and `overlap` keys refer to matches against the original training file **before filtering**. It reports ROUGE for the full set and both groups, plus readability for the full set and the group without an original match.

The readability analysis used Python 3.12.13, `textstat==0.7.13`, `pyphen==0.18.1` and `nltk==3.10.0`. With the dependencies above installed, run:

```bash
python -m pip install textstat==0.7.13 pyphen==0.18.1
python -m nltk.downloader cmudict
python recognition/FLAN_T5_s4983489/evaluate.py \
    --run-dirs ../runs/baseline ../runs/full ../runs/lora \
    --output ../runs/validation_analysis.json
```

Each run directory must contain `predictions.jsonl` for the complete official validation set. The output must be a new file in an existing directory. Reading grades use textstat's default word, syllable and sentence counting, without clipping negative grades. Outputs with no countable words are excluded from grade statistics and reported separately.

### Experiment Records

Records are kept outside Git. Local paths are relative to the repository root. Remote paths refer to my Rangpur account.

| Records | Location |
| --- | --- |
| Local zero-shot baseline | `../runs/rangpur-20261003/baseline/` |
| Local split audit, fine-tuning results and environments | `../runs/independent-20261117/` |
| ROUGE by original overlap group and readability | `../runs/independent-20261117/validation_analysis.json` |
| Remote baseline and original logs | `/home/Student/s4983489/comp3710-ass2/runs/20261003-formal/` |
| Remote fine-tuned weights and checkpoints | `/home/Student/s4983489/comp3710-ass2/runs/20261117-independent/{full,lora}/` |

Trained weights and checkpoints are stored on Rangpur and are not included in this repository.

## Artificial Intelligence Usage Disclosure

I used OpenAI Codex to assist with literature review, debugging, experiment execution, the manual-loop revision and saved-output analysis. I wrote the initial implementation and used ChatGPT to review and edit the wording and clinical error discussion in this README. I checked data preprocessing, parameter updates, evaluation and model save/reload behaviour. I checked the scores against saved predictions and compared the clinical examples with their source reports. I am responsible for checking the final code, measurements and conclusions.

## References

1. COMP3710 assessment specification.
2. Chung et al. (2022), [*Scaling Instruction-Finetuned Language Models*](https://arxiv.org/abs/2210.11416), Section 5 and Appendices A.1 and G.2.
3. Hu et al. (2021), [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), Sections 4.1 and 7.
4. [BioLaySumm LaymanRRG dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) and [FLAN-T5-base model card](https://huggingface.co/google/flan-t5-base).
5. Hugging Face documentation: [Datasets loading](https://huggingface.co/docs/datasets/loading), [summarisation](https://huggingface.co/docs/transformers/tasks/summarization), [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [generation](https://huggingface.co/docs/transformers/main_classes/text_generation), [PEFT](https://huggingface.co/docs/peft/main/en/quicktour) and [checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).
6. Google Research, [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge).
7. Course Ed clarification, [Radiology Report Summarisation Task Clarification](https://edstem.org/au/courses/38242/discussion/3634865).
8. [Textstat documentation](https://github.com/textstat/textstat), Flesch–Kincaid grade and text statistics.
