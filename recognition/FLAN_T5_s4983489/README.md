# FLAN-T5 for Layperson Radiology Report Generation

COMP3710 — s4983489 — Hard difficulty

## Problem and Approach

This project uses FLAN-T5 to rewrite radiology reports in plain English for patients. The target is a sixth-grade reading level without losing findings, anatomical details, negation or uncertainty. I compare a zero-shot baseline with full fine-tuning and LoRA, using ROUGE scores and a manual review of generated reports. The model works from report text only, not medical images.

`google/flan-t5-base` is an instruction-tuned encoder-decoder transformer. The encoder uses self-attention to process the instruction and report. The decoder uses causal self-attention and cross-attention to produce the explanation. Training uses teacher-forced token-level cross-entropy, covering attention, transfer learning and supervised sequence prediction. The zero-shot baseline has no parameter updates. Full fine-tuning updates all weights, while LoRA trains low-rank attention updates with the base model frozen. The code uses PyTorch, Transformers, PEFT and Accelerate.

[Chung et al.](https://arxiv.org/abs/2210.11416) report better held-out benchmark performance after instruction tuning, although this does not directly measure performance on radiology reports. [Hu et al.](https://arxiv.org/abs/2106.09685) describe LoRA updates as `W = W₀ + (α/r)BA` and discuss limitations of small ranks. LoRA updates can be merged into the base weights for inference, but the adapter in this project is kept separate.

## Dataset and Preprocessing

The official [BioLaySumm LaymanRRG dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) is used without re-splitting or augmentation:

| Split | Reports | Use |
| --- | ---: | --- |
| Training | 150,454 | Full and LoRA fine-tuning |
| Validation | 10,000 | ROUGE and validation loss; all references are non-empty |
| Test | 10,537 | Generation, timing and source-grounded error analysis |

The split sizes were checked after loading the data. Training uses only the official training split, and the quantitative comparison uses validation as required by the assessment. The test references are not public, so the test split is used for generation and source-based error analysis rather than ROUGE.

`radiology_report` is the input and `layman_report` the target. Source and image identifiers are retained for auditing, but images are not loaded. Each report follows this prompt and a newline:

```text
Explain this radiology report in plain English for a patient. Preserve the findings, negation, and uncertainty:
```

Inputs and targets are tokenized separately and truncated to 512 tokens each, including special tokens and, for inputs, the prompt. Dynamic batch padding uses attention masks for inputs and `-100` for target padding, which is ignored by the loss. References never enter source inputs, and complete original references are retained for ROUGE.

## Feasibility Review

**User need and scope.** Radiology reports contain terminology that can be difficult for patients to understand. The goal is to explain the report text more clearly while retaining the findings and qualifications. The system does not interpret images or provide diagnoses.

**Model choice and course concepts.** The Hard-difficulty design compares zero-shot FLAN-T5-base, full fine-tuning and rank-8 LoRA on the official BioLaySumm LaymanRRG dataset. This lets me apply attention, transfer learning and supervised sequence prediction in the same task. Full fine-tuning changes all model weights, while LoRA trains attention adapters. The official splits are used without modification.

**Five testable acceptance criteria.**

1. Keep training and validation separate, exclude reference answers from source inputs, and evaluate all methods on identical validation examples.
2. Improve ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-Lsum over the zero-shot baseline.
3. Preserve findings, negation, uncertainty and examination limitations without unsupported claims or important omissions in a five-case audit.
4. Explain technical terms clearly and avoid unexplained jargon or repetition in the audited outputs.
5. Complete each formal job within 12 hours on one 40 GB A100 and save weights that can be reloaded for inference.

**Initial feasibility evidence.** Data checks confirmed the split counts and that validation references were available. Small CPU tests checked tokenization, padding masks, finite losses and adapter gradients while the base weights stayed frozen. Saving and reloading also produced matching predictions. These checks confirmed that the training setup worked before the formal GPU runs.

**Budget, risks and fallback.** The compute budget was one A100 with a 12-hour limit per job. The main risks were slow validation, input truncation and inaccurate or incomplete explanations. The plan was to save a checkpoint after each epoch and, if needed, reduce the number of epochs for both fine-tuning methods. The validation set would remain unchanged. I also planned to check selected outputs against their source reports, since ROUGE does not assess factual accuracy directly.

**Experimental plan.** Compare the baseline, full fine-tuning and LoRA with the same prompt and decoding settings. Train both adapted models for two epochs using seed 42, record the quality and resource results, and inspect five test cases.

## Experimental Design

All methods used the same checkpoint, prompt, validation reports and greedy decoding. Formal runs used one NVIDIA A100-PCIE-40GB and two CPU cores on Rangpur, with a 12-hour limit per job. Both adaptations used the complete training split:

| Setting | Full fine-tuning | LoRA |
| --- | --- | --- |
| Starting checkpoint | `google/flan-t5-base` | Same |
| Epochs / optimizer updates | 2 / 18,808 | 2 / 18,808 |
| Batch / gradient accumulation | 2 / 8; effective batch 16 | Same |
| Optimizer / learning rate | AdamW / `5e-5` | AdamW / `3e-4` |
| Schedule / warm-up / weight decay | Linear decay / 0 / 0 | Same |
| Precision | FP32; no mixed precision | Same |
| Input / target limits | 512 / 512 tokens | Same |
| Seed | 42 | Same |
| Generation | Greedy; at most 512 new tokens | Same |
| Model selection | Final epoch | Final epoch |
| Adaptation | All weights | `q`, `v`; rank 8, alpha 16, dropout 0.05 |

LoRA used no trainable biases. No hyperparameter search was conducted. Both adapted models were evaluated after each epoch, and the final epoch was used for comparison. The learning rates differ, so the results compare these specific configurations rather than isolating the effect of the adaptation method. Saved identifiers and references were checked to confirm that all methods used the same validation examples.

An initial three-epoch full fine-tuning run timed out during final validation after **12:00:18** and did not save weights. I restarted both adapted models from the base checkpoint for two epochs and added epoch checkpoints. The baseline run was reused. The full and LoRA jobs then finished in **8:11:11** and **9:45:45**, including setup, validation, saving and test inference. Both saved models were reloaded successfully. The failed run used additional compute and is not included in the completed-run table.

## Results

### Validation Quality

Scores are mean per-example ROUGE F1 on a **0–100 scale**, with Porter stemming and NLTK sentence splitting for ROUGE-Lsum, evaluated on all 10,000 validation reports.

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot baseline | 12.19 | 4.13 | 10.84 | 11.32 | 2.4202 |
| Full fine-tuning, 2 epochs | **74.47** | **56.83** | **69.71** | **71.63** | **0.5123** |
| LoRA, 2 epochs | 71.25 | 52.10 | 65.97 | 68.19 | 0.7206 |

![Baseline, full fine-tuning and LoRA compared on four validation ROUGE metrics, using a shared 0–100 scale](figures/rouge_comparison.png)

*Scores are from the saved validation outputs. Both adapted models trained for two epochs, and the baseline was not updated. Each method was run once.*

Both adapted models scored much higher than the baseline. Full fine-tuning exceeded LoRA by **3.74 ROUGE-L points**. From epoch one to two, validation loss decreased from **0.5449 to 0.5123** for full fine-tuning and from **0.7630 to 0.7206** for LoRA. ROUGE-L increased from **69.11 to 69.71** and **65.14 to 65.97**, respectively.

### Resource Use and Inference

| Condition | Total parameters | Trainable parameters | Peak allocated VRAM (MiB) | Measured train/eval routine (s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 247,577,856 | 0 | 1,428.48 | 3,256.38 (validation only) |
| Full | 247,577,856 | 247,577,856 | 8,692.12 | 29,265.91 |
| LoRA | 248,462,592 | 884,736 | 5,353.66 | 34,980.96 |

For full and LoRA, `train_runtime` includes both validation passes and epoch checkpoint saving, but excludes final export and test inference. Baseline time covers validation alone. VRAM is peak PyTorch allocated memory, not total process memory. The saved field `peak_gpu_memory_mb` is expressed here in **MiB**, since the code divides bytes by `1024²`.

LoRA updated **0.356%** of its parameters and used **38.4%** less peak allocated VRAM than full fine-tuning, but took longer. Both methods still run the base model and generate full sequences during validation. The separate adapter may add overhead, although its contribution was not measured separately.

Inference used the same **100 test reports sampled with seed 42**, batch size one, on the same GPU model:

| Condition | Generated tokens | Generation time (s) | Mean seconds/report | Tokens/second | Inference peak allocated VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3,794 | 61.52 | 0.615 | 61.67 | 1,043.04 |
| Full | 6,916 | 107.99 | 1.080 | **64.04** | 1,043.04 |
| LoRA | 6,827 | 139.51 | 1.395 | 48.94 | 1,046.41 |

Timing includes only CUDA-synchronized `generate()` calls. It excludes loading, tokenization, input transfer, decoding and scoring. There was no warm-up. Token counts exclude decoder start and padding tokens but include EOS. The baseline generated shorter outputs, which partly explains its lower time per report. Timing was measured once per method.

## Clinical Safety Autopsy: Five Test Cases

I inspected the first 30 predictions from the 100-report test sample and selected five cases with clear omissions, unsupported statements or readability problems. Case numbers refer to positions in the seed-42 sample, not the original dataset indices. I compared the generated text with its source report because test references are not publicly available. This review used AI assistance and was not independently checked by a clinician.

The source reports and model outputs are shown below without edits. Input lengths were 57–131 tokens, and output lengths were 6–123 tokens. All were below the 512-token limits, so none of these five cases was truncated.

### Case 14: Repetition and missing location detail

Source: **PadChest**. Identifier: `216840111366964013590140476722013044085724775_02-062-040.png`.

**Original report:**

> Images of the lingular and right basal zones are consistent with bronchiectasis and COPD. A small nodular image of small size is noted in the left parahilar region. There is an increase in apical pleural thickening on the left. A complete study with a chest CT is recommended.

**Baseline:**

> A complete chest CT is recommended.

**Full fine-tuning:**

> The images of the left lower lung and the lower right lung are consistent with a condition called bronchiectasis and chronic obstructive pulmonary disease (COPD). There is a small round spot in the left lung area near the hilum. The thickening of the lung coverings at the top on the left side is more noticeable. It is suggested to do a full chest CT scan for further evaluation.

**LoRA:**

> The images of the lingular and right basal areas are consistent with bronchiectasis and COPD. A small nodule of small size is seen in the left parahilar region. There is an increase in thickening of the pleural pleural pleural pleural thickening on the left. It is recommended to have a full study with a chest CT scan.

**Analysis.** The baseline gives only the CT recommendation and leaves out every finding. Full fine-tuning covers most of the report but replaces the lingular location with the less precise “left lower lung” and does not explain bronchiectasis or hilum. LoRA repeats “pleural” four times, omits the apical location and keeps several technical terms. It also changes “nodular image” to the more definite “nodule.” The report contains several findings with specific anatomical locations, and the main failure is losing those details while simplifying the language.

### Case 15: Unsupported fluid findings

Source: **BIMCV-COVID19**. Identifier: `ses-E67984`.

**Original report:**

> Posteroanterior chest radiograph. Impression: Right basal and lingular consolidation, consider an inflammatory or infectious origin. Chronic parenchymal lung changes. Right apical thickening. Radiologic signs of EPO (extrapleural opacity). Skin fold projected over the right hemithorax.

**Baseline:**

> A chest radiograph.

**Full fine-tuning:**

> A chest x-ray taken from the back to the front shows a solid area in the lower right part of the lung and a part called the lingular, which could be due to inflammation or infection. There are also long-term changes in the lung tissue. The top part of the right lung is thicker than normal. There are signs of fluid buildup outside the lung. A skin fold is seen over the right side of the chest.

**LoRA:**

> The chest x-ray taken from the front shows signs of fluid buildup in the lower right part of the lung and a thickening of the lung tissue at the top. There are also signs of fluid buildup in the outer parts of the lungs. A skin fold is seen over the right side of the chest.

**Analysis.** The baseline returns only “A chest radiograph” and omits the findings. Full fine-tuning includes most findings and the possible inflammatory or infectious origin, but interprets “extrapleural opacity” as “fluid buildup.” The source does not mention fluid. LoRA also introduces fluid, while leaving out the lingular finding, chronic lung changes, the possible cause and the right-sided apical detail. Full fine-tuning still leaves “lingular” unexplained. The error occurs when an imaging term is rewritten as a more specific clinical finding, which could make a patient believe fluid was observed.

### Case 18: Incorrect symptom description

Source: **PadChest**. Identifier: `216840111366964012373310883942009104090852549_00-071-085.png`.

**Original report:**

> Reason for consultation: dyspnea, wheezing, and abundant expectoration. Posteroanterior and lateral chest radiograph. No parenchymal consolidation is visualized.

**Baseline:**

> a reconstructed pulmonary edema

**Full fine-tuning:**

> The patient came in because they were having trouble breathing, wheezing, and a lot of coughing up. We took a chest x-ray from the front and the side. There's no sign of lung tissue becoming solid or dense.

**LoRA:**

> The patient came in because they were having trouble breathing, coughing, and were looking for a good night. The chest x-ray was taken from the front and the side. There is no sign of lung tissue becoming solidified.

**Analysis.** The baseline adds pulmonary edema, which the report does not state, and omits both the symptoms and the negative consolidation finding. Full fine-tuning retains most of the report, although “a lot of coughing up” is unclear. LoRA inserts the unrelated phrase “looking for a good night” and replaces wheezing and abundant expectoration with a general mention of coughing. It does, however, preserve the absence of consolidation. The unrelated wording appears while paraphrasing the symptoms, showing that preserving one negative finding does not ensure the rest of the report is accurate.

### Case 19: Negation and uncertainty

Source: **BIMCV-COVID19**. Identifier: `ses-E46467`.

**Original report:**

> No clear pulmonary infiltrates are observed, with a questionable right basal opacity that should be evaluated over time.

**Baseline:**

> This report is a radiology report for a patient with a pulmonary infiltrate.

**Full fine-tuning:**

> There are no obvious signs of lung infections, but there might be a possible cloudiness at the bottom right part of the lung that needs to be checked over time.

**LoRA:**

> There are no clear signs of lung infections, but there might be a possible cloudiness at the bottom right part of the lung that should be checked over time.

**Analysis.** The baseline changes “no clear pulmonary infiltrates” into a positive claim about an infiltrate. It also drops the uncertain right basal opacity and the follow-up recommendation. Both adapted models keep the location, uncertainty and follow-up, but translate “infiltrates” as “lung infections,” which is more specific than the original report. Both also use the redundant phrase “might be a possible.” This sentence contains both qualified negation and a separate uncertain finding. The baseline fails to keep that distinction, which could make the finding seem confirmed.

### Case 20: Incorrect explanation and examination limits

Source: **BIMCV-COVID19**. Identifier: `ses-E28411`.

**Original report:**

> Findings: 22-year-old woman with acute appendicitis. Positive PCR for coronavirus. Requested examination performed. Suboptimal study as both lung apices are not included. In the visualized lung parenchyma, no consolidations or other intraparenchymal pulmonary infiltrates are observed. No pleural effusion is evident. Cardiac silhouette and pulmonary hila are of normal morphology and size.

**Baseline:**

> A 22-year-old woman with acute appendicitis. Positive PCR for coronavirus.

**Full fine-tuning:**

> A 22-year-old woman has acute appendicitis, which is a type of apex infection. She tested positive for coronavirus, and was asked to have an examination done. The scan didn't show the top parts of both lungs, which is a bit of a problem. The lung tissue that can be seen in the image doesn't show any signs of pneumonia or other lung infections. There's no fluid around the lungs. The heart and the areas where the lungs connect to the heart look normal in shape and size.

**LoRA:**

> The patient is a 22-year-old woman who has a severe appendicitis. She has a positive PCR test for coronavirus. She wanted to have an examination done. The examination was not very good because both the top parts of the lungs are not included. The lungs look clear without any signs of pneumonia or other lung infections. There is no fluid around the lungs. The heart and the areas where the lungs connect to the heart are normal in shape and size.

**Analysis.** The baseline only includes the patient background and leaves out the imaging findings. Full fine-tuning incorrectly calls appendicitis “a type of apex infection,” although it correctly limits the negative lung findings to the visible regions. LoRA changes “acute” to “severe” and says that the patient requested the examination, which is not stated in the source. It also says “the lungs look clear” without limiting this to the visible areas, even though both lung apices were excluded. Both adapted outputs describe the findings as infections rather than consolidation or infiltrates. LoRA also leaves PCR unexplained. Here, the key risk is losing the limit of the examination: the source only describes the visible lung tissue, not both lungs in full.

### What the Cases Show

In these five examples, the baseline often leaves out most of the findings. Full fine-tuning and LoRA cover more of the reports but sometimes add unsupported medical statements. Cases 15, 19 and 20 contain incorrect or overly specific explanations. Case 14 shows repetition, and case 18 includes an unrelated phrase. Across the cases, problems arise around unfamiliar anatomy, medical terms, qualified negation and examination limits.

ROUGE measures overlap with reference text, so it may not detect changes in negation, severity or which parts of the body were actually examined. The higher validation ROUGE scores do not rule out the errors shown here. Because the cases were selected to illustrate problems, they cannot be used to estimate the overall error rate or rank the models by clinical safety.

## Recommendation and Limitations

Full fine-tuning gave the highest ROUGE scores and faster measured generation than LoRA in these runs. LoRA used less training memory and saved a **3.56 MB** adapter, compared with **990.35 MB** for the full model weights (decimal MB). The adapter still requires the base model for inference.

The completed runs met the data-separation, ROUGE and resource criteria. However, the five reviewed outputs contained factual and readability problems. Full fine-tuning is the better starting point for further evaluation based on its ROUGE scores and measured generation speed, but neither model is ready for independent patient-facing use. Until clinical validation is completed, any generated summary should be checked against the source by a qualified reviewer. Further work would need:

- Source checks for findings, location, negation and uncertainty, including unsupported additions and severity changes.
- Coverage checks for important findings, follow-up recommendations and examination limitations.
- Verified plain-language definitions, repetition checks and clinician review before release. Summaries that change negation, omit important findings or add unsupported claims should be corrected before use.

These checks were not implemented in this project. The experiments used one seed and one checkpoint/prompt, with different learning rates and no rank or hyperparameter search. I did not measure reading grade, patient comprehension or clinical outcomes. Longer reports may be affected by truncation, patient-level independence of the dataset splits was not verified, and possible pretraining contamination was not assessed. The selected examples were not independently reviewed by a clinician. A larger, representative sample with clinician ratings would be needed to assess factual accuracy and readability. Any future confidence filter would also need to be tested against clinician-labelled errors.

## Implementation and Reproduction

| File | Purpose |
| --- | --- |
| `dataset.py` | Load official splits and tokenize source/target pairs |
| `modules.py` | Load FLAN-T5, attach LoRA and count parameters |
| `train.py` | Evaluate/train models and save predictions, metrics, weights and curves |
| `predict.py` | Reload a run, generate explanations and measure inference |

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
| NLTK | 3.10.0; `punkt_tab` installed |
| NumPy / Matplotlib | 2.4.6 / 3.11.1 |

To install the direct dependencies into an environment with compatible Python and PyTorch:

```bash
python -m pip install transformers==4.57.6 peft==0.19.1 \
    datasets==5.0.0 accelerate==1.14.0 sentencepiece==0.2.2 \
    rouge-score==0.1.2 nltk==3.10.0 numpy==2.4.6 matplotlib==3.11.1
python -m nltk.downloader punkt_tab
```

### Training and Prediction

Run these commands from the **repository root**, using new output directories. Formal experiments require an allocated CUDA GPU; the default prediction device is CPU.

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
```

Training defaults match the experiment table and use complete official splits. Each run saves its configuration, metrics and validation predictions. Fine-tuning also saves `model/`, the latest epoch checkpoint, `trainer_state.json` and `training_curves.png` with loss and four ROUGE curves. Checkpoints are written after epoch validation; the CLI does not expose resume support.

Prediction requires the run configuration and, for adapted models, its saved `model/`. The custom input above corresponds to case 19. Use `--split validation` to generate and score validation reports; test/custom text has no ROUGE. Sampling uses the saved seed. `--output` needs a new file in an existing directory; omitting it prints the result. `--help` lists the remaining options.

### Experiment Records

Records are kept outside Git. Local paths below are relative to the repository root; remote paths are in my Rangpur account.

| Records | Location |
| --- | --- |
| Local baseline results | `../runs/rangpur-20261003/` |
| Local full/LoRA results | `../runs/rangpur-20261004-epochs2/` |
| Remote baseline and original logs | `/home/Student/s4983489/comp3710-ass2/runs/20261003-formal/` |
| Remote full/LoRA weights and checkpoints | `/home/Student/s4983489/comp3710-ass2/runs/20261004-epochs2/{full,lora}/` |

Trained weights and checkpoints are stored on Rangpur and are not included in this repository.

## Artificial Intelligence Usage Disclosure

I used OpenAI Codex to help with literature review, debugging, running experiments. I write the initial code by myself. I analyze the results and write this README.md by myself.

## References

1. COMP3710 / COMP7710 assessment specification, Sections 2.6, 3, 5.3 and 6.1, with subsequent instructor clarifications.
2. Chung et al. (2022), [*Scaling Instruction-Finetuned Language Models*](https://arxiv.org/abs/2210.11416), Section 5 and Appendices A.1 and G.2.
3. Hu et al. (2021), [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), Sections 4.1 and 7.
4. [BioLaySumm LaymanRRG dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track) and [FLAN-T5-base model card](https://huggingface.co/google/flan-t5-base).
5. Hugging Face documentation: [Datasets loading](https://huggingface.co/docs/datasets/loading), [summarisation](https://huggingface.co/docs/transformers/tasks/summarization), [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), [generation](https://huggingface.co/docs/transformers/main_classes/text_generation), [PEFT](https://huggingface.co/docs/peft/main/en/quicktour) and [checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint).
6. Google Research, [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge).
