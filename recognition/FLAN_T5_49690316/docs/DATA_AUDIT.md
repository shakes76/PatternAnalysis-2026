# LaymanRRG data audit

## Dataset
BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track

Fields: source, images_path, radiology_report, layman_report.

## Original splits
- Training: 150,454 rows.
- Validation: 10,000 rows.
- Test: 10,537 rows.

Training and validation have no missing or blank reports or summaries.
All test reference summaries are blank, so local reference-based
evaluation cannot use the supplied test split.

## Overlap findings
- Training and validation share 2,148 distinct report texts.
- Many shared texts are routine findings, such as "No abnormalities."
- Training and validation share 138 individual image identifiers.
- No individual identifiers overlap with test in our checks.

## Preparation
Remove training rows sharing any source-qualified image or session
identifier with validation. Parse OpenI image lists before comparison.

Filtered training size: 150,316 rows.
Remaining training/validation identifier overlap: 0.

Retain repeated report wording because identical text alone does not
establish that records belong to the same patient.

## Remaining work
- Investigate patient-level identifiers or metadata.
- Create a development split that keeps related records together.
- Keep supplied validation untouched for final scored evaluation.
- Record the dataset revision for reproducibility.

Image/session identifier separation has been verified.
Patient-level separation has not been verified.



## Untuned FLAN-T5-small smoke test

- Device: Apple MPS on an M4 Mac with 16 GB unified memory.
- Model: google/flan-t5-small.
- Total parameters: 76,961,152.
- Generation: do_sample=False, max_new_tokens=128.
- Example: first row of filtered training data.

Input report:
No infiltrates or consolidations are observed in the study.

Generated output:
Report: No infiltrates or consolidations are observed in the study.

The pipeline successfully generated text, but this example was copied
rather than rewritten in plain language. This is a single-example
functional check, not a held-out performance evaluation.



## Training computation smoke test

FLAN-T5-small successfully processed a two-example batch on Apple MPS.

- Target padding used -100 so padded positions were excluded from loss.
- Evaluation-mode forward loss: 3.2585.
- Training-mode backward loss: 3.4796.
- Shared embedding gradient norm: 0.01508.
- One AdamW update completed with learning rate 0.0001.
- Encoder final-layer normalisation parameter change norm: 0.002263.

These checks demonstrate working forward, backward, and optimiser
operations. They do not measure held-out performance or establish
that summary quality improved. Updated weights were not saved.


## Two-example overfit diagnostic

Full fine-tuning was tested on two training examples using AdamW,
a learning rate of 0.0001, and a fixed seed of 42.

After 30 updates:
- The first example reproduced its reference and generated EOS.
- The second produced repetitive text and reached the 128-token cap
  without generating EOS.

After 100 updates:
- Both examples reproduced their reference summaries.
- Output lengths were 18 and 83 tokens, including EOS.
- Both generated EOS within the output budget.

This verifies that the pipeline can memorise and independently generate
the two training targets. It does not measure generalisation or clinical
factual fidelity.



## Checkpoint round-trip verification

The two-example diagnostic model and matching tokeniser were saved
under checkpoints/tiny_overfit, which is excluded from Git.

A separate predict.py process loaded the checkpoint and reproduced
both memorised reference summaries. This verifies local checkpoint
saving and loading; it is not held-out evaluation.




## Training and development split

Filtered training records were grouped by connected, source-qualified
image/session identifiers. OpenI lists were parsed into individual paths.

Groups were shuffled using seed 42, with 5% of groups reserved for
development.

- Training: 142,801 rows.
- Development: 7,515 rows.
- Training/development individual identifier overlap: 0.

The supplied validation split is reserved for final scored evaluation.

| Source | Training | Development | Final evaluation |
|---|---:|---:|---:|
| BIMCV-COVID19 | 29,686 (20.79%) | 1,540 (20.49%) | 2,042 (20.42%) |
| OpenI | 2,123 (1.49%) | 120 (1.60%) | 134 (1.34%) |
| PadChest | 110,992 (77.72%) | 5,855 (77.91%) | 7,824 (78.24%) |

Source proportions are similar across splits. Overall and per-source
metrics will be reported with sample counts.

This procedure verifies image/session identifier separation.
Patient-level separation remains unverified.



## Full fine-tuning pilot

FLAN-T5-small was trained on 128 training examples for two epochs.
Checkpoint selection used loss on 32 development examples.

- Initial development loss: 3.0052.
- Epoch 1 development loss: 2.5016.
- Epoch 2 development loss: 2.3554.
- Epoch 2 was selected.

Both the untuned baseline and selected pilot checkpoint were evaluated
on the same 32 development examples using deterministic generation
with a 256-new-token cap.

| Metric | Baseline | Pilot |
|---|---:|---:|
| ROUGE-1 | 0.2868 | 0.4537 |
| ROUGE-2 | 0.0750 | 0.2260 |
| ROUGE-L | 0.2525 | 0.4010 |
| ROUGE-Lsum | 0.2565 | 0.4221 |

Scores are mean per-example F1 values, using stemming and sentence
segmentation for ROUGE-Lsum.

All metrics improved, but inspected outputs still retained jargon
and sometimes used awkward wording. References also contained
questionable simplifications.

These are development results, not final held-out evaluation.
The generation cap may truncate outputs.



## LoRA pilot

LoRA targeted T5 query and value projections using rank 8,
alpha 16, dropout 0.05, and no trainable biases.

- Total parameters: 77,305,216.
- Trainable parameters: 344,064 (0.45%).

Pilots used 128 training examples, 32 development examples,
batch size 2, and two epochs.

At learning rate 0.0001, final development loss was 2.8041.
At learning rate 0.001, final development loss was 2.4511.

| Metric | Baseline | Full pilot | LoRA, learning rate 0.001 |
|---|---:|---:|---:|
| ROUGE-1 | 0.2868 | 0.4537 | 0.4709 |
| ROUGE-2 | 0.0750 | 0.2260 | 0.2401 |
| ROUGE-L | 0.2525 | 0.4010 | 0.4207 |
| ROUGE-Lsum | 0.2565 | 0.4221 | 0.4377 |

These comparisons used a 256-new-token generation cap.
LoRA received additional learning-rate tuning, so this is not
evidence that it is intrinsically better than full fine-tuning.

Inspected LoRA outputs still retained jargon and could omit or
blur findings.

## Recovery verification

A local LoRA run was interrupted and resumed from epoch 1,
batch offset 50. It completed the remaining batches and saved
an epoch-boundary recovery checkpoint.

Recovery includes model weights, optimiser state, random state,
batch position, and accumulated loss totals. Functional recovery
was verified; numerical equivalence to uninterrupted training
has not been established.

## Rangpur verification

- GPU: NVIDIA A100-PCIE-40GB.
- PyTorch: 2.13.0+cu130.
- CUDA tensor operation succeeded.
- A two-epoch LoRA pilot completed successfully in 43 seconds.
- Development loss decreased from 3.0052 to 2.4900.
- The dataset preparation reproduced the local split sizes.

The working Slurm configuration uses account comp3710 and
partition comp3710. Explicit system-memory requests were omitted
because the partition advertised 1 MB per node and rejected
the normal request.

A larger development run was submitted as job 659655:
10,000 training examples, 1,000 development examples, three
epochs, learning rate 0.001, and batch size 2.

Its results are pending.



## Split limitation

Splits are separated by source-qualified image/session identifiers.
Patient-level separation remains unverified because the released
LaymanRRG table does not include an explicit patient identifier.

Results therefore measure performance on identifier-separated
examples, not verified unseen patients.