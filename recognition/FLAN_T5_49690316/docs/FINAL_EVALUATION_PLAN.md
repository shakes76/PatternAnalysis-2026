# Final evaluation plan

Status: planned; final evaluation has not been run.

## Models

Compare:
- Untuned google/flan-t5-small.
- Full-data LoRA checkpoint: checkpoints/lora_all.
- Full-data full fine-tuning checkpoint: checkpoints/full_all.

For each trained model, use the checkpoint with the lowest development
loss from the three-epoch run. Confirm training completion and record
the selected epoch before evaluating.

## Data

Use all 10,000 examples in the supplied validation split, reserved
throughout development for final evaluation.

Use the same examples and order for every model.
The supplied test split cannot support reference-based evaluation
because its reference summaries are blank.

Image/session identifier overlap with training was removed.
Patient-level separation remains unverified.
Repeated report wording is retained and documented.

## Input and generation

Use the instruction recorded in config.py without further changes.
Preserve complete input reports without truncation.

Use greedy decoding with:
- do_sample=False
- num_beams=1
- max_new_tokens=1024
- min_new_tokens=0
- repetition_penalty=1.0
- no_repeat_ngram_size=0

Record generated lengths and whether outputs end with EOS.
Report outputs that reach the generation cap.

## Measurements

Report mean per-example F1 for ROUGE-1, ROUGE-2, ROUGE-L,
and ROUGE-Lsum, using stemming and the existing sentence splitting.

Report overall scores and scores for each source, with sample counts.

Measure generation throughput and peak allocated CUDA memory
under a documented timing procedure. Report training costs separately.

Save evaluation settings and per-example predictions.

## Qualitative analysis

Select 3–5 representative final failure cases using a documented
selection process. Compare all models against the source reports for:
- Unsupported claims.
- Important omissions.
- Changed negation or uncertainty.
- Unexplained jargon and confusing wording.

Include source excerpts supporting each judgment.
Describe this as a student source-faithfulness audit, not expert
clinical validation. Do not estimate population error rates from
a small selected case sample.

## Interpretation

Use final results for reporting and the engineering recommendation,
not further tuning. Higher ROUGE alone will not establish clinical
safety or suitability for independent patient-facing use.