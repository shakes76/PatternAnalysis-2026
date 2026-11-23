# Final evaluation: source-faithfulness review

## Scope and selection

This review compares the final full-data models on five reserved-evaluation
records. The selection rule was the lowest full-model ROUGE-L case in each
source, followed by the two lowest remaining cases with distinct report text.
Ties were resolved by subset index. This is a deliberately low-overlap sample,
not a random sample or an estimate of clinical error frequency.

Judgments below compare wording with the supplied source report. No images,
patient records, or expert clinical assessment were used. Patient-level split
separation remains unverified. Final results have not been used to tune models,
prompts, or decoding settings. The raw records remain in ignored local results.

## Case 1152 — PadChest: a faithful paraphrase with zero ROUGE-L

The source says clinical data was not provided for evaluation. The full model
preserves that statement concisely. LoRA also conveys the missing-data message,
although its wording is less natural. Neither adds clinical findings.

The reference uses different vocabulary and expands the purpose of the missing
information. The full output's zero ROUGE-L does not show a factual failure;
the main message is retained despite little matching vocabulary.

Classification: lexical metric limitation, not an observed factual error.
Both models generate EOS below the cap (LoRA 10 tokens; full 9).

## Case 8820 — BIMCV-COVID19: reference conflicts with source

The report lists an enlarged heart, prior surgical changes, a pacemaker,
upper-lung opacities, possible edema or infection, and recommended imaging
follow-up. Its reference instead says the study found nothing important.
The reference does not represent the supplied findings.

Both trained models retain the main findings and preserve uncertainty between
fluid buildup and infection. Full fine-tuning describes follow-up as another
imaging test; LoRA's follow-up wording refers awkwardly to a radiology report.
Their surgical descriptions could be clearer about prior surgery rather than
implying a current wound. These wording concerns do not explain the extremely
low scores; the contradictory reference does.

The baseline copies the technical report. Relative to that baseline, both
adaptations explain several technical terms. There is no source-based basis
here for preferring the reference's dismissal of the findings.

Classification: source/reference mismatch, with minor clarity concerns.
Both models generate EOS below the cap (LoRA 84 tokens; full 81).

## Case 8357 — OpenI: placeholder copying, contradiction, and degeneration

The source has missing-text placeholders but explicitly states large fluid
collections on both sides, possibly related to pleuritis or known pancreatitis.

The full model degenerates into a long sequence of placeholder characters.
It generates 1,024 tokens without EOS and reaches the output cap. It fails to
convey the substantive findings. This is confirmed generation failure, not
merely poor reference overlap. The input is only 108 tokens, so it exceeds
neither the tokenizer's listed 512-token length nor our preserved input range;
input length alone cannot explain this failure.

LoRA stops after 64 tokens, but states both absence and presence of fluid around
the lungs. Its absence claim conflicts with the source's explicit bilateral
large effusions. It also changes known pancreatitis to suspected pancreatitis,
does not clearly retain bilateral extent, leaves jargon unexplained, and repeats
normal-lung wording. Normal lung appearance does not justify negating the
separately reported pleural finding.

The baseline largely copies the damaged source and does not simplify it. The
reference interprets missing passages, so its details also need caution.

Classification: severe degeneration and omissions (full); contradiction,
changed certainty, incomplete location, and poor clarity (LoRA).
LoRA's ROUGE-L of 0.545 does not establish correctness. Full's score is 0.032.
Placeholder-related vulnerability is a hypothesis; causation is unproven.

## Case 2721 — PadChest: preserved finding, inadequate simplification

The source is a one-word diagnosis. The full model retains that diagnosis but
does not explain the medical term. LoRA adds that it affects the lungs, but also
retains the unexplained term rather than providing the reference's everyday
description. Neither introduces an observed contradiction of the source.

The full model scores zero ROUGE-L because it shares no scored words with the
short reference. Its factual preservation and its failure to simplify should
be assessed separately.

Classification: readability/task-completion limitation and short-reference
metric sensitivity; not an observed invented finding.
Both models generate EOS below the cap (LoRA 15 tokens; full 8).

## Case 2958 — PadChest: reference language mismatch

The source and predictions are English, but the reference is Chinese. English
predictions are therefore poorly served by word-overlap scoring against this
target. Both trained outputs score zero ROUGE-L.

The full output explains degeneration as wear-and-tear changes and retains
severity and approximate location. LoRA describes slight spinal changes but
does not explain their degenerative nature. Both retain air trapping and locate
it in the lungs, which is not explicitly named in the short source sentence.
This is an added contextual interpretation rather than a finding we can verify
from an image. There is no evidence of an output-cap problem.

Classification: target-language mismatch, with differences in explanatory
specificity. Zero ROUGE alone is not evidence of failed source preservation.
Both models generate EOS below the cap (LoRA 29 tokens; full 32).

## Synthesis and report implications

| Case | Dominant issue | Is low ROUGE evidence of factual failure? |
|---|---|---|
| 1152 | Different wording for the same missing-data message | No |
| 8820 | Reference contradicts substantive source findings | No |
| 8357 | Placeholder degeneration and source contradictions | Yes, supported independently by source comparison |
| 2721 | Diagnosis retained without a plain-language explanation | Primarily a readability issue |
| 2958 | English output compared with a Chinese target | No |

The findings support reporting ROUGE as reference overlap rather than medical
accuracy. Full fine-tuning has the strongest aggregate overlap and fewer cap
hits, but its severe OpenI failure prevents a claim of universal reliability.
LoRA can stop normally while still contradicting a source finding, so EOS and
cap diagnostics are not factual-safety tests.

Potential guardrails include flagging placeholder-heavy inputs, empty outputs,
repetition, and cap hits; checking important entities, side labels, and certainty;
and requiring human source comparison. These are proposals, not implemented or
validated safety mechanisms. Reference-language and pairing audits would help
interpret scores, but current final data and headline scores must remain
unchanged. A cleaned-data analysis would be a separately declared experiment.

The five-case low-overlap review does not supply five confirmed model failures.
The separately selected cap-hit cases below provide four failure autopsies,
including the already reviewed case 8357. Correct paraphrases and reference
defects are not relabelled as factual model errors. No further model tuning is
performed from these final findings within the frozen experiment.

## Supplementary selection: four cap-hit failure autopsies

Rule: scan records in their saved final-evaluation subset order and take the
first full-model cap hit from each source. Then add the first remaining cap hit
with a distinct source-report text, bringing the total to four. This gives
indices 246, 674, 8357, and 2431. Source coverage is deliberate, not proportional
sampling. Full-model cap hits comprise 20 BIMCV-COVID19, 15 PadChest, and one
OpenI record. These counts describe cap hits, not factual-error prevalence.

Aligned source, reference, and all model outputs are saved locally in
`results/final_cap_hit_review.jsonl`. Long repeated outputs are described below
rather than reproduced in public documentation.

### Case 246 — BIMCV-COVID19: invented diagnosis and lost addendum

The report describes stable bilateral opacities and support-device positions,
plus an addendum describing air around the mediastinal/pericardial contour and
in the left chest wall extending toward the neck. It does not state COVID-19.

Full fine-tuning invents a COVID-19 diagnosis and a net-like pattern, adds
unsupported negative findings and normal-heart statements, and misrepresents
the chest-wall air finding as collapsed lung tissue. It then repeats diagnostic
phrases to the 1,024-token cap without EOS. LoRA stops after 104 tokens but
repeats vague stability statements and does not communicate the key addendum
findings. It also treats administrative placeholder text as substantive content.
The baseline extracts the addendum's technical findings but leaves them
unexplained and omits other report content.

Input length is 184 tokens, ruling out a simple explanation based on exceeding
512 input tokens. Administrative boilerplate and duplicated text are possible
contributors, not proven root causes. The source dataset label is not supplied
as a model input and must not be used to justify the invented diagnosis.

Autopsy: unsupported diagnosis, anatomical substitution, omissions, and
repetition. A proposed review workflow should verify addendum coverage and
reject repetitive drafts, rather than accept diagnosis wording merely because
it appears plausible for the dataset.

### Case 674 — PadChest: fabricated explanation from an empty-content label

The source contains only an interpretation label, without clinical findings.
Its detailed reference is unsupported by that input.

Full fine-tuning fabricates fluid around the heart and possible heart failure,
then repeats normal-heart wording until the 1,024-token cap. LoRA stops after
19 tokens but invents a reassuring absence of pneumonia or other lung issues.
The baseline produces only a report label. Both trained models therefore
invent clinical content, even though only one triggers a cap warning.

Input length is 31 tokens. The trigger is an insufficient-information case,
not a long report. The dataset's upstream pairing/extraction error remains
unverified. Input-quality detection and an explicit insufficient-information
response are proposed safeguards; neither is validated by this analysis.

Autopsy: input/reference defect, unsupported abnormal findings (full), and
unsupported reassurance (LoRA). This illustrates why a normal stopping token
cannot establish factual safety.

### Case 8357 — OpenI: missing-text degeneration and contradictory fluid claims

See the detailed review above. Full fine-tuning emits placeholder characters
to the cap from a 108-token input. LoRA stops normally but contradicts the
explicit bilateral-fluid finding and changes known pancreatitis to suspected.
This is a third confirmed output-failure case, with distinct placeholder and
uncertainty problems.

### Case 2431 — BIMCV-COVID19: cross-structure substitutions in a long CT report

The 638-token source covers chest, abdomen, and pelvis findings and includes
lymph-node measurements, a suspected calcified hepatic microgranuloma, a patent
portal vein, absence of biliary dilation, and a qualified conclusion.

Full fine-tuning repeats abdominal findings, relocates the suspected calcified
finding toward a large vein near the heart, and fails to preserve all node
measurements accurately. It reaches 1,024 output tokens without EOS and ends
mid-conclusion. LoRA also reaches the cap, changes the suspected calcified
finding into a negative claim, substitutes bile-duct wording for the portal
vein finding, and repeatedly equates a bile duct with a calcified finding.
The baseline gives a short, incomplete statement about lung opacities rather
than an adequate explanation of the multi-region report.

Both models receive the full input: the current evaluator does not request
input truncation. Report length, many anatomical entities, and repeated output
are observed characteristics, but no causal experiment identifies which
produced these errors. The reference also has location wording that should not
be accepted without source comparison.

Autopsy: wrong anatomical association, altered certainty/negation, omitted
measurements, repetition, and an unfinished conclusion. Proposed guardrails
include entity/measurement checks and explicit detection of unfinished output;
these require validation and do not replace human review.

## Recommendation supported by the audit

Full fine-tuning is the stronger measured reference-overlap model and has fewer
cap hits, at higher training time and peak allocated memory. The selected cases
nevertheless show severe factual errors in both methods, including normally
terminated LoRA outputs. Neither should be recommended for independent
patient-facing use on this evidence. A restricted drafting prototype would
require human comparison with the source and validated input/output screening.
These four deliberately chosen cases support failure diagnosis, not a numerical
claim about the models' overall clinical error rates.
