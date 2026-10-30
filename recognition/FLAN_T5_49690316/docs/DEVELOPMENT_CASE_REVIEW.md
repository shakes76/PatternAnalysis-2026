# Development prediction review

## Failure candidate 30 — PadChest: placeholder input

Selection: one of the lowest remaining full-model ROUGE-L cases after source
coverage selection.

- **Source:** contains a bracketed translation placeholder rather than a clinical
  report. No medical conclusion can be grounded in this input.
- **Reference:** contains a long sequence of examination narratives with
  differing findings and recommendations. None is supported by the supplied
  placeholder. Concatenation or an upstream processing problem is a hypothesis,
  not an established cause.
- **Baseline and LoRA:** copy the placeholder. Their output is unusable as a
  patient explanation, but adds no medical findings.
- **Full fine-tuning:** invents a change in lung condition and infection. Neither
  is supported by the input. Its slightly higher ROUGE than LoRA does not make
  this output more faithful.

Failure categories: placeholder input, unsupported reference, hallucinated
finding, and failure to flag insufficient information.

Operational implication: detect report placeholders before generation and
request a valid source report. This is a proposed guardrail; it has not been
implemented or evaluated. A placeholder detector cannot detect every invalid
input or ensure clinical safety.

## Summary of the five ranked candidates

| Index | Main issue | Interpretation |
|---|---|---|
| 870 | Repetition, omissions, unsupported relationships | Both trained outputs fail on a detailed source |
| 974 | Label-only source and unsupported reference | Trained outputs invent numerical meaning |
| 640 | Wrong anatomy/side, omissions, repetition | Higher full-model ROUGE still accompanies factual errors |
| 841 | Apparent source/reference mismatch | Low ROUGE is not evidence of factual failure here |
| 30 | Placeholder source and unsupported reference | Full model invents a clinical finding; other models copy the placeholder |

These candidates were intentionally selected for low reference overlap with
source coverage. They are not representative of the dataset's error frequency.
They show the need to distinguish reference quality, input quality, readability,
and source faithfulness rather than combine them into a single ROUGE conclusion.
They do not establish the behaviour of the pending full-data models.

## Candidate 841 — BIMCV-COVID19: apparent source/reference mismatch

Selection: one of the lowest remaining full-model ROUGE-L cases after source
coverage selection. This candidate is not a confirmed factual model failure.

- **Source:** states that the AP chest examination has no significant findings.
- **Reference:** describes left-lower-lung atelectasis and comparison with a
  previous examination, neither of which is present in the supplied source.
  The reason for this mismatch cannot be established from the row alone.
- **Baseline:** preserves the general negative statement while retaining a label.
- **LoRA:** gives a concise, broadly source-consistent explanation of no major
  issues. It does not copy the unsupported reference findings.
- **Full fine-tuning:** preserves the general negative statement, but renders
  the AP examination label awkwardly as a front-chest description.

Interpretation: low ROUGE primarily reflects disagreement with an apparently
unsupported reference. It should not be counted automatically as a factual
model failure. The available text does not justify making claims about the
actual image or determining which upstream record is correct.

Operational implication: score reference overlap separately from source
faithfulness and document mismatched targets. Do not change current final
evaluation data after seeing this development result. Any systematic cleaning
rule should be independently specified and evaluated as a separate experiment.

## Failure candidate 640 — OpenI

Selection: lowest full-model ROUGE-L within OpenI among the 1,000 development
examples. The input contains repeated findings/impression sections and missing
text marked with placeholders. These are observed input characteristics, not
established causes of the model errors.

- **Baseline:** produces a sentence equating the breastbone with a fluid finding,
  confusing distinct anatomical and finding concepts.
- **LoRA:** relocates a rib fracture to the lung, introduces unsupported bilateral
  fracture wording, repeats extensively, and drifts between right and left.
  It also retains a missing-text placeholder in the patient-facing output.
- **Full fine-tuning:** replaces the main seventh-rib fracture with a fluid
  description, changes the healing eighth-rib finding from right to left, and
  shifts the possible breastbone defect toward a lung finding. It introduces
  causal wording where the source states association and repeats findings.
- **Reference:** includes duplicated technical and lay explanations and assigns
  interpretations to some missing-text passages. It should not be treated as
  sufficient evidence that an output preserves the available source text.

Failure categories: anatomical substitution, wrong-side attribution, omitted
fracture, unsupported causal relationship, repetition, and placeholder handling.

Full fine-tuning has higher ROUGE-L than LoRA in this case (0.345 versus 0.160),
but its output still changes important source facts. The score difference
therefore does not establish clinical acceptability.

Operational implication: preserve anatomical entities, side labels, and
uncertainty explicitly during review. Missing source text should be flagged,
not guessed. Entity/side consistency checks and placeholder detection are
possible guardrails requiring separate implementation and validation; they
cannot by themselves establish factual safety.

## Failure candidate 974 — PadChest

Selection: lowest full-model ROUGE-L within PadChest among the 1,000 development
examples. This is a source/target quality problem as well as a generation failure.

- **Source:** contains only a report label and the number 12. There are no clinical
  findings available to simplify. Whether this arose from extraction, pairing,
  or upstream data preparation cannot be established from this row alone.
- **Reference:** supplies numerous clinical claims unsupported by the source.
  It also describes both absence of fluid around the lungs and a small left-sided
  fluid finding without clarifying that apparent inconsistency.
- **Baseline:** copies the uninformative input. This is not a useful patient
  explanation, but it adds no clinical findings.
- **LoRA:** interprets 12 as the number of findings, which the source does not state.
- **Full fine-tuning:** invents a dimensional measurement and units absent from
  the source.

Failure categories: insufficient source information, unsupported target,
unsupported numerical interpretation, and invented measurement.

Low ROUGE here is not reliable evidence of poor summarisation: matching the
reference would require claims unavailable in the input. The trained outputs
still introduce unsupported information independently of their ROUGE scores.

Operational implication: a draft should not be generated as a clinical
explanation when the input contains only an identifier or report label.
Input-quality screening and an insufficient-information response are proposed
guardrails requiring validation. Retain this case in the current evaluation and
report the defect; do not remove inconvenient cases after inspecting scores.
Any later cleaned-data experiment should use a documented rule applied
consistently and be reported separately.

## Failure candidate 870 — BIMCV-COVID19

Selection: lowest full-model ROUGE-L within this source among the 1,000
development examples. This deliberately selected example cannot establish an
error rate. It comes from the 10,000-example training experiments, not the
pending full-data models.

- **Baseline:** produces whitespace rather than a usable explanation.
- **LoRA:** enters a repetition loop. It attributes imaging findings to the PCR
  test and invents causal relationships between spinal findings that are not
  stated in the source. It fails to communicate the regional lung findings,
  severity assessment, and other findings coherently.
- **Full fine-tuning:** begins with some reported symptoms but then repeatedly
  restates breathing difficulty. It omits the positive PCR result, lung opacity
  locations, moderate severity assessment, and imaging conclusion, among other
  source information.
- **Reference quality:** the target contains a non-English fragment and wording
  that extends spinal degenerative findings to the ribs despite the source
  reporting no rib findings. The reference is not an independent clinical gold
  standard.

Failure categories: repetitive generation, major omissions, unsupported
relationships, misattributed evidence, and an unusable baseline output.

The source is long and highly structured. This makes long-input handling and
decoding behaviour worth investigating, but the root cause is not established.
Neither input truncation nor an output-cap hit has been verified for this
historical case; its predictions lack token-level stopping diagnostics.

Operational implication: flag blank or heavily repetitive drafts for human
review. Such screening can detect unusable text but cannot verify medical
facts. It is a proposed guardrail, not an implemented or validated safety
mechanism. Do not change frozen final decoding settings based on this case
without documenting a separate development experiment.

These are preliminary language and source-faithfulness observations from the
10,000-example training experiments. They are not expert clinical validation
or final held-out evaluation. Source reports are the basis for checking claims;
reference summaries are comparison material and may contain imperfect wording.

## Case 281 — PadChest

### Observations

- The baseline copies the technical report, including an unnecessary report label.
- LoRA and full fine-tuning produce identical output for this case.
- Both adapted outputs retain the negative statement and shorten the wording.
- The wording about solid areas remains vague and may be confusing to a lay reader.
- The adapted models interpret imaging findings in terms of infection rather than
  simply describing the reported absence of infiltrates and consolidation. This
  is a possible shift in meaning that needs expert review; it is not established
  here as a definite hallucination.
- The reference also interprets the findings using fluid and pneumonia language.
  Agreement with the reference therefore cannot independently prove fidelity to
  the source report.

### Preliminary conclusion

This case shows a readability improvement over copying, but also a remaining
clarity issue and a possible semantic shift. There is no observable advantage
of full fine-tuning over LoRA on this individual example. It is insufficient
evidence to compare overall clinical safety or estimate error rates.

### Selection and limits

Index 281 is one of the five development positions selected using seed 42 in
`inspect_predictions.py`. It is a randomly inspected case, not a case selected
because of a known failure. The displayed source is PadChest. Patient-level
separation remains unverified. These notes deliberately omit verbatim clinical
text; the local ignored prediction files retain the original material.


## Comparison of five randomly selected development cases

Cases were selected using seed 42 before reviewing their outputs.
These are source-text comparisons, not expert clinical validation.

| Case | LoRA | Full fine-tuning |
|---|---|---|
| 654 | Loses the heart-to-chest ratio relationship and the specific anatomical angle. Preserves uncertainty. | Preserves those relationships more clearly, but retains “blunting.” Preserves uncertainty. |
| 114 | Incorrectly attributes wheezing to the X-ray; retains unexplained dysesthesia. Preserves the positive PCR result. | Incorrectly attributes wheezing to the X-ray, substitutes difficulty breathing for dysesthesia, and omits the positive PCR result. |
| 25 | Clear explanation preserving the main findings. | Clear explanation preserving the main findings. |
| 759 | Preserves the negative finding clearly. | Preserves the negative finding clearly. |
| 281 | Vague wording and a possible shift from imaging findings to infection language. | Identical output and concerns. |

Full fine-tuning achieved higher aggregate ROUGE, but was not more
faithful in every inspected case. Case 114 demonstrates that a
readable output can change symptoms, omit a test result, and
misattribute how a finding was obtained.

Five cases are insufficient to estimate error rates. Final evaluation
still requires a separately documented case audit.
