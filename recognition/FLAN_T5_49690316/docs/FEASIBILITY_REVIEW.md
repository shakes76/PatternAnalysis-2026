# Feasibility review

**Adin Sreekesh — s4969031 | FLAN-T5, Hard difficulty**

Completed and discussed with the tutor at the compulsory check-off on
**23 October 2026**.

### User need, scope and acceptance criteria

This project aims to help clinical reviewers explain radiology reports to
patients. The prototype will produce plain-language drafts for a reviewer to
check before use. The main question is whether fine-tuning makes these
explanations clearer without introducing incorrect findings, leaving out
important details, or changing the report's uncertainty.

I will consider the prototype successful if it:

1. Improves all four required ROUGE metrics over the baseline model.
2. In five audited cases, the selected model does not introduce any unsupported
   findings and preserves negation and uncertainty (suspected illness versus
   confirmed illness).
3. In those cases, all the important findings are preserved and medical terms
   are explained in simple language. Any remaining problems are documented.
4. Trains within one A100's 40 GB GPU memory in under 12 hours, with memory use,
   training time and generation speed reported.
5. Reloads saved weights and generates summaries through a documented
   prediction script.

### Model choice and course concepts

I will compare untuned FLAN-T5-small with LoRA and full fine-tuning using the
BioLaySumm 2025 LaymanRRG dataset. FLAN-T5 reads the report and an instruction,
then generates an explanation. LoRA trains a small set of added weights, while
full fine-tuning updates the whole model. I chose FLAN-T5 because its
encoder–decoder structure suits report-to-summary generation. GPT-2 is
decoder-only and would require a different input format and training-loss
setup. FLAN-T5-small also makes local development and small training experiments
feasible on my Mac.

The project applies several course concepts. Embeddings turn text tokens into
numerical representations, while attention helps the model use context from
across the report when generating an explanation. During training,
cross-entropy loss measures how well its predictions match the reference
summary and guides weight updates.

### Preliminary feasibility evidence

The data audit removed 138 training rows that shared image/session identifiers
with validation rows. Grouped splitting produced 142,801 training and 7,515
development examples without shared individual identifiers. Patient-level
separation remains unverified, as different reports may refer to the same
patient. Since the supplied test summaries are blank, I have reserved the
supplied validation set for final evaluation.

Training, prediction, checkpoint loading and interrupted-run recovery have
worked. Three-epoch trials using 10,000 training and 1,000 development examples
produced:

| Measurement | Baseline | LoRA | Full |
|---|---|---:|---:|
| Development ROUGE-Lsum | 0.285 | 0.622 | 0.678 |
| Final-epoch training time | Not applicable—no adaptation training | 526 s | 675 s |
| Final-epoch peak allocated GPU memory | Not measured during training | 4.21 GiB | 5.04 GiB |

These results support technical feasibility and improved reference overlap, but
do not necessarily establish medical accuracy. The methods used different
learning rates and received unequal tuning effort. Increasing LoRA's learning
rate from 0.0001 to 0.001 improved its development results.

### Risks, budget and fallback

The main risks include incorrect or missing findings, unclear wording,
imperfect reference summaries and outputs being cut short. Repeated reports and
unverified patient separation also limit what the results can establish.

The 10,000-example trials used one A100 with a three-hour allocation per
training job. Full-data runs request up to 12 hours each, with regular recovery
checkpoints and separate evaluation jobs.

The next planned experiment is to compare both methods on the full prepared
training set, using development loss to select checkpoints.

If resources become limited, I will retain FLAN-T5-small and the documented
training subset. If important factual errors remain, I will recommend further
development and mandatory human review.
