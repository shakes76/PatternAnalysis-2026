# GFNet for Alzheimer's Classification on ADNI Brain MRI

**Manav Sachdeva (s4967283).** COMP3710 Pattern Analysis Project.
Project 2.1 Classification. Model: GFNet. Difficulty: Hard.

---

## The problem

Alzheimer's disease destroys brain tissue. The damage shows up on a structural
MRI scan. The hippocampus gets smaller. The ventricles get larger and fill the
space. The cortex gets thinner. If a computer can find those changes early, a
clinician can start treatment sooner.

This project reads one 2D slice of a T1-weighted brain MRI. It puts the slice
in one of two classes: AD (Alzheimer's Disease) or NC (Cognitively Normal). The
data comes from the ADNI study. ADNI is a North American research programme
that has collected brain scans since 2004.

The hard part is not the difference between a sick brain and a healthy brain.
The hard part is that healthy brains differ a lot from person to person.
Healthy brains also get smaller with normal age. The model must separate "this
brain is sick" from "this brain belongs to a different person".

### The question this report answers

The assignment asks for more than accuracy. The open research question for this
field is about confidence.

> Does the confidence of the model match whether the model is correct? Or does
> the model stay confident when it is wrong? Can a confidence threshold
> automate the easy cases and send the unsure cases to a specialist?

A screening tool with 74% accuracy that knows when it is unsure is more useful
than one with 74% accuracy that always sounds certain. This report measures the
confidence of both models. It then says plainly whether you can trust that
confidence.

---

## How GFNet works

A Vision Transformer cuts an image into patches. It then uses self-attention,
so every patch can exchange information with every other patch. That global mix
is the strength of a transformer. But attention costs about n² operations for n
patches.

GFNet keeps the global mix and removes the attention. Each block does three
steps:

1. Take the 2D real FFT of the token grid. The result is complex.
2. Multiply the result by a learnable complex filter of the same shape. These
   weights are the parameters that the model learns. They are frequency gains.
3. Take the inverse FFT back to normal space.

The convolution theorem says that a multiply in the frequency domain equals a
convolution across the whole grid in normal space. So one multiply lets every
patch see every other patch. The cost is n log n instead of n².

### Why this architecture suits this disease

Alzheimer's atrophy is large-scale change in structure. The ventricles get
larger, the cortex gets thinner, the hippocampus gets smaller. These are
low-frequency features.

A model whose parameters are frequency weights is a good match. You can also
look inside it. `predict.py` plots the magnitude of each learned filter, so
this report can show which frequencies the model chose. A convolutional network
gives no equivalent picture.

### One detail that is easy to get wrong

A real-valued input has a symmetric spectrum. `torch.fft.rfft2` uses that fact
and returns only half of the frequencies. For a 14 × 14 token grid the output is
14 × **8**, not 14 × 14.

The learnable filter must match that shape. If the shape is wrong you get a
shape error, or worse, a silent broadcast that trains badly for no visible
reason. `modules.py` holds a self-test for this. The test sets every filter
weight to 1 + 0i and checks that the block returns its input without change. It
passes to within 4.8e-07, which is float32 rounding error.

![GFNet learned frequency filters](figures/filters_gfnet.png)

*Each panel is the learned filter magnitude of one block. Top-left is low
frequency, which means large structures. Bottom-right is high frequency, which
means fine texture.*

---

## Results

All numbers come from the held-out test split: 9,000 slices from 450 patients.
No model saw them during training or model selection.

Three models are reported, because they are good at different things and the
difference between them is this report's main finding.

| | BaselineCNN | GFNet, accuracy build | **GFNet, recommended** |
|---|---|---|---|
| Test accuracy, slice | 0.7421 | **0.7450** | 0.7381 |
| Test accuracy, patient | 0.7511 | 0.7578 | **0.7689** |
| Macro F1 | 0.7357 | 0.7367 | 0.7334 |
| AD precision | 0.8411 | 0.8506 | 0.8157 |
| AD recall | 0.5913 | 0.5729 | **0.6092** |
| NC precision | 0.6892 | 0.6854 | 0.6925 |
| NC recall | 0.8903 | **0.9141** | 0.8648 |
| AD cases missed, of 4,460 | 1,823 | 1,905 | **1,743** |
| Best validation accuracy | 0.9028 | 0.8689 | 0.8273 |
| Best epoch of 80 | 79 | 65 | 71 |

**The recommended GFNet wins at patient level by 1.8 points and misses 80 fewer
Alzheimer's cases than the baseline.** It gives up 0.4 points of slice accuracy
to do it.

A clinician classifies a patient, not a slice, so the patient-level row is the
one that matters clinically. The slice row is reported because the
specification's 0.80 target refers to it.

The three differ only in how GFNet was trained:

| Build | Augmentation | Stochastic depth | Mixup | Label smoothing |
|---|---|---|---|---|
| Accuracy build | normal | 0.1 | none | none |
| Recommended | strong | 0.3 | 0.8 | 0.1 |

Both use patch size 8, depth 12, AdamW, and a five-epoch warmup. That
combination is the recipe the GFNet paper trains with [1], which it takes
from DeiT [4]. My first runs used plain Adam with no warmup and no
stochastic depth, and closing that gap gave the largest single gain in the
project. The
differences and their effects are set out in the
experiment ledger section below.

### Resource profile

| Measure | BaselineCNN | GFNet, recommended | Ratio |
|---------|-------------|--------------------|-------|
| Trainable parameters | 421,474 | 9,117,954 | 21.6× |
| Peak GPU memory | 2.32 GB | 8.76 GB | 3.8× |
| Inference rate | 12,415 img/s | 977 img/s | 0.08× |
| Inference latency | 0.081 ms/img | 1.024 ms/img | 12.6× |
| Seconds per epoch | 13.6 | 62.4 | 4.6× |
| Total training time | 18.1 min | 83.5 min | 4.6× |

**GFNet costs 12.6 times the inference latency for 1.8 points of patient
accuracy.** On a screening queue of a few hundred scans a day that is irrelevant.
977 images a second clears a day's work in well under a second.
It would matter on a device at the bedside.

An earlier, smaller GFNet at patch size 16 used **less** memory than the
baseline, 1.58 GB against 2.32, despite having 11 times more parameters.
Parameters and activations are separate costs. The baseline's first convolution holds a 224 × 224 × 32 feature map, about
1.6 million values per image. GFNet at patch 16 reduces the image to 196 tokens
of 256 values straight away.
Quartering the patch size to 8 raises the token count to 784 and reverses that
advantage.

![BaselineCNN example predictions](figures/predictions_baseline.png)

![GFNet example predictions](figures/predictions_gfnet.png)

---

## The four findings that matter

### 1. GFNet is three times better calibrated, and wins at patient level

| Measure | BaselineCNN | GFNet, recommended |
|---|---|---|
| Expected Calibration Error | 0.1812 | **0.0614** |
| Worst single bin gap | 0.2603 | **0.1110** |
| Mean confidence when wrong | 0.8803 | **0.7378** |
| Confidence separation | 0.0579 | **0.0835** |
| Patients wrong on all 20 slices | 40 | **31** |

GFNet predicts better at patient level **and** knows better what it does not
know. Earlier in the project those two properties were split between the two
models, and getting both into one is what the final training recipe bought.

Even so, neither confidence score is good enough to automate on. A separation
of 0.0835 still leaves the right and wrong populations heavily overlapped, so
acceptance criterion 4 is reported as not achievable.

*Full detail in the "Investigation of the open research dilemma" section below.*

### 2. Temperature scaling is a correction, not an improvement

The recommended model's fitted temperature is **0.75**, below 1, meaning
mildly underconfident. Applying temperature scaling to it makes its calibration
**worse**, from 0.0614 to 0.1056.

Every earlier model in this project needed a temperature above 1. One needed
4.59. The method assumes overconfidence, and on a model that has been trained
not to be overconfident it does damage.

That is a condition my earlier recommendation did not carry, and it comes from
a measurement rather than from reading about the method.

### 3. The filename ID identifies a scan, not a person

Near-duplicate patients sit at a median ID gap of 2. Ordinary pairs sit at
47,026. So 16.1% of validation patients share a person with the training set,
even though the patient-level split and its overlap assertion are both correct.

Measured cost: 0.6 points of validation accuracy. **These splits are leak-free
by ID and are not leak-free by subject.**

*Full detail in the "The filename ID is a scan ID" section below.*

### 4. The constraint is the cohort, not the model

Twelve GFNet configurations across seven axes: training length, augmentation
strength, weight decay, token resolution, input representation, optimiser and
regularisation recipe.

**Validation moved 8.5 points across them. Test moved 6.1, and not in step with
it.** The run with the highest validation accuracy is not the best on test.

The strongest single piece of evidence: mixup reduced training accuracy from
0.9954 to 0.7759, which is memorisation eliminated, and validation loss stopped
rising. The intervention worked exactly as designed. **Test accuracy did not
improve.**

The best slice-level result is 0.7450, which is 5.5 points short of the 0.80
minimum the specification sets. The campaign is complete rather than abandoned,
and the ledger records all 20 experiments.

*Full detail in the "Experiments towards the 0.80 minimum" section below.*

---

## Engineering recommendation

**Use GFNet with the recommended training recipe, and do not let it decide
alone.** It beats the baseline at patient level, misses 80 fewer Alzheimer's
cases, and is three times better calibrated. It still does not reach the 0.80
accuracy the specification asks for, and its confidence is still not separable
enough to automate on.

### Which model

| | BaselineCNN | GFNet, accuracy build | **GFNet, recommended** |
|---|---|---|---|
| Patient accuracy | 0.7511 | 0.7578 | **0.7689** |
| Slice accuracy | 0.7421 | **0.7450** | 0.7381 |
| Calibration error | 0.1812 | 0.2231 | **0.0614** |
| AD cases missed | 1,823 | 1,905 | **1,743** |
| Patients wrong on all 20 slices | 40 | 40 | **31** |
| Inference | **12,415 img/s** | 977 img/s | 977 img/s |

The accuracy build holds the best slice accuracy, and that is the number the
0.80 target refers to. It is also the worst calibrated model in the project,
with a fitted temperature of 4.59 and all ten of its most confident mistakes at
exactly 1.0000. For a screening tool that routes uncertain cases to a
specialist, a model that is certain and wrong is the worst thing to have.

**The baseline remains the right choice when inference cost dominates.** It is
12.6 times faster and uses a twentieth of the parameters for 1.8 points less
patient accuracy.

### How to run it

1. **Decide at patient level, not slice level.** Average the 20 slice
   probabilities. It is the clinically meaningful unit and it is worth 3.1
   points for this model, 0.7381 to 0.7689.
2. **Move the AD decision threshold below 0.50.** On the baseline, a threshold
   of 0.10 saves 673 missed AD cases at no cost in overall accuracy. A missed case is the expensive error in screening. The error autopsy
   makes this urgent. Every one of the worst total failures, for every model
   tested, is a patient who has AD.
3. **Do not apply temperature scaling to this model.** Its fitted temperature
   is 0.75, so it is mildly underconfident, and scaling makes its calibration
   error worse, from 0.0614 to 0.1056. Temperature scaling corrects
   overconfidence. Fit the temperature on validation first and apply it only if
   it comes out above 1.
4. **Treat every output as a draft for a specialist.** A confidence separation
   of 0.0835 is the best measured in this project and it is still far too small
   to divide the right and wrong populations cleanly.
5. **Refer any patient the two architectures disagree about.** 23 of the 450 test patients are wrong on all 20 slices for
   both, so an ensemble cannot rescue them. But 17 fail only the baseline and 8
   only GFNet. Disagreement
   between a convolutional model and a frequency-domain model is cheap, and it
   is independent of the confidence score.

**Do not average the two models' probabilities.** The even-split ensemble
reaches 0.7518 at slice level, the best figure in the project, and it is the
wrong trade. Patient accuracy falls to 0.7644 and the number of patients wrong
on all 20 slices rises from 31 to 48. Buying 1.4 slice points by making the
worst failures 55% more common is the opposite of what a screening tool needs.

### Does GFNet earn its cost?

| GFNet is worse at | GFNet is better at |
|---|---|
| Slice accuracy, by 0.4 points | Patient accuracy, by 1.8 points |
| Parameter count, 21.6× more | Calibration error, 0.0614 against 0.1812 |
| Inference latency, 12.6× slower | Confidence separation, 0.0835 against 0.0579 |
| Peak memory, 3.8× more | AD cases missed, 1,743 against 1,823 |

**Yes, for this job.** A screening tool exists to decide which patients a
specialist should see. It is judged on patients rather than slices, on missed
disease rather than overall accuracy, and on whether its confidence can be
trusted. GFNet wins all three, and the cost it pays is inference latency that
is irrelevant at screening volumes.

The honest caveat is that this conclusion rests on 1.8 points of patient
accuracy, measured once, on one seed. The reproducibility test in the ledger puts the noise floor at 0.01 points,
so 1.8 points is real. A three-seed comparison would make it firmer, and that is
the first thing I would add.

### Where the real gain is

Not in either architecture. Twelve GFNet configurations moved test accuracy by
6.1 points in total, and the best of them is 5.5 points short of the target.
Mixup eliminated memorisation completely and test accuracy did not move.

The constraint is the difference between the training cohort and the test
cohort. This report could not explain it, after ruling out folder statistics
and subject leakage. Closing it needs better input, 3D volumes or
widely spaced slices rather than adjacent ones, not a better classifier on a
single 2D slice.

---

## Limitations

**The project did not meet the 0.80 accuracy target.** The best slice-level
result is 0.7450 and the best patient-level result is 0.7689. Twelve GFNet
configurations were tried across seven axes, and the ledger records all of
them. The short version:

| What I tried | Effect on test accuracy |
|---|---|
| 80 epochs instead of 30 | +2.2 points |
| Patch size 8 instead of 16 | +0.6 points |
| AdamW, warmup and stochastic depth | +1.5 points |
| The patient-level mean | +3.1 points |
| 150 epochs instead of 80 | −0.2 points |
| 2.5D, three adjacent slices | −2.4 points |
| Test-time augmentation | −0.06 points |
| Mixup and label smoothing | −0.7 points slice, **+1.1 patient** |

My view on the cause is that the training and test cohorts differ in a way
nothing I measured could reach. One 2D slice does not hold enough information
to bridge that difference. The ADNI literature mostly uses 3D volumes or several
slices. Three adjacent slices did not help here, and the reason is in my own data.
Adjacent slices correlate at 0.9612, so they are near-copies rather than new
information.

**`cudnn.benchmark` is on, and I measured what that costs.** An identical
configuration was accidentally run twice with the same seed. The two runs
differed by **one slice in 9,000**, 0.7450 against 0.7449. So results are not
bit-exact, and the noise floor is 0.01 accuracy points. Differences of 0.3
points and above in this report are real rather than run-to-run variation.

**One seed.** Every result uses seed 42. The measurement above shows the noise
floor is small, but it measures one configuration run twice, not the spread
across seeds. Three seeds per model with a mean and standard deviation would
make the comparison firmer, and it is the first thing I would add.

**The recommendation rests on 1.8 points of patient accuracy** between
BaselineCNN and the recommended GFNet, measured once. It is well above the
noise floor and it is still a single measurement.

**The horizontal flip** assumes a left-right symmetry that Alzheimer's atrophy
does not always keep. It deserves an ablation. Test-time augmentation gives indirect evidence that the model has
learned the symmetry. Averaging a slice with its mirror changed the result by
0.06 points, so the two predictions are nearly identical.

**The splits are leak-free by ID and not by subject.** 16.1% of validation
patients share a person with the training set, because the filename ID
identifies a scan rather than a person. Measured cost: 0.6 points of validation
accuracy and 0.2 points of test accuracy. The linked section sets out the
evidence and the fix I would apply with more time.

**The validation-to-test gap is unexplained.** It is 11 to 16 points depending
on the model. Folder statistics, subject leakage and memorisation were each
tested and none accounts for it. Mixup is the strongest of those tests. It cut training accuracy from
0.9954 to 0.7759 and stopped validation loss rising, so the memorisation was
genuinely removed. Test accuracy did not improve. I am
recording the gap as open rather than attributing it to a cause I cannot
demonstrate.

---

## Artificial Intelligence Usage Disclosure

### Tool used
Anthropic Claude, mainly through an agentic coding session over approximately one week.

### What I used it for
Claude helped me create the initial versions of several project files, including modules.py, dataset.py, train.py, predict.py, evaluate_calibration.py, visualise_data.py, and brightness_baseline.py. These were used as starting points rather than final solutions. I reviewed each file, made edits, corrected mistakes, and adapted the code to fit the requirements of the project.
It also helped generate an early draft of the README structure and some explanatory text, which I substantially revised and rewrote during the final documentation process.
During development, I used Claude as a debugging assistant when dealing with Python errors and configuring SLURM jobs on the university cluster. It was also useful for brainstorming potential analyses, including the brightness-ablation experiment, temperature scaling for calibration, patient-level prediction averaging, and strategies for addressing overfitting.

### How I checked the output
I did not accept Claude's suggestions without verification. The project includes several checks that allowed me to validate the code before running expensive training jobs.
In modules.py, I implemented and ran six self-tests. The most important test sets every GFNet filter weight to 1 + 0i and verifies that the FFT round-trip reconstruction reproduces the original input to within 4.8 x 10^-7. This confirmed that the frequency-domain operations were functioning correctly.
In dataset.py, I used an assertion to ensure that no patient appeared in more than one data split. I also manually checked the reported split sizes. The training and validation splits contained 915 and 161 patients respectively, giving a total of 1,076 patients, which matched the dataset audit.
I independently verified the evaluation metrics by calculating them directly from the confusion matrix. For example, in the baseline model, 4042 + 2637 = 6679 correct predictions out of 9000 samples, giving an accuracy of 0.7421, which matches the reported result.
I also confirmed with teaching staff that the identifiers contained in the filenames correspond to patient IDs, which was important for ensuring that splitting by patient was implemented correctly.
Several Claude-generated suggestions were incorrect and had to be fixed. One example was an invalid SLURM partition recommendation that appeared to come from outdated information. Other mistakes introduced during code edits included a duplicated keyword argument, a stray bracket causing a syntax error, and a missing .to(device) call. Each issue caused failed runs that I then diagnosed and corrected myself.

### What I did myself
I selected the project topic, chose GFNet as the model architecture, and decided to perform the dataset split at the patient level rather than the image level. I carried out the background reading on Alzheimer's disease anatomy that informed the error analysis and interpretation of the results.
All experimental decisions, result interpretation, discussion, conclusions, and recommendations are my own work. While Claude assisted with drafting code, troubleshooting, and generating ideas, I was responsible for verifying every implementation, testing every change, interpreting every result, and deciding what was ultimately included in the final report.
I take responsibility for all code, analyses, and claims presented in this submission.


---

## Feasibility review

*This is the one-page review I presented at the midpoint check-off, as Section 3
of the assignment specification requires.*

**Intended user.** A clinician who does early Alzheimer's screening. The
prototype does not replace the judgement of that clinician. It sorts scans into
two groups: cases the model is sure about, and cases it is not. The unsure cases
reach a specialist first.

**Purpose.** Put a 2D T1-weighted brain MRI slice in the AD class or the NC
class, and attach a confidence score that you can trust.

**Acceptance criteria.** I measured all of them on the held-out test split.

| # | Criterion | Target | Outcome |
|---|-----------|--------|---------|
| 1 | GFNet test accuracy, AD against NC | ≥ 0.80 | **Not met.** See Limitations |
| 2 | Baseline CNN trained on identical splits | reported | Met |
| 3 | Expected Calibration Error for both models | reported | Met |
| 4 | Reject-option rule that reaches ≥ 0.95 retained accuracy | ≤ 30% referral | **Not achievable.** See Calibration |
| 5 | Peak GPU memory and training time on one A100 | ≤ 16 GB, ≤ 2 h | Met. 2.32 GB, 18 min |

**Model choice.** GFNet (Rao et al., 2023) is a Hard-difficulty option. The
choice builds on the Fourier material in this course and on my own Demo 2 Part 1
work. There I measured a naive DFT at 34.2 s against an FFT at 0.13 ms for
N = 4096. The argument for GFNet against attention is the same argument one
level up.

**Baseline.** A plain four-block CNN written for this project, with 421,474
parameters. I did not tune it hard, so it stays an honest reference point and
not a weak target.

**Risks I found, and what happened.**

1. *Complexity of the GFNet filter block.* Closed by the FFT identity
   self-test, before I spent any GPU time.
2. *Cluster environment.* Confirmed to work. torch 2.14.0+cu130 on an
   A100-PCIE-40GB.
3. *Overfit on 915 training patients.* This risk happened. See the overfit
   section below for the diagnosis and the fix.

---

## Running the code

## Repository contents

```
PatternAnalysis-2026/
└── recognition/
    └── GFNet_s4967283/
        ├── README.md                 Main report page, links to the rest
        ├── progress_log.md           Dated record of each working day
        │
        ├── docs/
        │   ├── feasibility-review.md     Midpoint check-off review
        │   ├── running-the-code.md       This page
        │   ├── data-and-splits.md        Dataset, splits, preprocessing
        │   ├── dilemma-investigation.md  Calibration and the trade-off study
        │   ├── error-autopsy.md          Four named failure patients
        │   ├── scan-id-finding.md        The dataset ID problem
        │   └── experiment-ledger.md      All 17 experiments
        │
        ├── modules.py                GFNet and BaselineCNN, plus six self-tests
        ├── dataset.py                Patient-level splits, augmentation, loaders
        ├── train.py                  Training loop, validation, checkpoints
        ├── predict.py                Inference, confusion matrix, report figures
        │
        ├── audit_adni.py             Dataset audit, written before the loader
        ├── visualise_data.py         Raw data figures and a leakage measurement
        ├── brightness_baseline.py    One-feature classifier, audits for an artifact
        ├── evaluate_calibration.py   ECE, reliability, reject option, temperature
        ├── inspect_patient.py        Per-slice predictions for a named patient
        ├── try_ensemble.py           Combines the two models and measures the gain
        ├── audit_split_difference.py Compares the shipped train and test folders
        ├── find_duplicate_subjects.py Finds repeat scans of one person across splits
        ├── measure_leakage_impact.py Counts the leaked patients and scores without them
        │
        ├── run_smoke.sh              SLURM, two epochs, run before any full job
        ├── run_train.sh              SLURM, full run. Args: model seed epochs aug
        ├── run_figures.sh            SLURM, rebuilds every figure from checkpoints
        ├── run_recipe_figures.sh     SLURM, same but for a depth-12 patch-8 checkpoint
        ├── run_tta.sh                SLURM, measures what test-time augmentation buys
        │
        └── figures/
            ├── fig1_class_comparison.png        Example slices, AD against NC
            ├── fig2_one_patient_all_slices.png  All 20 slices of one patient
            ├── fig3_class_averages.png          Mean image per class
            ├── fig4_intensity_histograms.png    Pixel intensity by class
            ├── predictions_baseline.png         Example predictions
            ├── predictions_gfnet.png
            ├── failures_baseline.png            Most confident mistakes
            ├── failures_gfnet.png
            ├── filters_gfnet.png                Learned frequency filters
            ├── calibration_reliability.png      Reliability diagram
            ├── calibration_confidence.png       Confidence, right against wrong
            ├── reject_option_curves.png         Retained accuracy by referral rate
            ├── calibration_results.json         Every calibration number
            ├── test_predictions_*.npz           Saved test probabilities
            └── validation_predictions_*.npz     Saved validation probabilities
```

Model checkpoints and training logs are not committed. The spec forbids it and
they are regenerable from `run_train.sh`.

| File | Purpose |
|------|---------|
| `modules.py` | `BaselineCNN` and `GFNet` definitions, plus six self-tests |
| `dataset.py` | Data loader with patient-level splits and augmentation |
| `train.py` | Training loop, validation, checkpoints, metric plots |
| `predict.py` | Loads a checkpoint, runs inference, draws the report figures |
| `audit_adni.py` | Dataset inspection, written before the loader |
| `visualise_data.py` | Four figures of the raw data, plus a leakage measurement |
| `brightness_baseline.py` | A one-feature classifier, used to audit the result |
| `evaluate_calibration.py` | ECE, reliability, reject-option sweep, temperature scaling |
| `inspect_patient.py` | Per-slice predictions for one patient, used for the error autopsy |
| `try_ensemble.py` | Combines the two models and measures what that buys |
| `audit_split_difference.py` | Compares the shipped folders on eight measures, with effect sizes |
| `find_duplicate_subjects.py` | Detects repeat scans of one person, using test as a control |
| `measure_leakage_impact.py` | Counts leaked patients and recomputes accuracy without them |
| `run_smoke.sh` | SLURM smoke test, two epochs, run before any full job |
| `run_train.sh` | SLURM training run. Takes model, seed, epochs and augmentation |
| `run_figures.sh` | SLURM job that rebuilds every figure from a pair of checkpoints |

---

## Dependencies and reproducibility

I measured these on the Rangpur HPC cluster.

| Package | Version |
|---------|---------|
| Python | 3.11 |
| torch | 2.14.0+cu130 |
| torchvision | matching cu130 build |
| CUDA | 13.0 |
| numpy | used in `train.py`, `predict.py` and the analysis scripts only |
| matplotlib | for all figures |
| Pillow | image loading |

Hardware: one NVIDIA A100-PCIE-40GB.

**Note on `modules.py`.** The assignment says the model code must depend only on
PyTorch. `modules.py` imports nothing else. NumPy appears in `train.py`,
`predict.py` and the analysis scripts, where the assignment allows it.

### What is fixed, and what is not

`set_all_seeds` fixes the seed for Python `random`, for NumPy and for torch.
It fixes the CUDA seed as well. The train and validation split uses the same
seed, so the same patients go to the same split on every run.

An honest limit: `cudnn.benchmark` stays on. It picks the fastest convolution
algorithm for the hardware, and that choice is not guaranteed to be the same
between runs. So the results repeat to within a small margin. They are not
bit-exact. `deterministic=True` would fix that, at a clear cost in speed.

---

## How to run

Every command below shows the output you should expect. The output is real and
abridged, with `...` where lines are cut. If what you see differs, something is
wrong and the note under each block says what to check.

Run the whole pipeline in this order. Each step depends on the one before it.

### 1. Audit the dataset

```bash
python audit_adni.py
```

Expected output:

```
Auditing ADNI dataset at: /home/groups/comp3710/ADNI

1. FOLDER STRUCTURE (depth limited to 3)
ADNI/
    AD_NC/
        test/
            NC/   <- 4540 image files
            AD/   <- 4460 image files
        train/
            NC/   <- 11120 image files
            AD/   <- 10400 image files

TOTAL IMAGES FOUND: 30520

3. SAMPLE IMAGE PROPERTIES
  size (width, height): (256, 240)
  mode: L   (L means greyscale, RGB means colour)
  pixel min: 0
  pixel max: 255

4. PATIENT ID PATTERN (the important one)
Folder: .../AD_NC/train/AD
  files matching the <patient>_<slice> pattern: 10400 of 10400
  unique patient IDs: 520
  first few patient IDs: ['218391', '218393', '228875', '228879', '236920']
  slices per patient: min 20, max 20, average 20.0
...

5. DO ANY PATIENTS APPEAR IN MORE THAN ONE FOLDER?
No patient appears in two different folders. The shipped split is clean.
```

**Check:** every file must match the `<patient>_<slice>` pattern, and every
patient must have exactly 20 slices. The loader depends on both.

**A note on section 5.** That conclusion is true by ID and false by subject.
The section "The filename ID is a scan ID, not a person ID" below shows that
consecutive IDs are the same person. The first few training IDs in the output above hint at it already. 218391
and 218393 are two apart. 228875 and 228879 are four apart. I did not notice that until a validation accuracy
looked wrong three days later.

Run this first. Every later step assumes the layout it reports.

### 2. Produce the data figures

```bash
python visualise_data.py --data-root /home/groups/comp3710/ADNI/AD_NC \
  --output-dir figures
```

Expected output:

```
Reading train split from /home/groups/comp3710/ADNI/AD_NC
Found 21520 images from 1076 patients
  AD patients: 520
  NC patients: 556

wrote figures/fig1_class_comparison.png
wrote figures/fig2_one_patient_all_slices.png
wrote figures/fig3_class_averages.png
wrote figures/fig4_intensity_histograms.png

SLICE SIMILARITY (plain correlation between flattened images)
  same patient, adjacent slices:  mean 0.9612  (min 0.9041, max 0.9891, n=30)
  different patients, same class: mean 0.6769  (min 0.2880, max 0.8626, n=30)
  gap: 0.2843
```

**Check:** the similarity gap is the measurement that justifies patient-level
splitting. Two slices of one patient correlate at 0.96, two slices of different
patients at 0.68. A random image-level split would scatter one person across
train and validation and hand the model that 0.28 gap as a shortcut.

### 3. Check the splits and the loader

```bash
python dataset.py
```

Expected output:

```
Building splits...

SPLIT SUMMARY
train         18300 images    915 patients   NC: 9460   AD: 8840
validation     3220 images    161 patients   NC: 1660   AD: 1560
test           9000 images    450 patients   NC: 4540   AD: 4460

No patient appears in two splits (the assertion in build_splits passed).

Loading one batch to check shapes...
  image batch shape: (8, 1, 224, 224)
  label batch shape: (8,)
  pixel range after normalising: -0.518 to 3.779
```

**Check:** the patient counts must read 915, 161 and 450, and the assertion
line must appear. If `build_splits` raises an `AssertionError` instead, a
patient is in two splits and nothing after this is worth running.

### 4. Check the models before spending GPU time

```bash
python modules.py
```

Expected output:

```
GFNet and baseline self-tests

test: filter with unit weights returns its input unchanged
   largest absolute difference: 4.768e-07
   passed

test: filter shapes
   grid  7x7  -> rfft width 4, weight (7, 4, 4, 2)  OK
   grid 14x14 -> rfft width 8, weight (14, 8, 4, 2)  OK
   grid 16x16 -> rfft width 9, weight (16, 9, 4, 2)  OK
   passed

...

test: model output shapes and parameter counts
   BaselineCNN  output (4, 2)   parameters    421474
   GFNet        output (4, 2)   parameters   4788482
   GFNet is 11.4x the size of the baseline
   passed

all checks passed
```

**Check:** the first test is the important one. It sets every filter weight to
1 + 0i, so the block does FFT, multiply by one, inverse FFT, and must return
its input unchanged. A difference above 1e-4 means the transform plumbing is
wrong and no amount of training will fix it.

This runs on CPU in a few seconds. On the login node it prints a CUDA driver
warning, which is expected there and harmless.

### 5. Train

```bash
# Arguments, in order:
#   MODEL SEED EPOCHS AUG PATCH SLICES DEPTH DROPPATH WARMUP OPTIM MIXUP SMOOTH

# The baseline
sbatch run_train.sh baseline 42 80 strong

# The two GFNet builds this report compares
sbatch run_train.sh gfnet 42 80 normal 8 1 12 0.1 5 adamw
sbatch run_train.sh gfnet 42 80 strong 8 1 12 0.3 5 adamw 0.8 0.1

# Other configurations from the ledger
sbatch run_train.sh gfnet 42 80 normal 16 3
sbatch run_train.sh gfnet 42 150 normal 8 1 12 0.1 5 adamw
```

Expected output from `sbatch`:

```
Submitted batch job 692552
```

Then watch it:

```bash
squeue -u $USER
```

While the job runs, the log fills with one line per epoch:

```
epoch 148/150  train loss 0.2617 acc 0.8879  |  val loss 0.5767 acc 0.7724  |  13.6s
epoch 149/150  train loss 0.2649 acc 0.8862  |  val loss 0.5776 acc 0.7720  |  14.0s
epoch 150/150  train loss 0.2730 acc 0.8843  |  val loss 0.5779 acc 0.7714  |  13.7s
------------------------------------------------------------------------
Training finished in 35.2 minutes
Best validation accuracy 0.7907 at epoch 60

Loading best checkpoint and evaluating on the test set

TEST RESULTS
  accuracy: 0.6837
  macro F1: 0.6777
  NC: precision 0.6488  recall 0.8128  f1 0.7216
  AD: precision 0.7434  recall 0.5522  f1 0.6337

RESOURCE PROFILE
  trainable parameters: 4788482
  peak GPU memory: 1.58 GB
  total training time: 35.2 minutes
  mean seconds per epoch: 14.0
  inference: 5589 images/second (0.179 ms per image)

Wrote:
  outputs_gfnet_seed42_normal/best_gfnet.pt
  outputs_gfnet_seed42_normal/history_gfnet.json
  outputs_gfnet_seed42_normal/test_predictions_gfnet.npz
  outputs_gfnet_seed42_normal/summary_gfnet.json
```

**Check:** seconds per epoch tells you the job is healthy.

| Configuration | Seconds per epoch |
|---|---|
| Either model at patch 16, depth 8 | about 14 |
| Patch 8, depth 8 | about 42 |
| Patch 16, three slice channels | about 34 |
| Patch 8, depth 12, the builds this report uses | about 62 |

A figure far above these means the data loader is starving the GPU.

**The arguments that change the model's shape are `--patch-size` and
`--depth`.** Both must be repeated to `predict.py` when loading a checkpoint,
or the weights will not fit and the load fails. That failure is deliberate. A
silent fallback would fill the missing blocks with random weights and produce a
plausible-looking figure set.

The arguments that change only training are `--augmentation`, `--drop-path`,
`--warmup-epochs`, `--optimiser`, `--mixup` and `--label-smoothing`. None of
them need repeating at inference.

The output folder is named from the arguments. Patch 16 with one slice keeps
the plain name, and anything else gets a `_p<size>` or `_s<slices>` suffix, so
runs never overwrite each other.

### 6. Run a trained model and draw the figures

```bash
python predict.py --model gfnet \
  --checkpoint outputs_gfnet_seed42_strong_p8_recipe_mix/best_gfnet.pt \
  --patch-size 8 --depth 12 --output-dir figures
```

Expected output:

```
Using GPU: NVIDIA A100-PCIE-40GB

Loading data from /home/groups/comp3710/ADNI/AD_NC
Running on the test split: 141 batches

Loading gfnet from outputs_gfnet_seed42_strong/best_gfnet.pt
  4788482 trainable parameters

RESULTS
  slices: 9000
  accuracy: 0.7122

  confusion matrix
                 predicted NC   predicted AD
  actual NC            3814            726
  actual AD            1864           2596

  NC: precision 0.6717  recall 0.8401  (missed 726)
  AD: precision 0.7815  recall 0.5821  (missed 1864)

  patient level: 0.7444  (335 of 450 patients)

  FAILURE CASES FOR THE AUTOPSY
  Most confident false negatives (has AD, called normal):
    patient 985197  confidence 0.9990
    ...
```

**Check:** the confusion matrix rows must sum to 4,540 and 4,460. If they do
not, the loader returned the wrong split.

Add `--patch-size 8` or `--slice-context 3` when the checkpoint was trained
that way. A mismatch fails loudly when the weights do not fit the model, which
is the behaviour you want.

### 7. Validation predictions for temperature scaling

```bash
python predict.py --model gfnet \
  --checkpoint outputs_gfnet_seed42_strong/best_gfnet.pt \
  --split validation --no-figures --output-dir figures
```

Expected output ends with:

```
  patient level: 0.8199  (132 of 161 patients)

  wrote figures/validation_predictions_gfnet.npz
```

**Check:** 161 patients, not 450. The temperature must be fitted on validation.
Fitting it on test and then reporting test calibration would be tuning on the
test set.

### 8. The dilemma investigation

```bash
python evaluate_calibration.py \
  --predictions figures/test_predictions_baseline.npz figures/test_predictions_gfnet.npz \
  --val-predictions figures/validation_predictions_baseline.npz figures/validation_predictions_gfnet.npz \
  --names baseline gfnet --output-dir figures
```

Expected output, per model:

```
MODEL: gfnet
Slice level:   9000 slices, accuracy 0.7122
Patient level: 450 patients, accuracy 0.7444  (20-20 slices each)

Calibration:
  expected calibration error: 0.0934
  worst single bin gap:       0.1594
  mean confidence when right: 0.8281
  mean confidence when wrong: 0.7499
  separation:                 0.0782

Temperature scaling (fitted on validation):
  temperature: 1.11 (>1 means it was overconfident)
  ECE 0.0934 -> 0.0787
  accuracy 0.7122 -> 0.7122 (should be unchanged)

Recommended operating point:
  NOT ACHIEVABLE. No threshold reaches 0.95 accuracy on automated cases
  while referring no more than 30% of cases.
```

**Check:** accuracy before and after temperature scaling must be identical.
Temperature scaling divides the logits by one number, which cannot reorder
them. If accuracy moves, the implementation is wrong.

### Rebuild every figure at once

```bash
sbatch run_figures.sh
```

Runs steps 6, 7 and 8 for both models inside one GPU allocation. Use it after
a new pair of checkpoints, so that the figures and the tables beside them
always agree.

---

## Analysis scripts

These read the saved `.npz` files or the images directly. None needs a GPU, so
the login node is fine.

### The brightness control

```bash
python brightness_baseline.py --data-root /home/groups/comp3710/ADNI/AD_NC
```

Expected output ends with:

```
APPLIED TO THE TEST SPLIT
  slice-level accuracy:   0.5850
```

**Check:** this is a deliberately stupid classifier that uses one number per
slice. A result near 0.5 would mean brightness carries no signal. A result near
0.72 would mean the CNN is close to a brightness meter. At 0.585 it sits
between, so the network learns real structure and part of the signal is
brightness.

### Per-slice predictions for one patient

```bash
python inspect_patient.py --patient 985197 --worst 10
```

Expected output:

```
PATIENT 985197
------------------------------------------------------------------------
  baseline  true class AD   correct on 0 of 20 slices
             mean P(AD) 0.0000   min 0.0000   max 0.0000
  gfnet     true class AD   correct on 0 of 20 slices
             mean P(AD) 0.0086   min 0.0010   max 0.0290
```

**Check:** this is the single most important measurement in the error autopsy.
The baseline returns a probability below 0.00005 on all 20 slices of a patient
who has Alzheimer's.

### The ensemble

```bash
python try_ensemble.py
```

Expected output ends with:

```
THE EVEN SPLIT, WHICH NEEDS NO FITTING
  slice accuracy:   0.7494
  patient accuracy: 0.7600  (450 patients)
  patients wrong on all 20 slices: 42
```

**Check:** the script sweeps the mixing weight and then warns you not to report
the best one. That weight was chosen from the test split. The
even split needs no fitting, so that is the number the report uses.

### Compare the two shipped folders

```bash
python audit_split_difference.py
```

Expected output ends with:

```
IMAGE MEASURES, TRAIN AGAINST TEST
  measure                train mean   test mean    Cohen d   size
  --------------------------------------------------------------------
  tissue_mean            113.621      113.853      -0.014    negligible
  head_area_fraction     0.258        0.263        -0.215    small
  ...
```

**Check:** every measure should read negligible or small. Anything marked
LARGE would be a candidate explanation for the validation-to-test gap.

### Detect repeat scans of one person

```bash
python find_duplicate_subjects.py
python measure_leakage_impact.py
```

Expected output from the second:

```
ID DIFFERENCE FOR HIGH-CORRELATION PAIRS
  validation
    correlation >= 0.995 (n=26): median ID gap 2, 26 of them within 20
    correlation <  0.995 (n=135): median ID gap 47026, 31 of them within 20
```

**Check:** the gap between those two median values is the evidence that the
filename ID identifies a scan and not a person. See the section on that finding
below.

---

## Where to run what

| Needs a GPU, so use SLURM | Login node is fine |
|---|---|
| `train.py`, through `run_train.sh` | `modules.py`, `dataset.py`, `audit_adni.py` |
| `predict.py`, through `run_figures.sh` | `brightness_baseline.py`, `inspect_patient.py` |
| | `try_ensemble.py`, `evaluate_calibration.py` |
| | `audit_split_difference.py` and the two leakage scripts |

The rule I keep: a GPU, any `num_workers` above zero, or anything over a few
minutes goes to SLURM. I broke it once and ran `predict.py` on the login node
with four worker processes on a shared machine.

---

## The data, the splits and the preprocessing

```
/home/groups/comp3710/ADNI/AD_NC/
    train/AD   10,400 images from   520 patients
    train/NC   11,120 images from   556 patients
    test/AD     4,460 images from   223 patients
    test/NC     4,540 images from   227 patients
```

File names use the form `<patientID>_<slice>.jpeg`, for example
`1003730_100.jpeg`. Every patient gives exactly 20 slices. Images are 256 × 240,
greyscale, with pixel values from 0 to 255.

Teaching staff confirmed that the number before the underscore is a **patient**
ID, and not a scan ID or a session ID. That matters, because ADNI follows people
over many years and can scan the same person more than once.

![Example slices from different patients](figures/fig1_class_comparison.png)

### The splits, and why they are built this way

Each patient gives 20 slices cut from one 3D scan, so those 20 images look very
similar to each other.

![All 20 slices from one patient](figures/fig2_one_patient_all_slices.png)

A random split at image level would spread the slices of one person across train
and validation. The model could then score well because it recognises the
patient and not the disease. Validation accuracy would look excellent for the
wrong reason, and the test score would then collapse with no clear cause.

So every split happens at **ID** level. I choose which IDs go where first, and
all 20 slices of an ID follow that choice. `check_no_patient_overlap` raises an
`AssertionError` if an ID appears in two splits. It crashes and does not warn,
because if that check fires, every number after it is worthless.

**That is correct, and it is not enough.** The section "The filename ID is a scan ID, not a person
ID" below shows that one person holds several consecutive IDs. So this check
passes while the same person sits in two splits. I measured
how much that costs and report the corrected numbers there.

| Split | Images | Patients | NC | AD |
|-------|--------|----------|-----|-----|
| train | 18,300 | 915 | 9,460 | 8,840 |
| validation | 3,220 | 161 | 1,660 | 1,560 |
| test | 9,000 | 450 | 4,540 | 4,460 |

The test split comes exactly as shipped, and I never touched it during
development. The audit confirmed that it shares no patients with the train
folder.

Validation holds 15.0% of the train patients, stratified by class: 78 of 520 AD
patients and 83 of 556 NC patients. Both are within a tenth of a percent of 15%.

Class balance is NC 51.7% and AD 48.3%. That is mild enough that weighted loss
or resampling is not needed. I measured it and did not assume it.

### Preprocessing

**Resize 256 × 240 to 224 × 224.** GFNet needs a fixed square input, because it
works on a grid of patches. I resize and do not crop. A crop could remove the
outer cortex, and that is the tissue that atrophies in Alzheimer's. The stretch
is the same for every image, so the model sees a consistent world.

**Normalisation.** I measured these values over 2,000 train images. I used the
train split only, so nothing leaks in from validation or test.

```
PIXEL_MEAN = 0.1159
PIXEL_STD  = 0.2236
```

The mean is low because these slices are mostly black background. With
placeholder values of 0.5 the normalised range was −1.000 to 0.953, which
squeezed the brain tissue into a narrow band. With the measured values the range
is −0.518 to 3.849. Background sits near −0.52 and tissue spreads across the
positive range.

**Augmentation.** There are two settings. Select one with `--augmentation`.

| Transform | normal | strong |
|---|---|---|
| Horizontal flip | p = 0.5 | p = 0.5 |
| Rotation | ±5° | ±10° |
| Translation | ±5% | ±10% |
| Scale | none | 0.9 to 1.1 |
| Brightness and contrast jitter | none | ±25% |

The brightness jitter is not arbitrary. The brightness experiment below shows
that average intensity carries real signal, which means the model can lean on
brightness as a shortcut. Random brightness during training removes that
shortcut.

An honest caution about the horizontal flip. Alzheimer's atrophy is not always
symmetric between the two hemispheres, so a mirror image does lose a little real
signal. I kept the flip because the extra variety helps more at this dataset
size. But that choice deserves an ablation and not an assumption.

---

## Investigation of the open research dilemma

### Does the model learn anatomy, or brightness?

The intensity histograms show that AD tissue is darker than NC tissue on
average. That raised a question I had to answer before I trusted any result.
Does the model read anatomy, or does it read brightness?

![Pixel intensity by class](figures/fig4_intensity_histograms.png)

![Mean image per class, AD against NC](figures/fig3_class_averages.png)

*The average of every AD slice beside the average of every NC slice. The AD
mean shows visibly larger ventricles, which is the structural change the models
are meant to find.*

So I built the simplest classifier I could. `brightness_baseline.py` uses one
number per slice, which is the mean intensity of the tissue pixels. It fits the
threshold on the train split only.

| Classifier | Test accuracy | What it uses |
|---|---|---|
| Chance | 0.500 | nothing |
| Mean tissue brightness | 0.585 | one number per slice |
| GFNet | 0.7122 | learned frequency filters |
| BaselineCNN | 0.7421 | learned convolution features |

The CNN adds **15.7 points** over the brightness rule, so the network learns
real structure. But 0.585 is above chance, so about 8.5 points of the signal is
average brightness. An honest account of this result needs both numbers.

This measurement is also the reason for the brightness jitter in the strong
augmentation setting.

### Does the confidence mean anything?

This is the core of the dilemma. I measured it with `evaluate_calibration.py`.

Three models are compared, because the training recipe moved calibration far
more than it moved accuracy, and the direction reversed twice.

| Measure | BaselineCNN | GFNet, accuracy build | **GFNet, recommended** |
|---|---|---|---|
| Expected Calibration Error | 0.1812 | 0.2231 | **0.0614** |
| Worst single bin gap | 0.2603 | 0.3953 | **0.1110** |
| Mean confidence when correct | 0.9382 | 0.9760 | 0.8213 |
| Mean confidence when wrong | 0.8803 | 0.9427 | **0.7378** |
| Separation | 0.0579 | 0.0332 | **0.0835** |

![Reliability diagram](figures/calibration_reliability.png)

**The recommended GFNet is three times better calibrated than the baseline.**
Its calibration error is 0.0614 against 0.1812, and its worst single bin is out
by 0.111 rather than 0.260. When it is wrong it claims 74% certainty rather
than 88%.

**The accuracy build is the worst calibrated model in the project**, and it is
the most accurate at slice level. That is the trade-off this report is about.
Its ten most confident mistakes all sit at exactly 1.0000. Five patients it
says certainly do not have Alzheimer's who do, and five it says certainly do
who do not.

What separates the two GFNet builds is mixup, cutmix and label smoothing. The
accuracy build reaches 0.9954 training accuracy and becomes certain about
everything, including its errors. The recommended build reaches 0.7759 and
keeps some doubt.

**Even so, neither confidence score is good enough to automate on.** A
separation of 0.0835 is the best measured in this project. It is still far too
small for a threshold to divide the right and wrong populations cleanly.

![Confidence when right against when wrong](figures/calibration_confidence.png)

### Temperature scaling, and when not to use it

Temperature scaling divides the logits by one learned number before the
softmax. It changes no prediction, so accuracy stays the same, but it can make
the confidence honest.

I fit the temperature on the **validation** split. A fit on test, followed by a
report of test calibration, would be a tune on the test set. That is the same
mistake the patient-level split exists to avoid.

| Model | Temperature | ECE before | ECE after | Change |
|---|---|---|---|---|
| BaselineCNN | 1.20 | 0.1812 | 0.1651 | −8.9% |
| GFNet, accuracy build | **4.59** | 0.2231 | 0.1127 | **−49%** |
| GFNet, recommended | **0.75** | 0.0614 | 0.1056 | **+72%, worse** |

**A temperature below 1 means the model is underconfident, and scaling it makes
things worse.** The recommended model's fitted temperature is 0.75, so label
smoothing at 0.1 slightly overcorrected. Applying temperature scaling to it
raises its calibration error from 0.0614 to 0.1056.

That is a condition I did not have earlier in this project, and it came from a
measurement rather than from reading about the method. **Temperature scaling
corrects overconfidence. It is not a general improvement.** Fit the temperature
on validation, and apply it only if it comes out above 1.

The accuracy build is the case the method was designed for. One scalar cuts its
calibration error in half, from 0.2231 to 0.1127. Even then it ends up worse
calibrated than the recommended model is before any scaling at all.

### Accuracy and calibration are not the same axis

Three models, two axes, and they do not agree:

| | Slice accuracy | Calibration error |
|---|---|---|
| GFNet, accuracy build | **0.7450** | 0.2231 |
| BaselineCNN | 0.7421 | 0.1812 |
| GFNet, recommended | 0.7381 | **0.0614** |

The ordering is exactly reversed. The most accurate model is the least trustworthy, and the most trustworthy is
the least accurate. 0.7 points separate the extremes on accuracy. A factor of
3.6 separates them on calibration.

A screening tool routes uncertain cases to a specialist. For that job, a model
that is certain and wrong is worse than one that is slightly less accurate and
knows it. That is the reasoning behind the recommendation.

### Patient-level evaluation

A clinician classifies a patient and not a slice. Each patient gives 20 slices,
so the mean of those 20 probabilities is the clinically useful read-out.

| Model | Slice | Patient | Gain |
|---|---|---|---|
| BaselineCNN | 0.7421 | 0.7511 | +0.9 pts |
| GFNet, accuracy build | 0.7450 | 0.7578 | +1.3 pts |
| **GFNet, recommended** | 0.7381 | **0.7689** | **+3.1 pts** |

**The ordering changes between the two columns.** The accuracy build leads at
slice level and the recommended build leads at patient level by 1.1 points.
Averaging the 20 slices is worth three times more to the recommended model.

That difference says something about the shape of each model's errors, and you
can see the shape by counting how each patient fails:

| Model | All 20 slices wrong |
|---|---|
| BaselineCNN | 40 |
| GFNet, accuracy build | 40 |
| **GFNet, recommended** | **31** |

The mean only helps a patient whose slices disagree with each other. In the
all-wrong group the model has a consistent, confident misread of that person's
anatomy, and no amount of averaging fixes it.

**The recommended model gets all 20 slices wrong for 31 patients against 40.**
A patient wrong on all 20 slices is one the system gets wrong every time, with
no internal disagreement to flag the case for review. That is a real safety
property, and it is why the model with the lower slice accuracy is the one this
report recommends.

### Where the two architectures fail together

Counting the overlap separates a hard scan from a weakness of one architecture.

| Group | Patients |
|---|---|
| Wrong on all 20 slices, baseline only | 17 |
| Wrong on all 20 slices, GFNet only | 8 |
| Wrong on all 20 slices, **both** | **23** |
| Either model fails completely | 48 of 450 |

**23 patients, 5.1% of the test set, defeat both architectures.** A
convolutional network and a frequency-domain network read an image in
completely different ways and still agree on every slice of all 23.

That sets a floor. No ensemble of these two can rescue them. The other 25 failures are specific to one architecture. That is why
disagreement between the two is a usable referral signal, even though averaging
them is not.

### Averaging the two models is a bad trade

| | Slice | Patient | All 20 slices wrong |
|---|---|---|---|
| BaselineCNN | 0.7421 | 0.7511 | 40 |
| GFNet, recommended | 0.7381 | **0.7689** | **31** |
| Even-split ensemble | **0.7518** | 0.7644 | 48 |

The ensemble produces the best slice accuracy in the project, 0.7518. It also
loses patient accuracy to GFNet alone and raises the number of patients wrong
on every slice from 31 to 48, a 55% increase.

Averaging a model that hedges against one that does not pushes the hedging
model's borderline-correct slices over the line. Buying 1.4 slice points by
making the worst failures half again as common is the opposite of what a
screening tool needs.

A note on method. Sweeping the mixing weight finds 0.7778 at patient level with
90% weight on GFNet. That weight was chosen by looking at the test split, so it
is not a result and this report does not claim it. The even split needs no
fitting, which is why it is the one reported.

### The reject option

The rule is simple. If confidence is at or above a threshold, accept the answer
automatically. Below the threshold, send the case to a specialist.

![Reject option curves](figures/reject_option_curves.png)

**Acceptance criterion 4 is not achievable.** No threshold reaches 0.95 accuracy
on the automatic cases and refers 30% of cases or fewer. That follows from the
confidence separation of 0.058. The two groups overlap too much.

The three outcomes do not cost the same. That is why overall accuracy is the
wrong thing to optimise.

| Outcome | Cost |
|---|---|
| False positive, referred | A few minutes of specialist time |
| False positive, auto-flagged | Unnecessary follow-up, patient distress |
| **False negative, auto-cleared** | **A patient goes home with no diagnosis. Nobody checks again.** |

### The decision threshold should move

Both models prefer to say "normal". A move of the AD decision threshold below
0.5 trades NC recall for AD recall.

I measured this on the baseline.

| AD threshold | AD recall | AD precision | NC recall | Accuracy | AD cases missed |
|---|---|---|---|---|---|
| **0.10** | **0.6899** | 0.7598 | 0.7857 | **0.7382** | **1,383** |
| 0.20 | 0.6370 | 0.7909 | 0.8346 | 0.7367 | 1,619 |
| 0.30 | 0.6049 | 0.8149 | 0.8650 | 0.7361 | 1,762 |
| 0.40 | 0.5744 | 0.8315 | 0.8857 | 0.7314 | 1,898 |
| 0.50 (default) | 0.5390 | 0.8492 | 0.9059 | 0.7241 | 2,056 |
| 0.60 | 0.5074 | 0.8628 | 0.9207 | 0.7159 | 2,197 |

At a threshold of 0.10 the model misses 1,383 AD slices and not 2,056. That is
**673 fewer missed cases**, and accuracy goes slightly up and not down.

The default of 0.5 is the wrong operating point for a screening tool.

Note what this does not do. Accuracy stays near 0.73 at every threshold. The
threshold moves errors between the classes. It does not add power to separate
them.

### Overfit: the diagnosis and the fix

The gap from validation to test got wider as training went on. That is the mark
of a model that fits its training group and does not learn something general.

| Baseline run | Validation | Test | Gap |
|---|---|---|---|
| 30 epochs, normal aug | 0.8894 | 0.7241 | 0.165 |
| 80 epochs, normal aug | 0.9326 | 0.7360 | **0.197** |
| 80 epochs, strong aug, weight decay 1e-3 | 0.9028 | 0.7421 | **0.161** |

First I checked whether the two shipped folders differ. The intensity histograms
for the train and test splits look the same, so a shift in the distribution is
not the cause.

The fix was stronger augmentation plus ten times the weight decay. Every measure
moved the right way.

| Measure | Normal aug | Strong aug | Change |
|---|---|---|---|
| Validation | 0.9326 | 0.9028 | −3.0 pts |
| Test (slice) | 0.7360 | 0.7421 | +0.6 pts |
| Test (patient) | 0.7422 | 0.7511 | +0.9 pts |
| Gap | 0.197 | 0.161 | −3.6 pts |
| AD recall | 0.5466 | 0.5913 | **+4.5 pts** |
| Macro F1 | 0.7257 | 0.7357 | +1.0 pts |
| ECE | 0.2164 | 0.1812 | −16% |
| Confidence separation | 0.0408 | 0.0579 | **+42%** |

Validation fell and test rose. That is what less overfit looks like. AD recall
gained the most, and that is the measure that matters in a clinic. It means 199
fewer missed Alzheimer's cases.

---

## Error autopsy

`predict.py` selects the mistakes that the model was **most confident** about,
because those are the dangerous ones. A mistake at 51% confidence goes to a
human under any sensible threshold. A mistake at 99% confidence does not.

`inspect_patient.py` then prints every slice of a named patient for both
models. That separates a patient the model is unsure about from a patient the
model is confidently and completely wrong about.

![BaselineCNN, the mistakes it was most confident about](figures/failures_baseline.png)

![GFNet, the mistakes it was most confident about](figures/failures_gfnet.png)

**Which models this section covers.** Findings 1 to 4 and the image
observations below were made on BaselineCNN and the original depth-8 GFNet. The
recommended model came later and fails on different patients, which is itself a
finding and is recorded at the end.

### Finding 1. The baseline reached P(AD) = 0.0000 on a patient who has AD

Patient 985197 has Alzheimer's disease. The baseline gave this output.

| Model | Correct slices | Mean P(AD) | Min | Max |
|---|---|---|---|---|
| BaselineCNN | **0 of 20** | **0.0000** | 0.0000 | 0.0000 |
| GFNet | 0 of 20 | 0.0086 | 0.0010 | 0.0290 |

Every one of the 20 slices returned a probability below 0.00005. The softmax
saturated. The model did not report low confidence in Alzheimer's. It reported
none at all, twenty times, on twenty different images of a sick brain.

GFNet is also wrong on all 20 slices, and it is wrong by a factor of about 30
less. It never claims certainty.

This single patient is the clearest answer to the research question in this
report. A confidence score that reads 1.0000 while the answer is wrong carries
no information at all.

### Finding 2. Every catastrophic failure is a missed AD case

I listed the ten worst total failures for each model. A total failure is a
patient the model gets wrong on all 20 slices.

**All twenty entries, across both models, are patients who have AD.** Not one
healthy patient appears in either list.

| Model | Worst patient | True class | Mean distance from the truth |
|---|---|---|---|
| BaselineCNN | 985197 | AD | 1.0000 |
| BaselineCNN | 1240482 | AD | 0.9996 |
| BaselineCNN | 1398870 | AD | 0.9994 |
| GFNet | 985197 | AD | 0.9914 |
| GFNet | 1240482 | AD | 0.9856 |
| GFNet | 1061844 | AD | 0.9855 |

The failures are not spread evenly between the classes. When either model fails
completely, it fails by calling a sick person healthy. That is the expensive
direction, and it is the direction the cost table above names as the worst
outcome.

This is the measurement that justifies the threshold recommendation. A model
whose total failures are all false negatives must not run at a decision
threshold of 0.5.

### Finding 3. Twenty patients defeated both of the early architectures

| Group | Patients |
|---|---|
| Wrong on all 20 slices, BaselineCNN only | 20 |
| Wrong on all 20 slices, GFNet only | 8 |
| Wrong on all 20 slices, **both models** | **20** |
| Either model fails completely | 48 of 450 |

The 20 shared patients are a property of those scans and not of one
architecture. A convolutional network and a frequency-domain network, which
read an image in completely different ways, agree on every slice of all 20.

That sets a floor. An ensemble of the two models would still fail on those 20
patients, which is 4.4% of the test set. The other 28 failures are
architecture-specific, so an ensemble could recover some of them.

### Finding 4. The two models fail in different shapes

Patient 1274602 is healthy. Both models call the patient sick, but not in the
same way.

| Model | Correct slices | Mean P(AD) | Range across slices |
|---|---|---|---|
| BaselineCNN | 0 of 20 | 0.9656 | 0.767 to 0.998 |
| GFNet | 11 of 20 | 0.5402 | 0.216 to 0.921 |

The baseline is wrong on every slice and sure of itself on every slice. GFNet is
right on more than half the slices and sits near the fence on the rest. The
patient-level mean still lands on the wrong side for GFNet, at 0.5402, but it
lands there by 0.04 and not by 0.47.

The same shape appears for patient 1265863. The baseline is correct on 1 slice
of 20 at a mean of 0.8747. GFNet is correct on 2 of 20 at a mean of 0.7387.

**The baseline fails with certainty. GFNet fails with doubt.** The calibration
numbers earlier in this report, an ECE of 0.1812 against 0.0934, are an average
over 9,000 slices. These four patients are what that average looks like in one
case at a time.

### What the images show

Looking at the false negative cases, patient 985197 is quite interesting because the scan does not immediately stand out as having the classic appearance people would associate with Alzheimer's disease. The ventricles are visible but do not look dramatically enlarged, and there is no obvious severe cortical thinning from this single slice. What stands out more is the image quality itself. The scan is much noisier and higher contrast than the others, making it look somewhat unusual, but the underlying brain anatomy still appears ordinary. The slice seems to come from the middle portion of the brain because there is still a substantial amount of brain tissue visible, and the head is positioned reasonably well with only minor off centering.
Patient 1431781 also looks relatively ordinary when viewed anatomically. The ventricles appear moderately enlarged, but not to a degree that immediately screams Alzheimer's disease. The cortex still looks reasonably preserved, and although the hippocampal region cannot be clearly evaluated from this view alone, there is nothing dramatically abnormal visible. The image brightness appears normal compared to the surrounding scans, and the slice is likely taken near the centre of the brain because major internal structures are clearly visible.
The false positive cases all seem to share a different pattern. Patient 1265863 has a relatively large ventricular space, and some of the surrounding tissue gives the impression of mild atrophy. Even though the patient is normal, I can understand why the model was fooled because the anatomy visually resembles features often associated with Alzheimer's disease. The scan also appears brighter than what would be expected from an average normal-control image. The head is positioned well, and the slice appears to be from the middle of the scan stack where the largest amount of anatomical information is visible.
Patient 1253141 shows a similarly large ventricle and relatively open-looking spaces around the central part of the brain. The cortex appears slightly thinner than some other examples, although not enough to confidently diagnose disease from visual inspection alone. Like patient 1265863, the image looks brighter overall, which is important because the earlier brightness experiment suggested that intensity alone carries some predictive information. The scan is well centred and appears to be a central sagittal slice.
Patient 1243833 displays much the same behaviour. The ventricles are clearly visible and somewhat enlarged, while the cortex remains reasonably intact. Nothing in the image looks extreme, but it does contain several characteristics that could push a classifier toward an Alzheimer's prediction. The image brightness also seems relatively high, and the head position and slice selection look normal.
Patient 1274602 is probably the strongest example of this pattern. The ventricular region is quite large and dark, making the brain look more similar to the Alzheimer's examples than to a typical healthy control. The cortex also appears somewhat thin in several areas. On top of that, the image brightness is noticeably high, particularly in the white matter regions. This supports the idea that brightness may have contributed to the model's confident mistake.
For the first question, I would say patient 985197 looks somewhat unusual because of the image quality rather than because of the anatomy. The heavy noise and contrast make the scan visually different from most of the others, but the actual brain structures appear relatively ordinary. That makes this a difficult error to explain. If the scan contained obvious Alzheimer's-related atrophy, the model's prediction would at least make some intuitive sense. Instead, the anatomy looks normal while both models remain extremely confident, which suggests they may be responding to features that are not immediately obvious to a human observer.
For the second question, both patient 1274602 and patient 1265863 appear brighter than the normal-control average. The white matter regions are particularly bright, and the overall contrast is stronger than expected. Since the brightness-only experiment achieved an accuracy of 0.585, these examples provide visual evidence that image intensity could be influencing the model's decisions. I would not claim brightness is the sole reason for the errors, but it is a plausible contributing factor.
For the third question, the false negatives do seem to share a visible trait. Neither patient 985197 nor patient 1431781 displays the kind of obvious ventricular enlargement or severe cortical thinning that many people associate with Alzheimer's disease. Their brains look relatively ordinary from these slices, which may explain why the model incorrectly classified them as normal. In contrast, many of the false positives have ventricles that look larger or more AD-like despite belonging to normal patients. This suggests the model may rely heavily on visible structural changes and struggle when Alzheimer's disease presents more subtly.
Overall, I would not trust this prototype to handle borderline cases where Alzheimer's-related changes are subtle, image brightness differs from the training average, or normal ageing produces brain structures that visually resemble mild atrophy.


---

### Finding 5. Training the overconfidence out changed who fails

The recommended model, trained with mixup, cutmix and label smoothing, fails on
a different set of patients and in a visibly different way.

![GFNet, the mistakes it was most confident about](figures/failures_gfnet.png)

| | Accuracy build | Recommended |
|---|---|---|
| Highest confidence on a false negative | **1.0000** | 0.9810 |
| Highest confidence on a false positive | **1.0000** | 0.9645 |
| Patients wrong on all 20 slices | 40 | **31** |
| Calibration error | 0.2231 | **0.0614** |

**The accuracy build puts all ten of its worst mistakes at exactly 1.0000.**
Five patients it is certain do not have Alzheimer's who do, and five it is
certain do who do not. The recommended model's worst mistake is 0.9810, and
nothing it gets wrong reaches certainty.

That difference is the whole argument for the recommended model in one number.
Both are wrong about roughly the same number of slices. One of them says so.

### Finding 6. Patient 1265863 defeats every version

Patient 1265863 is in my own image observations above. I wrote that it has a
relatively large ventricular space and looks brighter than a normal control. I
also wrote that it resembles the Alzheimer's examples closely enough that I
understood why the model was fooled.

It is still there. It appears among the most confident false positives for the original depth-8
GFNet. It appears again for the recommended model, after every regulariser in
this report had been applied.

A patient that defeats one training run is a weakness of that run. A patient
that defeats three separate training recipes, including one that eliminated
memorisation entirely, is a property of the scan. Combine that with the 23 patients wrong on all 20 slices for both
architectures. Together they are the clearest evidence in this report that some
cases are not reachable by a better classifier on a single 2D slice.

### Finding 7. Every worst failure is still a missed AD case

The lists change, the direction does not. For the recommended model:

> False negatives: 1330894, 1398870, 1240482, 1017725, 1342389
> False positives: 1265863, 1199414, 1441119, 1215774

And the five worst total failures, the patients wrong on all 20 slices and
furthest from the truth, are 1398870, 1240482, 1342389, 992839 and 1223029.
**All five have AD.**

That has held for every model tested in this project. When any of them fails
completely, it fails by calling a sick person healthy. It is the expensive
direction, and it is why the recommendation moves the decision threshold below
0.50 rather than leaving it where the softmax puts it.

---

## The filename ID is a scan ID, not a person ID

This is the most important methodological finding in the project, and it was
found by pulling on a number that looked wrong.

### Why I went looking

Both models lose heavily between validation and test:

| Model | Validation | Test | Drop |
|---|---|---|---|
| BaselineCNN | 0.9028 | 0.7421 | 16 points |
| GFNet | 0.7907 | 0.6837 | 11 points |

Both splits are patient-disjoint by ID. A 16-point drop between two clean
splits of one dataset is too large to accept without an explanation.

### Step 1. The two shipped folders are indistinguishable

`audit_split_difference.py` compares the shipped `train/` and `test/` folders on
eight measures and reports Cohen's d for each.

| Measure | train | test | Cohen's d | Size |
|---|---|---|---|---|
| Tissue mean intensity | 113.62 | 113.85 | −0.014 | negligible |
| Tissue standard deviation | 54.03 | 53.85 | 0.028 | negligible |
| Tissue 99th percentile | 212.79 | 212.89 | −0.005 | negligible |
| Head area fraction | 0.258 | 0.263 | −0.215 | small |
| Bounding box width | 148.84 | 150.78 | −0.221 | small |
| Bounding box height | 169.28 | 170.35 | −0.107 | negligible |
| Centroid row | 102.59 | 101.95 | 0.084 | negligible |
| Centroid column | 142.13 | 141.26 | 0.075 | negligible |

Slice numbering: train covers 70 to 114, test covers 75 to 114. Test is a
subset, so the models train on five slice positions that never appear at test
time. That runs in the favourable direction.

Patients in both shipped folders, by ID: zero.

So the folders do not differ in any way a pixel measure can reach, and a whole
class of explanation is closed.

### Step 2. The same person holds several IDs

ADNI follows people for years and scans them repeatedly. If one person's sessions were filed under different IDs, a validation patient
could be the same person as a training patient. The ID check would pass
regardless.

`find_duplicate_subjects.py` reduces each patient to one 48 × 48 mean image and
normalises it. It then finds the closest training patient for every validation
and test patient, by correlation. Test acts as the control, because a high
correlation means nothing on its own when all brain MRIs resemble each other.

The closest pairs settled it:

| Validation patient | Closest training patient | Correlation | ID gap |
|---|---|---|---|
| 1037251 | 1037250 | 1.0000 | 1 |
| 1029553 | 1029554 | 1.0000 | 1 |
| 1142366 | 1142367 | 1.0000 | 1 |
| 982444 | 982443 | 0.9999 | 1 |
| 1069428 | 1069430 | 1.0000 | 2 |

`measure_leakage_impact.py` then separated the pairs by correlation and
compared their ID gaps.

| Group | n | Median ID gap | Within 20 of each other |
|---|---|---|---|
| Validation, correlation ≥ 0.995 | 26 | **2** | 26 of 26 |
| Validation, correlation < 0.995 | 135 | 47,026 | 31 of 135 |
| Test, correlation ≥ 0.995 | 4 | **1** | 3 of 4 |
| Test, correlation < 0.995 | 446 | 334,840 | **0 of 446** |

Four orders of magnitude separate the two groups. Every near-duplicate sits
within 20 IDs of its match. Not one of the 446 ordinary test patients does.

**The filename ID identifies a scan, not a person.** IDs are allocated in
sequence, so one person's repeat scans land next to each other.

**16.1% of the validation patients share a person with the training set.** In
test it is 0.9%, which is consistent with the folders being separate cohorts
that a few people straddle.

### Step 3. How much it cost

Patient-level accuracy with every leaked patient removed:

| Model | Split | All patients | Leaked removed | Change |
|---|---|---|---|---|
| BaselineCNN | validation | 0.9317 | 0.9259 | −0.6 pts |
| BaselineCNN | test | 0.7511 | 0.7489 | −0.2 pts |
| GFNet | validation | 0.8199 | 0.8148 | −0.5 pts |
| GFNet | test | 0.7444 | 0.7422 | −0.2 pts |

**The leakage is real and its effect is small.** About half a point of
validation accuracy. It does not explain a 16-point gap.

The reason is visible in the per-group scores. On the 26 leaked validation
patients the baseline reaches 0.9615 against 0.9259 on the rest, which is only
3.6 points better. A memorised patient would score near 1.00. The model has
seen the same person before, that helps a little, and that is the whole effect.

One observation I will not lean on: both models score 1.0000 on the four leaked
test patients, against a base rate near 0.74. Two architectures, four for four.
Suggestive, but four patients is not evidence.

### What this changes in the report

Every accuracy in this report is computed over all patients, because that is
what a marker reproducing the run would get. The corrected figures above differ
by 0.2 to 0.6 points and change no conclusion.

What it does change is the claim. **These splits are leak-free by ID and are
not leak-free by subject.** The earlier sections say the splitting is correct,
and it is correct given the only identifier the filenames offer. It is not
enough, and I would not have known that without checking.

### What I would do with more time

Group IDs into subjects by image correlation, then split by subject rather than
by ID. The grouping rule falls straight out of the measurement: correlation at
or above 0.995 together with an ID gap under 20. That would make the splits
genuinely leak-free.

I expect it would lower validation by roughly half a point and leave test
almost unchanged, since test is 99.1% clean. It would not improve accuracy. It
would make the headline number honest, which is the better reason to do it.

### The gap is still unexplained

Folder statistics: ruled out. Subject leakage: measured and too small. What remains is that the test cohort is genuinely harder
than the training cohort. Or the two differ in something no pixel measure
reaches, such as scanner model, field strength or acquisition site. ADNI does not ship that metadata
with these JPEGs, so I cannot test it here.

I am recording the gap as open rather than attributing it to a cause I have not
demonstrated.

---

## Experiments towards the 0.80 minimum

The project specification sets a minimum of 0.80 test accuracy for GFNet on
AD against CN. This section records every experiment run towards that number, and the
failures as well. The effort behind the result is then visible in one place,
rather than scattered through the report.

**Best achieved: 0.7450 at slice level and 0.7689 at patient level.** Both come
from GFNet. The target was not reached, and the campaign below is complete
rather than abandoned.

### The ledger

| # | Experiment | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| 1 | GFNet, 30 epochs, normal augmentation | Establish a starting point | 0.6890 | baseline for the rest |
| 2 | GFNet, 80 epochs, normal augmentation | More training helps | 0.7106 | **+2.2 pts** |
| 3 | GFNet, 80 epochs, strong augmentation, weight decay 1e-3 | Less overfitting helps | 0.7122 | +0.2 pts, no real gain |
| 4 | GFNet, 150 epochs, normal augmentation | The model is underfitting and needs longer | 0.6837 | **−2.7 pts. Hypothesis wrong** |
| 5 | Patient-level aggregation of the 20 slices | Errors are scattered across slices | 0.7444 | **+3.2 pts** |
| 6 | Temperature scaling, fitted on validation | Fixes confidence, not accuracy | accuracy unchanged | as predicted, confirms the method |
| 7 | Decision threshold sweep, 0.10 to 0.60 | A better operating point exists | accuracy flat near 0.73 | moves errors between classes, adds no accuracy |
| 8 | Even ensemble with BaselineCNN | The two models fail on different patients | 0.7600 patient level | +0.9 pts, but not GFNet, and costs 14 more total failures |
| 9 | Mean-brightness-only control classifier | Check for a dataset artifact | 0.585 | the network learns real structure |
| 10 | Patch size 8 instead of 16 | Frequency resolution is the constraint | val +2.9 pts, test +0.55 pts | **worked as designed, test barely moved** |
| 11 | 2.5D input, three adjacent slices as channels | One slice lacks depth context | test −2.4 pts, AD recall −11.9 pts | **failed. Adjacent slices correlate at 0.96** |
| 12 | Audit of the shipped train and test folders | Something systematic separates them | all 8 measures negligible or small | **ruled out** |
| 13 | Subject-identity check by image correlation, test as control | The same person holds several IDs | duplicates at median ID gap 2, others at 47,026 | **confirmed. IDs are scan IDs** |
| 14 | Accuracy recomputed with leaked patients removed | The leak explains the validation-to-test gap | validation −0.6 pts, test −0.2 pts | **too small. Gap still open** |
| 15 | AdamW instead of Adam, plus a 5-epoch warmup and stochastic depth 0.1, at depth 12 | My implementation differs from how GFNet is actually trained | 0.7106 → 0.7256 | **+1.5 pts, the largest single gain** |
| 16 | The same recipe at patch size 8 | The two gains should compound | 0.7450 | **+3.4 pts over the plain run. Beats the baseline** |
| 17 | The recipe at 150 epochs instead of 80 | Every recipe run peaked late, so the schedule is the limit | 0.7431 | −0.2 pts. Converged at epoch 94 |
| 18 | Test-time augmentation, averaging each slice with its mirror | Free accuracy from a label-preserving transform | −0.06 pts | **no gain. RandomHorizontalFlip already covered it** |
| 19 | An identical configuration run twice by accident | not planned | 0.7450 against 0.7449 | **the noise floor is one slice in 9,000** |
| 20 | Mixup, cutmix and label smoothing, with strong augmentation and stochastic depth 0.3 | The model has memorised the training set | train 0.9954 → 0.7759, test −0.7 pts, **patient +1.1 pts** | **memorisation removed, test unmoved** |

### Experiment 4 in detail, because it failed informatively

I read GFNet's small validation-to-test gap as underfitting and ran it for 150
epochs with lighter augmentation. Test accuracy fell by 2.7 points.

The diagnosis was wrong, and the number that disproves it was already in the
logs. GFNet reaches a training accuracy of 0.8843 against a validation accuracy
of 0.7714. An 11-point train-to-validation gap is overfitting, not
underfitting. I had compared GFNet's validation-to-test gap against the
baseline's instead of looking at its training accuracy.

The run also showed that GFNet converges at epoch 60 and then decays.
Validation peaked at 0.7907 at epoch 60 and fell to 0.7714 for the remaining 90
epochs. Three separate runs now peak between epoch 54 and epoch 61.

One methodological note. The learning rate schedule is tied to the epoch count. So epoch 60 of a
150-epoch schedule is at a different learning rate than epoch 60 of an
80-epoch schedule. Run 4 is therefore not "run 2 but longer", it is a
different trajectory. A clean test of training length would hold the schedule
fixed and stop early.

### Experiment 10 in detail, because the result is the clearest in the project

GFNet mixes tokens by multiplying in the frequency domain. At patch 16 the token grid is 14 × 14. `rfft2` keeps half the spectrum, so the
filter learns over 14 × 8. That is 112 frequency components. Alzheimer's atrophy is
subtle and spatially local, so 112 looked like a coarse instrument for it.

Patch 8 gives a 28 × 28 grid, so 28 × 15, which is 420 components. Nearly four
times the frequency resolution, on the one mechanism GFNet actually uses.

**It worked, and the test score barely noticed.**

| Patch | Epochs | Aug | Best epoch | Validation | Test | Gap |
|---|---|---|---|---|---|---|
| 16 | 80 | normal | 54 | 0.8134 | 0.7106 | 0.103 |
| 16 | 80 | strong | 61 | 0.7891 | 0.7122 | 0.077 |
| 16 | 150 | normal | 60 | 0.7907 | 0.6837 | 0.107 |
| **8** | 80 | normal | 55 | **0.8422** | **0.7161** | **0.126** |

Validation rose 2.9 points to the best of any run. Test rose 0.55 points. The
validation-to-test gap widened to 0.126, the largest of the four.

The cost was not small:

| | Patch 16 | Patch 8 | Ratio |
|---|---|---|---|
| Parameters | 4,788,482 | 6,151,426 | 1.28× |
| Peak GPU memory | 1.58 GB | 5.97 GB | 3.78× |
| Seconds per epoch | 14.0 | 41.7 | 2.97× |
| Inference rate | 5,589 img/s | 1,448 img/s | 0.26× |

Four times the inference cost for half a point. As an engineering decision that
is a clear no.

### Experiment 11 in detail, because my own data predicted the failure

One 2D slice is a cross-section. It cannot show how a structure changes as you
move through the brain, and ventricular enlargement is exactly that kind of
change. So `--slice-context 3` feeds the slice plus its two neighbours as three
channels.

It made things worse.

| | Single slice | Three slices | Change |
|---|---|---|---|
| Validation | 0.8134 | 0.7901 | −2.3 pts |
| Test | 0.7106 | 0.6863 | **−2.4 pts** |
| AD recall | 0.5821 | 0.4632 | **−11.9 pts** |
| Seconds per epoch | 14.0 | 33.7 | 2.41× |

**The answer was in my own day-one measurement.** `visualise_data.py` reported
the correlation between flattened slice images:

```
same patient, adjacent slices:  mean 0.9612
different patients, same class: mean 0.6769
```

Adjacent slices of one patient correlate at **0.9612**. Stacking three of them
does not add depth information, it adds two near-copies of the slice already
there. The model gets 2.4 times the data loading cost for about 4% new signal
per channel.

The damage to AD recall is the more serious part. It fell from 0.5821 to
0.4632, so the model now misses 54% of Alzheimer's cases. Three correlated
channels appear to have pushed it further towards its existing preference for
predicting "normal".

I should have checked the correlation figure before running this. It was
printed in my own output on the first day, and it predicts the result.

### Experiment 20 in detail, because it is the cleanest test in the project

The training history for the best run showed what was wrong, 50 epochs before
the calibration numbers said the same thing.

| Epoch | Train accuracy | Validation accuracy | Validation loss |
|---|---|---|---|
| 15 | 0.8767 | 0.7966 | **0.459** |
| 30 | 0.9515 | 0.8388 | 0.507 |
| 45 | 0.9783 | 0.8425 | 0.682 |
| 65 | 0.9928 | 0.8689 | 0.784 |
| 80 | **0.9954** | 0.8649 | **0.871** |

Training accuracy reaches 0.9954, so the model gets 99.5% of 18,300 training
slices right. Validation loss bottoms at epoch 15 and then rises 90%, while
validation accuracy keeps drifting up. That combination has one meaning: the
model is becoming more confident about the answers it gets wrong.

Three things were missing that attack memorisation directly, and the earlier
recipe work had not touched any of them: mixup, cutmix and label smoothing.

**They worked exactly as designed.**

| | Without | With |
|---|---|---|
| Train accuracy, final | 0.9954 | **0.7759** |
| Train-to-validation gap | 13.0 pts | **5.1 pts** |
| Validation loss, final | 0.871 | **0.449** |
| Validation loss behaviour | rose 90% above its minimum | **fell monotonically to the end** |
| Validation-to-test gap | 12.4 pts | **8.9 pts** |
| Test accuracy | 0.7450 | 0.7382 |
| **Patient accuracy** | 0.7578 | **0.7689** |
| **AD recall** | 0.5729 | **0.6092** |
| **Calibration error** | 0.2231 | **0.0614** |

Memorisation removed. Overconfidence removed. Calibration error cut by a factor
of 3.6. Patient accuracy up 1.1 points. 163 fewer missed Alzheimer's slices.

**And slice-level test accuracy fell 0.7 points.**

That is the strongest evidence in this report that memorisation was never the
cause of the gap between validation and test. The hypothesis was stated and the standard intervention was applied. The
intervention was confirmed to have worked on its own terms. The target metric
did not move.

### Experiment 19, which happened by accident and is the most useful

A configuration was re-run by mistake with identical arguments and the same
seed. The two runs returned **0.7450 and 0.7449**: a difference of one slice in
9,000, or 0.01 accuracy points.

That turns a vague limitation into a measurement. `cudnn.benchmark` is left on,
so results are not bit-exact, and the cost of that is now known rather than
guessed. It also means every difference of 0.3 points or more in this report is real
rather than run-to-run variation. That is what makes the rest of the ledger
readable.

### What the ledger shows

Lining all twelve GFNet runs up next to each other gives the central result of
this project.

| Patch | Epochs | Aug | Slices | Recipe | Best epoch | Validation | Test |
|---|---|---|---|---|---|---|---|
| 8 | 80 | normal | 1 | AdamW, dp 0.1 | 65 | 0.8689 | **0.7450** |
| 8 | 150 | normal | 1 | AdamW, dp 0.1 | 94 | **0.8742** | 0.7431 |
| 8 | 80 | strong | 1 | AdamW, dp 0.3, mixup | 71 | 0.8273 | 0.7382 |
| 16 | 150 | normal | 1 | AdamW, dp 0.1 | 116 | 0.8680 | 0.7294 |
| 16 | 80 | normal | 1 | AdamW, dp 0.1 | 73 | 0.8453 | 0.7256 |
| 8 | 80 | normal | 1 | plain Adam | 55 | 0.8422 | 0.7161 |
| 16 | 80 | strong | 1 | plain Adam | 61 | 0.7891 | 0.7122 |
| 16 | 80 | normal | 1 | plain Adam | 54 | 0.8134 | 0.7106 |
| 16 | 30 | normal | 1 | plain Adam | | | 0.6890 |
| 16 | 80 | normal | 3 | plain Adam | 63 | 0.7901 | 0.6863 |
| 16 | 150 | normal | 1 | plain Adam | 60 | 0.7907 | 0.6837 |

**Validation moved 8.5 points across these. Test moved 6.1, and the ordering
does not match.** The run with the highest validation accuracy of all, 0.8742,
is not the best on test. The run with the lowest, 0.7891, beats two runs with
higher validation scores.

The configurations span seven separate axes:

| Axis | Varied over |
|---|---|
| Training length | 30, 80 and 150 epochs |
| Augmentation strength | normal and strong |
| Weight decay | 1e-4 and 1e-3 |
| Token resolution | 112 and 420 frequency components |
| Input representation | one slice and three adjacent slices |
| Optimiser and schedule | Adam, and AdamW with warmup |
| Regularisation | stochastic depth 0, 0.1 and 0.3, mixup, cutmix, label smoothing |

**Validation responds to all seven. Test barely responds to any of them.** The
gap widens as validation improves, which is the signature of a model learning a
cohort rather than a disease.

So the binding constraint is not the model. It is the difference between the cohort the model trains on and the
cohort it is tested on. That is the same gap the section below could not
explain. Capacity cannot close it, because
capacity is spent fitting a cohort the test set does not resemble.

### The campaign is complete and the target was not reached

**The best GFNet result is 0.7450 at slice level, 5.5 points short of the 0.80
minimum.** The best patient-level result is 0.7689, which is 3.1 points short.

I am stopping the search here, and the reason is evidence rather than time.
Every lever I can identify acts on model capacity, training schedule or
regularisation, and all three axes are exhausted:

| Lever | Effect on test accuracy |
|---|---|
| Four times the frequency resolution | +0.6 |
| The paper's optimiser, warmup and stochastic depth | +1.5 |
| Nearly doubling the schedule | −0.2 |
| Three input channels instead of one | −2.4 |
| Averaging over a mirrored copy at inference | −0.06 |
| Eliminating memorisation entirely | −0.7 slice, +1.1 patient |

The last row is the one that closes the argument. Mixup removed the
memorisation completely, training accuracy fell from 0.9954 to 0.7759, and
validation loss stopped rising. The intervention did exactly what it claims to
do, and the test score did not improve.

What would be worth trying next, outside the scope available here:

| Idea | Why it might work | Why it is out of scope |
|---|---|---|
| True 3D volumes | Whole-brain context rather than near-duplicate slices | The shipped data is 2D JPEGs |
| Non-adjacent slices, widely spaced | Lower correlation than the 0.9612 of neighbours | Needs more than 20 slices per patient |
| ImageNet-pretrained GFNet weights | Features learned on a different and far larger distribution are the one thing that attacks transfer directly | Needs approval, and the released checkpoints use different layer widths |
| Subject-level splits plus retraining | Removes the 16.1% ID leakage | Lowers validation, unlikely to raise test |
| Train and test pooled, then re-split by subject | Removes the cohort difference entirely | Discards the shipped held-out set, so it answers a different question |

The last row is the honest one. The cohort difference could be removed by
pooling both folders and splitting afresh, and the number would very likely
clear 0.80. It would also no longer be a measurement on a held-out cohort,
which is the measurement that matters clinically. I would rather report 0.7450
on a genuine held-out set than 0.85 on a set I constructed to be easier.

### The unexplained gap

Both models lose heavily between validation and test, and this is the largest
unexplained effect in the project.

| Model | Validation | Test | Drop |
|---|---|---|---|
| BaselineCNN | 0.9028 | 0.7421 | 16 points |
| GFNet | 0.7907 | 0.6837 | 11 points |

Both splits are patient-disjoint. Validation is drawn from the shipped `train/`
folder and test is the shipped `test/` folder, used exactly as distributed.

Closing even half of that gap would put GFNet near the target, which makes it a
better target for effort than further tuning. `audit_split_difference.py`
compares the two folders on eight measures: slice
numbering, patient ID ranges, head area and position in the frame, tissue
contrast and intensity percentiles. It reports an effect size for each.

Pixel intensity histograms were checked earlier and matched, which rules out
brightness as the cause but not the rest.

---

---

## References

[1] Y. Rao, W. Zhao, Z. Zhu, J. Lu, J. Zhou, "Global filter networks for image
classification," *IEEE Transactions on Pattern Analysis and Machine
Intelligence*, vol. 45, no. 6, pp. 7837–7850, 2023.

[2] Alzheimer's Disease Neuroimaging Initiative. http://adni.loni.usc.edu

[3] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger, "On calibration of modern
neural networks," in *Proc. ICML*, 2017. This paper is the source of the
temperature scaling method and of the Expected Calibration Error metric used in
this report.

[4] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, H. Jegou,
"Training data-efficient image transformers and distillation through
attention," in *Proc. ICML*, 2021. This paper is the source of the training
recipe used here: AdamW with decoupled weight decay, a linear warmup,
stochastic depth, mixup, cutmix and label smoothing.
