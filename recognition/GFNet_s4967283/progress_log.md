# Progress log, COMP3710 Pattern Analysis Project

Project: GFNet vs BaselineCNN on ADNI (Alzheimer's vs Cognitively Normal)
Folder: `recognition/GFNet_s4967283/`
Branch: `topic-recognition`

---

## Day 1. Tuesday 13 October

### What I decided

Picked **GFNet on the ADNI dataset**, classifying brain MRI slices as AD
(Alzheimer's Disease) or NC (Cognitively Normal). This is a Hard difficulty
option, which matters because Easy caps the implementation marks at 10/20 and
Normal at 15/20. Hard is the only tier that leaves enough headroom for a 7.

Ruled out anything on the OASIS dataset, because I already did a UNet and a VAE
on OASIS for Demo 2 and the spec says the final model has to be substantially
different from previous work.

### What I built

**`audit_adni.py`** is a script to inspect the dataset before writing any loader.
I wrote this first because the data loader has to split by patient, and I could
not do that until I knew whether patient IDs were even recoverable from the file
names.

**`modules.py`** holds the baseline CNN. Four conv blocks (conv, batch norm, ReLU,
max pool) growing 1 → 32 → 64 → 128 → 256 channels, then global average pooling
and a small fully connected head. 421,474 trainable parameters.

Two choices worth remembering:
- Global average pooling instead of flattening a fixed grid, so the model works
  at any input resolution. Verified by running it at both 224 and 256.
- Two output logits rather than one sigmoid, so I get a proper probability per
  class after softmax. I need calibrated confidence later for the uncertainty
  investigation, and this keeps the baseline and GFNet handling identical.

**`dataset.py`** is the data loader, with patient-level splitting.

**`.gitignore`** blocks `__pycache__`, vim swap files, model weights and
dataset files. The spec says committing large files gets the pull request
rejected, so `*.pt` and the dataset patterns matter more than the editor junk.

### What the audit found

```
train/AD   10400 images   520 patients
train/NC   11120 images   556 patients
test/AD     4460 images   223 patients
test/NC     4540 images   227 patients
```

- File names are `<patientID>_<slice>.jpeg`, e.g. `1003730_100.jpeg`
- Every patient contributes exactly 20 slices
- Images are 256 wide × 240 high, greyscale, pixel values 0–255
- No patient appears in both train and test, so the shipped split is clean

### The leakage problem, and what I did about it

Each patient has 20 slices cut from the same 3D scan, so those images look very
similar to one another. A random image-level split would scatter one person's
slices across training and validation. The model could then recognise the
patient instead of the disease, validation accuracy would look great for the
wrong reason, and the test score would collapse with no obvious explanation.

So every split happens at the patient level: I choose which patients go where
first, then all 20 of their slices follow. `check_no_patient_overlap` raises an
`AssertionError` if a patient ever lands in two splits. It crashes rather than
warns, because if that fires every number downstream is worthless.

I also split the two classes separately, so validation gets a fair share of each
rather than whatever the shuffle happened to produce.

### Splits I ended up with

| Split      | Images | Patients | NC    | AD    |
|------------|--------|----------|-------|-------|
| train      | 18300  | 915      | 9460  | 8840  |
| validation | 3220   | 161      | 1660  | 1560  |
| test       | 9000   | 450      | 4540  | 4460  |

Validation is 15.0% of the training patients. Stratification held up: 78 of 520
AD patients and 83 of 556 NC patients, both within a tenth of a percent of 15%.

Class balance overall is NC 51.7% / AD 48.3%. Mild enough that I do not need
weighted loss or resampling. I checked rather than assumed.

### Normalisation

Measured over 2000 training images (training split only, so nothing leaks in
from validation or test):

```
PIXEL_MEAN = 0.1159
PIXEL_STD  = 0.2236
```

The mean is low because these slices are mostly black background. With the
placeholder 0.5/0.5 the normalised range was −1.000 to 0.953, which squashed the
actual brain tissue into a narrow band. With the measured values the range is
−0.518 to 3.849: background sits near −0.52 and tissue spreads across the
positive range. The asymmetry reflects the data honestly.

### Other design decisions I need to be able to defend

**Resize 256×240 → 224×224.** GFNet needs a fixed square input. I resize rather
than crop, because cropping could cut off the outer cortex, which is exactly the
tissue that atrophies in Alzheimer's. The stretch is identical for every image so
the model sees a consistent world.

**Horizontal flip augmentation.** The brain is roughly symmetric and these slices
are not labelled by side, so a mirrored brain is still plausible. Honest caveat:
Alzheimer's atrophy is not always symmetric, so this does throw away a little
real signal. I kept it for the extra variety, but it is a decision to test rather
than assume.

**Small rotation and shift (5°, 5%).** Patients are not positioned identically in
the scanner, so this is variation the model will genuinely meet. Kept small
because a heavily rotated slice is not realistic.

### Mistake I made

Committed my first two commits to `main` instead of `topic-recognition`. Fixed by
cherry-picking them onto the right branch rather than resetting history. The
commit log is a pass hurdle and deleting commits would have been worse than the
original mistake. Left the duplicates on `main`, since nobody marks that branch.

Added a habit: run `git branch --show-current` at the start of every session.

---

## Day 2. Wednesday 14 October

### The main job: the feasibility review

The midpoint check-off is an ungraded hurdle. If I do not complete it I cannot
pass the assessment at all, whatever the code looks like. So the whole evening
went on the one-page written review for Thursday morning.

It covers the four things the spec asks for:

1. **User need and acceptance criteria.** A clinician doing early screening,
   five testable criteria including a resource budget of 16 GB and 2 hours.
2. **Model choice.** GFNet, Hard tier, justified against the Fourier material
   and my own Demo 2 Part 1 work where I measured naive DFT at 34.2 s against
   FFT at 0.13 ms for N=4096.
3. **Preliminary evidence.** The data audit from Day 1, the splits, and the
   measured normalisation constants.
4. **Risks and fallback.** GFNet's FFT block is the part I am least sure of.
   fallback is ConvNeXt on the same dataset, same dilemma, same evaluation code.

I framed the whole thing around the confidence dilemma rather than around
accuracy, because that is what the spec actually asks me to investigate.

### What I built

**`train.py`** is the training loop. Validates every epoch, saves the best
checkpoint on validation accuracy rather than the last epoch, plots loss and
metric curves, and records peak GPU memory, timing and parameter count.

The important design choice: it saves `test_predictions_<model>.npz` with the
per-image probabilities, true labels and patient IDs. That means the whole
calibration investigation can run later without retraining anything, and the
patient IDs let me tell apart "five mistakes from one patient" and "five
mistakes from five patients". Those are completely different failure modes.

**`run_smoke.sh`** is a SLURM script for a short 3-epoch test.

### Mistakes I made

**Wrong partition.** I first wrote `--partition=a100`. My Demo 2 notes have two
findings about this and I used the stale one. The early note said the comp3710
partition rejects jobs. The later note said the rejection was caused by a
missing `--account=comp3710`. The right answer is `--partition=comp3710`, which
is the partition allocated to this course. The a100 partition shares nodes with
cosc3500 and a100-grind, so jobs sit pending behind other courses.

I now have the reasoning for every SLURM flag written into `run_smoke.sh` as
comments, so neither I nor anyone reading the repo has to rediscover it.

**Commits that did not land.** I pasted several `git add` and `git commit` pairs
for the same file. Only the last one took effect. Staging the same file twice
leaves nothing for the second commit. One commit per logical change, one file at
a time.

**Local environment.** My Windows machine had CPU-only `torch` with a CUDA
`torchvision`, which made `import torchvision` fail outright. Fixed with a
force-reinstall of both from the cu128 index in one command. Rule: install torch
and torchvision together, never separately.

---

## Day 3. Thursday 15 October

### The check-off is done

Hurdle cleared. The tutor read the review, gave brief feedback and recorded
completion. Their one warning was about data leakage.

I had already handled the main case with patient-level splitting. But there is
one thing I cannot verify from the preprocessed data: I assume the number before
the underscore is a patient ID rather than a scan or session ID. ADNI is a
longitudinal study, so the same person can be scanned more than once. I asked
and the tutor confirmed they are patient IDs.

### The smoke test worked

```
epoch 1: 75.4s    epoch 2: 16.1s    epoch 3: 15.4s
test accuracy 0.7081, peak VRAM 2.32 GB
```

Epoch 1 is slow because of cuDNN algorithm selection and file caching. Steady
state is about 16 s per epoch, so a 30-epoch run takes under 10 minutes. My
feasibility review estimated 6 to 8 A100-hours. The real figure is a small
fraction of that, which means I can afford multiple seeds and ablations.

This also closed risk 2 from the feasibility review: `torch 2.14.0+cu130`,
`cuda available: True`, `NVIDIA A100-PCIE-40GB`.

### What I built

**`visualise_data.py`** draws four figures of the actual data, plus a measured
within-patient against between-patient correlation. Figure 2 shows all 20 slices
of one patient side by side, which is the visual proof of why the split must be
by patient.

**GFNet in `modules.py`.** Patch embedding, the global filter block, the
feed-forward block, position embeddings and a mean-pooled classifier head.
4,788,482 parameters, which is 11.4 times the baseline.

### The test that mattered

Six self-tests, all passing. The important one sets every filter weight to
exactly 1 + 0i. FFT, multiply by one, inverse FFT should return the input
unchanged. Largest difference: **4.768e-07**, which is float32 rounding error.

That proves the transform plumbing is right: the dimensions, `norm="ortho"`,
the `s=` argument on the inverse, and the half-spectrum width. The rfft output
is 14 × 8, not 14 × 14, because a real input has a symmetric spectrum. Getting
that wrong gives either a shape error or a silent broadcast that trains badly
for no visible reason.

Risk 1 from the feasibility review is now closed, and it cost no GPU time.

---

## Day 4. Friday 16 October

I lost Thursday evening and most of Friday. Caught up in one long session.

### What I built

- Wired gfnet into `train.py` through a shared `build_model` in `modules.py`
- `run_train.sh`, one script that takes model, seed and epochs as arguments
- `evaluate_calibration.py`, which gives ECE, reliability diagrams, the reject-option
  threshold sweep, and patient-level aggregation
- `brightness_baseline.py`, a deliberately stupid classifier. See below

### First real results (30 epochs, learning rate 3e-4)

| Model | Test accuracy | Macro F1 | AD recall |
|---|---|---|---|
| Baseline | 0.7241 | 0.7138 | 0.5390 |
| GFNet | 0.6890 | 0.6749 | 0.4848 |

GFNet lost. That is a valid answer to the spec's question about whether the
extra complexity is justified, not a failure.

### The finding that matters most

Both models prefer to say "normal". The baseline misses 46% of AD patients. The
GFNet misses 52%. For a screening tool that is the worst possible failure mode,
because a missed case gets no follow-up and nobody ever checks.

AD precision is high (0.85, 0.81). So when a model says AD it is usually right.
The problem is that it rarely says AD.

### Patient aggregation does not rescue it

I expected a large gain from averaging each patient's 20 slices. I got +1.1 and
+2.0 points. The reason: 51 patients have every single one of their 20 slices
wrong. Averaging cannot fix a patient the model misreads 20 times out of 20.
The errors are patient-systematic, not slice-random.

### Both models are confidently wrong

Mean confidence 0.929 when right, 0.878 when wrong. Separation 0.051. ECE 0.191.

The spec asks whether the network "remains stubbornly overconfident even when
making incorrect predictions". Mine does, and now I have the number.

### A lower threshold is nearly free

At an AD threshold of 0.10 instead of 0.50, the baseline misses 1,383 AD slices
instead of 2,056. That is 673 fewer missed cases, and accuracy goes slightly
**up**, from 0.7241 to 0.7382.

So the default 0.5 is the wrong operating point for a screening tool. But the
threshold only moves errors between classes. It does not add discriminative
power, and accuracy stays near 0.73 at every threshold.

### The brightness experiment

The intensity histograms showed AD tissue is darker than NC tissue. That raised
a question I had to answer: does my model read anatomy, or does it read
brightness?

So I built a classifier that uses one number per slice, the mean intensity of
its tissue pixels, and fitted the threshold on the training split only.

```
Brightness rule:  0.585
GFNet:            0.689
BaselineCNN:      0.724
```

The CNN adds 13.9 points over the brightness rule, so the networks learn real
structure. But 0.585 is above chance, so about 8.5 points of the signal is
brightness. Both numbers go in the report.

I also checked the train and test intensity histograms against each other. They
look the same, so a distribution shift between the shipped folders is not the
explanation for the validation-to-test gap.

---

## Day 5. Saturday 17 October

### Results after 80 epochs

| Model | Val | Test slice | Test patient | Best epoch |
|---|---|---|---|---|
| Baseline | 0.9326 | 0.7360 | 0.7422 | 75 of 80 |
| GFNet | 0.8134 | 0.7106 | 0.7422 | 54 of 80 |

Doubling the training bought 1.2 points for the baseline and 2.2 for GFNet.
GFNet reached its best at epoch 54 and did not improve after. It has converged.

### The gap is overfitting, not a data problem

| Baseline | Validation | Test | Gap |
|---|---|---|---|
| 30 epochs | 0.8894 | 0.7241 | 0.165 |
| 80 epochs | 0.9326 | 0.7360 | 0.197 |

Validation rose 4.3 points. Test rose 1.2. The gap got wider. The model fits the
training cohort better and better, and the test cohort does not follow.

### Longer training made the confidence worse

| Baseline | ECE | Confidence when wrong | Separation |
|---|---|---|---|
| 30 epochs | 0.191 | 0.878 | 0.051 |
| 80 epochs | 0.216 | 0.922 | 0.041 |

More certain, not more correct. A wrong prediction now claims 92% certainty.

### The two models tie at patient level

Both get exactly 334 of 450 patients right. But the failure shapes differ:

| Model | All 20 slices right | All 20 slices wrong |
|---|---|---|
| Baseline | 246 | 60 |
| GFNet | 159 | 35 |

GFNet has 42% fewer catastrophic patients. A patient wrong on all 20 slices is
one the system misclassifies with no internal disagreement to flag it.

### Temperature scaling fixes GFNet but not the baseline

| Model | Temperature | ECE before | ECE after | Change |
|---|---|---|---|---|
| Baseline | 1.36 | 0.2164 | 0.1963 | −9% |
| GFNet | 1.90 | 0.1819 | 0.1102 | −39% |

Accuracy did not move for either model, which is the expected behaviour and
confirms the implementation is right.

GFNet's overconfidence is a single global scaling problem, and one parameter
fixes most of it. The baseline's is structural, so the same fix barely helps.

I fitted the temperature on the validation split only. Fitting it on test and
then reporting test calibration would be tuning on the test set, which is the
same sin the patient-level splitting exists to avoid.

### What I built

**`predict.py`** is the fifth required file. Loads a checkpoint, runs inference,
prints a confusion matrix and per-class metrics, saves the probabilities, and
draws three figures: example predictions, the most confident mistakes for the
error autopsy, and the learned GFNet frequency filters.

### Where the recommendation stands

| Measure | Baseline | GFNet |
|---|---|---|
| Slice accuracy | **0.7360** | 0.7106 |
| Patient accuracy | 0.7422 | 0.7422 |
| Catastrophic patients | 60 | **35** |
| ECE raw | 0.2164 | **0.1819** |
| ECE after scaling | 0.1963 | **0.1102** |
| Parameters | **421,474** | 4,788,482 |
| Inference | **12,263 img/s** | 4,640 img/s |
| Peak memory | 2.32 GB | **1.58 GB** |

The baseline is the better classifier. GFNet is the more trustworthy one. For a
triage system whose job is to route uncertain cases to a specialist, trustworthy
confidence is the property that matters.

### Mistakes today

1. Ran `predict.py` on the login node. It used CPU and four worker processes on
   a shared machine. Rule I am keeping: a GPU, any `num_workers` above zero, or
   anything over a few minutes goes to SLURM.
2. A stray `)` in `train.py` killed a job after 7 seconds. I now run
   `python -c "import ast; ast.parse(open('train.py').read())"` after every edit,
   before submitting anything.
3. An `srun` session timed out unused and wasted a 20-minute GPU allocation. Use
   a realistic `--time`, and `exit` when finished.

---

## Day 6. Monday 19 October

I took Sunday off. Both strong-augmentation jobs had finished by Monday
morning.

### The strong-augmentation results

| Measure | baseline_strong | gfnet_strong |
|---|---|---|
| Slice accuracy | **0.7421** | 0.7122 |
| Patient accuracy | **0.7511** | 0.7444 |
| Macro F1 | **0.7357** | 0.7069 |
| AD recall | **0.5913** | 0.5821 |
| ECE | 0.1812 | **0.0934** |
| Confidence separation | 0.0579 | **0.0782** |
| Patients wrong on all 20 slices | 40 | **28** |
| Best epoch of 80 | 79 | 61 |

### Finding. The intervention worked on every measure at once

Validation fell from 0.9326 to 0.9028 and test rose from 0.7360 to 0.7421. The
gap closed from 0.197 to 0.161. AD recall gained 4.5 points, which is 199 fewer
missed Alzheimer's cases. ECE fell 16% and the confidence separation rose 42%.

A change that improves every measure at once is rare. It means the diagnosis was
right: the model was fitting its training cohort, and less of that helped
everything downstream.

### Finding. GFNet is almost twice as well calibrated

GFNet's ECE is 0.0934 against the baseline's 0.1812. Its worst single bin is out
by 0.159 and not by 0.260. It also gains +3.2 points from patient aggregation
against the baseline's +0.9, which means its errors are scattered across slices
rather than locked to particular patients.

So the two models point in opposite directions. The baseline predicts better.
GFNet knows better what it does not know.

### Other work

- Wrote `README.md` in full and filled every number. It is the 10-mark
  deliverable and it was the last required file missing.
- Rewrote the whole README in plain language: no em dashes, no semicolons, no
  sentence over 25 words, and one word for one thing throughout. Twenty-eight
  em dashes came out.
- Committed `.gitignore` and `train.py`.

---

## Day 7. Tuesday 20 October

### Rebuilt every figure from the strong-augmentation checkpoints

The figures in `figures/` came from the normal-augmentation runs, but the README
now reports strong-augmentation numbers. A figure that disagrees with the table
beside it is worse than no figure. I wrote `run_figures.sh`, which runs five
predict and calibration commands inside one GPU allocation instead of five jobs
that each queue separately.

### Finding. The temperature refit corrected a claim I had made

| Run | Temperature | ECE before | ECE after | Change |
|---|---|---|---|---|
| GFNet, normal aug | 1.90 | 0.1819 | 0.1102 | −39% |
| GFNet, strong aug | 1.11 | 0.0934 | 0.0787 | −16% |
| BaselineCNN, normal aug | 1.36 | 0.2164 | 0.1963 | −9% |
| BaselineCNN, strong aug | 1.20 | 0.1812 | 0.1651 | −9% |

I had written that GFNet's miscalibration was fixable and the baseline's was
not, on the strength of a 39% cut against a 9% cut. On the strong-augmentation
models the gap is 16% against 9%. The original claim came from one run and was
too strong. I corrected it in the README.

The replacement finding is better than the one it replaced. GFNet's fitted
temperature fell from 1.90 to 1.11. A temperature near 1 means there is almost
no global overconfidence left to remove, so strong augmentation had already done
the work that temperature scaling was doing. The two treat the same problem.
Augmentation treats it better, because it also raises accuracy and temperature
scaling cannot.

### Wrote `inspect_patient.py` and found the best data in the report

`predict.py` names the patients each model was most confidently wrong about.
That is where the autopsy starts, not where it ends. `inspect_patient.py` reads
the saved `.npz` files and prints every slice of a named patient for both
models, so I can tell a patient the model is unsure about from a patient the
model is confidently and completely wrong about.

**Patient 985197 has AD. The baseline returned P(AD) below 0.00005 on all 20
slices.** It prints as 0.0000. The softmax saturated. The model did not report
low confidence in Alzheimer's, it reported none at all, twenty times, on twenty
images of a sick brain. GFNet is also wrong on all 20 slices, at a mean of
0.0086, which is about 30 times less extreme.

**Every one of the twenty worst total failures, across both models, is an AD
patient.** No healthy patient appears in either list. The catastrophic failures
all run in the false-negative direction, which is the expensive one.

**Twenty patients are wrong on all 20 slices for both models.** A convolutional
network and a frequency-domain network read an image in completely different
ways and still agree on every slice of all twenty. That is 4.4% of the test set
and it sets a floor for any ensemble. The other 28 failures are specific to one
architecture, so disagreement between the two models is a usable referral
signal. I added that as a fifth engineering recommendation.

**The two models fail in different shapes.** Patient 1274602 is healthy. The
baseline is wrong on 20 of 20 slices at a mean of 0.9656. GFNet is right on 11
of 20 and sits at 0.5402, so it lands on the wrong side by 0.04 rather than by
0.47. The ECE numbers are an average over 9,000 slices. These four patients are
what that average looks like one case at a time.

### Mistakes today

1. I deleted the course's own `README.md` at the repository root, in the belief
   that it stopped mine from displaying. It does not. GitHub renders a README in
   every directory, so the course README at the root and my report at
   `recognition/GFNet_s4967283/README.md` both display. Restored from upstream.
2. I edited a file on github.com, which created a commit my local clone did not
   have, and the next push was rejected with "fetch first". Fixed with
   `git pull --rebase`. Rule I am keeping: edit locally, commit, push. Never use
   the web editor on a repository I am working in.
3. I renamed `inspect_patient.py` to `inspect.py`, which shadows the Python
   standard library `inspect` module. NumPy imports `inspect` during startup and
   got my file instead, so `import numpy` failed with an error that looked like
   a NumPy bug. Python searches the script's own directory first. Renamed back.

---

## Day 8. Wednesday 21 October

No cluster time today. I audited the marks instead.

### Working out what 80% actually refers to

Three different figures were being confused, including by me.

| What | Means |
|---|---|
| Model accuracy 0.80 | GFNet on the ADNI test split |
| Project mark 80% | 32 of 40 marks, the grade 7 condition |
| Course mark 85% | across all four assessments |

I went back through the specification criterion by criterion and worked out
where the 40 marks sit, and which of them are already secure.

The useful outcome: **about 4.5 marks were sitting unclaimed and needed no
cluster time.** The pull request, the visual half of the error autopsy, the AI
disclosure, and the Turnitin export. The pull request alone is a pass hurdle,
so missing it would zero all forty marks.

I also drafted the pull request description.

---

## Day 9. Thursday 22 October

### I had the 0.80 rule wrong, and so did my notes

I had been treating the 0.80 accuracy target as something I set myself in the
feasibility review. It is in the specification, in the model table:

> GFNet [Hard Difficulty] Use ADNI dataset. Global Filter Network using 2D FFT
> token mixing; **minimum 0.8 test accuracy on AD vs CN.**

It also applies to GFNet, not to the baseline. So the gap is measured from
0.7122, not from the baseline's 0.7421, and it is 7 to 9 points rather than 5.

Worth recording because I spent part of a day planning against the wrong
target.

---

## Day 10. Friday 23 October

Tutor feedback arrived: drop the feasibility review from the README, add a
repository tree, and either hit 0.80 or show enough experimental evidence of
trying.

I kept the feasibility review. The written specification says its technical
quality is marked as part of the report, and the cost of keeping it is a
slightly longer README against the cost of losing marks the specification
allocates. I will ask for the instruction in writing.

The third point is the useful one. **A documented campaign can substitute for
the number.** So I built the experiment ledger, which collects every experiment
into one table a marker can read in thirty seconds.

### The ensemble

| | Slice | Patient | Patients wrong on all 20 |
|---|---|---|---|
| BaselineCNN | 0.7421 | 0.7511 | 40 |
| GFNet | 0.7122 | 0.7444 | **28** |
| Even split | **0.7494** | **0.7600** | 42 |

The ensemble buys 0.9 patient points and costs the best safety property in the
report. Catastrophic failures rise from 28 to 42, because averaging a
confidently wrong baseline against a hedging GFNet pushes GFNet's
borderline-correct slices over the line.

I report the even split. The sweep found 0.7622 at weight 0.4, but that weight
was chosen by looking at the test split, so claiming it would be tuning on test.

### The 150-epoch run failed, and the diagnosis behind it was wrong

I read GFNet's small validation-to-test gap as underfitting and ran it longer
with lighter augmentation. Test accuracy fell 2.7 points, from 0.7106 to 0.6837.

The number that disproves the diagnosis was already in the logs. GFNet reaches
a training accuracy of 0.8843 against a validation accuracy of 0.7714. An
11-point train-to-validation gap is overfitting. I had compared GFNet's
validation-to-test gap against the baseline's instead of looking at its
training accuracy.

One more thing the run showed: the learning rate schedule is tied to the epoch
count, so epoch 60 of a 150-epoch schedule sits at a different learning rate
than epoch 60 of an 80-epoch schedule. It is not the same run with more epochs.

### Chasing the validation-to-test gap

Both models lose 11 to 16 points between validation and test. Both splits are
patient-disjoint. That is too large to accept without an explanation.

**First I compared the two shipped folders** on eight measures. Slice
numbering, head size, head position, bounding box, tissue brightness, contrast
and intensity percentiles. Every one came back negligible or small, the largest
at Cohen's d 0.22. The folders are indistinguishable, so a whole class of
explanation is closed.

**Then I tested whether one person holds several IDs.** ADNI follows people for
years. If sessions were filed under different IDs, a validation patient could
be the same person as a training patient, and my ID-based check would pass
regardless.

It is true. The evidence is not subtle:

| Group | n | Median ID gap | Within 20 |
|---|---|---|---|
| Validation, correlation >= 0.995 | 26 | **2** | 26 of 26 |
| Validation, correlation < 0.995 | 135 | 47,026 | 31 of 135 |
| Test, correlation >= 0.995 | 4 | **1** | 3 of 4 |
| Test, correlation < 0.995 | 446 | 334,840 | **0 of 446** |

Four orders of magnitude. The filename ID identifies a scan, not a person, and
IDs run in sequence so one person's repeat scans land next to each other.
16.1% of my validation patients share a person with training.

**Then I measured what it costs.** About half a point of validation accuracy.
It is real and it does not explain a 16-point gap.

So I have two findings instead of one: a methodological flaw I can quantify,
and a gap that stays open. I recorded the gap as unexplained rather than
attributing it to a cause I cannot demonstrate.

### Mistakes today

1. I deleted the course's own `README.md` at the repository root, believing it
   stopped mine from displaying. It does not. GitHub renders a README in every
   directory. Restored from upstream.
2. I edited a file on github.com, which created a commit my local clone did not
   have, and the next push was rejected. Rule I am keeping: edit locally,
   commit, push. Never the web editor on a repository I am working in.
3. I named a script `inspect.py`, which shadows the Python standard library
   `inspect` module. NumPy imports it during startup and got my file instead,
   so `import numpy` failed with what looked like a NumPy bug. Python searches
   the script's own directory first.
4. My duplicate-detection script printed the wrong verdict, because I summarised
   with a difference in means. Only a minority of patients are duplicates, so
   the mean washed the effect out. The percentiles were in the same output and
   told the real story. I had printed them without reading them.

---

## Day 11. Saturday 24 October

### Patch size 8: it worked, and the test score did not care

GFNet mixes tokens by multiplying in the frequency domain. At patch 16 the
filter learns over 112 frequency components. At patch 8 it learns over 420.

| Patch | Epochs | Aug | Best epoch | Validation | Test | Gap |
|---|---|---|---|---|---|---|
| 16 | 80 | normal | 54 | 0.8134 | 0.7106 | 0.103 |
| 16 | 80 | strong | 61 | 0.7891 | 0.7122 | 0.077 |
| 16 | 150 | normal | 60 | 0.7907 | 0.6837 | 0.107 |
| **8** | 80 | normal | 55 | **0.8422** | **0.7161** | **0.126** |

Validation rose 2.9 points to the best of any run. Test rose 0.55 points. The
gap widened to the largest yet. The cost was 3.8 times the memory, 3 times the
training time, and inference down from 5,589 to 1,448 images per second.

### The result that explains all the others

Lining the four GFNet runs up gives the central finding of the project.

**Validation moved 5.3 points across these configurations. Test moved 3.2, and
not in step with it.** The run with the lowest validation accuracy beat the run
with 0.7907 on test.

Learning rate, epoch count, augmentation strength, weight decay, token
resolution and parameter count all change how well the model fits the training
cohort. Validation responds to all of them. Test responds to none.

So the constraint is not the model. It is the difference between the cohort the
model trains on and the cohort it is tested on, which is the same 16-point gap
I could not explain yesterday. Capacity cannot close it, because capacity is
spent fitting a cohort the test set does not resemble.

### 2.5D input

Added `--slice-context 3`: the slice plus its two neighbours, stacked as three
channels. The neighbours always come from the same patient, and edge slices
repeat so every item has three channels.

The trap is ordering. The three slices are merged into one image **before** the
transform runs. Transformed separately, each would draw its own random rotation
and shift, the channels would stop lining up, and the augmentation meant to
help would destroy the depth signal. Eight tests pass, including twenty random
draws with strong augmentation producing zero channel disagreement.

This is the last accuracy experiment I plan to run. It is the only remaining
lever that gives the model different information rather than more capacity over
the same information.

### Documentation

- Every command in the README now carries its real expected output and a line
  saying what to check. The specification asks for example inputs and outputs
  directly, and my tutor asked for the same thing.
- Added the repository tree.
- Found that the README referenced `find_duplicate_subjects.py` and
  `measure_leakage_impact.py` while the files were named `duplicate_subjects.py`
  and `Measure_leakage_impact.py`. A marker following my own instructions would
  have hit a missing file twice. Rangpur is case-sensitive, so the capital M
  mattered too.
- **Opened the pull request.** Two marks for opening it correctly and one for
  the description, and it is a pass hurdle that zeroes all forty marks if
  missed.

### Something I should have seen on day one

The day one audit output lists the first few training patient IDs:

```
first few patient IDs: ['218391', '218393', '228875', '228879', '236920']
```

218391 and 218393 are two apart. 228875 and 228879 are four apart. That is what
a sequential scan-ID scheme looks like, and it was printed on my screen on the
first day. I did not notice until a validation accuracy looked wrong ten days
later.

The audit also concluded "the shipped split is clean", which is true by ID and
false by subject.

---

## Day 12. Sunday 25 October

### 2.5D failed, and my own day-one measurement predicted it

| | Single slice | Three slices | Change |
|---|---|---|---|
| Validation | 0.8134 | 0.7901 | −2.3 pts |
| Test | 0.7106 | 0.6863 | **−2.4 pts** |
| AD recall | 0.5821 | 0.4632 | **−11.9 pts** |
| Seconds per epoch | 14.0 | 33.7 | 2.41× |

`visualise_data.py` printed this on day one:

```
same patient, adjacent slices:  mean 0.9612
different patients, same class: mean 0.6769
```

Adjacent slices correlate at 0.9612. Stacking three of them does not add
depth information, it adds two near-copies of the slice already there. I paid
2.4 times the data loading cost for about 4% new signal per channel.

The AD recall collapse is the worse part. The model now misses 54% of
Alzheimer's cases. Three highly correlated channels pushed it further towards
its existing preference for calling things normal.

I should have checked that correlation figure before running the experiment. It
was in my own output from the first day and it predicts the result.

### The campaign closed

Five GFNet configurations across four axes: training length, regularisation,
token resolution and input representation. Validation moved 5.3 points. Test
moved 3.2 and did not track it. Not one run crossed 0.72.

I recorded the search as complete rather than abandoned, with a table of what
would be worth trying next and why each is out of scope. One row in it is the
honest one: pooling the two shipped folders and re-splitting by subject would
very likely clear 0.80, and it would stop being a measurement on a held-out
cohort. I would rather report 0.7161 on a genuine held-out set than 0.85 on a
set I built to be easier.

---

## Day 13. Monday 26 October

### Tutor feedback and the restructure

Three points: the report is too long to go through, a README on GitHub can link
to separate pages, and either hit 0.80 or show enough experimental evidence.

The README was 11,442 words. It is now **2,594**, with the problem, the
results, three key findings and the recommendation on the main page, and seven
linked pages under `docs/` holding the detail.

`full_report.md` concatenates everything back into one document for the
Turnitin PDF, so GitHub readers get a short navigable page and the marker gets
the whole report.

### Expected output for every command

My tutor also asked for the expected output after each command. The
specification asks for the same thing under "Example inputs, outputs, and
visual plots". Every command in the report now shows its real output, abridged,
with a line saying what to check.

Real output only. A marker who runs my code and sees something different from
what I documented has found a worse problem than a missing output block.

### Something I found while doing it

The README referenced `find_duplicate_subjects.py` and
`measure_leakage_impact.py`. The files were named `duplicate_subjects.py` and
`Measure_leakage_impact.py`. A marker following my own instructions would have
hit a missing file twice, and Rangpur is case-sensitive so the capital M
mattered too.

### Submitted

Built the Word version with bordered tables, shaded header rows, boxed code
blocks and all twelve figures. Wrote the error autopsy observations and the AI
disclosure myself. Exported and submitted to Turnitin.

---

## Day 14. Tuesday 27 October

### Three gaps between my implementation and how GFNet is actually trained

I went looking at the reference architecture and found the training recipe
mattered more than the architecture.

1. **Plain `Adam`, not `AdamW`.** In `Adam` the weight decay goes through the
   same adaptive division as the gradient, so `--weight-decay 1e-3` regularised
   far less than the number suggests. Every ViT and GFNet recipe uses decoupled
   decay.
2. **No warmup.** Cosine annealing started at full rate from epoch 0. This also
   explains my smoke test at 1e-3 collapsing into a single class.
3. **No stochastic depth.** The standard regulariser for this family, and what
   makes depth usable at all.

Fixed all three and set depth to 12, which matches GFNet-Ti.

### The results

| Run | Validation | Test | Best epoch |
|---|---|---|---|
| depth 8, Adam, no warmup | 0.8134 | 0.7106 | 54 of 80 |
| recipe, patch 16, 80 ep | 0.8453 | 0.7256 | **73 of 80** |
| recipe, patch 16, 150 ep | 0.8680 | 0.7294 | **116 of 150** |
| **recipe, patch 8, 80 ep** | 0.8689 | **0.7450** | **65 of 80** |

**GFNet now beats the baseline.** 0.7450 against 0.7421 at slice level, and
0.7578 against 0.7511 at patient level.

Every recipe run also peaked late, at 73, 116 and 65 rather than the 54 to 63
of every earlier run. The recipe did not only raise the endpoint, it changed
the trajectory, and none of these runs had converged.

### The finding: accuracy and calibration traded directly

| | Baseline | GFNet, old | GFNet, recipe |
|---|---|---|---|
| Test accuracy | 0.7421 | 0.7122 | **0.7450** |
| Calibration error | 0.1812 | **0.0934** | 0.2231 |
| Confidence when wrong | 0.8803 | 0.7499 | **0.9427** |
| Separation | 0.0579 | **0.0782** | 0.0332 |
| Fitted temperature | 1.20 | 1.11 | **4.59** |

A fitted temperature of 4.59. The old GFNet needed 1.11. All ten of the new
model's most confident mistakes sit at exactly 1.0000, where the old model's
worst was 0.9990.

**So the recipe bought accuracy and paid for it in calibration**, and my
report's central sentence, that the baseline is the better classifier and
GFNet the more trustworthy one, is now false in both halves.

The twist that makes it useful: temperature scaling cuts the new model's error
by 49%, from 0.2231 to 0.1127, against the baseline's 9%. **After scaling,
GFNet is both more accurate and better calibrated than the baseline.** That
turns calibration from an optional extra into the thing that makes the
recommendation work.

### Mistakes today

1. I committed a change to `predict.py` with a message describing it, and the
   commit did not contain the change. The file on my laptop had been written a
   moment after I staged. Two jobs failed on `unrecognized arguments: --depth`
   before I checked `git status` on Rangpur instead of trusting the commit
   message. Rule I am keeping: `git show --stat HEAD` after any commit I am
   about to depend on.
2. I had two versions of `run_recipe_figures.sh`, one without `--depth 12`. The
   older one existed on Rangpur as an untracked file, so `git pull` left it
   alone and the job kept failing with the same error.
3. I read a job that printed `=== done ===` with an empty error file as a
   failure, because it finished in 30 seconds and I expected it to take longer.

### What the loud failure was worth

`predict.py` refused to load a depth-12 checkpoint into a depth-8 model. Had it
silently filled the missing four blocks with random weights, it would have
produced an accuracy near 0.5, a plausible-looking figure set, and a number I
would have put in the report.

---

## Still open

- **The 150-epoch run at patch 8 with the recipe.** Queued. The 80-epoch
  version peaked at epoch 65 of 80, so it had not converged.
- **Test-time augmentation and label smoothing.** Not yet implemented. Label
  smoothing should also attack the 4.59 temperature directly.
- **The report rewrite.** My central conclusion has reversed and the submitted
  version says the opposite of what the numbers now show. This is the work that
  matters most, and the hard cutoff for experiments is Thursday midday so it
  gets done properly.
- **The error autopsy.** My observations cover the baseline and the depth-8
  GFNet. The recipe model fails on completely different patients.
- **The AI disclosure.** It says staff confirmed the filename IDs are patient
  IDs. That is what I was told, and my own experiment later disproved it. The
  sentence needs to acknowledge both.
- **Phase 2.** Respond to the code review on the pull request within about
  seven days of receiving it. Two marks.

## Things I should be able to explain out loud

- Why patient-level splitting matters, and what would go wrong without it
- Why the measured normalisation constants changed the pixel range the way they did
- What GFNet does instead of self-attention, and why the convolution theorem
  makes that equivalent to global mixing
- Why the rfft filter is 14 x 8 and not 14 x 14, and why patch 8 gives 420
  frequency components instead of 112
- Why temperature scaling must be fitted on validation and not on test
- Why accuracy does not change when the temperature changes
- Why patient-level accuracy is the clinically honest number
- Why a patient wrong on all 20 slices is worse than 20 scattered wrong slices
- Why I ran a brightness-only classifier, and what its 0.585 tells me
- Why a median ID gap of 2 against 47,026 proves the IDs are scan IDs
- Why the three slices in 2.5D must be merged before the transform, not after
- Why adjacent slices correlating at 0.9612 predicts that 2.5D would fail
- Why plain Adam regularises less than AdamW at the same weight decay number
- Why stochastic depth must drop a whole sample's branch and not scatter zeros
- Why a fitted temperature of 4.59 means the model is badly overconfident, and
  why one scalar fixes most of it
- Why a confidence of 1.0000 on a wrong answer is the strongest evidence in my
  report that the confidence score cannot be automated on
