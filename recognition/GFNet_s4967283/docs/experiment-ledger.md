# Experiments towards the 0.80 minimum

[&larr; Back to the main README](../README.md)

---

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
