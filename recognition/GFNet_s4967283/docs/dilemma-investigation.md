# Investigation of the open research dilemma

[&larr; Back to the main README](../README.md)

---

### Does the model learn anatomy, or brightness?

The intensity histograms show that AD tissue is darker than NC tissue on
average. That raised a question I had to answer before I trusted any result.
Does the model read anatomy, or does it read brightness?

![Pixel intensity by class](../figures/fig4_intensity_histograms.png)

![Mean image per class, AD against NC](../figures/fig3_class_averages.png)

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

![Reliability diagram](../figures/calibration_reliability.png)

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

![Confidence when right against when wrong](../figures/calibration_confidence.png)

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

![Reject option curves](../figures/reject_option_curves.png)

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
