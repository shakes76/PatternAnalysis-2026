# The filename ID is a scan ID, not a person ID

[&larr; Back to the main README](../README.md)

---

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
