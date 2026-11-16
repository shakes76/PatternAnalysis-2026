# The data, the splits and the preprocessing

[&larr; Back to the main README](../README.md)

---

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

![Example slices from different patients](../figures/fig1_class_comparison.png)

### The splits, and why they are built this way

Each patient gives 20 slices cut from one 3D scan, so those 20 images look very
similar to each other.

![All 20 slices from one patient](../figures/fig2_one_patient_all_slices.png)

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
