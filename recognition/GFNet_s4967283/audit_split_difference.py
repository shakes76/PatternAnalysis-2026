"""
audit_split_difference.py

Compare the shipped train folder against the shipped test folder, feature by
feature, and report what differs.

The question this answers
-------------------------
Both models lose heavily between validation and test:

    BaselineCNN   validation 0.9028   test 0.7421   drop 16 points
    GFNet         validation 0.7907   test 0.6837   drop 11 points

Validation and test are both patient-disjoint. Validation comes from the
shipped train/ folder and test comes from the shipped test/ folder. A 16-point
drop between two clean splits of one dataset is too large to be ordinary.

Either the test folder is genuinely harder, or something systematic separates
the two folders. If it is the second, finding it is worth more than any amount
of further tuning.

I already checked mean intensity and found no difference. That rules out one
explanation, not all of them. This script checks the rest:

    - which slice numbers appear in each folder
    - the range of patient IDs
    - how much of the frame the head fills
    - where the head sits in the frame
    - contrast within the tissue

For each measure it reports Cohen's d, which is the difference between the two
means in units of their pooled standard deviation. A d below 0.2 is negligible.
Around 0.5 is moderate. Above 0.8 is large and worth investigating.

Usage:
    python audit_split_difference.py
    python audit_split_difference.py --patients-per-class 150
"""

import argparse
import os
import random
import re

import numpy as np
from PIL import Image

from dataset import scan_split_folder


# Pixels at or below this count as background. The histograms show a large
# spike at zero, which is the black space around the head.
BACKGROUND_CUTOFF = 10

# Filenames look like 1003730_100.jpeg. The number after the underscore is the
# slice position within that patient's 3D scan.
FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)\.(jpe?g|png)$", re.IGNORECASE)


def parse_filename(image_path):
    """Return (patient_id, slice_index) from a filename, or (None, None)."""
    match = FILENAME_PATTERN.match(os.path.basename(image_path))
    if match is None:
        return None, None
    return match.group(1), int(match.group(2))


def measure_image(image_path):
    """
    Measure one slice several ways.

    Each measure is chosen because a difference in it between the two folders
    would plausibly hurt a model trained on one and tested on the other.
    """
    with Image.open(image_path) as raw_image:
        pixels = np.array(raw_image.convert("L"), dtype=np.float64)

    tissue_mask = pixels > BACKGROUND_CUTOFF
    tissue = pixels[tissue_mask]

    if tissue.size == 0:
        return None

    # Where the tissue sits in the frame, and how much of it there is.
    rows, columns = np.nonzero(tissue_mask)

    return {
        # Brightness. Already checked, included so the comparison is complete.
        "tissue_mean": float(tissue.mean()),
        # Contrast inside the head. A flatter scan gives the model less to use.
        "tissue_std": float(tissue.std()),
        # How much of the 256x240 frame the head fills. A difference here means
        # the two folders were resized or cropped differently.
        "head_area_fraction": float(tissue_mask.mean()),
        # Where the head sits. A systematic offset would matter, because the
        # model has no translation invariance beyond what augmentation gives it.
        "centroid_row": float(rows.mean()),
        "centroid_column": float(columns.mean()),
        # The size of the head's bounding box, in pixels.
        "bbox_height": float(rows.max() - rows.min() + 1),
        "bbox_width": float(columns.max() - columns.min() + 1),
        # The brightest tissue. Scanner or normalisation differences show here.
        "tissue_p99": float(np.percentile(tissue, 99)),
    }


def sample_patients(samples, patients_per_class, seed=0):
    """
    Choose a fixed number of patients per class and keep all of their slices.

    I sample patients rather than slices so the comparison matches how the
    models are evaluated.
    """
    by_patient = {}
    for image_path, label, patient_id in samples:
        if patient_id not in by_patient:
            by_patient[patient_id] = {"label": label, "paths": []}
        by_patient[patient_id]["paths"].append(image_path)

    chooser = random.Random(seed)
    patient_ids = sorted(by_patient.keys())
    chooser.shuffle(patient_ids)

    chosen = {0: [], 1: []}
    for patient_id in patient_ids:
        label = by_patient[patient_id]["label"]
        if len(chosen[label]) < patients_per_class:
            chosen[label].append(patient_id)

    selected = []
    for label in (0, 1):
        for patient_id in chosen[label]:
            for image_path in sorted(by_patient[patient_id]["paths"]):
                selected.append((image_path, label, patient_id))

    return selected


def collect(split_name, data_root, patients_per_class, seed):
    """Measure a sample of one split and return every measurement as arrays."""
    print("Reading the %s folder..." % split_name)
    samples = scan_split_folder(data_root, split_name)
    selected = sample_patients(samples, patients_per_class, seed)

    measurements = {}
    patient_ids = []
    slice_indices = []
    labels = []

    for image_path, label, patient_id in selected:
        result = measure_image(image_path)
        if result is None:
            continue

        for key, value in result.items():
            measurements.setdefault(key, []).append(value)

        _, slice_index = parse_filename(image_path)
        patient_ids.append(patient_id)
        slice_indices.append(slice_index)
        labels.append(label)

    for key in measurements:
        measurements[key] = np.array(measurements[key])

    print("  %d slices from %d patients"
          % (len(labels), len(set(patient_ids))))

    return {
        "measurements": measurements,
        "patients": patient_ids,
        "slices": np.array([s for s in slice_indices if s is not None]),
        "labels": np.array(labels),
    }


def cohens_d(first, second):
    """
    The difference between two means, in units of their pooled spread.

    This is better than comparing raw means, because it says whether the
    difference is large relative to the natural variation in the data.
    """
    n1, n2 = len(first), len(second)
    if n1 < 2 or n2 < 2:
        return 0.0

    pooled_variance = (((n1 - 1) * first.var(ddof=1) +
                        (n2 - 1) * second.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_variance <= 0:
        return 0.0

    return float((first.mean() - second.mean()) / np.sqrt(pooled_variance))


def describe_effect(d):
    """Put a plain word to an effect size."""
    size = abs(d)
    if size < 0.2:
        return "negligible"
    if size < 0.5:
        return "small"
    if size < 0.8:
        return "moderate"
    return "LARGE"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Find what differs between the shipped train and test folders."
    )
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--patients-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    print("Comparing the shipped train and test folders.")
    print("Sampling %d patients per class from each." % arguments.patients_per_class)
    print()

    train = collect("train", arguments.data_root,
                    arguments.patients_per_class, arguments.seed)
    test = collect("test", arguments.data_root,
                   arguments.patients_per_class, arguments.seed)

    # ---- Slice numbers ----
    # If the two folders cover different slice positions, the models are being
    # tested on a different part of the brain than they trained on. That alone
    # would explain the gap.
    print()
    print("=" * 72)
    print("SLICE NUMBERS")
    print("=" * 72)
    for name, store in (("train", train), ("test", test)):
        slices = store["slices"]
        unique = sorted(set(slices.tolist()))
        print("  %-6s min %d   max %d   %d distinct values"
              % (name, slices.min(), slices.max(), len(unique)))
        preview = ", ".join(str(v) for v in unique[:12])
        if len(unique) > 12:
            preview += ", ..."
        print("         values: %s" % preview)

    train_set = set(train["slices"].tolist())
    test_set = set(test["slices"].tolist())
    shared = train_set & test_set

    print()
    print("  shared slice numbers: %d" % len(shared))
    print("  only in train:        %d" % len(train_set - test_set))
    print("  only in test:         %d" % len(test_set - train_set))
    if not shared:
        print()
        print("  WARNING. The two folders share NO slice numbers. The models are")
        print("  tested on a different part of the brain than they trained on.")

    # ---- Patient IDs ----
    print()
    print("=" * 72)
    print("PATIENT IDs")
    print("=" * 72)
    for name, store in (("train", train), ("test", test)):
        numeric = sorted(int(p) for p in set(store["patients"]))
        print("  %-6s min %d   max %d   count %d"
              % (name, numeric[0], numeric[-1], len(numeric)))

    train_patients = {int(p) for p in train["patients"]}
    test_patients = {int(p) for p in test["patients"]}
    overlap = train_patients & test_patients
    print()
    print("  patients in BOTH folders: %d" % len(overlap))
    if overlap:
        print("  WARNING. The shipped folders share patients. This is leakage")
        print("  in the dataset as distributed, not in my splitting code.")
        print("  shared IDs: %s" % sorted(overlap)[:20])

    # ---- Image measures ----
    print()
    print("=" * 72)
    print("IMAGE MEASURES, TRAIN AGAINST TEST")
    print("=" * 72)
    print("  %-22s %-12s %-12s %-9s %s"
          % ("measure", "train mean", "test mean", "Cohen d", "size"))
    print("  " + "-" * 68)

    large_differences = []

    for key in sorted(train["measurements"].keys()):
        train_values = train["measurements"][key]
        test_values = test["measurements"][key]
        d = cohens_d(train_values, test_values)
        size = describe_effect(d)

        print("  %-22s %-12.3f %-12.3f %-9.3f %s"
              % (key, train_values.mean(), test_values.mean(), d, size))

        if abs(d) >= 0.5:
            large_differences.append((key, d, size))

    # ---- The verdict ----
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if not large_differences and shared == train_set == test_set:
        print("  No measure differs by more than a small effect, and both folders")
        print("  cover the same slice numbers.")
        print()
        print("  So the validation-to-test gap is NOT explained by a measurable")
        print("  difference between the folders. The remaining explanations are")
        print("  that the test cohort is genuinely harder, or that the difference")
        print("  lies in something these measures do not capture, such as scanner")
        print("  model or acquisition site.")
    else:
        print("  Differences found:")
        for key, d, size in large_differences:
            print("    %-22s Cohen d %+.3f  (%s)" % (key, d, size))
        if len(shared) < len(train_set | test_set):
            print("    slice numbers differ between the folders")
        print()
        print("  Any LARGE entry is a candidate explanation for the")
        print("  validation-to-test gap, and is worth a targeted fix.")


if __name__ == "__main__":
    main()
