"""
brightness_baseline.py

A deliberately stupid classifier, built to test whether my real models are
learning anything.

The question this answers
------------------------
The intensity histograms show that AD tissue is darker than NC tissue on
average. That is a real difference in the pixels. But it raises a problem: a
model could score well by measuring average brightness and nothing else. That
would be a dataset artifact, not pattern recognition.

So I measure it. This script classifies a slice using ONE number: the mean
intensity of its tissue pixels. It picks the best threshold on the training
split and applies that threshold to the test split.

How to read the result
----------------------
If this scores near chance (0.50), the brightness difference is weak and my
CNN must be using shape. Good.

If this scores near my CNN (0.72), then my CNN is close to a brightness meter
and the result is much less impressive than it looks. That is a finding I must
report honestly.

Either answer is useful. The second answer is more useful.

Usage:
    python brightness_baseline.py --data-root /home/groups/comp3710/ADNI/AD_NC
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

from dataset import scan_split_folder, LABEL_TO_CLASS


# Pixels at or below this value count as background, not tissue. The histograms
# show a huge spike at zero, which is the black space around the head. Including
# it would make every image look similar and hide the tissue difference.
BACKGROUND_CUTOFF = 10


def mean_tissue_intensity(image_path):
    """
    Return the mean pixel value of the tissue in one slice.

    I ignore the background because the amount of black space depends on head
    size and position, not on the disease. Including it would measure how much
    of the frame the head fills.
    """
    with Image.open(image_path) as raw_image:
        pixels = np.array(raw_image.convert("L"), dtype=np.float64)

    tissue = pixels[pixels > BACKGROUND_CUTOFF]
    if tissue.size == 0:
        return 0.0
    return float(tissue.mean())


def collect_features(samples, max_patients_per_class, seed=0):
    """
    Measure every slice of a sample of patients.

    I sample patients rather than slices, and I keep all 20 slices of each
    chosen patient. That matches how the real models are evaluated, so the
    comparison is fair.
    """
    # Group by patient first, so the sampling happens at patient level.
    by_patient = {}
    for image_path, label, patient_id in samples:
        if patient_id not in by_patient:
            by_patient[patient_id] = {"label": label, "paths": []}
        by_patient[patient_id]["paths"].append(image_path)

    chooser = random.Random(seed)
    chosen_by_label = {0: [], 1: []}

    patient_ids = sorted(by_patient.keys())
    chooser.shuffle(patient_ids)

    for patient_id in patient_ids:
        label = by_patient[patient_id]["label"]
        if len(chosen_by_label[label]) < max_patients_per_class:
            chosen_by_label[label].append(patient_id)

    intensities = []
    labels = []
    patients = []

    for label in (0, 1):
        for patient_id in chosen_by_label[label]:
            for image_path in sorted(by_patient[patient_id]["paths"]):
                intensities.append(mean_tissue_intensity(image_path))
                labels.append(label)
                patients.append(patient_id)

    return np.array(intensities), np.array(labels), np.array(patients)


def find_best_threshold(intensities, labels):
    """
    Search for the intensity threshold that separates the classes best.

    The rule is: if mean intensity is below the threshold, predict AD. The
    histograms say AD is darker, so that is the right direction.

    I try every midpoint between sorted unique values. That is exhaustive for
    one feature, so there is no optimiser to get wrong.
    """
    candidates = np.unique(intensities)
    midpoints = (candidates[:-1] + candidates[1:]) / 2.0

    best_threshold = None
    best_accuracy = 0.0

    for threshold in midpoints:
        predictions = (intensities < threshold).astype(int)  # 1 means AD
        accuracy = float((predictions == labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def evaluate(intensities, labels, threshold):
    """Apply a fixed threshold and report accuracy plus per-class recall."""
    predictions = (intensities < threshold).astype(int)

    accuracy = float((predictions == labels).mean())

    results = {"accuracy": accuracy}
    for class_index in (0, 1):
        in_class = labels == class_index
        recall = float((predictions[in_class] == class_index).mean())
        results["%s_recall" % LABEL_TO_CLASS[class_index]] = recall

    return results, predictions


def evaluate_per_patient(intensities, labels, patients, threshold):
    """
    Same rule, but one decision per patient from the mean of their slices.

    This matches the patient-level numbers from the calibration analysis, so
    the two can be compared directly.
    """
    unique_patients = sorted(set(patients.tolist()))

    patient_predictions = []
    patient_labels = []

    for patient in unique_patients:
        mask = patients == patient
        patient_mean = intensities[mask].mean()
        patient_predictions.append(1 if patient_mean < threshold else 0)
        patient_labels.append(labels[mask][0])

    patient_predictions = np.array(patient_predictions)
    patient_labels = np.array(patient_labels)

    return float((patient_predictions == patient_labels).mean()), len(unique_patients)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Classify AD vs NC using mean tissue brightness alone."
    )
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--patients-per-class", type=int, default=120,
                        help="How many patients to sample from each class, per split.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    print("Measuring mean tissue brightness. Background cutoff is %d." % BACKGROUND_CUTOFF)
    print("Sampling %d patients per class from each split." % arguments.patients_per_class)
    print()

    print("Reading the training split...")
    train_samples = scan_split_folder(arguments.data_root, "train")
    train_intensity, train_labels, train_patients = collect_features(
        train_samples, arguments.patients_per_class, arguments.seed
    )
    print("  %d slices from %d patients" % (len(train_labels), len(set(train_patients.tolist()))))

    print("Reading the test split...")
    test_samples = scan_split_folder(arguments.data_root, "test")
    test_intensity, test_labels, test_patients = collect_features(
        test_samples, arguments.patients_per_class, arguments.seed
    )
    print("  %d slices from %d patients" % (len(test_labels), len(set(test_patients.tolist()))))

    # Mean brightness per class, to confirm the histograms numerically.
    print()
    print("MEAN TISSUE BRIGHTNESS")
    for split_name, intensity, labels in (
        ("train", train_intensity, train_labels),
        ("test", test_intensity, test_labels),
    ):
        ad_mean = intensity[labels == 1].mean()
        nc_mean = intensity[labels == 0].mean()
        print("  %-6s AD %.2f   NC %.2f   difference %.2f"
              % (split_name, ad_mean, nc_mean, nc_mean - ad_mean))

    # Fit on train only. Fitting on test would be cheating in exactly the way
    # the rest of this project avoids.
    threshold, train_accuracy = find_best_threshold(train_intensity, train_labels)

    print()
    print("THRESHOLD FITTED ON THE TRAINING SPLIT")
    print("  threshold: %.2f  (predict AD when mean tissue brightness is below this)" % threshold)
    print("  training accuracy at that threshold: %.4f" % train_accuracy)

    test_results, _ = evaluate(test_intensity, test_labels, threshold)
    patient_accuracy, patient_count = evaluate_per_patient(
        test_intensity, test_labels, test_patients, threshold
    )

    print()
    print("APPLIED TO THE TEST SPLIT")
    print("  slice-level accuracy:   %.4f" % test_results["accuracy"])
    print("  patient-level accuracy: %.4f  (%d patients)" % (patient_accuracy, patient_count))
    print("  AD recall: %.4f" % test_results["AD_recall"])
    print("  NC recall: %.4f" % test_results["NC_recall"])

    print()
    print("HOW TO READ THIS")
    print("  My BaselineCNN scores 0.7241 slice-level on the full test set.")
    print("  My GFNet scores 0.6890.")
    print()
    if test_results["accuracy"] >= 0.65:
        print("  This brightness rule is close to those numbers. That means a large")
        print("  part of my models' accuracy may come from average brightness and")
        print("  not from anatomy. I must report this.")
    else:
        print("  This brightness rule is well below those numbers. That means my")
        print("  models use more than average brightness. Good.")


if __name__ == "__main__":
    main()
