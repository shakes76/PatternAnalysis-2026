"""
try_ensemble.py

Combine the baseline and GFNet predictions and measure what that buys.

Why this is worth trying
------------------------
inspect_patient.py showed that only 20 of the 450 test patients are wrong on
all 20 slices for BOTH models. The baseline fails completely on 20 patients
that GFNet partly reads, and GFNet on 8 that the baseline partly reads.

Two models that fail on different people can cover for each other. This script
measures whether they actually do, and by how much.

It reads the saved .npz files, so there is no GPU and no retraining. If the
answer is good it is free accuracy. If the answer is bad, that is a finding
too, and it goes in the report either way.

Usage:
    python try_ensemble.py
    python try_ensemble.py --figures-dir figures
"""

import argparse
import os

import numpy as np


def load_predictions(path):
    """Read one predictions file written by predict.py."""
    data = np.load(path, allow_pickle=True)
    return {
        "probabilities": data["probabilities"],
        "labels": data["labels"],
        "patients": np.array([str(p) for p in data["patient_ids"]]),
    }


def slice_accuracy(probability_ad, labels):
    """Fraction of slices where the argmax matches the true label."""
    predictions = (probability_ad >= 0.5).astype(int)
    return float((predictions == labels).mean())


def patient_accuracy(probability_ad, labels, patients):
    """
    One decision per patient, from the mean of that patient's slices.

    This is the clinically meaningful number. A clinician classifies a person,
    not an image.
    """
    unique_patients = sorted(set(patients.tolist()))

    correct = 0
    for patient in unique_patients:
        mask = patients == patient
        mean_probability = probability_ad[mask].mean()
        prediction = 1 if mean_probability >= 0.5 else 0
        if prediction == int(labels[mask][0]):
            correct += 1

    return correct / len(unique_patients), len(unique_patients)


def count_total_failures(probability_ad, labels, patients):
    """How many patients are wrong on every single one of their slices."""
    total = 0
    for patient in sorted(set(patients.tolist())):
        mask = patients == patient
        predictions = (probability_ad[mask] >= 0.5).astype(int)
        true_label = int(labels[mask][0])
        if (predictions == true_label).sum() == 0:
            total += 1
    return total


def best_threshold_for_patients(probability_ad, labels, patients):
    """
    Find the patient-level decision threshold with the highest accuracy.

    IMPORTANT: this searches on the split it is given. Running it on test and
    then reporting that accuracy is tuning on the test set. Use it on
    validation, then apply the chosen value to test.
    """
    unique_patients = sorted(set(patients.tolist()))

    patient_means = []
    patient_labels = []
    for patient in unique_patients:
        mask = patients == patient
        patient_means.append(float(probability_ad[mask].mean()))
        patient_labels.append(int(labels[mask][0]))

    patient_means = np.array(patient_means)
    patient_labels = np.array(patient_labels)

    best_threshold = 0.5
    best_accuracy = 0.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        predictions = (patient_means >= threshold).astype(int)
        accuracy = float((predictions == patient_labels).mean())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure what an ensemble of the two models buys."
    )
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    # ---- Load test predictions for both models ----
    stores = {}
    for name in ("baseline", "gfnet"):
        path = os.path.join(arguments.figures_dir,
                            "test_predictions_%s.npz" % name)
        if not os.path.exists(path):
            print("Missing %s. Run predict.py first." % path)
            return
        stores[name] = load_predictions(path)

    labels = stores["baseline"]["labels"]
    patients = stores["baseline"]["patients"]

    # The two files must describe the same slices in the same order, or
    # averaging them is meaningless. Check rather than assume.
    assert np.array_equal(labels, stores["gfnet"]["labels"]), \
        "The two prediction files disagree on the labels."
    assert np.array_equal(patients, stores["gfnet"]["patients"]), \
        "The two prediction files disagree on the patient order."

    # Column 1 holds P(AD), because CLASS_TO_LABEL maps AD to 1.
    baseline_ad = stores["baseline"]["probabilities"][:, 1]
    gfnet_ad = stores["gfnet"]["probabilities"][:, 1]

    print("=" * 72)
    print("THE TWO MODELS ON THEIR OWN")
    print("=" * 72)
    for name, probability_ad in (("baseline", baseline_ad),
                                 ("gfnet", gfnet_ad)):
        slice_score = slice_accuracy(probability_ad, labels)
        patient_score, patient_count = patient_accuracy(
            probability_ad, labels, patients)
        failures = count_total_failures(probability_ad, labels, patients)
        print("  %-9s slice %.4f   patient %.4f   total failures %d"
              % (name, slice_score, patient_score, failures))

    # ---- Weighted averages ----
    # Weight 0.0 is pure GFNet and weight 1.0 is pure baseline. The sweep shows
    # whether the best mix is an even split or leans towards one model.
    print()
    print("=" * 72)
    print("WEIGHTED AVERAGE OF THE TWO PROBABILITIES")
    print("=" * 72)
    print("  weight is the share given to the baseline")
    print()
    print("  %-8s %-10s %-10s %-10s" % ("weight", "slice", "patient", "failures"))
    print("  " + "-" * 44)

    best_patient_score = 0.0
    best_weight = None

    for weight in np.arange(0.0, 1.01, 0.1):
        combined = weight * baseline_ad + (1.0 - weight) * gfnet_ad
        slice_score = slice_accuracy(combined, labels)
        patient_score, _ = patient_accuracy(combined, labels, patients)
        failures = count_total_failures(combined, labels, patients)

        marker = ""
        if patient_score > best_patient_score:
            best_patient_score = patient_score
            best_weight = float(weight)
            marker = "  <-- best so far"

        print("  %-8.1f %-10.4f %-10.4f %-10d%s"
              % (weight, slice_score, patient_score, failures, marker))

    # ---- The honest caveat ----
    print()
    print("=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  Best patient accuracy %.4f at weight %.1f." % (best_patient_score, best_weight))
    print()
    print("  WARNING. The weight above was chosen by looking at the TEST split.")
    print("  Reporting that number as a result would be tuning on test, which is")
    print("  the exact mistake the patient-level splitting exists to prevent.")
    print()
    print("  To use an ensemble honestly:")
    print("    1. Run this on the VALIDATION predictions to choose the weight.")
    print("    2. Apply that fixed weight to test, once, and report the result.")
    print()
    print("  The even split at weight 0.5 needs no fitting at all, so it is")
    print("  honest to report directly. Prefer it unless the sweep shows a large")
    print("  and consistent advantage somewhere else.")

    # ---- The even split, reported plainly ----
    even = 0.5 * baseline_ad + 0.5 * gfnet_ad
    even_slice = slice_accuracy(even, labels)
    even_patient, patient_count = patient_accuracy(even, labels, patients)
    even_failures = count_total_failures(even, labels, patients)

    baseline_patient, _ = patient_accuracy(baseline_ad, labels, patients)

    print()
    print("=" * 72)
    print("THE EVEN SPLIT, WHICH NEEDS NO FITTING")
    print("=" * 72)
    print("  slice accuracy:   %.4f" % even_slice)
    print("  patient accuracy: %.4f  (%d patients)" % (even_patient, patient_count))
    print("  patients wrong on all 20 slices: %d" % even_failures)
    print()
    print("  change against the better single model at patient level: %+.1f points"
          % ((even_patient - baseline_patient) * 100))


if __name__ == "__main__":
    main()
