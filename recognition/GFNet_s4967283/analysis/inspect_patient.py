"""
inspect_patient.py

Print every slice of one patient, for both models, side by side.

Why this exists
---------------
predict.py names the patients the models got most confidently wrong. That is
the start of the error autopsy, not the end. To write the autopsy I need to
know whether a failure covers the whole patient or only a few slices, and
whether both models fail on the same slices.

This script answers that from the saved predictions. No GPU and no model
loading, so it runs on the login node.

Usage:
    python inspect_patient.py --patient 985197
    python inspect_patient.py --patient 985197 1431781 1274602 1265863
    python inspect_patient.py --worst 10
"""

import argparse
import os

import numpy as np


def load_predictions(path):
    """Read one predictions file written by predict.py."""
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "probabilities": data["probabilities"],
        "labels": data["labels"],
        "patients": np.array([str(p) for p in data["patient_ids"]]),
    }


def patient_report(name, store, patient_id):
    """Print one line per slice for one patient under one model."""
    mask = store["patients"] == patient_id
    if not mask.any():
        print("  %-9s patient not found in this split" % name)
        return None

    # Column 1 holds P(AD), because CLASS_TO_LABEL maps AD to 1.
    probability_ad = store["probabilities"][mask][:, 1]
    true_label = int(store["labels"][mask][0])
    true_class = "AD" if true_label == 1 else "NC"

    predictions = (probability_ad >= 0.5).astype(int)
    correct_count = int((predictions == true_label).sum())
    slice_count = len(probability_ad)

    print("  %-9s true class %s   correct on %d of %d slices"
          % (name, true_class, correct_count, slice_count))
    print("             mean P(AD) %.4f   min %.4f   max %.4f"
          % (probability_ad.mean(), probability_ad.min(), probability_ad.max()))

    # The per-slice values. A patient wrong on every slice looks very different
    # from a patient the model is merely unsure about.
    values = "  ".join("%.3f" % value for value in probability_ad)
    print("             P(AD) per slice: %s" % values)

    return {
        "true_label": true_label,
        "correct_count": correct_count,
        "slice_count": slice_count,
        "mean_probability_ad": float(probability_ad.mean()),
    }


def find_worst_patients(store, how_many):
    """
    Return the patients this model gets wrong on every slice, worst first.

    "Worst" means furthest from the truth on average. A patient with AD whose
    mean P(AD) is 0.02 is a more serious failure than one at 0.45.
    """
    results = []
    for patient_id in sorted(set(store["patients"].tolist())):
        mask = store["patients"] == patient_id
        probability_ad = store["probabilities"][mask][:, 1]
        true_label = int(store["labels"][mask][0])
        predictions = (probability_ad >= 0.5).astype(int)

        if (predictions == true_label).sum() != 0:
            continue  # Not a total failure, so skip it.

        # Distance from the correct answer, averaged over the 20 slices.
        if true_label == 1:
            wrongness = float((1.0 - probability_ad).mean())
        else:
            wrongness = float(probability_ad.mean())

        results.append((wrongness, patient_id, true_label))

    results.sort(reverse=True)
    return results[:how_many]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inspect the per-slice predictions for chosen patients."
    )
    parser.add_argument("--patient", nargs="*", default=[],
                        help="One or more patient IDs to print in full.")
    parser.add_argument("--worst", type=int, default=0,
                        help="Also list the N worst total failures per model.")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    stores = {}
    for name in ("baseline", "gfnet"):
        path = os.path.join(arguments.figures_dir,
                            "test_predictions_%s.npz" % name)
        store = load_predictions(path)
        if store is None:
            print("Missing %s. Run predict.py first." % path)
            return
        stores[name] = store

    for patient_id in arguments.patient:
        print()
        print("PATIENT %s" % patient_id)
        print("-" * 72)
        for name in ("baseline", "gfnet"):
            patient_report(name, stores[name], patient_id)

    if arguments.worst > 0:
        for name in ("baseline", "gfnet"):
            print()
            print("WORST TOTAL FAILURES, %s" % name.upper())
            print("  (wrong on all 20 slices, furthest from the truth first)")
            print("-" * 72)
            for wrongness, patient_id, true_label in find_worst_patients(
                    stores[name], arguments.worst):
                true_class = "AD" if true_label == 1 else "NC"
                print("  patient %-9s true %s   mean distance from truth %.4f"
                      % (patient_id, true_class, wrongness))

        # The overlap between the two lists is the most interesting part. A
        # patient both models fail on is a hard scan. A patient only one model
        # fails on is a weakness of that model.
        baseline_failures = {
            patient_id for _, patient_id, _
            in find_worst_patients(stores["baseline"], 10000)
        }
        gfnet_failures = {
            patient_id for _, patient_id, _
            in find_worst_patients(stores["gfnet"], 10000)
        }
        shared = baseline_failures & gfnet_failures

        print()
        print("OVERLAP OF TOTAL FAILURES")
        print("-" * 72)
        print("  baseline only: %d patients" % len(baseline_failures - gfnet_failures))
        print("  gfnet only:    %d patients" % len(gfnet_failures - baseline_failures))
        print("  both models:   %d patients" % len(shared))
        if shared:
            print("  shared IDs: %s" % ", ".join(sorted(shared)))
        print()
        print("  A patient in the shared list is a hard scan, not a weakness of")
        print("  one architecture. A patient in one list only is the opposite.")


if __name__ == "__main__":
    main()
