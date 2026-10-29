"""
measure_leakage_impact.py

Measure how much the patient IDs leak, and what that costs in accuracy.

What came before this
---------------------
find_duplicate_subjects.py matched every validation and test patient to its
closest training patient. The closest pairs came back at correlation 1.0000
between IDs that differ by ONE:

    validation 1037251  matches training 1037250   correlation 1.0000
    validation 1029553  matches training 1029554   correlation 1.0000
    test       1344400  matches training 1344399   correlation 1.0000

Two strangers' brains do not correlate at 1.0000. Consecutive IDs are the same
person, so the filename ID identifies a SCAN and not a PERSON.

That breaks the central methodological claim of this project. My splitting code
separates patients by ID, and check_no_patient_overlap passes, while the same
person sits in two splits under two IDs.

What this script does
---------------------
1. Counts how many validation and test patients have a near-duplicate in the
   training set, at several correlation thresholds.
2. Reports the ID difference for those pairs. If duplicates sit at small ID
   differences and non-duplicates do not, the IDs are sequential per scan and
   the diagnosis is confirmed.
3. Recomputes validation and test accuracy with the leaked patients REMOVED,
   using the saved predictions. That is the honest number, and the difference
   between it and the reported number is the size of the problem.

Usage:
    python measure_leakage_impact.py
    python measure_leakage_impact.py --threshold 0.995
"""

import argparse
import os

import numpy as np
from PIL import Image

from dataset import build_splits


DEFAULT_THUMBNAIL = 48

# Correlations at or above these count as a probable duplicate. Several are
# reported because the right cut-off is not obvious in advance, and a finding
# that only holds at one threshold is not a finding.
THRESHOLDS = (0.9999, 0.999, 0.995, 0.99, 0.98)


def patient_thumbnail(image_paths, thumbnail_size):
    """Reduce one patient's slices to a single centred, unit-length vector."""
    accumulator = None
    count = 0

    for image_path in sorted(image_paths):
        with Image.open(image_path) as raw_image:
            small = raw_image.convert("L").resize(
                (thumbnail_size, thumbnail_size), Image.BILINEAR)
        pixels = np.asarray(small, dtype=np.float32)
        accumulator = pixels if accumulator is None else accumulator + pixels
        count += 1

    if count == 0:
        return None

    vector = (accumulator / count).ravel()
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return None if norm == 0 else vector / norm


def build_thumbnails(samples, thumbnail_size):
    """Return {patient_id: unit vector} for every patient in these samples."""
    by_patient = {}
    for image_path, _label, patient_id in samples:
        by_patient.setdefault(patient_id, []).append(image_path)

    thumbnails = {}
    for patient_id, paths in by_patient.items():
        vector = patient_thumbnail(paths, thumbnail_size)
        if vector is not None:
            thumbnails[patient_id] = vector

    return thumbnails


def best_matches(query_thumbnails, reference_thumbnails):
    """For each query patient, its best correlation and matching reference ID."""
    reference_ids = sorted(reference_thumbnails.keys())
    reference_matrix = np.stack([reference_thumbnails[p] for p in reference_ids])

    matches = {}
    for query_id, vector in query_thumbnails.items():
        correlations = reference_matrix @ vector
        best = int(np.argmax(correlations))
        matches[query_id] = {
            "match": reference_ids[best],
            "correlation": float(correlations[best]),
        }

    return matches


def id_difference(first, second):
    """Absolute difference between two numeric patient IDs."""
    try:
        return abs(int(first) - int(second))
    except ValueError:
        return None


def accuracy_from_predictions(path, excluded_patients=frozenset()):
    """
    Patient-level accuracy from a saved predictions file.

    Patients in excluded_patients are left out, which is how the honest
    number is produced: the leaked patients are removed and the rest scored.
    """
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    probabilities = data["probabilities"]
    labels = data["labels"]
    patients = np.array([str(p) for p in data["patient_ids"]])

    unique_patients = sorted(set(patients.tolist()))

    kept_correct = 0
    kept_total = 0
    removed_correct = 0
    removed_total = 0

    for patient in unique_patients:
        mask = patients == patient
        mean_probability_ad = probabilities[mask][:, 1].mean()
        prediction = 1 if mean_probability_ad >= 0.5 else 0
        is_correct = prediction == int(labels[mask][0])

        if patient in excluded_patients:
            removed_total += 1
            removed_correct += int(is_correct)
        else:
            kept_total += 1
            kept_correct += int(is_correct)

    return {
        "kept_accuracy": kept_correct / kept_total if kept_total else 0.0,
        "kept_total": kept_total,
        "removed_accuracy": removed_correct / removed_total if removed_total else None,
        "removed_total": removed_total,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure the size of the patient ID leakage."
    )
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI/AD_NC")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--thumbnail-size", type=int, default=DEFAULT_THUMBNAIL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.995,
                        help="Correlation at or above which a patient counts "
                             "as leaked, for the accuracy recomputation.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    train_samples, validation_samples, test_samples = build_splits(
        adni_root=arguments.data_root,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
    )

    print("Building thumbnails at %d x %d. This reads every image once."
          % (arguments.thumbnail_size, arguments.thumbnail_size))
    training = build_thumbnails(train_samples, arguments.thumbnail_size)
    validation = build_thumbnails(validation_samples, arguments.thumbnail_size)
    test = build_thumbnails(test_samples, arguments.thumbnail_size)
    print("  %d training, %d validation, %d test"
          % (len(training), len(validation), len(test)))

    validation_matches = best_matches(validation, training)
    test_matches = best_matches(test, training)

    # ---- 1. How many look like duplicates, at each threshold ----
    print()
    print("=" * 76)
    print("PATIENTS WITH A NEAR-DUPLICATE IN THE TRAINING SET")
    print("=" * 76)
    print("  %-12s %-22s %-22s" % ("threshold", "validation", "test (control)"))
    print("  " + "-" * 60)

    for threshold in THRESHOLDS:
        validation_hits = [p for p, m in validation_matches.items()
                           if m["correlation"] >= threshold]
        test_hits = [p for p, m in test_matches.items()
                     if m["correlation"] >= threshold]

        print("  %-12.4f %-4d of %-4d (%5.1f%%)   %-4d of %-4d (%5.1f%%)"
              % (threshold,
                 len(validation_hits), len(validation),
                 100.0 * len(validation_hits) / len(validation),
                 len(test_hits), len(test),
                 100.0 * len(test_hits) / len(test)))

    # ---- 2. The smoking gun: are duplicates at small ID differences? ----
    # If the IDs were allocated per person, a duplicate pair would have no
    # reason to sit at a small ID difference. If they are allocated per scan in
    # sequence, one person's scans land next to each other.
    print()
    print("=" * 76)
    print("ID DIFFERENCE FOR HIGH-CORRELATION PAIRS")
    print("=" * 76)

    for name, matches in (("validation", validation_matches),
                          ("test", test_matches)):
        duplicates = []
        others = []
        for query_id, record in matches.items():
            difference = id_difference(query_id, record["match"])
            if difference is None:
                continue
            if record["correlation"] >= 0.995:
                duplicates.append(difference)
            else:
                others.append(difference)

        print("  %s" % name)
        if duplicates:
            duplicates_array = np.array(duplicates)
            print("    correlation >= 0.995 (n=%d): median ID gap %.0f, "
                  "%d of them within 20"
                  % (len(duplicates), np.median(duplicates_array),
                     int((duplicates_array <= 20).sum())))
        else:
            print("    correlation >= 0.995: none")

        if others:
            others_array = np.array(others)
            print("    correlation <  0.995 (n=%d): median ID gap %.0f, "
                  "%d of them within 20"
                  % (len(others), np.median(others_array),
                     int((others_array <= 20).sum())))
        print()

    print("  If the high-correlation pairs sit at small ID gaps and the rest do")
    print("  not, the IDs are sequential per scan and one person holds several.")

    # ---- 3. What it costs in accuracy ----
    leaked_validation = {p for p, m in validation_matches.items()
                         if m["correlation"] >= arguments.threshold}
    leaked_test = {p for p, m in test_matches.items()
                   if m["correlation"] >= arguments.threshold}

    print()
    print("=" * 76)
    print("ACCURACY WITH THE LEAKED PATIENTS REMOVED (threshold %.3f)"
          % arguments.threshold)
    print("=" * 76)

    for model_name in ("baseline", "gfnet"):
        print("  %s" % model_name.upper())

        for split_name, prefix, leaked in (
            ("validation", "validation_predictions", leaked_validation),
            ("test", "test_predictions", leaked_test),
        ):
            path = os.path.join(arguments.figures_dir,
                                "%s_%s.npz" % (prefix, model_name))
            result = accuracy_from_predictions(path, leaked)

            if result is None:
                print("    %-11s file not found: %s" % (split_name, path))
                continue

            full = accuracy_from_predictions(path)
            line = ("    %-11s all %d patients %.4f   "
                    "clean %d patients %.4f   change %+.1f pts"
                    % (split_name, full["kept_total"], full["kept_accuracy"],
                       result["kept_total"], result["kept_accuracy"],
                       (result["kept_accuracy"] - full["kept_accuracy"]) * 100))
            print(line)

            if result["removed_accuracy"] is not None:
                print("                on the %d removed patients: %.4f"
                      % (result["removed_total"], result["removed_accuracy"]))
        print()

    print("=" * 76)
    print("HOW TO READ THIS")
    print("=" * 76)
    print("  If accuracy on the removed patients is much higher than on the")
    print("  rest, those patients were easy because the model had already seen")
    print("  the same person. The 'clean' column is then the honest number.")
    print()
    print("  If validation drops a lot and test barely moves, the")
    print("  validation-to-test gap is partly an artifact of the ID scheme and")
    print("  not a property of the models.")


if __name__ == "__main__":
    main()
