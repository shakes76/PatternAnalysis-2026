"""
Reproduce the supplementary pretrained patient-level error analysis.

Reads saved predictions from the three experimental configurations.
Patient identifiers are used only for matching records locally and
are never included in the printed report.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


DEFAULT_PATHS = {
    "baseline": "outputs/baseline/test/patient_predictions.csv",
    "scratch": "outputs/convnext/test/patient_predictions.csv",
    "pretrained": (
        "outputs/pretrained_trial/convnext/test/"
        "patient_predictions.csv"
    ),
}

THRESHOLDS = {
    "baseline": 0.78,
    "scratch": 0.45,
    "pretrained": 0.64,
}


def load_predictions(path, threshold):
    """Load and validate patient-level prediction records."""
    records = {}

    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            patient = row["patient_id"]

            if patient in records:
                raise ValueError("Duplicate patient record.")

            label = int(row["true_label"])
            probability = float(row["probability_ad"])
            prediction = int(row["predicted_label_selected"])
            default = int(row["predicted_label_default"])
            slices = int(row["num_slices"])

            if label not in (0, 1):
                raise ValueError("Invalid ground-truth label.")

            if not math.isfinite(probability):
                raise ValueError("Non-finite probability.")

            if not 0 <= probability <= 1:
                raise ValueError("Probability outside [0, 1].")

            if prediction != int(probability >= threshold):
                raise ValueError("Selected threshold mismatch.")

            if default != int(probability >= 0.50):
                raise ValueError("Default threshold mismatch.")

            if slices <= 0:
                raise ValueError("Invalid slice count.")

            records[patient] = {
                "label": label,
                "probability": probability,
                "prediction": prediction,
                "default": default,
                "margin": abs(probability - threshold),
                "slices": slices,
            }

    if not records:
        raise ValueError("Empty prediction file.")

    return records


def validate_alignment(models):
    """Ensure that all configurations use identical test patients."""
    patients = sorted(models["baseline"])

    for name, records in models.items():
        if set(records) != set(patients):
            raise ValueError(f"Patient mismatch in {name}.")

        for patient in patients:
            reference = models["baseline"][patient]
            current = records[patient]

            if current["label"] != reference["label"]:
                raise ValueError("Ground-truth label mismatch.")

            if current["slices"] != reference["slices"]:
                raise ValueError("Patient slice-count mismatch.")

    return patients


def error_set(records):
    """Identify incorrectly classified patients."""
    return {
        patient
        for patient, record in records.items()
        if record["label"] != record["prediction"]
    }


def summarize_errors(records, true_label, title):
    """Calculate probability and margin statistics for errors."""
    errors = [
        record for record in records.values()
        if record["label"] == true_label
        and record["prediction"] != true_label
    ]

    print(f"\n{title}")
    print("Count:", len(errors))

    if errors:
        print(
            "Mean AD probability:",
            f"{mean(r['probability'] for r in errors):.4f}",
        )
        print(
            "Mean decision margin:",
            f"{mean(r['margin'] for r in errors):.4f}",
        )
        print(
            "Margins >= 0.10:",
            sum(r["margin"] >= 0.10 for r in errors),
        )
        print(
            "Mean slices:",
            f"{mean(r['slices'] for r in errors):.2f}",
        )


def select_cases(models, patients):
    """
    Select representative cases by explicit failure patterns.

    Within each pattern, prioritize the strongest incorrect
    pretrained probability. For the threshold-dependent case,
    prioritize the largest probability below the selected threshold.

    Selection is retrospective, not independent validation.
    """
    baseline = models["baseline"]
    scratch = models["scratch"]
    pretrained = models["pretrained"]

    def correct(records, patient):
        record = records[patient]
        return record["label"] == record["prediction"]

    def choose(predicate, strongest_positive):
        candidates = [p for p in patients if predicate(p)]

        if not candidates:
            return None

        key = lambda p: pretrained[p]["probability"]

        return (
            max(candidates, key=key)
            if strongest_positive
            else min(candidates, key=key)
        )

    return [
        choose(
            lambda p:
                pretrained[p]["label"] == 0
                and correct(baseline, p)
                and not correct(scratch, p)
                and not correct(pretrained, p),
            True,
        ),
        choose(
            lambda p:
                pretrained[p]["label"] == 0
                and not correct(baseline, p)
                and not correct(scratch, p)
                and not correct(pretrained, p),
            True,
        ),
        choose(
            lambda p:
                pretrained[p]["label"] == 1
                and correct(baseline, p)
                and correct(scratch, p)
                and not correct(pretrained, p),
            False,
        ),
        choose(
            lambda p:
                pretrained[p]["label"] == 1
                and not correct(baseline, p)
                and not correct(scratch, p)
                and not correct(pretrained, p),
            False,
        ),
        choose(
            lambda p:
                pretrained[p]["label"] == 1
                and correct(baseline, p)
                and not correct(scratch, p)
                and pretrained[p]["default"] == 1
                and pretrained[p]["prediction"] == 0,
            True,
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    for name, path in DEFAULT_PATHS.items():
        parser.add_argument(f"--{name}", default=path)

    args = parser.parse_args()

    models = {
        name: load_predictions(
            getattr(args, name), THRESHOLDS[name]
        )
        for name in DEFAULT_PATHS
    }

    patients = validate_alignment(models)
    pretrained = models["pretrained"]

    print("=== PRETRAINED PATIENT ERROR AUDIT ===")
    print("Aligned test patients:", len(patients))

    for name, records in models.items():
        accuracy = 1 - len(error_set(records)) / len(patients)
        print(f"{name} accuracy: {accuracy:.4f}")

    summarize_errors(pretrained, 0, "False positives")
    summarize_errors(pretrained, 1, "False negatives")

    pretrained_errors = error_set(pretrained)

    print("\nError overlap")
    print(
        "Shared with baseline:",
        len(pretrained_errors & error_set(models["baseline"])),
    )
    print(
        "Shared with scratch:",
        len(pretrained_errors & error_set(models["scratch"])),
    )

    for prediction_key in ("default", "prediction"):
        tn = fp = fn = tp = 0

        for record in pretrained.values():
            actual = record["label"]
            predicted = record[prediction_key]

            tn += int(actual == 0 and predicted == 0)
            fp += int(actual == 0 and predicted == 1)
            fn += int(actual == 1 and predicted == 0)
            tp += int(actual == 1 and predicted == 1)

        print(
            f"\n{prediction_key}: "
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )
        print(f"Accuracy: {(tn + tp) / len(patients):.4f}")
        print(f"Sensitivity: {tp / (tp + fn):.4f}")

    print("\nRepresentative failure cases")
    print("| Case | True class | Baseline P(AD) | "
          "Scratch P(AD) | Pretrained P(AD) |")
    print("|---|---|---:|---:|---:|")

    for i, patient in enumerate(
        select_cases(models, patients), start=1
    ):
        if patient is None:
            print(f"| {i} | No matching case | - | - | - |")
            continue

        label = "AD" if pretrained[patient]["label"] else "NC"
        b = models["baseline"][patient]["probability"]
        s = models["scratch"][patient]["probability"]
        p = pretrained[patient]["probability"]

        print(
            f"| {i} | {label} | "
            f"{b:.4f} | {s:.4f} | {p:.4f} |"
        )

    print("\nAudit completed without disclosing patient IDs.")


if __name__ == "__main__":
    main()
