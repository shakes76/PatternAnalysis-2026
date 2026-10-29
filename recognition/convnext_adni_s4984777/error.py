
"""
COMP3710 - Representative ADNI Misclassification Analysis

Selects:
- 2 false negatives (AD classified as NC)
- 2 false positives (NC classified as AD)
- 1 additional high-confidence error

Verifies prediction-to-image ordering using the actual
test Dataset before associating predictions with filenames.

Requires:
    dataset.py
    predict.py
    splits.csv
    results/convnext/predictions.csv

No GPU or model checkpoint required.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from torch.utils.data import SequentialSampler

from dataset import DATA_DIR, SPLITS_CSV, get_dataloaders
from predict import get_transform, resolve_image_path


PREDICTIONS = Path("results/convnext/predictions.csv")
OUTPUT = Path("results/error_analysis")

LABELS = {"NC": 0, "AD": 1}
NAMES = {0: "NC", 1: "AD"}


def load_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def get_test_records():
    rows = load_csv(SPLITS_CSV)
    return [
        row for row in rows
        if row["split"].strip().lower() == "test"
    ]


def verify_alignment(records, predictions):
    """
    Check that test_index corresponds to the correct MRI.

    Checks:
    1. Test sampler is sequential.
    2. Test labels match the CSV.
    3. Loaded test image tensors match manifest images
       at several positions across the test set.

    Fails rather than silently producing mismatched
    patient/image associations.
    """

    loaders = get_dataloaders(
        csv_path=SPLITS_CSV,
        data_dir=DATA_DIR,
        batch_size=8,
        num_workers=0,
        image_size=224,
    )

    loader = loaders["test"]
    dataset = loader.dataset

    if not isinstance(loader.sampler, SequentialSampler):
        raise RuntimeError(
            "Test loader is not sequential. "
            "Cannot use prediction indices safely."
        )

    if len(dataset) != len(records):
        raise RuntimeError(
            "Manifest and test Dataset have different lengths."
        )

    if len(predictions) != len(records):
        raise RuntimeError(
            "Prediction and manifest lengths differ."
        )

    for i, prediction in enumerate(predictions):
        if int(prediction["test_index"]) != i:
            raise RuntimeError(
                f"Prediction index mismatch at row {i}."
            )

        expected_label = LABELS[
            records[i]["label"].strip().upper()
        ]

        if int(prediction["true_label"]) != expected_label:
            raise RuntimeError(
                f"Label order mismatch at row {i}."
            )

    transform = get_transform()

    # Verify image identity, not just label identity.
    positions = np.unique(
        np.linspace(
            0,
            len(records) - 1,
            min(15, len(records)),
            dtype=int,
        )
    )

    for i in positions:
        dataset_image, dataset_label = dataset[int(i)]

        path = resolve_image_path(
            records[int(i)]["path"],
            DATA_DIR,
        )

        with Image.open(path) as img:
            expected_image = transform(
                img.convert("RGB")
            )

        if int(dataset_label) != LABELS[
            records[int(i)]["label"].strip().upper()
        ]:
            raise RuntimeError(
                f"Dataset label mismatch at index {i}."
            )

        if not torch.allclose(
            dataset_image,
            expected_image,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                f"Image-order/preprocessing mismatch at {i}.\n"
                "Inspect dataset.py before linking patient IDs."
            )

    print(
        "Alignment checks passed:",
        len(predictions),
        "prediction rows.",
    )

    return dataset


def build_errors(records, predictions):
    errors = []

    for index, (record, pred) in enumerate(
        zip(records, predictions)
    ):
        true_label = int(pred["true_label"])
        predicted_label = int(pred["predicted_label"])

        if true_label == predicted_label:
            continue

        confidence = float(pred["confidence"])

        if true_label == 1 and predicted_label == 0:
            error_type = "False Negative (AD to NC)"
        else:
            error_type = "False Positive (NC to AD)"

        errors.append({
            "test_index": index,
            "patient_id": record.get("patient_id", ""),
            "scan_id": record.get("scan_id", ""),
            "path": record["path"],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "probability_AD": float(
                pred["probability_AD"]
            ),
            "confidence": confidence,
            "error_type": error_type,
        })

    return errors


def select_examples(errors):
    """
    Select one high- and one lower-confidence error
    of each type, plus a fifth high-confidence case.

    Prefer distinct patients to avoid near-duplicate
    slices from the same individual.
    """

    fn = [
        row for row in errors
        if row["true_label"] == 1
    ]

    fp = [
        row for row in errors
        if row["true_label"] == 0
    ]

    chosen = []
    used_patients = set()
    used_indices = set()

    def add_first(candidates):
        for row in candidates:
            idx = row["test_index"]

            patient = row["patient_id"]
            key = patient or f"index-{idx}"

            if (
                idx not in used_indices
                and key not in used_patients
            ):
                chosen.append(row)
                used_indices.add(idx)
                used_patients.add(key)
                return True

        return False

    for group in (fn, fp):
        # High-confidence failure.
        add_first(
            sorted(
                group,
                key=lambda r: -r["confidence"],
            )
        )

        # Lower-confidence, potentially ambiguous failure.
        add_first(
            sorted(
                group,
                key=lambda r: r["confidence"],
            )
        )

    # Fifth example: highest-confidence remaining failure.
    add_first(
        sorted(
            errors,
            key=lambda r: -r["confidence"],
        )
    )

    if len(chosen) < 5:
        print(
            f"WARNING: Selected only {len(chosen)} cases "
            "after patient-uniqueness filtering."
        )

    return chosen


def save_examples(examples):
    n = len(examples)

    if n == 0:
        raise RuntimeError("No errors selected.")

    fig, axes = plt.subplots(
        2, 3,
        figsize=(14, 9),
        constrained_layout=True,
    )   

    axes = axes.flatten()

    for i, example in enumerate(examples):
        path = resolve_image_path(
            example["path"], DATA_DIR
        )

        with Image.open(path) as image:
            display_image = image.convert("L").copy()

        ax = axes[i]
        ax.imshow(display_image, cmap="gray")
        ax.axis("off")

        truth = NAMES[example["true_label"]]
        predicted = NAMES[example["predicted_label"]]

        ax.set_title(
            f"Case {i + 1}: {truth} → {predicted}\n"
            f"Confidence: {example['confidence']:.2%}",
            fontsize=11,
        )

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        "ConvNeXt-Tiny: Representative Test Errors",
        fontsize=15,
    )

    fig.set_constrained_layout_pads(
        w_pad=0.1,
        h_pad=0.2,
        hspace=0.18,
    )   

    fig.savefig(
        OUTPUT / "representative_errors.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_csv(path, rows, columns):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    records = get_test_records()
    predictions = load_csv(PREDICTIONS)

    verify_alignment(records, predictions)

    errors = build_errors(records, predictions)
    examples = select_examples(errors)

    fn = sum(
        row["true_label"] == 1 for row in errors
    )
    fp = sum(
        row["true_label"] == 0 for row in errors
    )

    summary = [
        {"error_type": "False Negative", "count": fn},
        {"error_type": "False Positive", "count": fp},
        {"error_type": "Total", "count": len(errors)},
    ]

    save_csv(
        OUTPUT / "error_summary.csv",
        summary,
        ["error_type", "count"],
    )

    columns = [
        "test_index",
        "patient_id",
        "scan_id",
        "path",
        "true_label",
        "predicted_label",
        "probability_AD",
        "confidence",
        "error_type",
    ]

    save_csv(
        OUTPUT / "representative_errors.csv",
        examples,
        columns,
    )

    save_examples(examples)

    print("\n" + "=" * 45)
    print("REPRESENTATIVE ERROR ANALYSIS")
    print("=" * 45)

    print(f"Total errors: {len(errors)}")
    print(f"False negatives (AD to NC): {fn}")
    print(f"False positives (NC to AD): {fp}")

    for i, row in enumerate(examples, 1):
        print(
            f"Case {i}: {row['error_type']} | "
            f"Confidence={row['confidence']:.2%} | "
            f"Test index={row['test_index']}"
        )

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()

