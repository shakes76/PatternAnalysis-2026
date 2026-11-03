"""
Evaluation pipeline for ADNI MRI classification.

Supports:
    - Baseline CNN and ConvNeXt-Tiny
    - Patient-independent validation/test evaluation
    - Accuracy, precision, recall, F1, AUROC
    - Confusion matrix and ROC curve
    - Calibration analysis
    - Misclassification examples
    - Inference latency

Classes:
    NC = 0
    AD = 1
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from modules import create_model, count_parameters
from dataset import get_transforms

CLASS_NAMES = ["NC", "AD"]


class EvaluationDataset(Dataset):
    """Load ADNI MRI slices without random augmentation."""

    def __init__(self, records, image_size=224):
        self.records = records

        self.transform = get_transforms(
            image_size=image_size,
            training=False,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        # Adjust this key if your dataset.py uses a different name.
        image_path = record["path"]

        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("L"))

        return (
            tensor,
            int(record["label"]),
            str(image_path),
            str(record["patient_id"]),
        )


def load_evaluation_data(split_path, split, image_size, batch_size, workers):
    with open(split_path, "r") as f:
        splits = json.load(f)

    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be 'val' or 'test'.")

    dataset = EvaluationDataset(splits[split], image_size=image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model_name = checkpoint["model_name"]

    model = create_model(
        model_name=model_name,
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def collect_predictions(model, loader, device):
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []
    all_patient_ids = []

    elapsed_seconds = 0.0
    total_images = 0

    for images, labels, paths, patient_ids in loader:
        images = images.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)[:, 1]

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_seconds += time.perf_counter() - start
        total_images += len(labels)

        predictions = (probabilities >= 0.5).long()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())
        all_paths.extend(paths)
        all_patient_ids.extend(patient_ids)

    return {
        "labels": np.asarray(all_labels),
        "predictions": np.asarray(all_predictions),
        "probabilities": np.asarray(all_probabilities),
        "paths": all_paths,
        "patient_ids": all_patient_ids,
        "latency_ms_per_image": elapsed_seconds / total_images * 1000,
    }

def save_predictions(results, output_dir, patient_threshold=0.5):
    """
    Export individual slice predictions and patient-level
    aggregated predictions for reproducible analysis.

    These files are for local analysis and should not be
    committed to GitHub because they contain patient IDs.
    """

    import csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save slice-level predictions.
    with open(output_dir / "slice_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_id",
            "path",
            "true_label",
            "predicted_label",
            "probability_ad",
        ])

        for i in range(len(results["labels"])):
            writer.writerow([
                results["patient_ids"][i],
                results["paths"][i],
                int(results["labels"][i]),
                int(results["predictions"][i]),
                float(results["probabilities"][i]),
            ])

    # Aggregate predictions by patient.
    patient_data = {}

    for patient_id, label, probability in zip(
        results["patient_ids"],
        results["labels"],
        results["probabilities"],
    ):
        if patient_id not in patient_data:
            patient_data[patient_id] = {
                "labels": [],
                "probabilities": [],
            }

        patient_data[patient_id]["labels"].append(int(label))
        patient_data[patient_id]["probabilities"].append(
            float(probability)
        )

    with open(output_dir / "patient_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_id",
            "true_label",
            "predicted_label_default",
            "predicted_label_selected",
            "probability_ad",
            "num_slices",
        ])

        for patient_id, values in patient_data.items():
            if len(set(values["labels"])) != 1:
                raise ValueError(
                    f"Inconsistent labels for patient {patient_id}"
                )

            probability = float(np.mean(values["probabilities"]))

            writer.writerow([
                patient_id,
                values["labels"][0],
                int(probability >= 0.5),
                int(probability >= patient_threshold),
                probability,
                len(values["probabilities"]),
            ])

    print("Saved slice and patient predictions.")

def load_selected_threshold(model_name, threshold_file):
    """
    Load the patient-level threshold selected on validation data.

    Never select or optimise thresholds using test data.
    """
    with open(threshold_file, "r") as f:
        thresholds = json.load(f)

    if model_name not in thresholds:
        raise ValueError(
            f"No threshold found for model: {model_name}"
        )

    config = thresholds[model_name]

    if config.get("selection_split") != "validation":
        raise ValueError(
            "Threshold must have been selected on validation data."
        )

    if config.get("selection_metric") != "patient_balanced_accuracy":
        raise ValueError(
            "Unexpected threshold-selection criterion."
        )

    threshold = float(config["threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    return threshold

def resolve_patient_threshold(model_name, split, threshold_file):
    """Keep validation prediction generation independent of threshold selection."""
    if split == "val":
        return 0.5, "default_preselection"

    if split == "test":
        return (
            load_selected_threshold(model_name, threshold_file),
            "validation",
        )

    raise ValueError(f"Unsupported evaluation split: {split}")


def calculate_metrics(results, threshold=0.5):
    """
    Calculate classification metrics using a specified threshold.

    AD = 1 (positive class)
    NC = 0 (negative class)
    """

    y_true = np.asarray(results["labels"])
    y_prob = np.asarray(results["probabilities"])

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "brier_score": float(
            brier_score_loss(y_true, y_prob)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(
            roc_auc_score(y_true, y_prob)
        )
    else:
        metrics["auroc"] = None

    return metrics

def aggregate_by_patient(results):
    """
    Average slice probabilities per patient.

    This is a simple research aggregation strategy,
    not a clinically validated diagnostic method.
    """

    patients = {}

    for label, probability, patient_id in zip(
        results["labels"],
        results["probabilities"],
        results["patient_ids"],
    ):
        if patient_id not in patients:
            patients[patient_id] = {
                "labels": [],
                "probabilities": [],
            }

        patients[patient_id]["labels"].append(int(label))
        patients[patient_id]["probabilities"].append(float(probability))

    labels = []
    probabilities = []

    for patient_id, values in patients.items():
        unique_labels = set(values["labels"])

        if len(unique_labels) != 1:
            raise ValueError(
                f"Patient {patient_id} has inconsistent labels."
            )

        labels.append(values["labels"][0])
        probabilities.append(np.mean(values["probabilities"]))

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    return {
        "labels": labels,
        "predictions": (probabilities >= 0.5).astype(int),
        "probabilities": probabilities,
    }


def save_confusion_matrix(
    results,
    output_dir,
    name,
    threshold=0.5,
):
    """
    Save a confusion matrix at the specified threshold.
    """

    predictions = (
        np.asarray(results["probabilities"]) >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        results["labels"],
        predictions,
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title(
        f"{name.title()} Confusion Matrix "
        f"(threshold={threshold:.2f})"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / f"{name}_confusion_matrix.png",
        dpi=150,
    )

    plt.close()

def save_roc_curve(results, output_dir, name):
    if len(np.unique(results["labels"])) < 2:
        return

    fpr, tpr, _ = roc_curve(
        results["labels"],
        results["probabilities"],
    )

    auc = roc_auc_score(
        results["labels"],
        results["probabilities"],
    )

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_roc_curve.png", dpi=150)
    plt.close()


def save_calibration_plot(results, output_dir):
    fraction_positive, mean_predicted = calibration_curve(
        results["labels"],
        results["probabilities"],
        n_bins=10,
        strategy="uniform",
    )

    plt.figure(figsize=(6, 5))
    plt.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Model",
    )
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed AD Frequency")
    plt.title("Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close()


def save_misclassified_examples(results, output_dir, max_per_type=3):
    """
    Save representative false positives and false negatives.

    Prioritise confident mistakes and avoid repeatedly selecting
    slices from the same patient.
    """

    labels = results["labels"]
    predictions = results["predictions"]
    probabilities = results["probabilities"]
    patient_ids = results["patient_ids"]

    # NC incorrectly predicted as AD.
    false_positives = np.flatnonzero(
        (labels == 0) & (predictions == 1)
    )

    # AD incorrectly predicted as NC.
    false_negatives = np.flatnonzero(
        (labels == 1) & (predictions == 0)
    )

    def select_examples(indices, confidence_scores):
        """
        Select confident mistakes from distinct patients.
        """

        ranked = sorted(
            indices,
            key=lambda i: confidence_scores[i],
            reverse=True,
        )

        selected = []
        used_patients = set()

        for i in ranked:
            patient_id = patient_ids[i]

            if patient_id in used_patients:
                continue

            selected.append(i)
            used_patients.add(patient_id)

            if len(selected) >= max_per_type:
                break

        return selected

    # For false positives, confidence is P(AD).
    selected_fp = select_examples(
        false_positives,
        probabilities,
    )

    # For false negatives, confidence is P(NC) = 1 - P(AD).
    selected_fn = select_examples(
        false_negatives,
        1 - probabilities,
    )

    selected = [
        ("False Positive", i) for i in selected_fp
    ] + [
        ("False Negative", i) for i in selected_fn
    ]

    if not selected:
        print("No misclassified examples found.")
        return

    fig, axes = plt.subplots(
        2,
        max_per_type,
        figsize=(4 * max_per_type, 8),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for row, examples in enumerate([selected_fp, selected_fn]):
        error_type = "False Positive" if row == 0 else "False Negative"

        for col, index in enumerate(examples):
            ax = axes[row, col]

            with Image.open(results["paths"][index]) as image:
                ax.imshow(image.convert("L"), cmap="gray")

            true_label = CLASS_NAMES[int(labels[index])]
            predicted_label = CLASS_NAMES[int(predictions[index])]
            probability = float(probabilities[index])

            ax.set_title(
                f"{error_type}\n"
                f"True: {true_label} | Pred: {predicted_label}\n"
                f"P(AD): {probability:.3f}",
                fontsize=10,
            )

            ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_dir / "misclassified_examples_balanced.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved {len(selected_fp)} false-positive and "
        f"{len(selected_fn)} false-negative examples."
    )

def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, checkpoint = load_model(args.checkpoint, device)

    image_size = checkpoint.get("image_size", 224)

    loader = load_evaluation_data(
        args.splits,
        args.split,
        image_size,
        args.batch_size,
        args.workers,
    )

    output_dir = Path(args.output_dir) / checkpoint["model_name"] / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ADNI MRI EVALUATION")
    print("=" * 60)
    print("Model:", checkpoint["model_name"])
    print("Split:", args.split)
    print("Device:", device)
    print("Images:", len(loader.dataset))

    results = collect_predictions(model, loader, device)

    # Slice-level evaluation remains at the default threshold.
    slice_metrics = calculate_metrics(
        results,
        threshold=0.5,
    )

    # Aggregate slice probabilities into one probability per patient.
    patient_results = aggregate_by_patient(results)

    # Default patient-level classification.
    patient_metrics_default = calculate_metrics(
        patient_results,
        threshold=0.5,
    )

    # Validation predictions precede threshold selection.
    # Test predictions require a frozen validation-selected threshold.
    selected_threshold, threshold_source = resolve_patient_threshold(
        checkpoint["model_name"],
        args.split,
        args.threshold_file,
    )

    # Apply the frozen threshold to patient-level predictions.
    patient_metrics_selected = calculate_metrics(
        patient_results,
        threshold=selected_threshold,
    )

    slice_metrics["latency_ms_per_image"] = (
        results["latency_ms_per_image"]
    )

    slice_metrics["total_parameters"] = count_parameters(model)

    print("\nSLICE-LEVEL METRICS (threshold=0.50)")
    print(json.dumps(slice_metrics, indent=2))

    print("\nPATIENT-LEVEL METRICS (threshold=0.50)")
    print(json.dumps(patient_metrics_default, indent=2))

    print(
        f"\nPATIENT-LEVEL METRICS "
        f"(selected threshold={selected_threshold:.2f})"
    )
    print(json.dumps(patient_metrics_selected, indent=2))

    # Confusion matrices.
    save_confusion_matrix(
        results,
        output_dir,
        "slice",
        threshold=0.5,
    )

    save_confusion_matrix(
        patient_results,
        output_dir,
        "patient_default",
        threshold=0.5,
    )

    save_confusion_matrix(
        patient_results,
        output_dir,
        "patient_selected",
        threshold=selected_threshold,
    )

    # Threshold-independent evaluation figures.
    save_roc_curve(results, output_dir, "slice")
    save_roc_curve(patient_results, output_dir, "patient")

    save_calibration_plot(results, output_dir)

    # Error analysis.
    save_misclassified_examples(
        results,
        output_dir,
        max_per_type=3,
    )

    # Export individual predictions for reproducibility.
    save_predictions(results, output_dir, patient_threshold=selected_threshold)

    # Save complete metrics.
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "model": checkpoint["model_name"],
                "split": args.split,
                "checkpoint_epoch": checkpoint["epoch"],
                "threshold_selection": {
                    "source": threshold_source,
                    "criterion": (
                        "patient_balanced_accuracy"
                        if args.split == "test"
                        else "not_yet_selected"
                    ),
                    "selected_threshold": selected_threshold,
                },
                "slice_level": slice_metrics,
                "patient_level_default": patient_metrics_default,
                "patient_level_selected": patient_metrics_selected,
            },
            f,
            indent=2,
        )

    print("\nEvaluation complete.")
    print("Results saved to:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root output directory; model name and split are appended automatically.",
    )

    parser.add_argument(
        "--threshold-file",
        default="outputs/threshold_analysis/selected_thresholds.json",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
