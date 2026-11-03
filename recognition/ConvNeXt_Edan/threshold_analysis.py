"""
Threshold analysis for ADNI MRI classification.

Uses validation-set patient predictions to:
    - Compare classification thresholds.
    - Optimise patient-level balanced accuracy.
    - Compare baseline CNN and ConvNeXt-Tiny.
    - Save selected thresholds for held-out test evaluation.

IMPORTANT:
    Thresholds must be selected using validation data only.
    Never tune thresholds using the test set.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


MODELS = ["baseline", "convnext"]


def load_predictions(path):
    """Load saved patient-level validation predictions."""

    df = pd.read_csv(path)

    required_columns = {
        "patient_id",
        "true_label",
        "probability_ad",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    if df.empty:
        raise ValueError(f"No predictions found in {path}")

    if df["patient_id"].duplicated().any():
        raise ValueError(
            f"Duplicate patient IDs found in {path}"
        )

    labels = df["true_label"].to_numpy(dtype=int)
    probabilities = df["probability_ad"].to_numpy(dtype=float)

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities found")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    return df


def calculate_threshold_metrics(labels, probabilities, threshold):
    """Evaluate a classification threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def analyse_model(model_name, input_dir, thresholds):
    """Evaluate all thresholds for one model."""

    prediction_path = (
        input_dir
        / model_name
        / "val"
        / "patient_predictions.csv"
    )

    print(f"\nLoading {model_name}: {prediction_path}")

    df = load_predictions(prediction_path)

    labels = df["true_label"].to_numpy(dtype=int)
    probabilities = df["probability_ad"].to_numpy(dtype=float)

    results = [
        calculate_threshold_metrics(labels, probabilities, threshold)
        for threshold in thresholds
    ]

    results_df = pd.DataFrame(results)

    # Primary criterion: highest balanced accuracy.
    best_score = results_df["balanced_accuracy"].max()

    candidates = results_df[
        np.isclose(results_df["balanced_accuracy"], best_score)
    ].copy()

    # Tie-breaker: choose the threshold closest to 0.5.
    # This avoids selecting an unnecessarily extreme threshold.
    candidates["distance_from_default"] = (
        candidates["threshold"] - 0.5
    ).abs()

    best = candidates.sort_values(
        ["distance_from_default", "threshold"]
    ).iloc[0].to_dict()

    default = calculate_threshold_metrics(
        labels,
        probabilities,
        0.5,
    )

    auroc = float(roc_auc_score(labels, probabilities))

    return {
        "model": model_name,
        "num_patients": len(df),
        "auroc": auroc,
        "default": default,
        "selected": best,
        "results_df": results_df,
    }


def plot_threshold_metrics(analysis, output_dir):
    """Plot threshold sensitivity for one model."""

    df = analysis["results_df"]
    model_name = analysis["model"]

    plt.figure(figsize=(9, 6))

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]:
        plt.plot(
            df["threshold"],
            df[metric],
            label=metric.replace("_", " ").title(),
        )

    plt.axvline(
        0.5,
        linestyle="--",
        color="gray",
        label="Default threshold (0.5)",
    )

    plt.axvline(
        analysis["selected"]["threshold"],
        linestyle=":",
        color="black",
        label=(
            f"Selected threshold "
            f"({analysis['selected']['threshold']:.2f})"
        ),
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Metric")
    plt.title(
        f"{model_name.title()} - Patient-Level Threshold Analysis"
    )

    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / f"{model_name}_threshold_analysis.png",
        dpi=150,
    )

    plt.close()


def save_comparison(analyses, output_dir):
    """Save summary comparing default and selected thresholds."""

    comparison_rows = []

    for analysis in analyses:
        for setting in ["default", "selected"]:
            metrics = analysis[setting]

            comparison_rows.append({
                "model": analysis["model"],
                "setting": setting,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "auroc": analysis["auroc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        output_dir / "threshold_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        comparison_df[
            [
                "model",
                "setting",
                "threshold",
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
            ]
        ].to_string(index=False)
    )

    return comparison_df


def main(args):

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate thresholds from 0.05 to 0.95 in steps of 0.01.
    thresholds = np.round(
        np.arange(0.05, 0.951, 0.01),
        2,
    )

    analyses = []

    for model_name in MODELS:
        analysis = analyse_model(
            model_name,
            input_dir,
            thresholds,
        )

        analyses.append(analysis)

        analysis["results_df"].to_csv(
            output_dir / f"{model_name}_threshold_metrics.csv",
            index=False,
        )

        plot_threshold_metrics(analysis, output_dir)

    save_comparison(analyses, output_dir)

    selected_thresholds = {
        analysis["model"]: {
            "threshold": float(
                analysis["selected"]["threshold"]
            ),
            "selection_metric": "patient_balanced_accuracy",
            "selection_split": "validation",
            "num_validation_patients": int(
                analysis["num_patients"]
            ),
        }
        for analysis in analyses
    }

    with open(
        output_dir / "selected_thresholds.json",
        "w",
    ) as f:
        json.dump(selected_thresholds, f, indent=2)

    print("\nSelected thresholds:")
    print(json.dumps(selected_thresholds, indent=2))

    print("\nResults saved to:", output_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="outputs",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/threshold_analysis",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
