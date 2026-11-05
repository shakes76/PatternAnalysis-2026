"""Report test metrics and confidence analyses from saved model predictions."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import CLASS_NAMES, load_graph
from utils import (
    calibration_bins,
    classification_metrics,
    confidence_error_auroc,
    expected_calibration_error,
    precision_recall_data,
    risk_coverage_curve,
)


def choose_confidence_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_accuracy: float,
) -> tuple[float | None, float, float]:
    """Find the lowest validation confidence cutoff that meets the accuracy target.

    Returns the cutoff, accuracy of retained nodes, and their coverage. If no
    cutoff meets the target, returns None, NaN, and zero coverage.
    """
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    # lowest passing threshold keeps the largest possible share of nodes
    for threshold in np.sort(np.unique(confidence)):
        kept = confidence >= threshold
        if correct[kept].mean() >= target_accuracy:
            return float(threshold), float(correct[kept].mean()), float(kept.mean())
    return None, float("nan"), 0.0


def save_confidence_plots(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None,
) -> Path:
    """Save test confidence, calibration, and risk-coverage plots to figures/."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    bins = calibration_bins(y_true, probabilities)
    curve = risk_coverage_curve(y_true, probabilities)

    # compare confidence distributions for correct and incorrect predictions
    fig, (ax_conf, ax_reliability, ax_risk) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax_conf.hist(confidence[correct], bins=15, alpha=0.65, label="correct")
    ax_conf.hist(confidence[~correct], bins=15, alpha=0.65, label="incorrect")
    ax_conf.set(xlabel="maximum class probability", ylabel="node count", title="test confidence")
    ax_conf.legend()

    # empty bins have no accuracy or confidence value to plot
    populated = bins["count"] > 0
    # the diagonal represents perfect agreement between confidence and accuracy
    ax_reliability.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax_reliability.plot(
        bins["confidence"][populated],
        bins["accuracy"][populated],
        marker="o",
        label="model",
    )
    ax_reliability.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence", ylabel="accuracy", title="test reliability")
    ax_reliability.legend()

    ax_risk.plot(curve["coverage"], curve["risk"], label="test risk-coverage")
    if threshold is not None:
        # show test performance at the threshold chosen on validation
        kept = confidence >= threshold
        if kept.any():
            point_accuracy = correct[kept].mean()
            ax_risk.scatter([kept.mean()], [1.0 - point_accuracy], color="red", label="validation threshold")
    ax_risk.set(xlim=(0, 1), ylim=(0, 1), xlabel="coverage", ylabel="error risk", title="test risk-coverage")
    ax_risk.legend()

    fig.suptitle(f"{model_name.upper()} confidence analysis")
    fig.tight_layout()
    output_path = Path("figures") / f"{model_name}_confidence_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # close the figure after saving so repeated analysis does not hold memory
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_precision_recall_plot(
    model_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Path:
    """Save one-vs-rest precision-recall curves for each class."""
    curves = precision_recall_data(y_true, probabilities, CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 5))
    for class_name, curve in curves.items():
        precision = curve["precision"]
        recall = curve["recall"]
        pr_auc = curve["pr_auc"]
        if precision.size and recall.size:
            ax.plot(recall, precision, label=f"{class_name} (PR-AUC={pr_auc:.3f})")

    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="recall",
        ylabel="precision",
        title=f"{model_name.upper()} one-vs-rest precision-recall",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path = Path("figures") / f"{model_name}_precision_recall.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    """Print test metrics and validation-selected reject-rule results for both models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="five-seed",
        help="prediction checkpoint tag used in the saved file names",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="minimum validation accuracy among retained predictions",
    )
    args = parser.parse_args()
    if not 0 < args.target_accuracy <= 1:
        parser.error("--target-accuracy must be greater than 0 and at most 1")

    labels = load_graph().y.numpy()  # labels align with the saved node indices

    for model_name in ("mlp", "gcn"):
        # filenames match predict.py's default output naming
        prediction_path = Path("outputs") / f"predictions_{model_name}_seed42_{args.tag}.npz"
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Missing {prediction_path}; run predict.py for this model first."
            )

        with np.load(prediction_path) as saved:
            val_mask = saved["val_mask"]
            test_mask = saved["test_mask"]
            val_probabilities = saved["probabilities"][val_mask]
            test_probabilities = saved["probabilities"][test_mask]

        y_val = labels[val_mask]
        y_test = labels[test_mask]
        # probabilities are required for ranking and calibration metrics
        metrics = classification_metrics(y_test, test_probabilities, CLASS_NAMES)
        ece = expected_calibration_error(y_test, test_probabilities)
        # choose the threshold using validation labels, not test labels
        threshold, val_accuracy, val_coverage = choose_confidence_threshold(
            y_val, val_probabilities, args.target_accuracy
        )

        # ECE measures the gap between confidence and accuracy across 15 bins
        print(f"\n{model_name.upper()} — test set")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Macro one-vs-rest AUROC: {metrics['macro_auroc_ovr']:.4f}")
        print(f"Macro PR-AUC: {metrics['macro_pr_auc']:.4f}")
        print(f"ECE (15 bins): {ece:.4f}")
        print(f"Confidence error-detection AUROC: {confidence_error_auroc(y_test, test_probabilities):.4f}")
        if threshold is None:
            print(f"No validation threshold reached the {args.target_accuracy:.1%} target accuracy.")
        else:
            test_confidence = test_probabilities.max(axis=1)
            test_correct = test_probabilities.argmax(axis=1) == y_test
            # apply the unchanged validation threshold to test predictions
            test_kept = test_confidence >= threshold
            test_coverage = float(test_kept.mean())
            test_accuracy = float(test_correct[test_kept].mean()) if test_kept.any() else float("nan")
            print(
                f"Validation-selected threshold: {threshold:.4f} "
                f"(validation accuracy={val_accuracy:.4f}, coverage={val_coverage:.4f})"
            )
            print(
                f"Test at fixed threshold: accuracy={test_accuracy:.4f}, "
                f"coverage={test_coverage:.4f}"
            )
        print(f"Saved confidence plots: {save_confidence_plots(model_name, y_test, test_probabilities, threshold)}")
        print(
            "Saved precision-recall plot: "
            f"{save_precision_recall_plot(model_name, y_test, test_probabilities)}"
        )
        print("Class                 Precision  Recall  F1      AUROC   PR-AUC  Support")
        for class_name, scores in metrics["per_class"].items():
            # each row reports one class against all remaining classes.
            print(
                f"{class_name:20s} "
                f"{scores['precision']:9.4f} "
                f"{scores['recall']:7.4f} "
                f"{scores['f1']:7.4f} "
                f"{scores['auroc_ovr']:7.4f} "
                f"{scores['pr_auc']:7.4f} "
                f"{scores['support']:8d}"
            )


if __name__ == "__main__":
    main()
