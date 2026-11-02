from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from dataset import NC_LABEL, AD_LABEL

CLASS_NAMES = ["NC", "AD"]
CONFIDENCE_THRESHOLD = 0.90


def save_plot(fig, output_dir: Path, filename: str):
    """
    Save and close a Matplotlib figure.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / filename

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {save_path}")


def plot_learning_curves(metrics: dict, save_path: Path):
    """
    Plot training and validation loss and accuracy across epochs.
    """

    epochs = range(1, len(metrics["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(epochs, metrics["train_loss"], label="Training")
    axes[0].plot(epochs, metrics["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, metrics["train_accuracy"], label="Training")
    axes[1].plot(epochs, metrics["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Training and Validation Learning Curves")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved learning curves: {save_path}")


def plot_confusion_matrix(
    y_true,
    y_pred,
    output_dir: Path,
):
    """
    Plot raw and row-normalised confusion matrices.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[NC_LABEL, AD_LABEL],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Raw counts
    axes[0].imshow(cm, cmap="Blues")

    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].set_xticks([0, 1], CLASS_NAMES)
    axes[0].set_yticks([0, 1], CLASS_NAMES)

    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
            )

    # Row-normalised counts
    row_totals = cm.sum(axis=1, keepdims=True)

    cm_normalised = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=float),
        where=row_totals != 0,
    )

    axes[1].imshow(
        cm_normalised,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("Normalised Confusion Matrix")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")

    axes[1].set_xticks([0, 1], CLASS_NAMES)
    axes[1].set_yticks([0, 1], CLASS_NAMES)

    for i in range(2):
        for j in range(2):
            axes[1].text(
                j,
                i,
                f"{cm_normalised[i, j]:.2%}",
                ha="center",
                va="center",
                fontsize=12,
            )

    save_plot(fig, output_dir, "confusion_matrix.png")


def plot_roc_curve(
    y_true,
    y_prob,
    auroc,
    output_dir: Path,
):
    """
    Plot the receiver operating characteristic curve.
    """

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"ROC (AUROC = {auroc:.4f})",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")

    ax.legend()
    ax.grid(alpha=0.3)

    save_plot(fig, output_dir, "roc_curve.png")


def plot_precision_recall_curve(
    y_true,
    y_prob,
    pr_auc,
    output_dir: Path,
):
    """
    Plot the precision-recall curve.
    """

    if len(np.unique(y_true)) < 2:
        return

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recall,
        precision,
        label=f"PR curve (AP = {pr_auc:.4f})",
    )

    baseline = np.mean(y_true)

    ax.axhline(
        baseline,
        linestyle="--",
        label=f"Random baseline = {baseline:.3f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    ax.legend()
    ax.grid(alpha=0.3)

    save_plot(fig, output_dir, "precision_recall_curve.png")


def plot_prediction_distribution(
    y_true,
    y_prob,
    output_dir: Path,
):
    """
    Plot the distribution of predicted AD probabilities by true class.
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        y_prob[y_true == NC_LABEL],
        bins=20,
        alpha=0.6,
        label="True NC",
    )

    ax.hist(
        y_prob[y_true == AD_LABEL],
        bins=20,
        alpha=0.6,
        label="True AD",
    )

    ax.axvline(
        0.5,
        linestyle="--",
        color="black",
        label="Decision threshold",
    )

    ax.set_xlabel("Predicted probability of AD")
    ax.set_ylabel("Number of patients")
    ax.set_title("Patient-level Prediction Distribution")

    ax.legend()
    ax.grid(alpha=0.3)

    save_plot(fig, output_dir, "prediction_distribution.png")


def plot_confidence_coverage(
    y_true,
    y_pred,
    y_prob,
    output_dir: Path,
):
    """
    Plot the relationship between prediction coverage and selective accuracy.
    """

    confidence = np.maximum(y_prob, 1 - y_prob)
    correctness = y_true == y_pred

    thresholds = np.linspace(0.5, 1.0, 101)

    coverage = []
    selective_accuracy = []

    for threshold in thresholds:
        accepted = confidence >= threshold

        coverage.append(accepted.mean())

        if np.any(accepted):
            selective_accuracy.append(correctness[accepted].mean())
        else:
            selective_accuracy.append(np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        coverage,
        selective_accuracy,
        label="Confidence-coverage trade-off",
    )

    # Highlight the selected confidence threshold.
    accepted = confidence >= CONFIDENCE_THRESHOLD
    selected_coverage = accepted.mean()

    if np.any(accepted):
        selected_accuracy = correctness[accepted].mean()

        ax.scatter(
            selected_coverage,
            selected_accuracy,
            color="red",
            zorder=3,
            label=f"{CONFIDENCE_THRESHOLD:.0%} confidence threshold",
        )

    ax.set_xlabel("Coverage (fraction of patients accepted)")
    ax.set_ylabel("Accuracy among accepted patients")
    ax.set_title("Confidence-Coverage Trade-off")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    ax.legend()
    ax.grid(alpha=0.3)

    save_plot(fig, output_dir, "confidence_coverage.png")


def generate_evaluation_plots(
    y_true,
    y_pred,
    y_prob,
    classification_metrics: dict,
    output_dir: Path,
):
    """
    Generate all patient-level evaluation plots.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        y_true,
        y_pred,
        output_dir,
    )

    plot_roc_curve(
        y_true,
        y_prob,
        classification_metrics["auroc"],
        output_dir,
    )

    plot_precision_recall_curve(
        y_true,
        y_prob,
        classification_metrics["pr_auc"],
        output_dir,
    )

    plot_prediction_distribution(
        y_true,
        y_prob,
        output_dir,
    )

    plot_confidence_coverage(
        y_true,
        y_pred,
        y_prob,
        output_dir,
    )
