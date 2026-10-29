
"""
COMP3710 — Confidence Calibration and Selective Classification

Reads predictions.csv from BaselineCNN and ConvNeXt-Tiny.

Outputs:
- calibration_comparison.png
- confidence_distribution.png
- coverage_accuracy.png
- confidence_summary.csv
- referral_thresholds.csv

No GPU required.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


INPUTS = {
    "BaselineCNN": Path("results/baseline/predictions.csv"),
    "ConvNeXt-Tiny": Path("results/convnext/predictions.csv"),
}

OUTPUT = Path("results/confidence_analysis")
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
N_BINS = 10


def load_predictions(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No predictions in {path}")

    y_true = np.array(
        [int(row["true_label"]) for row in rows]
    )

    y_pred = np.array(
        [int(row["predicted_label"]) for row in rows]
    )

    p_ad = np.array(
        [float(row["probability_AD"]) for row in rows]
    )

    if not np.all((p_ad >= 0) & (p_ad <= 1)):
        raise ValueError(f"Invalid probabilities in {path}")

    confidence = np.maximum(p_ad, 1 - p_ad)
    correct = y_true == y_pred

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "p_ad": p_ad,
        "confidence": confidence,
        "correct": correct,
    }


def reliability_bins(data, n_bins=10):
    """
    Confidence reliability diagram.

    Compares predicted top-class confidence against
    empirical classification accuracy.
    """
    conf = data["confidence"]
    correct = data["correct"]

    edges = np.linspace(0, 1, n_bins + 1)
    indices = np.minimum(
        np.digitize(conf, edges[1:-1]),
        n_bins - 1,
    )

    mean_conf = []
    mean_acc = []
    counts = []

    for i in range(n_bins):
        mask = indices == i

        if not np.any(mask):
            continue

        mean_conf.append(float(conf[mask].mean()))
        mean_acc.append(float(correct[mask].mean()))
        counts.append(int(mask.sum()))

    ece = sum(
        count / len(conf) * abs(acc - avg_conf)
        for avg_conf, acc, count in zip(
            mean_conf, mean_acc, counts
        )
    )

    return (
        np.asarray(mean_conf),
        np.asarray(mean_acc),
        float(ece),
    )


def plot_calibration(datasets):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect calibration",
    )

    for name, data in datasets.items():
        mean_conf, mean_acc, ece = reliability_bins(data)

        ax.plot(
            mean_conf,
            mean_acc,
            marker="o",
            linewidth=2,
            label=f"{name} (ECE={ece:.4f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Observed Accuracy")
    ax.set_title("Confidence Reliability Comparison")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "calibration_comparison.png",
        dpi=200,
    )
    plt.close(fig)


def plot_confidence_distribution(datasets):
    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(12, 4.5),
        squeeze=False,
    )

    for ax, (name, data) in zip(axes[0], datasets.items()):
        conf = data["confidence"]
        correct = data["correct"]

        bins = np.linspace(0.5, 1.0, 16)

        ax.hist(
            conf[correct],
            bins=bins,
            alpha=0.6,
            label="Correct",
        )

        ax.hist(
            conf[~correct],
            bins=bins,
            alpha=0.6,
            label="Incorrect",
        )

        ax.set_title(name)
        ax.set_xlabel("Prediction Confidence")
        ax.set_ylabel("Number of MRI Slices")
        ax.legend()
        ax.grid(alpha=0.2)

    fig.suptitle("Confidence of Correct vs Incorrect Predictions")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / "confidence_distribution.png",
        dpi=200,
    )
    plt.close(fig)


def plot_coverage_accuracy(datasets):
    fig, ax = plt.subplots(figsize=(8, 5))

    thresholds = np.linspace(0.5, 0.99, 50)

    for name, data in datasets.items():
        coverages = []
        accuracies = []

        for threshold in thresholds:
            selected = data["confidence"] >= threshold

            coverages.append(float(selected.mean()))

            if selected.any():
                accuracies.append(
                    float(data["correct"][selected].mean())
                )
            else:
                accuracies.append(np.nan)

        ax.plot(
            np.array(coverages) * 100,
            np.array(accuracies) * 100,
            linewidth=2,
            marker="o",
            markersize=3,
            label=name,
        )

    ax.set_xlabel("Automated Coverage (%)")
    ax.set_ylabel("Accuracy of Accepted Predictions (%)")
    ax.set_title("Coverage vs Selective Accuracy")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "coverage_accuracy.png",
        dpi=200,
    )
    plt.close(fig)


def save_tables(datasets):
    with (OUTPUT / "confidence_summary.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "accuracy",
            "ece",
            "brier_score",
            "log_loss",
            "mean_correct_confidence",
            "mean_incorrect_confidence",
        ])

        for name, data in datasets.items():
            _, _, ece = reliability_bins(data)

            correct = data["correct"]
            conf = data["confidence"]

            writer.writerow([
                name,
                float(correct.mean()),
                ece,
                float(brier_score_loss(
                    data["y_true"],
                    data["p_ad"],
                )),
                float(log_loss(
                    data["y_true"],
                    data["p_ad"],
                    labels=[0, 1],
                )),
                float(conf[correct].mean())
                if correct.any() else "",
                float(conf[~correct].mean())
                if (~correct).any() else "",
            ])

    with (OUTPUT / "referral_thresholds.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "model",
            "confidence_threshold",
            "automated_coverage",
            "referral_rate",
            "accepted_accuracy",
            "accepted_errors",
            "accepted_AD_false_negatives",
        ])

        for name, data in datasets.items():
            for threshold in THRESHOLDS:
                selected = (
                    data["confidence"] >= threshold
                )

                coverage = float(selected.mean())

                accepted_accuracy = (
                    float(data["correct"][selected].mean())
                    if selected.any() else ""
                )

                accepted_errors = int(
                    ((~data["correct"]) & selected).sum()
                )

                # Actual AD=1 incorrectly predicted NC=0.
                accepted_false_negatives = int((
                    (data["y_true"] == 1)
                    & (data["y_pred"] == 0)
                    & selected
                ).sum())

                writer.writerow([
                    name,
                    threshold,
                    coverage,
                    1.0 - coverage,
                    accepted_accuracy,
                    accepted_errors,
                    accepted_false_negatives,
                ])


def main():
    datasets = {
        name: load_predictions(path)
        for name, path in INPUTS.items()
    }

    # Both models should have identical test labels/order.
    baseline = datasets["BaselineCNN"]["y_true"]
    convnext = datasets["ConvNeXt-Tiny"]["y_true"]

    if not np.array_equal(baseline, convnext):
        raise ValueError(
            "Baseline and ConvNeXt test-label sequences differ. "
            "Verify sample ordering before comparison."
        )

    plot_calibration(datasets)
    plot_confidence_distribution(datasets)
    plot_coverage_accuracy(datasets)
    save_tables(datasets)

    print("\nConfidence analysis completed.")
    print(f"Results saved to: {OUTPUT}")

    for name, data in datasets.items():
        _, _, ece = reliability_bins(data)
        brier = brier_score_loss(
            data["y_true"], data["p_ad"]
        )

        print(f"\n{name}")
        print(f"  Accuracy: {data['correct'].mean():.2%}")
        print(f"  ECE: {ece:.4f}")
        print(f"  Brier score: {brier:.4f}")


if __name__ == "__main__":
    main()

