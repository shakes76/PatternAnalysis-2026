
"""
COMP3710 — Training history visualizations.

Reads history.csv files from checkpoints/ and creates:
1. Training vs validation loss curves
2. Training vs validation accuracy curves
3. Validation accuracy comparison
4. Experiment summary CSV

No GPU or model checkpoint loading required.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = {
    "Baseline CNN": "checkpoints/baseline/history.csv",
    "ConvNeXt (LR 1e-4)": "checkpoints/convnext/history.csv",
    "ConvNeXt (LR 1e-5)": (
        "checkpoints/exp2_lr1e-5/convnext/history.csv"
    ),
    "ConvNeXt (Frozen)": (
        "checkpoints/exp3_frozen/convnext/history.csv"
    ),
    "ConvNeXt (Partial)": (
        "checkpoints/exp4_partial/convnext/history.csv"
    ),
}


def load_history(path):
    if not path.is_file():
        print(f"WARNING: Missing history file: {path}")
        return None

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print(f"WARNING: Empty history file: {path}")
        return None

    columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    for column in columns:
        if column not in rows[0]:
            raise ValueError(
                f"{path}: missing column '{column}'"
            )

    result = {}

    for column in columns:
        result[column] = np.array(
            [float(row[column]) for row in rows]
        )

    return result


def save_learning_curves(name, history, output_dir):
    """Save loss and accuracy figures for one experiment."""

    epochs = history["epoch"]
    slug = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )

    for metric, ylabel, filename in [
        ("loss", "Cross-Entropy Loss", "loss"),
        ("accuracy", "Accuracy (%)", "accuracy"),
    ]:
        train = history[f"train_{metric}"].copy()
        val = history[f"val_{metric}"].copy()

        if metric == "accuracy":
            # Training history stores accuracy as fraction.
            # Allow percentage-formatted values as well.
            if max(np.max(train), np.max(val)) <= 1.0:
                train *= 100
                val *= 100

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            epochs, train,
            marker="o",
            label="Training",
            linewidth=2,
        )

        ax.plot(
            epochs, val,
            marker="s",
            label="Validation",
            linewidth=2,
        )

        # Checkpoint chosen using minimum validation loss.
        best_idx = int(np.argmin(history["val_loss"]))

        ax.axvline(
            epochs[best_idx],
            linestyle="--",
            color="gray",
            alpha=0.7,
            label=f"Best checkpoint: epoch {int(epochs[best_idx])}",
        )

        ax.set_title(f"{name} — {ylabel}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xticks(epochs)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            output_dir / f"{slug}_{filename}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)


def plot_validation_comparison(histories, output_dir):
    """Compare validation accuracy across all experiments."""

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, history in histories.items():
        accuracy = history["val_accuracy"].copy()

        if np.max(accuracy) <= 1.0:
            accuracy *= 100

        ax.plot(
            history["epoch"],
            accuracy,
            marker="o",
            linewidth=2,
            label=name,
        )

    ax.set_title("Validation Accuracy Across Experiments")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig.savefig(
        output_dir / "validation_accuracy_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_summary(histories, output_dir):
    """Summarize best checkpoints selected by validation loss."""

    summary_path = output_dir / "experiment_summary.csv"

    with summary_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "experiment",
            "epochs_completed",
            "best_checkpoint_epoch",
            "best_val_loss",
            "val_accuracy_at_best_checkpoint",
            "highest_val_accuracy",
            "final_train_accuracy",
        ])

        for name, history in histories.items():
            idx = int(np.argmin(history["val_loss"]))

            val_acc = history["val_accuracy"]
            train_acc = history["train_accuracy"]

            if max(np.max(val_acc), np.max(train_acc)) <= 1:
                val_acc = val_acc * 100
                train_acc = train_acc * 100

            writer.writerow([
                name,
                len(history["epoch"]),
                int(history["epoch"][idx]),
                float(history["val_loss"][idx]),
                float(val_acc[idx]),
                float(np.max(val_acc)),
                float(train_acc[-1]),
            ])

    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/training_curves"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    histories = {}

    for name, path_string in EXPERIMENTS.items():
        path = Path(path_string)
        history = load_history(path)

        if history is None:
            continue

        histories[name] = history

        save_learning_curves(
            name,
            history,
            args.output_dir,
        )

        print(
            f"Processed {name}: "
            f"{len(history['epoch'])} epochs"
        )

    if not histories:
        raise RuntimeError(
            "No training history CSV files found."
        )

    plot_validation_comparison(
        histories,
        args.output_dir,
    )

    save_summary(
        histories,
        args.output_dir,
    )

    print(f"\nCompleted: {len(histories)} experiments")
    print(f"Figures saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

