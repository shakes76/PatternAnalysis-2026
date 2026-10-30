"""Plot training loss and validation prostate Dice from an experiment."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.history.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("History contains no epoch records.")

    epochs = [int(row["epoch"]) for row in rows]
    train_ce = [float(row["train_ce"]) for row in rows]
    val_ce = [float(row["validation_ce"]) for row in rows]
    dice = [float(row["validation_prostate_dice"]) for row in rows]
    best_index = max(range(len(rows)), key=lambda i: dice[i])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(epochs, train_ce, "o-", label="Training")
    axes[0].plot(epochs, val_ce, "o-", label="Validation")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, dice, "o-", label="Validation prostate Dice")
    axes[1].scatter(
        epochs[best_index], dice[best_index],
        color="red", zorder=3, label="Best epoch",
    )
    axes[1].set_title("Ground-truth-positive slices only")
    axes[1].set_ylabel("Mean slice Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.grid(alpha=0.3)

    fig.suptitle("Standard U-Net training trial - no test evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
