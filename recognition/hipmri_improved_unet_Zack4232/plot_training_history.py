"""Plot recorded training losses and validation metrics without running inference."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty history: {path}")
    keys = ("epoch", "train_ce", "validation_ce", "train_soft_dice_loss",
            "validation_soft_dice_loss", "validation_prostate_dice")
    data = {key: [float(row[key]) for row in rows] for key in keys}
    if not all(math.isfinite(x) for values in data.values() for x in values):
        raise ValueError(f"Non-finite history: {path}")
    if data["epoch"] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Non-consecutive epochs: {path}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--improved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, (name, path) in enumerate((("Standard U-Net", args.standard),
                                      ("Improved U-Net", args.improved))):
        data = read_history(path)
        epochs = data["epoch"]
        for col, (key, title) in enumerate((("ce", "Cross-entropy"),
                                           ("soft_dice_loss", "Soft Dice loss"))):
            ax = axes[row, col]
            ax.plot(epochs, data[f"train_{key}"], label="Training")
            ax.plot(epochs, data[f"validation_{key}"], label="Validation")
            ax.set_title(f"{name}: {title}")
            ax.set_ylabel("Loss")
            ax.set_ylim(bottom=0)
            ax.legend()
        ax = axes[row, 2]
        scores = data["validation_prostate_dice"]
        best = max(range(len(scores)), key=scores.__getitem__)
        ax.plot(epochs, scores, color="tab:green", label="Validation")
        ax.scatter([epochs[best]], [scores[best]], color="black", zorder=3,
                   label=f"Selected epoch {int(epochs[best])}")
        ax.set_title(f"{name}: prostate Dice")
        ax.set_ylabel("Equal mean over GT-positive slices")
        ax.set_ylim(0, 1)
        ax.legend()
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.25)
    fig.suptitle("Recorded CE + Dice experiments (seed 42); no test metrics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
