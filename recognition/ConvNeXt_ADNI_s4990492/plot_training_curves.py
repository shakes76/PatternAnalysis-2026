"""Plot recorded train/validation curves without loading images or models."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(run_dir):
    with (run_dir / "training_log.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty training log: {run_dir.name}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--convnext", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("figures/training_curves_20ep.png"),
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for column, (name, directory) in enumerate([
        ("SimpleCNN", args.baseline),
        ("ConvNeXt-Tiny", args.convnext),
    ]):
        rows = read_log(directory)
        epochs = [int(row["epoch"]) for row in rows]
        best = min(rows, key=lambda row: float(row["val_loss"]))
        best_epoch = int(best["epoch"])

        for line, (metric, ylabel) in enumerate([
            ("loss", "Cross-entropy loss"),
            ("accuracy", "Accuracy (%)"),
        ]):
            axis = axes[line, column]
            scale = 100 if metric == "accuracy" else 1

            for split, label in [
                ("train", "Training"), ("val", "Validation")
            ]:
                values = [
                    float(row[f"{split}_{metric}"]) * scale
                    for row in rows
                ]
                axis.plot(epochs, values, label=label)

            axis.axvline(
                best_epoch, color="gray", linestyle="--",
                label=f"Selected epoch: {best_epoch}",
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_title(name)
            axis.grid(alpha=0.25)
            axis.legend()

            if metric == "accuracy":
                axis.set_ylim(0, 100)

        print(
            f"{name}: selected epoch={best_epoch}, "
            f"val_loss={float(best['val_loss']):.6f}, "
            f"val_accuracy={float(best['val_accuracy']):.2%}"
        )

    fig.suptitle("Patient-independent training and validation — 20 epochs")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()