"""
This script was produced with the help of AI.

Plot training and validation loss and accuracy for the CNN and GFNet runs.

Rows: models. Columns: loss, accuracy. Accuracy is slice-level for both curves.

Usage (from the project folder):
    python scripts/plot_curves.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "CNN": "cnn_20261026_053134",
    "GFNET": "gfnet_20261026_051241",
}
RUNS_DIR = Path("runs")
OUT = Path("images/training_curves_run_1.png")


def load_history(run):
    with open(RUNS_DIR / run / "history.csv") as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return [float(r[key]) for r in rows]


fig, axes = plt.subplots(len(RUNS), 2, figsize=(10, 4 * len(RUNS)))

for row, (name, run) in enumerate(RUNS.items()):
    h = load_history(run)
    epochs = column(h, "epoch")

    ax = axes[row, 0]
    ax.plot(epochs, column(h, "train_loss"), label="Training")
    ax.plot(epochs, column(h, "val_loss"), label="Validation")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[row, 1]
    ax.plot(epochs, column(h, "train_acc"), label="Training")
    ax.plot(epochs, column(h, "val_slice_acc"), label="Validation")
    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
