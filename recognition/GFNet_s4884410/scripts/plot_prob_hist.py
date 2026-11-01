"""
This script was produced with the help of AI.

Histograms of per-slice AD probabilities on the test set, from evaluate.py's output.

Rows: models. Columns: truly AD slices, truly NC slices, all slices.
The dashed line marks the 0.5 threshold.

Usage (from the project folder):
    python scripts/plot_prob_hist.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("results/eval")
MODELS = {"CNN": "cnn", "GFNET": "gfnet"}
SPLIT = "test"
OUT = Path("images") / f"{SPLIT}_slice_prob_hist.png"
BINS = np.linspace(0, 1, 41)  # 40 bins of width 0.025


def load(model):
    with open(EVAL_DIR / model / f"{SPLIT}_slice.csv") as f:
        rows = list(csv.DictReader(f))
    prob = np.array([float(r["prob_ad"]) for r in rows])
    label = np.array([int(r["label"]) for r in rows])
    return prob, label


fig, axes = plt.subplots(len(MODELS), 3, figsize=(12, 4 * len(MODELS)))

for row, (name, model) in enumerate(MODELS.items()):
    prob, label = load(model)
    groups = {"AD": prob[label == 1], "NC": prob[label == 0], "All": prob}
    for col, (group, p) in enumerate(groups.items()):
        ax = axes[row, col]
        ax.hist(p, bins=BINS, color=f"C{col}")
        ax.axvline(0.5, color="black", linestyle="--")
        ax.set_title(f"{name} - {group}")
        ax.set_xlabel("Predicted AD probability")
        ax.set_ylabel("Slices")

fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"Saved {OUT}")
