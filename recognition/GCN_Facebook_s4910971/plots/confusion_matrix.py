"""Plot test confusion matrices."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Saved runs and split.
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
RUNS = {
    "MLP": ROOT / "training_results" / "mlp_20261030_000036",
    "GCN": ROOT / "training_results" / "gcn_20261030_002758",
}
SPLIT = "test"
CLASSES = ["company", "government", "politician", "tvshow"]


fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, name in zip(axes, RUNS):
    # Read saved predictions.
    true = []
    pred = []
    path = RUNS[name] / "prediction" / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == SPLIT:
                true.append(row["true"])
                pred.append(row["pred"])

    # Counts and percentages within each true class.
    counts = confusion_matrix(true, pred, labels=CLASSES)
    shares = confusion_matrix(true, pred, labels=CLASSES, normalize="true")

    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            label = f"{counts[i, j]}\n{shares[i, j]:.1%}"
            ax.text(j, i, label, ha="center", va="center")

    ax.set_title(name)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")


fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "confusion_matrices.png")
plt.close(fig)