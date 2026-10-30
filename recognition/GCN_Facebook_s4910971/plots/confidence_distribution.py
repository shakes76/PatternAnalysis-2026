"""Plot confidence percentages for correct and wrong predictions."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
RUNS = {
    "MLP": ROOT / "training_results" / "mlp_20261030_000036",
    "GCN": ROOT / "training_results" / "gcn_20261030_002758",
}
SPLIT = "test"
CLASSES = ["company", "government", "politician", "tvshow"]

BINS = [i / 20 for i in range(5, 21)]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

for ax, name in zip(axes, RUNS):
    correct = []
    wrong = []
    path = RUNS[name] / "prediction" / "predictions.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == SPLIT:
                scores = []
                for label in CLASSES:
                    scores.append(float(row[f"p_{label}"]))
                confidence = max(scores)

                if row["true"] == row["pred"]:
                    correct.append(confidence)
                else:
                    wrong.append(confidence)


    # Normalise correct and wrong groups separately.
    ax.hist(correct, bins=BINS, weights=[1 / len(correct)] * len(correct),
            alpha=0.5, label="Correct")
    ax.hist(wrong, bins=BINS, weights=[1 / len(wrong)] * len(wrong),
            alpha=0.5, label="Wrong")

    ax.set_title(name)
    ax.set_xlabel("Maximum softmax score")
    ax.set_ylabel("Within-group percentage")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()


fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "confidence_distribution.png")
plt.close(fig)