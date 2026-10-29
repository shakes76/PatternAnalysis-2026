"""Plot aggregate, patient-level held-out test results without private records."""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
with (ROOT / "test_tradeoffs.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

colors = {"Accuracy": "#38658C", "AD recall": "#B75B48", "CN specificity": "#6B8D77"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.45, 0.9]})

models = [row["model"] for row in rows]
x = np.arange(len(rows))
width = 0.24
for offset, (label, key) in zip((-width, 0, width), (("Accuracy", "accuracy"), ("AD recall", "ad_recall"), ("CN specificity", "cn_specificity"))):
    values = [100 * float(row[key]) for row in rows]
    bars = axes[0].bar(x + offset, values, width, color=colors[label], label=label)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
axes[0].axhline(80, color="#777777", lw=0.9, ls="--", zorder=0)
axes[0].text(2.48, 81.5, "80% course target", ha="right", color="#555555", fontsize=8)
axes[0].set_ylim(0, 106)
axes[0].set_xticks(x, models)
axes[0].set_ylabel("Patients (%)")
axes[0].set_title("A  Primary threshold = 0.5", loc="left", fontweight="bold")
axes[0].legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14), fontsize=8)

model = rows[-1]
groups = ["AD misses\n(of 44)", "CN false alarms\n(of 92)"]
main = [int(model["main_ad_misses"]), int(model["main_cn_false_alarms"])]
supp = [int(model["supplement_ad_misses"]), int(model["supplement_cn_false_alarms"])]
x2 = np.arange(2)
for offset, values, label, color in ((-0.18, main, "0.5 threshold", "#38658C"), (0.18, supp, "Validation-selected threshold", "#B75B48")):
    bars = axes[1].bar(x2 + offset, values, 0.34, color=color, label=label)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.35, str(value), ha="center", va="bottom", fontsize=9)
axes[1].set_ylim(0, 25)
axes[1].set_xticks(x2, groups)
axes[1].set_ylabel("Patients (count)")
axes[1].set_title("B  ConvNeXt sensitivity trade-off", loc="left", fontweight="bold")
axes[1].legend(ncol=1, loc="upper left", bbox_to_anchor=(0, -0.14), fontsize=8)

fig.text(0.5, 0.005, "Held-out test: 136 patients (44 AD, 92 CN). Calibrated three-seed ensembles.", ha="center", fontsize=8, color="#555555")
fig.subplots_adjust(bottom=0.25, left=0.075, right=0.98, top=0.88, wspace=0.30)
fig.savefig(ROOT / "test_tradeoffs.png", dpi=350, facecolor="white")
fig.savefig(ROOT / "test_tradeoffs.svg", facecolor="white")
fig.savefig(ROOT / "test_tradeoffs.pdf", facecolor="white")
svg = ROOT / "test_tradeoffs.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
