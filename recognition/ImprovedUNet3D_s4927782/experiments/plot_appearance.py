"""Plot all predeclared appearance conditions and every VAL patient's arm delta.

Reads only the anonymous review summary, never checkpoints, MRI or predictions.
This exploratory TRAIN/VAL follow-up does not replace the frozen v1 TEST result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

MODELS = ("seed42_reference", "seed43_reference", "gamma_only", "gamma_polarity")
CONDITIONS = ("clean", "gamma_0.7", "gamma_1.5", "polarity", "spatial_gain")
PATIENTS = ("B040", "J026", "K008", "K018", "M030", "S022")
MODEL_LABELS = ("Original seed 42", "Reference seed 43", "A: gamma only", "B: gamma + polarity")
CONDITION_LABELS = ("Clean", "Gamma 0.7", "Gamma 1.5", "Polarity", "Spatial gain")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value, lower, upper):
    if value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper,
            "Plot values must be finite numbers in their metric range, or null.")
    return float(value)


def validate_summary(summary):
    require(summary["experiment_id"] == "appearance_augmentation_v2" and summary["split"] == "validation",
            "Only the declared validation appearance summary is supported.")
    plot = summary["plot_data"]
    heatmap = plot["prostate_dice_heatmap"]
    require(tuple(heatmap["model_ids"]) == MODELS and tuple(heatmap["conditions"]) == CONDITIONS,
            "All four models and five fixed conditions must appear in their declared order.")
    values = heatmap["values"]
    require(len(values) == 4 and all(len(row) == 5 for row in values), "Heatmap must contain all 4 x 5 results.")
    values = [[scalar(v, 0, 1) for v in row] for row in values]
    rows = plot["patient_arm_deltas"]
    indexed = {row["patient"]: row for row in rows}
    require(len(rows) == len(indexed) == 6 and set(indexed) == set(PATIENTS),
            "Show all six unique validation patients, never selected patients or condition replicates.")
    deltas = [{"patient": patient, **{condition: scalar(indexed[patient][condition], -1, 1)
                                      for condition in ("clean", "polarity")}} for patient in PATIENTS]
    return values, deltas


def build_figure(summary):
    values, deltas = validate_summary(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 9})
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(7.1, 6.0), gridspec_kw={"height_ratios": [1, 1.25]})
    figure.subplots_adjust(left=.24, right=.94, top=.86, bottom=.24, hspace=.53)
    figure.suptitle("Exploratory validation appearance robustness", y=.975, fontsize=12)
    figure.text(.5, .925, "Four models × five conditions | six patients, 38 scans", ha="center", fontsize=9)
    array = np.array([[np.nan if v is None else v for v in row] for row in values])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    picture = top.imshow(np.ma.masked_invalid(array), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    top.set_xticks(range(5), CONDITION_LABELS)
    top.set_yticks(range(4), MODEL_LABELS)
    top.set_title("a  Patient-balanced prostate Dice — every condition", loc="left", pad=10)
    for row in range(4):
        for column in range(5):
            v = values[row][column]
            top.text(column, row, "NA" if v is None else f"{v:.3f}", ha="center", va="center",
                     color="black" if v is None or v >= .6 else "white", fontsize=9)
    colorbar = figure.colorbar(picture, ax=top, fraction=.035, pad=.035)
    colorbar.set_label("Dice", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    bottom.set_title("b  B − A for each patient; visits averaged within patient", loc="left", pad=10)
    bottom.axvline(0, color="#555555", linestyle="--", linewidth=.8)
    colors, markers = {"clean": "#1675a9", "polarity": "#cb6826"}, {"clean": "o", "polarity": "D"}
    defined = [row[k] for row in deltas for k in ("clean", "polarity") if row[k] is not None]
    extent = max([abs(v) for v in defined] + [.05])
    bottom.set_xlim(-extent * 1.3 - .025, extent * 1.3 + .025)
    shown = set()
    for index, row in enumerate(deltas):
        if row["clean"] is not None and row["polarity"] is not None:
            bottom.plot([row["clean"], row["polarity"]], [index - .14, index + .14], color="#bbbbbb", linewidth=.8, zorder=1)
        for condition, offset in (("clean", -.14), ("polarity", .14)):
            value = row[condition]
            if value is None:
                bottom.text(.99, index + offset, f"{condition}: NA", transform=bottom.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7.5, color=colors[condition])
                continue
            bottom.scatter(value, index + offset, s=22, marker=markers[condition], color=colors[condition],
                           label=condition.capitalize() if condition not in shown else None, zorder=3)
            shown.add(condition)
            bottom.annotate(f"{value:+.3f}", (value, index + offset), xytext=(5 if value >= 0 else -5, 0),
                            textcoords="offset points", ha="left" if value >= 0 else "right", va="center",
                            fontsize=7.8, color=colors[condition])
    bottom.set_yticks(range(6), PATIENTS)
    bottom.set_ylim(5.55, -.55)
    bottom.set_xlabel("Difference in prostate Dice (B − A)")
    bottom.grid(axis="x", alpha=.2)
    if shown:
        bottom.legend(loc="upper left", bbox_to_anchor=(0, -.26), ncol=2, frameon=False, fontsize=8)
    figure.text(.02, .04, "Image-only synthetic cache-space stress; conditions reuse the same six VAL patients.\n"
                "TRAIN/VAL follow-up after v1 TEST was seen. No new TEST or isolated causal-effect claim.",
                ha="left", va="bottom", fontsize=8)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    require(output.suffix.lower() == ".png" and not os.path.lexists(output), "Choose a new PNG output; existing files cannot be replaced.")
    source = args.summary.expanduser().resolve(strict=True)
    raw = source.read_bytes()
    summary = json.loads(raw)
    figure = build_figure(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, dpi=300, facecolor="white", metadata={"Description": "Exploratory VAL appearance review; input summary SHA256 " + hashlib.sha256(raw).hexdigest()})
        os.link(temporary, output)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Appearance plot saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
