# matplotlib calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot how much of the image each annotated lesion box covers, by class.

Reads artifacts/isic2018/split_manifest.csv (written by the data audit). The box area is the
box width times height, both as a fraction of the image.

    uv run python scripts/visualise/box_areas.py
    uv run python scripts/visualise/box_areas.py --split train
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.visualisation import use_report_style

# ---- settings -------------------------------------------------------------------------------
CLASS_NAMES = ("Benign", "Malignant")
CLASS_COLOURS = ("tab:green", "tab:red")
BINS = 20
ALPHA = 0.5
SHOW_MEDIANS = True
FIGSIZE = (7, 4.5)
DPI = 150
TITLE = "Lesion box area as a share of the image"
# ---------------------------------------------------------------------------------------------


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/isic2018/split_manifest.csv"))
    parser.add_argument("--split", choices=("train", "val", "test"), help="default: every retained image")
    parser.add_argument("--output", type=Path, default=Path("assets/box-areas.png"))
    args = parser.parse_args()

    areas: list[list[float]] = [[] for _ in CLASS_NAMES]
    with args.manifest.open(newline="") as file:
        for row in csv.DictReader(file):
            if args.split is None or row["split"] == args.split:
                areas[int(row["class_id"])].append(float(row["width"]) * float(row["height"]))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, colour, values in zip(CLASS_NAMES, CLASS_COLOURS, areas):
        ax.hist(values, bins=BINS, range=(0, 1), alpha=ALPHA, color=colour, label=f"{name} ({len(values)} images)")
        if SHOW_MEDIANS and values:
            ax.axvline(statistics.median(values), color=colour, linestyle=":", linewidth=1)
    ax.set_xlabel("Box area as a fraction of the image")
    ax.set_ylabel("Images")
    ax.set_title(TITLE + (f" ({args.split} split)" if args.split else ""))
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {args.output}")
    for name, values in zip(CLASS_NAMES, areas):
        print(f"  {name}: median {statistics.median(values):.2f}, {sum(v > 0.8 for v in values) / len(values):.1%} cover more than 80% of the image")


if __name__ == "__main__":
    main()
