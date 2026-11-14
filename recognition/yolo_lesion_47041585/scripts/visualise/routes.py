# matplotlib calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot what the routing policy does with each class: urgent, monitor or clear.

Reads the per-image outcomes written by scripts/final_test.py to artifacts/routes/. One bar
per model and true class, split into the share routed clear, monitor and urgent, with counts.

    uv run python scripts/visualise/routes.py --split test
    uv run python scripts/visualise/routes.py --split val       # after a dry run
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.visualisation import use_report_style

# ---- settings -------------------------------------------------------------------------------
BASELINE = "B0"
CANDIDATE = "A0"
CLASS_NAMES = ("benign", "malignant")
ROUTES = ("clear", "monitor", "urgent")
ROUTE_COLOURS = {"clear": "tab:green", "monitor": "gold", "urgent": "tab:red"}
SHOW_COUNTS = True  # write the image count inside each segment
FIGSIZE = (8, 4.5)
DPI = 150
BAR_HEIGHT = 0.6
TITLE = "How each true class is routed"
# ---------------------------------------------------------------------------------------------


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--routes-dir", type=Path, default=Path("artifacts/routes"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bars: list[tuple[str, Counter[str]]] = []
    for condition in (args.baseline, args.candidate):
        counts = {name: Counter[str]() for name in CLASS_NAMES}
        with (args.routes_dir / f"{args.split}-{condition}.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                counts[CLASS_NAMES[int(row["true_class"])]][row["route"]] += 1
        bars += [(f"{condition} {name}", counts[name]) for name in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for row, (_, counts) in enumerate(bars):
        total = sum(counts.values())
        left = 0.0
        for route in ROUTES:
            share = counts[route] / total
            ax.barh(row, share, left=left, height=BAR_HEIGHT, color=ROUTE_COLOURS[route], label=route if row == 0 else None)
            if SHOW_COUNTS and counts[route]:
                ax.text(left + share / 2, row, str(counts[route]), ha="center", va="center", fontsize=9)
            left += share
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([label for label, _ in bars])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(f"Share of images ({args.split} split)")
    ax.set_title(TITLE)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    output = args.output or Path("assets") / f"routes-{args.split}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
