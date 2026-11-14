# matplotlib calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot validation mAP50:95 by epoch: the mean over seeds, with the range as a band.

Reads each run's results.csv from runs/<condition>-s<seed>/.

    uv run python scripts/visualise/training_curves.py
    uv run python scripts/visualise/training_curves.py --conditions B0 A0 O35
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.visualisation import use_report_style

# ---- settings -------------------------------------------------------------------------------
CONDITIONS = ("B0", "A0")
METRIC_COLUMN = "metrics/mAP50-95(B)"
Y_LABEL = "Validation mAP50:95 (mean over seeds)"
SHOW_RANGE = True  # shade the lowest to highest seed at each epoch
FIGSIZE = (8, 5)
DPI = 150
LINE_WIDTH = 2.0
TITLE = "Validation mAP50:95 by training epoch"
# ---------------------------------------------------------------------------------------------


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--output", type=Path, default=Path("assets/training-curves.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for condition in args.conditions:
        per_seed: list[list[float]] = []
        finished = re.compile(rf"^{re.escape(condition)}-s\d+$")  # not the "-pilot" smoke tests
        for folder in sorted(f for f in args.runs_dir.iterdir() if finished.match(f.name)):
            with (folder / "results.csv").open(newline="") as file:
                per_seed.append([float(row[METRIC_COLUMN]) for row in csv.DictReader(file)])
        if not per_seed:
            raise FileNotFoundError(f"no runs for {condition} in {args.runs_dir}")
        epochs = range(1, min(len(c) for c in per_seed) + 1)
        mean = [statistics.fmean(c[e - 1] for c in per_seed) for e in epochs]
        (line,) = ax.plot(epochs, mean, linewidth=LINE_WIDTH, label=f"{condition} ({len(per_seed)} seeds)")
        if SHOW_RANGE:
            low = [min(c[e - 1] for c in per_seed) for e in epochs]
            high = [max(c[e - 1] for c in per_seed) for e in epochs]
            ax.fill_between(epochs, low, high, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(TITLE)
    ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
