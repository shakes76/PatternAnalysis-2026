# matplotlib calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot each condition's validation scores, one dot per training seed.

Reads artifacts/scores.json (written by scripts/score.py).

    uv run python scripts/visualise/conditions.py
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.visualisation import use_report_style

# ---- settings -------------------------------------------------------------------------------
HIGHLIGHT = "A0"  # drawn in HIGHLIGHT_COLOUR
HIGHLIGHT_COLOUR = "tab:blue"
OTHER_COLOUR = "gray"
CONDITIONS: tuple[str, ...] | None = None  # None: every condition in the file, in file order
MEASURES = (  # (key in scores.json, panel title, shown as a percentage)
    ("auc", "AUC", False),
    ("s30", "Sensitivity within 30% benign escalation", True),
    ("map50_95", "mAP50:95", False),
)
TARGET_LINES = {"s30": 0.90}  # a vertical reference line per measure
FIGSIZE = (13, 4.5)
DPI = 150
MARKER_SIZE = 40
TITLE = "Validation results by condition (one dot per seed, line is the mean)"
# ---------------------------------------------------------------------------------------------


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--scores", type=Path, default=Path("artifacts/scores.json"))
    parser.add_argument("--highlight", default=HIGHLIGHT)
    parser.add_argument("--output", type=Path, default=Path("assets/results-by-condition.png"))
    args = parser.parse_args()

    runs = json.loads(args.scores.read_text())["runs"]
    conditions = list(CONDITIONS or dict.fromkeys(r["condition"] for r in runs))

    fig, axes = plt.subplots(1, len(MEASURES), figsize=FIGSIZE, sharey=True)
    for ax, (key, title, percent) in zip(axes, MEASURES):
        for row, condition in enumerate(conditions):
            values = [r[key] for r in runs if r["condition"] == condition and r[key] is not None]
            colour = HIGHLIGHT_COLOUR if condition == args.highlight else OTHER_COLOUR
            ax.scatter(values, [row] * len(values), s=MARKER_SIZE, color=colour, alpha=0.8)
            if values:
                mean = statistics.fmean(values)
                ax.plot([mean, mean], [row - 0.3, row + 0.3], color=colour)
        if key in TARGET_LINES:
            ax.axvline(TARGET_LINES[key], color="black", linewidth=1, linestyle="--")
        ax.set_title(title)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.invert_yaxis()
        if percent:
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle(TITLE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
