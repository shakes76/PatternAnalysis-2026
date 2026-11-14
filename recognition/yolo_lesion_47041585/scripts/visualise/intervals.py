# matplotlib calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot differences with their patient-level 95% intervals, so "no clear difference" is visible.

  validation  each condition minus the reference condition, for AUC and for sensitivity at
              the 30% limit (from artifacts/scores.json, written by scripts/score.py)
  test        candidate minus baseline at the frozen operating point, for the rates in
              TEST_METRICS (from assets/test-results.json, written by scripts/final_test.py)

A dot is the difference and the bar is its 95% interval; a bar that crosses zero means the
difference is not clearly different from none.

    uv run python scripts/visualise/intervals.py --source validation
    uv run python scripts/visualise/intervals.py --source test
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.visualisation import use_report_style
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# ---- settings -------------------------------------------------------------------------------
TEST_METRICS = (  # (key in test-results.json, label)
    ("sensitivity", "malignant detected and alerted"),
    ("benign_urgent", "benign routed urgent (burden)"),
    ("benign_clear", "benign routed clear"),
    ("malignant_clear", "malignant routed clear"),
)
VALIDATION_MEASURES = (("auc", "AUC", False), ("s30", "Sensitivity within 30% escalation (points)", True))
DOT_COLOUR = "tab:blue"
ZERO_LINE_COLOUR = "black"
FIGSIZE = (9, 4.5)
DPI = 150
MARKER_SIZE = 6
# ---------------------------------------------------------------------------------------------


def validation_plot(scores: Path) -> tuple[Figure, str]:
    extra = json.loads(scores.read_text())["trade_off"]
    reference, differences = extra["reference"], extra["differences_vs_reference"]
    conditions = list(differences)
    fig, axes = plt.subplots(1, len(VALIDATION_MEASURES), figsize=FIGSIZE, sharey=True)
    for ax, (key, title, points) in zip(axes, VALIDATION_MEASURES):
        scale = 100 if points else 1
        for row, condition in enumerate(conditions):
            d = differences[condition][key]
            ax.errorbar(
                scale * d["difference"], row, xerr=[[scale * (d["difference"] - d["low"])], [scale * (d["high"] - d["difference"])]],
                fmt="o", color=DOT_COLOUR, capsize=3, markersize=MARKER_SIZE,
            )  # fmt: skip
        ax.axvline(0, color=ZERO_LINE_COLOUR, linewidth=1)
        ax.set_title(title)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    return fig, f"Validation: each condition minus {reference}, with patient-level 95% intervals"


def test_plot(results: Path) -> tuple[Figure, str]:
    data = json.loads(results.read_text())
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for row, (key, _) in enumerate(TEST_METRICS):
        low, high = data["intervals"][f"diff:{key}"]
        diff = data["observed"]["candidate"][key] - data["observed"]["baseline"][key]
        ax.errorbar(100 * diff, row, xerr=[[100 * (diff - low)], [100 * (high - diff)]], fmt="o", color=DOT_COLOUR, capsize=3, markersize=MARKER_SIZE)
    ax.axvline(0, color=ZERO_LINE_COLOUR, linewidth=1)
    ax.set_yticks(range(len(TEST_METRICS)))
    ax.set_yticklabels([label for _, label in TEST_METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Candidate minus baseline (percentage points)")
    return fig, f"{data['split'].capitalize()} split: candidate minus baseline at the frozen points, 95% intervals"


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", choices=("validation", "test"), required=True)
    parser.add_argument("--scores", type=Path, default=Path("artifacts/scores.json"))
    parser.add_argument("--test-results", type=Path, default=Path("assets/test-results.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fig, title = validation_plot(args.scores) if args.source == "validation" else test_plot(args.test_results)
    fig.suptitle(title)
    output = args.output or Path("assets") / f"intervals-{args.source}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
