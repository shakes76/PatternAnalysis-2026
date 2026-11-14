# matplotlib calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot sensitivity against biopsy burden for a baseline and a candidate.

Reads artifacts/tradeoff.csv (written by scripts/score.py): the rates at every threshold for
each condition and seed. The primary seed is drawn thick and the other seeds thin. Optional:
mark a chosen tau_m, and add the frozen test results from assets/test-results.json.

    uv run python scripts/visualise/tradeoff.py
    uv run python scripts/visualise/tradeoff.py --tau-m 0.07 --test-results assets/test-results.json
    uv run python scripts/visualise/tradeoff.py --csv artifacts/cls2/tradeoff.csv --baseline A0 \\
        --candidate A0-cls2 --candidate-colour tab:purple --mark-limits --output assets/tradeoff-cls2.png
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.visualisation import use_report_style

# ---- settings -------------------------------------------------------------------------------
BASELINE = "B0"
CANDIDATE = "A0"
BASELINE_COLOUR = "tab:orange"
CANDIDATE_COLOUR = "tab:blue"
FIGSIZE = (7, 6)
DPI = 150
LINE_WIDTH = 2.0
OTHER_SEED_ALPHA = 0.3
SHOW_OTHER_SEEDS = True  # thin lines for the seeds that are not the primary one
SHOW_TARGET_REGION = True  # shade 90% sensitivity at a burden of 30% or less
MAX_BURDEN = 0.30
MIN_SENSITIVITY = 0.90
SHOW_CHANCE_LINE = True
BURDEN_LIMITS = (0.30, 0.40, 0.50)  # the limits --mark-limits reads off, as in the score.py tables
MARK_POINTS = True  # number the lowest burden at 90% sensitivity, the best point within 30% burden, and --tau-m
TITLE = "Sensitivity against biopsy burden (validation)"
X_LABEL = "Benign images routed urgent (biopsy burden)"
Y_LABEL = "Malignant lesions detected and alerted (sensitivity)"
# ---------------------------------------------------------------------------------------------

Curve = tuple[list[float], list[float], list[float]]  # burden, sensitivity, tau_m


def read_curves(path: Path) -> dict[tuple[str, int], Curve]:
    curves: defaultdict[tuple[str, int], Curve] = defaultdict(lambda: ([], [], []))
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            burden, sensitivity, tau = curves[(row["condition"], int(row["seed"]))]
            burden.append(float(row["benign_urgent"]))
            sensitivity.append(float(row["sensitivity"]))
            tau.append(float(row["tau_m"]))
    return dict(curves)


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("artifacts/tradeoff.csv"))
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", default=CANDIDATE)
    parser.add_argument("--primary-seed", type=int, help="seed drawn thick (default: the smallest in the file)")
    parser.add_argument("--tau-m", type=float, help="mark this candidate threshold on the curve")
    parser.add_argument("--test-results", type=Path, help="test-results.json: add the frozen test points")
    parser.add_argument("--baseline-colour", default=BASELINE_COLOUR)
    parser.add_argument("--candidate-colour", default=CANDIDATE_COLOUR)
    parser.add_argument("--mark-limits", action="store_true", help="mark both curves at each burden limit instead of the numbered points")
    parser.add_argument("--output", type=Path, default=Path("assets/tradeoff.png"))
    args = parser.parse_args()

    curves = read_curves(args.csv)
    seeds = sorted({seed for _, seed in curves})
    primary = args.primary_seed if args.primary_seed is not None else seeds[0]
    colours = {args.baseline: args.baseline_colour, args.candidate: args.candidate_colour}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if SHOW_TARGET_REGION:
        ax.fill_between([0, MAX_BURDEN], MIN_SENSITIVITY, 1, color="tab:blue", alpha=0.1, label="target region")
    if SHOW_CHANCE_LINE:
        ax.plot([0, 1], [0, 1], color="lightgray", linewidth=1, label="chance")
    for condition, colour in colours.items():
        for (name, seed), (burden, sensitivity, _) in curves.items():
            if name != condition:
                continue
            if seed == primary:
                ax.plot(burden, sensitivity, color=colour, linewidth=LINE_WIDTH, label=f"{condition} (seed {seed})")
            elif SHOW_OTHER_SEEDS:
                ax.plot(burden, sensitivity, color=colour, linewidth=1, alpha=OTHER_SEED_ALPHA)

    if args.mark_limits:
        for limit in BURDEN_LIMITS:
            ax.axvline(limit, color="lightgray", linestyle=":", linewidth=1)
        for (condition, colour), (offset, align) in zip(colours.items(), (((6, -13), "left"), ((0, 8), "center"))):
            burden, sensitivity, _ = curves[(condition, primary)]
            for limit in BURDEN_LIMITS:
                within = [i for i, b in enumerate(burden) if b <= limit + 1e-9]
                i = max(within, key=lambda i: sensitivity[i])  # the first of any tie, as in score.py
                ax.plot(burden[i], sensitivity[i], "o", color=colour, markeredgecolor="black")
                ax.annotate(f"{sensitivity[i]:.0%}", (burden[i], sensitivity[i]), textcoords="offset points", xytext=offset, ha=align, fontsize=8)

    if MARK_POINTS and not args.mark_limits:
        burden, sensitivity, tau = curves[(args.candidate, primary)]
        marks: list[tuple[str, int]] = []
        reached = [i for i, s in enumerate(sensitivity) if s >= MIN_SENSITIVITY - 1e-9]
        if reached:
            marks.append(("lowest burden at 90% sensitivity", min(reached, key=lambda i: burden[i])))
        within = [i for i, b in enumerate(burden) if b <= MAX_BURDEN + 1e-9]
        if within:
            marks.append(("best sensitivity within 30% burden", max(within, key=lambda i: sensitivity[i])))
        if args.tau_m is not None:
            marks.append((f"chosen point, tau_m {args.tau_m}", min(range(len(tau)), key=lambda i: abs(tau[i] - args.tau_m))))
        for number, (label, i) in enumerate(marks, start=1):
            ax.plot(burden[i], sensitivity[i], "o", color="black")
            ax.annotate(str(number), (burden[i], sensitivity[i]), textcoords="offset points", xytext=(6, -12))
            ax.plot([], [], "o", color="black", label=f"{number}: {label}")

    if args.test_results:
        result = json.loads(args.test_results.read_text())
        for role, condition in (("candidate", args.candidate), ("baseline", args.baseline)):
            x, y = result["observed"][role]["benign_urgent"], result["observed"][role]["sensitivity"]
            ix, iy = result["intervals"].get(f"{role}:benign_urgent"), result["intervals"].get(f"{role}:sensitivity")
            ax.errorbar(
                x, y, xerr=[[x - ix[0]], [ix[1] - x]] if ix else None, yerr=[[y - iy[0]], [iy[1] - y]] if iy else None,
                fmt="s", markerfacecolor="white", color=colours[condition], capsize=3,
                label=f"{condition} {result['split']} result (frozen, 95% interval)",
            )  # fmt: skip

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(TITLE)
    ax.legend(loc="lower right", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
