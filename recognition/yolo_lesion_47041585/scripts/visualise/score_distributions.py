# matplotlib and numpy calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false
"""Plot the malignant score of every image, benign against malignant, one panel per condition.

The score is the highest malignant-class confidence among an image's detections (0 if it has
none), the quantity the routing policy compares with `tau_m`. A dashed line marks each
condition's recorded `tau_m` when a selection record is found. Where the two histograms
overlap, no threshold can separate the classes, which is what the AUC measures.

    uv run python scripts/visualise/score_distributions.py
    uv run python scripts/visualise/score_distributions.py --split test --record assets/selection.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from yolo_lesion_47041585.helpers.validation import arrays_for, cached_predictions, read_targets
from yolo_lesion_47041585.helpers.visualisation import use_report_style

# ---- settings -------------------------------------------------------------------------------
CONDITIONS = ("B0", "A0")
CLASS_NAMES = ("Benign", "Malignant")
CLASS_COLOURS = ("tab:green", "tab:red")
LOG_SCALE = True  # log-spaced bins from the confidence floor to 1; False uses 0 to 1
FLOOR = 0.001  # images with no malignant box (score 0) are drawn at this value
BINS = 25
ALPHA = 0.5
SHOW_THRESHOLD = True  # dashed line at the recorded tau_m
FIGSIZE = (11, 4.5)
DPI = 150
# ---------------------------------------------------------------------------------------------


def main() -> None:
    use_report_style()
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=47041585)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--record", type=Path, default=Path("assets/selection.json"), help="marks each condition's tau_m if the file exists")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/isic2018/yolo"))
    parser.add_argument("--cache-dir", type=Path, help="detections cache (default: artifacts/detection-cache/<split>)")
    parser.add_argument("--device", help="Ultralytics device, used only if detections are not cached")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    taus: dict[str, float] = {}
    if SHOW_THRESHOLD and args.record.is_file():
        record = json.loads(args.record.read_text())
        taus = {record[role]["condition"]: float(record[role]["tau_m"]) for role in ("candidate", "baseline")}
    cache_dir = args.cache_dir or Path("artifacts/detection-cache") / args.split
    targets = read_targets(args.data / "labels" / args.split)
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    true_class = np.array([t.class_id for t in targets])

    fig, axes = plt.subplots(1, len(args.conditions), figsize=FIGSIZE, sharey=True, squeeze=False)
    bins = np.logspace(np.log10(FLOOR), 0, BINS + 1) if LOG_SCALE else np.linspace(0, 1, BINS + 1)
    for ax, condition in zip(axes[0], args.conditions):
        run = f"{condition}-s{args.seed}"
        predictions = cached_predictions(args.runs_dir / run / "weights" / "best.pt", images, cache_dir, args.device)
        score = np.maximum(arrays_for(run, predictions, targets).score, FLOOR)
        for class_id, (name, colour) in enumerate(zip(CLASS_NAMES, CLASS_COLOURS)):
            here = score[true_class == class_id]
            label = f"{name} (n={len(here)})"
            if condition in taus:
                label += f", {np.mean(here >= taus[condition]):.0%} at or above tau_m"
            ax.hist(here, bins=bins, alpha=ALPHA, color=colour, label=label)
        if condition in taus:
            ax.axvline(taus[condition], color="black", linestyle="--", linewidth=1)
        if LOG_SCALE:
            ax.set_xscale("log")
        ax.set_title(f"{condition} (seed {args.seed})")
        ax.set_xlabel("Highest malignant confidence in the image")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Images")
    fig.suptitle(f"Malignant score by true class ({args.split} split; dashed line is the recorded tau_m)")
    output = args.output or Path("assets") / f"score-distributions-{args.split}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
