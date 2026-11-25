"""
compare.py -- Aggregate predict.py results across random seeds and models.

    python compare.py --eval-dir eval --out images

Reads every eval/<model>_s<seed>/eval.json, reports each metric as
mean +/- standard deviation over seeds, and writes:

    comparison.md              markdown tables ready to paste into README.md
    comparison_position.png    prostate Dice across the prostate extent, both models
    comparison_per_class.png   per-class Dice, both models

Reporting the seed-to-seed spread is what makes the comparison honest: a gap
between models smaller than the spread between seeds is not a real difference.
"""
import argparse
import glob
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
MODEL_ORDER = ["baseline", "improved", "improved3d", "improved3d_ds2"]
LABELS = {"baseline": "2D U-Net (baseline)", "improved": "2D Improved U-Net", "improved3d": "3D Improved U-Net",
          "improved3d_ds2": "3D Improved U-Net (half resolution)"}

# (row label, function extracting the value from one eval.json, decimals)
ACCURACY_ROWS = [
    ("Prostate Dice (all slices)", lambda r: r["per_class"]["prostate"]["dice"], 3),
    ("Prostate IoU", lambda r: r["per_class"]["prostate"]["iou"], 3),
    ("Prostate Dice: central slices", lambda r: r["prostate_by_stratum"]["central"]["dice"], 3),
    ("Prostate Dice: edge slices", lambda r: r["prostate_by_stratum"]["edge"]["dice"], 3),
    ("Prostate boundary F-score (2 px)", lambda r: r["prostate_boundary"]["f_score"], 3),
    ("Under-contouring (% of prostate missed)", lambda r: r["clinical_errors"]["under_contouring_pct"], 1),
    ("Leak into bladder (% of prostate area)", lambda r: r["clinical_errors"]["leak_into_bladder_pct"], 2),
    ("Leak into rectum (% of prostate area)", lambda r: r["clinical_errors"]["leak_into_rectum_pct"], 2),
    ("False-positive prostate-free slices (%)", lambda r: r["clinical_errors"]["false_positive_slice_rate_pct"], 1),
]
REVIEW_ROWS = [
    ("Slices flagged for manual review (%)", lambda r: r["review_band"]["flagged_slices_pct"], 1),
    ("Missed prostate area caught by the flag (%)", lambda r: r["review_band"]["missed_prostate_px_caught_pct"], 1),
    ("False-positive area caught by the flag (%)", lambda r: r["review_band"]["false_positive_px_caught_pct"], 1),
    ("Whole-slice omissions (count)", lambda r: r["review_band"]["omitted_slices"], 1),
    ("Whole-slice omissions caught (%)", lambda r: r["review_band"]["omitted_slices_caught_pct"], 1),
    ("Prostate Dice on auto-drafted (unflagged) slices", lambda r: r["review_band"]["prostate_dice_auto_drafted"], 3),
]
RESOURCE_ROWS = [
    ("Parameters (M)", lambda r: r["resources"]["parameters"] / 1e6, 2),
    ("Peak training VRAM (GB)", lambda r: r["training"]["peak_train_vram_gb"], 2),
    ("Training time per epoch (s)", lambda r: r["training"]["mean_epoch_time_s"], 1),
    ("Total training time (min)", lambda r: r["training"]["total_train_time_s"] / 60, 1),
    ("Inference per slice (ms; 3D: volume time / slices)", lambda r: r["resources"]["inference_ms_per_slice"], 2),
    ("Inference per scan (s)", lambda r: r["resources"]["inference_s_per_scan"], 2),
    ("Peak inference VRAM (GB)", lambda r: r["resources"]["peak_inference_vram_gb"], 2),
    ("Latency for one input, batch of 1 (ms; 3D: one volume)",
     lambda r: r["training"].get("inference_ms_per_slice", r["training"].get("inference_ms_per_volume")), 2),
]


def load_runs(eval_dir):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(eval_dir, "*", "eval.json"))):
        with open(path) as f:
            r = json.load(f)
        runs[r.get("variant", r["model"])].append(r)
    return runs


def stat(runs, fn):
    vals = []
    for r in runs:
        try:
            v = fn(r)
        except (KeyError, TypeError):
            continue
        if v is not None and not np.isnan(v):
            vals.append(v)
    # Sample standard deviation (ddof=1): the seeds are a sample of possible training runs.
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return (float(np.mean(vals)), sd, len(vals)) if vals else (float("nan"), float("nan"), 0)


def table(runs, rows, models):
    header = "| Metric | " + " | ".join(LABELS[m] for m in models) + " |"
    lines = [header, "|---|" + "---|" * len(models)]
    for label, fn, dec in rows:
        cells = []
        for m in models:
            mean, sd, n = stat(runs[m], fn)
            cells.append("n/a" if n == 0 else (f"{mean:.{dec}f} ± {sd:.{dec}f}" if n > 1 else f"{mean:.{dec}f}"))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", default="eval")
    p.add_argument("--out", default="images")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.eval_dir)
    models = [m for m in MODEL_ORDER if m in runs]
    if not models:
        raise SystemExit(f"No eval.json files found under {args.eval_dir}/")
    seeds = {m: len(runs[m]) for m in models}
    print("Runs found:", seeds)

    per_class_rows = []
    for c in CLASSES[1:]:
        per_class_rows.append((f"{c} Dice", (lambda c: lambda r: r["per_class"][c]["dice"])(c), 3))
        per_class_rows.append((f"{c} IoU", (lambda c: lambda r: r["per_class"][c]["iou"])(c), 3))
    md = [
        f"Mean ± standard deviation over random seeds ({', '.join(f'{LABELS[m]}: {n}' for m, n in seeds.items())}), "
        "held-out test split ({patients} patients, {slices} slices).".format(**runs[models[0]][0].get(
            "test_set", {"patients": "?", "slices": "?"})),
        "", "**Prostate accuracy and clinical error modes**", "", table(runs, ACCURACY_ROWS, models),
        "", "**Review-band rule (flag the slices around each end of the predicted prostate extent)**", "",
        table(runs, REVIEW_ROWS, models),
        "", "**Per-class Dice and IoU**", "", table(runs, per_class_rows, models),
        "", "**Resource profile**", "", table(runs, RESOURCE_ROWS, models),
    ]
    with open(os.path.join(args.out, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- Prostate Dice by position within the prostate extent ----
    plt.figure(figsize=(8, 4.5))
    for m in models:
        centres = np.array(runs[m][0]["prostate_dice_by_position"]["bin_centres"])
        curves = np.array([r["prostate_dice_by_position"]["dice"] for r in runs[m]], dtype=float)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # bins with no slices stay NaN
            mean, sd = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
        plt.plot(centres, mean, "o-", label=LABELS[m])
        plt.fill_between(centres, mean - sd, mean + sd, alpha=0.2)
    edge = runs[models[0]][0]["settings"]["edge_fraction"]
    plt.axvspan(0, edge, color="red", alpha=0.07, label="edge slices")
    plt.axvspan(1 - edge, 1, color="red", alpha=0.07)
    plt.axhline(0.75, color="red", ls="--", lw=1, label="0.75 target")
    plt.xlabel("position within the scan's prostate extent (0 = first slice, 1 = last)")
    plt.ylabel("prostate Dice (mean ± s.d. over seeds)")
    plt.ylim(0, 1)
    plt.title("Where do the models fail? Prostate Dice across the gland")
    plt.legend(loc="lower center")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "comparison_position.png"), dpi=130)
    plt.close()

    # ---- Per-class Dice bars ----
    x = np.arange(len(CLASSES) - 1)
    width = 0.8 / len(models)
    plt.figure(figsize=(8, 4.5))
    for k, m in enumerate(models):
        stats = [stat(runs[m], (lambda c: lambda r: r["per_class"][c]["dice"])(c)) for c in CLASSES[1:]]
        plt.bar(x + k * width - 0.4 + width / 2, [s[0] for s in stats], width,
                yerr=[s[1] for s in stats], capsize=3, label=LABELS[m])
    plt.xticks(x, CLASSES[1:])
    plt.axhline(0.75, color="red", ls="--", lw=1, label="0.75 prostate target")
    plt.ylabel("test Dice (mean ± s.d.)")
    plt.ylim(0, 1)
    plt.title("Per-class test Dice")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "comparison_per_class.png"), dpi=130)
    plt.close()
    print(f"\nSaved to {args.out}/: comparison.md, comparison_position.png, comparison_per_class.png")


if __name__ == "__main__":
    main()
