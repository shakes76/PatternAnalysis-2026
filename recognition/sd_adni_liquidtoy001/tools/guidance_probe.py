"""Summarise the guidance sweep (local; post-hoc, see tools/guidance_sweep.py).

Fits the class probe of tools/class_probe.py on the validation slices' features
(as before, never on test or generated data) and reads its AUC on E2's
samples at each guidance scale, next to the final batch-1 cohort on the same
seeds at 7.5. Collects the intensity statistics of the sweep and the five
autopsy seeds, and draws two figures: probe AUC and tissue intensity against
guidance, and the autopsy seeds at each guidance. Tissue is grey level above
0.1; the real reference is the middle validation slice of each patient.

    python tools/guidance_probe.py --real outputs/final-test/real \
        --probe-report outputs/final-test/probe/class-probe.json \
        --final-cohorts outputs/final-test/cohorts \
        --sweep outputs/guidance/guidance-e2-<job1> --sweep outputs/guidance/guidance-e2-<job2> \
        --json docs/guidance-sweep-2026-11-15.json --figure docs/figures/guidance-sweep.png \
        --autopsy-figure docs/figures/autopsy-guidance.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from class_probe import auc_with_interval, fit_probe  # noqa: E402

COLOURS = {"AD": "#c8640a", "NC": "#1f6fb4"}  # as in class-probe.png
INK, MUTED = "#222222", "#777777"


def probe_scores(probe, features):
    """Probe probability of NC."""
    return probe.predict_proba(features)[:, 1]


def main() -> None:
    """Write the summary JSON and the two figures."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--final-cohorts", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, action="append", required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--autopsy-figure", type=Path, required=True)
    args = parser.parse_args()

    real_report = json.loads((args.real / "report.json").read_text())
    if real_report["roles"]["floor"] != "validation":
        raise ValueError("The probe must be fitted on validation slices")
    real = np.load(args.real / "private-real-features.npz")
    probe = fit_probe(real)
    earlier = json.loads(args.probe_report.read_text())

    reports = [json.loads((s / "report.json").read_text()) for s in args.sweep]
    seeds = {(r["first_seed"], r["seeds_per_class"]) for r in reports}
    if len(seeds) != 1:
        raise ValueError("Sweep jobs used different seeds")
    first, count = seeds.pop()
    limit = first + count

    models = {}
    intensity = {}
    real_references = None
    autopsy = []
    for folder, report in zip(args.sweep, reports):
        features = np.load(folder / "features.npz")
        for guidance in report["guidance_values"]:
            key = f"cfg{guidance}"
            is_nc = features[f"{key}_is_nc"]
            values = probe_scores(probe, features[f"{key}_inception"])
            models[str(guidance)] = {
                **auc_with_interval(is_nc, values),
                "mean_nc_probability": {
                    "asked_ad": float(values[is_nc == 0].mean()),
                    "asked_nc": float(values[is_nc == 1].mean()),
                },
            }
            intensity[str(guidance)] = report["summary"][str(guidance)]
        real_references = real_references or report["real_references"]
        autopsy += report["autopsy"]

    # The final cohort (batch 1, guidance 7.5) on the same seeds.
    cohorts = np.load(args.final_cohorts / "private-cohort-features.npz")
    listing = json.loads((args.final_cohorts / "private-cohorts.json").read_text())
    keep = np.array([s < limit for s in listing["e2"]["seeds"]])
    is_nc = cohorts["e2_is_nc"][keep]
    values = probe_scores(probe, cohorts["e2_inception"][keep])
    final_same_seeds = auc_with_interval(is_nc, values)

    order = sorted(models, key=float)
    summary = {
        "post_hoc": True,
        "question": "Does guidance cause E2's class exaggeration (README 7.3) and "
        "the over-bright worst samples (README 8, autopsy 5)?",
        "probe_chosen_C": float(probe[-1].C_[0]),
        "real_test_auc_from_earlier_probe": earlier["real_test"],
        "e2_final_cohort_auc_from_earlier_probe": earlier["models"]["e2"],
        "seeds_per_class": count,
        "probe_auc_by_guidance": {g: models[g] for g in order},
        "final_cohort_same_seeds_batch1_cfg7.5": final_same_seeds,
        "intensity_by_guidance": {g: intensity[g] for g in order},
        "real_validation_references": real_references,
        "autopsy_seeds": sorted(autopsy, key=lambda r: (r["seed"], r["guidance"])),
        "sweep_reports": [s.as_posix() for s in args.sweep],
        "test_split_read": False,
    }
    args.json.write_text(json.dumps(summary, indent=2) + "\n")

    # Figure 1: probe AUC and tissue intensity against guidance.
    x = [float(g) for g in order]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    axis = axes[0]
    auc = [models[g]["auc"] for g in order]
    low = [models[g]["auc"] - models[g]["ci95"][0] for g in order]
    high = [models[g]["ci95"][1] - models[g]["auc"] for g in order]
    axis.errorbar(x, auc, yerr=[low, high], color=INK, marker="o", ms=6, lw=2, capsize=3)
    real_auc = earlier["real_test"]["auc"]
    axis.axhline(real_auc, color=MUTED, ls="--", lw=1)
    axis.text(x[-1], real_auc - 0.012, f"real slices {real_auc:.2f}", ha="right",
              va="top", fontsize=8, color=MUTED)
    axis.set(title="Probe AUC on E2 samples", xlabel="guidance scale",
             ylabel="AUC (asked NC vs asked AD)", xticks=x)
    for col, (stat, title) in enumerate(
        (("foreground_mean", "Tissue brightness"), ("foreground_std", "Tissue contrast (std)")),
        start=1,
    ):
        axis = axes[col]
        for label in ("AD", "NC"):
            y = [intensity[g][label][stat] for g in order]
            axis.plot(x, y, color=COLOURS[label], marker="o", ms=6, lw=2)
            axis.text(x[-1] + 0.15, y[-1], f"asked {label}", color=INK, fontsize=8,
                      va="center")
            if real_references:
                axis.axhline(real_references[label][stat], color=COLOURS[label],
                             ls="--", lw=1, alpha=0.7)
        axis.set(title=title, xlabel="guidance scale",
                 xticks=x, xlim=(x[0] - 0.4, x[-1] + 1.6))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.text(0.01, 0.01, f"E2, {count} seeds per class, batch 16. Post-hoc; probe "
                "fitted on validation slices only. Dashed: real validation slices (middle "
                "slice of each patient, so brighter than an average slice).", fontsize=8,
                color=MUTED)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(args.figure, dpi=150)
    plt.close(figure)

    # Figure 2: the autopsy seeds (columns, in README order) at each guidance
    # (rows); synthetic images only.
    folder = next(s for s, r in zip(args.sweep, reports) if r["autopsy"]) / "autopsy"
    seeds_shown = list(dict.fromkeys(r["seed"] for r in autopsy))
    scales = sorted({r["guidance"] for r in autopsy})
    size, header, margin = 160, 18, 84
    panel = Image.new(
        "L", (margin + len(seeds_shown) * size, header + len(scales) * size), 255
    )
    draw = ImageDraw.Draw(panel)
    for c, seed in enumerate(seeds_shown):
        draw.text((margin + c * size + 4, 3), f"NC-{seed}", fill=0)
    for r, guidance in enumerate(scales):
        draw.text((4, header + r * size + size // 2 - 6), f"guidance {guidance:g}", fill=0)
        for c, seed in enumerate(seeds_shown):
            with Image.open(folder / f"NC-{seed}-cfg{guidance}.png") as image:
                tile = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            panel.paste(tile, (margin + c * size, header + r * size))
    panel.save(args.autopsy_figure)
    print(json.dumps({k: summary[k] for k in ("probe_auc_by_guidance",
                                              "final_cohort_same_seeds_batch1_cfg7.5")},
                     indent=2))


if __name__ == "__main__":
    main()
