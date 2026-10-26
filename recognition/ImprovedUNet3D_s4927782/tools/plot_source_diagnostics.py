"""Plot reviewed D1/D2 summaries; never read MRI, predictions or model weights.

Requires the complete public source_diagnostic_review.json plot_data contract.
Missing or incomplete D2 evidence is an error, never an invitation to invent
data. Existing figures are never overwritten. Panel A uses scan-level ranges
and medians, not independent-patient confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_GROUPS = {"train": (143, 26), "validation": (38, 6), "test_M015": (8, 1)}
EXPECTED_CASES = {f"M015/Week{i}" for i in range(8)}


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite numeric {label}.")
    return float(value)


def load_plot_data(path):
    summary = json.loads(Path(path).read_text())
    data = summary.get("plot_data", {})
    groups = data.get("contrast_groups")
    if not isinstance(groups, list) or len(groups) != 3:
        raise ValueError("Require the three reviewed cache contrast groups.")
    indexed = {g["split"]: g for g in groups}
    if set(indexed) != set(EXPECTED_GROUPS):
        raise ValueError("Contrast groups must be TRAIN, validation and M015 only.")
    for split, (scans, patients) in EXPECTED_GROUPS.items():
        group = indexed[split]
        if group["n_scans"] != scans or group["n_patients"] != patients:
            raise ValueError("Cohort scan/patient counts differ from the reviewed scope.")
        bounds = [finite_number(group[k], f"{split} {k}") for k in ("min", "median", "max")]
        if not bounds[0] <= bounds[1] <= bounds[2]:
            raise ValueError("Invalid contrast range/median order.")
        if split == "test_M015":
            values, keys = group.get("values"), group.get("case_keys")
            if (not isinstance(values, list) or len(values) != 8 or not isinstance(keys, list)
                    or len(keys) != 8 or set(keys) != EXPECTED_CASES):
                raise ValueError("Require all eight paired M015 contrast values/case keys.")
            values = [finite_number(v, "M015 contrast") for v in values]
            for actual, expected in zip((min(values), statistics.median(values), max(values)), bounds):
                if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
                    raise ValueError("M015 values disagree with the reviewed range/median.")
    intervention = data.get("support_intervention")
    if not isinstance(intervention, list) or len(intervention) != 8:
        raise ValueError("Complete D2 M015 intervention results are required before plotting.")
    visits = {row["visit"]: row for row in intervention}
    if set(visits) != {f"Week{i}" for i in range(8)}:
        raise ValueError("Require each M015 visit exactly once in the D2 panel.")
    for row in visits.values():
        for model in ("unet2d", "improved3d"):
            for condition in ("baseline_dice", "without_support_dice"):
                value = finite_number(row[model][condition], "D2 prostate Dice")
                if not 0 <= value <= 1:
                    raise ValueError("D2 prostate Dice must be in [0,1], including actual zeros.")
    return indexed, [visits[f"Week{i}"] for i in range(8)]


def create_figure(groups, visits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    settings = {"font.family": "DejaVu Sans", "font.size": 9,
                "axes.titlesize": 9.5, "axes.labelsize": 9,
                "xtick.labelsize": 8.1, "ytick.labelsize": 8.5,
                "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.5), gridspec_kw={"width_ratios": [.88, 1.22]})
        fig.subplots_adjust(left=.095, right=.985, top=.81, bottom=.255, wspace=.34)
        fig.suptitle("Source contrast and a frozen-model input intervention", fontsize=11.5, y=.965)
        contrast, intervention = axes
        cohort_order = ("train", "validation", "test_M015")
        labels = ("TRAIN\n143 scans\n26 patients", "VAL\n38 scans\n6 patients", "M015 (TEST)\n8 visits\n1 patient")
        colors = ("#5b6573", "#148578", "#b33f47")
        for position, (split, color) in enumerate(zip(cohort_order, colors)):
            group = groups[split]
            contrast.vlines(position, group["min"], group["max"], color=color, lw=2, zorder=2)
            contrast.hlines([group["min"], group["max"]], position - .11, position + .11, color=color, lw=1.4)
            contrast.scatter(position, group["median"], s=37, marker="D", color="black", zorder=4)
        m015 = groups["test_M015"]
        values_by_case = dict(zip(m015["case_keys"], m015["values"]))
        contrast.scatter([2 - .20 + .40 * i / 7 for i in range(8)],
                         [values_by_case[f"M015/Week{i}"] for i in range(8)],
                         s=19, color=colors[2], edgecolors="white", linewidths=.4, zorder=5)
        contrast.axhline(0, color="#8a8a8a", lw=.8, ls=":", zorder=1)
        contrast.set_xticks(range(3), labels)
        contrast.set_xlim(-.45, 2.45)
        low = min(groups[s]["min"] for s in cohort_order)
        high = max(groups[s]["max"] for s in cohort_order)
        contrast.set_ylim(low - .08 * max(high - low, 1), high + .06 * max(high - low, 1))
        contrast.set_ylabel("(Bladder median − Body median) / Body IQR")
        contrast.set_title("(a) Contrast on the cached grid", loc="left", pad=12)
        contrast.grid(axis="y", color="#e5e7eb", lw=.6, zorder=0)
        contrast.legend(handles=[Line2D([0], [0], color="#5b6573", lw=2, label="Min–max"),
                                 Line2D([0], [0], marker="D", color="black", ls="none", markersize=4, label="Median"),
                                 Line2D([0], [0], marker="o", color=colors[2], ls="none", markersize=4, label="M015 visit")],
                        loc="upper right", frameon=False, fontsize=7.4, handlelength=1.2, borderaxespad=.3)
        for model, label, color in (("unet2d", "2D", "#2567a9"), ("improved3d", "3D", "#b33f47")):
            for field, suffix, style, marker in (("baseline_dice", "cached", "-", "o"),
                                                  ("without_support_dice", "omit zeroing", "--", "x")):
                intervention.plot(range(8), [row[model][field] for row in visits],
                                  color=color, ls=style, marker=marker, markersize=4,
                                  linewidth=1.2, label=f"{label}: {suffix}")
        intervention.set_xticks(range(8), [str(i) for i in range(8)])
        intervention.set_xlabel("M015 visit (Week)")
        intervention.set_ylabel("Prostate Dice (FP32)")
        intervention.set_ylim(-.025, 1.025)
        intervention.set_xlim(-.3, 7.3)
        intervention.set_title("(b) Remove only final support zeroing", loc="left", pad=12)
        intervention.grid(axis="y", color="#e5e7eb", lw=.6)
        intervention.legend(loc="upper left", ncol=2, frameon=False, fontsize=7.3,
                            handlelength=1.8, columnspacing=.9, borderaxespad=.3)
        fig.text(.5, .095, "Exploratory diagnostics after v1 TEST inspection; fixed seed42 weights; no new TEST benchmark.",
                 ha="center", va="center", fontsize=7.7)
        fig.text(.5, .052, "Repeated visits; GT-defined regions. The intervention does not isolate GroupNorm or establish a root cause.",
                 ha="center", va="center", fontsize=7.7)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "figures/source_model_diagnostics.png")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    if output.suffix.lower() != ".png" or os.path.lexists(output):
        parser.error("Choose a new PNG output; existing figures are never overwritten.")
    groups, visits = load_plot_data(args.summary.expanduser().resolve(strict=True))
    figure = create_figure(groups, visits)
    temporary = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.stem + ".", suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, format="png", dpi=280, facecolor="white")
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        import matplotlib.pyplot as plt
        plt.close(figure)
    print(f"Reviewed diagnostic figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
