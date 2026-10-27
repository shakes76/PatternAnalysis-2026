"""Plot scalar R1 VAL results only; no MRI, weights, predictions or TEST inputs.

Read the complete audited summary. Display every model and all six patients;
fixed development gates are descriptive criteria, never significance tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

MODELS = ("seed42_reference", "gamma_only", "gamma_polarity", "clean_adapt", "polarity_rehearsal")
CONDITIONS = ("clean", "gamma_0.7", "gamma_1.5", "polarity", "spatial_gain")
PATIENTS = ("B040", "J026", "K008", "K018", "M030", "S022")
ARMS = ("clean_adapt", "polarity_rehearsal")
SHORT = {name: label for name, label in zip(MODELS, ("v1", "A", "B", "C", "D"))}
COLORS = ("#596a7c", "#97a5b1", "#758da1", "#1675a9", "#cb6826")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value):
    if value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, "Dice must be finite in [0,1], or null.")
    return float(value)


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def validate_summary(summary):
    require(summary["experiment_id"] == "appearance_recovery_v1" and summary["split"] == "validation"
            and summary["status"] == "completed", "Only completed R1 validation summaries are supported.")
    scope = summary["scope"]
    require(all(scope[k] == v for k, v in {"n_models": 5, "n_conditions": 5, "n_scans": 38, "n_patients": 6,
                "n_case_evaluations": 950, "test_arrays_read_by_summarizer": 0, "arrays_loaded": False}.items()), "Require the complete 5x5x38 VAL scope.")
    require(set(summary["models"]) == set(summary["results"]) == set(MODELS), "Five-model scope differs.")
    patients, values = {}, {}
    for name in MODELS:
        require(set(summary["results"][name]) == set(CONDITIONS), "All five conditions must be retained.")
        values[name] = {}
        for condition in CONDITIONS:
            group = summary["results"][name][condition]
            rows, cases = group["patients"], group["cases_prostate"]
            indexed = {r["patient"]: r for r in rows}
            require(len(rows) == len(indexed) == 6 and set(indexed) == set(PATIENTS), "Every group must retain the same six patients.")
            require(len(cases) == len({(r["patient"], r["visit"]) for r in cases}) == 38
                    and {r["patient"] for r in cases} == set(PATIENTS), "Every group must retain all38 case keys.")
            per_patient = {}
            for patient in PATIENTS:
                row = indexed[patient]
                require([r["class_index"] for r in row["per_class"]] == list(range(6)), "Expected six ordered classes per patient.")
                per_patient[patient] = scalar(row["per_class"][5]["dice"])
                require(row["num_cases"] == sum(c["patient"] == patient for c in cases), "Patient visit counts differ.")
            value = scalar(group["aggregate"]["prostate_dice"])
            expected = average(per_patient.values())
            require((value is None and expected is None) or (value is not None and expected is not None
                    and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10)), "Patient mean differs from prostate headline.")
            values[name][condition] = {"prostate_dice": value, "foreground_mean_dice": scalar(group["aggregate"]["foreground_mean_dice"])}
            if condition == "clean": patients[name] = per_patient
    choice = summary["recommendation"]
    require(choice["selection_basis"] == "validation_only" and choice["selected_model_id"] in (*ARMS, None)
            and type(choice["same_platform_verified"]) is bool, "Candidate must come from VAL only.")
    if choice["selected_model_id"] is not None:
        require(choice["same_platform_verified"] and choice["selected_model_id"] in choice["eligible_model_ids"], "Confounded/ineligible candidate cannot be displayed.")
    thresholds = [values["seed42_reference"]["clean"]["prostate_dice"],
                  values["gamma_polarity"]["polarity"]["prostate_dice"],
                  values["seed42_reference"]["clean"]["foreground_mean_dice"]]
    require(all(v is not None for v in thresholds), "Reference values must be defined for fixed gates.")
    thresholds = [thresholds[0], thresholds[1] - .02, thresholds[2] - .005]
    differences = {name: [patients[name][p] - patients["seed42_reference"][p]
                   if patients[name][p] is not None and patients["seed42_reference"][p] is not None else None for p in PATIENTS] for name in ARMS}
    return values, differences, thresholds


def build_figure(summary):
    values, differences, thresholds = validate_summary(summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(settings):
        figure = plt.figure(figsize=(8.7, 6.8))
        grid = figure.add_gridspec(2, 3, left=.09, right=.98, top=.80, bottom=.235, hspace=.72, wspace=.38,
                                   height_ratios=(1, 1.2))
        figure.suptitle("Exploratory recovery: clean accuracy and synthetic polarity robustness", y=.975, fontsize=12)
        figure.text(.5, .928, "v1: original | A: gamma | B: gamma + polarity | C: clean adapt | D: polarity rehearsal",
                    ha="center", fontsize=8.5)
        choice = summary["recommendation"]
        candidate = SHORT[choice["selected_model_id"]] if choice["selected_model_id"] else "none"
        qualifier = "paired execution verified" if choice["same_platform_verified"] else "execution confound"
        figure.text(.5, .89, f"VAL-only candidate: {candidate} | {qualifier} | six patients, 38 scans", ha="center", fontsize=9)
        columns = (("clean", "prostate_dice", "a  Clean prostate Dice"),
                   ("polarity", "prostate_dice", "b  Polarity prostate Dice"),
                   ("clean", "foreground_mean_dice", "c  Clean foreground Dice"))
        for column, (condition, metric, title) in enumerate(columns):
            axis = figure.add_subplot(grid[0, column])
            for position, name in enumerate(MODELS):
                value = values[name][condition][metric]
                if value is None:
                    axis.text(position, .035, "NA", ha="center", fontsize=8)
                else:
                    axis.bar(position, value, width=.65, color=COLORS[position], zorder=2)
                    axis.text(position, value + .018, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
            axis.axhline(thresholds[column], color="#222222", linestyle="--", linewidth=1, zorder=3)
            axis.set_title(title + f"\nCandidate gate ≥ {thresholds[column]:.4f}", loc="left", pad=10)
            axis.set_xticks(range(5), [SHORT[n] for n in MODELS])
            axis.set_ylim(min(-.025, thresholds[column] - .025), 1.09)
            axis.set_yticks([0, .2, .4, .6, .8, 1.0])
            axis.grid(axis="y", alpha=.2, zorder=0)
            if column == 0: axis.set_ylabel("Patient-balanced Dice")
        bottom = figure.add_subplot(grid[1, :])
        bottom.set_title("d  Every patient's clean prostate Dice difference versus v1", loc="left", pad=10)
        bottom.axvline(0, color="#555555", linestyle="--", linewidth=.9)
        extent = max([abs(v) for rows in differences.values() for v in rows if v is not None] + [.02])
        bottom.set_xlim(-extent * 1.42 - .006, extent * 1.42 + .006)
        for name, offset, color, marker, label in ((ARMS[0], -.15, COLORS[3], "o", "C: clean adapt − v1"),
                                                   (ARMS[1], .15, COLORS[4], "D", "D: rehearsal − v1")):
            shown = False
            for patient_index, value in enumerate(differences[name]):
                y = patient_index + offset
                if value is None:
                    bottom.text(.99, y, SHORT[name] + ": NA", transform=bottom.get_yaxis_transform(), ha="right", color=color, fontsize=8)
                    continue
                bottom.scatter(value, y, color=color, marker=marker, s=25, label=label if not shown else None, zorder=3)
                shown = True
                bottom.annotate(f"{value:+.4f}", (value, y), xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                                ha="left" if value >= 0 else "right", va="center", fontsize=8, color=color)
        bottom.set_yticks(range(6), PATIENTS); bottom.set_ylim(5.55, -.55)
        bottom.set_xlabel("Clean prostate Dice difference; visits averaged within each patient")
        bottom.grid(axis="x", alpha=.2)
        bottom.legend(loc="upper center", bbox_to_anchor=(.5, -.27), ncol=2, frameon=False, fontsize=8)
        figure.text(.02, .045, "Dashed upper-panel lines are prespecified development gates, not significance tests.\n"
                    "Same six checkpoint-selecting VAL patients; repeated conditions are not new patients.\n"
                    "Post-test-informed follow-up; no M015/TEST scores, independent confirmation or significance claim shown.",
                    ha="left", va="bottom", fontsize=8)
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    require(output.suffix.lower() == ".png" and not os.path.lexists(output), "Choose a new PNG; refusing overwrite.")
    raw = args.summary.expanduser().resolve(strict=True).read_bytes()
    figure = build_figure(json.loads(raw))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".png", delete=False) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, dpi=300, facecolor="white", metadata={"Description":
                       "Exploratory VAL recovery; input summary SHA256 " + hashlib.sha256(raw).hexdigest()})
        os.link(temporary, output)
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
        if temporary is not None: temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
