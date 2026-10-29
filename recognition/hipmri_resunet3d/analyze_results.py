"""Summarize paired patient-level differences in a completed test run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METRICS = (
    "native_prostate_dice",
    "native_apex_dice",
    "native_base_dice",
    "native_hd95_mm",
    "native_false_positive_bladder_fraction",
    "native_false_positive_rectum_fraction",
    "latency_ms",
)


def paired_patient_summary(
    rows: list[dict[str, str]], metric: str, seed: int = 3710,
    bootstrap_samples: int = 10_000,
) -> dict | None:
    """Average visits within each patient, then resample patients in pairs."""
    by_model_case = defaultdict(dict)
    for row in rows:
        key = (row["patient_id"], row["case_id"])
        if row["model"] in by_model_case[key]:
            raise ValueError(f"Repeated {row['model']} result for {key}")
        by_model_case[key][row["model"]] = row
    if any(set(pair) != {"2d", "3d"} for pair in by_model_case.values()):
        raise ValueError("2D and 3D results must contain the same patient/case pairs")

    per_patient = defaultdict(lambda: {"2d": [], "3d": []})
    for (patient_id, _), pair in by_model_case.items():
        if any(pair[kind].get(metric, "") == "" for kind in ("2d", "3d")):
            continue
        for kind in ("2d", "3d"):
            per_patient[patient_id][kind].append(float(pair[kind][metric]))
    complete = [
        values for _, values in sorted(per_patient.items())
        if values["2d"] and values["3d"]
    ]
    if not complete:
        return None
    baseline = np.array([np.mean(values["2d"]) for values in complete])
    improved = np.array([np.mean(values["3d"]) for values in complete])
    difference = improved - baseline
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(bootstrap_samples, len(difference)))
    bootstrap = difference[draws].mean(axis=1)
    low, high = np.percentile(bootstrap, (2.5, 97.5))
    return {
        "patients": len(complete),
        "baseline_patient_mean": float(baseline.mean()),
        "improved_patient_mean": float(improved.mean()),
        "improved_minus_baseline": float(difference.mean()),
        "paired_patient_bootstrap_95pct_interval": [float(low), float(high)],
        "bootstrap_samples": bootstrap_samples,
    }


def plot_patient_dice(
    rows: list[dict[str, str]], path: Path,
    baseline_label: str = "2D U-Net", improved_label: str = "3D Improved U-Net",
) -> None:
    """Draw paired visit-averaged native Dice values for each test patient."""
    by_patient = defaultdict(lambda: {"2d": [], "3d": []})
    for row in rows:
        value = row.get("native_prostate_dice", "")
        if value != "":
            by_patient[row["patient_id"]][row["model"]].append(float(value))
    figure, axis = plt.subplots(figsize=(5.6, 4.4))
    patient_points = []
    for patient_id, values in sorted(by_patient.items()):
        if not values["2d"] or not values["3d"]:
            continue
        baseline, improved = (np.mean(values[kind]) for kind in ("2d", "3d"))
        patient_points.append((patient_id, float(improved)))
        axis.plot([0, 1], [baseline, improved], color="#8da8b4", linewidth=1.2)
        axis.scatter([0, 1], [baseline, improved], color=("#176b83", "#b35b38"), s=38)
    # Keep close patient scores readable without moving their plotted points.
    previous_label_y = -1.0
    for patient_id, improved in sorted(patient_points, key=lambda point: point[1]):
        label_y = max(improved, previous_label_y + 0.038)
        shifted = label_y - improved > 0.005
        axis.annotate(
            patient_id, xy=(1, improved), xytext=(1.05, label_y),
            textcoords="data", va="center", fontsize=8,
            arrowprops=(dict(arrowstyle="-", color="#8da8b4", linewidth=0.7)
                        if shifted else None),
        )
        previous_label_y = label_y
    axis.set(xlim=(-0.25, 1.5), ylim=(0, 1), ylabel="Native prostate Dice",
             xticks=[0, 1], xticklabels=[baseline_label, improved_label],
             title="Test patients, visits averaged within patient")
    axis.grid(axis="y", color="#e3eaed")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--baseline-label", default="2D U-Net")
    parser.add_argument("--improved-label", default="3D Improved U-Net")
    args = parser.parse_args()
    with args.per_case.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Empty per-case results")
    report = {
        metric: result for metric in METRICS
        if (result := paired_patient_summary(rows, metric)) is not None
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.figure is not None:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plot_patient_dice(rows, args.figure, args.baseline_label, args.improved_label)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
