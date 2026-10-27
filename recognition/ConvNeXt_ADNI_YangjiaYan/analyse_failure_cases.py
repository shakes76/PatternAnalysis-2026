"""Inspect fixed predictions for five previously selected test errors."""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

BASE = Path.home() / "data"
EVAL = BASE / "comp3710_a3_evaluation"
CASES = BASE / "comp3710_a3_figures/failure_case_details.csv"
OUT = Path("results/failure_audit")


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    cases = read_csv(CASES)
    check(len(cases) == 5, "Expected five selected cases.")
    summaries, scan_details = [], []

    for model in ("cnn", "convnext"):
        tables = {
            level: read_csv(
                EVAL / model / "test" / f"{level}_predictions.csv"
            )
            for level in ("slice", "scan", "patient")
        }

        for case in cases:
            number = int(case["case"])
            patient = case["patient_id"]
            label = int(case["true_label"])
            selected = {
                level: [
                    r for r in rows if r["patient_id"] == patient
                ]
                for level, rows in tables.items()
            }
            check(
                len(selected["patient"]) == 1
                and bool(selected["slice"])
                and bool(selected["scan"]),
                f"Missing or duplicate patient records: case {number}",
            )

            for rows in selected.values():
                for row in rows:
                    score = float(row["prob_ad"])
                    check(
                        int(row["label"]) == label
                        and math.isfinite(score)
                        and 0 <= score <= 1
                        and int(row["prediction"]) == int(score >= 0.5),
                        f"Invalid prediction: {model}, case {number}",
                    )

            groups = defaultdict(list)
            for row in selected["slice"]:
                groups[row["scan_id"]].append(float(row["prob_ad"]))

            saved_scans = {
                row["scan_id"]: row for row in selected["scan"]
            }
            check(
                len(saved_scans) == len(selected["scan"])
                and set(groups) == set(saved_scans),
                f"Scan alignment failed: {model}, case {number}",
            )

            scan_scores = []
            for index, scan_id in enumerate(sorted(groups), start=1):
                scores = groups[scan_id]
                saved = saved_scans[scan_id]
                score = mean(scores)
                check(
                    len(scores) == int(saved["n_members"])
                    and math.isclose(
                        score, float(saved["prob_ad"]), abs_tol=1e-6
                    ),
                    f"Scan aggregation failed: {model}, case {number}",
                )
                scan_scores.append(score)
                scan_details.append({
                    "case": number,
                    "model": model,
                    "scan_index": index,
                    "label": label,
                    "n_slices": len(scores),
                    "prob_ad": score,
                    "prediction": int(score >= 0.5),
                    "slice_error_fraction": mean(
                        int(int(p >= 0.5) != label) for p in scores
                    ),
                })

            slices = [
                float(r["prob_ad"]) for r in selected["slice"]
            ]
            patient_score = mean(scan_scores)
            saved_patient = selected["patient"][0]
            check(
                len(scan_scores) == int(saved_patient["n_members"])
                and math.isclose(
                    patient_score,
                    float(saved_patient["prob_ad"]),
                    abs_tol=1e-6,
                )
                and math.isclose(
                    patient_score,
                    float(case[f"{model}_patient_prob_ad"]),
                    abs_tol=1e-6,
                ),
                f"Patient aggregation failed: {model}, case {number}",
            )

            summaries.append({
                "case": number,
                "model": model,
                "label": label,
                "n_slices": len(slices),
                "n_scans": len(scan_scores),
                "patient_prob_ad": patient_score,
                "patient_prediction": int(patient_score >= 0.5),
                "slice_error_fraction": mean(
                    int(int(p >= 0.5) != label) for p in slices
                ),
                "scan_error_fraction": mean(
                    int(int(p >= 0.5) != label) for p in scan_scores
                ),
                "slice_prob_min": min(slices),
                "slice_prob_max": max(slices),
                "slice_prob_std": pstdev(slices),
                "scan_prob_min": min(scan_scores),
                "scan_prob_max": max(scan_scores),
                "scan_prediction_disagreement": int(
                    any(p >= 0.5 for p in scan_scores)
                    and any(p < 0.5 for p in scan_scores)
                ),
            })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "case_summary.csv", summaries)
    write_csv(OUT / "scan_details.csv", scan_details)

    print("Aggregation reproduction: PASSED")
    for row in summaries:
        print(
            f"Case {row['case']} | {row['model']} | "
            f"slices={row['n_slices']} scans={row['n_scans']} | "
            f"P(AD)={row['patient_prob_ad']:.4f} | "
            f"wrong slices={row['slice_error_fraction']:.1%} | "
            f"wrong scans={row['scan_error_fraction']:.1%} | "
            f"scan range={row['scan_prob_min']:.4f}"
            f"–{row['scan_prob_max']:.4f}"
        )
    print("Saved to:", OUT.resolve())


if __name__ == "__main__":
    main()
