"""Compare a tuned 2D baseline with the selected 3D model on identical test cases.

Inputs are saved per-case CSVs from completed, one-time test evaluations. This
script only analyzes those files; it never loads test images or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_results import METRICS, paired_patient_summary, plot_patient_dice


def read_model_rows(path: Path, expected_model: str) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    """Read one model's results, rejecting missing IDs and duplicate cases."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"model", "case_id", "patient_id"}.issubset(reader.fieldnames):
            raise ValueError(f"Missing model, case_id, or patient_id column: {path}")
        rows = {}
        for row in reader:
            if row["model"] != expected_model:
                if expected_model == "3d" and row["model"] == "2d":
                    continue  # The selected run also saved its matched 2D baseline.
                raise ValueError(f"Unexpected model in {path}: {row['model']}")
            patient_id, case_id = row["patient_id"], row["case_id"]
            if not patient_id or not case_id:
                raise ValueError(f"Missing patient or case ID in {path}")
            key = (patient_id, case_id)
            if key in rows:
                raise ValueError(f"Duplicate result for {key} in {path}")
            rows[key] = row
        if not rows:
            raise ValueError(f"No {expected_model} results in {path}")
        return reader.fieldnames, rows


def pair_rows(selected_path: Path, baseline_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return matching 2D/3D rows after checking the complete patient/case set."""
    selected_fields, selected = read_model_rows(selected_path, "3d")
    baseline_fields, baseline = read_model_rows(baseline_path, "2d")
    if selected_fields != baseline_fields:
        raise ValueError("Per-case CSV columns differ")
    if selected.keys() != baseline.keys():
        missing_2d = sorted(selected.keys() - baseline.keys())
        missing_3d = sorted(baseline.keys() - selected.keys())
        raise ValueError(
            f"Patient/case sets differ: missing 2D={missing_2d}; missing 3D={missing_3d}"
        )
    paired = []
    for key in sorted(selected):
        paired.extend((baseline[key], selected[key]))
    return selected_fields, paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-per-case", type=Path, required=True)
    parser.add_argument("--baseline-per-case", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    fields, rows = pair_rows(args.selected_per_case, args.baseline_per_case)
    report = {
        metric: result for metric in METRICS
        if (result := paired_patient_summary(rows, metric)) is not None
    }
    for output in (args.out_csv, args.out_json, args.figure):
        output.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_patient_dice(
        rows, args.figure, baseline_label="Tuned 2D U-Net",
        improved_label="Selected 3D U-Net",
    )
    print(json.dumps(report["native_prostate_dice"], indent=2))


if __name__ == "__main__":
    main()
