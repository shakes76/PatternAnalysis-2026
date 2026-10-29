"""Evaluate one validation-selected 2D baseline on a matching test split once.

This is a robustness check for the independently tuned 2D baseline. It leaves
the original validation report unchanged, while the shared test marker blocks
any later test evaluation of the candidate run.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dataset import VolumeDataset, load_split, read_manifest, verify_manifest_provenance
from evaluate_test import claim_test_evaluation, load_pending_report
from predict import load_model
from train import choose_device, evaluate


def matching_split(candidate: dict, reference: dict) -> bool:
    """Require identical visit IDs in all three patient partitions."""
    return all(
        [case.case_id for case in candidate[name]] ==
        [case.case_id for case in reference[name]]
        for name in ("train", "val", "test")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = load_pending_report(args.run_dir)
    provenance = report["provenance"]
    verify_manifest_provenance(provenance, args.manifest)

    cases = read_manifest(args.manifest)
    candidate_split = load_split(cases, args.run_dir / "split.json")
    reference_split = load_split(cases, args.reference_run / "split.json")
    if not matching_split(candidate_split, reference_split):
        raise ValueError("Baseline and selected 3D runs use different patient splits")
    reference_report = json.loads(
        (args.reference_run / "results.json").read_text(encoding="utf-8")
    )
    reference_provenance = reference_report["provenance"]
    verify_manifest_provenance(reference_provenance, args.manifest)
    if any(
        provenance.get(name) != reference_provenance.get(name)
        for name in ("prostate_label", "bladder_label", "rectum_label")
    ):
        raise ValueError("Baseline and reference runs use different anatomical labels")
    if (
        reference_report.get("provenance", {}).get("validation_only") is not False
        or "test" not in reference_report.get("3d", {})
    ):
        raise ValueError("Reference 3D run needs a completed test evaluation")

    device = choose_device(args.device)
    baseline, config = load_model(args.run_dir / "2d_best.pt", device)
    if config["kind"] != "2d" or (
        tuple(config["size"]) != tuple(provenance["size_dhw"])
        or config["prostate_label"] != provenance["prostate_label"]
        or config.get("crop_xy") != provenance.get("crop_xy")
    ):
        raise ValueError("2D checkpoint preprocessing differs from its training report")

    for name in ("2d_baseline_test.json", "2d_baseline_per_case.csv"):
        if (args.run_dir / name).exists():
            raise ValueError("2D baseline test output already exists")

    test_data = VolumeDataset(
        candidate_split["test"], tuple(config["size"]),
        config["prostate_label"], crop_xy=config.get("crop_xy"),
    )
    claim_test_evaluation(args.run_dir, args.manifest)
    summary, rows = evaluate(
        baseline, test_data, "2d", device,
        provenance.get("bladder_label"), provenance.get("rectum_label"),
        native=True,
    )

    result = {
        "model": "2d",
        "selection": "highest 2D patient-balanced validation Dice among four candidates",
        "validation_patient_mean_prostate_dice": report["2d"]["validation"][
            "patient_mean_prostate_dice"
        ],
        "test": summary,
        "provenance": {
            **provenance,
            "reference_run": str(args.reference_run.resolve()),
            "same_patient_split": True,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "prior_test_exposure": "Earlier one-epoch pilot opened this split; exploratory only",
        },
    }
    csv_pending = args.run_dir / "2d_baseline_per_case.pending.csv"
    with csv_pending.open("w", newline="", encoding="utf-8") as handle:
        all_rows = [{"model": "2d", **row} for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    json_pending = args.run_dir / "2d_baseline_test.pending.json"
    json_pending.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_pending.replace(args.run_dir / "2d_baseline_per_case.csv")
    json_pending.replace(args.run_dir / "2d_baseline_test.json")
    print(
        "Independently selected 2D patient-balanced native test Dice: "
        f"{summary['patient_mean_native_prostate_dice']:.3f}"
    )


if __name__ == "__main__":
    main()
