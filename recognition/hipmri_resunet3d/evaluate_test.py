"""Evaluate a selected validation-only run on its locked test split once."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from dataset import VolumeDataset, load_split, read_manifest, verify_manifest_provenance
from predict import load_model
from train import choose_device, evaluate


def load_pending_report(run_dir: Path) -> dict:
    """Reject runs whose test split may already have been opened."""
    report = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if report.get("provenance", {}).get("validation_only") is not True:
        raise ValueError("Expected a validation-only run with no test evaluation")
    if any(kind not in report for kind in ("2d", "3d")):
        raise ValueError("Both validation model reports are required")
    if any("test" in report.get(kind, {}) for kind in ("2d", "3d")):
        raise ValueError("Test results already exist; do not evaluate twice")
    if any((run_dir / name).exists() for name in (
        "test_evaluation_started.json", "per_case.csv",
        "per_case.pending.csv", "results.pending.json",
    )):
        raise ValueError("Test evaluation has already started; inspect the run before retrying")
    return report


def claim_test_evaluation(run_dir: Path, manifest: Path) -> None:
    """Atomically record that the held-out test split is about to be opened."""
    marker = run_dir / "test_evaluation_started.json"
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.resolve()),
    }
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError("Test evaluation has already started for this run") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    report = load_pending_report(args.run_dir)
    verify_manifest_provenance(report["provenance"], args.manifest)
    device = choose_device(args.device)
    baseline, baseline_config = load_model(args.run_dir / "2d_best.pt", device)
    improved, improved_config = load_model(args.run_dir / "3d_best.pt", device)
    if (
        baseline_config["size"] != improved_config["size"]
        or baseline_config["prostate_label"] != improved_config["prostate_label"]
        or baseline_config.get("crop_xy") != improved_config.get("crop_xy")
    ):
        raise ValueError("The two checkpoints use different preprocessing")
    if (
        report["provenance"].get("crop_xy") != improved_config.get("crop_xy")
        or tuple(report["provenance"]["size_dhw"]) != tuple(improved_config["size"])
        or report["provenance"]["prostate_label"] != improved_config["prostate_label"]
    ):
        raise ValueError("Checkpoint preprocessing differs from the training report")

    split = load_split(read_manifest(args.manifest), args.run_dir / "split.json")
    test_data = VolumeDataset(
        split["test"], tuple(improved_config["size"]),
        improved_config["prostate_label"],
        crop_xy=improved_config.get("crop_xy"),
    )
    claim_test_evaluation(args.run_dir, Path(args.manifest))
    bladder_label = report["provenance"].get("bladder_label")
    rectum_label = report["provenance"].get("rectum_label")
    all_rows = []
    for kind, model in (("2d", baseline), ("3d", improved)):
        summary, rows = evaluate(
            model, test_data, kind, device, bladder_label,
            rectum_label, native=True,
        )
        report[kind]["test"] = summary
        all_rows.extend({"model": kind, **row} for row in rows)

    report["provenance"]["validation_only"] = False
    report["provenance"]["finalized_from_validation_only"] = True
    report["provenance"]["test_evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()

    csv_path = args.run_dir / "per_case.csv"
    csv_pending = args.run_dir / "per_case.pending.csv"
    with csv_pending.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    report_path = args.run_dir / "results.json"
    report_pending = args.run_dir / "results.pending.json"
    report_pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_pending.replace(csv_path)
    report_pending.replace(report_path)
    for kind in ("2d", "3d"):
        score = report[kind]["test"]["patient_mean_native_prostate_dice"]
        print(f"{kind} patient-balanced native test Dice: {score:.3f}")


if __name__ == "__main__":
    main()
