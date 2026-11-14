"""Exercise data invariants against an actual prepared manifest on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import FIELDS, ADNISliceDataset, read_manifest
from prepare_data import assign_patients


def check(root: Path, manifest: Path, seed: int) -> dict:
    rows = read_manifest(manifest)
    patient_classes = {r["patient_id"]: r["class_name"] for r in rows}
    assignment = assign_patients(patient_classes, seed)
    reordered = assign_patients(dict(reversed(list(patient_classes.items()))), seed)
    if assignment != reordered or any(
        assignment[r["patient_id"]] != r["split"] for r in rows
    ):
        raise AssertionError(
            "Split assignment is not deterministic or does not match seed"
        )
    source_paths = {
        p.relative_to(root).as_posix() for p in (root / "AD_NC").glob("*/*/*.jpeg")
    }
    if source_paths != {r["relative_path"] for r in rows}:
        raise AssertionError(
            "Manifest does not cover the available source JPEGs exactly"
        )

    batches = {}
    for split in ("train", "validation"):
        data = ADNISliceDataset(root, manifest, split, image_size=256)
        # Sequential batches spanning distinct parts of each split; no test quality inspection.
        indices = sorted({0, len(data) // 2, len(data) - 1})
        for index in indices:
            item = data[index]
            values = item["pixel_values"]
            if values.shape != (3, 256, 256) or values.dtype != torch.float32:
                raise AssertionError("Unexpected tensor contract")
            if (
                not torch.isfinite(values).all()
                or values.min() < -1
                or values.max() > 1
            ):
                raise AssertionError("Invalid tensor values")
            if not torch.equal(values[0], values[1]) or not torch.equal(
                values[1], values[2]
            ):
                raise AssertionError("Grayscale channel replication failed")
            with (root / item["relative_path"]).open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != data.rows[index]["sha256"]:
                raise AssertionError("Image changed since manifest creation")
        batch = next(iter(DataLoader(data, batch_size=2, num_workers=0)))
        if tuple(batch["pixel_values"].shape) != (2, 3, 256, 256):
            raise AssertionError("DataLoader collation failed")
        batches[split] = {
            "sample_checks": len(indices),
            "batch_shape": list(batch["pixel_values"].shape),
        }

    # Negative controls deliberately corrupt real manifest rows, not training data.
    rejected = []
    with tempfile.TemporaryDirectory() as temporary:
        for case in (
            "patient_overlap",
            "path_escape",
            "label_mismatch",
            "duplicate_row",
        ):
            changed = [dict(row) for row in rows]
            if case == "patient_overlap":
                changed[0]["split"] = (
                    "validation" if changed[0]["split"] != "validation" else "train"
                )
            elif case == "path_escape":
                changed[0]["relative_path"] = "../outside.jpeg"
            elif case == "label_mismatch":
                changed[0]["metadata_label"] = "999"
            else:
                changed.append(dict(changed[0]))
            candidate = Path(temporary) / f"{case}.csv"
            with candidate.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(changed)
            try:
                read_manifest(candidate)
            except ValueError:
                rejected.append(case)
            else:
                raise AssertionError(f"Invalid manifest accepted: {case}")
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "coverage_images": len(source_paths),
        "deterministic_patient_assignment": True,
        "batches": batches,
        "negative_controls_rejected": rejected,
        "test_model_outputs_or_quality_metrics_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3710)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run source-data checks in a CPU allocation")
    torch.set_num_threads(1)
    print(
        json.dumps(check(args.root, args.manifest, args.seed), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
