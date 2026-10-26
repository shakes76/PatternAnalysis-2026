"""Inventory the HipMRI environment and file names without loading scan data.

This standard-library-only tool writes one explicitly requested JSON report. It
does not infer patient identities, validate image/mask pairs, allocate a GPU,
load NIfTI pixels, or train a model. A discovered inventory is only a starting
point for the separate patient and spatial-alignment audit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")
DATA_FOLDERS = ("semantic_MRs", "semantic_labels_only")
DISTRIBUTIONS = ("torch", "nibabel", "numpy", "scipy")
MAX_SAMPLES = 8


def distribution_versions() -> dict[str, str | None]:
    """Read installed distribution metadata without importing ML packages."""
    versions: dict[str, str | None] = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def discover_folder(data_root: Path, folder_name: str) -> dict:
    """Count NIfTI file names recursively, retaining at most eight examples."""
    directory = data_root / folder_name
    record = {
        "relative_directory": folder_name,
        "exists": directory.is_dir(),
        "nifti_count": 0,
        "sample_relative_paths": [],
        "errors": [],
    }
    if not record["exists"]:
        return record

    def record_error(error: OSError) -> None:
        record["errors"].append(str(error))

    # os.walk does not follow directory symlinks. No file contents are opened.
    for current_directory, subdirectories, filenames in os.walk(
        directory, followlinks=False, onerror=record_error
    ):
        subdirectories.sort()
        for filename in sorted(filenames):
            if not filename.lower().endswith((".nii", ".nii.gz")):
                continue
            record["nifti_count"] += 1
            if len(record["sample_relative_paths"]) < MAX_SAMPLES:
                relative = (Path(current_directory) / filename).relative_to(
                    data_root
                )
                record["sample_relative_paths"].append(relative.as_posix())
    return record


def gpu_inventory() -> dict:
    """Query visible GPU metadata if nvidia-smi exists; never allocate a GPU."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "nvidia-smi not found on PATH"}

    command = [
        executable,
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "reason": "nvidia-smi exceeded 5 seconds"}
    except OSError as error:
        return {"status": "error", "reason": str(error)}
    if result.returncode != 0:
        return {
            "status": "error",
            "returncode": result.returncode,
            "reason": result.stderr.strip() or "nvidia-smi query failed",
        }

    devices = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 4:
            return {"status": "error", "reason": "Unexpected nvidia-smi CSV output"}
        devices.append(
            dict(zip(("index", "name", "memory_total_mib", "driver_version"), row))
        )
    return {"status": "queried", "devices": devices}


def audit_environment(data_root: Path) -> dict:
    """Return a descriptive inventory, explicitly retaining unverified checks."""
    data_root = data_root.expanduser().resolve()
    root_exists = data_root.is_dir()
    folders = {name: discover_folder(data_root, name) for name in DATA_FOLDERS}
    blockers = []
    if not root_exists:
        blockers.append(f"Data root is missing or is not a directory: {data_root}")
    else:
        for name, record in folders.items():
            if not record["exists"]:
                blockers.append(f"Required data directory is missing: {name}")
            elif record["errors"]:
                blockers.append(f"Could not completely inventory data directory: {name}")
            elif record["nifti_count"] == 0:
                blockers.append(f"No NIfTI file names found in: {name}")

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "inventory_complete",
        "blockers": blockers,
        "data_root": str(data_root),
        "data_root_exists": root_exists,
        "folders": folders,
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "distribution_versions": distribution_versions(),
            "gpu": gpu_inventory(),
        },
        "validation": {
            "patient_ids_inferred": False,
            "patient_split_validated": False,
            "image_mask_pairing_validated": False,
            "spatial_alignment_validated": False,
            "volume_pixels_read": False,
            "training_started": False,
        },
        "limitations": [
            "NIfTI counts are based on file names, not file validity or pixel contents.",
            "Image/mask pairing, patient identity, split independence, and spatial alignment remain unverified.",
            "GPU query describes visible devices only; it does not request cluster resources.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    args = parser.parse_args(argv)
    report = audit_environment(args.data_root)
    output = args.output.expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        parser.exit(2, f"Cannot write audit report: {error}\n")
    print(f"Audit status: {report['status']}; report: {output.resolve()}")
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
