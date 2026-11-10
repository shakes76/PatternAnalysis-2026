"""Read-only course ADNI inventory; emit aggregates without patient identifiers.

Run inside a Rangpur CPU allocation. Image headers are inspected, but pixel
decoding, content deduplication, clinical assessment, and GPU tests are not done.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys

from PIL import Image


def audit(root: Path) -> dict:
    """Summarize file structure and metadata-backed patient split overlap."""
    metadata_path = root / "meta_data_with_label.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    patients = {}
    for scan_id, record in metadata.items():
        match = re.search(r"ADNI_(\d{3}_S_\d{4})_", record.get("raw", ""))
        if match:
            patients[scan_id] = match.group(1)

    files = sorted((root / "AD_NC").glob("*/*/*.jpeg"))
    if not files:
        raise ValueError("No JPEG files found in AD_NC/<split>/<class>/")
    counts = Counter()
    sizes = Counter()
    modes = Counter()
    labels = Counter()
    scans = defaultdict(set)
    subjects = defaultdict(set)
    per_scan = Counter()
    patient_classes = defaultdict(set)
    slice_indices = Counter()
    issues = Counter()
    total_bytes = 0
    for path in files:
        split, label, _ = path.relative_to(root / "AD_NC").parts
        group = f"{split}/{label}"
        counts[group] += 1
        total_bytes += path.stat().st_size
        match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
        if match:
            scan_id, slice_index = match.groups()
            scans[split].add(scan_id)
            per_scan[scan_id] += 1
            slice_indices[int(slice_index)] += 1
            if scan_id not in metadata:
                issues["images_without_metadata"] += 1
            else:
                labels[f"{label}:metadata_label={metadata[scan_id].get('label')}"] += 1
            if scan_id in patients:
                patient_id = patients[scan_id]
                subjects[split].add(patient_id)
                patient_classes[patient_id].add(label)
            else:
                issues["images_without_patient_mapping"] += 1
        else:
            issues["unexpected_filename"] += 1
        try:
            with Image.open(path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
        except (OSError, ValueError):
            issues["unreadable_image_header"] += 1

    versions = {}
    for package in (
        "torch", "torchvision", "diffusers", "transformers", "accelerate",
        "peft", "safetensors", "nibabel", "numpy", "Pillow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    disks = {}
    for name, path in (("home", Path.home()), ("job_tmp", Path(os.environ["TMPDIR"]))):
        usage = shutil.disk_usage(path)
        disks[name] = {"total_bytes": usage.total, "free_bytes": usage.free}

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "packages": versions,
        "filesystem_snapshot": disks,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "metadata_records": len(metadata),
        "metadata_records_with_patient_mapping": len(patients),
        "image_count": len(files),
        "image_bytes": total_bytes,
        "split_class_counts": dict(sorted(counts.items())),
        "image_sizes": dict(sizes),
        "image_modes": dict(modes),
        "folder_metadata_label_counts": dict(sorted(labels.items())),
        "scan_counts_by_split": {k: len(v) for k, v in scans.items()},
        "patient_counts_by_split": {k: len(v) for k, v in subjects.items()},
        "unique_scans": len(set().union(*scans.values())),
        "unique_patients": len(patient_classes),
        "train_test_shared_scans": len(scans["train"] & scans["test"]),
        "train_test_shared_patients": len(subjects["train"] & subjects["test"]),
        "patients_in_multiple_class_folders": sum(len(v) > 1 for v in patient_classes.values()),
        "images_per_scan_histogram": dict(sorted(Counter(per_scan.values()).items())),
        "slice_index_counts": dict(sorted(slice_indices.items())),
        "issues": dict(issues),
        "limits": [
            "Header inspection only; pixels were not fully decoded or clinically assessed.",
            "No image-content deduplication or GPU/model execution.",
            "Patient identity is inferred from the supplied metadata raw-path convention.",
            "Disk free space is a snapshot, not a reservation or quota guarantee.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("TMPDIR"):
        parser.error("Run this audit inside a Slurm CPU allocation with TMPDIR set.")
    print(json.dumps(audit(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
