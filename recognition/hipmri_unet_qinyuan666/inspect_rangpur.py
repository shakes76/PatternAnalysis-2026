"""Bounded, read-only data/environment inventory; no training or label inference.

Only Python's standard library is required. Optional explicit NIfTI samples use
nibabel/numpy if already installed. Output can contain private paths/patient IDs.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys


DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")


def inventory(root: Path, max_depth: int = 2, max_entries: int = 120,
              per_directory: int = 20) -> dict:
    """Inspect at most max_entries entries; never follow directory symlinks.

    Directory order is filesystem order, so these are samples, not totals.
    A directory at max_depth is listed but its contents are not opened.
    """
    if min(max_depth, max_entries, per_directory) < 1:
        raise ValueError("All inventory limits must be positive.")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Data root is not an accessible directory: {root}")
    entries: list[dict] = []
    errors: list[dict] = []
    pending = deque([(root, 0)])
    directories_opened = 0
    while pending and len(entries) < max_entries:
        directory, depth = pending.popleft()
        try:
            with os.scandir(directory) as children:
                directories_opened += 1
                # Do not sort/list an entire shared directory before limiting it.
                for _ in range(min(per_directory, max_entries - len(entries))):
                    child = next(children, None)
                    if child is None:
                        break
                    path = Path(child.path)
                    relative = path.relative_to(root).as_posix()
                    try:
                        if child.is_symlink():
                            kind = "symlink_not_followed"
                        elif child.is_dir(follow_symlinks=False):
                            kind = "directory"
                            if depth + 1 < max_depth:
                                pending.append((path, depth + 1))
                        else:
                            kind = "file"
                        entries.append({"path": relative, "kind": kind})
                    except OSError as exc:
                        entries.append({"path": relative, "kind": "unreadable"})
                        errors.append({"path": relative, "error": str(exc)})
        except OSError as exc:
            errors.append({"path": directory.relative_to(root).as_posix(), "error": str(exc)})
    return {
        "root": str(root), "inventory_kind": "bounded_sample_not_dataset_totals",
        "limits": {"max_depth": max_depth, "max_entries": max_entries,
                   "entries_per_directory": per_directory},
        "directories_opened": directories_opened, "entries_sampled": len(entries),
        "entries": entries, "errors": errors,
        "possible_documentation": [entry["path"] for entry in entries
            if entry["kind"] == "file" and Path(entry["path"]).suffix.lower()
            in {".csv", ".tsv", ".json", ".txt", ".md", ".yaml", ".yml"}],
        "patient_mapping": "unknown; confirm with authoritative dataset metadata",
        "label_meanings": "unknown; numeric values alone do not identify the prostate",
        "training_ready": False,
    }


def nifti_sample(path: Path, read_mask_values: bool = False) -> dict:
    """Read an explicitly selected header, optionally a small mask's raw values."""
    result = {"path": str(path), "label_meanings": "unknown"}
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        result["status"] = "skipped; optional nibabel/numpy are not installed"
        return result
    try:
        # Bound compressed input size as well as decompressed voxel count.
        if path.stat().st_size > 16 * 1024 * 1024:
            result["status"] = "skipped; sample file exceeds 16 MiB"
            return result
        sample = nib.load(str(path))
        result.update({"status": "header_read", "shape": list(sample.shape),
                       "zooms": [float(value) for value in sample.header.get_zooms()],
                       "dtype": str(sample.get_data_dtype())})
        if read_mask_values:
            if math.prod(sample.shape) > 1_048_576:
                result["mask_values_status"] = "skipped; more than 1,048,576 voxels"
            else:
                data = np.asarray(sample.dataobj)
                finite = bool(np.isfinite(data).all())
                result["all_values_finite"] = finite
                if finite:
                    values = np.unique(data)
                    result["unique_value_count"] = int(values.size)
                    result["first_32_raw_values"] = values[:32].tolist()
                    result["all_values_integer"] = bool(np.equal(data, np.round(data)).all())
                result["mask_values_status"] = "read; anatomy meanings still unverified"
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
        result.update({"status": "error", "error": str(exc)})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-entries", type=int, default=120)
    parser.add_argument("--per-directory", type=int, default=20)
    parser.add_argument("--nifti", type=Path, action="append", default=[],
                        help="Explicit image/header sample; at most 3 samples total")
    parser.add_argument("--mask", type=Path, action="append", default=[],
                        help="Explicit known mask sample; read bounded numeric values")
    parser.add_argument("--output", type=Path,
                        help="Optional private JSON outside the source data directory")
    args = parser.parse_args()
    if len(args.nifti) + len(args.mask) > 3:
        parser.error("At most three explicit NIfTI/mask samples are allowed.")
    if not (1 <= args.max_depth <= 4 and 1 <= args.max_entries <= 500
            and 1 <= args.per_directory <= 100):
        parser.error("Limits: depth 1-4, total entries 1-500, entries/directory 1-100.")
    root = args.data_root.expanduser().resolve()
    if args.output and args.output.expanduser().resolve().is_relative_to(root):
        parser.error("Output must be outside the read-only source data directory.")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "python": platform.python_version(),
        "python_executable": sys.executable,
        "commands_available": {name: shutil.which(name) is not None for name in
            ("git", "python3", "nvidia-smi", "sinfo", "squeue", "srun", "sbatch", "qstat", "qsub")},
        # Record only these non-secret allocation indicators, never all env vars.
        "job_environment": {name: os.environ[name] for name in
            ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "PBS_JOBID", "JOB_ID") if name in os.environ},
        "allocation_note": "Command/job-variable presence does not prove GPU permission or allocation.",
        "privacy_note": "Private report: may contain patient identifiers/paths; keep outside Git.",
    }
    try:
        report["dataset"] = inventory(root, args.max_depth, args.max_entries, args.per_directory)
    except (OSError, ValueError) as exc:
        report["dataset"] = {"root": str(root), "error": str(exc), "training_ready": False}
    report["explicit_samples"] = [nifti_sample(path) for path in args.nifti]
    report["explicit_samples"] += [nifti_sample(path, True) for path in args.mask]
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved private inventory: {args.output}")
    else:
        print(rendered)
    dataset = report["dataset"]
    return 1 if "error" in dataset or dataset.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
