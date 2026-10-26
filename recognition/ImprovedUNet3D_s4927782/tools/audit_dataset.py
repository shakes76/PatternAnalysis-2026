"""Audit deidentified HipMRI pairs and propose patient-disjoint splits.

Only letter-plus-three-digit/Week filename keys observed in the dataset are accepted. These are grouping keys,
not real identities; this tool makes no external identity queries. Images remain
header-only, and at most three masks have their voxel values inspected. No model
is trained or evaluated, including on the proposed test partition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    nib = None
    np = None


DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")
FOLDERS = {"image": "semantic_MRs", "mask": "semantic_labels_only"}
EXPECTED_KINDS = {"image": "LFOV", "mask": "SEMANTIC"}
SCAN_NAME = re.compile(
    r"^(?P<patient>[A-Z]\d{3})_(?P<visit>Week\d+)_(?P<kind>LFOV|SEMANTIC)\.nii(?:\.gz)?$"
)
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_SEED = 3710
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
MAX_INSPECTED_MASKS = 3
GEOMETRY_ATOL = 1e-5
GEOMETRY_RTOL = 1e-5


def issue(code: str, message: str, paths: Sequence[str] = ()) -> dict:
    return {"code": code, "message": message, "relative_paths": list(paths)}


def parse_scan_name(filename: str, expected_kind: str) -> tuple[str, str]:
    """Extract only the explicitly supported, deidentified grouping keys."""
    match = SCAN_NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unsupported NIfTI filename: {filename}")
    if match["kind"] != expected_kind:
        raise ValueError(f"Expected {expected_kind} in this directory: {filename}")
    return match["patient"], match["visit"]


def discover_pairs(data_root: Path) -> dict:
    """Inventory every NIfTI name; retain missing and duplicate pair records."""
    data_root = Path(data_root)
    groups: dict[tuple[str, str], dict[str, list[str]]] = {}
    blockers, unknown_files, folders = [], [], {}
    if not data_root.is_dir():
        blockers.append(issue("missing_data_root", "Data root is not a directory."))
    for role, folder_name in FOLDERS.items():
        directory = data_root / folder_name
        folders[role] = {"relative_directory": folder_name, "nifti_count": 0}
        if not directory.is_dir():
            blockers.append(issue("missing_directory", f"Missing {folder_name}."))
            continue

        def onerror(error: OSError) -> None:
            blockers.append(issue("directory_read_error", str(error)))

        for current, directories, filenames in os.walk(
            directory, followlinks=False, onerror=onerror
        ):
            directories.sort()
            for filename in sorted(filenames):
                if not filename.lower().endswith((".nii", ".nii.gz")):
                    continue
                relative = (Path(current) / filename).relative_to(data_root).as_posix()
                folders[role]["nifti_count"] += 1
                try:
                    key = parse_scan_name(filename, EXPECTED_KINDS[role])
                except ValueError as error:
                    unknown_files.append(relative)
                    blockers.append(issue("unsupported_filename", str(error), [relative]))
                    continue
                groups.setdefault(key, {"image": [], "mask": []})[role].append(relative)
        if folders[role]["nifti_count"] == 0:
            blockers.append(issue("empty_directory", f"No NIfTI files in {folder_name}."))

    records = []
    for (patient, visit), paths in sorted(groups.items()):
        valid = True
        for role in FOLDERS:
            if not paths[role]:
                valid = False
                blockers.append(issue(
                    "missing_pair", f"{patient}/{visit} has no {role}.",
                    paths["mask" if role == "image" else "image"],
                ))
            elif len(paths[role]) != 1:
                valid = False
                blockers.append(issue(
                    "duplicate_pair", f"{patient}/{visit} has multiple {role} files.",
                    paths[role],
                ))
        records.append({
            "patient": patient, "visit": visit,
            "image_paths": paths["image"], "mask_paths": paths["mask"],
            "pairing_valid": valid, "geometry_valid": False,
            "image": None, "mask": None,
        })
    return {
        "folders": folders, "records": records,
        "unknown_files": unknown_files, "blockers": blockers,
    }


def validate_patient_splits(
    splits: Mapping[str, Sequence[str]], expected_patients: Iterable[str]
) -> None:
    """Fail explicitly on overlap, duplicates, missing patients or empty splits."""
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Splits must be train, validation and test.")
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        patients = list(splits[name])
        if not patients:
            raise ValueError(f"Patient split {name} is empty.")
        if len(set(patients)) != len(patients) or seen.intersection(patients):
            raise ValueError("Patient overlap or duplicate detected in splits.")
        seen.update(patients)
    if seen != set(expected_patients):
        raise ValueError("Patient splits do not exactly cover the audited patients.")


def generate_patient_splits(
    patient_ids: Iterable[str], seed: int = SPLIT_SEED,
    fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> dict[str, list[str]]:
    """Shuffle unique patients deterministically, rounding with at least one each."""
    patients = sorted(set(patient_ids))
    if len(patients) < 3:
        raise ValueError("At least three distinct patients are needed for three splits.")
    if (len(fractions) != 3 or any(not math.isfinite(f) or f <= 0 for f in fractions)
            or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9)):
        raise ValueError("Three positive split fractions must sum to one.")
    targets = [len(patients) * fraction for fraction in fractions]
    counts = [max(1, math.floor(target)) for target in targets]
    while sum(counts) < len(patients):
        index = max(range(3), key=lambda i: targets[i] - counts[i])
        counts[index] += 1
    while sum(counts) > len(patients):
        index = max((i for i in range(3) if counts[i] > 1),
                    key=lambda i: counts[i] - targets[i])
        counts[index] -= 1
    random.Random(seed).shuffle(patients)
    splits, start = {}, 0
    for name, count in zip(SPLIT_NAMES, counts):
        splits[name] = sorted(patients[start:start + count])
        start += count
    validate_patient_splits(splits, patients)
    return splits


def _safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _inspect_header(path: Path, relative: str) -> tuple[object | None, dict, list]:
    record = {"relative_path": relative, "shape": None, "zooms": None,
              "orientation": None, "affine": None, "header_valid": False}
    try:
        volume = nib.load(str(path))
        shape = tuple(int(size) for size in volume.shape)
        zooms = tuple(float(value) for value in volume.header.get_zooms())
        affine = np.asarray(volume.affine, dtype=float)
        record.update(shape=list(shape), zooms=[_safe_float(v) for v in zooms])
        record["affine"] = [[_safe_float(v) for v in row] for row in affine]
        errors = []
        if len(shape) != 3 or any(size <= 0 for size in shape):
            errors.append(issue("invalid_dimensions", "Expected a nonempty 3D volume.", [relative]))
        if len(zooms) != 3 or any(not math.isfinite(v) or v <= 0 for v in zooms):
            errors.append(issue("invalid_spacing", "Expected finite positive 3D spacing.", [relative]))
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            errors.append(issue("invalid_affine", "Affine must be a finite 4x4 matrix.", [relative]))
        else:
            axes = nib.aff2axcodes(affine)
            record["orientation"] = list(axes)
            if any(axis is None for axis in axes):
                errors.append(issue("invalid_orientation", "Affine has undefined spatial axes.", [relative]))
        record["header_valid"] = not errors
        return volume, record, errors
    except Exception as error:
        return None, record, [issue("header_read_error", str(error), [relative])]


def _inspect_labels(volume: object, relative: str) -> tuple[dict, list]:
    """Read scaled mask values only; never assign unverified label meanings."""
    try:
        values = np.asanyarray(volume.dataobj)
        finite = np.isfinite(values)
        nonfinite_count = int(np.count_nonzero(~finite))
        finite_values = values[finite]
        noninteger_count = int(np.count_nonzero(finite_values != np.rint(finite_values)))
        inspection = {
            "status": "inspected", "voxel_count": int(values.size),
            "nonfinite_count": nonfinite_count, "noninteger_count": noninteger_count,
            "label_ids_and_counts": None,
        }
        errors = []
        if nonfinite_count:
            errors.append(issue("nonfinite_labels", "Mask contains nonfinite labels.", [relative]))
        if noninteger_count:
            errors.append(issue("fractional_labels", "Mask contains noninteger labels.", [relative]))
        if not errors:
            ids, counts = np.unique(values, return_counts=True)
            inspection["label_ids_and_counts"] = [
                {"label_id": int(label), "count": int(count)}
                for label, count in zip(ids, counts)
            ]
        return inspection, errors
    except Exception as error:
        return {"status": "read_error"}, [issue("label_read_error", str(error), [relative])]


def audit_dataset(data_root: Path, inspect_labels_limit: int = 3) -> dict:
    """Validate all pair headers and bounded mask samples before proposing splits."""
    if not 0 <= inspect_labels_limit <= MAX_INSPECTED_MASKS:
        raise ValueError("inspect_labels_limit must be between 0 and 3.")
    data_root = Path(data_root).expanduser().resolve()
    report = discover_pairs(data_root)
    blockers = report["blockers"]
    inspected_masks = 0
    if nib is None or np is None:
        blockers.append(issue("missing_dependency", "Install the declared nibabel and NumPy dependencies."))
    else:
        for pair in report["records"]:
            if not pair["pairing_valid"]:
                continue
            volumes = {}
            for role in FOLDERS:
                relative = pair[f"{role}_paths"][0]
                volume, header, errors = _inspect_header(data_root / relative, relative)
                pair[role] = header
                volumes[role] = volume
                blockers.extend(errors)
            pair["mask"]["label_inspection"] = {"status": "header_only"}
            if not all(pair[role]["header_valid"] for role in FOLDERS):
                continue
            pair_paths = pair["image_paths"] + pair["mask_paths"]
            mismatches = []
            if pair["image"]["shape"] != pair["mask"]["shape"]:
                mismatches.append("shape")
            for field in ("affine", "zooms"):
                if not np.allclose(pair["image"][field], pair["mask"][field],
                                   rtol=GEOMETRY_RTOL, atol=GEOMETRY_ATOL):
                    mismatches.append(field)
            if pair["image"]["orientation"] != pair["mask"]["orientation"]:
                mismatches.append("orientation")
            pair["geometry_valid"] = not mismatches
            if mismatches:
                blockers.append(issue("geometry_mismatch", "Image/mask mismatch: " + ", ".join(mismatches), pair_paths))
            if inspected_masks < inspect_labels_limit:
                inspected_masks += 1
                inspection, errors = _inspect_labels(volumes["mask"], pair["mask_paths"][0])
                pair["mask"]["label_inspection"] = inspection
                blockers.extend(errors)

    patients = sorted({pair["patient"] for pair in report["records"]})
    splits = None
    if not blockers:
        try:
            splits = generate_patient_splits(patients)
        except ValueError as error:
            blockers.append(issue("invalid_patient_split", str(error)))
    report.update({
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root), "status": "blocked" if blockers else "audit_complete",
        "distinct_patients": len(patients), "patient_grouping_keys": patients,
        "pair_count": sum(pair["pairing_valid"] for pair in report["records"]),
        "mask_voxel_inspections_attempted": inspected_masks,
        "inspect_labels_limit": inspect_labels_limit,
        "geometry_tolerance": {"atol": GEOMETRY_ATOL, "rtol": GEOMETRY_RTOL},
        "label_semantics": {"status": "unverified", "mapping": None},
        "patient_splits": splits,
        "split_configuration": {"seed": SPLIT_SEED,
                                "requested_fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS))},
        "validation": {
            "all_pairs_and_headers_valid": not any(
                not (pair["pairing_valid"] and pair["geometry_valid"])
                for pair in report["records"]
            ) and not blockers,
            "patient_split_independence_and_coverage_validated": splits is not None,
            "all_mask_voxels_validated": False,
            "label_semantics_verified": False,
            "training_started": False, "model_evaluated_on_test": False,
        },
        "limitations": [
            "Dataset codes are deidentified grouping keys only; no real identities were inferred or queried.",
            "Headers establish grid compatibility, not visual/anatomical correctness of registration.",
            "Only the stated bounded mask sample was inspected; other mask values and all image voxels remain unchecked.",
            "Label IDs have no verified anatomical meanings; confirm dataset documentation before training.",
            "The proposed test partition was not used for model evaluation or model selection.",
        ],
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="JSON audit report path")
    parser.add_argument("--inspect-labels-limit", type=int, choices=range(4), default=3,
                        help="Number of masks to read, from 0 to the hard limit of 3")
    args = parser.parse_args(argv)
    report = audit_dataset(args.data_root, args.inspect_labels_limit)
    output = args.output.expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Cannot write audit report: {error}\n")
    print(f"Dataset audit: {report['status']}; patients: {report['distinct_patients']}; report: {output.resolve()}")
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
