"""Prepare a shared canonical HipMRI grid from an unblocked dataset audit.

No label semantics are inferred. Raw label IDs must be supplied explicitly. The
optional --limit creates a training-only development cache, never a full dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dataset import validate_patient_splits

try:
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom
except ImportError:
    nib = None
    np = None
    zoom = None


def _label_mapping(label_ids: Sequence[int]) -> dict[int, int]:
    if (not label_ids or len(label_ids) > 256 or any(type(label) is not int for label in label_ids)
            or len(set(label_ids)) != len(label_ids)):
        raise ValueError("Declare 1-256 distinct integer --label-ids in class-index order.")
    return {label: index for index, label in enumerate(label_ids)}


def _source_metadata(volume: object) -> dict:
    return {"shape_xyz": list(volume.shape), "affine_xyz": np.asarray(volume.affine).tolist()}


def preprocess_pair(
    image_path: Path, mask_path: Path, target_shape_dhw: Sequence[int],
    label_ids: Sequence[int],
) -> tuple[dict, dict]:
    """Apply matching RAS orientation and centre-aware resampling to one pair."""
    if nib is None or np is None or zoom is None:
        raise ImportError("Cache preparation requires nibabel, NumPy and SciPy.")
    if len(target_shape_dhw) != 3 or any(type(s) is not int or s <= 0 for s in target_shape_dhw):
        raise ValueError("Target shape must contain three positive D,H,W integers.")
    mapping = _label_mapping(label_ids)
    original_image, original_mask = nib.load(str(image_path)), nib.load(str(mask_path))
    if len(original_image.shape) != 3 or len(original_mask.shape) != 3:
        raise ValueError("Image and mask must both be 3D.")
    image_volume = nib.as_closest_canonical(original_image)
    mask_volume = nib.as_closest_canonical(original_mask)
    if image_volume.shape != mask_volume.shape or not np.allclose(
        image_volume.affine, mask_volume.affine, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Canonical image/mask grids are not aligned.")
    if (nib.aff2axcodes(image_volume.affine) != ("R", "A", "S")
            or not np.isfinite(image_volume.affine).all()
            or not np.isfinite(mask_volume.affine).all()):
        raise ValueError("Expected finite RAS canonical geometry.")
    old_spacing = np.asarray(image_volume.header.get_zooms()[:3], dtype=np.float64)
    mask_spacing = np.asarray(mask_volume.header.get_zooms()[:3], dtype=np.float64)
    if (not np.isfinite(old_spacing).all() or np.any(old_spacing <= 0)
            or not np.allclose(old_spacing, mask_spacing, rtol=1e-5, atol=1e-5)):
        raise ValueError("Image/mask spacing must be finite, positive and aligned.")
    image = image_volume.get_fdata(dtype=np.float32, caching="unchanged")
    raw_mask = np.asanyarray(mask_volume.dataobj)
    if not np.isfinite(image).all() or not np.isfinite(raw_mask).all():
        raise ValueError("Image or mask contains nonfinite voxel values.")
    if np.any(raw_mask != np.rint(raw_mask)):
        raise ValueError("Mask contains fractional labels.")
    observed = {int(raw) for raw in np.unique(raw_mask)}
    if not observed.issubset(mapping):
        raise ValueError(f"Mask contains undeclared raw label IDs: {sorted(observed - set(mapping))}.")
    mask = np.empty(raw_mask.shape, dtype=np.uint8)
    for raw, index in mapping.items():
        mask[raw_mask == raw] = index

    foreground = image != 0
    count = int(np.count_nonzero(foreground))
    mean = float(np.mean(image[foreground], dtype=np.float64)) if count else 0.0
    standard_deviation = float(np.std(image[foreground], dtype=np.float64)) if count else 0.0
    normalized = np.zeros(image.shape, dtype=np.float32)
    if count:
        normalized[foreground] = np.clip(
            (image[foreground].astype(np.float64) - mean) / max(standard_deviation, 1e-8), -5.0, 5.0
        ).astype(np.float32)
    target_xyz = tuple(reversed(target_shape_dhw))
    old_shape = np.asarray(image.shape, dtype=np.float64)
    ratio = np.asarray(target_xyz, dtype=np.float64) / old_shape
    options = {"grid_mode": True, "mode": "nearest", "prefilter": False}
    resized_image = zoom(normalized, ratio, order=1, **options)
    resized_mask = zoom(mask, ratio, order=0, **options)
    resized_foreground = zoom(foreground.astype(np.uint8), ratio, order=0, **options).astype(bool)
    resized_image[~resized_foreground] = 0.0
    if resized_image.shape != target_xyz or resized_mask.shape != target_xyz:
        raise ValueError("Resampling did not produce the requested grid.")
    if not set(int(label) for label in np.unique(resized_mask)).issubset({mapping[raw] for raw in observed}):
        raise ValueError("Nearest-neighbour resampling unexpectedly introduced a label.")
    new_spacing = old_spacing * old_shape / np.asarray(target_xyz)
    # grid_mode=True maps new centre x to old centre (x + .5) * scale - .5.
    transform = np.eye(4)
    scale = old_shape / np.asarray(target_xyz)
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = 0.5 * (scale - 1.0)
    new_affine = image_volume.affine @ transform
    arrays = {
        "image": np.ascontiguousarray(resized_image.transpose(2, 1, 0), dtype=np.float32),
        "mask": np.ascontiguousarray(resized_mask.transpose(2, 1, 0), dtype=np.uint8),
        "spacing": np.asarray(new_spacing[::-1], dtype=np.float32),
    }
    metadata = {
        "original_image": _source_metadata(original_image),
        "original_mask": _source_metadata(original_mask),
        "canonical_image": _source_metadata(image_volume),
        "canonical_mask": _source_metadata(mask_volume),
        "target_shape_xyz": list(target_xyz), "target_shape_dhw": list(target_shape_dhw),
        "resampled_affine_xyz": new_affine.tolist(),
        "spacing_dhw_mm": arrays["spacing"].tolist(),
        "observed_raw_label_ids": sorted(observed),
        "retained_class_indices": [int(label) for label in np.unique(resized_mask)],
        "class_voxel_counts": np.bincount(resized_mask.ravel(), minlength=len(label_ids)).tolist(),
        "normalization": {"method": "per_case_nonzero_zscore_then_clip", "nonzero_count": count,
                          "mean": mean, "standard_deviation": standard_deviation,
                          "minimum_scale": 1e-8, "clip": [-5.0, 5.0],
                          "background": "zero; target support uses matching nearest-neighbour transform"},
        "resampling": {"image_order": 1, "mask_order": 0, **options},
        "metrics_grid": "downsampled_resampled_grid",
        "native_resolution_accuracy_claimed": False,
    }
    return arrays, metadata


def prepare_cache(
    audit_path: str | Path, output_dir: str | Path,
    target_shape_dhw: Sequence[int] = (64, 128, 128),
    label_ids: Sequence[int] = (), limit: int | None = None,
) -> Path:
    """Write pair caches and a manifest only after validating audit partitions."""
    mapping = _label_mapping(label_ids)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("--limit must be a positive number of TRAIN records.")
    audit_path, output_dir = Path(audit_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    audit_bytes = audit_path.read_bytes()
    source = json.loads(audit_bytes)
    if source.get("status") != "audit_complete" or source.get("blockers") != []:
        raise ValueError("Cache preparation requires an audit_complete report without blockers.")
    records = source.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Audit has no pair records.")
    if any(not record.get("pairing_valid") or not record.get("geometry_valid") for record in records):
        raise ValueError("Every audited record must have valid pairing and geometry.")
    splits = source.get("patient_splits")
    validate_patient_splits(splits, {record["patient"] for record in records})
    memberships = {patient: name for name, patients in splits.items() for patient in patients}
    seen = set()
    for record in records:
        pair = (record["patient"], record["visit"])
        if pair in seen:
            raise ValueError("Duplicate patient/visit pair in source audit.")
        seen.add(pair)
    selected = sorted(records, key=lambda record: (record["patient"], record["visit"]))
    if limit is not None:
        selected = [record for record in selected if memberships[record["patient"]] == "train"][:limit]
    if not selected:
        raise ValueError("No training records available for development cache.")
    manifest_path = output_dir / "manifest.json"
    destinations = [output_dir / "volumes" / f"{record['patient']}_{record['visit']}.npz" for record in selected]
    if manifest_path.exists() or any(path.exists() for path in destinations):
        raise FileExistsError("Refusing to overwrite an existing manifest or cache volume; choose a fresh output directory.")
    data_root = Path(source["data_root"]).expanduser().resolve()
    output_records = []
    for record, destination in zip(selected, destinations):
        paths = {}
        for role in ("image", "mask"):
            relative = Path(record[role]["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Audit source paths must be relative to data_root.")
            paths[role] = data_root / relative
        arrays, metadata = preprocess_pair(paths["image"], paths["mask"], target_shape_dhw, label_ids)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **arrays)
        output_records.append({
            "patient": record["patient"], "visit": record["visit"],
            "split": memberships[record["patient"]],
            "cache_path": destination.relative_to(output_dir).as_posix(),
            "source_image": record["image"]["relative_path"],
            "source_mask": record["mask"]["relative_path"], **metadata,
        })
        print(f"Prepared {len(output_records)}/{len(selected)}: {record['patient']}/{record['visit']}")
    manifest = {
        "schema_version": 1, "records": output_records,
        "num_classes": len(label_ids), "label_ids": list(label_ids),
        "raw_label_to_index": {str(raw): index for raw, index in mapping.items()},
        "label_semantics": {"status": "unverified", "mapping": None},
        "target_shape_dhw": list(target_shape_dhw),
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
        "patient_splits": splits, "partial": limit is not None,
        "development_only": limit is not None,
        "permitted_use": "training_smoke_test_only" if limit is not None else "full_patient_disjoint_experiment",
        "metrics_grid": "downsampled_resampled_grid", "native_resolution_accuracy_claimed": False,
        "preprocessing_fit_scope": "each individual image only; no split-level fitting",
        "source_audit_record_count": len(records), "prepared_record_count": len(output_records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=[64, 128, 128], metavar=("D", "H", "W"))
    parser.add_argument("--label-ids", type=int, nargs="+", required=True)
    parser.add_argument("--limit", type=int, help="Development only: process at most N TRAIN records")
    args = parser.parse_args(argv)
    try:
        manifest_path = prepare_cache(args.audit, args.output_dir, args.shape, args.label_ids, args.limit)
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.exit(2, f"Cache preparation blocked: {error}\n")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
