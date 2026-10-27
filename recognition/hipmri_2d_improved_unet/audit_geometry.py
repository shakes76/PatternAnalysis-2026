"""Supplement prior audits with train/validation NIfTI header geometry checks."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np

from dataset import AFFINE_ATOL, AFFINE_RTOL, collect_pairs, patient_ids, validate_pair_geometry


def audit_split(pairs, limit, progress_every):
    selected = pairs if limit is None else pairs[:limit]
    summary = {
        "total_pairs": len(pairs), "selected_count": len(selected),
        "case_group_count": len(patient_ids(pairs)), "processed_count": 0,
        "passed_count": 0, "error_count": 0, "errors": [],
        "all_pairs_attempted": False, "max_affine_abs_difference": 0.0,
        "exact_affine_matches": 0, "identity_affine_pairs": 0,
    }
    sizes, units, form_codes, matrices = Counter(), Counter(), Counter(), Counter()
    for image_path, mask_path in selected:
        try:
            image = nib.load(image_path, mmap="r", keep_file_open=False)
            mask = nib.load(mask_path, mmap="r", keep_file_open=False)
            difference = validate_pair_geometry(image, mask)
            summary["max_affine_abs_difference"] = max(summary["max_affine_abs_difference"], difference)
            summary["exact_affine_matches"] += int(np.array_equal(image.affine, mask.affine))
            summary["identity_affine_pairs"] += int(
                np.array_equal(image.affine, np.eye(4)) and np.array_equal(mask.affine, np.eye(4))
            )
            sizes["x".join(str(value) for value in image.shape)] += 1
            units[f"MRI:{image.header.get_xyzt_units()[0]}|mask:{mask.header.get_xyzt_units()[0]}"] += 1
            form_codes[
                f"MRI:q{int(image.header['qform_code'])},s{int(image.header['sform_code'])}|"
                f"mask:q{int(mask.header['qform_code'])},s{int(mask.header['sform_code'])}"
            ] += 1
            matrices[tuple(image.affine.reshape(-1).tolist())] += 1
            summary["passed_count"] += 1
        except Exception as error:
            summary["error_count"] += 1
            summary["errors"].append({
                "image": image_path.name, "mask": mask_path.name,
                "type": type(error).__name__, "message": str(error),
            })
            print(f"Geometry ERROR: {image_path.name}: {error}", flush=True)
        summary["processed_count"] += 1
        if summary["processed_count"] % progress_every == 0 or summary["processed_count"] == len(selected):
            print(f"Geometry: {summary['processed_count']}/{len(selected)}; errors={summary['error_count']}", flush=True)
    summary["all_pairs_attempted"] = summary["processed_count"] == len(pairs)
    summary["shape_counts"] = dict(sorted(sizes.items()))
    summary["declared_spatial_unit_pairs"] = dict(sorted(units.items()))
    summary["qform_sform_code_pairs"] = dict(sorted(form_codes.items()))
    summary["selected_mri_affine_matrices"] = [
        {"matrix": np.asarray(values).reshape(4, 4).tolist(), "count": count}
        for values, count in sorted(matrices.items())
    ]
    manifest = "\n".join(f"{image.name}|{mask.name}" for image, mask in selected)
    summary["selected_manifest_sha256"] = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parents[4]
    parser.add_argument("--data-root", type=Path, default=workspace / "Final Report/HipMRI_2D_Improved_UNet_学习/data/keras_slices_data")
    parser.add_argument("--output", type=Path, default=Path("outputs/geometry_audit.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.progress_every <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("progress-every and any limit must be positive")
    if args.output.exists():
        raise FileExistsError("Use a new audit output path to preserve previous evidence.")
    report = {
        "audit_kind": "supplementary_nifti_header_geometry", "status": "running",
        "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
        "performed_by": "Codex automation; not user verification",
        "scope": ["training", "validation"], "test_split_read": False,
        "voxel_arrays_materialized": False,
        "affine_policy": {"atol": AFFINE_ATOL, "rtol": AFFINE_RTOL,
                          "basis": "selected NIfTI affine; absolute tolerance in its coordinate units, not assumed mm"},
        "claim_boundary": "Header consistency only. Equal selected affines do not independently prove anatomical alignment, correct physical units, or valid original geometry. This is a new audit; prior preprocessing evidence did not check affines.",
        "limit_per_split": args.limit, "splits": {}, "setup_errors": [],
    }
    started = time.perf_counter()
    try:
        pairs = {
            "training": collect_pairs(args.data_root / "keras_slices_train", args.data_root / "keras_slices_seg_train"),
            "validation": collect_pairs(args.data_root / "keras_slices_validate", args.data_root / "keras_slices_seg_validate"),
        }
        overlap = patient_ids(pairs["training"]) & patient_ids(pairs["validation"])
        report["shared_case_ids"] = sorted(overlap)
        if overlap:
            raise ValueError("Training and validation have shared numeric case groups.")
        for split, paths in pairs.items():
            print(f"{split}: supplementary header check of {len(paths)} pairs", flush=True)
            report["splits"][split] = audit_split(paths, args.limit, args.progress_every)
        failed = any(item["error_count"] for item in report["splits"].values())
        full = all(item["all_pairs_attempted"] for item in report["splits"].values())
        report["status"] = "failed" if failed else ("passed_full" if full else "passed_limited")
    except Exception as error:
        report["status"] = "failed"
        report["setup_errors"].append({"type": type(error).__name__, "message": str(error)})
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("dataset.py", "audit_geometry.py", "tests/test_geometry.py", "tests/test_preprocessing.py")
    }
    report["source_hash_basis"] = "Verified working-tree file bytes before Git line-ending normalization."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Geometry audit: {report['status']}; {args.output.resolve()}", flush=True)
    return 0 if report["status"] in ("passed_full", "passed_limited") else 1


if __name__ == "__main__":
    raise SystemExit(main())
