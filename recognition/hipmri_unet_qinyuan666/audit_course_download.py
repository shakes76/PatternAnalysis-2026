"""Read-only acquisition checks for course slices, without creating a manifest.

Public output contains aggregates only. Filenames, case IDs, hashes and error
details go exclusively to the explicit private output. Case IDs parsed from
names are grouping evidence, not independently verified patient identities.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import re
import zlib

import nibabel as nib
import numpy as np


SPLITS = ("train", "validate", "test")


def inspect_pair(item: tuple) -> dict:
    split, key, image_path, mask_path = item
    result = {"split": split, "key": list(key), "files": [], "errors": []}
    loaded = []
    for role, path in (("image", image_path), ("mask", mask_path)):
        try:
            compressed = path.read_bytes()
            # Reading the complete gzip stream checks its trailer CRC and length,
            # including bytes beyond those needed to load the NIfTI array.
            payload = gzip.decompress(compressed)
            nii = nib.Nifti1Image.from_bytes(payload)
            array = nii.get_fdata()
            result["files"].append({"role": role, "name": path.name,
                                    "bytes": len(compressed),
                                    "sha256": hashlib.sha256(compressed).hexdigest()})
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            if array.ndim != 2 or min(array.shape) < 2:
                result["errors"].append(role + "_not_2d")
            if not np.isfinite(array).all() or not np.isfinite(nii.affine).all():
                result["errors"].append(role + "_nonfinite")
            loaded.append((nii, array))
        except (OSError, ValueError, EOFError, zlib.error, nib.filebasedimages.ImageFileError) as exc:
            result["errors"].append(role + "_read_error")
            result.setdefault("error_details", []).append(str(exc))
            loaded.append(None)
    if any(value is None for value in loaded):
        return result
    (image, x), (mask, y) = loaded
    if x.shape != y.shape:
        result["errors"].append("shape_mismatch")
    if not np.allclose(image.affine, mask.affine, rtol=0, atol=1e-5):
        result["errors"].append("affine_mismatch")
    if not np.isfinite(y).all() or not np.array_equal(y, np.rint(y)):
        result["errors"].append("mask_not_finite_integer")
    else:
        result["mask_values"] = [int(value) for value in np.unique(y)]
    result["shape"] = "x".join(map(str, x.shape))
    result["image_dtype"] = str(image.get_data_dtype())
    result["mask_dtype"] = str(mask.get_data_dtype())
    result["spatial_units"] = [image.header.get_xyzt_units()[0], mask.header.get_xyzt_units()[0]]
    result["identity_affines"] = bool(np.array_equal(image.affine, np.eye(4))
                                      and np.array_equal(mask.affine, np.eye(4)))
    return result


def audit_course(root: Path, workers: int = 4, progress=None) -> tuple[dict, dict]:
    root = Path(root).resolve()
    if not root.is_dir() or workers < 1:
        raise ValueError("An existing course directory and positive worker count are required.")
    expected = {"keras_slices_" + prefix + split
                for split in SPLITS for prefix in ("", "seg_")}
    issues = []
    for path in root.iterdir():
        if path.name not in expected or not path.is_dir() or path.is_symlink():
            issues.append({"kind": "unexpected_root_entry", "path": path.name})
    pairs, split_summary, case_sets, private_inventory = [], {}, {}, []
    total_files = total_bytes = 0
    for split in SPLITS:
        tables = []
        for role, folder_prefix, file_prefix in (("image", "", "case"), ("mask", "seg_", "seg")):
            folder = root / ("keras_slices_" + folder_prefix + split)
            table = {}
            if not folder.is_dir() or folder.is_symlink():
                issues.append({"kind": "missing_or_symlink_directory", "path": folder.name})
            else:
                for path in sorted(folder.iterdir()):
                    match = re.fullmatch(file_prefix + r"_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz", path.name)
                    if path.is_symlink() or not path.is_file() or not match:
                        issues.append({"kind": "unexpected_entry", "path": str(path.relative_to(root))})
                        continue
                    key = tuple(map(int, match.groups()))
                    total_files += 1
                    total_bytes += path.stat().st_size
                    if key in table:
                        issues.append({"kind": "duplicate_normalized_key", "path": str(path.relative_to(root))})
                    else:
                        table[key] = path
            if not table:
                issues.append({"kind": "empty_expected_directory", "path": folder.name})
            tables.append(table)
        images, masks = tables
        all_keys = images.keys() | masks.keys()
        common = images.keys() & masks.keys()
        for key in sorted(images.keys() ^ masks.keys()):
            issues.append({"kind": "unpaired_key", "split": split, "key": list(key)})
        case_sets[split] = {key[0] for key in all_keys}
        scans = Counter(key[:2] for key in all_keys)
        split_summary[split] = {
            "images": len(images), "masks": len(masks), "matched_pairs": len(common),
            "case_groups_from_names": len(case_sets[split]), "case_week_groups": len(scans),
            "slice_counts_per_case_week": dict(Counter(map(str, scans.values()))),
        }
        pairs.extend((split, key, images[key], masks[key]) for key in sorted(common))
        private_inventory.append({"split": split, "case_groups": sorted(case_sets[split])})
    overlap_counts, private_overlaps = {}, {}
    for a, b in (("train", "validate"), ("train", "test"), ("validate", "test")):
        shared = sorted(case_sets[a] & case_sets[b])
        overlap_counts[a + "_" + b] = len(shared)
        private_overlaps[a + "_" + b] = shared
    shapes, image_dtypes, mask_dtypes, units, errors = (Counter() for _ in range(5))
    labels, pair_results = set(), []
    identity_count = valid_count = files_read = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(inspect_pair, pairs), 1):
            pair_results.append(result)
            files_read += len(result["files"])
            errors.update(result["errors"])
            if not result["errors"]:
                valid_count += 1
                shapes[result["shape"]] += 1
                image_dtypes[result["image_dtype"]] += 1
                mask_dtypes[result["mask_dtype"]] += 1
                units.update(result["spatial_units"])
                labels.update(result["mask_values"])
                identity_count += int(result["identity_affines"])
            if progress and (index % 1000 == 0 or index == len(pairs)):
                progress(index, len(pairs))
    summary = {
        "scope": "local course acquisition check; no training, resplitting or anatomical validation",
        "local_integrity_passed": bool(pairs) and not issues and not errors,
        "source_manifest_or_hashes_compared": False,
        "files_observed": total_files, "compressed_file_bytes": total_bytes,
        "pairs_checked": len(pairs), "pairs_passed": valid_count,
        "gzip_and_nifti_files_read": files_read,
        "splits": split_summary,
        "case_groups_from_names": len(set().union(*case_sets.values())),
        "case_group_overlap_counts": overlap_counts,
        "case_groups_disjoint_by_filename": not any(overlap_counts.values()),
        "patient_identity_mapping_independently_verified": False,
        "inventory_issue_counts": dict(Counter(issue["kind"] for issue in issues)),
        "pair_error_counts": dict(errors),
        "valid_pair_shapes": dict(shapes),
        "image_dtypes": dict(image_dtypes), "mask_dtypes": dict(mask_dtypes),
        "observed_mask_values": sorted(labels),
        "pairs_with_identity_affines": identity_count,
        "spatial_units_file_counts": dict(units),
        "global_label_semantics_verified": False,
        "physical_spacing_verified": False,
        "training_ready": False,
    }
    private = {"root": str(root), "inventory_issues": issues,
               "case_inventory": private_inventory, "case_overlaps": private_overlaps,
               "pairs": pair_results}
    return summary, private


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # Reports cannot overwrite or alter the downloaded data tree.
    for destination in (args.output, args.private_output):
        if destination.resolve().is_relative_to(args.root.resolve()):
            parser.error("Write reports outside the downloaded data directory.")
    if args.output.resolve() == args.private_output.resolve():
        parser.error("Public and private outputs must differ.")
    report, private = audit_course(args.root, args.workers,
                                  lambda done, total: print(f"Checked {done}/{total} pairs", flush=True))
    for destination, content in ((args.output, report), (args.private_output, private)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["local_integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
