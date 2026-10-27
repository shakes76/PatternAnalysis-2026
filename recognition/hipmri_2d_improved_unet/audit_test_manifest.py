"""Audit frozen test inputs using filenames, headers and compressed file bytes only."""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import nibabel as nib
import numpy as np

from dataset import AFFINE_ATOL, AFFINE_RTOL, collect_pairs, patient_ids, validate_pair_geometry
from train import manifest_hash


PATTERN = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")
SPLIT_FOLDERS = {
    "training": ("keras_slices_train", "keras_slices_seg_train"),
    "validation": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}


def sha256_file(path):
    """Hash original file bytes without decoding NIfTI voxel arrays."""
    path = Path(path)
    before = path.stat()
    digest, byte_count = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    after = path.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or byte_count != after.st_size):
        raise ValueError(f"File changed while hashing: {path.name}")
    return digest.hexdigest(), byte_count


def pair_metadata(pairs):
    """Reject numeric aliases and malformed basenames, retaining manifest order."""
    rows, seen = [], set()
    for image, mask in pairs:
        match = PATTERN.fullmatch(image.name)
        if match is None or mask.name != image.name.replace("case_", "seg_", 1):
            raise ValueError(f"Unexpected MRI/mask basename pair: {image.name}|{mask.name}")
        key = tuple(map(int, match.groups()))
        if key in seen:
            raise ValueError(f"Duplicate numeric case/week/slice key: {key}")
        seen.add(key)
        rows.append({"image": image.name, "mask": mask.name,
                     "case": key[0], "week": key[1], "slice": key[2]})
    return rows


def collect_checked_splits(data_root, prior_plan, test_folders=None):
    """Read train/validation/test names only; reject mismatches and case leakage."""
    folders = dict(SPLIT_FOLDERS)
    if test_folders is not None:
        if len(test_folders) != 2 or any(Path(name).name != name or name in ("", ".", "..") for name in test_folders):
            raise ValueError("Test folder names must be basenames inside data-root.")
        folders["test"] = tuple(test_folders)
    pairs, summaries = {}, {}
    for split, (image_folder, mask_folder) in folders.items():
        paths = collect_pairs(Path(data_root) / image_folder, Path(data_root) / mask_folder)
        metadata = pair_metadata(paths)
        groups = sorted(patient_ids(paths))
        digest = manifest_hash(paths)
        if split != "test":
            expected = prior_plan["data"][split]
            if digest != expected["manifest_sha256"] or len(paths) != expected["pairs"]:
                raise ValueError(f"{split} filename manifest differs from the prior frozen plan.")
        pairs[split] = paths
        summaries[split] = {"pairs": len(paths), "case_proxy_ids": groups,
                            "case_proxy_groups": len(groups), "manifest_sha256": digest,
                            "image_folder": image_folder, "mask_folder": mask_folder,
                            "numeric_keys_unique": len(metadata) == len(paths)}
    overlaps = {}
    for left, right in (("training", "validation"), ("training", "test"), ("validation", "test")):
        shared = sorted(set(summaries[left]["case_proxy_ids"]) & set(summaries[right]["case_proxy_ids"]))
        overlaps[f"{left}__{right}"] = shared
    if any(overlaps.values()):
        raise ValueError(f"Numeric case-group leakage between splits: {overlaps}")
    return pairs, summaries, overlaps


def audit_test_headers_and_hashes(pairs, progress_every=1000):
    """No get_fdata/dataobj materialization; hash compressed streams after headers."""
    if progress_every <= 0:
        raise ValueError("progress-every must be positive.")
    metadata = pair_metadata(pairs)
    sizes, units, forms, matrices = Counter(), Counter(), Counter(), Counter()
    summary = {"processed_count": 0, "passed_count": 0, "error_count": 0, "errors": [],
               "all_pairs_attempted": False, "max_affine_abs_difference": 0.0,
               "exact_affine_matches": 0, "identity_affine_pairs": 0}
    files = []
    for (image_path, mask_path), row in zip(pairs, metadata):
        entry = dict(row)
        try:
            image = nib.load(image_path, mmap="r", keep_file_open=False)
            mask = nib.load(mask_path, mmap="r", keep_file_open=False)
            difference = validate_pair_geometry(image, mask)
            image_hash, image_bytes = sha256_file(image_path)
            mask_hash, mask_bytes = sha256_file(mask_path)
            entry.update(image_sha256=image_hash, mask_sha256=mask_hash,
                         image_bytes=image_bytes, mask_bytes=mask_bytes)
            summary["max_affine_abs_difference"] = max(summary["max_affine_abs_difference"], difference)
            summary["exact_affine_matches"] += int(np.array_equal(image.affine, mask.affine))
            summary["identity_affine_pairs"] += int(np.array_equal(image.affine, np.eye(4)) and np.array_equal(mask.affine, np.eye(4)))
            sizes["x".join(map(str, image.shape))] += 1
            units[f"MRI:{image.header.get_xyzt_units()[0]}|mask:{mask.header.get_xyzt_units()[0]}"] += 1
            forms[f"MRI:q{int(image.header['qform_code'])},s{int(image.header['sform_code'])}|mask:q{int(mask.header['qform_code'])},s{int(mask.header['sform_code'])}"] += 1
            matrices[tuple(image.affine.reshape(-1).tolist())] += 1
            summary["passed_count"] += 1
            entry["status"] = "passed"
        except Exception as error:
            entry.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            summary["error_count"] += 1
            summary["errors"].append({"image": image_path.name, "mask": mask_path.name, **entry["error"]})
        files.append(entry)
        summary["processed_count"] += 1
        if summary["processed_count"] % progress_every == 0 or summary["processed_count"] == len(pairs):
            print(f"Test header/byte audit: {summary['processed_count']}/{len(pairs)}; errors={summary['error_count']}", flush=True)
    summary.update(all_pairs_attempted=summary["processed_count"] == len(pairs),
                   shape_counts=dict(sorted(sizes.items())), declared_spatial_unit_pairs=dict(sorted(units.items())),
                   qform_sform_code_pairs=dict(sorted(forms.items())),
                   selected_mri_affine_matrices=[{"matrix": np.asarray(values).reshape(4, 4).tolist(), "count": count} for values, count in sorted(matrices.items())])
    content_manifest = None
    if not summary["error_count"] and summary["all_pairs_attempted"]:
        text = "\n".join(f"{row['image']}|{row['mask']}|{row['image_sha256']}|{row['mask_sha256']}" for row in files)
        content_manifest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return summary, files, content_manifest


def verify_test_content(pairs, audit):
    """Recheck frozen compressed bytes only; no header/pixel decode or forward."""
    if audit.get("status") != "passed_full":
        raise ValueError("Only a full successful test input audit may be used.")
    if manifest_hash(pairs) != audit["splits"]["test"]["manifest_sha256"]:
        raise ValueError("Test filename manifest changed after the input audit.")
    metadata = pair_metadata(pairs)
    expected = audit["test_files"]
    if len(expected) != len(pairs):
        raise ValueError("Test pair count changed after the input audit.")
    rows = []
    for (image, mask), actual, frozen in zip(pairs, metadata, expected):
        if any(actual[key] != frozen[key] for key in actual) or frozen.get("status") != "passed":
            raise ValueError("Test pair identity/order differs from the successful audit.")
        image_sha, image_bytes = sha256_file(image)
        mask_sha, mask_bytes = sha256_file(mask)
        if (image_sha, image_bytes, mask_sha, mask_bytes) != (
                frozen["image_sha256"], frozen["image_bytes"], frozen["mask_sha256"], frozen["mask_bytes"]):
            raise ValueError(f"Test compressed file bytes changed: {image.name}|{mask.name}")
        rows.append(f"{image.name}|{mask.name}|{image_sha}|{mask_sha}")
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    if digest != audit["test_content_manifest_sha256"]:
        raise ValueError("Test content manifest differs from the successful audit.")
    return {"pairs": len(pairs), "compressed_files_verified": 2 * len(pairs),
            "content_manifest_sha256": digest, "all_compressed_files_match": True,
            "voxel_arrays_materialized": False, "model_forward_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prior-plan", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-image-folder", default=SPLIT_FOLDERS["test"][0])
    parser.add_argument("--test-mask-folder", default=SPLIT_FOLDERS["test"][1])
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    if args.output.exists():
        raise FileExistsError("Use a new audit output to preserve prior evidence.")
    report = {"audit_kind": "frozen_test_manifest_header_and_compressed_bytes", "status": "running",
              "created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
              "performed_by": "Codex automation; not user verification", "test_split_metadata_read": True,
              "voxel_arrays_materialized": False, "model_forward_calls": 0,
              "affine_policy": {"atol": AFFINE_ATOL, "rtol": AFFINE_RTOL},
              "claim_boundary": "Filename case fields are grouping proxies, not independently confirmed patient identity. Headers/byte hashes do not prove anatomical alignment, physical units, label semantics or absence of prior test exposure.",
              "errors": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    started = time.perf_counter()
    try:
        prior = json.loads(args.prior_plan.read_text(encoding="utf-8"))
        report["prior_plan_sha256"] = sha256_file(args.prior_plan)[0]
        report["frozen_plan_sha256"] = sha256_file(args.frozen_plan)[0]
        pairs, summaries, overlaps = collect_checked_splits(args.data_root, prior, (args.test_image_folder, args.test_mask_folder))
        report["splits"], report["pairwise_case_overlap"] = summaries, overlaps
        headers, files, digest = audit_test_headers_and_hashes(pairs["test"], args.progress_every)
        report.update(test_header_geometry=headers, test_files=files, test_content_manifest_sha256=digest,
                      content_manifest_definition="SHA256 of newline-joined image_basename|mask_basename|image_compressed_SHA256|mask_compressed_SHA256 in existing lexicographic pair order; no trailing newline")
        report["status"] = "passed_full" if headers["error_count"] == 0 and headers["all_pairs_attempted"] else "failed"
        if sha256_file(args.prior_plan)[0] != report["prior_plan_sha256"] or sha256_file(args.frozen_plan)[0] != report["frozen_plan_sha256"]:
            raise ValueError("A plan changed during the audit.")
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        project = Path(__file__).resolve().parent
        report["source_sha256"] = {name: sha256_file(project / name)[0] for name in ("dataset.py", "train.py", "audit_test_manifest.py")}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Frozen test input audit: {report['status']}; {args.output.resolve()}", flush=True)
    return 0 if report["status"] == "passed_full" else 1


if __name__ == "__main__":
    raise SystemExit(main())
