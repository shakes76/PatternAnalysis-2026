"""Freeze existing course groups into a private CSV after checking audit hashes.

No split is randomized, no slice is selected by model performance, and no model
is run. patient_id is the dataset's pseudonymous Case grouping convention, not
an independently verified identity. All weeks/slices of a Case stay together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

from dataset import read_manifest, validate_label_values, validate_patient_splits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_manifest(root: Path, summary_path: Path, private_path: Path,
                     labels_path: Path, manifest: Path, progress=None) -> dict:
    root, manifest = root.resolve(), manifest.resolve()
    if manifest.is_relative_to(root) or manifest.suffix != ".csv":
        raise ValueError("The private CSV must be outside the source data directory.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    config = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = validate_label_values(config["label_values"])
    if config["prostate_raw_label"] == 0 or config["prostate_raw_label"] not in labels:
        raise ValueError("The dictionary must specify a nonzero prostate label.")
    if not summary["local_integrity_passed"] or not summary["case_groups_disjoint_by_filename"]:
        raise ValueError("A successful, case-disjoint acquisition audit is required.")
    if Path(private["root"]).resolve() != root:
        raise ValueError("The private audit describes a different data root; rerun the audit.")
    pairs = private["pairs"]
    if len(pairs) != summary["pairs_checked"] or not pairs:
        raise ValueError("Public and private acquisition records disagree.")
    directories = {"keras_slices_" + prefix + split for split in ("train", "validate", "test") for prefix in ("", "seg_")}
    if {path.name for path in root.iterdir()} != directories:
        raise ValueError("Course directories differ from the acquisition contract.")
    actual_files = set()
    for name in directories:
        folder = root / name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Expected regular course directories.")
        for path in folder.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ValueError("Unexpected nested entry or symlink in course data.")
            actual_files.add(path.resolve())
    records, expected_files, keys = [], set(), set()
    for index, pair in enumerate(pairs, 1):
        split = pair["split"]
        key = tuple(pair["key"])
        if split not in ("train", "validate", "test") or len(key) != 3:
            raise ValueError("Invalid acquisition split/key.")
        if any(type(value) is not int or value < 0 for value in key):
            raise ValueError("Case/week/slice keys must be nonnegative integers.")
        if key in keys or pair["errors"]:
            raise ValueError("Repeated slice key or failed acquisition pair.")
        keys.add(key)
        if not set(pair["mask_values"]).issubset(labels):
            raise ValueError("Acquisition labels are outside the project dictionary.")
        row = {"patient_id": f"case_{key[0]:03d}", "split": "val" if split == "validate" else split}
        if len(pair["files"]) != 2 or {f["role"] for f in pair["files"]} != {"image", "mask"}:
            raise ValueError("Each acquisition pair requires one image and one mask.")
        for item in pair["files"]:
            role, name = item["role"], item["name"]
            prefix = "case" if role == "image" else "seg"
            match = re.fullmatch(prefix + r"_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz", name)
            if not match or tuple(map(int, match.groups())) != key:
                raise ValueError("Filename and recorded acquisition key disagree.")
            path = root / ("keras_slices_" + ("" if role == "image" else "seg_") + split) / name
            if path.resolve() not in actual_files:
                raise ValueError("A source file is missing.")
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("A source file changed since the acquisition audit; rerun it.")
            expected_files.add(path.resolve())
            row[role] = str(path.resolve())
        records.append(row)
        if progress and (index % 2000 == 0 or index == len(pairs)):
            progress(index, len(pairs))
    if expected_files != actual_files or len(actual_files) != summary["files_observed"]:
        raise ValueError("Local file inventory differs from the audited pairs.")
    validate_patient_splits(records)
    if {row["split"] for row in records} != {"train", "val", "test"}:
        raise ValueError("All three course groups must be populated.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("image", "mask", "patient_id", "split"), lineterminator="\n")
    writer.writeheader()
    for row in records:
        writer.writerow({**row, **{role: os.path.relpath(row[role], manifest.parent).replace("\\", "/") for role in ("image", "mask")}})
    payload = stream.getvalue().encode("utf-8")
    if manifest.exists() and manifest.read_bytes() != payload:
        raise FileExistsError("Refusing to overwrite a different frozen manifest; use a new versioned path.")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        with manifest.open("xb") as output:
            output.write(payload)
    checked = read_manifest(manifest)
    return {
        "status": "course_manifest_prepared_no_real_training_run",
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "label_dictionary_sha256": sha256(labels_path),
        "acquisition_summary_sha256": sha256(summary_path),
        "rows": len(checked), "source_file_hashes_rechecked": len(actual_files),
        "splits": {split: {"slices": sum(row["split"] == split for row in checked),
                           "case_groups": len({row["patient_id"] for row in checked if row["split"] == split})}
                   for split in ("train", "val", "test")},
        "split_policy": "preserve course folders; validate renamed to val; all weeks/slices grouped by original Case identifier",
        "patient_id_basis": "pseudonymous Case grouping convention matched to original case/week names; not independent verification of personal identity",
        "label_values": list(labels), "prostate_raw_label": config["prostate_raw_label"],
        "input_policy": {"image_size": 256, "resize_mode": "pad", "normalization": "per-slice nonzero z-score clipped to [-5,5]"},
        "test_used_for_model_selection": False, "real_data_training_run": False,
        "physical_spacing_verified": False,
    }


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project / "private_data/keras_slices_data")
    parser.add_argument("--audit-summary", type=Path, default=project / "results/course_download_audit.json")
    parser.add_argument("--audit-private", type=Path, default=project / "private_data/source_metadata/course_download_audit_private.json")
    parser.add_argument("--labels", type=Path, default=project / "labels.json")
    parser.add_argument("--manifest", type=Path, default=project / "private_data/manifests/course_v1.csv")
    parser.add_argument("--report", type=Path, default=project / "results/data_preparation.json")
    args = parser.parse_args()
    protected = {path.resolve() for path in (args.manifest, args.audit_summary, args.audit_private, args.labels)}
    if args.report.resolve() in protected or args.report.resolve().is_relative_to(args.root.resolve()):
        parser.error("The report must not overwrite inputs or source data.")
    report = prepare_manifest(args.root, args.audit_summary, args.audit_private, args.labels, args.manifest,
                              lambda done, total: print(f"Rechecked {done}/{total} source pairs", flush=True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
