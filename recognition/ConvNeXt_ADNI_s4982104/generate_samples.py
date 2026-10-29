"""Build a read-only ADNI slice manifest using Python's standard library.

Each JPEG gets one row, including files with missing/contradictory metadata.
The supplied JSON has no documented acquisition-date field. scan_date stays
blank; the timestamp embedded in a raw filename is NOT treated as that date.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

FIELDS = ("file_path", "original_split", "image_id", "patient_id",
          "scan_date", "label")
ISSUE_FIELDS = ("file_path", "image_id", "field", "issue", "detail")
LABELS = {"AD": 2, "NC": 0}
DATE_NOTE = (
    "No verified acquisition-date source in the supplied metadata schema; "
    "raw filename timestamps are not assumed to be scan dates."
)


def make_row(path, split, folder_label, metadata):
    """提取可核实的信息；缺失或冲突字段留空，不删除图片记录。"""
    row = dict.fromkeys(FIELDS, "")
    row.update(file_path=str(path), original_split=split)
    issues = []

    def issue(field, code, detail):
        issues.append(dict(file_path=str(path), image_id=row["image_id"],
                           field=field, issue=code, detail=detail))

    match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
    record = None
    if match:
        row["image_id"] = match.group(1)
        record = metadata.get(row["image_id"])
    else:
        issue("image_id", "filename_unparsed",
              "Expected <image_id>_<slice_index>.jpeg or .jpg.")
    if not isinstance(record, dict):
        issue("patient_id,label", "metadata_missing",
              "No metadata object for the image ID; patient and label unverified.")
    else:
        raw = record.get("raw")
        raw = raw if isinstance(raw, str) else ""
        raw_id = re.search(r"_I(\d+)\.nii(?:\.gz)?$", raw)
        patient = re.search(r"ADNI_(\d{3}_S_\d{4})_", raw)
        if raw_id is None:
            issue("patient_id", "raw_image_id_missing",
                  "Cannot cross-check the image ID in the raw source path.")
        elif raw_id.group(1) != row["image_id"]:
            issue("patient_id", "raw_image_id_mismatch",
                  "raw source path image ID disagrees with the JPEG image ID.")
        elif patient:
            row["patient_id"] = patient.group(1)
        else:
            issue("patient_id", "patient_id_unparsed",
                  "No NNN_S_NNNN patient ID in the raw source path.")

        actual = record.get("label")
        if folder_label not in LABELS:
            issue("label", "folder_label_unknown", "Expected AD or NC folder.")
        elif type(actual) is not int or actual not in (0, 2):
            issue("label", "metadata_label_unknown",
                  "Expected integer label 2 (AD) or 0 (NC).")
        elif actual != LABELS[folder_label]:
            issue("label", "label_mismatch",
                  "Folder class disagrees with metadata label; label left blank.")
        else:
            row["label"] = folder_label

    # 当前来源没有经过核实的采集日期字段，禁止从文件名时间戳猜测。
    issue("scan_date", "scan_date_unavailable", DATE_NOTE)
    return row, issues


def build_manifest(data_root, output_dir):
    root = Path(data_root).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source data tree.")
    required_dirs = [
        root / "AD_NC" / split / label
        for split in ("train", "test") for label in LABELS
    ]
    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError("Missing source directory: " + str(directory))
    destinations = [output / name for name in
                    ("samples.csv", "samples_issues.csv", "manifest_summary.json")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Outputs already exist; use a new output directory.")

    metadata_bytes = (root / "meta_data_with_label.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be an object keyed by image ID.")

    rows, issues = [], []

    def walk_error(error):
        # 目录无法读取时明确失败，避免输出看似完整但实际缺文件的表。
        raise error

    for split in ("train", "test"):
        split_root = root / "AD_NC" / split
        for directory, subdirs, filenames in os.walk(
                split_root, onerror=walk_error, followlinks=False):
            subdirs.sort()
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.suffix.lower() not in (".jpeg", ".jpg"):
                    continue
                relative = path.relative_to(split_root)
                folder_label = relative.parts[0] if len(relative.parts) > 1 else ""
                row, found = make_row(path, split, folder_label, metadata)
                rows.append(row)
                issues.extend(found)
    if not rows:
        raise ValueError("No JPEG files found; no manifest written.")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "fields": list(FIELDS),
        "rows_written": len(rows),
        "issues_by_code": dict(sorted(Counter(x["issue"] for x in issues).items())),
        "scan_date_policy": DATE_NOTE,
        "original_split_policy": "Source location only; not a new train/test split.",
        "scope": "Filename/metadata manifest; no pixel decoding or full B3 audit.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for destination, fields, values in (
        (destinations[0], FIELDS, rows),
        (destinations[1], ISSUE_FIELDS, issues),
    ):
        with destination.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    with destinations[2].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/groups/comp3710/ADNI",
                        help="Read-only course ADNI root.")
    parser.add_argument("--output-dir", required=True,
                        help="Personal cluster directory outside the source tree.")
    args = parser.parse_args()
    try:
        summary = build_manifest(args.data_root, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, "Manifest generation failed: " + str(error) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
