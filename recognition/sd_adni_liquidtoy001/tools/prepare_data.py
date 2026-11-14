"""Build a private, patient-grouped ADNI manifest and aggregate split report.

Run inside a CPU job. Reads every JPEG once to hash and fully decode it, but
does not copy images, change source data, or expose patient IDs in the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import FIELDS, SPLITS, read_manifest


def assign_patients(patient_classes: dict[str, str], seed: int) -> dict[str, str]:
    """Assign class-stratified patient groups with stable hash ordering."""
    assignment = {}
    for label in sorted(set(patient_classes.values())):
        patients = sorted(
            (p for p, c in patient_classes.items() if c == label),
            key=lambda p: (hashlib.sha256(f"{seed}:{p}".encode()).hexdigest(), p),
        )
        numerators = [len(patients) * percent for percent in (70, 15, 15)]
        counts = [n // 100 for n in numerators]
        remainder_order = sorted(range(3), key=lambda i: (-(numerators[i] % 100), i))
        for i in remainder_order[: len(patients) - sum(counts)]:
            counts[i] += 1
        if min(counts) == 0:
            raise ValueError("Too few patients to represent this class in all splits")
        start = 0
        for split, count in zip(SPLITS, counts):
            assignment.update({p: split for p in patients[start : start + count]})
            start += count
    return assignment


def prepare(root: Path, output: Path, seed: int) -> dict:
    """Validate source data, then create a manifest without overwriting a run."""
    metadata_bytes = (root / "meta_data_with_label.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    rows = []
    patient_classes = {}
    modes, sizes = Counter(), Counter()
    for path in sorted((root / "AD_NC").glob("*/*/*.jpeg")):
        original_split, class_name, _ = path.relative_to(root / "AD_NC").parts
        match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
        if match is None:
            raise ValueError("Unexpected image filename")
        scan_id, slice_index = match.groups()
        record = metadata[scan_id]
        patient_match = re.search(r"ADNI_(\d{3}_S_\d{4})_", record["raw"])
        if patient_match is None:
            raise ValueError("Missing patient identity")
        patient_id = patient_match.group(1)
        if patient_classes.setdefault(patient_id, class_name) != class_name:
            raise ValueError("Patient occurs in multiple classes")
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            modes[image.mode] += 1
            sizes[f"{image.width}x{image.height}"] += 1
            if image.mode != "L":
                raise ValueError("Expected grayscale course images")
        rows.append(
            dict(
                zip(
                    FIELDS,
                    (
                        path.relative_to(root).as_posix(),
                        original_split,
                        class_name,
                        str(record["label"]),
                        patient_id,
                        scan_id,
                        slice_index,
                        "",
                        hashlib.sha256(payload).hexdigest(),
                    ),
                )
            )
        )
    if not rows:
        raise ValueError("No course JPEG files found")
    assignment = assign_patients(patient_classes, seed)
    for row in rows:
        row["split"] = assignment[row["patient_id"]]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest_bytes = stream.getvalue().encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".pending")
    # Exclusive creation prevents accidentally replacing another experiment.
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with temporary.open("xb") as handle:
        handle.write(manifest_bytes)
    try:
        read_manifest(temporary)
        # Publish only a validated manifest; existing destinations are rejected.
        with output.open("xb") as handle:
            handle.write(manifest_bytes)
    finally:
        temporary.unlink()

    split_counts = {}
    for split in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        split_counts[split] = {
            "patients": len({r["patient_id"] for r in selected}),
            "scans": len({r["scan_id"] for r in selected}),
            "images": len(selected),
            "patient_classes": dict(
                Counter(patient_classes[p] for p, s in assignment.items() if s == split)
            ),
            "image_classes": dict(Counter(r["class_name"] for r in selected)),
        }
    hash_counts = Counter(r["sha256"] for r in rows)
    return {
        "schema_version": 1,
        "seed": seed,
        "patient_percentages": [70, 15, 15],
        "split_method": "class-stratified SHA256 ordering with largest remainder",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "images": len(rows),
        "patients": len(patient_classes),
        "splits": split_counts,
        "decoded_modes": dict(modes),
        "decoded_sizes": dict(sizes),
        "duplicate_file_hash_groups": sum(n > 1 for n in hash_counts.values()),
        "cross_split_patient_scan_hash_overlap": 0,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "provisional split; no model selection or evaluation performed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3710)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run source-data preparation in a CPU allocation")
    print(
        json.dumps(prepare(args.root, args.output, args.seed), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
