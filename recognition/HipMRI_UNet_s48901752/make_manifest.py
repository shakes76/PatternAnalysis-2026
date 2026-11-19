"""Build the patient-level HipMRI scan manifest (metadata.csv).

Ported from ``metadata.py`` (``read_pairs_csv`` + ``assign_splits``) in the
author's earlier COMP3710-Project, which produced the frozen split recorded in
``dataset.EXPECTED_SPLIT_SHA256``.

Procedure (README Section 2.2)
    1. List ``<data-root>/semantic_MRs/*_LFOV.nii.gz`` in sorted filename order.
       Each image ``<scan>_LFOV.nii.gz`` is paired with
       ``<data-root>/semantic_labels_only/<scan>_SEMANTIC.nii.gz``; the patient
       ID is the part of the scan ID before ``_Week`` (e.g. ``B040_Week3`` ->
       ``B040``).
    2. Take patient IDs in order of first appearance, permute them with
       ``numpy.random.default_rng(seed)``, and assign the first 15% (6 of 38)
       to test, the next 15% (6) to validation and the rest (26) to training.
    3. Write one row per scan, then re-read the file with
       ``dataset.load_manifest``, which checks for patient leakage, the 26/6/6
       patient counts and the frozen split SHA-256.

Usage
    python make_manifest.py --data-root /path/to/HipMRI_Study_open --out metadata.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import dataset

DEFAULT_SEED = 3710
IMAGE_DIR = "semantic_MRs"
MASK_DIR = "semantic_labels_only"
IMAGE_SUFFIX = "_LFOV.nii.gz"
MASK_SUFFIX = "_SEMANTIC.nii.gz"
SCAN_PATTERN = re.compile(r"^(?P<patient>.+?)_Week\d+$")
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15


def discover_scans(data_root: str | Path) -> pd.DataFrame:
    """Return one row per scan (patient_id, scan_id, image_path, mask_path), sorted."""
    root = Path(data_root).absolute()
    image_dir, mask_dir = root / IMAGE_DIR, root / MASK_DIR
    images = sorted(image_dir.glob(f"*{IMAGE_SUFFIX}"))
    if not images:
        raise ValueError(f"No images matched {image_dir / ('*' + IMAGE_SUFFIX)}")

    rows = []
    for image_path in images:
        scan_id = image_path.name[: -len(IMAGE_SUFFIX)]
        match = SCAN_PATTERN.match(scan_id)
        if match is None:
            raise ValueError(f"Cannot parse patient ID from {image_path.name}")
        mask_path = mask_dir / f"{scan_id}{MASK_SUFFIX}"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"No mask for {image_path.name}: expected {mask_path}"
            )
        rows.append(
            {
                "patient_id": match["patient"],
                "scan_id": scan_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )

    mask_ids = {p.name[: -len(MASK_SUFFIX)] for p in mask_dir.glob(f"*{MASK_SUFFIX}")}
    orphans = sorted(mask_ids - {row["scan_id"] for row in rows})
    if orphans:
        raise ValueError(f"Masks without images: {orphans[:8]}")
    return pd.DataFrame(rows)


def assign_splits(frame: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Add a patient-level ``split`` column (same rule as the original assign_splits)."""
    patients = frame["patient_id"].astype(str).drop_duplicates().to_numpy()
    count = len(patients)
    n_test = max(1, round(count * TEST_FRACTION))
    n_val = max(1, round(count * VAL_FRACTION))
    if n_test + n_val >= count:
        raise ValueError(f"Too few patients ({count}) for three non-empty splits")

    order = np.random.default_rng(seed).permutation(count)
    test_ids = set(patients[order[:n_test]])
    val_ids = set(patients[order[n_test : n_test + n_val]])
    result = frame.copy()
    result["split"] = result["patient_id"].map(
        lambda p: "test" if p in test_ids else "val" if p in val_ids else "train"
    )
    return result


def build_manifest(data_root: str | Path, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    frame = assign_splits(discover_scans(data_root), seed=seed)
    return frame[list(dataset.REQUIRED_COLUMNS)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=f"folder containing {IMAGE_DIR}/ and {MASK_DIR}/",
    )
    parser.add_argument("--out", type=Path, required=True, help="output CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    frame = build_manifest(args.data_root, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    # Re-read through the same checks every training/prediction script uses.
    # The frozen hash only applies to the default seed.
    expected = dataset.EXPECTED_SPLIT_SHA256 if args.seed == DEFAULT_SEED else None
    checked = dataset.load_manifest(args.out, expected_sha256=expected)
    patients = checked.groupby("split")["patient_id"].nunique()
    scans = checked["split"].value_counts()
    print(f"Wrote {len(checked)} scans to {args.out}")
    for split in dataset.SPLITS:
        print(
            f"  {split:5s}: {patients.get(split, 0)} patients, {scans.get(split, 0)} scans"
        )
    print(f"split_sha256: {dataset.split_sha256(checked)}")
    if expected is not None:
        print("Matches dataset.EXPECTED_SPLIT_SHA256 (frozen split).")


if __name__ == "__main__":
    main()
