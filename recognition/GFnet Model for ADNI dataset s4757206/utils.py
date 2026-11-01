"""Prepare ADNI inventories and patient-independent CSVs from extracted images.

Place beside the project's original dataset.py. This entry point reproduces
01_data_exploration and the final 70/15/15 workflow in 02_preprocessing.
It does not extract archives, alter images, or verify diagnostic annotations.

Workflow:
    1. Scan the four original class directories in deterministic order.
    2. Match image IDs to patient IDs extracted from metadata raw paths.
    3. Delegate duplicate exclusion and patient-level splitting to dataset.py.
    4. Optionally compare all generated records with existing notebook CSVs.
    5. Save outputs in a new directory without overwriting previous results.

Dependencies:
    pandas is required for inventory construction and CSV comparison.
    Running preparation also imports dataset.py and therefore requires that
    module's dependencies, including scikit-learn, Pillow, torch and torchvision.

Outputs under --output-dir:
    processed/image_records.csv: original inventory, including the known duplicate.
    splits/train.csv, val.csv, test.csv: cleaned patient-independent partitions.
    preparation_summary.json: counts and optional reference-comparison status.

The files are records of existing images; no image pixels are copied or changed.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Keep this order aligned with the tuple returned by create_patient_splits.
SPLITS = ("train", "val", "test")
# Preserve the original notebook column order for exact CSV comparisons.
COLUMNS = ("split", "label", "filename", "image_id", "path", "patient_id")


def build_inventory(image_root, metadata_path, source_root=None):
    """Scan train/test × AD/NC in notebook order and attach patient IDs.

    image_root must contain the extracted train/AD, train/NC, test/AD and
    test/NC directories. source_root optionally controls the path prefix stored
    in CSVs (e.g. the original Drive root when scanning a local Colab cache).
    Labels come from folders, not the numeric metadata label field.

    Args:
        image_root: Directory containing the actual extracted image files.
        metadata_path: JSON mapping image IDs to records containing a raw path.
        source_root: Optional prefix written into the path column. This does not
            change which files are scanned or move any files.

    Returns:
        A DataFrame with the six columns in COLUMNS, in notebook scan order.

    Raises:
        FileNotFoundError: A required directory or metadata file is absent.
        ValueError: The metadata format, image inventory, patient mapping, or
            within-patient label consistency fails validation.

    Notes:
        File existence is checked, but image decoding and diagnostic label
        correctness are not checked. The known duplicate remains here because
        the original exploration notebook saved it before preprocessing.
    """
    # Resolve the physical root independently from the CSV path prefix.
    image_root = Path(image_root).resolve()
    stored_root = str(source_root) if source_root is not None else image_root.as_posix()
    # Avoid a doubled slash when appending original split/class/file components.
    stored_root = stored_root.rstrip("/")
    with Path(metadata_path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be an image-ID-to-record JSON object.")

    # Example: ADNI_037_S_4001_... maps to patient ID 037_S_4001.
    # Unusable metadata entries are skipped; referenced unmapped images fail below.
    image_to_patient = {}
    for image_id, info in metadata.items():
        raw = info.get("raw", "") if isinstance(info, dict) else ""
        match = re.search(r"\d{3}_S_\d{4}", raw) if isinstance(raw, str) else None
        if match:
            image_to_patient[str(image_id)] = match.group()

    # Match notebook order exactly: train before test, AD before NC, then
    # lexicographically sorted filenames within each directory (not numeric sort).
    records = []
    for split in ("train", "test"):
        for label in ("AD", "NC"):
            folder = image_root / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing image directory: {folder}")
            files = sorted(p for p in folder.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if not files:
                raise ValueError(f"No supported images found in {folder}")
            for path in files:
                if not path.is_file():
                    raise ValueError(f"Expected an image file: {path}")
                # Example: 218391_78.jpeg -> image ID "218391". Keep IDs as
                # strings so matching does not depend on numeric dtype inference.
                image_id = path.stem.split("_")[0]
                # split is the original folder assignment at this stage. The
                # splitter later replaces it with the new train/val/test assignment.
                records.append({
                    "split": split, "label": label, "filename": path.name,
                    "image_id": image_id,
                    "path": f"{stored_root}/{split}/{label}/{path.name}",
                    "patient_id": image_to_patient.get(image_id),
                })

    # Reject missing mappings instead of silently dropping images and changing
    # the experimental population. Reject ambiguous patient labels as well.
    frame = pd.DataFrame(records, columns=COLUMNS)
    missing = frame["patient_id"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} images have no mapped patient ID; "
                         f"examples: {frame.loc[missing, 'filename'].head().tolist()}")
    if frame.groupby("patient_id")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting folder labels found within patients.")
    if not frame["path"].is_unique:
        raise ValueError("Duplicate paths in inventory.")
    return frame


def compare_csv(frame, reference):
    """Require equal columns and cell values in the same row order.

    This comparison does not sort rows or strip path prefixes. Supply
    --source-root to match the original CSV paths when scanning elsewhere.
    """
    # Compare logical CSV values rather than bytes or line-ending conventions.
    # String loading preserves identifiers; disabling NA inference prevents
    # literal text from being interpreted as missing values.
    expected = pd.read_csv(reference, dtype=str, keep_default_na=False)
    # The in-memory pandas index is not written to CSV, so discard it here.
    # Row order and column order are still checked by assert_frame_equal.
    actual = frame.astype(str).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    except AssertionError as error:
        raise ValueError(f"CSV does not match {reference}:\n{error}") from error


def prepare_data(args):
    """Run inventory creation, splitting, optional verification, and CSV export.

    Args:
        args: argparse namespace produced by main(), including required input
            paths, output_dir, seed, and optional source/reference paths.

    Returns:
        None. Writes CSVs and a JSON audit, and prints a summary.

    Existing reference files are read only. A failed reference comparison raises
    ValueError before output files are created. Input data is never modified.
    """
    # Import only when preparing data so --help works without PyTorch installed.
    # Reuse the existing implementation: do not duplicate splitting logic here.
    from dataset import create_patient_splits, DUPLICATE_FILENAME

    # Protect the existing experiment: refuse any non-empty destination.
    # No overwrite flag is provided; use a fresh directory for another run.
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new or empty --output-dir; existing CSVs are protected.")
    records = build_inventory(args.image_root, args.metadata, args.source_root)
    # dataset.py excludes DUPLICATE_FILENAME, sorts patients, and applies two
    # stratified splits (70/30, then 50/50 of the holdout) using the same seed.
    # All slices of a patient remain together; source image paths are preserved.
    # Its validation checks labels, unique paths, separation and image counts.
    frames = create_patient_splits(records, seed=args.seed)

    # Comparisons run before any outputs are written. A mismatch is a hard error.
    if args.reference_inventory:
        compare_csv(records, args.reference_inventory)
    if args.reference_split_dir:
        for name, frame in zip(SPLITS, frames):
            compare_csv(frame, Path(args.reference_split_dir) / f"{name}.csv")

    # Original-folder overlap is an audit statistic, not overlap in the new splits.
    original_patients = [set(records.loc[records['split'] == s, 'patient_id'])
                         for s in ('train', 'test')]
    # True means a requested reference check passed. None is serialized as null
    # and means no reference was supplied, not that equivalence was verified.
    report = {
        "seed": args.seed,
        "inventory_images": len(records),
        "inventory_patients": int(records["patient_id"].nunique()),
        "original_patient_overlap": len(original_patients[0] & original_patients[1]),
        "excluded_duplicate_records": int((records["filename"] == DUPLICATE_FILENAME).sum()),
        "reference_inventory_matched": True if args.reference_inventory else None,
        "reference_splits_matched": True if args.reference_split_dir else None,
        "splits": {},
    }
    # Report slice counts separately from unique patient counts.
    for name, frame in zip(SPLITS, frames):
        report["splits"][name] = {
            "images": len(frame), "patients": int(frame["patient_id"].nunique()),
            "images_by_class": {str(k): int(v) for k, v in frame['label'].value_counts().items()},
        }

    # Only write after validation and requested comparisons succeed. Filesystem
    # errors during writing may leave partial outputs; use a new directory on retry.
    (output / "processed").mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir()
    # Save the unfiltered inventory, as notebook 01 did; cleaned rows are in splits.
    records.to_csv(output / "processed" / "image_records.csv", index=False)
    for name, frame in zip(SPLITS, frames):
        frame.to_csv(output / "splits" / f"{name}.csv", index=False)
    (output / "preparation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Echo the audit so a Colab run leaves a visible verification record.
    print(json.dumps(report, indent=2))
    print(f"Saved preparation outputs to: {output.resolve()}")


def main():
    """Parse command-line options and invoke the preparation workflow.

    Use --help for argument descriptions. Exceptions intentionally propagate so
    unsuccessful preparation exits with a nonzero status and a useful traceback.
    """
    # Required arguments identify input images, metadata, and a safe destination.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, help="Extracted AD_NC directory or equivalent cache root.")
    parser.add_argument("--metadata", required=True, help="Path to meta_data_with_label.json.")
    parser.add_argument("--output-dir", required=True, help="New/empty directory for processed/ and splits/.")
    # Optional compatibility settings reproduce the original path prefix/seed.
    parser.add_argument("--source-root", help="Original root to record in CSV paths; defaults to image-root.")
    parser.add_argument("--seed", type=int, default=42)
    # Optional references enable strict regression checks against notebook outputs.
    parser.add_argument("--reference-inventory", help="Optional existing image_records.csv for exact comparison.")
    parser.add_argument("--reference-split-dir", help="Optional existing final splits directory for exact comparison.")
    args = parser.parse_args()
    prepare_data(args)


# Importing this module exposes helpers without starting preparation.
if __name__ == "__main__":
    main()
