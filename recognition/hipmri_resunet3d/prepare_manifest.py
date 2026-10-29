"""Pair Nifti images and masks by a shared case key and verified patient ID."""

import argparse
import csv
import re
from pathlib import Path


def nifti_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else name[:-4]


def collect(directory: Path, suffix: str = "") -> dict[str, Path]:
    """Index Nifti files after removing a known suffix from each filename stem."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or not path.name.endswith((".nii", ".nii.gz")):
            continue
        stem = nifti_stem(path)
        if suffix and not stem.endswith(suffix):
            raise ValueError(f"Expected suffix {suffix!r} in {path.name}")
        case_key = stem[:-len(suffix)] if suffix else stem
        if not case_key:
            raise ValueError(f"Empty case key after removing suffix from {path.name}")
        if case_key in result:
            raise ValueError(
                f"Repeated case key {case_key}: {result[case_key]}, {path}"
            )
        result[case_key] = path.resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument(
        "--image-suffix", default="",
        help="Known image stem suffix to remove before pairing",
    )
    parser.add_argument(
        "--mask-suffix", default="",
        help="Known mask stem suffix to remove before pairing",
    )
    parser.add_argument(
        "--patient-regex", required=True,
        help="Regex with one capture group that extracts patient ID from case stem",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    images = collect(args.image_dir, args.image_suffix)
    masks = collect(args.mask_dir, args.mask_suffix)
    if not images or images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())[:10]
        missing_images = sorted(masks.keys() - images.keys())[:10]
        raise ValueError(
            f"Unpaired cases; missing masks {missing_masks}, missing images {missing_images}"
        )
    regex = re.compile(args.patient_regex)
    rows = []
    for stem in sorted(images):
        match = regex.search(stem)
        if match is None or len(match.groups()) != 1:
            raise ValueError(f"Patient regex must capture exactly one ID from {stem}")
        rows.append({
            "case_id": stem,
            "patient_id": match.group(1),
            "image": str(images[stem]),
            "mask": str(masks[stem]),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "patient_id", "image", "mask"])
        writer.writeheader()
        writer.writerows(rows)
    patients = {row["patient_id"] for row in rows}
    print(f"Wrote {len(rows)} cases from {len(patients)} patients to {args.out}")


if __name__ == "__main__":
    main()
