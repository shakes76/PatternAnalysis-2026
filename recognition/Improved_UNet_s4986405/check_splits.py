"""Check patient overlap between HipMRI dataset splits."""

import argparse
from itertools import combinations
from pathlib import Path
import re


PATTERN = re.compile(r"case_(\d+)_week_\d+_slice_\d+\.nii\.gz")


def collect_patients(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    files = sorted(directory.glob("case_*.nii.gz"))
    if not files:
        raise ValueError(f"No case_*.nii.gz images found: {directory}")

    patients = set()
    for path in files:
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected image filename: {path}")
        patients.add(int(match.group(1)))

    return patients, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()

    patients_by_split = {}
    for name, directory in [
        ("train", args.train_dir),
        ("validation", args.val_dir),
        ("test", args.test_dir),
    ]:
        try:
            patients, count = collect_patients(directory)
        except ValueError as error:
            parser.exit(1, f"ERROR: {error}\n")
        patients_by_split[name] = patients
        print(f"{name}: {count} slices, {len(patients)} patients")
        print(f"  Patient IDs: {sorted(patients)}")

    has_overlap = False
    for left, right in combinations(patients_by_split, 2):
        overlap = patients_by_split[left] & patients_by_split[right]
        if overlap:
            has_overlap = True
            print(f"FAIL: {left}/{right} overlap: {sorted(overlap)}")
        else:
            print(f"PASS: {left}/{right} have no shared patients")

    if has_overlap:
        parser.exit(1, "Patient split check: FAIL\n")
    print("Patient split check: PASS")


if __name__ == "__main__":
    main()
