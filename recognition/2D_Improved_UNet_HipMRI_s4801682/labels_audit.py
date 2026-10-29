from collections import Counter 
from pathlib import Path 
import nibabel as nib 
import numpy as np 
from tqdm import tqdm

DATA_DIR = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")


MASK_FOLDERS = {
    "train": "keras_slices_seg_train",
    "validation": "keras_slices_seg_validate",
    "test": "keras_slices_seg_test"
}

def audit_labels(data_dir: Path):
    """Count segmentation label values across all dataset splits."""
    all_label_counts = {}
    total_masks = 0

    for split_name, folder_name in MASK_FOLDERS.items():
        mask_dir = data_dir / folder_name

        print("=" * 60) 
        print("AUDITING SEGMENTATION MASK LABELS")
        print(f"SCANNING {split_name.upper()} MASKS") 
        print("=" * 60)

        if not mask_dir.is_dir():
            print(f"ERROR: Folder not found: {mask_dir}")
            continue

        mask_paths = sorted(mask_dir.glob("*.nii.gz"))

        if not mask_paths:
            print("WARNING: No .nii.gz mask files found.")
            continue

        split_label_counts = {}
        split_masks = 0
        invalid_value_files = []

        for mask_path in mask_paths:
            # Load one mask at a time
            mask_nii = nib.load(str(mask_path))
            mask = np.asanyarray(mask_nii.dataobj)

            # Find each unique value and its pixel count
            labels, counts = np.unique(
                mask,
                return_counts=True
            )

            split_masks += 1
            total_masks += 1

            for label, count in zip(labels, counts):
                label_value = float(label)

                # Accumulate counts for this split
                split_label_counts[label_value] = (
                    split_label_counts.get(label_value, 0)
                    + int(count)
                )

                # Also accumulate counts across the dataset
                all_label_counts[label_value] = (
                    all_label_counts.get(label_value, 0)
                    + int(count)
                )

            # Flag labels that are not finite integers
            if (
                not np.all(np.isfinite(labels))
                or not np.all(labels == np.floor(labels))
            ):
                invalid_value_files.append(mask_path.name)

        print(f"Masks scanned: {split_masks}")
        print("Label pixel counts:")

        for label in sorted(split_label_counts):
            print(
                f"  Label {label:g}: "
                f"{split_label_counts[label]:,} pixels"
            )

        if invalid_value_files:
            print(
                "WARNING: Non-integer or non-finite values found "
                f"in {len(invalid_value_files)} mask(s)."
            )
            for filename in invalid_value_files[:10]:
                print(f"  {filename}")

    print(f"\n{'=' * 50}")
    print("OVERALL LABEL AUDIT")
    print(f"{'=' * 50}")
    print(f"Total masks scanned: {total_masks}")
    print("All label values found:")

    for label in sorted(all_label_counts):
        print(
            f"  Label {label:g}: "
            f"{all_label_counts[label]:,} pixels"
        )

    print("\nAudit complete.")

if __name__ == "__main__":
    audit_labels(DATA_DIR)