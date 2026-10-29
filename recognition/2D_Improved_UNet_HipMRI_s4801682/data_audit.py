import os
import re
from pathlib import Path
from collections import Counter

# Rangpur HipMRI data set location
DATA_DIR = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")

# Expected naming conventions
IMAGE_PATTERN = re.compile(
    r"^case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

MASK_PATTERN = re.compile(
    r"^seg_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

def run_data_audit(data_dir):
    print("=" * 60)
    print("-------- HipMRI dataset Audit Report --------")
    print("=" * 60)

    if not os.path.exists(data_dir):
        print(f"[Error] Path does not exist: {data_dir}")
        return

    # 1. Total File Count & Extensions
    # Find all files
    all_files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file()
    )

    print(f"\nDataset path: {data_dir}")

    if not all_files:
        print("[WARNING] no files found.")
        return

    # Count file extensions correctly
    extensions = Counter()

    for path in all_files:
        name = path.name.lower()

        if name.endswith(".nii.gz"):
            extensions[".nii.gz"] += 1
        else: 
            extensions [path.suffix.lower() or "No Extension"] += 1

    print(f"\n1. TOTAL FILES FOUND: {len(all_files)}")
    print(" File Extensions Breakdown:")
    for ext, count in sorted(extensions.items()):
        print(f"    {ext}: {count} files")

    # Count files in each immediate subdirectory
    print("\n2. FILE COUNTS BY FOLDER")
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir():
            count = sum(1 for p in folder.iterdir() if p.is_file())
            print(f"    {folder.name}: {count} files")

    # 2. Folder Organisation (Image vs Masks)
    img_files = []
    mask_files = []

    for path in all_files:
        if not path.name.lower().endswith(".nii.gz"):
            continue

        if "seg" in path.parent.name.lower():
            mask_files.append(path)
        else:
            img_files.append(path)

    print("\n3. IMAGE AND MASK COUNTS:")
    print(f"    Images count: {len(img_files)}") 
    print(f"    Masks count: {len(mask_files)}")

    # 3. Patient / Volume Identifiers
    case_ids = set() 
    unmatched_image_names = []
    unmatched_mask_names = []

    for path in img_files:
        match = IMAGE_PATTERN.match(path.name)

        if match:
            case_ids.add(match.group(1))
        else:
            unmatched_image_names.append(path.name)

    for path in mask_files:
        if not MASK_PATTERN.match(path.name):
            unmatched_mask_names.append(path.name)

    print("\n4. CASE IDENTIFIERS:") 
    print(f"    Unique case IDs: {len(case_ids)}")
    print(f"    Sample case IDs: {sorted(case_ids)[:10]}")

    if unmatched_image_names:
        print(
            "  Image filenames not matching the expected pattern: "
            f"{len(unmatched_image_names)}"
        )
        print(f"  Examples: {unmatched_image_names[:5]}")

    if unmatched_mask_names:
        print(
            "  Mask filenames not matching the expected pattern: "
            f"{len(unmatched_mask_names)}"
        )
        print(f"  Examples: {unmatched_mask_names[:5]}")

    # Compare filename identifiers
    img_keys = set()
    mask_keys = set()

    for path in img_files:
        match = IMAGE_PATTERN.match(path.name)
        if match:
            img_keys.add(match.groups())

    for path in mask_files:
        match = MASK_PATTERN.match(path.name)
        if match:
            mask_keys.add(match.groups())

    matched = img_keys & mask_keys

    print("\n5. IMAGE-MASK FILENAME MATCHING")
    print(f"    Unique image identifiers: {len(img_keys)}")
    print(f"    Unique mask identifiers:  {len(mask_keys)}")
    print(f"    Identifiers found in both: {len(matched)}")
    print(f"  Images without matching mask identifiers: "
          f"{len(img_keys - mask_keys)}")
    print(f"  Masks without matching image identifiers: "
          f"{len(mask_keys - img_keys)}")

    print(
        "  Note: Matching identifiers do not prove that the "
        "image and mask are spatially aligned."
    )

    # Display examples
    print("\n6. EXAMPLE FILES")
    for path in img_files[:3]:
        print(f"  Image: {path.relative_to(data_dir)}")

    for path in mask_files[:3]:
        print(f"  Mask:  {path.relative_to(data_dir)}")

    print("=" * 60)

if __name__ == "__main__":
    run_data_audit(DATA_DIR)