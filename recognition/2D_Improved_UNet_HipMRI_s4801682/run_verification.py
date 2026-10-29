from pathlib import Path 
import re 
import time
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
import nibabel as nib
from tqdm import tqdm

"""
check enviroment: confirm PyTorch, NiBabel and NumPy are installed and check whether CUDA is avaliable
Inspect the data set: count the NIfTI files and sow the folder organisation
Check image-mask pairs: match filenames using case, week and slice IDs and report missing or duplicate matches. 
inspect sample data: load a few image-mask pairs, check their dimensions, intensity values and mask labels.
check the existing splits: check is the current folder organisation seperates case IDs across training, validation and testing. 
save a visual example: create an image showing an MRI slice with its mask overlay, so you can manually inspect it. 
"""

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")

# Regex pattern matching case_XXX_week_Y_slice_Z.nii.gz and seg_XXX_week_y_slice_z.nii.gz
IMAGE_PATTERN = re.compile(
    r"^case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

MASK_PATTERN = re.compile(
    r"^seg_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$",
    re.IGNORECASE
)

# APPENDIX B LISTING 1: course standard NIfTI loading functions
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Converts a label array into multi-channel 
       categorical (one-hot) encoding.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels: 
        c = int(c) 
        res[..., c : c + 1][arr == c] = 1 
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """Load medical image data from names, cases list provided into a list for each. 
       This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.
       nornImage: bool (normalise the image 0.0-1.0)
       early_stop: Stop loading pre-maturely, leaves arrays mostly empty, 
            for quick loading and testin scripts.      
    """
    affines = []

    # Get fixed size
    num = len(imageNames)

    # Read first case to get fixed 2D dimensions
    first_case = nib.load(str(imageNames[0])).get_fdata(caching='unchanged')

    # Remove extra 3D dimensions if present
    if len(first_case.shape) == 3: 
        first_case = first_case[:, :, 0]

    # Multi-channel categorical mask encoding if requested
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, desc="Loadin NIfTI Slices")):
        niftiImage = nib.load(str(inName))
        inImage = niftiImage.get_fdata(caching="unchanged") # caching='unchanged' avoids memory leaks
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0] #sometimes extra dims in HipMRI_study data

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


#1. Verify Image and Mask Alignment:
def verify_alignment(data_dir: Path):
    """ Load a simaple image and its mask.
        Compare their dimensions.
        Display the MRI, segmentation mask, and overlay for visual inspection.
    """
    print("\n1. VERIFYING IMAGE & MASK ALIGNMENT")

    # Load sample image slices from training folder
    img_dir = data_dir / "keras_slices_train"
    mask_dir = data_dir / "keras_slices_seg_train"

    # Select one training slice and matching mask 
    sample_img_paths = sorted(list(img_dir.glob("case_*.nii.gz")))[:1]
    
    if not sample_img_paths:
        raise FileNotFoundError(f"No NIfTI images found in {img_dir}")

    # Select 1 image for inspection
    sample_img_path = sample_img_paths[0]

    # Validate the image filename
    image_match = IMAGE_PATTERN.match(sample_img_path.name)

    if image_match is None:
        raise ValueError( 
            f"Unexpected image filename: {sample_img_path.name}"
        )

    # Match the mask using case, week and slice identifiers
    case_id, week_id, slice_id = image_match.groups()

    sample_mask_name = (f"seg_{case_id}_week_{week_id}_slice_{slice_id}.nii.gz")
    sample_mask_path = mask_dir / sample_mask_name

    # Checking is the matching file exists:
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    if not sample_mask_path.is_file():
        raise FileNotFoundError(
            f"Matching mask not found: {sample_mask_path}"
        )
    
    # Load image and mask using the course standard loader
    raw_img = load_data_2D([sample_img_path], normImage=False)
    norm_img = load_data_2D([sample_img_path], normImage=True)
    mask_data = load_data_2D([sample_mask_path], categorical=False)

    # Extract the 2D slices from the batch dimension
    raw_img = raw_img[0]
    norm_img = norm_img[0]
    mask_data = mask_data[0]

    print(f"    Image Path: {sample_img_path.relative_to(data_dir)}")
    print(f"    * Raw Array Shape: {raw_img.shape} | Intensity Range: [{raw_img.min():.2f}, {raw_img.max():.2f}]")
    print(f"    Mask Path: {sample_mask_path.relative_to(data_dir)}")
    print(f"    * Mask Array Shape: {mask_data.shape} | Unique Label Values: {np.unique(mask_data)}")

    # Check spatial dimensions
    if raw_img.shape != mask_data.shape:
        raise ValueError(
            f"Image and mask dimensions do not match:"
            f"{raw_img.shape} vs {mask_data.shape}"
        )

    print("Image and mask dimensions match.")

    # Generate 3-panel visualisation: MRI slice, Mask, and Overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(norm_img, cmap="gray")
    axes[0].set_title("Input Pelvic MRI Slice (Normalized)")

    axes[1].imshow(mask_data, cmap="gray")
    axes[1].set_title("Segmentation Mask")

    axes[2].imshow(norm_img, cmap="gray")
    axes[2].imshow(np.ma.masked_where(mask_data == 0, mask_data), cmap="jet", alpha=0.5)
    axes[2].set_title("Overlay Spatial Alignment Check")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("alignment_check.png")
    plt.close(fig)
    print("     Saved visual alignment overlay plot to 'alignment_check.png'")

def verify_multiple_pairs(data_dir: Path, num_samples=5):
    """Visually inspect several MRI and mask pairs."""

    print(f"\\nINSPECTING {num_samples} IMAGE-MASK PAIRS...")

    img_dir = data_dir / "keras_slices_train"
    mask_dir = data_dir / "keras_slices_seg_train"

    # Pick evenly spaced slices across the dataset 
    all_imgs = sorted(list(img_dir.glob("case_*.nii.gz")))
    step = max(1, len(all_imgs) // num_samples) 
    selected_imgs = [all_imgs[i] for i in range(0, len(all_imgs), step)][ :num_samples ]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for row, img_path in enumerate(selected_imgs):
        match = IMAGE_PATTERN.match(img_path.name)
        case_id, week_id, slice_id = match.groups()
        mask_path = mask_dir / f"seg_{case_id}_week_{week_id}_slice_{slice_id}.nii.gz"

        img_data = load_data_2D([img_path], normImage=True)[0]
        mask_data = load_data_2D([mask_path], categorical=False)[0]

        # Column 0: MRI Slice 
        axes[row, 0].imshow(img_data, cmap="gray") 
        axes[row, 0].set_title(f"MRI: {img_path.name}", fontsize=10)

        # Column 1: Segmentation Mask 
        axes[row, 1].imshow(mask_data, cmap="gray") 
        axes[row, 1].set_title(f"Mask (Labels: {np.unique(mask_data)})", fontsize=10)

        # Column 2: Color Overlay 
        axes[row, 2].imshow(img_data, cmap="gray") 
        axes[row, 2].imshow( np.ma.masked_where(mask_data == 0, mask_data), cmap="jet", alpha=0.5) 
        axes[row, 2].set_title("Overlay Alignment Check", fontsize=10)

        for col in range(3): 
            axes[row, col].axis("off")

    plt.tight_layout() 
    plt.savefig("sample_gallery_check.png", dpi=150) 
    plt.close(fig) 
    print(f"Saved {num_samples} sample slice overlays to 'sample_gallery_check.png'" )

# Check every image has a matching mask in the same split folder. 
def verify_image_mask_pairs(data_dir: Path):
    """ check every image has a matching mask i the same split.
        Find images with missing masks.
        Find masks with no matching image.
        Detect duplicate image or mask identifiers.
        Check whether image/mask identifiers occur 
        in the wrong split or across multiple splits.
    """
    split_folders = {
        "train": (
            data_dir / "keras_slices_train", 
            data_dir / "keras_slices_seg_train" 
        ),
        "validation": (
            data_dir / "keras_slices_validate",
            data_dir / "keras_slices_seg_validate"
        ),
        "test": (
            data_dir / "keras_slices_test",
            data_dir / "keras_slices_seg_test"
        )
    }

    # Track where each slice identifier occurs across splits.
    image_loc = {}
    mask_loc = {}
    errors_found = False

    for split, (img_dir, mask_dir) in split_folders.items():
        print(f"\n{split.upper()}")

        if not img_dir.is_dir() or not mask_dir.is_dir():
            print("ERROR: Image or mask folder is missing.")
            errors_found = True
            continue

        images = {}
        masks = {}

        # Read image filenames and detect duplicate identifiers.
        for path in img_dir.glob("case_*.nii.gz"):
            match = IMAGE_PATTERN.fullmatch(path.name)

            if match:
                key = match.groups()

                if key in images:
                    print(f"Duplicate image identifier: {key}")
                    errors_found = True
                else:
                    images[key] = path

                image_loc.setdefault(key, []).append(split)

        # Read mask filenames and detect duplicate identifiers.
        for path in mask_dir.glob("seg_*.nii.gz"):
            match = MASK_PATTERN.fullmatch(path.name)

            if match:
                key = match.groups()

                if key in masks:
                    print(f"Duplicate mask identifier: {key}")
                    errors_found = True
                else:
                    masks[key] = path

                mask_loc.setdefault(key, []).append(split)

        image_keys = set(images)
        mask_keys = set(masks)

        missing_masks = image_keys - mask_keys
        unmatched_masks = mask_keys - image_keys

        print(f"Images found: {len(images)}")
        print(f"Masks found: {len(masks)}")
        print(f"Images without masks: {len(missing_masks)}")
        print(f"Masks without images: {len(unmatched_masks)}")

        for key in sorted(missing_masks)[:10]:
            print(f"  Missing mask for image: {images[key].name}")

        for key in sorted(unmatched_masks)[:10]:
            print(f"  Missing image for mask: {masks[key].name}")

        if missing_masks or unmatched_masks:
            errors_found = True

    # Check whether the same slice identifier occurs in
    # multiple image splits or multiple mask splits.
    for label, locations in [
        ("image", image_loc),
        ("mask", mask_loc)
    ]:
        for key, splits in locations.items():
            if len(splits) > 1:
                print(
                    f"Duplicate {label} identifier across splits: "
                    f"{key}, found in {splits}"
                )
                errors_found = True

     # Check whether an image and its matching mask are
    # assigned to the same split.
    for key in set(image_loc) & set(mask_loc):
        image_splits = set(image_loc[key])
        mask_splits = set(mask_loc[key])

        if image_splits != mask_splits:
            print(
                f"Image-mask split mismatch for {key}: "
                f"images in {sorted(image_splits)}, "
                f"masks in {sorted(mask_splits)}"
            )
            errors_found = True

    print("\nIMAGE-MASK PAIRING RESULT")

    if errors_found:
        print("WARNING: Issues were found. Review the messages above.")
    else:
        print("PASSED: No pairing or duplicate-identifier issues found.")

# Confirm Whether existing splits are seperated by case ID
def confirm_patient_split(data_dir: Path):
    """ Scan the 3 image folders.
        Extract unique case IDs.
        Report the number of cases in each split.
        Check for case IDs appearing in multiple splits.
        Report whether any overlap was found.
    """
    print("\n2. EXISTING SPLIT AND CASE-ID AUDIT")

    split_folders = {
        "train": data_dir / "keras_slices_train", 
        "validation": data_dir / "keras_slices_validate",
        "test": data_dir / "keras_slices_test",
    }

    case_ids_by_split = {}
    slice_counts = {}

    for split, folder in split_folders.items():
        if not folder.is_dir():
            print(f"\nWARNING: Folder not found: {folder}")
            case_ids_by_split[split] = set()
            slice_counts[split] = 0
            continue

        case_ids = set()
        slice_count = 0

        for path in folder.glob("case_*.nii.gz"):
            match = IMAGE_PATTERN.fullmatch(path.name)

            if match:
                case_id = match.group(1)
                case_ids.add(case_id)
                slice_count += 1

        case_ids_by_split[split] = case_ids
        slice_counts[split] = slice_count

        print(f"\n{split.upper()}")
        print(f"    Image slices: {slice_count}")
        print(f"    Unique case IDs: {len(case_ids)}")

    # Count unique cases across all existing splits
    all_case_ids = set().union(*case_ids_by_split.values())
    total_cases = len(all_case_ids)

    print("\nEXISTING SPLIT DISTIBUTION")
    print(f"Total unique case IDs: {total_cases}")

    if total_cases > 0:
        for split, case_ids in case_ids_by_split.items():
            percentage = 100 * len(case_ids) / total_cases
            print(
                f"  {split}: {len(case_ids)} cases"
                f"({percentage:.1f}%)"
            )

    # Check for case IDs shared between splits.
    print("\nCROSS-SPLIT CASE-ID OVERLAP")
    overlap_found = False

    split_names = list(split_folders.keys())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            split_a = split_names[i]
            split_b = split_names[j]

            overlap = (
                case_ids_by_split[split_a]
                & case_ids_by_split[split_b]
            )

            print(
                f"{split_a} vs {split_b}: "
                f"{len(overlap)} overlapping case IDs"
            )

            if overlap:
                overlap_found = True
                print(f"    IDs: {sorted(overlap)}")

    if total_cases == 0:
        print("\nWARNING: No valid case IDs found. Check the folders and filenames.")
    elif any(not case_ids_by_split[name] for name in split_names):
        print("\nWARNING: At least one split is empty or missing.")
    elif overlap_found:
        print("\nWARNING: Investigate overlapping case IDs before training.")
    else:
        print("\nNo case-ID overlap found between the existing splits.")

    print(
        "\nIMPORTANT: This checks case-ID separation, not patient "
        "identity. Confirm that each case ID represents one patient."
    )

if __name__ == "__main__":
    verify_alignment(DATA_DIR)
    verify_image_mask_pairs(DATA_DIR)
    confirm_patient_split(DATA_DIR)
    verify_multiple_pairs(DATA_DIR, num_samples=5)