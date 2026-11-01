import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import re
import nibabel as nib
import numpy as np

# Extract the patient ID while checking the expected filename structure.
IMAGE_PATTERN = re.compile(
    r"case_(\d+)_week_\d+_slice_\d+\.nii\.gz"
)
SPLITS = ("train", "validate", "test")

def find_pairs(data_dir, split):
    """Return matching image and mask paths for one supplied split."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    data_dir = Path(data_dir)
    image_dir = data_dir / f"keras_slices_{split}"
    mask_dir = data_dir / f"keras_slices_seg_{split}"

    # Catch incorrect paths before attempting to load any images.
    for folder in (image_dir, mask_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")

    images = sorted(image_dir.glob("*.nii.gz"))
    if not images:
        raise ValueError(f"No NIfTI images found in {image_dir}")

    pairs = []
    expected_masks = set()

    for image_path in images:
        if IMAGE_PATTERN.fullmatch(image_path.name) is None:
            raise ValueError(f"Unexpected image filename: {image_path.name}")

        # Pair by filename rather than relying on directory ordering.
        mask_name = image_path.name.replace("case_", "seg_", 1)
        mask_path = mask_dir / mask_name

        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        pairs.append((image_path, mask_path))
        expected_masks.add(mask_name)

    # Also detect masks that have no corresponding image.
    actual_masks = {p.name for p in mask_dir.glob("*.nii.gz")}
    extra_masks = actual_masks - expected_masks
    if extra_masks:
        raise ValueError(
            f"{split}: {len(extra_masks)} masks have no matching image"
        )

    return pairs

# Sanity test - make sure all of the image pairs actually work
def load_pair(image_path, mask_path):
    """Load and validate one 2D MRI–mask pair."""
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)

    # Remove singleton dimensions, keeping the stored orientation.
    image = np.squeeze(image_nii.get_fdata(dtype=np.float32))
    mask = np.squeeze(mask_nii.get_fdata())

    if image.ndim != 2 or mask.ndim != 2:
        raise ValueError("Expected 2D image and mask")

    if image.shape != mask.shape:
        raise ValueError("Image and mask dimensions differ")

    # Check spatial metadata and numerical validity before preprocessing.
    if not np.allclose(image_nii.affine, mask_nii.affine):
        raise ValueError("Image and mask affines differ")

    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("Image or mask contains NaN or infinite values")

    if not np.allclose(mask, np.round(mask)) or mask.min() < 0:
        raise ValueError("Mask must contain non-negative integer labels")

    # Standardise MRI intensities; leave segmentation labels unchanged.
    image = (image - image.mean()) / (image.std() + 1e-8)
    mask = np.round(mask).astype(np.int64)

    return image, mask

def patient_ids(pairs):
    """Read patient IDs from the verified image filenames."""
    return {
        IMAGE_PATTERN.fullmatch(image.name).group(1)
        for image, _ in pairs
    }


def audit_splits(data_dir):
    """Check all supplied splits and reject overlapping patient IDs."""
    splits = {split: find_pairs(data_dir, split) for split in SPLITS}
    patients = {
        split: patient_ids(pairs) for split, pairs in splits.items()
    }

    for first, second in (
        ("train", "validate"),
        ("train", "test"),
        ("validate", "test"),
    ):
        overlap = patients[first] & patients[second]
        if overlap:
            raise ValueError(
                f"Patient overlap between {first} and {second}: "
                f"{sorted(overlap)}"
            )

    print(f"{'Split':<12} {'Pairs':>8} {'Patients':>10}")
    for split in SPLITS:
        print(
            f"{split:<12} {len(splits[split]):>8} "
            f"{len(patients[split]):>10}"
        )

    print("All image–mask pairs matched; no patient overlap.")
    return splits


class HipMRIDataset(Dataset):
    """Load paired slices and pad them to a common size."""

    def __init__(self, pairs, target_size=(256, 144)):
        self.pairs = list(pairs)
        self.target_size = target_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image, mask = load_pair(image_path, mask_path)

        image = torch.from_numpy(
            np.ascontiguousarray(image)
        ).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        height, width = mask.shape
        target_h, target_w = self.target_size

        # Stop rather than silently crop a larger image.
        if height > target_h or width > target_w:
            raise ValueError(
                f"{image_path.name}: shape {(height, width)} exceeds "
                f"target {self.target_size}"
            )

        # Apply identical centred padding to the image and mask.
        pad_h, pad_w = target_h - height, target_w - width
        padding = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
        )

        # Zero is the mean intensity after normalisation.
        image = F.pad(image, padding, value=0)

        # Mark artificial pixels so the loss ignores them.
        mask = F.pad(mask, padding, value=-100)

        return image, mask

def get_datasets(data_dir):

    splits = audit_splits(data_dir)

    train_dataset = HipMRIDataset(splits["train"])
    val_dataset = HipMRIDataset(splits["validate"])
    test_dataset = HipMRIDataset(splits["test"])

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(data_dir, batch_size, device):

    # Define datasets TODO
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        # Allocate data batches into RAM or GPU memory depending on the device.
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    return train_loader, val_loader, test_loader
    
