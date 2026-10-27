"""Data loading and preprocessing helpers for the guided HipMRI rebuild."""

from pathlib import Path
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_MULTIPLE = 16
# Absolute tolerance in the affine's declared coordinate units, not assumed mm.
# No relative tolerance: a large origin must not hide a translation mismatch.
AFFINE_ATOL = 1e-5
AFFINE_RTOL = 0.0


def minmax_normalize(values):
    if not torch.isfinite(values).all():
        raise ValueError("MRI intensities must be finite before normalization.")
    minimum = values.min()
    maximum = values.max()
    if maximum == minimum:
        return torch.zeros_like(values)
    if not torch.isfinite(maximum - minimum):
        raise ValueError("MRI normalization range must be finite.")
    return (values - minimum) / (maximum - minimum)


def preprocess_pair(image, target, multiple=DEFAULT_MULTIPLE):
    """Normalize, then pad bottom/right; every original pixel remains valid."""
    if not isinstance(multiple, int) or isinstance(multiple, bool) or multiple <= 0:
        raise ValueError("Padding multiple must be a positive integer.")
    if image.ndim != 3 or target.ndim != 2 or image.shape[0] != 1:
        raise ValueError("Expected a single-channel [1,H,W] image and [H,W] target.")
    if image.shape[-2:] != target.shape or min(target.shape) == 0:
        raise ValueError("Image and target must have matching nonempty spatial shapes.")
    if not image.is_floating_point() or target.dtype != torch.long:
        raise TypeError("Expected floating-point image and int64 class-ID target.")
    if image.device != target.device or (target < 0).any():
        raise ValueError("Image/target must share a device; original class IDs must be nonnegative.")
    normalized = minmax_normalize(image.to(torch.float32))
    height, width = target.shape
    padding = (0, (-width) % multiple, 0, (-height) % multiple)
    valid = torch.ones_like(target, dtype=torch.bool)
    return {
        "image": F.pad(normalized, padding, value=0),
        "target": F.pad(target, padding, value=IGNORE_INDEX),
        "valid_mask": F.pad(valid, padding, value=False),
        "original_size": (height, width),
    }


def collate_padded(samples):
    """Stack mixed-size samples, preserving each sample's existing valid region."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")
    height = max(sample["target"].shape[0] for sample in samples)
    width = max(sample["target"].shape[1] for sample in samples)
    images, targets, valid_masks = [], [], []
    for sample in samples:
        image, target, valid = sample["image"], sample["target"], sample["valid_mask"]
        if image.shape[-2:] != target.shape or valid.shape != target.shape:
            raise ValueError("Sample image, target and valid_mask shapes must match.")
        if target.dtype != torch.long or valid.dtype != torch.bool:
            raise TypeError("Expected int64 target and boolean valid_mask.")
        if (target[valid] < 0).any() or (target[~valid] != IGNORE_INDEX).any():
            raise ValueError("Only invalid padded pixels may contain the ignore index.")
        padding = (0, width - target.shape[1], 0, height - target.shape[0])
        images.append(F.pad(image, padding, value=0))
        targets.append(F.pad(target, padding, value=IGNORE_INDEX))
        valid_masks.append(F.pad(valid, padding, value=False))
    return {
        "image": torch.stack(images),
        "target": torch.stack(targets),
        "valid_mask": torch.stack(valid_masks),
        "original_sizes": [sample["original_size"] for sample in samples],
    }


def crop_to_original(values, original_size):
    """Restore a single sample's original spatial extent without interpolation."""
    height, width = original_size
    if height <= 0 or width <= 0 or height > values.shape[-2] or width > values.shape[-1]:
        raise ValueError("Original size must fit within the padded tensor.")
    return values[..., :height, :width]


def valid_pixels(prediction, target, valid_mask):
    """Select class IDs only; padding cannot enter Dice/IoU pixel counts."""
    if prediction.shape != target.shape or valid_mask.shape != target.shape:
        raise ValueError("Prediction, target and valid_mask must have the same shape.")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean.")
    if (target[valid_mask] < 0).any():
        raise ValueError("Valid pixels must contain nonnegative class IDs.")
    return prediction[valid_mask], target[valid_mask]


def validate_pair_geometry(image_nifti, mask_nifti):
    """Check selected NIfTI affines and header shapes before reading voxel arrays.

    Matching matrices are a consistency check, not independent evidence of
    anatomical alignment, valid source geometry, or physical units.
    Return the maximum absolute elementwise affine difference for auditing.
    """
    if (len(image_nifti.shape) != 2 or image_nifti.shape != mask_nifti.shape
            or min(image_nifti.shape) <= 0):
        raise ValueError("Expected matching 2D MRI and mask shapes.")
    affines = []
    for name, nifti in (("MRI", image_nifti), ("Mask", mask_nifti)):
        affine = np.asarray(nifti.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError(f"{name} affine must be a finite 4x4 matrix.")
        affines.append(affine)
    difference = float(np.max(np.abs(affines[0] - affines[1])))
    if not np.allclose(affines[0], affines[1], atol=AFFINE_ATOL, rtol=AFFINE_RTOL):
        raise ValueError(
            f"MRI/mask affine mismatch: max absolute difference={difference:.9g}; "
            f"atol={AFFINE_ATOL}, rtol={AFFINE_RTOL}. Verify source geometry."
        )
    return difference


def load_pair(image_path, mask_path, multiple=DEFAULT_MULTIPLE):
    """Check paired header geometry, then read arrays and preprocess the sample."""
    image_nifti = nib.load(image_path)
    mask_nifti = nib.load(mask_path)
    validate_pair_geometry(image_nifti, mask_nifti)
    image = image_nifti.get_fdata()
    mask = mask_nifti.get_fdata()
    if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("Mask values must be finite integers before casting.")
    if (mask < 0).any():
        raise ValueError("Original mask class IDs must be nonnegative.")
    image_tensor = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask).to(torch.long)
    return preprocess_pair(image_tensor, mask_tensor, multiple=multiple)


def collect_pairs(image_dir, mask_dir):
    """Match existing case_/seg_ filenames without reading NIfTI contents."""
    image_paths = sorted(
        path for path in Path(image_dir).glob("case_*.nii.gz") if path.is_file()
    )
    if not image_paths:
        raise ValueError(f"No MRI files found in {image_dir}")
    pairs = []
    for image_path in image_paths:
        mask_name = image_path.name.replace("case_", "seg_", 1)
        mask_path = Path(mask_dir) / mask_name
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        pairs.append((image_path, mask_path))
    mask_count = sum(
        path.is_file() for path in Path(mask_dir).glob("seg_*.nii.gz")
    )
    if mask_count != len(pairs):
        raise ValueError("Extra mask files without a matching MRI; check filenames.")
    return pairs


def patient_ids(pairs):
    """Use the numeric case field as a patient-group key; read filenames only."""
    ids = set()
    for image_path, _ in pairs:
        match = re.fullmatch(
            r"case_(\d+)_week_\d+_slice_\d+\.nii\.gz", image_path.name
        )
        if match is None:
            raise ValueError(f"Unexpected MRI filename: {image_path.name}")
        ids.add(int(match.group(1)))
    return ids


class HipMRIPairDataset(Dataset):
    """Load single samples from a caller-verified list of MRI/mask path pairs."""

    def __init__(self, pairs, multiple=DEFAULT_MULTIPLE):
        self.pairs = list(pairs)
        self.multiple = multiple

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        return load_pair(image_path, mask_path, multiple=self.multiple)
