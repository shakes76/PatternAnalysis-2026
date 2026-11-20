"""
dataset.py - Data loading for HipMRI 2D slice segmentation.

Loads 2D NIfTI (.nii.gz) slices and their segmentation masks, centre-crops them
to a fixed size, normalises the images, optionally applies augmentation, and
exposes them through PyTorch Datasets / DataLoaders.

Expected directory layout under `root`:
    keras_slices_train/        keras_slices_seg_train/
    keras_slices_validate/     keras_slices_seg_validate/
    keras_slices_test/         keras_slices_seg_test/

Naming convention:
    image: case_040_week_0_slice_0.nii.gz
    mask:  seg_040_week_0_slice_0.nii.gz

Mask classes: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate.

Course:  COMP3710 - 2.2 Segmentation (HipMRI)
Author:  <your name / student number>
"""

import glob
import math
import os
import random

import nibabel as nib
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# Every slice is centre-cropped to this (rows, cols) shape
TARGET_SHAPE = (256, 128)


# --------------------------------------------------------------------------- #
# Pre-processing helpers
# --------------------------------------------------------------------------- #
def center_crop_to_shape(arr, target_shape=TARGET_SHAPE):
    """Centre-crop a 2D array down to `target_shape`.

    Args:
        arr: 2D numpy array of shape (rows, cols).
        target_shape: Desired (rows, cols).

    Returns:
        The cropped 2D array.

    Raises:
        ValueError: If `arr` is smaller than `target_shape` in either dimension.
    """
    crop_r = arr.shape[0] - target_shape[0]
    crop_c = arr.shape[1] - target_shape[1]
    if crop_r < 0 or crop_c < 0:
        raise ValueError(f"Slice {arr.shape} is smaller than target {target_shape}")

    # Trim equal amounts from each side (any odd pixel is dropped from the end)
    r0 = crop_r // 2
    c0 = crop_c // 2
    return arr[r0:r0 + target_shape[0], c0:c0 + target_shape[1]]


def load_nifti_2d(path, norm_image=False, dtype=np.float32):
    """Load a single 2D slice from a .nii.gz file and centre-crop it.

    Follows the same handling as the reference `load_data_2D` function.

    Args:
        path: Path to the .nii.gz file.
        norm_image: If True, normalise to zero mean / unit variance.
            Leave False for masks, which are label maps.
        dtype: Numpy dtype of the returned array.

    Returns:
        2D numpy array of shape TARGET_SHAPE.
    """
    img = nib.load(path).get_fdata(caching="unchanged")

    if img.ndim == 3:
        img = img[:, :, 0]  # HipMRI slices carry an extra singleton dimension

    img = center_crop_to_shape(img.astype(dtype))

    if norm_image:
        img = (img - img.mean()) / (img.std() + 1e-8)  # epsilon avoids divide-by-zero

    return img


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def augment(image, mask):
    """Randomly augment an image / mask pair (each transform applied with p=0.5).

    Geometric transforms (flip, rotation) are applied identically to the image
    and mask so they stay aligned. The intensity transform (gamma) only touches
    the image.

    Args:
        image: Float tensor of shape (1, H, W).
        mask: Long tensor of shape (H, W) holding class indices.

    Returns:
        The augmented (image, mask) tuple.
    """
    # Horizontal flip
    if random.random() < 0.5:
        image = torch.flip(image, dims=[-1])
        mask = torch.flip(mask, dims=[-1])

    # Small rotation of up to +/-10 degrees. The image is padded with its
    # minimum value; the mask uses nearest-neighbour so labels stay integers.
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR,
                          fill=[float(image.min())])
        mask = TF.rotate(mask.unsqueeze(0), angle,
                         interpolation=InterpolationMode.NEAREST,
                         fill=[0]).squeeze(0)

    # Random gamma (brightness curve) adjustment
    if random.random() < 0.5:
        # Log-uniform sampling so brightening and darkening are equally likely
        gamma = math.exp(random.uniform(math.log(0.7), math.log(1.5)))
        lo, hi = image.min(), image.max()
        if hi > lo:  # skip constant (e.g. empty) slices
            img01 = (image - lo) / (hi - lo)          # rescale to [0, 1]
            image = img01.pow(gamma) * (hi - lo) + lo  # gamma, then rescale back

    return image, mask


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class HipMRIDataset(Dataset):
    """2D HipMRI slice dataset (the whole split is pre-loaded into memory).

    Args:
        root: Directory containing the keras_slices_* folders.
        split: One of "train", "validate" or "test".
        num_classes: Number of segmentation classes (used to validate masks).
        norm_image: Normalise each image to zero mean / unit variance.
        early_stop: Use only the first 21 slices (quick testing).
        augment: Apply random augmentation when fetching items (train only).

    Each item is a tuple:
        image: float tensor (1, H, W), zero mean / unit variance
        mask:  long tensor (H, W) of class indices
    """

    def __init__(self, root, split="train", num_classes=6, norm_image=True,
                 early_stop=False, augment=False):
        self.num_classes = num_classes
        self.augment = augment

        image_dir = os.path.join(root, f"keras_slices_{split}")
        mask_dir = os.path.join(root, f"keras_slices_seg_{split}")

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No .nii.gz files found in {image_dir}")

        # Match each image to its mask by swapping the "case" prefix for "seg"
        mask_paths = []
        for p in image_paths:
            name = os.path.basename(p)
            mask_name = "seg" + name[len("case"):] if name.startswith("case") else name
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask for {name}: {mask_path}")
            mask_paths.append(mask_path)

        if early_stop:  # quick testing with a small subset
            image_paths, mask_paths = image_paths[:21], mask_paths[:21]

        self.names = [os.path.basename(p) for p in image_paths]

        # Pre-allocate arrays using the shape of the first (cropped) slice
        first = load_nifti_2d(image_paths[0])
        rows, cols = first.shape
        n = len(image_paths)

        self.images = np.zeros((n, rows, cols), dtype=np.float32)
        self.masks = np.zeros((n, rows, cols), dtype=np.int64)

        # Pre-load everything into memory (slices are small), like load_data_2D
        for i in tqdm(range(n), desc=f"Loading {split}"):
            self.images[i] = load_nifti_2d(image_paths[i], norm_image=norm_image)
            # Masks are label maps: no normalisation, round to integer classes
            self.masks[i] = np.rint(load_nifti_2d(mask_paths[i])).astype(np.int64)

        # Sanity check: every label must be a valid class index
        assert self.masks.min() >= 0 and self.masks.max() < num_classes

    def __len__(self):
        """Number of slices in the split."""
        return len(self.images)

    def __getitem__(self, idx):
        """Return the (image, mask) pair at `idx`, augmented if enabled."""
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(self.masks[idx])                 # (H, W)

        if self.augment:
            image, mask = augment(image, mask)

        return image, mask


def get_dataloaders(root, batch_size=16, num_classes=6, num_workers=0):
    """Build train / validation / test DataLoaders.

    Only the training set is shuffled and augmented.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_set = HipMRIDataset(root, "train", num_classes, augment=True)
    val_set = HipMRIDataset(root, "validate", num_classes)
    test_set = HipMRIDataset(root, "test", num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick check: python dataset.py [/path/to/keras_slices_data]
    import sys

    default_root = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    root = sys.argv[1] if len(sys.argv) > 1 else default_root

    ds = HipMRIDataset(root, "train", early_stop=True)
    img, msk = ds[0]
    print(len(ds), img.shape, img.dtype, msk.shape, msk.dtype, msk.unique())