"""
Load and preprocess HipMRI data for U-Net segmentation.

Provides NIfTI image loading, per-slice Z-score normalisation,
six-class one-hot mask encoding, centre-padding, training-only
intensity augmentation and train/validation/test DataLoaders.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

NUM_CLASSES = 6
# Target Width to ensure all slices are same size
TARGET_WIDTH = 144
SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}


def to_channels(arr: np.ndarray, dtype=np.uint8, num_classes=NUM_CLASSES) -> np.ndarray:
    """Convert a class-index mask to one-hot channels.

    Adapted from Appendix B: use a FIXED class count because individual
    HipMRI slices do not always contain all six labels.

    Args:
        arr: 2D integer-valued segmentation mask.
        dtype: Output array data type.
        num_classes: Total number of labels in the dataset.

    Returns:
        Array with shape (height, width, num_classes).
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c] = (arr == c).astype(dtype)
    return res


def load_data_2D(imageNames, normImage=False, categorical=False,
                 dtype=np.float32, getAffines=False, early_stop=False):
    """Load NIfTI slices into a NumPy array.

    Args:
        imageNames: List of NIfTI file paths (same-size images).
        normImage: Apply per-slice Z-score normalisation if True.
        categorical: Return six-channel one-hot masks if True.
        dtype: Data type for the returned NumPy array.
        getAffines: Include affine matrices in the return value if True.
        early_stop: Load only the first 22 files for a quick test.

    Returns:
        Stacked NumPy images, optionally with a list of affine matrices.

    Note:
        The Appendix B pre-allocation assumes all input files have the
        same dimensions. The class HipMRIDataset calls this with ONE file at a time,
        then pads the paired image and mask to a common width.
    """
    affines = []
    if early_stop:
        imageNames = imageNames[:20]

    num = len(imageNames)
    first_case = nib.load(str(imageNames[0])).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Appendix B: possible singleton dimension
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames, disable=(num == 1))):
        niftiImage = nib.load(str(inName))
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        affines.append(affine)

    if getAffines:
        return images, affines
    else:
        return images


class HipMRIDataset(Dataset):
    """Load matched MRI/mask pairs from an existing split."""

    def __init__(self, root, split="train", augment=False):
        """Identify files for the requested HipMRI split. Defaults to training split.

        Args:
            root: Folder containing all six keras_slices_* directories.
            split: "train", "validate", or "test".
            augment: Use training-only random intensity augmentation.

        Returns:
            None.
        """
        image_folder, mask_folder = SPLIT_DIRS[split]
        self.image_dir = Path(root) / image_folder
        self.mask_dir = Path(root) / mask_folder
        self.files = sorted(self.image_dir.glob("case_*.nii.gz"))
        self.augment = augment and split == "train"

    def __len__(self):
        """Return the number of MRI slices in the selected split.

        Returns:
            int: Number of images.
        """
        return len(self.files)

    def __getitem__(self, index):
        """Load and preprocess one MRI and its matching segmentation mask.

        Args:
            index: Index of the requested slice.

        Returns:
            dict: image [1, 256, 144], mask [256, 144],
            mask_onehot [6, 256, 144], filename, original_width.
        """
        image_path = self.files[index]
        mask_path = self.mask_dir / image_path.name.replace("case_", "seg_", 1)

        # Appendix B loader for both MRI and mask.
        image = load_data_2D([image_path], normImage=True)[0]
        mask = load_data_2D([mask_path])[0].astype(np.int64)
        if image.shape != mask.shape:
            raise ValueError(f"MRI/mask dimensions differ: {image_path.name}")

        # Standardise 128- and 144-pixel-wide slices without resizing anatomy.
        original_width = image.shape[1]
        extra = TARGET_WIDTH - original_width
        padding = ((0, 0), (extra // 2, extra - extra // 2))
        image = np.pad(image, padding)
        mask = np.pad(mask, padding)  # Zero is the visually identified background.

        # Appendix B one-hot conversion, now with six fixed channels.
        onehot = to_channels(mask, dtype=np.float32)

        # Augment MRI intensity
        if self.augment:
            image = image * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)

        # Convert NumPy arrays into PyTorch tensors
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        onehot_tensor = torch.tensor(onehot, dtype=torch.float32).permute(2, 0, 1)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "mask_onehot": onehot_tensor,
            "filename": image_path.name,
            "original_width": original_width,
        }

def make_dataloaders(root, batch_size=4, num_workers=0):
    """Create PyTorch loaders without changing the supplied patient splits.

    Args:
        root: Root folder for HipMRI keras_slices_data.
        batch_size: Number of slices per batch.
        num_workers: Number of parallel data-loading processes.

    Returns:
        dict: "train", "validate", and "test" DataLoaders.
    """
    loaders = {}
    for split in SPLIT_DIRS:
        dataset = HipMRIDataset(root, split, augment=(split == "train"))
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=(split == "train"), num_workers=num_workers
        )
    return loaders


