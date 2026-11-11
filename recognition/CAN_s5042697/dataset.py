"""
Data loading and preprocessing for the HipMRI 2D prostate slices.

Expected layout (copied from Rangpur /home/groups/comp3710/HipMRI_Study_open):

    data/keras_slices_data/
        keras_slices_train/       case_004_week_0_slice_0.nii.gz ...
        keras_slices_seg_train/   seg_004_week_0_slice_0.nii.gz  ...
        keras_slices_validate/    keras_slices_seg_validate/
        keras_slices_test/        keras_slices_seg_test/

The provided train/validate/test folders are already split by patient
(31 / 4 / 3 patients). We keep that split and assert that no patient id
appears in more than one split, so slices from one patient never leak
across splits.

Label values in the masks (checked visually against overlays):
    0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate
"""

import glob
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
BLADDER = 3
PROSTATE = 5
IMAGE_SHAPE = (256, 128)
SPLITS = ("train", "validate", "test")

_PATIENT_RE = re.compile(r"^(?:case|seg)_(\d+)_")


def patient_id(path):
    """Return the patient id string ('004') from an image or mask filename."""
    match = _PATIENT_RE.match(os.path.basename(path))
    if match is None:
        raise ValueError(f"Unexpected filename: {path}")
    return match.group(1)


def list_pairs(root, split, limit=None):
    """
    List (image_path, mask_path) pairs for one split.

    Masks share the image filename with the 'case_' prefix swapped for 'seg_'.
    `limit` keeps only the first N pairs, for quick smoke tests.
    """
    image_paths = sorted(glob.glob(os.path.join(root, f"keras_slices_{split}", "*.nii*")))
    if not image_paths:
        raise FileNotFoundError(f"No images found for split '{split}' under {root}")

    pairs = []
    for image_path in image_paths:
        name = os.path.basename(image_path).replace("case_", "seg_", 1)
        mask_path = os.path.join(root, f"keras_slices_seg_{split}", name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask for {image_path}")
        pairs.append((image_path, mask_path))

    return pairs[:limit] if limit else pairs


def check_no_patient_overlap(split_pairs):
    """Raise if any patient id appears in more than one split."""
    seen = {}
    for split, pairs in split_pairs.items():
        for patient in {patient_id(image_path) for image_path, _ in pairs}:
            if patient in seen and seen[patient] != split:
                raise RuntimeError(
                    f"Patient {patient} appears in both '{seen[patient]}' and '{split}'"
                )
            seen[patient] = split


def _fit_shape(arr, shape=IMAGE_SHAPE):
    """
    Centre-crop or zero-pad a 2D slice to `shape`.

    Most slices are 256x128, but a few (case_019_week_1) are 256x144.
    Cropping keeps the pixel spacing unchanged, unlike resizing.
    """
    if arr.ndim == 3:  # some HipMRI slices carry an extra trailing dim
        arr = arr[:, :, 0]

    out = np.zeros(shape, dtype=arr.dtype)
    src, dst = [], []
    for size, target in zip(arr.shape, shape):
        if size >= target:
            start = (size - target) // 2
            src.append(slice(start, start + target))
            dst.append(slice(0, target))
        else:
            start = (target - size) // 2
            src.append(slice(0, size))
            dst.append(slice(start, start + size))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def load_slice(path, is_mask=False):
    """Load one Nifti slice as a fixed-size numpy array."""
    arr = np.asarray(nib.load(path).dataobj)
    arr = _fit_shape(arr)
    return arr.astype(np.int64) if is_mask else arr.astype(np.float32)


def normalise(image, eps=1e-8):
    """Per-slice z-score normalisation (zero mean, unit variance)."""
    return (image - image.mean()) / (image.std() + eps)


def one_hot(mask, num_classes=NUM_CLASSES):
    """Convert an integer mask batch (B, H, W) to one-hot channels (B, C, H, W)."""
    return F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()


def train_transform():
    """
    Light augmentation applied jointly to image and mask.

    Arrays are (256, 128) with the 256 axis running left-right across the
    pelvis, so RandomVerticalFlip (which flips dim -2) is a left-right mirror.
    A flip along the other axis would turn the coronal slice upside down
    (bladder below the prostate), which is not anatomically plausible. Masks are resampled with nearest neighbour
    automatically because they are wrapped as tv_tensors.Mask.
    """
    return v2.Compose([
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ])


class HipMRISlices(Dataset):
    """
    2D HipMRI slices with their segmentation masks.

    Returns (image, mask) where image is a float tensor (1, H, W) and mask is
    a long tensor (H, W) of class indices in [0, NUM_CLASSES).
    """

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = torch.from_numpy(load_slice(image_path)).unsqueeze(0)
        mask = torch.from_numpy(load_slice(mask_path, is_mask=True))

        # Augment on raw intensities so padded regions fill with 0, matching
        # the scanner background, then normalise.
        if self.transform is not None:
            image, mask = self.transform(tv_tensors.Image(image), tv_tensors.Mask(mask))
            image, mask = image.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)

        return normalise(image), mask.long()


def get_datasets(root, augment=True, limit=None):
    """Build train/validate/test datasets after checking for patient leakage."""
    split_pairs = {split: list_pairs(root, split, limit) for split in SPLITS}
    check_no_patient_overlap(split_pairs)

    return {
        split: HipMRISlices(pairs, train_transform() if (augment and split == "train") else None)
        for split, pairs in split_pairs.items()
    }


def get_dataloaders(root, batch_size=16, num_workers=2, augment=True, limit=None):
    """Build train/validate/test DataLoaders. Only the train loader shuffles."""
    datasets = get_datasets(root, augment=augment, limit=limit)
    return {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        for split, ds in datasets.items()
    }


if __name__ == "__main__":
    # Quick sanity check: split sizes, patient counts and one batch.
    root = os.path.join(os.path.dirname(__file__), "data", "keras_slices_data")
    datasets = get_datasets(root)
    for split, ds in datasets.items():
        patients = sorted({patient_id(p) for p, _ in ds.pairs})
        print(f"{split:9s} {len(ds):6d} slices  {len(patients):2d} patients")

    images, masks = next(iter(DataLoader(datasets["train"], batch_size=8, shuffle=True)))
    print("image batch", tuple(images.shape), images.dtype,
          f"mean {images.mean():.3f} std {images.std():.3f}")
    print("mask batch ", tuple(masks.shape), masks.dtype, "labels", masks.unique().tolist())
    print("one-hot    ", tuple(one_hot(masks).shape))
