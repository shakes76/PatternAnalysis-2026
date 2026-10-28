"""
Data loading, preprocessing, and leakage-free patient-level splitting for
the HipMRI 2D Prostate Radiotherapy dataset.

Implements:
1. Nifti 2D slice loading with Z-score intensity normalization and fixed 6-class
   one-hot mask encoding (adapted from COMP3710 Specification Appendix B).
2. Strict patient-level train/val/test splitting to prevent cross-slice data leakage.
3. Random spatial data augmentations (horizontal flip, affine shift/scale) applied
   identically to paired images and segmentation masks.
"""

import os
import re
import glob
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ============================================================================
# Part 1: Nifti Loading & One-Hot Encoding (Adapted from Spec Appendix B)
# ============================================================================

def to_channels(arr: np.ndarray, num_classes: int = 6, dtype=np.float32) -> np.ndarray:
    """
    Converts a 2D integer segmentation mask (H, W) into a multi-channel
    one-hot encoded array (num_classes, H, W).
    Note: Fixed to num_classes=6 (Background, Body, Bone, Bladder, Rectum, Prostate)
    even if a specific slice does not contain all organs (e.g., superior/inferior slices).
    """
    h, w = arr.shape
    res = np.zeros((num_classes, h, w), dtype=dtype)
    arr_int = np.rint(arr).astype(np.int64)
    for c in range(num_classes):
        res[c, arr_int == c] = 1.0
    return res


def load_nifti_slice_2d(filepath: str, norm_image: bool = False) -> np.ndarray:
    """
    Loads a single 2D Nifti file using Nibabel (following Appendix B Listing 1).
    Removes redundant 3rd dimension if present and optionally applies Z-score
    intensity standardization.
    """
    nifti_img = nib.load(filepath)
    arr = nifti_img.get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]  # Remove extra singleton dimension in HipMRI_study data
    arr = arr.astype(np.float32)

    if norm_image:
        mean = arr.mean()
        std = arr.std()
        arr = (arr - mean) / (std + 1e-8)
    return arr


# ============================================================================
# Part 2: Patient-Level Leakage-Free Splitting & File Discovery
# ============================================================================

def extract_patient_id(filename: str) -> str:
    """
    Extracts the unique patient/case identifier from a HipMRI slice filename
    (e.g., 'case_012_slice_045.nii.gz' -> 'case_012') to enforce strict
    patient-level separation across train, validation, and test splits.
    """
    base = os.path.basename(filename)
    # Match patterns like case_XX, patient_XX, patXX, or prefix before '_slice'
    match = re.search(r"([a-zA-Z]+_?\d+)", base)
    if match:
        return match.group(1)
    if "_slice" in base:
        return base.split("_slice")[0]
    # Fallback: group by the first two underscore-separated tokens
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def discover_hipmri_splits(data_dir: str, seed: int = 42):
    """
    Discovers paired image and label Nifti files in `keras_slices_data`.
    Supports both pre-split directory structures (keras_slices_train/validate/test)
    and flat image/label directories with automatic patient-level splitting.
    """
    # Structure A: Pre-split folders on Rangpur (/home/groups/comp3710/HipMRI_Study_open/keras_slices_data)
    train_img_dir = os.path.join(data_dir, "keras_slices_train")
    val_img_dir = os.path.join(data_dir, "keras_slices_validate")
    test_img_dir = os.path.join(data_dir, "keras_slices_test")

    train_lbl_dir = os.path.join(data_dir, "keras_slices_seg_train")
    val_lbl_dir = os.path.join(data_dir, "keras_slices_seg_validate")
    test_lbl_dir = os.path.join(data_dir, "keras_slices_seg_test")

    def pair_folder(img_folder: str, lbl_folder: str):
        imgs = sorted(glob.glob(os.path.join(img_folder, "*.nii*")))
        lbls = sorted(glob.glob(os.path.join(lbl_folder, "*.nii*")))
        if len(imgs) != len(lbls) or len(imgs) == 0:
            raise RuntimeError(
                f"Mismatch or empty Nifti files between {img_folder} ({len(imgs)}) "
                f"and {lbl_folder} ({len(lbls)})."
            )
        return list(zip(imgs, lbls))

    if os.path.isdir(train_img_dir) and os.path.isdir(train_lbl_dir):
        train_pairs = pair_folder(train_img_dir, train_lbl_dir)
        val_pairs = pair_folder(val_img_dir, val_lbl_dir)
        test_pairs = pair_folder(test_img_dir, test_lbl_dir)
        return train_pairs, val_pairs, test_pairs

    # Structure B: Alternative folder naming (e.g., images/ and labels/ or semantic_MRs/)
    candidates_img = ["images", "MRs", "semantic_MRs", "data"]
    candidates_lbl = ["labels", "segs", "semantic_labels_only", "masks"]
    img_root, lbl_root = None, None

    for c_img, c_lbl in zip(candidates_img, candidates_lbl):
        p_img = os.path.join(data_dir, c_img)
        p_lbl = os.path.join(data_dir, c_lbl)
        if os.path.isdir(p_img) and os.path.isdir(p_lbl):
            img_root, lbl_root = p_img, p_lbl
            break

    if img_root is None:
        raise FileNotFoundError(
            f"Could not locate HipMRI sub-folders inside '{data_dir}'. "
            f"Expected 'keras_slices_train'/'keras_slices_seg_train' or 'images'/'labels'."
        )

    all_pairs = pair_folder(img_root, lbl_root)

    # Group strictly by Patient ID to prevent slice-level data leakage
    patient_to_pairs = {}
    for img_path, lbl_path in all_pairs:
        pid = extract_patient_id(img_path)
        patient_to_pairs.setdefault(pid, []).append((img_path, lbl_path))

    patients = sorted(patient_to_pairs.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = max(1, int(0.70 * n_total))
    n_val = max(1, int(0.15 * n_total))

    train_pids = set(patients[:n_train])
    val_pids = set(patients[n_train:n_train + n_val])
    test_pids = set(patients[n_train + n_val:])

    # Assert zero patient overlap across splits
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)

    train_pairs = [p for pid in train_pids for p in patient_to_pairs[pid]]
    val_pairs = [p for pid in val_pids for p in patient_to_pairs[pid]]
    test_pairs = [p for pid in test_pids for p in patient_to_pairs[pid]]

    return train_pairs, val_pairs, test_pairs


# ============================================================================
# Part 3: PyTorch Dataset & Synchronized Spatial Augmentations
# ============================================================================

class HipMRISliceDataset(Dataset):
    """
    PyTorch Dataset for 2D HipMRI Prostate Segmentation.
    Returns:
        image_tensor: FloatTensor of shape [1, H, W] (Z-score normalized)
        mask_onehot:  FloatTensor of shape [6, H, W] (One-hot encoded)
    """
    def __init__(
        self,
        pairs: list[tuple[str, str]] = None,
        num_classes: int = 6,
        target_size: tuple[int, int] = (256, 256),
        augment: bool = False,
        dummy_mode: bool = False,
        dummy_length: int = 32
    ):
        super().__init__()
        self.pairs = pairs or []
        self.num_classes = num_classes
        self.target_size = target_size
        self.augment = augment
        self.dummy_mode = dummy_mode
        self.dummy_length = dummy_length

    def __len__(self) -> int:
        if self.dummy_mode:
            return self.dummy_length
        return len(self.pairs)

    def _apply_joint_augmentations(
        self,
        image: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies identical spatial transformations to both the input MRI slice
        and its multi-channel one-hot segmentation mask.
        """
        # Random horizontal flip (pelvic anatomy is bilaterally symmetric)
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random slight rotation (-10 to +10 deg) and translation (up to 5%)
        if random.random() < 0.5:
            angle = random.uniform(-10.0, 10.0)
            max_dx = int(0.05 * image.shape[2])
            max_dy = int(0.05 * image.shape[1])
            translate = (random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy))
            scale = random.uniform(0.95, 1.05)

            image = TF.affine(
                image, angle=angle, translate=translate, scale=scale, shear=0.0,
                interpolation=TF.InterpolationMode.BILINEAR
            )
            mask = TF.affine(
                mask, angle=angle, translate=translate, scale=scale, shear=0.0,
                interpolation=TF.InterpolationMode.NEAREST
            )
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.dummy_mode:
            # Synthetic pelvic-like slice & 6-class one-hot mask for local smoke testing
            h, w = self.target_size
            img_tensor = torch.randn(1, h, w, dtype=torch.float32)
            raw_labels = np.zeros((h, w), dtype=np.int64)
            raw_labels[40:210, 40:210] = 1   # Body outline
            raw_labels[80:160, 70:110] = 2   # Bone
            raw_labels[70:115, 115:155] = 3  # Bladder
            raw_labels[145:185, 115:150] = 4 # Rectum
            raw_labels[115:145, 120:145] = 5 # Prostate (<2% area)
            mask_onehot = torch.from_numpy(to_channels(raw_labels, num_classes=self.num_classes))
            return img_tensor, mask_onehot

        img_path, lbl_path = self.pairs[idx]

        # 1. Load 2D Nifti slice & Z-score normalize intensities
        img_np = load_nifti_slice_2d(img_path, norm_image=True)
        lbl_np = load_nifti_slice_2d(lbl_path, norm_image=False)

        # 2. Convert to tensors: Image [1, H, W], Mask [6, H, W]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        mask_onehot = torch.from_numpy(to_channels(lbl_np, num_classes=self.num_classes))

        # 3. Ensure consistent spatial dimensions (resize if needed)
        if self.target_size is not None and tuple(img_tensor.shape[1:]) != self.target_size:
            img_tensor = TF.resize(
                img_tensor, list(self.target_size),
                interpolation=TF.InterpolationMode.BILINEAR, antialias=True
            )
            mask_onehot = TF.resize(
                mask_onehot, list(self.target_size),
                interpolation=TF.InterpolationMode.NEAREST
            )

        # 4. Apply paired training augmentations
        if self.augment:
            img_tensor, mask_onehot = self._apply_joint_augmentations(img_tensor, mask_onehot)

        return img_tensor, mask_onehot


def get_hipmri_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    target_size: tuple[int, int] = (256, 256),
    num_workers: int = 2,
    dummy_mode: bool = False
):
    """
    Builds and returns (train_loader, val_loader, test_loader) for HipMRI 2D.
    Falls back to synthetic dummy_mode only if explicitly requested for local testing.
    """
    if dummy_mode:
        print("[Dataset] Running in synthetic DUMMY mode for pipeline verification.")
        train_ds = HipMRISliceDataset(target_size=target_size, augment=True, dummy_mode=True, dummy_length=32)
        val_ds = HipMRISliceDataset(target_size=target_size, augment=False, dummy_mode=True, dummy_length=16)
        test_ds = HipMRISliceDataset(target_size=target_size, augment=False, dummy_mode=True, dummy_length=16)
    else:
        train_pairs, val_pairs, test_pairs = discover_hipmri_splits(data_dir)
        print(f"[Dataset] Loaded HipMRI splits -> Train: {len(train_pairs)} | "
              f"Val: {len(val_pairs)} | Test: {len(test_pairs)} slices")

        train_ds = HipMRISliceDataset(train_pairs, target_size=target_size, augment=True)
        val_ds = HipMRISliceDataset(val_pairs, target_size=target_size, augment=False)
        test_ds = HipMRISliceDataset(test_pairs, target_size=target_size, augment=False)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Smoke test for dataset.py (automatically tests real path if available, else dummy mode)
    default_rangpur_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    use_dummy = not os.path.exists(default_rangpur_path)

    tr_loader, va_loader, te_loader = get_hipmri_dataloaders(
        data_dir=default_rangpur_path,
        batch_size=4,
        num_workers=0,
        dummy_mode=use_dummy
    )
    sample_imgs, sample_masks = next(iter(tr_loader))
    print(f"[Smoke Test] Batch Image Tensor : {tuple(sample_imgs.shape)} | dtype: {sample_imgs.dtype}")
    print(f"[Smoke Test] Batch Mask One-Hot : {tuple(sample_masks.shape)} | unique values: {torch.unique(sample_masks).tolist()}")