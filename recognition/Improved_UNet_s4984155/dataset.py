"""
Data loader and preprocessing for HipMRI 2D prostate segmentation.
Handles NIfTI slice loading, intensity standardization, and patient-level splitting.
"""

import os
import glob
from typing import Tuple, List, Optional
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


class HipMRI2DSliceDataset(Dataset):
    """
    Dataset for loading 2D HipMRI NIfTI slices and extracting prostate masks (Label 5).
    """
    PROSTATE_LABEL = 5

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        is_train: bool = False,
        target_size: Optional[Tuple[int, int]] = (256, 128),
        prostate_only: bool = True,
        use_cache: bool = True
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.target_size = target_size
        self.prostate_only = prostate_only
        self.use_cache = use_cache
        self.cache = {}

        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
        if not self.image_files:
            raise FileNotFoundError(f"No NIfTI files found in {image_dir}")

        self.mask_files = []
        for img_path in self.image_files:
            fname = os.path.basename(img_path)
            mask_fname = fname.replace("case_", "seg_")
            mask_path = os.path.join(mask_dir, mask_fname)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask file: {mask_path}")
            self.mask_files.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_nifti(self, path: str) -> np.ndarray:
        nii = nib.load(path)
        data = nii.get_fdata(caching="unchanged")
        # Squeeze trailing singleton dimension present in some 2D HipMRI slices
        if len(data.shape) == 3:
            data = data[:, :, 0]
        return data.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cache and idx in self.cache:
            image_tensor, mask_tensor = self.cache[idx]
        else:
            img_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            image_np = self._load_nifti(img_path)
            mask_np = self._load_nifti(mask_path)

            if self.prostate_only:
                mask_np = (mask_np == self.PROSTATE_LABEL).astype(np.float32)

            # Per-slice z-score normalization
            mean = image_np.mean()
            std = image_np.std() + 1e-8
            image_np = (image_np - mean) / std

            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

            if self.target_size is not None and image_tensor.shape[1:] != self.target_size:
                image_tensor = TF.resize(image_tensor, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.resize(mask_tensor, self.target_size, interpolation=TF.InterpolationMode.NEAREST)

            if self.use_cache:
                self.cache[idx] = (image_tensor, mask_tensor)

        if self.is_train:
            image_tensor = image_tensor.clone()
            mask_tensor = mask_tensor.clone()

            if random.random() > 0.5:
                image_tensor = TF.hflip(image_tensor)
                mask_tensor = TF.hflip(mask_tensor)

            if random.random() > 0.5:
                image_tensor = TF.vflip(image_tensor)
                mask_tensor = TF.vflip(mask_tensor)

            if random.random() > 0.5:
                angle = random.uniform(-10.0, 10.0)
                image_tensor = TF.rotate(image_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_tensor = TF.rotate(mask_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image_tensor, mask_tensor


def get_data_loaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (256, 128),
    prostate_only: bool = True,
    use_cache: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_img_dir = os.path.join(data_root, "keras_slices_train")
    train_mask_dir = os.path.join(data_root, "keras_slices_seg_train")

    val_img_dir = os.path.join(data_root, "keras_slices_validate")
    val_mask_dir = os.path.join(data_root, "keras_slices_seg_validate")

    test_img_dir = os.path.join(data_root, "keras_slices_test")
    test_mask_dir = os.path.join(data_root, "keras_slices_seg_test")

    def extract_patients(folder: str) -> set:
        if not os.path.exists(folder):
            return set()
        files = glob.glob(os.path.join(folder, "*.nii*"))
        return set(os.path.basename(f).split("_week")[0] for f in files)

    train_pts = extract_patients(train_img_dir)
    val_pts = extract_patients(val_img_dir)
    test_pts = extract_patients(test_img_dir)

    # Check for patient overlap across splits
    if train_pts and val_pts:
        assert not (train_pts & val_pts), "Leakage error: patient overlap between train and val"
    if train_pts and test_pts:
        assert not (train_pts & test_pts), "Leakage error: patient overlap between train and test"
    if val_pts and test_pts:
        assert not (val_pts & test_pts), "Leakage error: patient overlap between val and test"

    print(f"Patient split audit: {len(train_pts)} train, {len(val_pts)} val, {len(test_pts)} test patients.")

    train_dataset = HipMRI2DSliceDataset(train_img_dir, train_mask_dir, is_train=True, target_size=target_size, prostate_only=prostate_only, use_cache=use_cache)
    val_dataset = HipMRI2DSliceDataset(val_img_dir, val_mask_dir, is_train=False, target_size=target_size, prostate_only=prostate_only, use_cache=use_cache)
    test_dataset = HipMRI2DSliceDataset(test_img_dir, test_mask_dir, is_train=False, target_size=target_size, prostate_only=prostate_only, use_cache=use_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    default_hpc_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    default_local_path = "./keras_slices_data"
    data_dir = default_hpc_path if os.path.exists(default_hpc_path) else default_local_path

    if os.path.exists(data_dir):
        test_img_dir = os.path.join(data_dir, "keras_slices_test")
        test_mask_dir = os.path.join(data_dir, "keras_slices_seg_test")

        if os.path.exists(test_img_dir) and os.path.exists(test_mask_dir):
            test_ds = HipMRI2DSliceDataset(test_img_dir, test_mask_dir, is_train=False, target_size=(256, 128), prostate_only=True)
            test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
            images, masks = next(iter(test_loader))
            print(f"Loaded test batch: images {images.shape}, masks {masks.shape}")
        else:
            print(f"Test subfolders not found under {data_dir}")
    else:
        print(f"Data directory not found at {data_dir}")
