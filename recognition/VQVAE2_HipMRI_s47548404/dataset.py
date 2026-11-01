"""
dataset.py - Data loading for HipMRI 2D slices (COMP3710, task 2.4).

Uses the provided patient-level split (train / validate / test).
Each item is one 2D slice, normalised per slice to [0, 1],
returned as a float32 tensor of shape (1, 256, 128).

Slices of any other shape are excluded. In the provided data this is one
training scan (case_019_week_1, 256x144): a crop check against the segmentation
masks showed its extra columns contain labelled anatomy, so cropping was rejected.
"""
import argparse
import glob
import os
import re
from collections import Counter

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validate", "test")
TARGET_SHAPE = (256, 128)  # all slices except one 256x144 scan, which is excluded


def patient_id(path):
    """'case_004_week_0_slice_10.nii.gz' -> '004'."""
    return re.search(r"_(\d+)_week_", os.path.basename(path)).group(1)


def load_slice(path):
    """Load one NIfTI slice as a float32 numpy array."""
    return nib.load(path).get_fdata(dtype=np.float32)


def normalise(img):
    """Per-slice min-max scaling to [0, 1]. Flat slices become all zeros."""
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


class HipMRIDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        folder = os.path.join(data_dir, f"keras_slices_{split}")
        all_files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
        if not all_files:
            raise FileNotFoundError(f"No .nii.gz files found in {folder}")
        # Header-only shape check (fast): keep TARGET_SHAPE slices, exclude the rest
        self.files, self.excluded = [], []
        for f in all_files:
            (self.files if nib.load(f).shape == TARGET_SHAPE else self.excluded).append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = normalise(load_slice(self.files[idx]))
        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


def check_split(data_dir, split):
    """Sanity checks: file count, patients, shapes of ALL slices, one batch."""
    ds = HipMRIDataset(data_dir, split)
    patients = {patient_id(f) for f in ds.files}
    patients_all = patients | {patient_id(f) for f in ds.excluded}
    print(f"[{split}] {len(ds)} slices kept from {len(patients)} patients "
          f"({len(patients_all)} patients before exclusion)")
    if ds.excluded:
        scans = Counter(re.sub(r"_slice_\d+.*", "", os.path.basename(f)) for f in ds.excluded)
        print(f"[{split}] excluded (not {TARGET_SHAPE}): {dict(scans)}")
    shapes = Counter(load_slice(f).shape for f in ds.files)
    print(f"[{split}] shapes of kept slices: {dict(shapes)}")

    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    print(f"[{split}] batch shape {tuple(batch.shape)}, "
          f"range {batch.min():.3f} to {batch.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the HipMRI dataset.")
    parser.add_argument("--data-dir", required=True,
                        help="Path to the keras_slices_data folder")
    args = parser.parse_args()
    for split in SPLITS:
        check_split(args.data_dir, split)