"""
Data loading and preprocessing for the HipMRI Study 2D slices (keras_slices_data).

Each slice is a 2D Nifti file. Images are z-score normalised per slice and masks are kept as
integer class labels (no one-hot needed, as CrossEntropyLoss takes class indices directly).
"""
import os
import re

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import Dataset


# ==========================================
# Data Paths
# ==========================================
# Root of the HipMRI dataset, set via HIPMRI_ROOT in a .env file next to this script (not committed)
# or as an environment variable (e.g. /home/groups/comp3710/HipMRI_Study_open on Rangpur).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
HIPMRI_ROOT = os.environ["HIPMRI_ROOT"]
SLICES_PATH = os.path.join(HIPMRI_ROOT, "keras_slices_data")

TRAIN_IMG_PATH = os.path.join(SLICES_PATH, "keras_slices_train")
TRAIN_SEG_PATH = os.path.join(SLICES_PATH, "keras_slices_seg_train")
VALIDATE_IMG_PATH = os.path.join(SLICES_PATH, "keras_slices_validate")
VALIDATE_SEG_PATH = os.path.join(SLICES_PATH, "keras_slices_seg_validate")
TEST_IMG_PATH = os.path.join(SLICES_PATH, "keras_slices_test")
TEST_SEG_PATH = os.path.join(SLICES_PATH, "keras_slices_seg_test")

# 0 = Background, 1 = Body, 2 = Bone, 3 = Bladder, 4 = Rectum, 5 = Prostate
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE_CLASS = 5

# Most slices are 256x128; any that differ are resized so they can be batched together
IMG_SIZE = (256, 128)


def load_nifti_slice(path):
    """Load a single 2D Nifti slice as a float32 NumPy array (adapted from spec Appendix B)."""
    image = nib.load(path).get_fdata(caching="unchanged")
    if image.ndim == 3:
        # Some HipMRI slices carry an extra trailing dimension
        image = image[:, :, 0]
    return image.astype(np.float32)


def pair_key(file_name):
    """Shared key between an image and its mask, e.g. 'case_004_week_0_slice_0' -> '004_week_0_slice_0'."""
    return file_name.split("_", 1)[1]


def patient_id(file_name):
    """Patient ID is the first number in the file name (e.g. 'case_004_...' -> '004')."""
    match = re.search(r"\d+", file_name)
    return match.group() if match else file_name


def is_nifti(file_name):
    return file_name.endswith(".nii") or file_name.endswith(".nii.gz")


class HipMRISliceDataset(Dataset):
    """Returns (image, mask) pairs of 2D HipMRI slices: image (1, H, W) float, mask (H, W) long."""

    def __init__(self, image_dir, seg_dir, img_size=IMG_SIZE):
        self.img_size = img_size

        # Pair images and masks by name rather than relying on sorted order lining up
        images = sorted(f for f in os.listdir(image_dir) if is_nifti(f))
        masks = {pair_key(f): f for f in os.listdir(seg_dir) if is_nifti(f)}

        self.pairs = []
        for f in images:
            key = pair_key(f)
            if key not in masks:
                raise FileNotFoundError(f"No mask found for {f} in {seg_dir}")
            self.pairs.append((os.path.join(image_dir, f), os.path.join(seg_dir, masks[key])))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]

        # 1. Load Image and z-score normalise per slice
        image = load_nifti_slice(img_path)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = torch.from_numpy(image)[None, None]  # (1, 1, H, W) for interpolate

        # 2. Load Mask as integer class labels
        mask = np.rint(load_nifti_slice(seg_path)).astype(np.int64)
        mask = torch.from_numpy(mask)[None, None].float()

        # Bilinear for the image, NEAREST for the mask so class labels are never blended
        image = F.interpolate(image, size=self.img_size, mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask, size=self.img_size, mode="nearest")[0, 0].long()

        return image, mask

    def name(self, idx):
        return os.path.basename(self.pairs[idx][0])

    def patient_ids(self):
        return {patient_id(os.path.basename(img_path)) for img_path, _ in self.pairs}


def check_patient_leakage(**splits):
    """Print any patient IDs shared between splits. Returns True if the splits are patient-disjoint."""
    ids = {name: ds.patient_ids() for name, ds in splits.items()}
    names = list(ids)
    clean = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ids[names[i]] & ids[names[j]]
            if overlap:
                clean = False
                print(f"WARNING: patients shared between {names[i]} and {names[j]}: {sorted(overlap)}")
    for name in names:
        print(f"{name}: {len(splits[name])} slices from {len(ids[name])} patients")
    return clean
