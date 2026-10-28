"""
Data loading, patient-level splitting and augmentation for the HipMRI Study 3D volumes.

Each volume is a 3D Nifti file (semantic_MRs) with a matching label volume (semantic_labels_only).
Images are z-score normalised per volume and both are downsampled to a fixed size. Masks are kept as
integer class labels (CrossEntropyLoss takes class indices directly).

There is no predefined 3D split, so patients are split here with a fixed seed, keeping every weekly
scan of a patient in the same split.
"""
import math
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
MR_PATH = os.path.join(HIPMRI_ROOT, "semantic_MRs")
SEG_PATH = os.path.join(HIPMRI_ROOT, "semantic_labels_only")

# 0 = Background, 1 = Body, 2 = Bone, 3 = Bladder, 4 = Rectum, 5 = Prostate
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE_CLASS = 5

# Volumes are 256x256x128 (one is 256x256x144) with ~1.6 mm near-isotropic voxels. Full resolution keeps
# the prostate (0.1-0.3% of voxels) boundary as sharp as possible; half resolution (128, 128, 64) was the
# first experiment. Every axis must stay divisible by 16 for the 4 downsampling steps.
VOLUME_DEPTH = 128
VOLUME_SIZE = (256, 256, 128)


def load_nifti_volume(path):
    """Load a single 3D Nifti volume as a float32 NumPy array (adapted from spec Appendix B)."""
    volume = nib.load(path).get_fdata(caching="unchanged")
    if volume.ndim == 4:
        # Some HipMRI volumes carry an extra trailing dimension
        volume = volume[:, :, :, 0]
    # Clip extra slices so every volume has the same depth and voxel spacing before resizing
    return volume[:, :, :VOLUME_DEPTH].astype(np.float32)


def pair_key(file_name):
    """Shared key between an image and its mask, e.g. 'B006_Week0_LFOV.nii.gz' -> 'B006_Week0'."""
    return file_name.rsplit("_", 1)[0]


def patient_id(file_name):
    """Patient ID is the first number in the file name ('B006_...' -> '006', 'case_006_...' -> '006')."""
    match = re.search(r"\d+", file_name)
    return match.group() if match else file_name


def is_nifti(file_name):
    return file_name.endswith(".nii") or file_name.endswith(".nii.gz")


# ==========================================
# Patient-level Split
# ==========================================
def split_volumes(mr_dir=MR_PATH, val_fraction=0.1, test_fraction=0.1, seed=42):
    """Split volume file names into train/validate/test by patient.

    Patients (not volumes) are shuffled with a fixed seed, so every weekly scan of a patient goes to
    the same split and train.py / predict.py always reproduce the same split.
    """
    files = sorted(f for f in os.listdir(mr_dir) if is_nifti(f))
    patients = sorted({patient_id(f) for f in files})

    order = torch.randperm(len(patients), generator=torch.Generator().manual_seed(seed)).tolist()
    n_test = round(test_fraction * len(patients))
    n_val = round(val_fraction * len(patients))
    test_patients = {patients[i] for i in order[:n_test]}
    val_patients = {patients[i] for i in order[n_test:n_test + n_val]}

    splits = {"train": [], "validate": [], "test": []}
    for f in files:
        pid = patient_id(f)
        if pid in test_patients:
            splits["test"].append(f)
        elif pid in val_patients:
            splits["validate"].append(f)
        else:
            splits["train"].append(f)
    return splits


# ==========================================
# 3D Augmentation
# ==========================================
# Only mirror left-right: the pelvis is roughly symmetric that way, whereas flipping the other axes
# would put the bladder below the prostate or the rectum in front of it, which never occurs in real scans.
# Axis 0 is left-right for RAS/LAS-oriented volumes (check with nib.aff2axcodes(nib.load(path).affine)).
FLIP_AXES = (0,)


def random_flip(image, mask):
    """Mirror along each axis in FLIP_AXES with probability 0.5."""
    for axis in FLIP_AXES:
        if torch.rand(1).item() < 0.5:
            image = image.flip(axis + 1)  # image has a leading channel dimension
            mask = mask.flip(axis)
    return image, mask


def random_rotate_scale(image, mask, max_angle=10.0, scale_range=(0.9, 1.1)):
    """Small random in-plane rotation and isotropic scaling, applied identically to image and mask."""
    angle = math.radians((torch.rand(1).item() * 2 - 1) * max_angle)
    scale = scale_range[0] + torch.rand(1).item() * (scale_range[1] - scale_range[0])
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale

    # affine_grid uses (x, y, z) = (W, H, D); the in-plane axes are D and H, so rotate in the (y, z) plane
    theta = torch.tensor([[[1 / scale, 0, 0, 0],
                           [0, cos, -sin, 0],
                           [0, sin, cos, 0]]], dtype=torch.float32)
    grid = F.affine_grid(theta, (1,) + tuple(image.shape), align_corners=False)

    # Border padding for the image (z-scored background is not 0), nearest for the mask
    image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
    mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest", padding_mode="zeros",
                         align_corners=False)[0, 0].long()
    return image, mask


def random_intensity(image, scale=0.1, shift=0.1, noise=0.05):
    """Random brightness/contrast change and Gaussian noise (image only)."""
    image = image * (1 + (torch.rand(1).item() * 2 - 1) * scale)
    image = image + (torch.rand(1).item() * 2 - 1) * shift
    return image + torch.randn_like(image) * noise


def augment_volume(image, mask):
    image, mask = random_flip(image, mask)
    image, mask = random_rotate_scale(image, mask)
    image = random_intensity(image)
    return image, mask


# ==========================================
# Dataset
# ==========================================
class HipMRIVolumeDataset(Dataset):
    """Returns (image, mask) pairs of 3D HipMRI volumes: image (1, D, H, W) float, mask (D, H, W) long.

    Volumes are preloaded and downsampled once in __init__, since decompressing the full-size .nii.gz
    files every epoch would dominate training time (~32 MB per volume at full resolution).
    """

    def __init__(self, file_names, mr_dir=MR_PATH, seg_dir=SEG_PATH, volume_size=VOLUME_SIZE,
                 augment=False):
        self.file_names = list(file_names)
        self.augment = augment

        # Pair images and masks by name rather than relying on sorted order lining up
        masks = {pair_key(f): f for f in os.listdir(seg_dir) if is_nifti(f)}

        self.images, self.masks = [], []
        for f in self.file_names:
            key = pair_key(f)
            if key not in masks:
                raise FileNotFoundError(f"No mask found for {f} in {seg_dir}")

            # 1. Load Image and z-score normalise per volume
            image = load_nifti_volume(os.path.join(mr_dir, f))
            image = (image - image.mean()) / (image.std() + 1e-8)
            image = torch.from_numpy(image)[None, None]  # (1, 1, D, H, W) for interpolate

            # 2. Load Mask as integer class labels
            mask = np.rint(load_nifti_volume(os.path.join(seg_dir, masks[key]))).astype(np.int64)
            mask = torch.from_numpy(mask)[None, None].float()

            # Trilinear for the image, NEAREST for the mask so class labels are never blended
            image = F.interpolate(image, size=volume_size, mode="trilinear", align_corners=False)[0]
            mask = F.interpolate(mask, size=volume_size, mode="nearest")[0, 0].to(torch.uint8)

            self.images.append(image)
            self.masks.append(mask)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx].long()
        if self.augment:
            image, mask = augment_volume(image, mask)
        return image, mask

    def name(self, idx):
        return self.file_names[idx]

    def patient_ids(self):
        return {patient_id(f) for f in self.file_names}


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
        print(f"{name}: {len(splits[name])} volumes from {len(ids[name])} patients")
    return clean
