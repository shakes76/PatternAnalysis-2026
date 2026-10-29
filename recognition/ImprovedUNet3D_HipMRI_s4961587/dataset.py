"""
dataset.py - HipMRI 3D prostate MRI loading, splitting and preprocessing.

Author: Angus Slater (s4961587)
Course: COMP3710 Pattern Recognition, The University of Queensland, 2026
Project: Improved 3D U-Net for prostate MRI segmentation (HipMRI)

Data: HipMRI Study (Dowling et al., 2015, doi:10.1016/j.ijrobp.2015.08.045),
read in place from /home/groups/comp3710/HipMRI_Study_open. 211 T2 MRI volumes
from 38 patients (1-8 weekly scans each), 256x256x128, LPS orientation.

DataLoaders yield batches of:
- image: (B, 1, 128, 128, 64) float32, z-scored per volume.
- mask:  (B, 128, 128, 64) int64, labels 0-5
         (background, body, bone, bladder, rectum, prostate).

Design decisions:
- Split by patient, not by file: weekly re-scans of one patient are near-
  duplicates, so splitting files would leak test anatomy into training.
- 2x downsample (256x256x128 -> 128x128x64): 8x fewer voxels, so ~8x faster
  training and far less GPU memory; the spec asks for downsampled 3D data.
  Cost: ~3.1 mm voxels, coarser prostate boundaries at base/apex.
- Voxel spacing (1.41-1.88 mm in-plane) is not resampled to a common size;
  tested later as an ablation.

Run `python dataset.py` for a smoke test (split sizes, leakage and
orientation checks, one preprocessed volume).
"""
import glob
import os
import random
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = "/home/groups/comp3710/HipMRI_Study_open"
TARGET_SHAPE = (128, 128, 64)  # 2x downsample on every axis 
NUM_CLASSES = 6                # background, body, bone, bladder, rectum, prostate


def patient_id(path):
    """Extract the patient ID from a scan or mask path.

    Args:
        path: file path, e.g. '.../B040_Week3_LFOV.nii.gz'.

    Returns:
        Patient ID, e.g. 'B040'. Every weekly scan of one patient shares it.
    """
    return os.path.basename(path).split("_")[0]


def list_pairs(root=ROOT):
    """Find every scan and its mask, and pair them.

    Args:
        root: HipMRI_Study_open folder containing semantic_MRs/ and
            semantic_labels_only/.

    Returns:
        Sorted list of (image_path, label_path) tuples (211 for the full set).

    Raises:
        AssertionError: if counts differ or any scan/mask names don't match.
    """
    imgs = sorted(glob.glob(os.path.join(root, "semantic_MRs", "*.nii.gz")))
    labs = sorted(glob.glob(os.path.join(root, "semantic_labels_only", "*.nii.gz")))

    # Smoke test: assert num images = num labels
    assert len(imgs) == len(labs), f"{len(imgs)} images vs {len(labs)} labels"

    for i, l in zip(imgs, labs):
        # 'B040_Week0_LFOV.nii.gz' and 'B040_Week0_SEMANTIC.nii.gz' -> 'B040_Week0.nii.gz'
        assert (os.path.basename(i).replace("_LFOV", "")
                == os.path.basename(l).replace("_SEMANTIC", "")), (i, l)
    return list(zip(imgs, labs))


def split_by_patient(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    """Assign whole patients to train/val/test so no patient appears in two splits.

    Patients have 1-8 scans, so test and val are filled to ~15% of *volumes*
    (not patients) by walking a seeded random order of patients.

    Args:
        pairs: (image_path, label_path) tuples from list_pairs.
        val_frac, test_frac: target fraction of volumes for val and test.
        seed: shuffle seed; the same seed always gives the same split.

    Returns:
        Dict {"train", "val", "test"} -> list of (image_path, label_path).
    """
    by_patient = defaultdict(list)
    for pair in pairs:
        by_patient[patient_id(pair[0])].append(pair)

    patients = sorted(by_patient)          # sort first so the shuffle is reproducible
    random.Random(seed).shuffle(patients)

    n = len(pairs)
    targets = {"test": test_frac * n, "val": val_frac * n}
    splits = {"train": [], "val": [], "test": []}

    for p in patients:
        vols = by_patient[p]
        for name in ("test", "val"):
            if len(splits[name]) + len(vols) <= targets[name] + 4:  # slack so an 8-scan patient fits
                splits[name].extend(vols)
                break
        else:                              # runs only if no `break`: val and test are full
            splits["train"].extend(vols)

    return splits


def load_volume(img_path, lab_path, shape=TARGET_SHAPE):
    """Load one scan + mask, crop to 128 slices, z-score the scan, downsample both.

    One of the 211 scans has 144 slices; cropping drops its top 16 (head end,
    well above the prostate) so every volume has the same shape.

    Args:
        img_path, lab_path: paths to the scan and its mask (.nii.gz).
        shape: output grid size (default 2x downsample to 128x128x64).

    Returns:
        image: (1, X, Y, Z) float32, z-scored, trilinear-downsampled.
        mask: (X, Y, Z) uint8 labels 0-5, nearest-downsampled.
    """
    img = nib.load(img_path).get_fdata(dtype=np.float32)[:, :, :128]
    lab = nib.load(lab_path).get_fdata(dtype=np.float32)[:, :, :128]

    # Per-volume z-score: MRI brightness has no fixed scale between scans.
    img = (img - img.mean()) / (img.std() + 1e-8)

    img = torch.from_numpy(img)[None, None]  # (1, 1, X, Y, Z): F.interpolate needs batch + channel dims
    lab = torch.from_numpy(lab)[None, None]
    img = F.interpolate(img, size=shape, mode="trilinear", align_corners=False)
    lab = F.interpolate(lab, size=shape, mode="nearest")
    return img[0], lab[0, 0].round().to(torch.uint8)        # Round before casting: interpolated labels can be 4.9999, which a cast would truncate to 4.

class HipMRI3D(Dataset):
    """One split's volumes, preprocessed once and held in RAM (~5 MB each),
    so each epoch doesn't re-decompress the .nii.gz files from disk.

    Args:
        pairs: list of (image_path, label_path) from split_by_patient.
        augment: if True, random left-right flip (training split only).
    """

    def __init__(self, pairs, augment=False):
        self.augment = augment
        self.items = [load_volume(i, l) for i, l in pairs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Return (image (1, X, Y, Z) float32, mask (X, Y, Z) int64) for volume idx."""
        img, lab = self.items[idx]
        # Left-right flip (axis 0 in LPS): the pelvis is roughly mirror-symmetric.
        if self.augment and torch.rand(1).item() < 0.5:
            img, lab = img.flip(-3), lab.flip(-3)
        return img, lab.long()  # CrossEntropy / one-hot need int64


def get_loaders(root=ROOT,batch_size=2, seed=42, num_workers=4, augment=False, limit=None):
    """Build train/val/test DataLoaders from a patient-level split.

    Args:
        root: HipMRI_Study_open folder (override with --data_dir in train.py).
        batch_size: training batch size; val/test always use 1 (whole-volume eval).
        seed: split seed, so the same patients land in the same split every run.
        num_workers: DataLoader worker processes (match --cpus-per-task).
        augment: enable left-right flips on the training split.
        limit: cap volumes per split, for quick smoke tests.

    Returns:
        (loaders, splits): dicts keyed "train"/"val"/"test" holding the
        DataLoaders and the (image_path, label_path) lists.
    """
    splits = split_by_patient(list_pairs(root), seed=seed)
    loaders = {}
    for name, pairs in splits.items():
        ds = HipMRI3D(pairs[:limit], augment=augment and name == "train")
        loaders[name] = DataLoader(ds, batch_size=batch_size if name == "train" else 1,
                                   shuffle=(name == "train"), num_workers=num_workers,
                                   pin_memory=True)
    return loaders, splits


if __name__ == "__main__":
    pairs = list_pairs()
    print(f"{len(pairs)} image/label pairs, names match")

    # Orientation check: every scan must use the same axis convention.
    codes = {nib.aff2axcodes(nib.load(i).affine) for i, _ in pairs}
    print("orientations found:", codes)
    assert codes == {("L", "P", "S")}, "mixed orientations!"

    # Split sizes + leakage check.
    splits = split_by_patient(pairs)
    pats = {n: {patient_id(i) for i, _ in ps} for n, ps in splits.items()}
    for n, ps in splits.items():
        print(f"{n:5s}: {len(ps):3d} volumes, {len(pats[n]):2d} patients")
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"]
                or pats["val"] & pats["test"]), "patient leakage!"
    print("no patient overlap between splits")

    # One preprocessed volume.
    img, lab = load_volume(*splits["train"][0])
    print("image", tuple(img.shape), img.dtype, f"mean {img.mean():.2f} std {img.std():.2f}")
    print("mask ", tuple(lab.shape), lab.dtype, "classes", lab.unique().tolist())
    print("prostate voxels after downsampling:", int((lab == 5).sum()))