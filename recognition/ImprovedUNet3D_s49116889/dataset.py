"""
dataset.py - Data loading, preprocessing and leakage-free splitting for the
HipMRI 3D prostate study.

Raw data (Rangpur): /home/groups/comp3710/HipMRI_Study_open
    semantic_MRs/          B006_Week0_LFOV.nii.gz       (211 volumes)
    semantic_labels_only/  B006_Week0_SEMANTIC.nii.gz   (211 volumes)
    38 patients, each scanned at several weeks. Volumes are 256x256x128 or
    256x256x144 voxels at ~1.56 mm isotropic. Labels take values 0-5:
    0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate.

Preprocessing (done once and cached to a .pt file):
    1. z-score normalise each MR volume (zero mean, unit std).
    2. Pad the slice axis to 160 so every volume has the same shape without
       cropping away any anatomy (128- and 144-slice volumes both fit).
    3. Downsample x2 in every axis -> 128 x 128 x 80 voxels at ~3.1 mm.
       Images use average pooling (anti-aliased); labels use nearest
       neighbour so class values are never blended.
    4. Store as (D, H, W) = (slices, rows, cols), float16 / uint8.

Splitting:
    Patients (not volumes, not slices) are assigned to train / val / test,
    so all weekly scans of one patient stay in one split. Splitting by volume
    would leak near-identical anatomy from the same patient into the test set.

Datasets:
    Volume3DDataset - whole volumes for the 3D Improved U-Net.
    Slice2DDataset  - every axial slice of the same volumes for the 2D U-Net.
"""

import json
import os
import random
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
DEFAULT_CACHE = os.path.expanduser("~/hipmri_cache/hipmri_3d_128x128x80.pt")

CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE = 5

PAD_SLICES = 160        # slice-axis size after padding (multiple of 32)
DOWNSAMPLE = 2          # factor applied to every axis


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def case_key(filename: str) -> str:
    """'B006_Week0_LFOV.nii.gz' -> 'B006_Week0' (patient + week)."""
    parts = filename.split("_")
    return f"{parts[0]}_{parts[1]}"


def patient_id(key: str) -> str:
    """'B006_Week0' -> 'B006'."""
    return key.split("_")[0]


def find_cases(root: str = DEFAULT_ROOT) -> list:
    """
    Pair every MR volume with its label volume by case key.

    Returns a sorted list of (key, mr_path, label_path). Raises if any MR
    volume has no matching label, so a silent mismatch can't corrupt training.
    """
    mr_dir = os.path.join(root, "semantic_MRs")
    lab_dir = os.path.join(root, "semantic_labels_only")
    mrs = {case_key(f): os.path.join(mr_dir, f)
           for f in os.listdir(mr_dir) if f.endswith(".nii.gz")}
    labs = {case_key(f): os.path.join(lab_dir, f)
            for f in os.listdir(lab_dir) if f.endswith(".nii.gz")}
    missing = sorted(set(mrs) ^ set(labs))
    if missing:
        raise RuntimeError(f"Unpaired cases (no matching image/label): {missing}")
    return [(k, mrs[k], labs[k]) for k in sorted(mrs)]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI file as a 3D float32 array (drop any trailing 4th dim)."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 4:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def preprocess_case(mr_path: str, label_path: str):
    """
    Load one image/label pair and apply steps 1-4 from the module docstring.

    Returns (image, label) tensors of shape (D, H, W) = (80, 128, 128).
    """
    img = _load_nifti(mr_path)
    lab = _load_nifti(label_path)
    if img.shape != lab.shape:
        raise ValueError(f"Shape mismatch {img.shape} vs {lab.shape}: {mr_path}")

    # 1. z-score normalisation over the whole volume
    img = (img - img.mean()) / (img.std() + 1e-8)

    # NIfTI (X, Y, Z) -> torch (D=Z, H=X, W=Y)
    img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    lab_t = torch.from_numpy(np.rint(lab).astype(np.int64)).permute(2, 0, 1).contiguous()

    # 2. pad slice axis to PAD_SLICES (image padded with its minimum = air)
    depth = img_t.shape[0]
    if depth > PAD_SLICES:
        raise ValueError(f"Volume has {depth} slices > {PAD_SLICES}: {mr_path}")
    pad = PAD_SLICES - depth
    before, after = pad // 2, pad - pad // 2
    img_t = F.pad(img_t, (0, 0, 0, 0, before, after), value=float(img_t.min()))
    lab_t = F.pad(lab_t, (0, 0, 0, 0, before, after), value=0)

    # 3. downsample x2: average pool for image, strided nearest for labels
    img_t = F.avg_pool3d(img_t[None, None], kernel_size=DOWNSAMPLE)[0, 0]
    lab_t = lab_t[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]

    return img_t.half(), lab_t.to(torch.uint8)


def build_cache(root: str = DEFAULT_ROOT, cache_path: str = DEFAULT_CACHE,
                limit: int = None) -> dict:
    """
    Preprocess every case once and save to `cache_path`.

    The cache holds {'keys', 'images' (N,D,H,W) float16, 'labels' (N,D,H,W)
    uint8}; about 0.8 GB for all 211 volumes. `limit` processes only the first
    few cases (useful for a quick smoke test).
    """
    cases = find_cases(root)
    if limit:
        cases = cases[:limit]
    keys, images, labels = [], [], []
    for i, (key, mr, lab) in enumerate(cases):
        img_t, lab_t = preprocess_case(mr, lab)
        keys.append(key)
        images.append(img_t)
        labels.append(lab_t)
        if (i + 1) % 20 == 0 or i + 1 == len(cases):
            print(f"  preprocessed {i + 1}/{len(cases)}", flush=True)
    data = {"keys": keys, "images": torch.stack(images), "labels": torch.stack(labels)}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(data, cache_path)
    print(f"Saved cache to {cache_path}")
    return data


def load_data(root: str = DEFAULT_ROOT, cache_path: str = DEFAULT_CACHE) -> dict:
    """Load the preprocessed cache, building it on first use."""
    if not os.path.exists(cache_path):
        print(f"No cache at {cache_path}; preprocessing raw NIfTI files...")
        return build_cache(root, cache_path)
    return torch.load(cache_path)


# ---------------------------------------------------------------------------
# Leakage-free patient-level split
# ---------------------------------------------------------------------------

def patient_split(keys: list, val_frac: float = 0.15, test_frac: float = 0.15,
                  seed: int = 42) -> dict:
    """
    Split case indices into train/val/test by patient.

    Patients are shuffled with a fixed seed, so the split is reproducible.
    Returns {'train': [...], 'val': [...], 'test': [...]} of indices into
    `keys`, plus the patient IDs in each split under '<split>_patients'.
    """
    by_patient = defaultdict(list)
    for i, k in enumerate(keys):
        by_patient[patient_id(k)].append(i)
    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    groups = {
        "test": patients[:n_test],
        "val": patients[n_test:n_test + n_val],
        "train": patients[n_test + n_val:],
    }
    split = {}
    for name, pats in groups.items():
        split[name] = sorted(i for p in pats for i in by_patient[p])
        split[f"{name}_patients"] = sorted(pats)
    # Safety check: no patient appears in two splits
    assert not (set(groups["train"]) & set(groups["val"])), "train/val leak"
    assert not (set(groups["train"]) & set(groups["test"])), "train/test leak"
    assert not (set(groups["val"]) & set(groups["test"])), "val/test leak"
    return split


def save_split(split: dict, keys: list, path: str) -> None:
    """Write the split (with case keys) to JSON for the README / reproducibility."""
    out = {name: [keys[i] for i in split[name]] for name in ("train", "val", "test")}
    out.update({k: v for k, v in split.items() if k.endswith("_patients")})
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Augmentation (pure PyTorch, applied on the fly to training data only)
# ---------------------------------------------------------------------------

def random_affine_3d(img: torch.Tensor, lab: torch.Tensor,
                     max_rot_deg: float = 10.0, scale_range=(0.9, 1.1)):
    """
    Random in-plane rotation + isotropic scaling of a (D, H, W) volume.

    The image is resampled bilinearly, the label with nearest neighbour so
    labels stay valid integers. Rotation is about the slice axis, matching
    the realistic variation in patient positioning on the couch.
    """
    angle = torch.empty(1).uniform_(-max_rot_deg, max_rot_deg).item() * torch.pi / 180
    s = torch.empty(1).uniform_(*scale_range).item()
    cos, sin = torch.cos(torch.tensor(angle)) / s, torch.sin(torch.tensor(angle)) / s
    # affine_grid uses (x=W, y=H, z=D) order
    theta = torch.tensor([[cos, -sin, 0, 0],
                          [sin, cos, 0, 0],
                          [0, 0, 1 / s, 0]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1) + tuple(img.shape), align_corners=False)
    img = F.grid_sample(img[None, None].float(), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)[0, 0]
    lab = F.grid_sample(lab[None, None].float(), grid, mode="nearest",
                        padding_mode="zeros", align_corners=False)[0, 0].long()
    return img, lab


def augment(img: torch.Tensor, lab: torch.Tensor):
    """
    Training augmentation for a (D, H, W) volume or (H, W) slice:
      * left-right flip (pelvic anatomy is roughly symmetric), p = 0.5
      * random rotation/scale (3D volumes only), p = 0.5
      * random intensity scale and shift, simulating scanner variation
    """
    if torch.rand(1) < 0.5:
        img, lab = img.flip(-1), lab.flip(-1)
    if img.ndim == 3 and torch.rand(1) < 0.5:
        img, lab = random_affine_3d(img, lab)
    img = img * torch.empty(1).uniform_(0.9, 1.1) + torch.empty(1).uniform_(-0.1, 0.1)
    return img, lab


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class Volume3DDataset(Dataset):
    """
    Whole preprocessed volumes for the 3D model.

    Item: (image (1, D, H, W) float32, label (D, H, W) int64, case key)
    """

    def __init__(self, data: dict, indices: list, train: bool = False):
        self.images = data["images"]
        self.labels = data["labels"]
        self.keys = data["keys"]
        self.indices = list(indices)
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.images[idx].float()
        lab = self.labels[idx].long()
        if self.train:
            img, lab = augment(img, lab)
        return img[None], lab, self.keys[idx]


class Slice2DDataset(Dataset):
    """
    Every axial slice of the selected volumes, for the 2D baseline.

    Uses exactly the same preprocessed volumes and patient split as the 3D
    model, so the two are evaluated on identical test data.

    Item: (image (1, H, W) float32, label (H, W) int64)
    """

    def __init__(self, data: dict, indices: list, train: bool = False,
                 skip_empty: bool = False):
        self.images = data["images"]
        self.labels = data["labels"]
        self.train = train
        self.items = []
        for idx in indices:
            for z in range(self.images.shape[1]):
                # Padding slices are kept by default: the 3D model sees them
                # too, and at test time the 2D model must label every slice
                # of the volume, including empty ones.
                if skip_empty and self.labels[idx, z].max() == 0:
                    continue
                self.items.append((idx, z))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        idx, z = self.items[i]
        img = self.images[idx, z].float()
        lab = self.labels[idx, z].long()
        if self.train:
            img, lab = augment(img, lab)
        return img[None], lab


# ---------------------------------------------------------------------------
# Quick self-test:  python dataset.py [--limit N]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build cache and check the split")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--limit", type=int, default=None,
                        help="only preprocess the first N cases (smoke test)")
    args = parser.parse_args()

    data = build_cache(args.root, args.cache, args.limit) if args.limit \
        else load_data(args.root, args.cache)
    keys = data["keys"]
    print(f"Volumes: {len(keys)}  image tensor: {tuple(data['images'].shape)}")

    split = patient_split(keys)
    for name in ("train", "val", "test"):
        print(f"{name:5s}: {len(split[name + '_patients']):2d} patients, "
              f"{len(split[name]):3d} volumes")

    counts = torch.bincount(data["labels"].flatten().long(), minlength=NUM_CLASSES)
    for c, name in enumerate(CLASS_NAMES):
        print(f"  class {c} {name:10s}: {100 * counts[c].item() / counts.sum().item():6.3f}% of voxels")

    img, lab, key = Volume3DDataset(data, split["train"], train=True)[0]
    print(f"3D item {key}: image {tuple(img.shape)}, label {tuple(lab.shape)}")
    s_img, s_lab = Slice2DDataset(data, split["train"], train=True)[0]
    print(f"2D item: image {tuple(s_img.shape)}, label {tuple(s_lab.shape)}")
