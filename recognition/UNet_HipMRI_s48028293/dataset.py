"""
dataset.py manages the data loading, preprocessing, splitting and image augmentation of the HipMRI prostate dataset.
"""

import glob
import math
import os
import random
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Constants
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE = 5
IMG_SIZE = (256, 128)

SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "val": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

# 3D volumes
DATA_ROOT_2D = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
DATA_ROOT_3D = "/home/groups/comp3710/HipMRI_Study_open"
VOLUME_DIRS = ("semantic_MRs", "semantic_labels_only")
DOWNSAMPLE = (2, 2, 1)      # downsampling factor per axis of the native volume
VOL_SIZE = (128, 128, 128)   # after downsampling; each axis divisible by 16

# SPLIT AND FILE HANDLING FUNCTIONS
def parse_ids(path):
    # Extract the patient, week and slice numbers from a 2D slice file name.
    nums = re.findall(r"\d+", os.path.basename(path))
    if len(nums) < 3:
        raise ValueError(
            f"Expected case, week and slice numbers in '{os.path.basename(path)}'. "
            "Update parse_ids() to match the real file naming."
        )
    return int(nums[0]), int(nums[1]), int(nums[2])

def list_pairs(img_dir, seg_dir):
    #Pair every image with its mask by (case, week, slice).
    imgs = {parse_ids(p): p for p in glob.glob(os.path.join(img_dir, "*.nii*"))}
    segs = {parse_ids(p): p for p in glob.glob(os.path.join(seg_dir, "*.nii*"))}
    if not imgs or not segs:
        raise FileNotFoundError(f"No Nifti files found in {img_dir} or {seg_dir}")
    if imgs.keys() != segs.keys():
        unmatched = len(imgs.keys() ^ segs.keys())
        raise ValueError(f"{unmatched} images/masks have no partner in {img_dir} / {seg_dir}")
    return [(imgs[k], segs[k], k) for k in sorted(imgs)]

def parse_volume_ids(path):
    # Extract the patient and week numbers from a 3D volume file name.
    nums = re.findall(r"\d+", os.path.basename(path))
    if len(nums) < 2:
        raise ValueError(
            f"Expected case and week numbers in '{os.path.basename(path)}'. "
            "Update parse_volume_ids() to match the real file naming."
        )
    return int(nums[0]), int(nums[1])

def list_volume_pairs(img_dir, seg_dir):
    # Pair every 3D scan with its label volume by (case, week).
    imgs = {parse_volume_ids(p): p for p in glob.glob(os.path.join(img_dir, "*.nii*"))}
    segs = {parse_volume_ids(p): p for p in glob.glob(os.path.join(seg_dir, "*.nii*"))}
    if not imgs or not segs:
        raise FileNotFoundError(f"No Nifti files found in {img_dir} or {seg_dir}")
    if imgs.keys() != segs.keys():
        unmatched = len(imgs.keys() ^ segs.keys())
        raise ValueError(f"{unmatched} images/masks have no partner in {img_dir} / {seg_dir}")
    return [(imgs[k], segs[k], k) for k in sorted(imgs)]

def patients_of(pairs):
    return {key[0] for _, _, key in pairs}

def assert_no_leakage(splits):
    #Checks if any patient appears in more than one split
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = patients_of(splits[names[i]]) & patients_of(splits[names[j]])
            if shared:
                raise ValueError(
                    f"Patient leakage between {names[i]} and {names[j]}: "
                    f"cases {sorted(shared)}. Re-run with --resplit."
                )

def get_splits(data_root):
    #Load the provided 2D train/validate/test split.
    splits = {
        name: list_pairs(os.path.join(data_root, img), os.path.join(data_root, seg))
        for name, (img, seg) in SPLIT_DIRS.items()
    }
    assert_no_leakage(splits)
    return splits

def get_volume_splits(data_root=DATA_ROOT_3D, slices_root=DATA_ROOT_2D):
    # Determine the train/validate/test split for the 3D volumes based on the 2D slice split.
    pairs = list_volume_pairs(*(os.path.join(data_root, d) for d in VOLUME_DIRS))
    slice_splits = get_splits(slices_root)
    owner = {case: name for name, sp in slice_splits.items() for case in patients_of(sp)}
    missing = sorted(patients_of(pairs) - owner.keys())
    if missing:
        print(f"  patients with no 2D slices, assigned to train: {missing}")
    splits = {name: [p for p in pairs if owner.get(p[2][0], "train") == name]
            for name in SPLIT_DIRS}
    assert_no_leakage(splits)
    return splits

# LOAD AND PREPROCESSING FUNCTIONS
def load_slice(path):
    #Load one 2D Nifti slice as a NumPy array (drops a trailing extra dim)
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr

def fit_to_shape(arr, shape):
    #Centre-crop and/or zero-pad a 2D array to the target shape
    out = np.zeros(shape, dtype=arr.dtype)
    src, dst = [], []
    for have, want in zip(arr.shape, shape):
        if have >= want:  # crop
            start = (have - want) // 2
            src.append(slice(start, start + want))
            dst.append(slice(0, want))
        else:  # pad
            start = (want - have) // 2
            src.append(slice(0, have))
            dst.append(slice(start, start + have))
    out[tuple(dst)] = arr[tuple(src)]
    return out

def preprocess_image(arr, img_size):
    # Resize and z-score a 2D image slice
    arr = fit_to_shape(arr.astype(np.float32), img_size)
    return (arr - arr.mean()) / (arr.std() + 1e-8)  # per-slice z-score

def preprocess_mask(arr, img_size):
    # Resize and round a 2D mask slice
    return fit_to_shape(np.rint(arr).astype(np.uint8), img_size)

def load_volume(path):
    #Load one 3D Nifti volume as a NumPy array (drops a trailing extra dim)
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 4:
        arr = arr[:, :, :, 0]
    return arr

def downsampled_shape(shape, factor):
    # Compute the shape of a volume after downsampling
    return tuple(max(1, round(s / f)) for s, f in zip(shape, factor))

def downsample_image(arr, factor):
    # Area (block-average) downsampling, which also avoids aliasing.
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=downsampled_shape(arr.shape, factor), mode="area")
    return t[0, 0].numpy()

def downsample_mask(arr, factor):
    # Area downsampling of a mask: convert to one-hot, downsample, then take argmax.
    labels = torch.from_numpy(np.rint(arr).astype(np.int64))
    onehot = F.one_hot(labels, NUM_CLASSES).movedim(-1, 0).float()
    onehot = F.interpolate(onehot[None], size=downsampled_shape(arr.shape, factor), mode="area")
    return onehot[0].argmax(0).numpy().astype(np.uint8)

def preprocess_volume(img, seg, vol_size=VOL_SIZE, factor=DOWNSAMPLE):
    # Downsample, crop and pad a 3D volume and its mask, then z-score the image.
    img = fit_to_shape(downsample_image(img, factor), vol_size)
    seg = fit_to_shape(downsample_mask(seg, factor), vol_size)
    img = (img - img.mean()) / (img.std() + 1e-8) 
    return img.astype(np.float32), seg

# AUGMENTATION CLASSES
class Augment:
    # Random small rotation, scaling and translation applied identically to
    # the image (bilinear) and mask (nearest), plus intensity jitter on the image.
    def __init__(self, max_rot_deg=10.0, max_scale=0.1, max_shift=0.05, intensity=0.1):
        self.max_rot = math.radians(max_rot_deg)
        self.max_scale = max_scale
        self.max_shift = max_shift
        self.intensity = intensity

    # Apply the augmentation to a 2D image and mask.
    def __call__(self, img, mask):
        h, w = mask.shape
        angle = random.uniform(-self.max_rot, self.max_rot)
        scale = 1.0 + random.uniform(-self.max_scale, self.max_scale)
        tx = random.uniform(-self.max_shift, self.max_shift) * 2
        ty = random.uniform(-self.max_shift, self.max_shift) * 2
        cos, sin = math.cos(angle) * scale, math.sin(angle) * scale
        theta = torch.tensor([[cos, -sin * h / w, tx],
                              [sin * w / h, cos, ty]], dtype=torch.float32)
        grid = F.affine_grid(theta[None], (1, 1, h, w), align_corners=False)
        img = F.grid_sample(img[None], grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[0]
        mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[0, 0].long()

        img = img * (1.0 + random.uniform(-self.intensity, self.intensity))
        img = img + random.uniform(-self.intensity, self.intensity)
        return img, mask

class Augment3D(Augment):
    # Random small rotation, scaling and translation applied identically to
    #  the image (bilinear) and mask (nearest), plus intensity jitter on the image.
    def __call__(self, img, mask):
        nx, ny, nz = mask.shape
        angle = random.uniform(-self.max_rot, self.max_rot)
        s = 1.0 + random.uniform(-self.max_scale, self.max_scale)
        sz = 1.0 + random.uniform(-self.max_scale, self.max_scale) / 2
        tx = random.uniform(-self.max_shift, self.max_shift) * 2
        ty = random.uniform(-self.max_shift, self.max_shift) * 2
        cos, sin = math.cos(angle) * s, math.sin(angle) * s
        theta = torch.tensor([[sz, 0.0, 0.0, 0.0],
                              [0.0, cos, -sin * nx / ny, ty],
                              [0.0, sin * ny / nx, cos, tx]], dtype=torch.float32)
        grid = F.affine_grid(theta[None], (1, 1, nx, ny, nz), align_corners=False)
        img = F.grid_sample(img[None], grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[0]
        mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[0, 0].long()

        img = img * (1.0 + random.uniform(-self.intensity, self.intensity))
        img = img + random.uniform(-self.intensity, self.intensity)
        return img, mask

# DATASETS AND LOADERS CLASSES
class HipMRI2D(Dataset):
    # HipMRI 2D slices, downsampled.
    def __init__(self, pairs, img_size=IMG_SIZE, augment=None, preload=True):
        self.pairs = pairs
        self.img_size = tuple(img_size)
        self.augment = augment
        self.cache = None
        if preload:
            n = len(pairs)
            images = np.zeros((n,) + self.img_size, dtype=np.float32)
            masks = np.zeros((n,) + self.img_size, dtype=np.uint8)
            for i, (img_path, seg_path, _) in enumerate(pairs):
                images[i] = preprocess_image(load_slice(img_path), self.img_size)
                masks[i] = preprocess_mask(load_slice(seg_path), self.img_size)
                if (i + 1) % 2000 == 0:
                    print(f"  loaded {i + 1}/{n} slices", flush=True)
            if masks.max() >= NUM_CLASSES:
                raise ValueError(f"Found label {masks.max()} but NUM_CLASSES is {NUM_CLASSES}")
            self.cache = (images, masks)

    # Return the number of slices in this dataset.
    def __len__(self):
        return len(self.pairs)

    #  Return the (image, mask, case_id, week, slice_idx) tuple for the given index.
    def __getitem__(self, idx):
        img_path, seg_path, (case, week, slice_idx) = self.pairs[idx]
        if self.cache is not None:
            img, mask = self.cache[0][idx], self.cache[1][idx]
        else:
            img = preprocess_image(load_slice(img_path), self.img_size)
            mask = preprocess_mask(load_slice(seg_path), self.img_size)
        img = torch.from_numpy(np.array(img))[None]             # (1, H, W)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (H, W)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return img, mask, case, week, slice_idx

class HipMRI3D(Dataset):
    # HipMRI 3D weekly scans, downsampled.
    def __init__(self, pairs, vol_size=VOL_SIZE, downsample=DOWNSAMPLE, augment=None, preload=True):
        self.pairs = pairs
        self.vol_size = tuple(vol_size)
        self.downsample = tuple(downsample)
        self.augment = augment
        self.cache = None
        if preload:
            n = len(pairs)
            images = np.zeros((n,) + self.vol_size, dtype=np.float32)
            masks = np.zeros((n,) + self.vol_size, dtype=np.uint8)
            for i in range(n):
                images[i], masks[i] = self._load(i)
                if (i + 1) % 25 == 0:
                    print(f"  loaded {i + 1}/{n} volumes", flush=True)
            if masks.max() >= NUM_CLASSES:
                raise ValueError(f"Found label {masks.max()} but NUM_CLASSES is {NUM_CLASSES}")
            self.cache = (images, masks)

    # Load and preprocess one 3D volume and its mask.
    def _load(self, idx):
        img_path, seg_path, _ = self.pairs[idx]
        return preprocess_volume(load_volume(img_path), load_volume(seg_path),
                                self.vol_size, self.downsample)

    # Return the number of volumes in this dataset.
    def __len__(self):
        return len(self.pairs)

    # Return the (image, mask, case_id, week, -1) tuple for the given index.
    def __getitem__(self, idx):
        case, week = self.pairs[idx][2]
        if self.cache is not None:
            img, mask = self.cache[0][idx], self.cache[1][idx]
        else:
            img, mask = self._load(idx)
        img = torch.from_numpy(np.array(img))[None]             # (1, X, Y, Z)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # (X, Y, Z)
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return img, mask, case, week, -1
    
def get_loaders(data_root, batch_size=32, img_size=IMG_SIZE, workers=4, limit=None, preload=True, dims=2):
    #Build the train/val/test loaders for the 2D slices (dims=2) or the 3D volumes (dims=3).
    if dims == 2:
        splits = get_splits(data_root)
    else:
        splits = get_volume_splits(data_root)
    if limit:
        splits = {name: pairs[:limit] for name, pairs in splits.items()}
    unit = "slices" if dims == 2 else "volumes"
    for name, pairs in splits.items():
        print(f"{name:5s}: {len(pairs):6d} {unit} from {len(patients_of(pairs))} patients")

    if dims == 2:
        sets = {
            "train": HipMRI2D(splits["train"], img_size, augment=Augment(), preload=preload),
            "val": HipMRI2D(splits["val"], img_size, preload=preload),
            "test": HipMRI2D(splits["test"], img_size, preload=preload),
        }
    else:
        sets = {
            "train": HipMRI3D(splits["train"], img_size, augment=Augment3D(), preload=preload),
            "val": HipMRI3D(splits["val"], img_size, preload=preload),
            "test": HipMRI3D(splits["test"], img_size, preload=preload),
        }
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=True,
                    persistent_workers=workers > 0)
    return {
        "train": DataLoader(sets["train"], shuffle=True, drop_last=True, **common),
        "val": DataLoader(sets["val"], shuffle=False, **common),
        "test": DataLoader(sets["test"], shuffle=False, **common),
    }

def inspect_data(data_root, n_files=300):
    # Print patient counts, leakage status, image shapes and label values."""
    splits = {
        name: list_pairs(os.path.join(data_root, img), os.path.join(data_root, seg))
        for name, (img, seg) in SPLIT_DIRS.items()
    }
    for name, pairs in splits.items():
        print(f"{name:5s}: {len(pairs):6d} slices, patients {sorted(patients_of(pairs))}")
        print(f"       example file: {os.path.basename(pairs[0][0])}")
    try:
        assert_no_leakage(splits)
        print("Provided split is leakage-free at the patient level.")
    except ValueError as err:
        print("WARNING:", err)

    pairs = splits["train"]
    step = max(1, len(pairs) // n_files)
    shapes, labels = {}, set()
    for img_path, seg_path, _ in pairs[::step]:
        shape = nib.load(img_path).shape
        shapes[shape] = shapes.get(shape, 0) + 1
        labels.update(np.unique(load_slice(seg_path)).tolist())
    print("Image shapes in sample:", shapes)
    print("Label values in sample:", sorted(labels), f"(expecting 0..{NUM_CLASSES - 1})")

if __name__ == "__main__":
    import sys
    default_root = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    inspect_data(sys.argv[1] if len(sys.argv) > 1 else default_root)