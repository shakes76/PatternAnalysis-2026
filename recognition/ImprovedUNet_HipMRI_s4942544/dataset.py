"""
dataset.py -- Data loading and preprocessing for HipMRI 2D prostate segmentation.

Design decisions (each one is backed by data_audit.py on the real dataset):

* Splits: the provided keras_slices_{train,validate,test} folders are used as-is.
  The audit confirmed they are already patient-level (31 / 4 / 3 patients, zero
  overlap), so no slice from a test patient is ever seen during training.
* Geometry: 99.5% of slices are 256x128, but one scan is 256x144. Every slice is
  centre-cropped / zero-padded to 256x128. Cropping is used instead of resizing so
  that mask labels are never interpolated and anatomical scale is preserved.
* Intensity: raw MR intensities are unitless and vary between scans (audit: max
  582-738), so each slice is z-score normalised individually.
* Labels: 0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate. Masks are
  returned as class indices; `to_one_hot` produces the multi-channel encoding used
  by the Dice loss.
* Augmentation (training only): random flips, small rotations/scaling and intensity
  jitter. Geometric transforms are applied identically to image and mask, with
  nearest-neighbour sampling for the mask so no invalid labels are created.

Every item also carries metadata (patient, week, slice index, relative position in
its scan). This is what later allows failures to be analysed at the prostate base
and apex -- the end slices that the research question focuses on.
"""
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
TARGET_SHAPE = (256, 128)
NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE = 5

SPLIT_DIRS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}

# Images: case_004_week_0_slice_12.nii.gz   Masks: seg_004_week_0_slice_12.nii.gz
NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)")


def _parse_key(filename):
    m = NAME_RE.search(filename)
    return None if m is None else (m.group(1), m.group(2), int(m.group(3)))


def _load_2d(path):
    """Load a Nifti slice as a 2D float array (drop the occasional trailing singleton dim)."""
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def crop_or_pad(arr, target=TARGET_SHAPE):
    """Centre-crop or zero-pad a 2D array to `target` without any interpolation."""
    out = np.zeros(target, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for dim, size in enumerate(target):
        n = arr.shape[dim]
        if n >= size:  # crop
            start = (n - size) // 2
            src_slices.append(slice(start, start + size))
            dst_slices.append(slice(0, size))
        else:  # pad
            start = (size - n) // 2
            src_slices.append(slice(0, n))
            dst_slices.append(slice(start, start + n))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def normalise(img):
    """Per-slice z-score normalisation (robust to scan-to-scan intensity differences)."""
    return (img - img.mean()) / (img.std() + 1e-8)


def to_one_hot(mask, num_classes=NUM_CLASSES):
    """(N, H, W) class-index mask -> (N, C, H, W) one-hot float tensor."""
    return F.one_hot(mask.long(), num_classes).permute(0, 3, 1, 2).float()


class Augment:
    """
    Paired, label-safe augmentation for one (image, mask) sample.

    Image: bilinear resampling. Mask: nearest-neighbour resampling, so a pixel can
    only ever take an existing label value.
    """

    def __init__(self, max_rotate_deg=10.0, scale_range=(0.9, 1.1), p_flip=0.5,
                 intensity_jitter=0.1):
        self.max_rotate = math.radians(max_rotate_deg)
        self.scale_range = scale_range
        self.p_flip = p_flip
        self.intensity_jitter = intensity_jitter

    def __call__(self, image, mask):
        # image: (1, H, W) float, mask: (H, W) long
        if torch.rand(1).item() < self.p_flip:
            # Mirror across the left-right axis. Pelvic anatomy is approximately
            # left-right symmetric, so this produces realistic new examples.
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        angle = (torch.rand(1).item() * 2 - 1) * self.max_rotate
        scale = self.scale_range[0] + torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
        cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, (1, 1, *image.shape[-2:]), align_corners=False)

        image = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False).squeeze(0)
        mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                             padding_mode="zeros", align_corners=False)[0, 0].long()

        # Mild contrast/brightness jitter mimics coil and scanner variation.
        gain = 1 + (torch.rand(1).item() * 2 - 1) * self.intensity_jitter
        bias = (torch.rand(1).item() * 2 - 1) * self.intensity_jitter
        image = image * gain + bias
        return image, mask


class HipMRISliceDataset(Dataset):
    """
    2D HipMRI slices held in memory.

    Returns (image, mask, meta):
        image: (1, 256, 128) float32, z-score normalised
        mask:  (256, 128) int64 class indices in [0, 5]
        meta:  dict with patient, week, slice, rel_pos (0 = first slice, 1 = last)
    """

    def __init__(self, root=DEFAULT_ROOT, split="train", augment=False, limit=None, num_threads=8):
        if split not in SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(SPLIT_DIRS)}")
        img_dir, seg_dir = (os.path.join(root, d) for d in SPLIT_DIRS[split])

        img_files = {_parse_key(f): f for f in os.listdir(img_dir) if f.endswith(".nii.gz")}
        seg_files = {_parse_key(f): f for f in os.listdir(seg_dir) if f.endswith(".nii.gz")}
        keys = sorted(k for k in img_files if k is not None and k in seg_files)
        if not keys:
            raise RuntimeError(f"No image/mask pairs found in {img_dir} and {seg_dir}")
        if limit:
            keys = keys[:limit]

        # Number of slices in each scan, to express each slice's relative position.
        slices_per_scan = {}
        for p, w, s in keys:
            slices_per_scan[(p, w)] = max(slices_per_scan.get((p, w), 0), s + 1)

        def load_pair(key):
            img = crop_or_pad(_load_2d(os.path.join(img_dir, img_files[key])).astype(np.float32))
            seg = crop_or_pad(_load_2d(os.path.join(seg_dir, seg_files[key])).astype(np.uint8))
            return normalise(img).astype(np.float16), seg

        # Decompressing ~11k .nii.gz files is I/O + zlib bound; threads parallelise it.
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            pairs = list(pool.map(load_pair, keys))

        # float16 storage halves RAM (~0.75 GB for the training split); items are
        # converted back to float32 on access.
        self.images = np.stack([p[0] for p in pairs])
        self.masks = np.stack([p[1] for p in pairs])
        self.meta = [
            {"patient": p, "week": w, "slice": s,
             "rel_pos": s / max(slices_per_scan[(p, w)] - 1, 1)}
            for p, w, s in keys
        ]
        self.augment = Augment() if augment else None
        self.split = split

        bad = set(np.unique(self.masks).tolist()) - set(range(NUM_CLASSES))
        if bad:
            raise ValueError(f"Unexpected label values in {split} masks: {sorted(bad)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return image, mask, self.meta[idx]


def get_dataloaders(root=DEFAULT_ROOT, batch_size=16, num_workers=4, limit=None):
    """Build train (augmented, shuffled), validation and test loaders."""
    train_ds = HipMRISliceDataset(root, "train", augment=True, limit=limit)
    val_ds = HipMRISliceDataset(root, "validate", augment=False, limit=limit)
    test_ds = HipMRISliceDataset(root, "test", augment=False, limit=limit)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


# --------------------------------------------------------------------------- #
# 3D volumes (for the 3D Improved U-Net)                                       #
# --------------------------------------------------------------------------- #
# Design decisions (backed by map_2d_to_3d.py run on the real data):
#
# * Volumes are 256 x 256 x 128 voxels (~1.6 x 1.6 x 1.56 mm), stored in LPS
#   order: axis 0 = left-right, axis 1 = anterior-posterior, axis 2 = superior-
#   inferior. Each 2D scan is 60 consecutive coronal slices (axis 1) of its volume.
# * Splits: the volumes of each split are exactly the 3D counterparts of the 2D
#   split's scans (splits_3d.json, written by map_2d_to_3d.py), so the 3D model
#   is trained, validated and tested on the same patients as the 2D models.
# * Geometry: the anterior-posterior axis is centre-cropped to 128 slices (the
#   prostate lies well inside it; the crop keeps the full left-right and
#   superior-inferior extent) giving a 256 x 128 x 128 input. Like the 2D
#   pipeline this is cropping, not resizing, so labels are never interpolated.
# * Optional downsampling (--downsample 2) averages 2x2x2 blocks of the image and
#   takes the centre voxel of each block of the mask, giving 128 x 64 x 64.
# * Intensity: per-volume z-score normalisation.
# * Augmentation runs on the GPU, batched (VolumeAugment) -- a Slurm job has one
#   CPU core, which is far too slow for 3D resampling.
VOLUME_ROOT = "/home/groups/comp3710/HipMRI_Study_open"
VOLUME_SHAPE = (256, 128, 128)  # (left-right, anterior-posterior crop, superior-inferior)


def fit_volume(arr, target=VOLUME_SHAPE):
    """
    Centre-crop / zero-pad every axis of a 3D array to `target` (no interpolation).
    Returns the fitted array and the placement needed to undo it (see unfit_volume).
    """
    out = np.zeros(target, dtype=arr.dtype)
    placement = []
    for dim, size in enumerate(target):
        n = arr.shape[dim]
        if n >= size:
            start = (n - size) // 2
            placement.append((slice(start, start + size), slice(0, size)))
        else:
            start = (size - n) // 2
            placement.append((slice(0, n), slice(start, start + n)))
    out[tuple(p[1] for p in placement)] = arr[tuple(p[0] for p in placement)]
    return out, placement


def unfit_volume(arr, native_shape, placement, fill=0):
    """Inverse of fit_volume: put a fitted array back into a native-sized one."""
    out = np.full(native_shape, fill, dtype=arr.dtype)
    out[tuple(p[0] for p in placement)] = arr[tuple(p[1] for p in placement)]
    return out


def downsample_volume(image, mask, factor):
    """Average-pool the image and take the centre voxel of each block of the mask."""
    if factor == 1:
        return image, mask
    img = F.avg_pool3d(torch.from_numpy(image.astype(np.float32))[None, None], factor)[0, 0].numpy()
    off = factor // 2
    return img, mask[off::factor, off::factor, off::factor] if mask is not None else None


def load_volume_pair(root, info):
    """Read one MR volume and its label volume (as listed in volumes_3d.json)."""
    def read(path):
        a = np.asarray(nib.load(path).dataobj)
        while a.ndim > 3:
            a = a[..., 0]
        return np.ascontiguousarray(a)  # Nifti arrays are Fortran-ordered; slicing them is very slow
    image = read(os.path.join(root, "semantic_MRs", info["mr_file"])).astype(np.float32)
    label = read(os.path.join(root, "semantic_labels_only", info["label_file"])).astype(np.uint8)
    if image.shape != label.shape:
        raise ValueError(f"MR/label shape mismatch for {info['label_file']}: {image.shape} vs {label.shape}")
    return image, label


def preprocess_volume(image, label, downsample=1):
    """Crop/pad, normalise and (optionally) downsample. Returns image, label, placement."""
    image, placement = fit_volume(image)
    label = fit_volume(label)[0] if label is not None else None
    image = normalise(image)
    image, label = downsample_volume(image, label, downsample)
    return image, label, placement


class VolumeAugment:
    """
    Batched, label-safe 3D augmentation applied on the GPU inside the training loop:
    left-right flip, rotation (+/-10 deg in the axial plane) and isotropic scaling
    (0.9-1.1), and brightness/contrast jitter. The mask is resampled with
    nearest-neighbour, so it can only ever contain existing labels.
    """

    def __init__(self, max_rotate_deg=10.0, scale_range=(0.9, 1.1), p_flip=0.5, intensity_jitter=0.1):
        self.max_rotate = math.radians(max_rotate_deg)
        self.scale_range = scale_range
        self.p_flip = p_flip
        self.intensity_jitter = intensity_jitter

    def __call__(self, images, masks):
        # images (N, 1, LR, AP, SI) float, masks (N, LR, AP, SI) long, on any device
        n, dev = images.shape[0], images.device
        flip = torch.rand(n, device=dev) < self.p_flip
        images = torch.where(flip[:, None, None, None, None], images.flip(2), images)
        masks = torch.where(flip[:, None, None, None], masks.flip(1), masks)

        angle = (torch.rand(n, device=dev) * 2 - 1) * self.max_rotate
        scale = self.scale_range[0] + torch.rand(n, device=dev) * (self.scale_range[1] - self.scale_range[0])
        cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
        # affine_grid orders coordinates (x, y, z) = (SI, AP, LR); rotate the LR-AP (axial) plane.
        theta = torch.zeros(n, 3, 4, device=dev)
        theta[:, 0, 0] = 1 / scale
        theta[:, 1, 1], theta[:, 1, 2] = cos, -sin
        theta[:, 2, 1], theta[:, 2, 2] = sin, cos
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        # "border" pads with edge values (air), not 0 -- after z-scoring, 0 is mid-grey tissue intensity.
        images = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
        masks = F.grid_sample(masks[:, None].float(), grid, mode="nearest", padding_mode="zeros",
                              align_corners=False)[:, 0].long()

        gain = 1 + (torch.rand(n, 1, 1, 1, 1, device=dev) * 2 - 1) * self.intensity_jitter
        bias = (torch.rand(n, 1, 1, 1, 1, device=dev) * 2 - 1) * self.intensity_jitter
        return images * gain + bias, masks


class HipMRIVolumeDataset(Dataset):
    """
    HipMRI 3D volumes of one split, held in memory.

    Returns (image, mask, meta):
        image: (1, 256, 128, 128) float32 (or (1, 128, 64, 64) with downsample=2)
        mask:  matching (D, H, W) int64 class indices in [0, 5]
        meta:  dict with the volume name, e.g. "B040_Week0"
    """

    def __init__(self, split="train", mapping_dir="mapping_out", root=VOLUME_ROOT, downsample=1,
                 limit=None, num_threads=8):
        with open(os.path.join(mapping_dir, "splits_3d.json")) as f:
            names = json.load(f)["splits"][split]
        with open(os.path.join(mapping_dir, "volumes_3d.json")) as f:
            info = json.load(f)
        if limit:
            names = names[:limit]
        if not names:
            raise RuntimeError(f"No 3D volumes listed for split '{split}' in {mapping_dir}/splits_3d.json")

        def load(name):
            image, label = load_volume_pair(root, info[name])
            # Fraction of the prostate that survives the crop (should be 1.0).
            kept = float((fit_volume(label)[0] == PROSTATE).sum() / max((label == PROSTATE).sum(), 1))
            image, label, _ = preprocess_volume(image, label, downsample)
            return image.astype(np.float16), label, kept

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            loaded = list(pool.map(load, names))
        self.images = np.stack([v[0] for v in loaded])
        self.masks = np.stack([v[1] for v in loaded])
        self.meta = [{"volume": n} for n in names]
        self.split = split
        # Sanity check: the crop must not cut away any prostate.
        self.prostate_kept = min(v[2] for v in loaded)
        if self.prostate_kept < 0.999:
            print(f"WARNING: the crop removed prostate voxels in a {split} volume "
                  f"(min kept fraction {self.prostate_kept:.4f})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx].astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))
        return image, mask, self.meta[idx]


def get_volume_dataloaders(mapping_dir="mapping_out", root=VOLUME_ROOT, batch_size=2, num_workers=2,
                           downsample=1, limit=None):
    """Train (shuffled), validation and test loaders over 3D volumes. Augmentation is applied on the GPU."""
    datasets = [HipMRIVolumeDataset(s, mapping_dir, root, downsample, limit) for s in ["train", "validate", "test"]]
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(datasets[0], shuffle=True, drop_last=True, **loader_kwargs),
        DataLoader(datasets[1], shuffle=False, **loader_kwargs),
        DataLoader(datasets[2], shuffle=False, **loader_kwargs),
    )


if __name__ == "__main__":
    # Quick self-check: python dataset.py [--root PATH]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    ds = HipMRISliceDataset(args.root, "train", augment=True, limit=args.limit)
    img, mask, meta = ds[0]
    print(f"{len(ds)} slices loaded | image {tuple(img.shape)} {img.dtype} | mask {tuple(mask.shape)} {mask.dtype}")
    print(f"image mean={img.mean():.3f} std={img.std():.3f} | labels present={torch.unique(mask).tolist()}")
    print("meta:", meta)
