"""
dataset.py — Data loading for HipMRI 2D prostate MRI slices (keras_slices_data)

Purpose
-------
Loads the HipMRI 2D slices, scales them into the range a GAN generator
produces ([-1, 1]), resizes them to 128 x 64 (half resolution, keeping the 2:1 
aspect ratio)

Data splitting (leakage prevention)
-----------------------------------
The provided train/validate/test folders are already split at PATIENT level.
This was audited before writing this file:
    train    : 11,460 slices, 31 patients
    validate :    660 slices,  4 patients
    test     :    540 slices,  3 patients
    patient overlap between any two splits: none

How the splits are used in this project:
    * train         -> the ONLY data the GANs ever see during training
    * test          -> unseen real patients, used as a reference in the
                       memorisation audit (are synthetic images closer to the
                       training set than real unseen patients are?)
    * validate      -> not used (no model selection for an unconditional GAN)

No data augmentation is applied: for an unconditional GAN, flips/rotations
would be learned as part of the data distribution (see README §4).

Usage
-----
    python dataset.py --root ~/keras_slices_data
Runs a shape/patient audit, a scaling smoke test, and saves data_preview.png.
"""
import argparse
import glob
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# CONFIGURATION
# =============================================================================

# Maps our split names to the folder names inside keras_slices_data/.
# We deliberately ignore the keras_slices_seg_* folders: they hold
# segmentation masks, which an unconditional GAN does not need.
SPLIT_DIRS = {
    "train": "keras_slices_train",
    "validate": "keras_slices_validate",
    "test": "keras_slices_test",
}

# Filenames look like: case_004_week_0_slice_10.nii.gz
#   group 1 -> patient ID (3 digits)
#   group 2 -> scan week (same patient scanned at several timepoints)
#   group 3 -> slice index within that scan
NAME_RE = re.compile(r"case_(\d{3})_week_(\d+)_slice_(\d+)")

# Native slice shape for almost all files (one train scan is 256x144 and is
# centre-cropped to this width). Images are 2:1, so we keep that aspect ratio.
NATIVE_WIDTH = 128
DEFAULT_SIZE = (128, 64)   # (height, width) = half native resolution

# =============================================================================
# FILE DISCOVERY AND FILENAME PARSING
# =============================================================================

def list_slices(root, split):
    """Return a sorted list of NIfTI file paths for one split.

    Sorting makes the file order deterministic across machines, which keeps
    any max_files subset (and therefore our smoke tests) reproducible.
    """
    files = sorted(glob.glob(os.path.join(root, SPLIT_DIRS[split], "*.nii*")))
    if not files:
        # Fail loudly: a wrong --root path would otherwise give an empty dataset
        raise FileNotFoundError(f"No slices found for '{split}' under {root}")
    return files


def parse_name(path):
    """Extract (patient_id, week, slice_idx) from a slice's filename.

    Patient IDs are what make patient-level leakage checks possible.
    """
    m = NAME_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"Unexpected filename format: {path}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# =============================================================================
# LOADING AND PRE-PROCESSING A SINGLE SLICE
# =============================================================================

def load_slice(path):
    """Load one NIfTI slice from disk as a 2D float32 array."""
    # caching="unchanged" avoids nibabel keeping an extra cached copy in memory
    img = nib.load(path).get_fdata(caching="unchanged")

    # Some HipMRI files are stored as (H, W, 1); drop the trailing singleton dim
    if img.ndim == 3:
        img = img[:, :, 0]

    # Centre-crop wider slices (256x144) to the common 256x128 so every
    # slice has the same pixel scale before resizing
    w = img.shape[1]
    if w > NATIVE_WIDTH:
        start = (w - NATIVE_WIDTH) // 2
        img = img[:, start:start + NATIVE_WIDTH]

    return img.astype(np.float32)


def to_gan_range(img):
    """Min-max scale a single slice to [-1, 1].

    Why [-1, 1]: the generator will end in tanh, whose output range is
    [-1, 1]. Real images must share that range, otherwise the discriminator
    could separate real from fake by intensity alone.

    Why per-slice: MRI intensities have no fixed physical scale between scans.
    Limitation: absolute brightness differences between scans are discarded.

    The 1e-8 term prevents division by zero on blank (constant) slices.
    """
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8) * 2.0 - 1.0


# =============================================================================
# LOADING A WHOLE SPLIT (WITH CACHING)
# =============================================================================

def load_split(root, split, image_size=DEFAULT_SIZE, max_files=None, cache_dir="data/cache"):
    """Load an entire split as a tensor of shape [N, 1, H, W], where image_size = (H, W).

    Args:
        root:       path to the keras_slices_data folder
        split:      "train", "validate" or "test"
        image_size: (height, width) in pixels
        max_files:  load only the first N files (for quick smoke tests)
        cache_dir:  where the fully loaded tensor is saved (gitignored)

    Returns:
        dict with
            "images":      float32 tensor [N, 1, H, W] in [-1, 1]
            "patient_ids": int tensor [N], one patient ID per slice

    Reading ~11k separate files is slow, so a full load is saved once to a
    single .pt file and reused afterwards (it can also be copied to another 
    machine instead of the raw NIfTI folders)
    """
    cache = os.path.join(cache_dir, f"{split}_{image_size[0]}x{image_size[1]}.pt")

    # Reuse the cached tensor, but only for full loads (never for smoke tests)
    if max_files is None and os.path.exists(cache):
        return torch.load(cache)

    files = list_slices(root, split)[:max_files]
    images, patient_ids = [], []

    for f in files:
        # 1. read + scale the slice, then add batch and channel dims -> [1, 1, H, W]
        x = torch.from_numpy(to_gan_range(load_slice(f)))[None, None]

        # 2. resize to image_size (H, W); bilinear interpolation of values
        #    in [-1, 1] stays within [-1, 1]
        x = F.interpolate(x, size=image_size,
                          mode="bilinear", align_corners=False)

        images.append(x[0])                    # back to [1, H, W]
        patient_ids.append(parse_name(f)[0])   # keep the patient ID per slice

    out = {
        "images": torch.stack(images),             # [N, 1, H, W]
        "patient_ids": torch.tensor(patient_ids),  # [N]
    }

    # Only cache complete splits, so a partial smoke-test load never
    # masquerades as the full dataset later
    if max_files is None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(out, cache)

    return out


# =============================================================================
# DATALOADER
# =============================================================================

def get_loader(images, batch_size=64, shuffle=True):
    """Wrap an image tensor in a DataLoader for training.

    drop_last=True when shuffling keeps every training batch the same size,
    which avoids a tiny final batch destabilising BatchNorm in the GAN.
    """
    return DataLoader(TensorDataset(images), batch_size=batch_size,
                      shuffle=shuffle, drop_last=shuffle)


# =============================================================================
# SELF-TEST: DATA AUDIT + SMOKE TEST + VISUAL PREVIEW
# =============================================================================

if __name__ == "__main__":
    from collections import Counter
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Audit and preview HipMRI 2D slices")
    parser.add_argument("--root", default=os.path.expanduser("~/keras_slices_data"),
                        help="path to the keras_slices_data folder")
    args = parser.parse_args()

    # --- 1. Audit: counts per split, read from headers only (fast) ----------
    patients_per_split = {}
    for split in SPLIT_DIRS:
        files = list_slices(args.root, split)
        ids = [parse_name(f) for f in files]
        shapes = Counter(nib.load(f).shape for f in files)   # header read only
        patients_per_split[split] = {i[0] for i in ids}
        print(f"{split}: {len(files)} slices, "
              f"{len(patients_per_split[split])} patients, "
              f"{len({(i[0], i[1]) for i in ids})} scans, "
              f"shapes {dict(shapes)}")

    # --- 2. Leakage check: no patient may appear in two splits --------------
    splits = list(SPLIT_DIRS)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = patients_per_split[splits[i]] & patients_per_split[splits[j]]
            assert not overlap, f"Patient leakage {splits[i]}/{splits[j]}: {overlap}"
    print("Leakage check passed: no patient appears in more than one split.")

    # --- 3. Smoke test: load a few slices and check shape/range -------------
    sample = load_split(args.root, "train", max_files=16)
    x = sample["images"]
    print("sample batch:", tuple(x.shape), x.dtype,
          f"range [{x.min():.2f}, {x.max():.2f}]")
    assert x.min() >= -1.0001 and x.max() <= 1.0001, "scaling out of range"

    # --- 4. Visual check: do the slices look like pelvic MRI? ---------------
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for ax, img in zip(axes.flat, x):
        ax.imshow(img[0], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("data_preview.png", dpi=100)
    print("Saved data_preview.png")