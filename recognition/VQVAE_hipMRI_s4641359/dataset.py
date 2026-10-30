# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Wrote the Nifti loader (following Appendix B of the project spec),
#   per-slice min-max scaling to [0, 1], resizing to 256x128, and the
#   PyTorch Dataset and DataLoaders for the provided splits.
# - Wrote the data audit (slice counts, shapes, intensity range) and the
#   patient-overlap check based on case IDs in the filenames.
# - Added an optional intensity-jitter augmentation (not used in the
#   final runs).
#
# What I did:
# - Audited and verified the code
# - Copied the HipMRI data from Rangpur and ran the audit on it
#   (11,460 / 660 / 540 slices; 31 / 4 / 3 patients; no overlap).
# - Decided to keep the provided patient-level split and not to use
#   flips or rotations, which would create anatomically wrong slices.
# - Noticed that patient 040 makes up most of the test set and used the
#   audit results in my feasibility review and README.


import argparse
import os
import re
from collections import defaultdict
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Folder names inside keras_slices_data for each split (image slices only;
# the matching "seg" folders hold segmentation masks, which we don't need).
SPLIT_FOLDERS = {
    "train": "keras_slices_train",
    "validate": "keras_slices_validate",
    "test": "keras_slices_test",
}

# Every slice is resized to this (height, width) so batches stack cleanly.
DEFAULT_SHAPE = (256, 128)

# Load one Nifti slice as a 2D NumPy array
def load_slice(path):

    image = nib.load(path).get_fdata(caching="unchanged")
    if image.ndim == 3:  # some HipMRI slices carry an extra trailing dim
        image = image[:, :, 0]
    return image.astype(np.float32)

# Min-max scale a slice to [0, 1]; constant slices become all zeros.
def normalise(image):

    low, high = image.min(), image.max()
    if high - low < 1e-8:
        return np.zeros_like(image)
    return (image - low) / (high - low)

# Extract the patient/case ID from a filename
# Falls back to the filename stem if no case number is found, so the
# overlap check still runs (it will just be per-file).
def patient_id(path):

    name = os.path.basename(path)
    match = re.search(r"case_?(\d+)", name, flags=re.IGNORECASE)
    return match.group(1) if match else name.split(".")[0]

# Return sorted Nifti paths for one split ('train', 'validate', 'test').
def list_split_files(data_root, split):

    folder = os.path.join(data_root, SPLIT_FOLDERS[split])
    files = sorted(glob(os.path.join(folder, "*.nii*")))
    if not files:
        raise FileNotFoundError(f"No Nifti files found in {folder}")
    return files

# In-memory dataset of normalised, resized HipMRI 2D slices.
# Each item is a float tensor of shape (1, H, W) with values in [0, 1].
class HipMRIDataset(Dataset):

    def __init__(self, files, shape=DEFAULT_SHAPE, augment=False):
        self.files = files
        self.augment = augment
        slices = []
        for path in tqdm(files, desc="Loading slices", leave=False):
            image = torch.from_numpy(normalise(load_slice(path)))
            image = image[None, None]  # (1, 1, H, W) for interpolate
            if tuple(image.shape[-2:]) != tuple(shape):
                image = F.interpolate(image, size=shape, mode="bilinear",
                                      align_corners=False)
            slices.append(image[0])  # back to (1, H, W)
        self.images = torch.stack(slices)  # (N, 1, H, W)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        # Light augmentation only: small random intensity jitter.
        # No flips, so generated anatomy keeps its real left/right layout.
        if self.augment:
            image = (image * (1 + 0.05 * (torch.rand(1) - 0.5))).clamp(0, 1)
        return image

# Build train / validate / test DataLoaders from the provided splits.
def get_dataloaders(data_root, batch_size=32, shape=DEFAULT_SHAPE,
                    num_workers=0, augment_train=False):

    loaders = {}
    for split in SPLIT_FOLDERS:
        dataset = HipMRIDataset(list_split_files(data_root, split), shape,
                                augment=(augment_train and split == "train"))
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=num_workers,
                                    pin_memory=torch.cuda.is_available())
    return loaders

# Report any patient IDs that appear in more than one split.
def check_patient_overlap(data_root):

    split_ids = {split: {patient_id(p) for p in list_split_files(data_root, split)}
                 for split in SPLIT_FOLDERS}
    owners = defaultdict(list)
    for split, ids in split_ids.items():
        for pid in ids:
            owners[pid].append(split)
    leaks = {pid: splits for pid, splits in owners.items() if len(splits) > 1}
    return split_ids, leaks

# Print the data audit used in the feasibility review.
def audit(data_root):

    print(f"Data root: {data_root}\n")
    for split in SPLIT_FOLDERS:
        files = list_split_files(data_root, split)
        shapes = defaultdict(int)
        low, high = float("inf"), float("-inf")
        for path in tqdm(files, desc=f"Auditing {split}", leave=False):
            image = load_slice(path)
            shapes[image.shape] += 1
            low, high = min(low, image.min()), max(high, image.max())
        print(f"[{split}] {len(files)} slices")
        print(f"  shapes: {dict(shapes)}")
        print(f"  raw intensity range: {low:.1f} to {high:.1f}")
        print(f"  example file: {os.path.basename(files[0])}")

    split_ids, leaks = check_patient_overlap(data_root)
    print("\nPatients per split:",
          {split: len(ids) for split, ids in split_ids.items()})
    if leaks:
        print(f"WARNING: {len(leaks)} patient IDs appear in multiple splits:")
        for pid, splits in sorted(leaks.items())[:10]:
            print(f"  {pid}: {splits}")
    else:
        print("No patient overlap across splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit HipMRI 2D slices.")
    parser.add_argument("--data_root", required=True,
                        help="Folder containing keras_slices_train/validate/test")
    args = parser.parse_args()
    audit(args.data_root)