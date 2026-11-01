"""dataset.py: OASIS 2D brain MRI data loading for segmentation.

The OASIS data arrives pre-split into train / validate / test folders, with one
PNG per 2D slice and a matching mask PNG:

    keras_png_slices_train/case_001_slice_0.nii.png
    keras_png_slices_seg_train/seg_001_slice_0.nii.png

Masks hold the grey values 0, 85, 170, 255. Dividing by 85 gives the class
indices 0, 1, 2, 3.

Usage check (prints split sizes, runs the leakage check, loads one batch):
    python dataset.py --root ~/oasis_data
"""

import argparse
import random
import re
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

NUM_CLASSES = 4
MASK_STEP = 85  # mask grey levels 0, 85, 170, 255 -> classes 0, 1, 2, 3

# split name -> (image folder, mask folder)
SPLIT_FOLDERS = {
    "train": ("keras_png_slices_train", "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test": ("keras_png_slices_test", "keras_png_slices_seg_test"),
}

_NAME_RE = re.compile(r"case_(\d+)_slice_(\d+)")

def list_pairs(root, split):
    """Return a sorted list of (image_path, mask_path) for one split."""
    img_folder, seg_folder = SPLIT_FOLDERS[split]
    img_dir = Path(root).expanduser() / img_folder
    seg_dir = Path(root).expanduser() / seg_folder

    pairs = []
    for img_path in sorted(img_dir.glob("case_*.png")):
        if _NAME_RE.match(img_path.name) is None:
            continue
        # case_001_slice_0.nii.png -> seg_001_slice_0.nii.png
        seg_path = seg_dir / img_path.name.replace("case_", "seg_", 1)
        if not seg_path.exists():
            raise FileNotFoundError(f"No mask for {img_path.name}: expected {seg_path}")
        pairs.append((img_path, seg_path))

    if not pairs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return pairs

def case_ids(root, split):
    """Set of case (patient) IDs present in a split."""
    return {_NAME_RE.match(img.name).group(1) for img, _ in list_pairs(root, split)}


def assert_no_case_overlap(root):
    """Raise an error if any case appears in more than one split.

    The data is already split, so this is the leakage check: all slices of a
    patient must stay in one split. Returns the number of cases per split.
    """
    ids = {split: case_ids(root, split) for split in SPLIT_FOLDERS}
    for a, b in combinations(ids, 2):
        shared = ids[a] & ids[b]
        if shared:
            raise ValueError(
                f"Case leakage between {a} and {b}: "
                f"{len(shared)} shared cases, e.g. {sorted(shared)[:5]}"
            )
    return {split: len(case_set) for split, case_set in ids.items()}

def load_image(path):
    """Greyscale image as a (H, W) uint8 array."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def load_mask(path):
    """Mask as a (H, W) uint8 array of class indices 0..NUM_CLASSES-1."""
    raw = np.array(Image.open(path).convert("L"), dtype=np.int16)
    classes = np.rint(raw / MASK_STEP).astype(np.int16)
    if classes.max() >= NUM_CLASSES or not np.array_equal(classes * MASK_STEP, raw):
        raise ValueError(
            f"Unexpected mask values in {path}: {np.unique(raw).tolist()}"
        )
    return classes.astype(np.uint8)

def to_one_hot(mask, num_classes=NUM_CLASSES):
    """(H, W) long class indices -> (C, H, W) float one-hot."""
    return torch.nn.functional.one_hot(mask, num_classes).permute(2, 0, 1).float()

class OASISDataset(Dataset):
    """One split of the OASIS 2D slices.

    Each item is (image, mask):
        image: float32 (1, H, W), scaled to [0, 1]
        mask:  float32 (NUM_CLASSES, H, W) one-hot if one_hot=True,
               otherwise int64 (H, W) class indices
    """

    def __init__(self, root, split, augment=False, one_hot=True):
        self.pairs = list_pairs(root, split)
        self.augment = augment
        self.one_hot = one_hot

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _augment(image, mask):
        """Simple paired augmentation; the same transform hits image and mask."""
        if random.random() < 0.5:  # horizontal flip
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.5:  # small rotation; nearest-neighbour keeps mask labels intact
            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        return image, mask

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        image = torch.from_numpy(load_image(img_path)).unsqueeze(0)  # (1, H, W) uint8
        mask = torch.from_numpy(load_mask(seg_path)).unsqueeze(0)  # (1, H, W) uint8
        if self.augment:
            image, mask = self._augment(image, mask)
        image = image.float() / 255.0
        mask = mask.squeeze(0).long()  # (H, W)
        if self.one_hot:
            mask = to_one_hot(mask)
        return image, mask

def get_dataloaders(root, batch_size=8, num_workers=0, augment=True, one_hot=True):
    """Return (train_loader, val_loader, test_loader).

    Only the training split is augmented and shuffled. Raises an error first if
    any patient appears in more than one split.
    """
    assert_no_case_overlap(root)
    pin = torch.cuda.is_available()

    train_set = OASISDataset(root, "train", augment=augment, one_hot=one_hot)
    val_set = OASISDataset(root, "validate", augment=False, one_hot=one_hot)
    test_set = OASISDataset(root, "test", augment=False, one_hot=one_hot)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the OASIS data pipeline.")
    parser.add_argument("--root", required=True,
                        help="folder holding the six keras_png_slices_* folders")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("Cases per split (no overlap found):", assert_no_case_overlap(args.root))

    loaders = get_dataloaders(args.root, batch_size=args.batch_size)
    for name, loader in zip(("train", "validate", "test"), loaders):
        print(f"{name}: {len(loader.dataset)} slices, {len(loader)} batches")

    images, masks = next(iter(loaders[0]))
    print("image batch:", tuple(images.shape), images.dtype,
          f"min={images.min():.3f} max={images.max():.3f}")
    print("mask batch: ", tuple(masks.shape), masks.dtype)
    print("classes present in batch:", torch.unique(masks.argmax(dim=1)).tolist())