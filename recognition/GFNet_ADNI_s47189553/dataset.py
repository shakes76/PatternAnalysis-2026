"""
dataset.py — ADNI (AD vs NC) data loading for GFNet.

Labels: NC = 0, AD = 1.

python dataset.py --root "/Users/ryanhuang/Documents/UQ/Sem 2 2026/COMP3710/ADNI/AD_NC"
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CLASSES = ("NC", "AD")                      # index = label
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)} # {"NC": 0, "AD": 1}
IMG_EXTS = {".jpeg", ".jpg", ".png"}

DEFAULT_ROOT = "/home/groups/comp3710/ADNI/AD_NC"

# Placeholder grayscale stats — replace with the output of
# `python dataset.py --root ... --stats` on your training split.
DEFAULT_MEAN = 0.1155
DEFAULT_STD = 0.2224


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def patient_id(path: str | Path) -> str:
    """'388206_78.jpeg' -> '388206'."""
    return Path(path).stem.split("_")[0]


def scan_split(root: str | Path, split: str) -> list[tuple[str, int]]:
    """Return sorted (path, label) pairs for one split ('train' or 'test')."""
    samples = []
    for cls in CLASSES:
        cls_dir = Path(root) / split / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {cls_dir}")
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((str(f), CLASS_TO_IDX[cls]))
    if not samples:
        raise RuntimeError(f"No images found under {Path(root) / split}")
    return samples


def patient_level_split(
    samples: list[tuple[str, int]], val_frac: float = 0.15, seed: int = 42
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Split samples into train/val so no patient appears in both.
    Stratified by class so AD/NC proportions are preserved."""
    by_class_patient: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path, label in samples:
        by_class_patient[label][patient_id(path)].append((path, label))

    rng = random.Random(seed)
    train, val = [], []
    for label, patients in by_class_patient.items():
        ids = sorted(patients)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_frac))
        for i, pid in enumerate(ids):
            (val if i < n_val else train).extend(patients[pid])
    return train, val


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ADNIDataset(Dataset):
    """Slice-level ADNI dataset. Returns (image_tensor, label).

    `self.patients[i]` gives the patient ID for sample i, which predict.py can
    use to aggregate slice predictions into a per-patient vote.
    """

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform
        self.patients = [patient_id(p) for p, _ in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")      # MRI slices are grayscale
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict[str, int]:
        counts = {c: 0 for c in CLASSES}
        for _, y in self.samples:
            counts[CLASSES[y]] += 1
        return counts


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(
    train: bool,
    img_size: int = 224,
    in_chans: int = 1,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
):
    """Train: light geometric + intensity augmentation. Eval: deterministic.
    in_chans=3 repeats the grey channel (for ImageNet-pretrained GFNet weights)."""
    ops: list = []
    if train:
        ops += [
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    else:
        ops += [transforms.Resize((img_size, img_size))]

    ops += [transforms.ToTensor(), transforms.Normalize([mean], [std])]
    if in_chans == 3:
        ops.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    return transforms.Compose(ops)


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def get_dataloaders(
    root: str = DEFAULT_ROOT,
    batch_size: int = 64,
    img_size: int = 224,
    in_chans: int = 1,
    val_frac: float = 0.15,
    num_workers: int | None = None,
    seed: int = 42,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 0)
    pin = torch.cuda.is_available()             # pin_memory is a no-op/warning on MPS

    train_samples, val_samples = patient_level_split(
        scan_split(root, "train"), val_frac=val_frac, seed=seed
    )
    test_samples = scan_split(root, "test")

    tf_train = get_transforms(True, img_size, in_chans, mean, std)
    tf_eval = get_transforms(False, img_size, in_chans, mean, std)

    train_ds = ADNIDataset(train_samples, tf_train)
    val_ds = ADNIDataset(val_samples, tf_eval)
    test_ds = ADNIDataset(test_samples, tf_eval)

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def compute_mean_std(root: str = DEFAULT_ROOT, img_size: int = 224) -> tuple[float, float]:
    """Pixel mean/std over the training split (after resize, before normalisation)."""
    ds = ADNIDataset(
        scan_split(root, "train"),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()]),
    )
    loader = DataLoader(ds, batch_size=128, num_workers=min(8, os.cpu_count() or 0))
    total, s, sq = 0, 0.0, 0.0
    for x, _ in loader:
        total += x.numel()
        s += x.sum().item()
        sq += (x ** 2).sum().item()
    mean = s / total
    std = (sq / total - mean ** 2) ** 0.5
    return mean, std


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ADNI dataset sanity check")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--stats", action="store_true", help="compute train mean/std")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    if args.stats:
        m, s = compute_mean_std(args.root)
        print(f"train mean={m:.4f}  std={s:.4f}")

    tr, va, te = get_dataloaders(args.root, batch_size=args.batch_size, num_workers=0)
    for name, dl in (("train", tr), ("val", va), ("test", te)):
        ds = dl.dataset
        print(f"{name:5s}: {len(ds):6d} slices, {len(set(ds.patients)):4d} patients, {ds.class_counts()}")

    overlap = set(tr.dataset.patients) & set(va.dataset.patients)
    print("train/val patient overlap:", len(overlap))   # should be 0

    x, y = next(iter(tr))
    print("batch:", tuple(x.shape), x.dtype, "labels:", y[:8].tolist(),
          f"range=[{x.min():.2f}, {x.max():.2f}]")