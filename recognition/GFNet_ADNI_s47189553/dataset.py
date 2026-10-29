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
 
# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CLASSES = ("NC", "AD")                               # index = label
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}  # {"NC": 0, "AD": 1}
IMG_EXTS = {".jpeg"}
 
DEFAULT_ROOT = "/Users/ryanhuang/Documents/UQ/Sem 2 2026/COMP3710/ADNI/AD_NC"
# DEFAULT_ROOT = "/home/groups/comp3710/ADNI/AD_NC"
 
# Dataset split & channel configurations
DEFAULT_VAL_FRAC = 0.15
DEFAULT_SEED = 42
DEFAULT_N_FOLDS = 5
MIN_N_FOLDS = 2
DEFAULT_IMG_SIZE = 224
DEFAULT_IN_CHANS = 1
RGB_CHANNEL_COUNT = 3
MIN_PATIENTS_IN_VAL = 1
MAX_DATALOADER_WORKERS = 8
 
# Baseline grayscale normalization statistics
DEFAULT_MEAN = 0.1155
DEFAULT_STD = 0.2224
 
# Data augmentation parameters
AUG_CROP_FLIP = True            # False = no RandomResizedCrop / RandomHorizontalFlip
AUG_SCALE_RANGE = (0.85, 1.0)
AUG_RATIO_RANGE = (0.95, 1.05)
AUG_AFFINE_DEGREES = 10
AUG_TRANSLATE_RANGE = (0.05, 0.05)
AUG_BRIGHTNESS_JITTER = 0.1
AUG_CONTRAST_JITTER = 0.1
 
# Batch size defaults for utilities & verification
DEFAULT_BATCH_SIZE = 64
CLI_DEFAULT_BATCH_SIZE = 32
STATS_COMPUTE_BATCH_SIZE = 128
 
Sample = tuple[str, int]
 
 
# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def patient_id(path: str | Path) -> str:
    """'388206_78.jpeg' -> '388206'."""
    return Path(path).stem.split("_")[0]
 
 
def scan_split(root: str | Path, split: str) -> list[Sample]:
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
 
 
def _group_by_class_patient(
    samples: list[Sample],
) -> dict[int, dict[str, list[Sample]]]:
    """{label: {patient_id: [samples...]}}"""
    grouped: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for path, label in samples:
        grouped[label][patient_id(path)].append((path, label))
    return grouped
 
 
def patient_level_split(
    samples: list[Sample],
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into train/val so no patient appears in both.
    Stratified by class so AD/NC proportions are preserved."""
    by_class_patient = _group_by_class_patient(samples)
 
    rng = random.Random(seed)
    train, val = [], []
    for label, patients in by_class_patient.items():
        ids = sorted(patients)
        rng.shuffle(ids)
        n_val = max(MIN_PATIENTS_IN_VAL, round(len(ids) * val_frac))
        for i, pid in enumerate(ids):
            (val if i < n_val else train).extend(patients[pid])
    return train, val
 
 
def subsample_patients(
    samples: list[Sample], frac: float, seed: int = DEFAULT_SEED
) -> list[Sample]:
    """Keep ~frac of the patients in each class (all their slices), for quick runs.
    frac >= 1 returns samples unchanged."""
    if frac >= 1.0:
        return samples
    if frac <= 0.0:
        raise ValueError(f"subset fraction must be in (0, 1], got {frac}")
    rng = random.Random(seed)
    kept: list[Sample] = []
    for label in sorted(by := _group_by_class_patient(samples)):
        ids = sorted(by[label])
        rng.shuffle(ids)
        for pid in ids[: max(MIN_PATIENTS_IN_VAL, round(len(ids) * frac))]:
            kept.extend(by[label][pid])
    return kept
 
 
def patient_level_kfold(
    samples: list[Sample],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = DEFAULT_SEED,
) -> list[tuple[list[Sample], list[Sample]]]:
    """Stratified, patient-grouped K-fold.
 
    Returns a list of n_folds (train, val) pairs. Guarantees:
      * every patient's slices are in exactly one val fold (never split),
      * each fold gets ~1/n_folds of the NC patients and ~1/n_folds of the AD
        patients, so class balance matches the full training set,
      * deterministic for a given seed.
    """
    if n_folds < MIN_N_FOLDS:
        raise ValueError(f"n_folds must be >= {MIN_N_FOLDS}, got {n_folds}")
 
    by_class_patient = _group_by_class_patient(samples)
    rng = random.Random(seed)
    folds: list[list[Sample]] = [[] for _ in range(n_folds)]
 
    # Round-robin shuffled patients into folds, class by class. The running
    # offset spreads the "remainder" patients across folds instead of always
    # dumping them in fold 0.
    offset = 0
    for label in sorted(by_class_patient):
        patients = by_class_patient[label]
        ids = sorted(patients)
        if len(ids) < n_folds:
            raise ValueError(
                f"Class {CLASSES[label]} has {len(ids)} patients; "
                f"need at least {n_folds} for {n_folds}-fold CV"
            )
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            folds[(i + offset) % n_folds].extend(patients[pid])
        offset += len(ids)
 
    splits = []
    for k in range(n_folds):
        val = folds[k]
        train = [s for j, fold in enumerate(folds) if j != k for s in fold]
        splits.append((train, val))
    return splits
 
 
# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ADNIDataset(Dataset):
    """Slice-level ADNI dataset. Returns (image_tensor, label).
 
    `self.patients[i]` gives the patient ID for sample i, which predict.py can
    use to aggregate slice predictions into a per-patient vote.
    """
 
    def __init__(self, samples: list[Sample], transform=None):
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
    img_size: int = DEFAULT_IMG_SIZE,
    in_chans: int = DEFAULT_IN_CHANS,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
    crop_flip: bool = AUG_CROP_FLIP,
):
    """Train: light geometric + intensity augmentation. Eval: deterministic.
    crop_flip=False removes RandomResizedCrop and RandomHorizontalFlip, so
    training sees the same framing as evaluation (affine + jitter still apply).
    in_chans=3 repeats the grey channel (for ImageNet-pretrained GFNet weights)."""
    ops: list = []
    if train:
        if crop_flip:
            ops += [
                transforms.RandomResizedCrop(
                    img_size, scale=AUG_SCALE_RANGE, ratio=AUG_RATIO_RANGE
                ),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            ops += [transforms.Resize((img_size, img_size))]
        ops += [
            transforms.RandomAffine(
                degrees=AUG_AFFINE_DEGREES, translate=AUG_TRANSLATE_RANGE
            ),
            transforms.ColorJitter(
                brightness=AUG_BRIGHTNESS_JITTER, contrast=AUG_CONTRAST_JITTER
            ),
        ]
    else:
        ops += [transforms.Resize((img_size, img_size))]
 
    ops += [transforms.ToTensor(), transforms.Normalize([mean], [std])]
    if in_chans == RGB_CHANNEL_COUNT:
        ops.append(transforms.Lambda(lambda x: x.repeat(RGB_CHANNEL_COUNT, 1, 1)))
    return transforms.Compose(ops)
 
 
# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def _build_loaders(
    train_samples: list[Sample],
    val_samples: list[Sample],
    test_samples: list[Sample],
    batch_size: int,
    img_size: int,
    in_chans: int,
    num_workers: int | None,
    mean: float,
    std: float,
    subset: float = 1.0,
    seed: int = DEFAULT_SEED,
    crop_flip: bool = AUG_CROP_FLIP,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_samples = subsample_patients(train_samples, subset, seed)
    val_samples = subsample_patients(val_samples, subset, seed)
    test_samples = subsample_patients(test_samples, subset, seed)
    if num_workers is None:
        num_workers = min(MAX_DATALOADER_WORKERS, os.cpu_count() or 0)
    pin = torch.cuda.is_available()             # pin_memory is a no-op/warning on MPS
 
    tf_train = get_transforms(True, img_size, in_chans, mean, std, crop_flip)
    tf_eval = get_transforms(False, img_size, in_chans, mean, std)
 
    train_ds = ADNIDataset(train_samples, tf_train)
    val_ds = ADNIDataset(val_samples, tf_eval)
    test_ds = ADNIDataset(test_samples, tf_eval)
 
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
 
 
def get_dataloaders(
    root: str = DEFAULT_ROOT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    in_chans: int = DEFAULT_IN_CHANS,
    val_frac: float = DEFAULT_VAL_FRAC,
    num_workers: int | None = None,
    seed: int = DEFAULT_SEED,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
    subset: float = 1.0,
    crop_flip: bool = AUG_CROP_FLIP,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Single hold-out split. Return (train_loader, val_loader, test_loader).
    `subset` < 1 keeps that fraction of patients in every split (quick runs)."""
    train_samples, val_samples = patient_level_split(
        scan_split(root, "train"), val_frac=val_frac, seed=seed
    )
    test_samples = scan_split(root, "test")
    return _build_loaders(
        train_samples, val_samples, test_samples,
        batch_size, img_size, in_chans, num_workers, mean, std, subset, seed, crop_flip,
    )
 
 
def get_fold_dataloaders(
    fold: int,
    root: str = DEFAULT_ROOT,
    n_folds: int = DEFAULT_N_FOLDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    img_size: int = DEFAULT_IMG_SIZE,
    in_chans: int = DEFAULT_IN_CHANS,
    num_workers: int | None = None,
    seed: int = DEFAULT_SEED,
    mean: float = DEFAULT_MEAN,
    std: float = DEFAULT_STD,
    subset: float = 1.0,
    crop_flip: bool = AUG_CROP_FLIP,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Loaders for one CV fold (0-indexed). Return (train, val, test).
 
    The test split is never touched by CV — it stays the same held-out set
    for every fold. Use the same `seed` across folds so the partitions are
    consistent.
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")
    splits = patient_level_kfold(scan_split(root, "train"), n_folds, seed)
    train_samples, val_samples = splits[fold]
    test_samples = scan_split(root, "test")
    return _build_loaders(
        train_samples, val_samples, test_samples,
        batch_size, img_size, in_chans, num_workers, mean, std, subset, seed, crop_flip,
    )
 
 
def compute_mean_std(
    root: str = DEFAULT_ROOT,
    img_size: int = DEFAULT_IMG_SIZE,
    samples: list[Sample] | None = None,
) -> tuple[float, float]:
    """Pixel mean/std (after resize, before normalisation).
    Defaults to the full training split; pass `samples` (e.g. one fold's
    training samples) to compute fold-specific statistics."""
    ds = ADNIDataset(
        samples if samples is not None else scan_split(root, "train"),
        transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        ),
    )
    loader = DataLoader(
        ds,
        batch_size=STATS_COMPUTE_BATCH_SIZE,
        num_workers=min(MAX_DATALOADER_WORKERS, os.cpu_count() or 0),
    )
    total, s, sq = 0, 0.0, 0.0
    for x, _ in loader:
        total += x.numel()
        s += x.sum().item()
        sq += (x ** 2).sum().item()
    mean = s / total
    std = (sq / total - mean ** 2) ** 0.5
    return mean, std
 
 
# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def _describe(name: str, ds: ADNIDataset) -> None:
    print(
        f"{name:7s}: {len(ds):6d} slices, {len(set(ds.patients)):4d} patients, "
        f"{ds.class_counts()}"
    )
 
 
def _check_cv(root: str, n_folds: int, seed: int) -> None:
    train_all = scan_split(root, "train")
    all_patients = {patient_id(p) for p, _ in train_all}
    test_patients = {patient_id(p) for p, _ in scan_split(root, "test")}
    splits = patient_level_kfold(train_all, n_folds, seed)
 
    seen_in_val: list[str] = []
    for k, (tr, va) in enumerate(splits):
        tr_ds, va_ds = ADNIDataset(tr), ADNIDataset(va)
        print(f"--- fold {k} ---")
        _describe("train", tr_ds)
        _describe("val", va_ds)
        tr_p, va_p = set(tr_ds.patients), set(va_ds.patients)
        assert not tr_p & va_p, f"fold {k}: train/val patient overlap"
        assert len(tr) + len(va) == len(train_all), f"fold {k}: samples lost"
        seen_in_val.extend(va_p)
 
    assert len(seen_in_val) == len(set(seen_in_val)), "patient in >1 val fold"
    assert set(seen_in_val) == all_patients, "some patients never validated"
    print(f"\nOK: every one of {len(all_patients)} patients is in exactly one val fold")
    print("train/test patient overlap:", len(all_patients & test_patients))
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ADNI dataset sanity check")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--stats", action="store_true", help="compute train mean/std")
    ap.add_argument("--batch-size", type=int, default=CLI_DEFAULT_BATCH_SIZE)
    ap.add_argument("--cv", action="store_true", help="check K-fold CV splits")
    ap.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-crop-flip", dest="crop_flip", action="store_false",
                    help="disable RandomResizedCrop and RandomHorizontalFlip")
    args = ap.parse_args()
 
    if args.stats:
        m, s = compute_mean_std(args.root)
        print(f"train mean={m:.4f}  std={s:.4f}")
 
    if args.cv:
        _check_cv(args.root, args.n_folds, args.seed)
        tr, _, _ = get_fold_dataloaders(
            0, args.root, n_folds=args.n_folds, batch_size=args.batch_size,
            num_workers=0, seed=args.seed, crop_flip=args.crop_flip,
        )
    else:
        tr, va, te = get_dataloaders(
            args.root, batch_size=args.batch_size, num_workers=0, seed=args.seed,
            crop_flip=args.crop_flip,
        )
        for name, dl in (("train", tr), ("val", va), ("test", te)):
            _describe(name, dl.dataset)
        overlap = set(tr.dataset.patients) & set(va.dataset.patients)
        print("train/val patient overlap:", len(overlap))   # should be 0
 
    print("train transform:", tr.dataset.transform)
    x, y = next(iter(tr))
    print(
        "batch:",
        tuple(x.shape),
        x.dtype,
        "labels:",
        y.tolist(),
        f"range=[{x.min():.2f}, {x.max():.2f}]",
    )