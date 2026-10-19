"""
Data loading for the ADNI AD/NC brain MRI slice dataset.

Expected layout (the preprocessed ADNI release used in COMP3710):

    <root>/train/AD/<subject>_<slice>.jpeg
    <root>/train/NC/<subject>_<slice>.jpeg
    <root>/test/AD/...
    <root>/test/NC/...

Each subject contributes 20 axial slices (256 x 240, greyscale). Because
slices of the same subject are highly correlated, the validation split is
made at the *subject* level so no subject appears in both train and val.

All images are decoded once and cached as a uint8 array, which keeps every
epoch I/O free. Augmentation is done on the GPU in batches (see
``gpu_augment``) so the data pipeline never bottlenecks training on Windows,
where DataLoader worker processes are slow to start.
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASSES = ("NC", "AD")  # label 0 = cognitive normal, label 1 = Alzheimer's
IMG_SIZE = 224
DEFAULT_ROOT = os.environ.get("ADNI_ROOT", r"D:\ADNI\AD_NC")
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def subject_id(filename):
    """``1003730_100.jpeg`` -> ``1003730``."""
    return os.path.basename(filename).split("_")[0]


def list_split(root, split):
    """Return ``(paths, labels)`` for ``split`` in {"train", "test"}, sorted for determinism."""
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(root, split, cls)
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(folder, name))
                labels.append(label)
    return paths, labels


def subject_split(paths, labels, val_frac=0.1, seed=42):
    """Split indices into train/val by subject, stratified by class.

    Returns two lists of indices into ``paths``.
    """
    by_subject = defaultdict(list)
    subject_label = {}
    for i, (p, y) in enumerate(zip(paths, labels)):
        s = subject_id(p)
        by_subject[s].append(i)
        subject_label[s] = y

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for label in range(len(CLASSES)):
        subjects = sorted(s for s, y in subject_label.items() if y == label)
        rng.shuffle(subjects)
        n_val = max(1, round(len(subjects) * val_frac))
        for k, s in enumerate(subjects):
            (val_idx if k < n_val else train_idx).extend(by_subject[s])
    return sorted(train_idx), sorted(val_idx)


def _load_images(paths):
    first = np.asarray(Image.open(paths[0]).convert("L"))
    out = np.empty((len(paths), *first.shape), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = np.asarray(Image.open(p).convert("L"))
    return out


def load_split(split, root=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE):
    """Load a whole split as ``(images uint8 [N,H,W], labels int64 [N], paths)``.

    The decoded array is cached to ``cache_dir`` so subsequent runs start instantly.
    """
    paths, labels = list_split(root, split)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{split}_{len(paths)}.npy")
    if os.path.exists(cache):
        images = np.load(cache)
    else:
        print(f"Decoding {len(paths)} {split} images (cached to {cache})...")
        images = _load_images(paths)
        np.save(cache, images)
    return torch.from_numpy(images), torch.tensor(labels, dtype=torch.long), paths


class TensorBatcher:
    """Minimal in-memory batch iterator over uint8 image tensors.

    Yields ``(images uint8 [B,H,W], labels [B], indices [B])`` on ``device``.
    """

    def __init__(self, images, labels, batch_size, shuffle, device, drop_last=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        for b in range(len(self)):
            idx = order[b * self.batch_size:(b + 1) * self.batch_size]
            yield (
                self.images[idx].to(self.device, non_blocking=True),
                self.labels[idx].to(self.device, non_blocking=True),
                idx,
            )


def compute_stats(images, sample=2000, seed=0):
    """Mean / std of pixel intensities in [0, 1] estimated on a random sample."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(images), generator=g)[:sample]
    x = images[idx].float() / 255.0
    return x.mean().item(), x.std().item()


def preprocess(batch_u8, mean, std, size=IMG_SIZE):
    """uint8 [B,H,W] -> normalised float [B,1,size,size] (deterministic, used for eval)."""
    x = batch_u8.unsqueeze(1).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - mean) / std


def gpu_augment(batch_u8, mean, std, size=IMG_SIZE, max_rot=10.0, scale=(0.9, 1.1),
                max_shift=0.06, hflip=0.5, intensity=0.1):
    """Per-sample random affine (rotation, scale, shift, flip) + brightness/contrast jitter.

    Implemented with ``affine_grid``/``grid_sample`` so each image in the batch gets
    its own random transform, all on the GPU.
    """
    B = batch_u8.shape[0]
    dev = batch_u8.device
    x = batch_u8.unsqueeze(1).float() / 255.0

    theta_rad = torch.empty(B, device=dev).uniform_(-max_rot, max_rot) * torch.pi / 180.0
    s = torch.empty(B, device=dev).uniform_(*scale)
    flip = torch.where(torch.rand(B, device=dev) < hflip, -1.0, 1.0)
    tx = torch.empty(B, device=dev).uniform_(-max_shift, max_shift) * 2
    ty = torch.empty(B, device=dev).uniform_(-max_shift, max_shift) * 2

    cos, sin = torch.cos(theta_rad) / s, torch.sin(theta_rad) / s
    A = torch.stack([
        torch.stack([cos * flip, -sin, tx], dim=1),
        torch.stack([sin * flip, cos, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(A, (B, 1, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # brightness / contrast jitter
    c = 1 + torch.empty(B, 1, 1, 1, device=dev).uniform_(-intensity, intensity)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-intensity, intensity) * 0.5
    x = (x * c + b).clamp(0, 1)
    return (x - mean) / std


if __name__ == "__main__":
    # Quick sanity check of the split and loading.
    paths, labels = list_split(DEFAULT_ROOT, "train")
    tr, va = subject_split(paths, labels)
    tr_subj = {subject_id(paths[i]) for i in tr}
    va_subj = {subject_id(paths[i]) for i in va}
    assert not tr_subj & va_subj, "subject leakage between train and val"
    print(f"train slices {len(tr)} ({len(tr_subj)} subjects), val slices {len(va)} ({len(va_subj)} subjects)")
    images, y, _ = load_split("train")
    print("train array", tuple(images.shape), "label counts", torch.bincount(y).tolist())
    print("mean/std", compute_stats(images))
