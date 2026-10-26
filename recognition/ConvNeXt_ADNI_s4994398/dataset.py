"""
Data loading for the ADNI AD/NC brain MRI slice dataset.

Expected layout (the preprocessed ADNI release used in COMP3710):

    <root>/train/AD/<scan>_<slice>.jpeg
    <root>/train/NC/<scan>_<slice>.jpeg
    <root>/test/AD/...
    <root>/test/NC/...

Each MRI *scan* contributes 20 axial slices (256 x 240, greyscale). The number
in the file name is the ADNI image (scan) ID, NOT a patient ID: one patient can
have up to 7 scans from different visits. ``meta_data_with_label.json`` maps
scan IDs to patient IDs (e.g. ``068_S_0473``).

Leakage found and fixed: the official train/ and test/ folders share 216
patients, and a scan-level validation split puts other visits of the same
patient on both sides. A model trained that way scored 0.97 on test patients it
had already seen and 0.39 on unseen ones, i.e. it learned to recognise patients,
not disease. ``patient_split`` therefore pools both folders and re-splits them
so that every patient is in exactly one of train / val / test.

All images are decoded once and cached as a uint8 array, which keeps every
epoch I/O free. Augmentation is done on the GPU in batches (see
``gpu_augment``) so the data pipeline never bottlenecks training on Windows,
where DataLoader worker processes are slow to start.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASSES = ("NC", "AD")  # label 0 = cognitive normal, label 1 = Alzheimer's
IMG_SIZE = 224
DEFAULT_ROOT = os.environ.get("ADNI_ROOT", r"D:\ADNI\AD_NC")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "cache")
DEFAULT_META = os.environ.get("ADNI_META", os.path.join(HERE, "meta_data_with_label.json"))


def scan_id(filename):
    """``1003730_100.jpeg`` -> ``1003730`` (ADNI image ID of the scan)."""
    return os.path.basename(filename).split("_")[0]


def load_patient_map(meta_path=DEFAULT_META):
    """Map scan ID -> patient ID using the ADNI metadata file.

    The patient ID (site_S_subject, e.g. ``068_S_0473``) is parsed from the raw
    NIfTI file name stored for each scan.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    return {k: re.search(r"ADNI_(\d+_S_\d+)_", v["raw"]).group(1) for k, v in meta.items()}


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


def patient_split(paths, labels, patient_of, val_frac=0.1, test_frac=0.2, seed=42):
    """Split slice indices into train/val/test by *patient*, stratified by class.

    Returns three sorted lists of indices into ``paths``.
    """
    # Group slice indices by patient. Each patient has a single diagnosis in this
    # dataset (checked: no patient appears under both AD and NC).
    by_patient = defaultdict(list)
    patient_label = {}
    for i, (p, y) in enumerate(zip(paths, labels)):
        pid = patient_of[scan_id(p)]
        by_patient[pid].append(i)
        assert patient_label.setdefault(pid, y) == y, f"patient {pid} has mixed labels"

    # Shuffle the *patients* (not slices or scans) of each class separately and cut
    # the list into test / val / train. Doing it per class keeps the AD/NC ratio the
    # same in all splits (stratification). The seeded RNG makes it reproducible.
    rng = random.Random(seed)
    splits = ([], [], [])  # train, val, test
    for label in range(len(CLASSES)):
        patients = sorted(pid for pid, y in patient_label.items() if y == label)  # sort first: deterministic
        rng.shuffle(patients)
        n_test = round(len(patients) * test_frac)
        n_val = round(len(patients) * val_frac)
        for k, pid in enumerate(patients):
            which = 2 if k < n_test else 1 if k < n_test + n_val else 0
            # every slice of every scan of this patient goes to the same split -> no leakage
            splits[which].extend(by_patient[pid])
    return tuple(sorted(s) for s in splits)


def balanced_weights(paths, labels, patient_of):
    """Per-slice sampling weights so every *patient* and every *class* counts equally.

    AD patients have ~3.4 scans each vs ~1.7 for NC, so uniform sampling lets the
    network score well by memorising the frequently repeated AD brains. Weighting
    each slice by 1 / (slices of its patient) gives every patient the same total
    weight; dividing by the number of patients in the class then balances AD vs NC.
    """
    pids = [patient_of[scan_id(p)] for p in paths]
    per_patient = Counter(pids)
    patients_in_class = Counter()
    for pid, y in set(zip(pids, labels)):
        patients_in_class[y] += 1
    w = [1.0 / (per_patient[pid] * patients_in_class[y]) for pid, y in zip(pids, labels)]
    return torch.tensor(w, dtype=torch.double)


def _load_images(paths):
    # Pre-allocate one uint8 array (1 byte/pixel: ~1.3 GB for the training set
    # instead of ~5 GB as float32) and fill it slice by slice.
    first = np.asarray(Image.open(paths[0]).convert("L"))  # "L" = 8-bit greyscale
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


def load_all(root=DEFAULT_ROOT, cache_dir=DEFAULT_CACHE):
    """Both official folders concatenated (they are re-split by patient afterwards)."""
    x1, y1, p1 = load_split("train", root, cache_dir)
    x2, y2, p2 = load_split("test", root, cache_dir)
    return torch.cat([x1, x2]), torch.cat([y1, y2]), p1 + p2


def load_patient_splits(root=DEFAULT_ROOT, meta_path=DEFAULT_META, val_frac=0.1, test_frac=0.2, seed=42):
    """Convenience wrapper used by train.py / predict.py.

    Returns ``{"train"|"val"|"test": (images, labels, paths)}`` and the patient map.
    """
    images, labels, paths = load_all(root)
    patient_of = load_patient_map(meta_path)
    idx = patient_split(paths, labels.tolist(), patient_of, val_frac, test_frac, seed)
    out = {}
    for name, ii in zip(("train", "val", "test"), idx):
        t = torch.tensor(ii)
        out[name] = (images[t], labels[t], [paths[i] for i in ii])
    return out, patient_of


class TensorBatcher:
    """Minimal in-memory batch iterator over uint8 image tensors.

    Yields ``(images uint8 [B,H,W], labels [B], indices [B])`` on ``device``.
    If ``weights`` is given, each epoch draws ``len(labels)`` samples *with
    replacement* in proportion to the weights (like WeightedRandomSampler).
    """

    def __init__(self, images, labels, batch_size, shuffle, device, drop_last=False, weights=None):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.drop_last = drop_last
        self.weights = weights

    def __len__(self):
        n = len(self.labels)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    # Why not torch.utils.data.DataLoader? The whole dataset already sits in RAM as one
    # tensor and augmentation runs on the GPU, so worker processes would add start-up
    # cost (slow on Windows) without any benefit. Indexing the tensor is enough.
    def __iter__(self):
        n = len(self.labels)
        if self.weights is not None:
            order = torch.multinomial(self.weights, n, replacement=True)  # balanced resampling
        elif self.shuffle:
            order = torch.randperm(n)  # new random order every epoch
        else:
            order = torch.arange(n)
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
    x = batch_u8.unsqueeze(1).float() / 255.0  # add channel dim, scale to [0, 1]
    # 256x240 -> 224x224, the resolution ConvNeXt's ImageNet weights were trained at
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - mean) / std  # zero mean / unit variance using training-set statistics


def gpu_augment(batch_u8, mean, std, size=IMG_SIZE, max_rot=10.0, scale=(0.9, 1.1),
                max_shift=0.06, hflip=0.0, intensity=0.1):
    """Per-sample random affine (rotation, scale, shift, optional flip) + brightness/contrast jitter.

    ``hflip`` defaults to 0: on these sagittal slices a horizontal flip swaps anterior and
    posterior, an anatomically impossible image (the README's ablation measures its cost).

    Implemented with ``affine_grid``/``grid_sample`` so each image in the batch gets
    its own random transform, all on the GPU.
    """
    B = batch_u8.shape[0]
    dev = batch_u8.device
    x = batch_u8.unsqueeze(1).float() / 255.0  # [B,1,H,W] in [0,1]

    # Draw independent random parameters for every image in the batch.
    theta_rad = torch.empty(B, device=dev).uniform_(-max_rot, max_rot) * torch.pi / 180.0
    s = torch.empty(B, device=dev).uniform_(*scale)
    flip = torch.where(torch.rand(B, device=dev) < hflip, -1.0, 1.0)  # -1 mirrors the x axis
    # affine_grid works in normalised coordinates where the image spans [-1, 1]
    # (width 2), so a shift of max_shift * image width is max_shift * 2.
    tx = torch.empty(B, device=dev).uniform_(-max_shift, max_shift) * 2
    ty = torch.empty(B, device=dev).uniform_(-max_shift, max_shift) * 2

    # Build one 2x3 affine matrix per image: [rotation/scale | translation].
    # affine_grid maps *output* pixels back to *input* locations, so dividing by s
    # samples a smaller/larger input region, i.e. zooms in for s > 1.
    cos, sin = torch.cos(theta_rad) / s, torch.sin(theta_rad) / s
    A = torch.stack([
        torch.stack([cos * flip, -sin, tx], dim=1),
        torch.stack([sin * flip, cos, ty], dim=1),
    ], dim=1)  # [B,2,3]
    # The grid has output size size x size, so this also resizes 256x240 -> 224x224 in
    # the same interpolation step (no separate resize needed). Out-of-image pixels are 0
    # (black), which matches the black background of the skull-stripped scans.
    grid = F.affine_grid(A, (B, 1, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Brightness / contrast jitter mimics scanner-to-scanner intensity differences:
    # x' = c * x + b with c in [1 - intensity, 1 + intensity] and a small offset b.
    c = 1 + torch.empty(B, 1, 1, 1, device=dev).uniform_(-intensity, intensity)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-intensity, intensity) * 0.5
    x = (x * c + b).clamp(0, 1)
    return (x - mean) / std  # standardise with the training-set statistics


if __name__ == "__main__":
    # Sanity check: patient-disjoint splits and their sizes.
    splits, patient_of = load_patient_splits()
    pats = {}
    for name, (x, y, paths) in splits.items():
        pats[name] = {patient_of[scan_id(p)] for p in paths}
        scans = {scan_id(p) for p in paths}
        print(f"{name:5s}: {len(y):5d} slices, {len(scans):4d} scans, {len(pats[name]):3d} patients, "
              f"AD slice fraction {y.float().mean():.3f}")
    assert not pats["train"] & pats["val"] and not pats["train"] & pats["test"] and not pats["val"] & pats["test"], \
        "patient leakage between splits"
    print("OK: no patient appears in more than one split")
    x_tr = splits["train"][0]
    print("train mean/std", compute_stats(x_tr))
