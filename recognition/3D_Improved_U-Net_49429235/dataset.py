import glob
import json
import os
import random
import re

import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PATIENT_RE = re.compile(r"^([A-Za-z]+_?\d+)")
WEEK_RE = re.compile(r"(Week\d+)", re.IGNORECASE)
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]

def _key(path):
    name = os.path.basename(path)
    p = PATIENT_RE.search(name)
    w = WEEK_RE.search(name)
    stem = name.split(".")[0]
    return (p.group(1) if p else stem, w.group(1) if w else "")


def find_pairs(root):
    img_files = sorted(glob.glob(os.path.join(root, "semantic_MRs", "*.nii*")))
    lab_files = sorted(glob.glob(os.path.join(root, "semantic_labels_only", "*.nii*")))
    if not img_files or not lab_files:
        raise FileNotFoundError(f"No NIfTI files found under {root} "
                                "(expected semantic_MRs/ and semantic_labels_only/)")
    labels = {_key(f): f for f in lab_files}
    pairs, unmatched = [], []
    for f in img_files:
        k = _key(f)
        if k in labels:
            pairs.append({"patient": k[0], "week": k[1], "image": f, "label": labels[k]})
        else:
            unmatched.append(f)
    if unmatched:
        print(f"WARNING: {len(unmatched)} image(s) had no matching label, e.g. {unmatched[0]}")
    return pairs


def split_by_patient(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    patients = sorted({p["patient"] for p in pairs})
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    test_p = set(patients[:n_test])
    val_p = set(patients[n_test:n_test + n_val])
    train_p = set(patients[n_test + n_val:])
    splits = {
        "train": [p for p in pairs if p["patient"] in train_p],
        "val": [p for p in pairs if p["patient"] in val_p],
        "test": [p for p in pairs if p["patient"] in test_p],
    }
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits):
    sets = {k: {p["patient"] for p in v} for k, v in splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise AssertionError(f"Patient leakage between {names[i]} and {names[j]}: {sorted(overlap)}")


def save_splits(splits, path):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path):
    with open(path) as f:
        splits = json.load(f)
    check_no_leakage(splits)
    return splits


def describe_splits(splits):
    for k, v in splits.items():
        print(f"{k:>5}: {len({p['patient'] for p in v}):3d} patients, {len(v):3d} volumes")

def _load_nifti(path):
    arr = nib.load(path).get_fdata(caching="unchanged")
    if arr.ndim == 4:          # some HipMRI files carry an extra trailing dim
        arr = arr[..., 0]
    return torch.from_numpy(arr).float()


class HipMRI3DDataset(Dataset):

    def __init__(self, pairs, target_size=(128, 128, 64), num_classes=6,
                 augment=False, flip_axes=(0,), cache=True):
        if any(s % 16 for s in target_size):
            raise ValueError("target_size dims must be divisible by 16")
        self.pairs = pairs
        self.target_size = tuple(target_size)
        self.num_classes = num_classes
        self.augment = augment
        self.flip_axes = flip_axes   # axes are (D,H,W) of the stored volume; verify left-right axis
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        p = self.pairs[idx]
        img = _load_nifti(p["image"])
        lab = _load_nifti(p["label"]).round().long()
        if lab.min() < 0 or lab.max() >= self.num_classes:
            raise ValueError(f"{p['label']}: label values {lab.unique().tolist()} "
                             f"outside [0,{self.num_classes - 1}]")
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = F.interpolate(img[None, None], size=self.target_size, mode="trilinear",
                            align_corners=False)[0]
        lab = F.interpolate(lab[None, None].float(), size=self.target_size,
                            mode="nearest")[0, 0].long()
        if self._cache is not None:
            self._cache[idx] = (img, lab)
        return img, lab

    def __getitem__(self, idx):
        img, lab = self._load(idx)
        if self.augment:
            img, lab = self._augment(img.clone(), lab.clone())
        return img, lab

    def _augment(self, img, lab):
        for ax in self.flip_axes:                       # random flips
            if random.random() < 0.5:
                img = torch.flip(img, dims=[ax + 1])    # +1 for the channel dim
                lab = torch.flip(lab, dims=[ax])
        img = img * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)   # intensity jitter
        if random.random() < 0.5:                       # gaussian noise
            img = img + 0.05 * torch.randn_like(img)
        return img, lab


def get_dataloaders(root, splits=None, target_size=(128, 128, 64), batch_size=1,
                    num_workers=0, num_classes=6, seed=42):
    if splits is None:
        splits = split_by_patient(find_pairs(root), seed=seed)
    describe_splits(splits)
    for name, pairs in splits.items():
        if not pairs:
            raise ValueError(f"'{name}' split is empty - check file-name matching in "
                             "dataset.py (PATIENT_RE / WEEK_RE)")
    loaders = {}
    for name, pairs in splits.items():
        ds = HipMRI3DDataset(pairs, target_size, num_classes, augment=(name == "train"))
        loaders[name] = DataLoader(ds, batch_size=batch_size if name == "train" else 1,
                                   shuffle=(name == "train"), num_workers=num_workers,
                                   persistent_workers=num_workers > 0,
                                   pin_memory=torch.cuda.is_available())
    return loaders, splits