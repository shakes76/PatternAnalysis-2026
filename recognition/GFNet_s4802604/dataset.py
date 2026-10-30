"""Data loading for ADNI AD vs NC slices, with a subject-level split.

splits.json (made by make_splits.py) lists scan IDs per split, and every
subject is in exactly one split. Each scan has 20 slices. Each item returned
is (image, label, scan_id) so predictions can be aggregated per scan later.

Set leaky=True to use the provided train/test folders instead (they share
subjects), which is only for the leakage comparison in the README.
"""
import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = "/home/groups/comp3710/ADNI"
WORK = os.path.expanduser("~/adni_work")
SPLITS = f"{WORK}/splits.json"
STATS = f"{WORK}/stats.json"
META = f"{ROOT}/meta_data_with_label.json"
LABEL_MAP = {0: 0, 2: 1}  # metadata: 0 = NC, 2 = AD  ->  NC = 0, AD = 1
IMG_SIZE = 224


def build_index():
    """scan_id -> {folder, orig (train/test), files}. Walks the disk once."""
    index = {}
    for orig in ("train", "test"):
        for cls in ("AD", "NC"):
            folder = f"{ROOT}/AD_NC/{orig}/{cls}"
            for fname in sorted(os.listdir(folder)):
                scan_id = fname.rsplit("_", 1)[0]
                entry = index.setdefault(scan_id, {"folder": folder, "orig": orig, "files": []})
                entry["files"].append(fname)
    return index


def get_scan_lists(index, leaky=False, seed=42):
    """Return {'train': [...], 'val': [...], 'test': [...]} of scan IDs."""
    if not leaky:
        return json.load(open(SPLITS))
    # Provided split: scan-level validation carve-out, subjects still overlap
    train = sorted(s for s, e in index.items() if e["orig"] == "train")
    test = sorted(s for s, e in index.items() if e["orig"] == "test")
    random.Random(seed).shuffle(train)
    n_val = int(0.15 * len(train))
    return {"val": train[:n_val], "train": train[n_val:], "test": test}


class ADNISlices(Dataset):
    def __init__(self, split, leaky=False, augment=False, mean=0.0, std=1.0, size=IMG_SIZE, strong=False):
        meta = json.load(open(META))
        index = build_index()
        self.samples = []  # (path, label, scan_id), one per slice
        for scan_id in get_scan_lists(index, leaky)[split]:
            entry = index[scan_id]
            label = LABEL_MAP[meta[scan_id]["label"]]
            for fname in entry["files"]:
                self.samples.append((f"{entry['folder']}/{fname}", label, scan_id))

        ops = [transforms.Resize((size, size))]
        if augment and strong:  # training only; no flips, since left/right matters in brain MRI
            ops += [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3)]
        elif augment:
            ops += [transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)]
        ops += [transforms.ToTensor(), transforms.Normalize([mean], [std])]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, scan_id = self.samples[i]
        img = Image.open(path).convert("L")
        return self.transform(img), label, scan_id


def compute_mean_std(leaky=False, num_workers=4):
    """Pixel mean/std over the TRAIN split only (never val/test), cached to disk."""
    key = "leaky" if leaky else "clean"
    stats = json.load(open(STATS)) if os.path.exists(STATS) else {}
    if key in stats:
        return stats[key]["mean"], stats[key]["std"]
    loader = DataLoader(ADNISlices("train", leaky=leaky), batch_size=256, num_workers=num_workers)
    total, total_sq, n = 0.0, 0.0, 0
    for x, _, _ in loader:
        total += x.sum().item()
        total_sq += (x ** 2).sum().item()
        n += x.numel()
    mean = total / n
    std = (total_sq / n - mean ** 2) ** 0.5
    stats[key] = {"mean": mean, "std": std}
    json.dump(stats, open(STATS, "w"))
    return mean, std


def get_loaders(batch_size=64, leaky=False, num_workers=4, strong_aug=False):
    """Return train, val, test DataLoaders (shuffle and augmentation on train only)."""
    mean, std = compute_mean_std(leaky, num_workers)
    sets = {
        "train": ADNISlices("train", leaky, augment=True, mean=mean, std=std, strong=strong_aug),
        "val": ADNISlices("val", leaky, mean=mean, std=std),
        "test": ADNISlices("test", leaky, mean=mean, std=std),
    }
    return tuple(
        DataLoader(sets[s], batch_size=batch_size, shuffle=(s == "train"),
                   num_workers=num_workers, pin_memory=True)
        for s in ("train", "val", "test")
    )


if __name__ == "__main__":
    # Sanity checks: sizes, no scan overlap between splits, one batch's shape
    train_dl, val_dl, test_dl = get_loaders(batch_size=8, num_workers=2)
    ids = {n: {s[2] for s in dl.dataset.samples}
           for n, dl in zip(("train", "val", "test"), (train_dl, val_dl, test_dl))}
    for n, dl in zip(ids, (train_dl, val_dl, test_dl)):
        print(n, len(dl.dataset), "slices", len(ids[n]), "scans")
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) \
        and not (ids["val"] & ids["test"]), "scan overlap between splits!"
    x, y, sid = next(iter(train_dl))
    print("batch:", x.shape, y.tolist(), sid[:3])
    print("mean/std:", compute_mean_std())
