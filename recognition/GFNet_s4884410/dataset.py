"""
Data loading and preprocessing for ADNI AD vs NC classification.

Data layout: <root>/AD_NC/{train,test}/{AD,NC}/<patientID>_<sliceNo>.jpeg
    - 256 x 240 greyscale JPEGs, 20 consecutive slices per patient
    - near-midline sagittal slices, skull-stripped, rotated ~90 degrees

Splits:
    - train / val: patients from the official train folder, assigned by the patient
      lists in splits/val_split.json (created by scripts/audit_data.py), so every
      run uses the same split
    - test: the official test folder, used only for final evaluation
    No patient may appear in more than one split; this is checked on every load.

Preprocessing (all splits): 224 x 224 centre crop, scale to [0, 1], and normalise
with a mean and standard deviation computed on the training patients only.

Training augmentation (training split only, switchable): a random affine transform
(rotation, translation, scaling) and brightness/contrast jitter, set by the AUG_*
constants below. No flips: in these rotated sagittal views a flip would produce
anatomically impossible images.

Each slice sample is (image, label, patient_id, slice_pos), where label is
NC = 0, AD = 1 and slice_pos is the slice's position (0-19) within its patient.
slice_pos is used for analysis only and is never given to the model.

Usage:
    python dataset.py    # print split sizes, normalisation stats and sample shapes
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = "/home/groups/comp3710/ADNI"
DEFAULT_SPLIT_FILE = PROJECT_DIR / "splits" / "val_split.json"
DEFAULT_STATS_FILE = PROJECT_DIR / "splits" / "norm_stats.json"

LABELS = {"NC": 0, "AD": 1}  # AD is the positive class
IMG_SIZE = 224
IMG_EXTS = {".jpeg", ".jpg", ".png"}

# Training augmentation (applied to the training split only)
AUG_ROTATION = 15
AUG_TRANSLATE = 0.10
AUG_SCALE = (0.90, 1.10)
AUG_BRIGHTNESS = 0.0
AUG_CONTRAST = 0.0


def parse_name(path: Path):
    """'<patientID>_<sliceNo>.jpeg' -> (patient_id, slice_no as int)."""
    pid, slice_no = path.stem.rsplit("_", 1)
    return pid, int(slice_no)


def _scan_folder(folder: Path, label: int, keep_ids=None):
    """List one class folder (AD or NC) as records, optionally keeping only some patients.

    Each record is a dict: {"path", "label", "pid", "slice_no"}.
    If keep_ids is given, only slices of those patient IDs are returned.
    """
    records = []
    for p in folder.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        pid, slice_no = parse_name(p)
        if keep_ids is not None and pid not in keep_ids:
            continue
        records.append({"path": p, "label": label, "pid": pid, "slice_no": slice_no})
    return records


def _add_slice_positions(records):
    """Give each slice its position within its patient, sorting slice numbers numerically.

    Adds "slice_pos" (0-19) to every record in place, then sorts the list by
    patient ID and slice position. Slice numbers differ between patients, so the
    position, not the raw number, is what is comparable across patients.
    """
    by_pid = defaultdict(list)
    for r in records:
        by_pid[r["pid"]].append(r)
    for patient_records in by_pid.values():
        patient_records.sort(key=lambda r: r["slice_no"])
        for pos, r in enumerate(patient_records):
            r["slice_pos"] = pos
    records.sort(key=lambda r: (r["pid"], r["slice_pos"]))


def check_no_leakage(records_by_split):
    """Assert no patient is in two splits, and every patient has a single label.

    Args:
        records_by_split: {"train": [...], "val": [...], "test": [...]} slice records.

    Raises:
        AssertionError: naming the offending patients if either check fails.
    """
    ids = {name: {r["pid"] for r in recs} for name, recs in records_by_split.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = ids[a] & ids[b]
            assert not shared, f"Patients in both {a} and {b}: {sorted(shared)[:5]}"
    for name, recs in records_by_split.items():
        labels = defaultdict(set)
        for r in recs:
            labels[r["pid"]].add(r["label"])
        mixed = [pid for pid, ls in labels.items() if len(ls) > 1]
        assert not mixed, f"Patients with both labels in {name}: {mixed[:5]}"


def build_records(root=DEFAULT_ROOT, split_file=DEFAULT_SPLIT_FILE):
    """Return {'train': [...], 'val': [...], 'test': [...]} slice records.

    Train/val membership comes from the saved patient lists; nothing is
    re-randomised here, so every run uses identical splits.
    """
    base = Path(root) / "AD_NC"
    with open(split_file) as f:
        split = json.load(f)

    out = {"train": [], "val": [], "test": []}
    for cls, label in LABELS.items():
        for part in ("train", "val"):
            wanted = set(split[part][cls])
            found = _scan_folder(base / "train" / cls, label, keep_ids=wanted)
            missing = wanted - {r["pid"] for r in found}
            assert not missing, (
                f"{len(missing)} {part}/{cls} patients from split file not found"
            )
            out[part] += found
        out["test"] += _scan_folder(base / "test" / cls, label)

    for recs in out.values():
        _add_slice_positions(recs)
    check_no_leakage(out)
    return out


def build_transform(mean=None, std=None, augment=False):
    """Image transform: optional augmentation, 224 centre crop, scale to [0, 1], normalise.

    Args:
        mean, std: normalisation statistics; if mean is None, no normalisation.
        augment: if True, add the random affine and brightness/contrast jitter
            (training split only). The affine runs on the full 256 x 240 image
            before the crop, and fill=0 keeps uncovered corners black.
    """
    steps = []
    if augment:
        steps += [
            T.RandomAffine(
                degrees=AUG_ROTATION,
                translate=(AUG_TRANSLATE, AUG_TRANSLATE),
                scale=AUG_SCALE,
                fill=0,
            ),
            T.ColorJitter(brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST),
        ]
    steps += [T.CenterCrop(IMG_SIZE), T.ToTensor()]  # ToTensor scales to [0, 1]
    if mean is not None:
        steps.append(T.Normalize([mean], [std]))
    return T.Compose(steps)


def compute_norm_stats(train_records, num_workers=4, batch_size=256):
    """Pixel mean and std over training patients only (cropped, no augmentation).

    Computed in a single pass from running sums, in double precision.
    """
    ds = ADNISliceDataset(train_records, build_transform())
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    total, total_sq, n = 0.0, 0.0, 0
    for x, *_ in loader:
        x = x.double()
        total += x.sum().item()
        total_sq += (x**2).sum().item()
        n += x.numel()
    mean = total / n
    std = (total_sq / n - mean**2) ** 0.5
    return mean, std


def load_or_compute_norm_stats(
    train_records, stats_file=DEFAULT_STATS_FILE, num_workers=4
):
    """Load cached stats if they match the current training set, otherwise recompute and save.

    The cache (stats_file) is reused only if its image count, patient count and
    image size match the current training set. Delete the file after changing
    the split to force a recompute.

    Returns:
        (mean, std)
    """
    key = {
        "n_images": len(train_records),
        "n_patients": len({r["pid"] for r in train_records}),
        "img_size": IMG_SIZE,
    }
    stats_file = Path(stats_file)
    if stats_file.exists():
        cached = json.loads(stats_file.read_text())
        if all(cached.get(k) == v for k, v in key.items()):
            return cached["mean"], cached["std"]

    mean, std = compute_norm_stats(train_records, num_workers=num_workers)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps({**key, "mean": mean, "std": std}, indent=2))
    return mean, std


class ADNISliceDataset(Dataset):
    """One item per slice: (image [1, 224, 224], label, patient_id, slice_pos).

    Images are read from disk when requested, not held in memory.
    """

    def __init__(self, records, transform):
        """records: slice records for one split; transform: applied to each image."""
        self.records = records
        self.transform = transform

    def __len__(self):
        """Number of slices."""
        return len(self.records)

    def load_image(self, i):
        """Open slice i as greyscale and apply the transform."""
        with Image.open(self.records[i]["path"]) as im:
            img = im.convert("L")
        return self.transform(img)

    def __getitem__(self, i):
        """Return (image, label, patient_id, slice_pos) for slice i."""
        r = self.records[i]
        return self.load_image(i), r["label"], r["pid"], r["slice_pos"]


class ADNIPatientDataset(Dataset):
    """One item per patient: (images [20, 1, 224, 224] in slice order, label, patient_id).

    Wraps a slice dataset and reuses its transform. Use with
    DataLoader(..., batch_size=None) so that each batch is one patient.
    """

    def __init__(self, slice_dataset: ADNISliceDataset):
        """Group the slice dataset's records by patient, in slice order."""
        self.slices = slice_dataset
        groups = defaultdict(list)
        for i, r in enumerate(slice_dataset.records):
            groups[r["pid"]].append(i)
        recs = slice_dataset.records
        self.pids = sorted(groups)
        self.index = {
            pid: sorted(idx, key=lambda i: recs[i]["slice_pos"])
            for pid, idx in groups.items()
        }

    def __len__(self):
        """Number of patients."""
        return len(self.pids)

    def __getitem__(self, k):
        """Return (images, label, patient_id) for the k-th patient (sorted by ID)."""
        pid = self.pids[k]
        idx = self.index[pid]
        images = torch.stack([self.slices.load_image(i) for i in idx])
        label = self.slices.records[idx[0]]["label"]
        return images, label, pid


def get_datasets(
    root=DEFAULT_ROOT,
    split_file=DEFAULT_SPLIT_FILE,
    stats_file=DEFAULT_STATS_FILE,
    num_workers=4,
    augment=False,
):
    """Build train/val/test slice datasets sharing one set of training-only norm stats.

    Only the training split is augmented, and only if augment is True;
    validation and test always use the deterministic transform.

    Returns:
        ({"train": ..., "val": ..., "test": ...} ADNISliceDatasets, (mean, std))
    """
    records = build_records(root, split_file)
    mean, std = load_or_compute_norm_stats(records["train"], stats_file, num_workers)
    eval_tf = build_transform(mean, std)
    datasets = {
        "train": ADNISliceDataset(
            records["train"], build_transform(mean, std, augment)
        ),
        "val": ADNISliceDataset(records["val"], eval_tf),
        "test": ADNISliceDataset(records["test"], eval_tf),
    }
    return datasets, (mean, std)


def get_loaders(datasets, batch_size=64, num_workers=4):
    """Slice-level DataLoaders for the three splits. Only train is shuffled.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def _print_split_sizes(datasets):
    """Print the number of patients (total, AD, NC) and slices in each split."""
    print("\n== Split sizes ==")
    print(f"{'split':6} {'patients':>8} {'AD':>5} {'NC':>5} {'slices':>7}")
    for name, ds in datasets.items():
        label_of = {r["pid"]: r["label"] for r in ds.records}
        ad = sum(label_of.values())
        print(
            f"{name:6} {len(label_of):8d} {ad:5d} {len(label_of) - ad:5d} {len(ds):7d}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=DEFAULT_ROOT, help="ADNI data folder")
    ap.add_argument(
        "--split-file", default=DEFAULT_SPLIT_FILE, help="train/val patient lists"
    )
    ap.add_argument(
        "--stats-file", default=DEFAULT_STATS_FILE, help="cached normalisation stats"
    )
    ap.add_argument(
        "--num-workers", type=int, default=4, help="image-loading processes"
    )
    args = ap.parse_args()

    datasets, (mean, std) = get_datasets(
        args.root, args.split_file, args.stats_file, num_workers=args.num_workers
    )
    _print_split_sizes(datasets)
    print("PASS: no patient-level leakage between train, val and test")
    print(f"\nNormalisation (training patients only): mean={mean:.4f} std={std:.4f}")

    x, y, pid, pos = datasets["train"][0]
    print(
        f"\nSlice sample: image {tuple(x.shape)}, label {y}, patient {pid}, slice pos {pos}"
    )
    images, y, pid = ADNIPatientDataset(datasets["val"])[0]
    print(f"Patient sample: images {tuple(images.shape)}, label {y}, patient {pid}")
