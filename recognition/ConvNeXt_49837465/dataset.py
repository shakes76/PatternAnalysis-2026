"""
ADNI MRI slices with patient-disjoint, class-stratified splits.

Load an existing manifest or regroup the course folders by patient.
Required columns: path, label (NC/AD), patient_id, scan_id, split.
All scans and slices from one patient remain in the same subset.
Default patient proportions: 70% train, 10% validation, 20% test.
"""
import os
import json
import zipfile
from pathlib import Path
import re
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    from torchvision import transforms
except ImportError:  # Minimal fallback when torchvision is unavailable.
    transforms = None


# Colab archive, extraction, and output paths.
DEFAULT_ZIP_PATH = "/content/drive/MyDrive/COMP3710/Final_Project_dataset/ADNI.zip"
DEFAULT_DATA_ROOT = "/content/comp3710_adni/ADNI"
DEFAULT_OUTPUT_DIR = (
    "/content/drive/MyDrive/COMP3710/Final_Project_dataset/convnext_results_4/"
    "convnext_narrow_final7740_reproduction"
)


def ensure_data_root(data_root=DEFAULT_DATA_ROOT, zip_path=DEFAULT_ZIP_PATH):
    """Reuse extracted data, or extract ADNI.zip into the parent directory."""
    root = Path(data_root).expanduser().resolve()
    image_root = root / "AD_NC" if (root / "AD_NC").is_dir() else root
    metadata_candidates = [root / "meta_data_with_label.json",
                           root.parent / "meta_data_with_label.json"]
    if ((image_root / "train").is_dir() and (image_root / "test").is_dir()
            and any(p.is_file() for p in metadata_candidates)):
        print(f"Dataset already extracted: {root}")
        return str(root)

    archive = Path(zip_path).expanduser()
    if not archive.is_file() and str(archive) == DEFAULT_ZIP_PATH:
        alternative = archive.with_name("ADNI(2).zip")
        if alternative.is_file():
            archive = alternative
    if not archive.is_file():
        raise FileNotFoundError(
            f"Incomplete dataset: {root}; archive not found: {archive}. "
            "Mount Google Drive or specify the archive with --zip_path."
        )
    # The course archive contains ADNI/AD_NC and its metadata JSON.
    if root.name != "ADNI":
        raise FileNotFoundError(
            f"Custom data directory is not ready: {root}. Automatic extraction requires a root named ADNI."
        )
    destination = root.parent
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.filename.startswith("__MACOSX/")]
        for member in members:
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError(f"Invalid zip member: {member.filename}")
        z.extractall(destination, members=members)
    if not ((root / "AD_NC" / "train").is_dir()
            and (root / "AD_NC" / "test").is_dir()
            and (root / "meta_data_with_label.json").is_file()):
        raise FileNotFoundError(f"Expected AD_NC folder or metadata missing after extraction: {root}")
    print(f"Dataset extracted: {root}")
    return str(root)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")

# ADNI subject ID format, e.g. 006_S_4153 or 037_S_4430.
_ADNI_SUBJECT_RE = re.compile(r"(\d{3}_S_\d{4})")


# --------------------------------------------------------------------------- #
# Parse patient and scan IDs from filenames
# --------------------------------------------------------------------------- #
def parse_patient_scan_id(filename, metadata=None):
    """Resolve patient/scan IDs; numeric filenames require metadata."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    short = re.fullmatch(r"(\d+)_(\d+)", stem)
    if short:
        scan_id = short.group(1)
        record = (metadata or {}).get(scan_id)
        if not isinstance(record, dict):
            raise ValueError(f"Missing metadata for scan {scan_id}; cannot resolve the patient.")
        subjects = {
            match.group(1)
            for value in record.values() if isinstance(value, str)
            for match in _ADNI_SUBJECT_RE.finditer(value)
        }
        if len(subjects) != 1:
            raise ValueError(f"Missing or ambiguous patient ID for scan {scan_id}: {subjects}")
        return subjects.pop(), scan_id
    m = _ADNI_SUBJECT_RE.search(stem)
    if m:
        patient_id = m.group(1)
        rest = stem[m.end():].lstrip("_").split("_")
        scan_tokens = [t for t in rest if t][:4]  # Use the leading scan-related fields.
        scan_id = "_".join([patient_id] + scan_tokens) if scan_tokens else patient_id
        return patient_id, scan_id
    raise ValueError(f"Cannot resolve patient/scan IDs from filename: {filename}")


# --------------------------------------------------------------------------- #
# Build a manifest and assign patient-disjoint subsets
# --------------------------------------------------------------------------- #
def build_manifest(
    data_root,
    out_csv=None,
    test_ratio=0.20,
    val_ratio=0.10,
    seed=42,
    folder_split=True,
    metadata_path=None,
):
    """Scan the course folders and stratify patients into train/val/test."""
    data_root = os.path.abspath(data_root)
    if os.path.isdir(os.path.join(data_root, "AD_NC")):
        data_root = os.path.join(data_root, "AD_NC")
    if metadata_path is None:
        for directory in (data_root, os.path.dirname(data_root)):
            candidate = os.path.join(directory, "meta_data_with_label.json")
            if os.path.isfile(candidate):
                metadata_path = candidate
                break
    metadata = None
    if metadata_path is not None:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    rows = []
    if folder_split:
        split_dirs = [d for d in os.listdir(data_root)
                      if os.path.isdir(os.path.join(data_root, d))]
    else:
        split_dirs = ["."]

    for split_name in split_dirs:
        split_dir = os.path.join(data_root, split_name)
        for label in ("AD", "CN", "NC"):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            label_norm = "NC" if label in ("CN", "NC") else "AD"
            for dirpath, _, files in os.walk(label_dir):
                for f in sorted(files):
                    if f.lower().endswith(IMG_EXTS):
                        path = os.path.join(dirpath, f)
                        pid, sid = parse_patient_scan_id(f, metadata)
                        rows.append({
                            "path": path,
                            "label": label_norm,
                            "patient_id": pid,
                            "scan_id": sid,
                            "orig_split": split_name,
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found in {data_root}; check the path and folder structure.")

    # ---- Regroup by patient to prevent overlap between subsets ----
    # Check label conflicts before deduplication can hide them.
    if (df.groupby("patient_id")["label"].nunique() > 1).any():
        raise ValueError("Conflicting class labels for the same patient.")
    rng = random.Random(seed)
    patients = (df[["patient_id", "label"]]
                .drop_duplicates("patient_id")
                .reset_index(drop=True))
    # Each patient must have exactly one label.
    assert patients["patient_id"].is_unique

    assign = {}
    for label in ("NC", "AD"):
        group = patients[patients["label"] == label]["patient_id"].tolist()
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_ratio)) if n > 2 else 0
        n_val = max(1, round(n * val_ratio)) if n - n_test > 2 else 0
        for i, pid in enumerate(group):
            if i < n_test:
                assign[pid] = "test"
            elif i < n_test + n_val:
                assign[pid] = "val"
            else:
                assign[pid] = "train"

    df["split"] = df["patient_id"].map(assign)
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Manifest saved to {out_csv}: {len(df)} slices, "
              f"{df['patient_id'].nunique()} patients.")
    return df


def load_manifest(csv_path, data_root=None):
    df = pd.read_csv(csv_path)
    need = {"path", "label", "patient_id", "scan_id", "split"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")
    df["label"] = df["label"].replace({"CN": "NC"})
    # Resolve image paths without changing labels, patients, or splits.
    def resolve_image_path(value):
        path = Path(value)
        if path.is_file():
            return str(path)
        if data_root is not None:
            root = Path(data_root)
            image_root = root / "AD_NC" if (root / "AD_NC").is_dir() else root
            if "AD_NC" in path.parts:
                candidate = image_root.joinpath(*path.parts[path.parts.index("AD_NC") + 1:])
            elif not path.is_absolute():
                candidate = root / path
            else:
                candidate = path
            if candidate.is_file():
                return str(candidate)
        candidate = Path(csv_path).resolve().parent / path
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Image in manifest not found: {value}")
    df["path"] = df["path"].map(resolve_image_path)
    return df


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #
def load_image(path):
    if path.lower().endswith(".npy"):
        import numpy as np
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]  # HW -> 1HW
        return torch.from_numpy(arr).float()
    with Image.open(path) as im:
        im = im.convert("L")  # Grayscale
        arr = torch.from_numpy(
            __import__("numpy").array(im, dtype="float32") / 255.0
        )
    return arr.unsqueeze(0)  # 1 x H x W


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_transform(img_size=224, train=True, jitter=0.0):
    """Shared preprocessing; apply light augmentation only during training."""

    if transforms is not None:
        ops = [
            transforms.Resize((img_size, img_size))
        ]

        if train:
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                ),

            ]

        # Jitter belongs to a later experiment, not the final report model.
        if train and jitter > 0:
            ops.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))

        ops += [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.5],
                std=[0.5],
            ),
        ]

        return transforms.Compose(ops)

    # Fallback preprocessing without torchvision.
    def _tf(img):
        x = load_image(img)

        x = torch.nn.functional.interpolate(
            x[None],
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )[0]

        return (x - 0.5) / 0.5

    return _tf


class ADNISliceDataset(Dataset):
    """Return image, label, and path/patient/scan metadata for evaluation."""

    def __init__(self, dataframe, img_size=224, train=True, jitter=0.0):
        self.df = dataframe.reset_index(drop=True)
        self.tf = build_transform(img_size, train=train, jitter=jitter)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if transforms is not None:
            with Image.open(row["path"]) as im:
                x = self.tf(im.convert("L"))
        else:
            x = self.tf(row["path"])
        y = 0 if row["label"] == "NC" else 1
        meta = {
            "path": row["path"],
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
        }
        return x, y, meta


def split_dataframe(df):
    """Return train, validation, and test DataFrames."""
    out = {}
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        out[s] = sub.reset_index(drop=True)
        print(f"split={s:5s}: {len(sub):6d} slices, "
              f"{sub['scan_id'].nunique():4d} scans, "
              f"{sub['patient_id'].nunique():4d} patients")
    # Check patient isolation.
    tr = set(out["train"]["patient_id"]) | set(out["val"]["patient_id"])
    te = set(out["test"]["patient_id"])
    overlap = tr & te
    print(f"Patient overlap between train/val and test: {len(overlap)} (expected 0)")
    if overlap or (set(out["train"]["patient_id"]) &
                   set(out["val"]["patient_id"])):
        raise ValueError("Patient overlap detected between train/val/test.")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--zip_path", default=DEFAULT_ZIP_PATH)
    ap.add_argument("--out_csv", default="manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metadata_path", default=None)
    args = ap.parse_args()
    args.data_root = ensure_data_root(args.data_root, args.zip_path)
    df = build_manifest(args.data_root, args.out_csv, seed=args.seed,
                        metadata_path=args.metadata_path)
    split_dataframe(df)
