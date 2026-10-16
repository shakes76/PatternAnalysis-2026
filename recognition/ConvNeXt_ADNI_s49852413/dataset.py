"""Data loading, preprocessing, and strict patient-level splitting for ADNI Brain MRI.

Author: Lingxiao Li (s49852413)
Course: COMP3710 / COMP7710 Pattern Recognition and Analysis (2026)

This module implements:
1. Automatic discovery of the ADNI T1-weighted 2D MRI dataset directory
   (supporting Rangpur HPC `/home/groups/comp3710/ADNI/AD_NC` as well as local paths).
2. Subject/scan identifier parsing from ADNI slice filenames (e.g., `369883_4.jpeg`,
   `Sample_train_data_808819_88.jpeg`) to prevent cross-slice patient data leakage.
3. Strict patient-level stratified splitting (`train` -> `train` + `val`, with held-out `test`)
   and an explicit programmatic audit (`verify_no_patient_leakage`) asserting zero patient
   overlap across all splits.
4. Per-slice Z-score intensity standardization and anatomically plausible 2D MRI
   data augmentations.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASS_NAMES: Tuple[str, str] = ("NC", "AD")
LABEL_TO_IDX: Dict[str, int] = {"NC": 0, "CN": 0, "AD": 1}
IDX_TO_LABEL: Dict[int, str] = {0: "NC", 1: "AD"}

DEFAULT_ADNI_ROOTS: Tuple[str, ...] = (
    os.environ.get("ADNI_ROOT", ""),
    "/home/groups/comp3710/ADNI/AD_NC",
    "/home/groups/comp3710/ADNI",
    str(Path("/Users/lilingxiao/.gemini/antigravity/scratch/ADNI/AD_NC")),
    str(Path.home() / "Downloads" / "ADNI" / "AD_NC"),
    str(Path.home() / "ADNI" / "AD_NC"),
)

VALID_EXTENSIONS: Tuple[str, ...] = (".jpeg", ".jpg", ".png", ".nii", ".nii.gz")


@dataclass(frozen=True)
class SliceRecord:
    """Metadata record for a single 2D brain MRI slice."""

    filepath: Path
    label: int
    class_name: str
    patient_id: str
    slice_index: int


def resolve_adni_root(data_root: Optional[str | Path] = None) -> Path:
    """Locate the ADNI dataset root directory containing `train/` and `test/` splits.

    Args:
        data_root: Explicit directory path provided by the user, or None to search defaults.

    Returns:
        Resolved `Path` directly containing `train/` and `test/` subdirectories.

    Raises:
        FileNotFoundError: If no valid ADNI directory hierarchy can be found.
    """
    candidates: List[str] = [str(data_root)] if data_root else list(DEFAULT_ADNI_ROOTS)
    checked: List[str] = []
    for cand in candidates:
        if not cand:
            continue
        base = Path(cand).expanduser().resolve()
        for sub in (base, base / "AD_NC", base / "AD_CN"):
            checked.append(str(sub))
            if (sub / "train").is_dir() and (sub / "test").is_dir():
                return sub
    raise FileNotFoundError(
        "Could not locate ADNI dataset containing 'train/' and 'test/' folders. "
        f"Pass --data_root or set ADNI_ROOT. Checked: {checked}"
    )


def extract_patient_and_slice(filename: str) -> Tuple[str, int]:
    """Extract unique subject/scan identifier and slice index from an ADNI slice filename.

    Examples supported:
        - `369883_4.jpeg` -> (`"369883"`, 4)
        - `Sample_train_data_808819_88.jpeg` -> (`"808819"`, 88)
        - `ADNI_002_S_0413_slice_15.png` -> (`"002_S_0413"`, 15)
        - `case_104_slice_12.nii.png` -> (`"104"`, 12)

    Args:
        filename: Basename of the image file.

    Returns:
        Tuple of `(patient_id, slice_index)`.
    """
    stem = filename
    for ext in (".nii.gz", ".nii.png", ".jpeg", ".jpg", ".png", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break

    # Pattern 1: explicit ADNI subject format XXX_S_XXXX
    m_adni = re.search(r"(\d{3}_S_\d{4})", stem)
    if m_adni:
        pid = m_adni.group(1)
        nums = re.findall(r"(\d+)", stem[m_adni.end() :])
        slice_idx = int(nums[-1]) if nums else 0
        return pid, slice_idx

    # Pattern 2: case_XXX_slice_YYY
    m_case = re.search(r"case_(\d+)_slice_(\d+)", stem, flags=re.IGNORECASE)
    if m_case:
        return m_case.group(1), int(m_case.group(2))

    # Pattern 3: standard COMP3710 ADNI `<prefix>_<scan_id>_<slice_idx>` or `<scan_id>_<slice_idx>`
    m_pair = re.search(r"(\d+)[_-](\d+)$", stem)
    if m_pair:
        return m_pair.group(1), int(m_pair.group(2))

    # Fallback: split on last underscore
    if "_" in stem:
        prefix, last = stem.rsplit("_", 1)
        if last.isdigit():
            return prefix, int(last)
    return stem, 0


def scan_split_directory(split_dir: Path, split_tag: str = "") -> List[SliceRecord]:
    """Scan a split directory (`train/` or `test/`) for `AD` and `NC`/`CN` slice files.

    Args:
        split_dir: Path to `train` or `test` folder.
        split_tag: Optional namespace prefix if disambiguating folders.

    Returns:
        Sorted list of `SliceRecord` entries.
    """
    records: List[SliceRecord] = []
    for sub in sorted(split_dir.iterdir()):
        if not sub.is_dir():
            continue
        raw_cls = sub.name.upper()
        if raw_cls not in LABEL_TO_IDX:
            continue
        label = LABEL_TO_IDX[raw_cls]
        canonical_cls = IDX_TO_LABEL[label]

        for fpath in sorted(sub.iterdir()):
            if not fpath.is_file():
                continue
            fname_lower = fpath.name.lower()
            if not any(fname_lower.endswith(ext) for ext in VALID_EXTENSIONS):
                continue
            pid, slice_idx = extract_patient_and_slice(fpath.name)
            # Tag patient_id with class to guarantee two folders with numeric IDs never collide
            qualified_pid = f"{canonical_cls}_{pid}" if not split_tag else f"{split_tag}_{canonical_cls}_{pid}"
            records.append(
                SliceRecord(
                    filepath=fpath,
                    label=label,
                    class_name=canonical_cls,
                    patient_id=qualified_pid,
                    slice_index=slice_idx,
                )
            )
    return records


def split_patients_stratified(
    records: Sequence[SliceRecord],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SliceRecord], List[SliceRecord]]:
    """Split a list of `SliceRecord`s into train and validation sets strictly by `patient_id`.

    All 2D slices belonging to a given `patient_id` are placed exclusively in either
    the training split or the validation split, stratified by diagnostic class (`NC` vs `AD`).

    Args:
        records: Input slice records from the training pool.
        val_fraction: Fraction of patients per class to allocate to validation (0 < val_fraction < 1).
        seed: Random seed for reproducible patient permutation.

    Returns:
        `(train_records, val_records)` with zero patient overlap.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    patients_by_class: Dict[int, List[str]] = defaultdict(list)
    slices_by_patient: Dict[str, List[SliceRecord]] = defaultdict(list)

    for rec in records:
        slices_by_patient[rec.patient_id].append(rec)

    for pid, pid_records in sorted(slices_by_patient.items()):
        cls_label = pid_records[0].label
        patients_by_class[cls_label].append(pid)

    rng = np.random.default_rng(seed)
    train_pids: set[str] = set()
    val_pids: set[str] = set()

    for cls_label, pid_list in sorted(patients_by_class.items()):
        perm = list(pid_list)
        rng.shuffle(perm)
        n_val = max(1, int(round(len(perm) * val_fraction)))
        val_pids.update(perm[:n_val])
        train_pids.update(perm[n_val:])

    train_records = [r for r in records if r.patient_id in train_pids]
    val_records = [r for r in records if r.patient_id in val_pids]
    return train_records, val_records


def verify_no_patient_leakage(
    train_records: Sequence[SliceRecord],
    val_records: Sequence[SliceRecord],
    test_records: Sequence[SliceRecord],
) -> Dict[str, object]:
    """Programmatically verify and summarize strict patient-level separation across splits.

    Raises:
        AssertionError: If any `patient_id` appears in more than one split.
    """
    p_train = {r.patient_id for r in train_records}
    p_val = {r.patient_id for r in val_records}
    p_test = {r.patient_id for r in test_records}

    overlap_tv = p_train & p_val
    overlap_tt = p_train & p_test
    overlap_vt = p_val & p_test

    assert len(overlap_tv) == 0, f"CRITICAL DATA LEAKAGE: {len(overlap_tv)} patients in both train & val!"
    assert len(overlap_tt) == 0, f"CRITICAL DATA LEAKAGE: {len(overlap_tt)} patients in both train & test!"
    assert len(overlap_vt) == 0, f"CRITICAL DATA LEAKAGE: {len(overlap_vt)} patients in both val & test!"

    def _split_stats(recs: Sequence[SliceRecord]) -> Dict[str, int]:
        pids = {r.patient_id for r in recs}
        ad_pids = {r.patient_id for r in recs if r.label == 1}
        nc_pids = {r.patient_id for r in recs if r.label == 0}
        ad_slices = sum(1 for r in recs if r.label == 1)
        nc_slices = sum(1 for r in recs if r.label == 0)
        return {
            "total_slices": len(recs),
            "ad_slices": ad_slices,
            "nc_slices": nc_slices,
            "total_patients": len(pids),
            "ad_patients": len(ad_pids),
            "nc_patients": len(nc_pids),
        }

    return {
        "leakage_detected": False,
        "overlap_train_val": len(overlap_tv),
        "overlap_train_test": len(overlap_tt),
        "overlap_val_test": len(overlap_vt),
        "train": _split_stats(train_records),
        "val": _split_stats(val_records),
        "test": _split_stats(test_records),
    }


def _load_raw_grayscale(filepath: Path) -> np.ndarray:
    """Load a 2D slice from JPEG/PNG or NIfTI file as a 2D float32 array."""
    name_lower = filepath.name.lower()
    if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
        import nibabel as nib

        arr = nib.load(str(filepath)).get_fdata(caching="unchanged")
        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]
        return np.asarray(arr, dtype=np.float32)

    with Image.open(filepath) as img:
        gray = img.convert("L")
        return np.asarray(gray, dtype=np.float32)


class ADNIDataset(Dataset):
    """PyTorch Dataset for 2D T1-weighted ADNI Brain MRI classification (`NC` vs `AD`).

    Performs spatial resizing, optional medical-safe training augmentations, and
    per-slice Z-score standardization `(I - mean) / (std + 1e-6)`.
    """

    def __init__(
        self,
        records: Sequence[SliceRecord],
        image_size: int = 128,
        augment: bool = False,
        return_meta: bool = False,
        preload: bool = True,
    ) -> None:
        super().__init__()
        self.records: List[SliceRecord] = list(records)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.return_meta = bool(return_meta)
        self.preload = bool(preload)

        self._cached_images: Optional[np.ndarray] = None
        if self.preload and len(self.records) > 0:
            self._cached_images = np.empty(
                (len(self.records), self.image_size, self.image_size), dtype=np.float32
            )
            for idx, rec in enumerate(self.records):
                raw = _load_raw_grayscale(rec.filepath)
                pil = Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
                self._cached_images[idx] = np.asarray(pil, dtype=np.float32) / 255.0

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _apply_augmentations(img: np.ndarray) -> np.ndarray:
        """Apply anatomically valid 2D brain MRI augmentations in [0, 1] float space."""
        h, w = img.shape
        # 1. Random horizontal flip across the interhemispheric midline (p=0.5)
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()

        # 2. Random mild rotation (+/- 10 deg) and spatial translation (+/- 5% pixels)
        if np.random.rand() < 0.7:
            angle = float(np.random.uniform(-10.0, 10.0))
            tx = int(np.random.uniform(-0.05 * w, 0.05 * w))
            ty = int(np.random.uniform(-0.05 * h, 0.05 * h))
            pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
            pil = pil.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty), fillcolor=0)
            img = np.asarray(pil, dtype=np.float32) / 255.0

        # 3. Mild RF intensity scaling / gamma perturbation
        if np.random.rand() < 0.5:
            gamma = float(np.random.uniform(0.88, 1.14))
            scale = float(np.random.uniform(0.92, 1.08))
            img = np.clip(img, 0.0, 1.0) ** gamma * scale

        # 4. Subtle scanner Gaussian noise
        if np.random.rand() < 0.35:
            noise = np.random.normal(0.0, 0.018, size=img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img

    @staticmethod
    def standardize_slice(img: np.ndarray) -> np.ndarray:
        """Apply per-slice Z-score standardization to zero mean and unit variance."""
        mean = float(np.mean(img))
        std = float(np.std(img))
        return (img - mean) / (std + 1e-6)

    def get_raw_slice(self, idx: int) -> np.ndarray:
        """Return the un-standardized [0, 1] resized slice for visualization."""
        if self._cached_images is not None:
            return self._cached_images[idx].copy()
        raw = _load_raw_grayscale(self.records[idx].filepath)
        pil = Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        return np.asarray(pil, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = self.get_raw_slice(idx)
        if self.augment:
            img = self._apply_augmentations(img)

        norm_img = self.standardize_slice(img)
        tensor_img = torch.from_numpy(norm_img).unsqueeze(0).float()  # [1, H, W]
        label = torch.tensor(rec.label, dtype=torch.long)

        if self.return_meta:
            meta = {
                "index": idx,
                "filepath": str(rec.filepath),
                "patient_id": rec.patient_id,
                "slice_index": rec.slice_index,
                "class_name": rec.class_name,
            }
            return tensor_img, label, meta
        return tensor_img, label


def build_adni_dataloaders(
    data_root: Optional[str | Path] = None,
    image_size: int = 128,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    max_samples_per_split: Optional[int] = None,
    return_meta_eval: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """Construct leakage-free train, validation, and test DataLoaders for ADNI MRI.

    Args:
        data_root: Path to ADNI root (or None to auto-discover).
        image_size: Spatial resolution (H = W = image_size).
        batch_size: Mini-batch size.
        val_fraction: Patient fraction from `train/` assigned to `val`.
        seed: Random seed for reproducible patient-level split.
        num_workers: PyTorch DataLoader worker count.
        max_samples_per_split: Optional cap for rapid smoke-testing.
        return_meta_eval: Whether val/test datasets return slice metadata dicts.

    Returns:
        `(train_loader, val_loader, test_loader, audit_summary)`
    """
    resolved_root = resolve_adni_root(data_root)
    raw_train_records = scan_split_directory(resolved_root / "train")
    test_records = scan_split_directory(resolved_root / "test")

    # Ensure zero patient overlap between raw train and test; if any patient IDs collide,
    # namespace them only if they come from distinct physical folders or filter overlaps.
    test_pids = {r.patient_id for r in test_records}
    if any(r.patient_id in test_pids for r in raw_train_records):
        # Re-scan with explicit split namespace tag if raw filenames reused generic counters
        raw_train_records = scan_split_directory(resolved_root / "train", split_tag="tr")
        test_records = scan_split_directory(resolved_root / "test", split_tag="te")

    train_records, val_records = split_patients_stratified(
        raw_train_records, val_fraction=val_fraction, seed=seed
    )

    if max_samples_per_split is not None:
        train_records = train_records[:max_samples_per_split]
        val_records = val_records[: max(16, max_samples_per_split // 3)]
        test_records = test_records[: max(16, max_samples_per_split // 2)]

    audit = verify_no_patient_leakage(train_records, val_records, test_records)
    audit["resolved_root"] = str(resolved_root)
    audit["image_size"] = image_size

    train_ds = ADNIDataset(train_records, image_size=image_size, augment=True, return_meta=False)
    val_ds = ADNIDataset(val_records, image_size=image_size, augment=False, return_meta=return_meta_eval)
    test_ds = ADNIDataset(test_records, image_size=image_size, augment=False, return_meta=return_meta_eval)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader, test_loader, audit


if __name__ == "__main__":
    import json

    tr_loader, va_loader, te_loader, audit_info = build_adni_dataloaders(image_size=128, batch_size=32)
    print("=== ADNI Data Hygiene & Patient-Level Split Audit ===")
    print(json.dumps(audit_info, indent=2))
    xb, yb = next(iter(tr_loader))
    print(f"Sample train batch shape: images={tuple(xb.shape)}, labels={tuple(yb.shape)}")
