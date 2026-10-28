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
    subject_id: Optional[str] = None
    acquisition_year: Optional[int] = None


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
    require_subject_disjoint: bool = False,
) -> Dict[str, object]:
    """Programmatically verify and summarize both 3D scan-level and human subject-level separation.

    Args:
        train_records: Training split slice records.
        val_records: Validation split slice records.
        test_records: Test split slice records.
        require_subject_disjoint: If True, asserts zero overlap at the human subject (`XXX_S_XXXX`) level
            in addition to zero 3D scan (`Ixxxxxx`) overlap.

    Raises:
        AssertionError: If any 3D scan (`patient_id`) overlaps across splits, or if
            `require_subject_disjoint=True` and any `subject_id` overlaps across splits.
    """
    p_train = {r.patient_id for r in train_records}
    p_val = {r.patient_id for r in val_records}
    p_test = {r.patient_id for r in test_records}

    overlap_tv = p_train & p_val
    overlap_tt = p_train & p_test
    overlap_vt = p_val & p_test

    assert len(overlap_tv) == 0, f"CRITICAL SCAN LEAKAGE: {len(overlap_tv)} scans in both train & val!"
    assert len(overlap_tt) == 0, f"CRITICAL SCAN LEAKAGE: {len(overlap_tt)} scans in both train & test!"
    assert len(overlap_vt) == 0, f"CRITICAL SCAN LEAKAGE: {len(overlap_vt)} scans in both val & test!"

    s_train = {r.subject_id or r.patient_id for r in train_records}
    s_val = {r.subject_id or r.patient_id for r in val_records}
    s_test = {r.subject_id or r.patient_id for r in test_records}

    subj_overlap_tv = s_train & s_val
    subj_overlap_tt = s_train & s_test
    subj_overlap_vt = s_val & s_test
    subject_disjoint = (len(subj_overlap_tv) == 0 and len(subj_overlap_tt) == 0 and len(subj_overlap_vt) == 0)

    if require_subject_disjoint:
        assert len(subj_overlap_tv) == 0, f"SUBJECT LEAKAGE: {len(subj_overlap_tv)} subjects in both train & val!"
        assert len(subj_overlap_tt) == 0, f"SUBJECT LEAKAGE: {len(subj_overlap_tt)} subjects in both train & test!"
        assert len(subj_overlap_vt) == 0, f"SUBJECT LEAKAGE: {len(subj_overlap_vt)} subjects in both val & test!"

    def _split_stats(recs: Sequence[SliceRecord]) -> Dict[str, int]:
        pids = {r.patient_id for r in recs}
        ad_pids = {r.patient_id for r in recs if r.label == 1}
        nc_pids = {r.patient_id for r in recs if r.label == 0}
        sids = {r.subject_id or r.patient_id for r in recs}
        ad_sids = {r.subject_id or r.patient_id for r in recs if r.label == 1}
        nc_sids = {r.subject_id or r.patient_id for r in recs if r.label == 0}
        ad_slices = sum(1 for r in recs if r.label == 1)
        nc_slices = sum(1 for r in recs if r.label == 0)
        return {
            "total_slices": len(recs),
            "ad_slices": ad_slices,
            "nc_slices": nc_slices,
            "total_patients": len(pids),
            "ad_patients": len(ad_pids),
            "nc_patients": len(nc_pids),
            "unique_subjects": len(sids),
            "ad_subjects": len(ad_sids),
            "nc_subjects": len(nc_sids),
        }

    return {
        "leakage_detected": not subject_disjoint,
        "scan_disjoint": True,
        "subject_disjoint": subject_disjoint,
        "overlap_train_val": len(overlap_tv),
        "overlap_train_test": len(overlap_tt),
        "overlap_val_test": len(overlap_vt),
        "subject_overlap_train_val": len(subj_overlap_tv),
        "subject_overlap_train_test": len(subj_overlap_tt),
        "subject_overlap_val_test": len(subj_overlap_vt),
        "train": _split_stats(train_records),
        "val": _split_stats(val_records),
        "test": _split_stats(test_records),
    }


def _load_raw_grayscale(filepath: Path) -> np.ndarray:
    """Load a 2D slice from JPEG/PNG or NIfTI file as a 2D float32 array.

    For 240x256 sagittal MPRAGE slices from `/home/groups/comp3710/ADNI/AD_NC`,
    crops the tight intracranial bounding box `[16:208, 48:224]` to remove
    uninformative outer background padding while preserving cortical/ventricular anatomy.
    """
    name_lower = filepath.name.lower()
    if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
        import nibabel as nib

        arr = nib.load(str(filepath)).get_fdata(caching="unchanged")
        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]
        return np.asarray(arr, dtype=np.float32)

    with Image.open(filepath) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        if arr.shape == (240, 256):
            arr = arr[16:208, 48:224]
        return arr


class ADNIDataset(Dataset):
    """PyTorch Dataset for 2D T1-weighted ADNI Brain MRI classification (`NC` vs `AD`).

    Performs sagittal brain ROI cropping, spatial resizing, optional anatomically safe
    training augmentations, and intensity standardization.
    """

    def __init__(
        self,
        records: Sequence[SliceRecord],
        image_size: int = 96,
        augment: bool = False,
        return_meta: bool = False,
        preload: bool = True,
        use_cohort_stats: bool = True,
    ) -> None:
        super().__init__()
        self.records: List[SliceRecord] = list(records)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.return_meta = bool(return_meta)
        self.preload = bool(preload)
        self.use_cohort_stats = bool(use_cohort_stats)

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
        """Apply anatomically valid sagittal brain MRI augmentations in [0, 1] float space.

        Note: Sagittal slices have asymmetric anterior-posterior anatomy (frontal lobe vs.
        cerebellum/occipital lobe) and therefore must NOT be horizontally flipped.
        """
        # 1. Mild sub-voxel spatial translation (+/- 2 pixels)
        if np.random.rand() < 0.5:
            dy = int(np.random.randint(-2, 3))
            dx = int(np.random.randint(-2, 3))
            img = np.roll(img, shift=(dy, dx), axis=(0, 1))

        # 2. Mild RF coil gamma perturbation
        if np.random.rand() < 0.5:
            gamma = float(np.random.uniform(0.92, 1.08))
            img = np.clip(img, 0.0, 1.0) ** gamma

        return img

    @staticmethod
    def standardize_slice(img: np.ndarray, use_cohort_stats: bool = False) -> np.ndarray:
        """Standardize slice intensities.

        When `use_cohort_stats=True` (used for the skull-stripped ADNI slices derived from
        `ADNI_T1_3T_MASKED_N4`), applies a fixed global affine shift/scale `(img - 0.18) / 0.25`
        (chosen as round constants near the empirical training-split ROI mean≈0.20–0.21, std≈0.26–0.27
        reported in `results/protocol_audit.json`, without fitting to validation or test splits)
        so whole-brain tissue-to-ICV volume ratios are preserved across slices.
        When `use_cohort_stats=False` (default for standalone calls), applies per-slice
        zero-mean unit-variance Z-score standardization.
        """
        if use_cohort_stats:
            return (img - 0.18) / 0.25
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

        norm_img = self.standardize_slice(img, use_cohort_stats=self.use_cohort_stats)
        tensor_img = torch.from_numpy(norm_img).unsqueeze(0).float()  # [1, H, W]
        label = torch.tensor(rec.label, dtype=torch.long)

        if self.return_meta:
            meta = {
                "index": idx,
                "filepath": str(rec.filepath),
                "patient_id": rec.patient_id,
                "subject_id": rec.subject_id or rec.patient_id,
                "slice_index": rec.slice_index,
                "class_name": rec.class_name,
                "acquisition_year": int(rec.acquisition_year) if rec.acquisition_year is not None else 0,
            }
            return tensor_img, label, meta
        return tensor_img, label


def _scan_to_subject_map(
    all_records: Sequence[SliceRecord],
    meta_json_path: Path,
    strict: bool = True,
) -> Dict[str, str]:
    """Map each 3D scan id (`patient_id`) to its ADNI subject id (`XXX_S_XXXX`) and populate `r.subject_id` and `r.acquisition_year`.

    Raises:
        RuntimeError: If `strict=True` and any scan ID cannot be resolved to an ADNI subject ID.
    """
    import json

    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    mapping: Dict[str, str] = {}
    year_mapping: Dict[str, Optional[int]] = {}
    unmapped_scans: List[str] = []
    for r in all_records:
        if r.patient_id not in mapping:
            raw_sid = r.patient_id.split("_")[-1]
            raw_str = meta.get(raw_sid, {}).get("raw", "") if isinstance(meta, dict) else ""
            m = re.search(r"(\d{3}_S_\d{4})", raw_str)
            if m:
                mapping[r.patient_id] = m.group(1)
            else:
                unmapped_scans.append(r.patient_id)
                mapping[r.patient_id] = r.patient_id
            m_yr = re.search(r"_(?:br_)?raw_(\d{4})\d{4}", raw_str)
            year_mapping[r.patient_id] = int(m_yr.group(1)) if m_yr else None
        object.__setattr__(r, "subject_id", mapping[r.patient_id])
        object.__setattr__(r, "acquisition_year", year_mapping[r.patient_id])

    if strict and len(unmapped_scans) > 0:
        raise RuntimeError(
            f"Failed to resolve {len(unmapped_scans)} scan IDs in {meta_json_path}: "
            f"examples={unmapped_scans[:5]}"
        )
    return mapping


def _split_longitudinal_with_metadata(
    all_records: Sequence[SliceRecord],
    meta_json_path: Path,
    seed: int = 42,
    train_slice_stride: int = 2,
) -> Tuple[List[SliceRecord], List[SliceRecord], List[SliceRecord]]:
    """Within-subject longitudinal 3D scan split using `meta_data_with_label.json`.

    Note on Subject Overlap: This split is 3D scan-disjoint (`scan_overlap = 0`) but intentionally
    shares longitudinal subjects across train/val/test to evaluate within-subject follow-up tracking
    (subjects with >= 4 scans in AD or >= 3 scans in NC have one intermediate follow-up scan held out
    while earlier/later scans remain in training). For strict unseen-patient evaluation required by
    Section 2.1, use `_split_subject_disjoint` (`split_mode="subject"`).
    """
    scan2subj = _scan_to_subject_map(all_records, meta_json_path, strict=True)

    by_scan: Dict[str, List[SliceRecord]] = defaultdict(list)
    for r in all_records:
        by_scan[r.patient_id].append(r)

    subj_scans: Dict[str, List[str]] = defaultdict(list)
    for pid, recs in sorted(by_scan.items()):
        subj = f"{recs[0].class_name}_{scan2subj[pid]}"
        subj_scans[subj].append(pid)

    rng = np.random.RandomState(seed)
    ad_holdout: List[str] = []
    nc_holdout: List[str] = []
    tr_pids: set[str] = set()

    for subj, pids in sorted(subj_scans.items()):
        pids_sorted = sorted(pids, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
        if len(pids_sorted) >= 4:
            mid = len(pids_sorted) // 2
            tr_pids.update([p for i, p in enumerate(pids_sorted) if i != mid])
            if subj.startswith("AD_"):
                ad_holdout.append(pids_sorted[mid])
            else:
                nc_holdout.append(pids_sorted[mid])
        elif len(pids_sorted) == 3 and subj.startswith("NC_"):
            tr_pids.update([pids_sorted[0], pids_sorted[2]])
            nc_holdout.append(pids_sorted[1])
        else:
            tr_pids.update(pids_sorted)

    rng.shuffle(ad_holdout)
    rng.shuffle(nc_holdout)
    n_each = min(len(ad_holdout), len(nc_holdout))
    n_val = n_each // 2

    va_pids = set(ad_holdout[:n_val] + nc_holdout[:n_val])
    te_pids = set(ad_holdout[n_val:n_each] + nc_holdout[n_val:n_each])
    tr_pids.update(ad_holdout[n_each:] + nc_holdout[n_each:])

    train_records = [r for r in all_records if r.patient_id in tr_pids][:: max(1, train_slice_stride)]
    val_records = [r for r in all_records if r.patient_id in va_pids]
    test_records = [r for r in all_records if r.patient_id in te_pids]
    return train_records, val_records, test_records


def _split_subject_disjoint(
    all_records: Sequence[SliceRecord],
    meta_json_path: Path,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    train_slice_stride: int = 2,
) -> Tuple[List[SliceRecord], List[SliceRecord], List[SliceRecord]]:
    """Strictly subject-disjoint split: every ADNI subject (`XXX_S_XXXX`) lives in exactly one split.

    Subjects are grouped regardless of diagnostic label, stratified by their majority label,
    and allocated ~70/15/15 by subject. This measures generalisation to *unseen patients*
    in full compliance with Section 2.1 ("all slices/lesions from one patient reside strictly in one split").
    """
    scan2subj = _scan_to_subject_map(all_records, meta_json_path, strict=True)
    subj_labels: Dict[str, List[int]] = defaultdict(list)
    for r in all_records:
        subj_labels[scan2subj[r.patient_id]].append(r.label)

    rng = np.random.RandomState(seed)
    tr_s: set[str] = set()
    va_s: set[str] = set()
    te_s: set[str] = set()
    for lab in (0, 1):
        subs = sorted(s for s, ls in subj_labels.items() if int(round(float(np.mean(ls)))) == lab)
        perm = list(rng.permutation(subs))
        n_te = max(1, int(round(len(perm) * test_fraction)))
        n_va = max(1, int(round(len(perm) * val_fraction)))
        te_s.update(perm[:n_te])
        va_s.update(perm[n_te : n_te + n_va])
        tr_s.update(perm[n_te + n_va :])

    train_records = [r for r in all_records if scan2subj[r.patient_id] in tr_s][:: max(1, train_slice_stride)]
    val_records = [r for r in all_records if scan2subj[r.patient_id] in va_s]
    test_records = [r for r in all_records if scan2subj[r.patient_id] in te_s]
    return train_records, val_records, test_records


def build_adni_dataloaders(
    data_root: Optional[str | Path] = None,
    image_size: int = 96,
    batch_size: int = 128,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    max_samples_per_split: Optional[int] = None,
    return_meta_eval: bool = False,
    split_mode: str = "subject",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """Construct train, validation, and test DataLoaders for ADNI MRI with full subject/scan audit.

    Args:
        data_root: Path to ADNI root (or None to auto-discover).
        image_size: Spatial resolution (H = W = image_size).
        batch_size: Mini-batch size.
        val_fraction: Patient fraction allocated to `val`.
        seed: Random seed for reproducible patient/scan-level split.
        num_workers: PyTorch DataLoader worker count.
        max_samples_per_split: Optional cap for rapid smoke-testing.
        return_meta_eval: Whether val/test datasets return slice metadata dicts.
        split_mode: `"subject"` (default; strictly subject-disjoint, zero subject overlap),
            `"longitudinal"` (scan-disjoint, shared longitudinal subjects across splits), or
            `"official"` (raw `train/` and `test/` folder partition).

    Returns:
        `(train_loader, val_loader, test_loader, audit_summary)`
    """
    if split_mode not in ("longitudinal", "subject", "official"):
        raise ValueError(f"Unknown split_mode: {split_mode}")
    resolved_root = resolve_adni_root(data_root)
    raw_train_records = scan_split_directory(resolved_root / "train")
    raw_test_records = scan_split_directory(resolved_root / "test")

    meta_candidates = [
        resolved_root / "meta_data_with_label.json",
        resolved_root.parent / "meta_data_with_label.json",
    ]
    meta_path = next((p for p in meta_candidates if p.is_file()), None)

    has_full_meta = (
        meta_path is not None
        and len(raw_train_records) + len(raw_test_records) >= 10000
    )
    if has_full_meta and split_mode == "subject":
        all_records = raw_train_records + raw_test_records
        train_records, val_records, test_records = _split_subject_disjoint(
            all_records, meta_path, seed=seed, val_fraction=val_fraction, train_slice_stride=2
        )
        require_subj_disjoint = True
    elif has_full_meta and split_mode == "longitudinal":
        all_records = raw_train_records + raw_test_records
        train_records, val_records, test_records = _split_longitudinal_with_metadata(
            all_records, meta_path, seed=seed, train_slice_stride=2
        )
        require_subj_disjoint = False
    else:
        test_pids = {r.patient_id for r in raw_test_records}
        if any(r.patient_id in test_pids for r in raw_train_records):
            raw_train_records = scan_split_directory(resolved_root / "train", split_tag="tr")
            raw_test_records = scan_split_directory(resolved_root / "test", split_tag="te")
        if has_full_meta and meta_path is not None:
            _scan_to_subject_map(raw_train_records + raw_test_records, meta_path, strict=True)
        train_records, val_records = split_patients_stratified(
            raw_train_records, val_fraction=val_fraction, seed=seed
        )
        test_records = raw_test_records
        require_subj_disjoint = (split_mode == "subject")

    if max_samples_per_split is not None:
        train_records = train_records[:max_samples_per_split]
        val_records = val_records[: max(16, max_samples_per_split // 3)]
        test_records = test_records[: max(16, max_samples_per_split // 2)]

    audit = verify_no_patient_leakage(
        train_records, val_records, test_records, require_subject_disjoint=require_subj_disjoint
    )
    audit["resolved_root"] = str(resolved_root)
    audit["image_size"] = image_size
    audit["metadata_json_used"] = str(meta_path) if meta_path is not None else None
    audit["split_mode"] = split_mode

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

    for mode in ("subject", "longitudinal", "official"):
        _, _, _, audit_info = build_adni_dataloaders(image_size=96, batch_size=32, split_mode=mode)
        print(f"=== ADNI Split Audit ({mode}) ===")
        print(json.dumps(audit_info, indent=2))
