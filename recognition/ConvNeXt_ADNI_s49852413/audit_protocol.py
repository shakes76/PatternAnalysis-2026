"""Reproducible ADNI Metadata & Protocol Confound Audit Script.

Author: Lingxiao Li (Student ID: 49852413)
Course: COMP3710 / COMP7710 — Pattern Recognition and Analysis (2026)

Inspects `/home/groups/comp3710/ADNI/AD_NC` (or local copy) together with
`meta_data_with_label.json` to quantify:
1. Scan ID (`Ixxxxxx`) chronological ranges across `train/AD`, `train/NC`, `test/AD`, `test/NC`.
2. MRI acquisition sequence breakdown (`MPRAGE`, `Accelerated_Sagittal_MPRAGE`, `IR-FSPGR`).
3. Exact cross-split subject (`XXX_S_XXXX`) overlap between official `train/` and `test/`.
4. Training-split cohort intensity statistics (`mean`, `std`) inside the cropped brain ROI.

Outputs a verifiable JSON report to `results/protocol_audit.json`.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from dataset import (
    _load_raw_grayscale,
    _split_subject_disjoint,
    resolve_adni_root,
    scan_split_directory,
)


def classify_protocol(raw_path_str: str) -> str:
    """Categorize ADNI NIfTI filename by MRI pulse sequence naming convention.

    Note: Sequence naming (`Standard_Sagittal_MPRAGE`, `Accelerated_Sagittal_MPRAGE`,
    `Accelerated_Sag_IR-FSPGR`) reflects scanner pulse sequence parameters, whereas
    acquisition era (`< 2017` ADNI-2 vs `>= 2017` ADNI-3) is determined from `extract_year`.
    """
    if not raw_path_str:
        return "Unmapped"
    if "IR-FSPGR" in raw_path_str or "IR_FSPGR" in raw_path_str:
        return "Accelerated_Sag_IR-FSPGR"
    if "Accelerated_Sagittal_MPRAGE" in raw_path_str or "Accelerated_Sag_MPRAGE" in raw_path_str:
        return "Accelerated_Sagittal_MPRAGE"
    if "MPRAGE" in raw_path_str or "MP-RAGE" in raw_path_str:
        return "Standard_Sagittal_MPRAGE"
    return "Other_T1_Sequence"


def extract_year(raw_path_str: str) -> int | None:
    """Extract 4-digit acquisition year from ADNI NIfTI timestamp (`_br_raw_YYYYMMDD` or `__raw_YYYYMMDD`)."""
    m = re.search(r"_(?:br_)?raw_(\d{4})\d{4}", raw_path_str)
    if m:
        return int(m.group(1))
    return None


def run_protocol_audit(data_root: str | Path | None = None) -> Dict[str, object]:
    resolved_root = resolve_adni_root(data_root)
    meta_candidates = [
        resolved_root / "meta_data_with_label.json",
        resolved_root.parent / "meta_data_with_label.json",
    ]
    meta_path = next((p for p in meta_candidates if p.is_file()), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"meta_data_with_label.json not found near {resolved_root}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_records = scan_split_directory(resolved_root / "train", split_tag="tr")
    test_records = scan_split_directory(resolved_root / "test", split_tag="te")

    def summarize_partition(records, part_name: str):
        scans: Dict[str, dict] = {}
        slice_indices: List[int] = []
        for r in records:
            slice_indices.append(r.slice_index)
            sid = r.patient_id.split("_")[-1]
            if r.patient_id not in scans:
                entry = meta.get(sid, {}) if isinstance(meta, dict) else {}
                raw_str = entry.get("raw", "")
                m_subj = re.search(r"(\d{3}_S_\d{4})", raw_str)
                subj_id = m_subj.group(1) if m_subj else f"UNMAPPED_{sid}"
                scans[r.patient_id] = {
                    "class_name": r.class_name,
                    "scan_id": sid,
                    "subject_id": subj_id,
                    "protocol": classify_protocol(raw_str),
                    "year": extract_year(raw_str),
                }

        by_class_proto: Dict[str, Counter] = defaultdict(Counter)
        by_class_era: Dict[str, Counter] = defaultdict(Counter)
        by_class_years: Dict[str, List[int]] = defaultdict(list)
        by_class_scan_ids: Dict[str, List[int]] = defaultdict(list)
        subjects_by_class: Dict[str, set] = defaultdict(set)

        for info in scans.values():
            cls = info["class_name"]
            by_class_proto[cls][info["protocol"]] += 1
            if info["year"] is not None:
                by_class_years[cls].append(info["year"])
                era_key = "pre_2017_ADNI2" if info["year"] < 2017 else "post_2017_ADNI3"
                by_class_era[cls][era_key] += 1
            if info["scan_id"].isdigit():
                by_class_scan_ids[cls].append(int(info["scan_id"]))
            subjects_by_class[cls].add(info["subject_id"])

        summary = {
            "partition": part_name,
            "total_slices": len(records),
            "total_scans": len(scans),
            "slice_index_min": int(min(slice_indices)),
            "slice_index_max": int(max(slice_indices)),
            "classes": {},
        }
        for cls in ("AD", "NC"):
            sids = sorted(by_class_scan_ids[cls])
            yrs = sorted(by_class_years[cls])
            summary["classes"][cls] = {
                "num_scans": len(sids),
                "num_unique_subjects": len(subjects_by_class[cls]),
                "scan_id_min": int(sids[0]) if sids else None,
                "scan_id_median": int(sids[len(sids) // 2]) if sids else None,
                "scan_id_max": int(sids[-1]) if sids else None,
                "year_min": int(yrs[0]) if yrs else None,
                "year_median": int(yrs[len(yrs) // 2]) if yrs else None,
                "year_max": int(yrs[-1]) if yrs else None,
                "era_counts": dict(sorted(by_class_era[cls].items())),
                "protocol_counts": dict(sorted(by_class_proto[cls].items())),
            }
        return summary, subjects_by_class, scans

    train_summary, train_subjs, train_scans = summarize_partition(train_records, "official_train")
    test_summary, test_subjs, test_scans = summarize_partition(test_records, "official_test")

    overlap_ad = train_subjs["AD"] & test_subjs["AD"]
    overlap_nc = train_subjs["NC"] & test_subjs["NC"]
    all_train_subjs = train_subjs["AD"] | train_subjs["NC"]
    all_test_subjs = test_subjs["AD"] | test_subjs["NC"]
    total_overlap = all_train_subjs & all_test_subjs

    # Compute pixel mean/std on (a) official_train and (b) subject-disjoint train split (seed=42)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(train_records), size=min(1000, len(train_records)), replace=False)
    pix_vals = [_load_raw_grayscale(train_records[int(idx)].filepath) / 255.0 for idx in sample_indices]
    stacked = np.stack(pix_vals, axis=0)
    train_cohort_mean = float(np.mean(stacked))
    train_cohort_std = float(np.std(stacked))

    subj_tr_records, _, _ = _split_subject_disjoint(
        scan_split_directory(resolved_root / "train") + scan_split_directory(resolved_root / "test"),
        meta_path,
        seed=42,
        val_fraction=0.15,
        train_slice_stride=2,
    )
    rng_subj = np.random.default_rng(42)
    subj_sample_idx = rng_subj.choice(len(subj_tr_records), size=min(1000, len(subj_tr_records)), replace=False)
    subj_pix = np.stack([_load_raw_grayscale(subj_tr_records[int(idx)].filepath) / 255.0 for idx in subj_sample_idx], axis=0)
    subj_tr_mean = float(np.mean(subj_pix))
    subj_tr_std = float(np.std(subj_pix))

    report = {
        "dataset_total_slices": len(train_records) + len(test_records),
        "dataset_total_scans": len(train_scans) + len(test_scans),
        "dataset_total_unique_subjects": len(all_train_subjs | all_test_subjs),
        "official_train": train_summary,
        "official_test": test_summary,
        "official_split_subject_overlap": {
            "ad_overlapping_subjects": len(overlap_ad),
            "nc_overlapping_subjects": len(overlap_nc),
            "total_overlapping_subjects": len(total_overlap),
            "cross_diagnosis_subjects": len((train_subjs["AD"] | test_subjs["AD"]) & (train_subjs["NC"] | test_subjs["NC"])),
        },
        "train_only_intensity_statistics": {
            "cropped_roi": "[16:208, 48:224]",
            "fixed_pipeline_shift_scale": [0.18, 0.25],
            "official_train_1000_slice_sample": {
                "mean": round(train_cohort_mean, 4),
                "std": round(train_cohort_std, 4),
            },
            "subject_disjoint_train_1000_slice_sample_seed42": {
                "mean": round(subj_tr_mean, 4),
                "std": round(subj_tr_std, 4),
            },
        },
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "protocol_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved protocol audit report to: {out_file}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_protocol_audit()
