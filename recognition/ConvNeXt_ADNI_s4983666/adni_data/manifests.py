"""Load verified rows from the frozen patient-level manifests.

Patient roles (fixed once by ``adni_splits.py prepare``; every scan of a
patient lives in exactly one role, checked by ``adni_splits.py verify``):

* development (70%), split into five folds. For fold k:
  ``train`` - fits the weights. The frozen manifests also hold a small
              ``early_stop.csv``; it is no longer used for model selection, so
              its patients are added to ``train`` when a fold is loaded;
  ``val``   - fold k's held-out validation patients, monitored every epoch and
              used to compare configurations, never to pick a checkpoint.
* ``calibration`` (10%) - fits Platt scaling and the referral threshold.
* ``test`` (20%) - scored once, only by the frozen final pipeline.
"""

import argparse
from collections import Counter
from pathlib import Path

from . import splits as adni_splits


def _open_verified(data_root: Path, splits_dir: Path) -> tuple[Path, dict, bytes]:
    """Re-audit sources and manifests, then return the sealed hash list."""
    data_root, splits_dir = Path(data_root).resolve(), Path(splits_dir).resolve()
    adni_splits.verify(argparse.Namespace(data_root=data_root, output=splits_dir))
    seal_bytes = (splits_dir / "COMPLETED.json").read_bytes()
    return splits_dir, adni_splits.read_json(splits_dir / "COMPLETED.json"), seal_bytes


def _read_rows(splits_dir: Path, seal: dict, relative: str, partition: str, expected_slices: int) -> list[dict]:
    """Read one manifest whose bytes still match the verified seal."""
    path = splits_dir / relative
    adni_splits.require(adni_splits.digest(path.read_bytes()) == seal["sha256"][relative],
                        f"Manifest changed after verification: {relative}")
    rows = adni_splits.read_csv(path, adni_splits.FIELDS)
    adni_splits.require(rows and all(row["partition"] == partition for row in rows),
                        f"{relative} must contain only {partition} rows.")
    scans = Counter(row["image_id"] for row in rows)
    adni_splits.require(all(count == expected_slices for count in scans.values()),
                        f"{relative} contains an incomplete scan.")
    return rows


def load_fold(data_root, splits_dir, fold: int) -> dict:
    """Return the train and val rows of one development fold (train includes early_stop.csv)."""
    splits_dir, seal, seal_bytes = _open_verified(data_root, splits_dir)
    report = adni_splits.read_json(splits_dir / "report.json")
    folds = report["config"]["folds"]
    adni_splits.require(type(fold) is int and 1 <= fold <= folds, f"fold must be between 1 and {folds}.")
    expected = report["config"]["expected_slices"]
    result = {"report": report, "manifest_sha256": adni_splits.digest(seal_bytes)}
    rows = {role: _read_rows(splits_dir, seal, f"fold_{fold:02d}/{role}.csv", "development", expected)
            for role in ("train", "early_stop", "val")}
    result["train"] = rows["train"] + rows["early_stop"]
    result["val"] = rows["val"]
    train_patients = {row["patient_id"] for row in result["train"]}
    adni_splits.require(train_patients.isdisjoint(row["patient_id"] for row in result["val"]),
                        "A patient appears in both train and val.")
    return result


def load_holdout(data_root, splits_dir, role: str) -> dict:
    """Return the calibration or test rows, which never enter training or selection."""
    adni_splits.require(role in ("calibration", "test"), "Held-out role must be calibration or test.")
    splits_dir, seal, seal_bytes = _open_verified(data_root, splits_dir)
    report = adni_splits.read_json(splits_dir / "report.json")
    rows = _read_rows(splits_dir, seal, f"{role}.csv", role, report["config"]["expected_slices"])
    return {"report": report, "manifest_sha256": adni_splits.digest(seal_bytes), role: rows}


def manifest_sha256(splits_dir) -> str:
    """Fingerprint the frozen manifest set for checkpoints and run records."""
    return adni_splits.digest((Path(splits_dir) / "COMPLETED.json").read_bytes())
