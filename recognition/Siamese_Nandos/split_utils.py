"""Frozen-manifest validation and the development notebook's split definition."""

from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SPLIT_NAMES = ("train", "val", "test")
DUPLICATES_FILENAME = "ISIC_2020_Training_Duplicates.csv"


def normalize_frame(frame, name):
    """Validate metadata without changing row order or removing examples."""
    required = {"image_name", "patient_id", "lesion_id", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    frame = frame.copy()
    if frame.empty:
        raise ValueError(f"{name}: empty manifest")
    for column in ("image_name", "patient_id", "lesion_id"):
        if frame[column].isna().any():
            raise ValueError(f"{name}: missing {column}")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{name}: blank {column}")
    frame["image_name"] = frame.image_name.map(lambda s: Path(s).stem)
    if frame.image_name.eq("").any() or frame.image_name.duplicated().any():
        raise ValueError(f"{name}: empty or duplicate normalized image IDs")
    if frame.target.isna().any() or not frame.target.isin([0, 1]).all():
        raise ValueError(f"{name}: target must contain only binary labels 0 and 1")
    if set(frame.target) != {0, 1}:
        raise ValueError(f"{name}: both classes must be represented")
    frame["target"] = frame.target.astype(int)
    if (frame.groupby("lesion_id").target.nunique() > 1).any():
        raise ValueError(f"{name}: conflicting labels within a lesion")
    return frame


def validate_splits(frames, duplicates):
    """Reject image, patient, lesion and known-duplicate leakage in all partitions."""
    if set(frames) != set(SPLIT_NAMES):
        raise ValueError("All three manifests (train, val, test) are required")
    clean = {name: normalize_frame(frames[name], name) for name in SPLIT_NAMES}
    for a, b in combinations(SPLIT_NAMES, 2):
        for column in ("image_name", "patient_id", "lesion_id"):
            overlap = set(clean[a][column]) & set(clean[b][column])
            if overlap:
                raise ValueError(f"{column} overlap between {a} and {b}: {sorted(overlap)[:5]}")
    columns = ("image_name_1", "image_name_2")
    if not set(columns).issubset(duplicates.columns):
        raise ValueError(f"Duplicate metadata requires columns {columns}")
    owners = {row.image_name: (name, row.target)
              for name, frame in clean.items() for row in frame.itertuples()}
    for pair in duplicates.loc[:, list(columns)].itertuples(index=False, name=None):
        if any(pd.isna(value) or not str(value).strip() for value in pair):
            raise ValueError("Duplicate metadata contains missing image IDs")
        a, b = (Path(str(value).strip()).stem for value in pair)
        if a not in owners or b not in owners:
            raise ValueError(f"Duplicate pair references images absent from manifests: {a}, {b}")
        if owners[a][0] != owners[b][0]:
            raise ValueError(f"Known duplicate pair crosses splits: {a}, {b}")
        if owners[a][1] != owners[b][1]:
            raise ValueError(f"Known duplicate pair has conflicting labels: {a}, {b}")
    return clean


def load_splits(split_dir, duplicates_path=None):
    """Read all frozen manifests; never generate partitions during model runs."""
    split_dir = Path(split_dir)
    duplicates_path = Path(duplicates_path) if duplicates_path else split_dir.parent / DUPLICATES_FILENAME
    paths = {name: split_dir / f"{name}.csv" for name in SPLIT_NAMES}
    for path in (*paths.values(), duplicates_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required leakage-audit file not found: {path}")
    frames = {name: pd.read_csv(path, dtype={"image_name": str, "patient_id": str, "lesion_id": str})
              for name, path in paths.items()}
    return validate_splits(frames, pd.read_csv(duplicates_path, dtype=str))


def define_splits(metadata):
    """Reproduce development notebook cell 15, including candidate tie-breaking.

    Labels contribute only to this one-time cohort partition definition, not
    model or threshold selection. Input row order must match original metadata.
    """
    frame = normalize_frame(metadata, "metadata")
    best, best_score, best_seed = None, float("inf"), None
    overall_rate = frame.target.mean()
    for seed in range(100):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_val_idx, test_idx = next(outer.split(frame, groups=frame.patient_id))
        train_val, test = frame.iloc[train_val_idx], frame.iloc[test_idx]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
        train_idx, val_idx = next(inner.split(train_val, groups=train_val.patient_id))
        parts = [train_val.iloc[train_idx], train_val.iloc[val_idx], test]
        score = sum(abs(len(part) / len(frame) - proportion)
                    + 2 * abs(part.target.mean() - overall_rate)
                    for part, proportion in zip(parts, [0.70, 0.15, 0.15]))
        if score < best_score:
            best_score, best_seed = float(score), seed
            best = {name: part.copy() for name, part in zip(SPLIT_NAMES, parts)}
    return best, {"seed": best_seed, "score": best_score, "candidate_seeds": [0, 99]}
