"""Manifest-based ADNI JPEG loading for COMP3710.

Uses the finalized CSVs produced by prepare_patient_split.py; it NEVER derives a
split from source directories or changes assignments. The source directory name
`test` is provenance only and may legitimately occur in the new training CSV.

Initial experiment design: one greyscale channel; aspect-preserving resize into
224 x 224, black padding, and fixed (pixel / 255 - 0.5) / 0.5 scaling. No fitted
normalization statistics and no augmentation by default. An optional small
training-only rotation is available for a later validation-selected experiment.

The model implementation belongs in modules.py. Pillow/NumPy below are used only
for data preparation, not to implement the neural network.

Development assistance: ChatGPT assisted with this draft and synthetic tests.
The submitting student must review, understand, test and disclose that assistance.
API references (not claims of latest versions):
https://docs.pytorch.org/docs/stable/data.html
https://pillow.readthedocs.io/en/stable/reference/Image.html
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

PARTITIONS = ("train", "validation", "test")
COLUMNS = ("image_path", "subject_id", "scan_id", "slice_index", "source_split",
           "source_class", "metadata_label", "label", "split")
LABEL_MAP = {"NC": 0, "AD": 1}
SOURCE_SIZE = (256, 240)  # Pillow convention: width, height; confirmed by audit.
INPUT_SIZE = 224
FINAL_STATUS = "FINALIZED_UNDER_USER_CONFIRMED_PATIENT_RULE_AND_TUTOR_APPROVAL"


class DataError(ValueError):
    """Validation error whose message contains no patient identifiers."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Summary contains a duplicate JSON key.")
        result[key] = value
    return result


@dataclass(frozen=True)
class SliceRecord:
    image_path: str
    subject_id: str
    scan_id: str
    slice_index: int
    source_split: str
    source_class: str
    metadata_label: int
    label: int
    split: str


@dataclass(frozen=True)
class SplitBundle:
    root: Path
    records: dict[str, tuple[SliceRecord, ...]]
    summary: dict[str, Any]
    manifest_hashes: dict[str, str]


def _parse_row(row: dict[str, str], split: str) -> SliceRecord:
    require(set(row) == set(COLUMNS) and all(isinstance(row[k], str) and row[k]
                                           for k in COLUMNS),
            "A CSV row has missing, empty, or extra fields.")
    require(row["split"] == split, "A CSV row has the wrong final split.")
    require(row["source_split"] in ("train", "test"), "Invalid original split.")
    require(row["source_class"] in LABEL_MAP, "Invalid source class.")
    require(row["label"] in ("0", "1"), "Invalid binary model label.")
    require(row["metadata_label"] in ("0", "2"), "Invalid original metadata label.")
    require(row["slice_index"].isdigit(), "Invalid slice index.")
    label = int(row["label"])
    require(label == LABEL_MAP[row["source_class"]], "Class and binary label disagree.")
    require(int(row["metadata_label"]) == 2 * label, "Metadata and binary label disagree.")
    text = row["image_path"]
    path = PurePosixPath(text)
    require("\\" not in text and ":" not in text and not path.is_absolute()
            and ".." not in path.parts and len(path.parts) == 4
            and path.as_posix() == text,
            "Image path is not a safe canonical relative path.")
    require(path.parts[:3] == ("AD_NC", row["source_split"], row["source_class"]),
            "Image path disagrees with source-folder provenance.")
    require(path.suffix.lower() in (".jpeg", ".jpg"), "Expected a JPEG path.")
    prefix, sep, suffix = path.stem.rpartition("_")
    require(bool(sep) and suffix.isdigit() and prefix == row["scan_id"]
            and int(suffix) == int(row["slice_index"]),
            "Image filename and scan/slice columns disagree.")
    return SliceRecord(text, row["subject_id"], row["scan_id"], int(row["slice_index"]),
                       row["source_split"], row["source_class"],
                       int(row["metadata_label"]), label, split)


def counts_for_records(records: tuple[SliceRecord, ...]) -> dict[str, Any]:
    """Use the prior split-summary field names for a direct consistency check."""
    return {
        "candidate_subjects": len({r.subject_id for r in records}),
        "scans": len({r.scan_id for r in records}),
        "images": len(records),
        "by_class": {
            name: {
                "candidate_subjects": len({r.subject_id for r in records if r.label == label}),
                "scans": len({r.scan_id for r in records if r.label == label}),
                "images": sum(r.label == label for r in records),
            } for name, label in LABEL_MAP.items()
        },
    }


def load_finalized_splits(root: str | Path, splits_dir: str | Path) -> SplitBundle:
    """Verify small manifests against their summary; do not decode/hash images.

    The previous audit established image integrity. This verifies that the exact
    finalized assignments are used. Checking manifest rows is not model evaluation
    on the test partition. No patient IDs appear in the returned aggregate counts.
    """
    root, splits_dir = Path(root).resolve(), Path(splits_dir).resolve()
    require(root.is_dir() and splits_dir.is_dir(), "Dataset or split directory is missing.")
    try:
        summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8-sig"),
                             object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataError("Cannot read the finalized split summary.") from exc
    require(isinstance(summary, dict) and summary.get("format_version") == 1,
            "Unsupported split-summary format.")
    require(summary.get("status") == FINAL_STATUS, "Splits have not been finalized.")
    for key in ("mapping_structural_checks_passed", "patient_identifier_semantics_confirmed_by_user",
                "tutor_split_change_approved_as_reported_by_user", "zero_group_overlap_verified",
                "splits_finalized"):
        require(summary.get(key) is True, "A required split-finalization check is missing.")
    require(summary.get("new_pairwise_candidate_subject_overlap") == 0,
            "Split summary reports overlapping subject groups.")
    require(summary.get("model_label_map") == LABEL_MAP
            and summary.get("metadata_to_model_labels") == {"0": 0, "2": 1},
            "Split summary has an unexpected label encoding.")
    require(isinstance(summary.get("manifest_sha256"), dict)
            and isinstance(summary.get("proposed_or_final_counts"), dict),
            "Split summary lacks manifest hashes or counts.")

    records: dict[str, tuple[SliceRecord, ...]] = {}
    hashes: dict[str, str] = {}
    path_seen: set[str] = set()
    subject_info: dict[str, tuple[str, int]] = {}
    scan_info: dict[str, tuple[str, str, int]] = {}
    scan_slices: dict[str, set[int]] = {}
    for split in PARTITIONS:
        name = f"{split}.csv"
        try:
            raw = (splits_dir / name).read_bytes()
        except OSError as exc:
            raise DataError("A finalized CSV manifest could not be read.") from exc
        digest = sha256_bytes(raw)
        require(digest == summary["manifest_sha256"].get(name),
                "A manifest SHA-256 differs from the finalized summary; do not edit/re-save CSVs.")
        hashes[name] = digest
        try:
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
            require(tuple(reader.fieldnames or ()) == COLUMNS, "Unexpected manifest CSV columns.")
            rows = tuple(_parse_row(row, split) for row in reader)
        except (UnicodeError, csv.Error) as exc:
            raise DataError("A manifest is not valid UTF-8 CSV.") from exc
        require(bool(rows), "A split is empty.")
        require({r.label for r in rows} == {0, 1}, "A split is missing a diagnostic class.")
        for row in rows:
            require(row.image_path not in path_seen, "An image path is repeated across manifests.")
            path_seen.add(row.image_path)
            subject = (row.split, row.label)
            require(row.subject_id not in subject_info or subject_info[row.subject_id] == subject,
                    "A subject spans splits or has conflicting labels.")
            subject_info[row.subject_id] = subject
            scan = (row.split, row.subject_id, row.label)
            require(row.scan_id not in scan_info or scan_info[row.scan_id] == scan,
                    "A scan spans splits, subjects or labels.")
            scan_info[row.scan_id] = scan
            indices = scan_slices.setdefault(row.scan_id, set())
            require(row.slice_index not in indices, "A scan repeats a slice index.")
            indices.add(row.slice_index)
        require(counts_for_records(rows) == summary["proposed_or_final_counts"].get(split),
                "Manifest counts disagree with the finalized summary.")
        records[split] = rows
    require(all(len(indices) == 20 for indices in scan_slices.values()),
            "This audited dataset requires exactly 20 distinct slices per scan.")
    evidence = summary.get("evidence", {})
    require(len(path_seen) == evidence.get("images")
            and len(scan_info) == evidence.get("scan_prefixes")
            and len(subject_info) == evidence.get("candidate_subjects"),
            "Manifest totals differ from the split audit.")
    return SplitBundle(root, records, summary, hashes)


def preprocessing_spec(rotation_degrees: float = 0.0) -> dict[str, Any]:
    return {
        "source": {"mode": "L", "width": 256, "height": 240},
        "input_channels": 1,
        "output_height": INPUT_SIZE, "output_width": INPUT_SIZE,
        "resize": "Pillow bilinear, preserve aspect ratio; 256x240 -> 224x210",
        "padding": "7 black pixels at the top and bottom, before normalization",
        "scale": "float32: (pixel / 255.0 - 0.5) / 0.5; nominal range [-1,1]",
        "fitted_dataset_statistics": False,
        "training_rotation_degrees": float(rotation_degrees),
        "validation_test_rotation_degrees": 0.0,
        "horizontal_vertical_flips": False,
        "cropping": False,
    }


def image_to_tensor(image: Image.Image, angle: float = 0.0) -> torch.Tensor:
    """Prepare an audited L-mode JPEG; preserve geometry rather than stretching."""
    require(image.mode == "L" and image.size == SOURCE_SIZE,
            "An image differs from the audited greyscale dimensions.")
    # Integer dimensions chosen from the audited source aspect ratio.
    resized = image.resize((224, 210), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (INPUT_SIZE, INPUT_SIZE), 0)
    canvas.paste(resized, (0, 7))
    if angle:
        canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR,
                               expand=False, fillcolor=0)
    array = np.array(canvas, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor.div_(255.0).sub_(0.5).div_(0.5)


class ADNISliceDataset(Dataset):
    """Return an image plus labels and grouping metadata for later aggregation.

    DataLoader collates each item into a dict. Use batch['image'] and batch['label']
    for the loss. IDs and relative paths remain private and are needed to average
    slice probabilities within scans and then scans within patients later.
    """

    def __init__(self, bundle: SplitBundle, split: str, *,
                 rotation_degrees: float = 0.0, allow_test: bool = False) -> None:
        require(split in PARTITIONS, "Unknown split name.")
        require(split != "test" or allow_test,
                "Final test image access is disabled; enable explicitly only for final evaluation.")
        require(0.0 <= rotation_degrees <= 5.0, "Training rotation must be between 0 and 5 degrees.")
        require(split == "train" or rotation_degrees == 0.0,
                "Random augmentation is not allowed for validation or test.")
        self.root = bundle.root
        self.records = bundle.records[split]
        self.split = split
        self.rotation_degrees = float(rotation_degrees)

    def __len__(self) -> int:
        return len(self.records)

    def source_path(self, index: int) -> Path:
        path = (self.root / self.records[index].image_path).resolve()
        require(self.root in path.parents, "An image resolves outside the dataset root.")
        return path

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        angle = 0.0
        if self.rotation_degrees:
            angle = (2.0 * torch.rand(()).item() - 1.0) * self.rotation_degrees
        try:
            with Image.open(self.source_path(index)) as image:
                image.load()
                tensor = image_to_tensor(image, angle)
        except (OSError, ValueError) as exc:
            if isinstance(exc, DataError):
                raise
            raise DataError("An image could not be loaded; source files may have changed.") from exc
        return {
            "image": tensor, "label": row.label, "subject_id": row.subject_id,
            "scan_id": row.scan_id, "slice_index": row.slice_index,
            "image_path": row.image_path, "split": row.split,
        }


def seed_worker(worker_id: int) -> None:
    """Top-level function so DataLoader can pickle it on Windows."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset: ADNISliceDataset, *, batch_size: int = 16,
                num_workers: int = 0, seed: int = 42,
                shuffle: bool | None = None, pin_memory: bool = False) -> DataLoader:
    """Windows-safe defaults; keep the last partial batch and every sample.

    For num_workers > 0 call from a script protected by if __name__ == '__main__'.
    Reproducibility is scoped to the same settings/software, not every platform.
    """
    require(batch_size > 0 and num_workers >= 0, "Invalid batch size or worker count.")
    if shuffle is None:
        shuffle = dataset.split == "train"
    require(dataset.split == "train" or not shuffle, "Do not shuffle evaluation loaders.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=False, generator=generator,
                      worker_init_fn=seed_worker, pin_memory=pin_memory)
