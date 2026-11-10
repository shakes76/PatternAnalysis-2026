"""Validated ADNI manifest loading and deterministic image preprocessing."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path, PurePosixPath

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

SPLITS = ("train", "validation", "test")
FIELDS = (
    "relative_path",
    "original_split",
    "class_name",
    "metadata_label",
    "patient_id",
    "scan_id",
    "slice_index",
    "split",
    "sha256",
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Validate the entire manifest before filtering to a requested split."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Unexpected manifest schema")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty manifest")
    paths: set[str] = set()
    identities: set[tuple[str, str]] = set()
    groups: dict[tuple[str, str], str] = {}
    patient_classes: dict[str, str] = {}
    scan_patients: dict[str, str] = {}
    for row in rows:
        if set(row) != set(FIELDS) or any(not value for value in row.values()):
            raise ValueError("Incomplete manifest row")
        relative = PurePosixPath(row["relative_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in str(relative)
            or ":" in str(relative)
        ):
            raise ValueError("Unsafe relative image path")
        if row["split"] not in SPLITS or row["original_split"] not in ("train", "test"):
            raise ValueError("Unknown split")
        expected_label = {"AD": "2", "NC": "0"}.get(row["class_name"])
        if expected_label is None or row["metadata_label"] != expected_label:
            raise ValueError("Class and metadata label disagree")
        if len(row["sha256"]) != 64 or any(
            c not in "0123456789abcdef" for c in row["sha256"]
        ):
            raise ValueError("Invalid content hash")
        if not row["slice_index"].isdigit() or not row["scan_id"].isdigit():
            raise ValueError("Invalid scan or slice identity")
        expected_path = (
            f"AD_NC/{row['original_split']}/{row['class_name']}/"
            f"{row['scan_id']}_{row['slice_index']}.jpeg"
        )
        if row["relative_path"] != expected_path:
            raise ValueError("Path and recorded identity disagree")
        identity = (row["scan_id"], row["slice_index"])
        if row["relative_path"] in paths or identity in identities:
            raise ValueError("Duplicate path or scan/slice identity")
        paths.add(row["relative_path"])
        identities.add(identity)
        for field in ("patient_id", "scan_id", "sha256"):
            key = (field, row[field])
            previous = groups.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError(f"Cross-split overlap: {field}")
        if (
            patient_classes.setdefault(row["patient_id"], row["class_name"])
            != row["class_name"]
        ):
            raise ValueError("Inconsistent patient class")
        if (
            scan_patients.setdefault(row["scan_id"], row["patient_id"])
            != row["patient_id"]
        ):
            raise ValueError("Inconsistent scan/patient mapping")
    if {row["split"] for row in rows} != set(SPLITS):
        raise ValueError("One or more project splits are empty")
    return rows


class ADNISliceDataset(Dataset):
    """Return RGB tensors in [-1,1] and explicit class/slice metadata.

    No prompt construction or data augmentation is performed here. The manifest
    is validated as a whole, so selecting training rows cannot hide test leakage.
    """

    def __init__(
        self,
        root: Path,
        manifest: Path,
        split: str,
        image_size: int = 256,
        verify_hash: bool = False,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if image_size <= 0 or image_size % 8:
            raise ValueError("image_size must be a positive multiple of eight")
        self.root = root.resolve()
        self.rows = [row for row in read_manifest(manifest) if row["split"] == split]
        self.image_size = image_size
        # Checking each file against its recorded SHA-256 catches a changed or
        # substituted source image before it reaches training or evaluation.
        self.verify_hash = verify_hash

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image resolves outside dataset root")
        content = path.read_bytes()
        if self.verify_hash and hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Source image hash changed: {row['relative_path']}")
        with Image.open(io.BytesIO(content)) as source:
            image = source.convert("L")
            width, height = image.size
            side = max(width, height)
            left, top = (side - width) // 2, (side - height) // 2
            image = ImageOps.expand(
                image,
                border=(left, top, side - width - left, side - height - top),
                fill=0,
            )
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            tensor = (
                torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                .reshape(1, self.image_size, self.image_size)
                .repeat(3, 1, 1)
                .to(torch.float32)
                .div_(127.5)
                .sub_(1)
            )
        return {
            "pixel_values": tensor,
            "class_name": row["class_name"],
            "slice_index": int(row["slice_index"]),
            "relative_path": row["relative_path"],
        }


# A slice counts as near-empty when under 5% of its pixels exceed 10% intensity.
FOREGROUND_THRESHOLD = 0.1
NEAR_EMPTY_BELOW = 0.05


def foreground_share(pixel_values: torch.Tensor) -> torch.Tensor:
    """Share of pixels brighter than FOREGROUND_THRESHOLD, per image.

    Takes loader output in [-1, 1], shaped [3, H, W] or [B, 3, H, W].
    """
    intensity = (pixel_values[..., 0, :, :] + 1) / 2
    return (intensity > FOREGROUND_THRESHOLD).float().mean(dim=(-2, -1))


def defective_scans(rows: list[dict], shares: list[float]) -> set[str]:
    """Scan IDs with at least one near-empty slice.

    The whole scan is excluded, not just the empty slices, because a scan with
    blank slices failed acquisition or preprocessing and its other slices are
    suspect too. Decided from training data only, before any E2 run.
    """
    if len(rows) != len(shares):
        raise ValueError("One foreground share is needed per row")
    return {
        row["scan_id"] for row, share in zip(rows, shares) if share < NEAR_EMPTY_BELOW
    }
