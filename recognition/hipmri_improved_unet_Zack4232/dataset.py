"""Load and validate the course HipMRI 2D segmentation dataset."""

import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_DATA_ROOT = Path(
    "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
)

LABEL_NAMES = (
    "Background",
    "Body",
    "Bones",
    "Bladder",
    "Rectum",
    "Prostate",
)

FILE_PATTERN = re.compile(
    r"^(case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$"
)


@dataclass(frozen=True)
class SliceRecord:
    """Store file paths and identifiers for one paired slice."""

    image_path: Path
    mask_path: Path
    case_id: int
    week: int
    slice_id: int


def index_files(folder, expected_prefix):
    """Map (case, week, slice) keys to paths without relying on file order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Data directory not found: {folder}")

    records = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue

        match = FILE_PATTERN.fullmatch(path.name)
        if match is None or match.group(1) != expected_prefix:
            raise ValueError(f"Unexpected file: {path}")

        key = tuple(int(value) for value in match.groups()[1:])
        if key in records:
            raise ValueError(f"Duplicate sample key {key} in {folder}")

        records[key] = path

    if not records:
        raise ValueError(f"No samples found in {folder}")

    return records


class HipMRIDataset(Dataset):
    """Load paired slices from one supplied data split."""

    def __init__(self, data_root=DEFAULT_DATA_ROOT, split="train"):
        if split not in {"train", "validate", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.data_root = Path(data_root)
        self.split = split

        images = index_files(
            self.data_root / f"keras_slices_{split}", "case"
        )
        masks = index_files(
            self.data_root / f"keras_slices_seg_{split}", "seg"
        )

        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())

        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired files in {split}: "
                f"{len(missing_masks)} missing masks "
                f"(examples: {missing_masks[:5]}); "
                f"{len(missing_images)} missing images "
                f"(examples: {missing_images[:5]})"
            )

        self.samples = [
            SliceRecord(
                image_path=images[key],
                mask_path=masks[key],
                case_id=key[0],
                week=key[1],
                slice_id=key[2],
            )
            for key in sorted(images)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """Load, validate, and normalize one paired slice."""
        sample = self.samples[index]
        image = nib.load(sample.image_path)
        mask = nib.load(sample.mask_path)

        x = image.get_fdata(dtype=np.float32)
        y = mask.get_fdata(dtype=np.float32)

        if x.ndim != 2 or x.shape != y.shape:
            raise ValueError(
                f"Invalid image/mask shapes: {x.shape}, {y.shape}; "
                f"{sample.image_path.name}"
            )
        if min(x.shape) < 16 or any(size % 16 for size in x.shape):
            raise ValueError(
                f"Image dimensions must be positive multiples of 16: "
                f"{x.shape}; {sample.image_path.name}"
            )
        if not np.allclose(
            image.affine, mask.affine, rtol=0, atol=1e-5
        ):
            raise ValueError(f"Affine mismatch: {sample.image_path.name}")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite values: {sample.image_path.name}")
        if not np.equal(y, np.round(y)).all():
            raise ValueError(f"Non-integer labels: {sample.mask_path.name}")
        if not np.isin(y, np.arange(len(LABEL_NAMES))).all():
            raise ValueError(f"Unknown labels: {sample.mask_path.name}")

        # Compute statistics from this image only, without using its mask.
        mean = float(x.mean(dtype=np.float64))
        std = float(x.std(dtype=np.float64))
        x = (x - mean) / max(std, 1e-8)

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(x, dtype=np.float32)
        ).unsqueeze(0)
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(y, dtype=np.int64)
        )

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_id": sample.case_id,
            "week": sample.week,
            "slice_id": sample.slice_id,
        }
