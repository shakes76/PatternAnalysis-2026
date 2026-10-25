"""Patient-isolated 2D NIfTI loading and clearly labelled synthetic smoke data.

Manifest columns: image,mask,patient_id,split. Paths are relative to the CSV.
Use a recorded dataset grouping convention for patient IDs, validated against
source metadata; slice numbers alone do not identify patients. Obtain label
meanings from authoritative documentation. ``label_values`` fixes a global mapping,
with background 0 first. No mapping here asserts which class is the prostate.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


SPLITS = ("train", "val", "test")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read explicit split assignments and resolve paths relative to the CSV."""
    manifest = Path(path).expanduser().resolve()
    records = []
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "mask", "patient_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires image,mask,patient_id,split columns.")
        for row_number, row in enumerate(reader, start=2):
            record = {key: (row.get(key) or "").strip() for key in required}
            if not all(record.values()):
                raise ValueError(f"Manifest row {row_number} contains an empty required field.")
            if record["split"] not in SPLITS:
                raise ValueError(f"Manifest row {row_number}: split must be train, val, or test.")
            for key in ("image", "mask"):
                file_path = Path(record[key]).expanduser()
                if not file_path.is_absolute():
                    file_path = manifest.parent / file_path
                record[key] = str(file_path.resolve())
            records.append(record)
    if not records:
        raise ValueError("Manifest contains no data rows.")
    validate_patient_splits(records)
    return records


def validate_patient_splits(records: Iterable[dict[str, str]]) -> None:
    """Reject patient leakage and file reuse across split boundaries.

    This checks recorded identity and resolved paths, not duplicate file contents.
    Correct patient identity still depends on the supplied dataset metadata.
    """
    patients: dict[str, str] = {}
    files: dict[Path, str] = {}
    for record in records:
        split = record.get("split")
        patient = record.get("patient_id", "").strip()
        if split not in SPLITS or not patient:
            raise ValueError("Each record needs a patient_id and a train/val/test split.")
        if patient in patients and patients[patient] != split:
            raise ValueError("Patient leakage: one patient appears in multiple splits.")
        patients[patient] = split
        for key in ("image", "mask"):
            if not record.get(key):
                raise ValueError(f"Each record requires a {key} path.")
            path = Path(record[key]).expanduser().resolve()
            if path in files and files[path] != split:
                raise ValueError("File leakage: one image or mask appears in multiple splits.")
            files[path] = split


def validate_label_values(label_values: Sequence[int]) -> tuple[int, ...]:
    """Validate a global raw-label-to-channel mapping, with zero background."""
    values = tuple(label_values)
    if len(values) < 2 or values[0] != 0:
        raise ValueError("label_values must start with background 0 and include a foreground label.")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
           for value in values):
        raise ValueError("label_values must contain non-negative integer raw labels.")
    if len(values) != len(set(values)):
        raise ValueError("label_values must not contain duplicates.")
    return tuple(int(value) for value in values)


def map_labels(mask: np.ndarray, label_values: Sequence[int]) -> np.ndarray:
    """Map documented raw labels to fixed channels; never infer per-slice labels."""
    values = validate_label_values(label_values)
    if not np.isfinite(mask).all():
        raise ValueError("Mask contains non-finite values.")
    if not np.array_equal(mask, np.rint(mask)):
        raise ValueError("Mask contains non-integer labels.")
    observed = np.unique(mask)
    unknown = observed[~np.isin(observed, values)]
    if unknown.size:
        raise ValueError(f"Mask contains unknown raw labels: {unknown.tolist()}.")
    mapped = np.zeros(mask.shape, dtype=np.int64)
    for class_index, raw_label in enumerate(values):
        mapped[mask == raw_label] = class_index
    return mapped


def _two_dimensional(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError(f"{name} must be 2D or (H,W,1); received shape {array.shape}.")
    return array


def load_nifti_pair(record: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Load aligned slice arrays without selecting slices from a 3D volume."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Real NIfTI loading requires nibabel; install the project requirements.") from exc
    image_nii, mask_nii = nib.load(record["image"]), nib.load(record["mask"])
    if not (np.isfinite(image_nii.affine).all() and np.isfinite(mask_nii.affine).all()):
        raise ValueError("Image or mask affine contains non-finite values.")
    if not np.allclose(image_nii.affine, mask_nii.affine, rtol=1e-5, atol=1e-4):
        raise ValueError("Image and mask affines disagree; registration/orientation must be checked.")
    image_array = _two_dimensional(image_nii.get_fdata(dtype=np.float32), "Image")
    # Keep float64 here so anomalous/non-integral labels are not hidden by casting.
    mask_array = _two_dimensional(mask_nii.get_fdata(dtype=np.float64), "Mask")
    if image_array.shape != mask_array.shape:
        raise ValueError("Image and mask shapes disagree.")
    if not np.isfinite(image_array).all():
        raise ValueError("Image contains non-finite values.")
    if not np.isfinite(mask_array).all():
        raise ValueError("Mask contains non-finite values.")
    return image_array, mask_array


def normalize_slice(image: np.ndarray) -> np.ndarray:
    """Z-score nonzero slice voxels, clip to [-5,5], and keep zero padding zero.

    Each slice uses its own nonzero voxels, so no validation/test statistics are
    fitted into preprocessing. Empty or constant nonzero slices become zeros.
    """
    result = np.zeros(image.shape, dtype=np.float32)
    nonzero = image != 0
    if nonzero.any():
        foreground = image[nonzero].astype(np.float64)
        std = float(foreground.std())
        if std > 1e-8:
            result[nonzero] = np.clip((foreground - foreground.mean()) / std, -5, 5)
    return result


def resize_pair(image: torch.Tensor, mask: torch.Tensor, image_size: int,
                mode: str = "pad") -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a CHW image/HW mask into a square, preserving array aspect in pad mode.

    Image interpolation is bilinear, labels nearest-neighbour. Padding is zero
    after normalization (image) and background class zero (mask). This preserves
    array proportions, not independently verified physical spacing.
    """
    if mode not in ("pad", "stretch") or image_size < 8:
        raise ValueError("resize mode must be pad or stretch and image_size >= 8.")
    if mask.ndim != 2 or image.shape != (1, *mask.shape):
        raise ValueError("Expected an aligned one-channel CHW image and HW mask.")
    height, width = mask.shape
    if min(height, width) < 1:
        raise ValueError("Image and mask dimensions must be positive.")
    if mode == "pad":
        scale = image_size / max(height, width)
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
    else:
        size = (image_size, image_size)
    image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear",
                          align_corners=False)[0]
    mask = F.interpolate(mask[None, None].float(), size=size, mode="nearest")[0, 0].long()
    top, left = (image_size - size[0]) // 2, (image_size - size[1]) // 2
    padding = (left, image_size - size[1] - left, top, image_size - size[0] - top)
    return F.pad(image, padding, value=0), F.pad(mask, padding, value=0)


class HipMRIDataset(Dataset):
    """2D slices; the same explicit raw-label mapping must be used for all splits."""

    def __init__(self, manifest: str | Path, split: str, image_size: int = 128,
                 label_values: Sequence[int] = (0, 1), augment: bool = False,
                 resize_mode: str = "stretch"):
        if split not in SPLITS:
            raise ValueError("split must be train, val, or test.")
        if image_size < 8:
            raise ValueError("image_size must be at least 8 pixels.")
        if augment and split != "train":
            raise ValueError("Augmentation is allowed only for the training split.")
        if resize_mode not in ("pad", "stretch"):
            raise ValueError("resize_mode must be pad or stretch.")
        self.label_values = validate_label_values(label_values)
        self.records = [record for record in read_manifest(manifest) if record["split"] == split]
        if not self.records:
            raise ValueError(f"Manifest has no samples in the {split} split.")
        self.image_size = image_size
        self.augment = augment
        # Keep the historical dataset API default; new training CLI selects pad.
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image_array, raw_mask = load_nifti_pair(record)
        mapped_mask = map_labels(raw_mask, self.label_values)
        image, mask = resize_pair(torch.from_numpy(normalize_slice(image_array)).unsqueeze(0),
                                  torch.from_numpy(mapped_mask), self.image_size,
                                  self.resize_mode)
        if self.augment:
            # Geometrically identical flips for image and mask, using torch's seed.
            if torch.rand(()) < 0.5:
                image, mask = image.flip(-1), mask.flip(-1)
            if torch.rand(()) < 0.5:
                image, mask = image.flip(-2), mask.flip(-2)
        return {"image": image, "mask": mask, "patient_id": record["patient_id"],
                "image_path": record["image"]}


class SyntheticSegmentationDataset(Dataset):
    """Deterministic geometric shapes, ONLY for software smoke tests, not HipMRI."""

    def __init__(self, samples: int = 16, image_size: int = 64, seed: int = 0):
        if samples < 1 or image_size < 8:
            raise ValueError("Synthetic data requires samples >= 1 and image_size >= 8.")
        self.samples, self.image_size, self.seed = samples, image_size, seed
        self.label_values = (0, 1)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < self.samples:
            raise IndexError(index)
        # A pair of seed/index values avoids overlap between adjacent split seeds.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        side = self.image_size
        yy, xx = np.mgrid[:side, :side]
        cy, cx = rng.uniform(0.35, 0.65, size=2) * side
        ry, rx = rng.uniform(0.12, 0.25, size=2) * side
        mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.int64)
        noise = rng.normal(0, 0.12, size=(side, side)).astype(np.float32)
        image_array = normalize_slice(noise + mask.astype(np.float32))
        return {"image": torch.from_numpy(image_array).unsqueeze(0),
                "mask": torch.from_numpy(mask), "patient_id": f"synthetic-{self.seed}-{index}",
                "image_path": f"synthetic://seed-{self.seed}/sample-{index}"}
