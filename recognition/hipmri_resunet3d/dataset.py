"""Leakage-safe HipMRI Nifti loading, preprocessing, and patient splits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Case:
    case_id: str
    patient_id: str
    image: Path
    mask: Path


def manifest_sha256(path: str | Path) -> str:
    """Fingerprint the exact case, patient, and file mapping used by a run."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest_provenance(provenance: dict, path: str | Path) -> None:
    """Reject a moved or changed manifest before opening evaluation cases.

    Older saved runs lack a fingerprint, so they retain their original path
    check. New runs additionally lock the manifest contents.
    """
    if Path(provenance["manifest"]).resolve() != Path(path).resolve():
        raise ValueError("Manifest path differs from the one recorded during training")
    recorded_hash = provenance.get("manifest_sha256")
    if recorded_hash is not None and manifest_sha256(path) != recorded_hash:
        raise ValueError("Manifest contents differ from the one recorded during training")


def read_manifest(path: str | Path) -> list[Case]:
    """Read an explicit case-to-patient map; never guess patient IDs."""
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty")
    required = {"case_id", "patient_id", "image", "mask"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest needs columns {sorted(required)}")
    cases = []
    seen_case_ids = set()
    seen_images = set()
    seen_masks = set()
    for row in rows:
        case_id = row["case_id"].strip()
        patient_id = row["patient_id"].strip()
        if not case_id or not patient_id or case_id in seen_case_ids:
            raise ValueError(f"Missing or repeated case/patient ID: {row}")
        seen_case_ids.add(case_id)
        image = (path.parent / row["image"]).resolve()
        mask = (path.parent / row["mask"]).resolve()
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing image/mask for {case_id}: {image}, {mask}")
        if image in seen_images or mask in seen_masks:
            raise ValueError(f"Repeated image or mask path in manifest: {case_id}")
        seen_images.add(image)
        seen_masks.add(mask)
        cases.append(Case(case_id, patient_id, image, mask))
    return cases


def split_by_patient(
    cases: list[Case], seed: int = 3710, train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict[str, list[Case]]:
    """Assign whole patients to train/validation/test with a fixed seed."""
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ValueError("Invalid split fractions")
    patients = sorted({case.patient_id for case in cases})
    if len(patients) < 3:
        raise ValueError("At least three patients are required for three disjoint splits")
    random.Random(seed).shuffle(patients)
    n_train = max(1, round(len(patients) * train_fraction))
    n_val = max(1, round(len(patients) * val_fraction))
    if n_train + n_val >= len(patients):
        n_train = len(patients) - 2
        n_val = 1
    patient_sets = {
        "train": set(patients[:n_train]),
        "val": set(patients[n_train:n_train + n_val]),
        "test": set(patients[n_train + n_val:]),
    }
    return {name: [case for case in cases if case.patient_id in ids]
            for name, ids in patient_sets.items()}


def save_split(split: dict[str, list[Case]], path: str | Path) -> None:
    data = {name: [case.case_id for case in cases] for name, cases in split.items()}
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_split(cases: list[Case], path: str | Path) -> dict[str, list[Case]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup = {case.case_id: case for case in cases}
    if set(data) != {"train", "val", "test"}:
        raise ValueError("Split JSON must contain train, val, test")
    if set(sum(data.values(), [])) != set(lookup):
        raise ValueError("Split and manifest contain different cases")
    if len(sum(data.values(), [])) != len(lookup):
        raise ValueError("Repeated case in split")
    split = {name: [lookup[case_id] for case_id in ids] for name, ids in data.items()}
    sets = [{case.patient_id for case in split[name]} for name in ("train", "val", "test")]
    groups_missing = any(not group for group in sets)
    patients_overlap = any(
        sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)
    )
    if groups_missing or patients_overlap:
        raise ValueError("Empty split or patient leakage")
    return split


def _read_canonical_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 3D Nifti volume, allowing one trailing singleton dimension."""
    nifti = nib.as_closest_canonical(nib.load(str(path)))
    volume = np.asarray(nifti.dataobj, dtype=np.float32)
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D Nifti volume: {path}")
    return volume, nifti.affine


def load_native_anatomy(case: Case) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return original-resolution labels and voxel spacing in [D,H,W] order."""
    labels, affine = _read_canonical_volume(case.mask)
    if not np.allclose(labels, np.rint(labels), atol=1e-3):
        raise ValueError(f"Non-integer segmentation mask in {case.case_id}")
    anatomy = np.transpose(np.rint(labels).astype(np.int16), (2, 1, 0)).copy()
    spacing_dhw = tuple(float(value) for value in nib.affines.voxel_sizes(affine)[::-1])
    return anatomy, spacing_dhw


def _normalise_image(image: np.ndarray, case_id: str) -> np.ndarray:
    """Clip extreme intensities and standardize using nonzero tissue voxels."""
    tissue = image[np.abs(image) > 1e-6]
    if tissue.size < 100:
        raise ValueError(f"Too few nonzero image voxels in {case_id}")
    lower, upper = np.percentile(tissue, (0.5, 99.5))
    clipped = np.clip(image, lower, upper)
    return ((clipped - float(tissue.mean())) /
            max(float(tissue.std()), 1e-6)).astype(np.float32)


def central_xy_slices(
    shape_xy: tuple[int, int], crop_xy: int | None
) -> tuple[slice, slice]:
    """Return matching centered X/Y index ranges for an optional fixed crop."""
    if crop_xy is None:
        return slice(0, shape_xy[0]), slice(0, shape_xy[1])
    if crop_xy <= 0 or crop_xy % 2 or crop_xy > min(shape_xy):
        raise ValueError(
            f"crop_xy must be a positive even integer no larger than both "
            f"image axes {shape_xy}; got {crop_xy}"
        )
    x0 = (shape_xy[0] - crop_xy) // 2
    y0 = (shape_xy[1] - crop_xy) // 2
    return slice(x0, x0 + crop_xy), slice(y0, y0 + crop_xy)


def _resize_to_network_grid(
    image: np.ndarray, labels: np.ndarray, size: tuple[int, int, int],
    crop_xy: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop in canonical X/Y, then convert [X,Y,Z] to network [D,H,W]."""
    x_range, y_range = central_xy_slices(image.shape[:2], crop_xy)
    image = image[x_range, y_range, :]
    labels = labels[x_range, y_range, :]
    image_dhw = np.transpose(image, (2, 1, 0)).copy()
    labels_dhw = np.transpose(np.rint(labels), (2, 1, 0)).copy()

    image_tensor = torch.from_numpy(image_dhw)[None, None]
    labels_tensor = torch.from_numpy(labels_dhw)[None, None]
    image_tensor = F.interpolate(
        image_tensor, size=size, mode="trilinear", align_corners=False
    )[0]
    labels_tensor = F.interpolate(labels_tensor, size=size, mode="nearest")[0, 0].long()
    return image_tensor, labels_tensor


def load_volume(
    case: Case, size: tuple[int, int, int], prostate_label: int,
    crop_xy: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the image, binary prostate target, and anatomical label IDs."""
    image, image_affine = _read_canonical_volume(case.image)
    labels, mask_affine = _read_canonical_volume(case.mask)
    if image.shape != labels.shape:
        raise ValueError(f"Image/mask shape mismatch for {case.case_id}")
    if not np.allclose(image_affine, mask_affine, atol=1e-3):
        raise ValueError(f"Image/mask affine mismatch for {case.case_id}")
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError(f"Non-finite voxel in {case.case_id}")
    if not np.allclose(labels, np.rint(labels), atol=1e-3):
        raise ValueError(f"Non-integer segmentation mask in {case.case_id}")
    if not np.any(labels == prostate_label):
        raise ValueError(
            f"Prostate label {prostate_label} absent in {case.case_id}; audit label IDs"
        )
    image = _normalise_image(image, case.case_id)
    image_tensor, anatomy = _resize_to_network_grid(image, labels, size, crop_xy)
    prostate_mask = (anatomy == prostate_label).long()
    if not prostate_mask.any():
        raise ValueError(
            f"Crop or resampling removed prostate target in {case.case_id}; "
            "check crop size and grid resolution"
        )
    return image_tensor, prostate_mask, anatomy


class VolumeDataset(Dataset):
    def __init__(
        self, cases: list[Case], size: tuple[int, int, int], prostate_label: int,
        crop_xy: int | None = None,
    ):
        self.cases = cases
        self.size = size
        self.prostate_label = prostate_label
        self.crop_xy = crop_xy

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        case = self.cases[index]
        image, mask, anatomy = load_volume(
            case, self.size, self.prostate_label, self.crop_xy
        )
        return {"image": image, "mask": mask, "anatomy": anatomy, "case_id": case.case_id,
                "patient_id": case.patient_id}


def mirror_left_right(sample: dict) -> dict:
    """Mirror image and labels together across the canonical left-right axis."""
    mirrored = sample.copy()
    for key in ("image", "mask", "anatomy"):
        if key in mirrored:
            mirrored[key] = torch.flip(mirrored[key], dims=(-1,))
    return mirrored


class TrainingVolumeDataset(Dataset):
    """Apply random training augmentation without changing validation data."""

    def __init__(self, volumes: VolumeDataset):
        self.volumes = volumes

    def __len__(self) -> int:
        return len(self.volumes)

    def __getitem__(self, index: int) -> dict:
        sample = self.volumes[index]
        if torch.rand(()) < 0.5:
            sample = mirror_left_right(sample)
        return sample


class SliceDataset(Dataset):
    """Materialize resized training volumes once for efficient 2D sampling."""

    def __init__(self, volume_dataset: VolumeDataset, augment: bool = False):
        self.volumes = []
        self.augment = augment
        for i in range(len(volume_dataset)):
            sample = volume_dataset[i]
            self.volumes.append({
                "image": sample["image"],
                "mask": sample["mask"],
                "case_id": sample["case_id"],
            })
        self.index = [(i, z) for i, vol in enumerate(self.volumes)
                      for z in range(vol["mask"].shape[0])]
        self.weights = [5.0 if self.volumes[i]["mask"][z].any() else 1.0
                        for i, z in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        i, z = self.index[index]
        vol = self.volumes[i]
        sample = {"image": vol["image"][:, z], "mask": vol["mask"][z],
                  "case_id": vol["case_id"]}
        if self.augment and torch.rand(()) < 0.5:
            sample = mirror_left_right(sample)
        return sample
