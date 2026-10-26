"""Patient-disjoint cached HipMRI volumes and matched 2D slices.

The cache uses a shared resampled grid. Spacing and resulting geometric metrics
refer to that grid, not to native-resolution segmentation accuracy. Label indices
are declared by the manifest; this module assigns no anatomical meanings.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

try:
    import numpy as np
    import torch
    from torch.utils.data import Dataset
except ImportError:
    np = None
    torch = None
    Dataset = object


SPLIT_NAMES = ("train", "validation", "test")


def validate_patient_splits(
    splits: Mapping[str, Sequence[str]], expected_patients: Iterable[str] | None = None
) -> None:
    """Validate nonempty partitions without relying on disabled Python asserts."""
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise ValueError("Expected train, validation and test patient partitions.")
    seen = set()
    for name in SPLIT_NAMES:
        patients = splits[name]
        if (not isinstance(patients, list) or not patients
                or any(not isinstance(p, str) or not p for p in patients)):
            raise ValueError(f"Invalid or empty patient partition: {name}.")
        if len(set(patients)) != len(patients) or seen.intersection(patients):
            raise ValueError("Patient overlap or duplicate in partition assignments.")
        seen.update(patients)
    if expected_patients is not None and seen != set(expected_patients):
        raise ValueError("Patient partitions do not cover the expected patients exactly.")


def load_manifest(path: str | Path) -> dict:
    """Read and validate all partition assignments before selecting a split."""
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    splits = manifest.get("patient_splits")
    validate_patient_splits(splits)
    labels = manifest.get("label_ids")
    if (not isinstance(labels, list) or not labels or len(labels) > 256
            or any(type(label) is not int for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("label_ids must contain 1-256 distinct integer raw IDs.")
    if type(manifest.get("num_classes")) is not int or manifest["num_classes"] != len(labels):
        raise ValueError("num_classes must equal the number of declared label_ids.")
    expected_mapping = {str(raw): index for index, raw in enumerate(labels)}
    mapping = manifest.get("raw_label_to_index")
    if mapping is None and labels == list(range(len(labels))):
        mapping = expected_mapping
    if mapping != expected_mapping:
        raise ValueError("Explicit raw_label_to_index must follow the declared label_ids order.")
    manifest["raw_label_to_index"] = mapping
    shape = manifest.get("target_shape_dhw")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(s) is not int or s <= 0 for s in shape):
        raise ValueError("target_shape_dhw must contain three positive integers.")
    if not isinstance(manifest.get("audit_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", manifest["audit_sha256"]):
        raise ValueError("Manifest must identify its source audit by SHA-256.")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain at least one cache record.")
    partial = manifest.get("partial", False)
    if type(partial) is not bool or type(manifest.get("development_only", False)) is not bool:
        raise ValueError("partial and development_only must be booleans.")
    if partial != manifest.get("development_only", False):
        raise ValueError("Partial manifests must explicitly be development_only.")
    seen_pairs, seen_paths, observed_patients = set(), set(), set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each cache record must be an object.")
        for key in ("patient", "visit", "split", "cache_path"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Missing or invalid record field: {key}.")
        patient, split = record["patient"], record["split"]
        if split not in SPLIT_NAMES or patient not in splits[split]:
            raise ValueError("Record split disagrees with patient-level partition assignment.")
        if partial and split != "train":
            raise ValueError("A partial development manifest may contain training records only.")
        pair = (patient, record["visit"])
        relative = Path(record["cache_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npz":
            raise ValueError("cache_path must be a safe relative .npz path.")
        resolved = (path.parent / relative).resolve()
        if not resolved.is_relative_to(path.parent):
            raise ValueError("cache_path escapes the manifest directory.")
        if pair in seen_pairs or resolved in seen_paths:
            raise ValueError("Duplicate patient/visit or duplicate cache file in manifest.")
        seen_pairs.add(pair)
        seen_paths.add(resolved)
        observed_patients.add(patient)
    if not partial:
        validate_patient_splits(splits, observed_patients)
    manifest.setdefault("partial", False)
    manifest.setdefault("development_only", False)
    return manifest


class HipMRIVolumeDataset(Dataset):
    """Load image (1,D,H,W), mask (D,H,W), and spacing (D,H,W millimetres)."""

    def __init__(self, manifest_path: str | Path, split: str, augment: bool = False):
        if torch is None or np is None:
            raise ImportError("HipMRI datasets require PyTorch and NumPy.")
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {split}.")
        if augment and split != "train":
            raise ValueError("Augmentation is permitted for the training split only.")
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.records = [record for record in self.manifest["records"] if record["split"] == split]
        self.split, self.augment = split, bool(augment)
        self.num_classes = self.manifest["num_classes"]
        self.label_ids = self.manifest["label_ids"]
        self.target_shape_dhw = tuple(self.manifest["target_shape_dhw"])
        self.development_only = self.manifest["development_only"]

    def __len__(self) -> int:
        return len(self.records)

    def _load_volume(self, index: int) -> dict:
        record = self.records[index]
        with np.load(self.manifest_path.parent / record["cache_path"], allow_pickle=False) as cache:
            image, mask, spacing = cache["image"], cache["mask"], cache["spacing"]
        if image.dtype != np.float32 or mask.dtype != np.uint8:
            raise ValueError("Cache image/mask must be float32/uint8 respectively.")
        if image.shape != self.target_shape_dhw or mask.shape != image.shape:
            raise ValueError("Cache shape differs from the manifest target grid.")
        if not np.isfinite(image).all() or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Cache contains invalid image values or spacing.")
        if int(mask.max()) >= self.num_classes:
            raise ValueError("Cached mask contains an undeclared class index.")
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "patient": record["patient"], "visit": record["visit"],
            "spacing": torch.from_numpy(np.asarray(spacing, dtype=np.float32).copy()),
        }

    def _augment(self, sample: dict) -> dict:
        if self.augment:
            for dimension in (-2, -1):
                if bool(torch.rand(()) < 0.5):
                    sample["image"] = torch.flip(sample["image"], [dimension])
                    sample["mask"] = torch.flip(sample["mask"], [dimension])
        return sample

    def __getitem__(self, index: int) -> dict:
        return self._augment(self._load_volume(index))


class HipMRISliceDataset(HipMRIVolumeDataset):
    """Expose every axial slice from the same volume records and patient splits.

    records remains the list of volume records; len(self) counts slices. Returned
    spacing retains the shared D,H,W convention, and slice_index identifies D.

    Unaugmented volume tensors use a least-recently-used cache, bounded by
    volume_cache_bytes PER DATASET INSTANCE PER DATA LOADER WORKER (2 GiB by
    default). Multiple workers/datasets therefore multiply the total allowance;
    num_workers=0 keeps one cache for this dataset. The allowance counts retained
    image, long-mask and spacing tensor storage, not temporary decompression,
    returned slice copies, Python overhead, or other datasets. Oversized volumes
    are served without retention. A cache belongs only to this instance's split.
    """

    def __init__(
        self, manifest_path: str | Path, split: str, augment: bool = False,
        volume_cache_bytes: int = 2 * 1024 ** 3,
    ):
        super().__init__(manifest_path, split, augment)
        if type(volume_cache_bytes) is not int or volume_cache_bytes < 0:
            raise ValueError("volume_cache_bytes must be a nonnegative integer.")
        self.volume_cache_bytes = volume_cache_bytes
        self._volume_cache = OrderedDict()
        self._volume_cache_size_bytes = 0
        self._slice_indices = [
            (volume_index, depth)
            for volume_index in range(len(self.records))
            for depth in range(self.target_shape_dhw[0])
        ]

    def __len__(self) -> int:
        return len(self._slice_indices)

    @staticmethod
    def _volume_tensor_bytes(volume: dict) -> int:
        return sum(volume[key].numel() * volume[key].element_size()
                   for key in ("image", "mask", "spacing"))

    def _get_volume(self, volume_index: int) -> dict:
        if volume_index in self._volume_cache:
            self._volume_cache.move_to_end(volume_index)
            return self._volume_cache[volume_index]
        volume = self._load_volume(volume_index)
        size_bytes = self._volume_tensor_bytes(volume)
        if size_bytes <= self.volume_cache_bytes:
            while self._volume_cache_size_bytes + size_bytes > self.volume_cache_bytes:
                _, evicted = self._volume_cache.popitem(last=False)
                self._volume_cache_size_bytes -= self._volume_tensor_bytes(evicted)
            self._volume_cache[volume_index] = volume
            self._volume_cache_size_bytes += size_bytes
        return volume

    def __getitem__(self, index: int) -> dict:
        volume_index, depth = self._slice_indices[index]
        volume = self._get_volume(volume_index)
        # Copy only the requested slice. Neither augmentation nor an in-place
        # transform by a caller can change the retained, unaugmented volume.
        sample = {
            "image": volume["image"][:, depth].clone(),
            "mask": volume["mask"][depth].clone(),
            "patient": volume["patient"], "visit": volume["visit"],
            "spacing": volume["spacing"].clone(), "slice_index": depth,
        }
        return self._augment(sample)
