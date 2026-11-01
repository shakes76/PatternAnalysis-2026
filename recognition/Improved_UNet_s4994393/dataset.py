"""NIfTI pairing, patient-level splits and foreground-aware patches."""

from pathlib import Path
import random
import hashlib
import json
from functools import lru_cache

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def discover_pairs(root):
    """Return sorted MRI/mask path pairs, rejecting missing masks or an empty root.

    HipMRI pairs share the patient/week prefix; only their suffixes differ.
    """
    root = Path(root)
    pairs = []
    for image in sorted((root / "semantic_MRs").glob("*_LFOV.nii.gz")):
        mask = (
            root / "semantic_labels_only" / image.name.replace("_LFOV.nii.gz", "_SEMANTIC.nii.gz")
        )
        if not mask.is_file():
            raise FileNotFoundError(mask)
        pairs.append((str(image), str(mask)))
    if not pairs:
        raise ValueError(f"No MRI volumes found in {root}")
    return pairs


def split_pairs(pairs, fraction=0.2, seed=42):
    """Build the two-way patient split used by the smoke test.

    Production experiments use split_three_way to reserve test patients too.
    """
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    # All Week scans for a patient stay in one partition.
    patients = sorted({Path(p[0]).name.split("_")[0] for p in pairs})
    if len(patients) < 2:
        raise ValueError("At least two patients are required")
    random.Random(seed).shuffle(patients)
    count = max(1, min(len(patients) - 1, round(len(patients) * fraction)))
    validation = set(patients[:count])
    return (
        [p for p in pairs if Path(p[0]).name.split("_")[0] not in validation],
        [p for p in pairs if Path(p[0]).name.split("_")[0] in validation],
    )


def load_volume(pair, num_classes):
    """Validate a shared NIfTI grid and return normalized MRI and integer labels.

    Nonzero MRI voxels are standardized per volume; zero-valued background
    is preserved. Arrays retain their stored voxel-axis order: no reorientation
    or resampling occurs here. Labels have shape (D, H, W), without a channel.
    """
    image_nii, mask_nii = (nib.load(p) for p in pair)
    if len(image_nii.shape) != 3 or image_nii.shape != mask_nii.shape:
        raise ValueError(f"Expected matching 3D volumes: {pair}")
    if not np.allclose(image_nii.affine, mask_nii.affine, atol=1e-4):
        raise ValueError(f"MRI and mask affines differ: {pair}")
    image = image_nii.get_fdata(dtype=np.float32)
    labels = mask_nii.get_fdata(dtype=np.float32)
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError(f"Non-finite voxel values: {pair}")
    if (
        not np.equal(labels, np.rint(labels)).all()
        or labels.min() < 0
        or labels.max() >= num_classes
    ):
        raise ValueError(f"Invalid class labels: {pair}")
    nonzero = image != 0
    values = image[nonzero]
    if not values.size:
        raise ValueError(f"Empty MRI: {pair[0]}")
    image[nonzero] = (values - values.mean()) / max(float(values.std()), 1e-6)
    return image, labels.astype(np.int64)


def split_three_way(pairs, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Freeze test patients before any model selection; fractions are patient counts."""
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Positive validation/test fractions must sum to less than one")
    patients = sorted({Path(p[0]).name.split("_")[0] for p in pairs})
    if len(patients) < 3:
        raise ValueError("Train/validation/test splitting requires at least three patients")
    random.Random(seed).shuffle(patients)
    n_test = max(1, min(len(patients) - 2, round(len(patients) * test_fraction)))
    n_val = max(1, min(len(patients) - n_test - 1, round(len(patients) * val_fraction)))
    groups = {
        "test": set(patients[:n_test]),
        "validation": set(patients[n_test : n_test + n_val]),
        "train": set(patients[n_test + n_val :]),
    }
    return {
        name: [p for p in pairs if Path(p[0]).name.split("_")[0] in ids]
        for name, ids in groups.items()
    }


class PatchDataset(Dataset):
    """Sample random training patches with optional foreground-class balancing.

    Each scan contributes patches_per_volume samples per epoch. Foreground
    sampling first chooses a present class uniformly, then one of its voxels,
    so large organs do not monopolize all foreground-centred patches.
    """

    def __init__(
        self,
        pairs,
        patch_size,
        num_classes=6,
        patches_per_volume=8,
        foreground_probability=0.7,
        cache_dir=None,
    ):
        self.pairs = pairs
        self.patch_size = tuple(patch_size)
        self.num_classes = num_classes
        self.patches_per_volume = patches_per_volume
        self.foreground_probability = foreground_probability
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.pairs) * self.patches_per_volume

    def __getitem__(self, index):
        """Return an MRI (1, D, H, W) and class-index mask (D, H, W)."""
        pair = self.pairs[index // self.patches_per_volume]
        locations_by_class = None
        if self.cache_dir:
            image, mask, locations_by_class = cached_volume(pair, self.num_classes, self.cache_dir)
        else:
            image, mask = load_volume(pair, self.num_classes)
        # Cached indices refer to the unpadded grid; padding extends only high ends.
        original_shape = image.shape
        padding = [(0, max(0, p - s)) for p, s in zip(self.patch_size, image.shape)]
        if any(after for _, after in padding):
            image, mask = np.pad(image, padding), np.pad(mask, padding)
        classes = (
            np.array(sorted(locations_by_class))
            if locations_by_class is not None
            else np.unique(mask)
        )
        classes = classes[classes > 0]
        if classes.size and np.random.random() < self.foreground_probability:
            label = np.random.choice(classes)
            if locations_by_class is None:
                locations = np.argwhere(mask == label)
                center = locations[np.random.randint(len(locations))]
            else:
                locations = locations_by_class[int(label)]
                center = np.unravel_index(
                    int(locations[np.random.randint(len(locations))]), original_shape
                )
            starts = [
                max(0, min(int(c) - p // 2, s - p))
                for c, p, s in zip(center, self.patch_size, image.shape)
            ]
        else:
            starts = [np.random.randint(s - p + 1) for s, p in zip(image.shape, self.patch_size)]
        region = tuple(slice(s, s + p) for s, p in zip(starts, self.patch_size))
        image, mask = image[region], mask[region]
        # Intensity augmentation preserves anatomical class identity (no left/right flips).
        if np.random.random() < 0.5:
            image = image * np.random.uniform(0.9, 1.1)
        # Copy slices so tensors own writable memory, even when the source is mmap.
        return torch.from_numpy(image.copy())[None].float(), torch.from_numpy(mask.copy()).long()


def cache_path(pair, num_classes, cache_dir):
    # Invalidate when paths, sizes, modification times, class count or format change.
    """Derive a cache directory from source metadata and preprocessing version."""
    fingerprint = [
        (str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in pair
    ]
    key = hashlib.sha256(json.dumps([1, num_classes, fingerprint]).encode()).hexdigest()
    return Path(cache_dir) / key


def prepare_cache(pairs, num_classes, cache_dir):
    """Build once in the parent process before workers start; never cache random patches."""
    for index, pair in enumerate(pairs):
        folder = cache_path(pair, num_classes, cache_dir)
        if (folder / "ready.json").is_file():
            continue
        image, mask = load_volume(pair, num_classes)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "image.npy", image)
        dtype = np.uint8 if num_classes <= 256 else np.int64
        np.save(folder / "mask.npy", mask.astype(dtype))
        classes = [int(c) for c in np.unique(mask) if c > 0]
        for label in classes:
            indices = np.flatnonzero(mask == label)
            np.save(
                folder / f"class_{label}.npy",
                indices.astype(np.uint32 if mask.size <= 2**32 else np.int64),
            )
        # Publish readiness last so partially written caches are rebuilt.
        (folder / "ready.json").write_text(json.dumps(classes))
        print(f"Prepared cache {index+1}/{len(pairs)}: {Path(pair[0]).name}", flush=True)


@lru_cache(maxsize=4)
def _open_cached(folder):
    """Keep a small per-process LRU of read-only, memory-mapped volume arrays."""
    folder = Path(folder)
    classes = json.loads((folder / "ready.json").read_text())
    return (
        np.load(folder / "image.npy", mmap_mode="r"),
        np.load(folder / "mask.npy", mmap_mode="r"),
        {c: np.load(folder / f"class_{c}.npy", mmap_mode="r") for c in classes},
    )


def cached_volume(pair, num_classes, cache_dir):
    """Open a prepared volume and its per-class flattened foreground indices."""
    return _open_cached(str(cache_path(pair, num_classes, cache_dir)))


def clear_volume_cache():
    """Release mmap references, including Windows file locks, after a run."""
    _open_cached.cache_clear()
