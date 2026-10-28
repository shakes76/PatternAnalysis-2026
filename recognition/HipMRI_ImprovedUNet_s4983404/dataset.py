"""HipMRI NIfTI pairs; preserve provided patient splits and native 256x128 grids.

Loading follows the assignment Appendix B conceptually; fixed six-channel masks
avoid its per-slice unique-label channel-count problem. No external loader copied.
"""
from pathlib import Path
import re
import logging
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path('/home/groups/comp3710/HipMRI_Study_open/keras_slices_data')
PATTERN = re.compile(r'case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$')


def pair_paths(root: Path, split: str) -> list[tuple[Path, Path]]:
    if split not in ('train', 'validate', 'test'):
        raise ValueError('unknown split')
    image_dir = root / f'keras_slices_{split}'
    label_dir = root / f'keras_slices_seg_{split}'
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f'missing image/label directory for {split}')
    images = sorted(image_dir.glob('*.nii.gz'))
    if not images or any(PATTERN.fullmatch(path.name) is None for path in images):
        raise ValueError('empty split or unexpected filename')
    pairs = [(path, label_dir/path.name.replace('case_', 'seg_', 1)) for path in images]
    if {label.name for _, label in pairs} != {p.name for p in label_dir.glob('*.nii.gz')}:
        raise ValueError(f'image/label key mismatch: {split}')
    return pairs


def check_patient_splits(root: Path) -> dict[str, list[str]]:
    subjects = {split: sorted({PATTERN.fullmatch(image.name).group(1)
                              for image, _ in pair_paths(root, split)})
                for split in ('train', 'validate', 'test')}
    for left, right in (('train', 'validate'), ('train', 'test'), ('validate', 'test')):
        if set(subjects[left]) & set(subjects[right]):
            raise ValueError(f'patient leakage: {left}/{right}')
    return subjects


def normalize_image(image: np.ndarray) -> torch.Tensor:
    """Use the same native-grid normalization for training and unlabeled inference."""
    image = np.asarray(image, dtype=np.float32).squeeze()
    if image.ndim != 2:
        raise ValueError('expected a 2D slice, not a 3D volume')
    if not np.isfinite(image).all():
        raise ValueError('nonfinite image pixels')
    image = (image - image.mean()) / max(float(image.std()), 1e-6)
    return torch.from_numpy(image.copy()).unsqueeze(0)


def load_pair(image_path: Path, label_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    image_nifti, label_nifti = nib.load(image_path), nib.load(label_path)
    image = np.asarray(image_nifti.dataobj, dtype=np.float32).squeeze()
    labels = np.asarray(label_nifti.dataobj, dtype=np.float32).squeeze()
    if image.ndim != 2 or image.shape != labels.shape:
        raise ValueError(f'non-2D or mismatched grids: {image_path.name}')
    if not np.allclose(image_nifti.affine, label_nifti.affine, atol=1e-5):
        raise ValueError(f'affine mismatch: {image_path.name}')
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise ValueError('nonfinite pixels')
    if not np.isin(labels, np.arange(6)).all():
        raise ValueError('labels must be integer IDs 0..5')
    # Normalize each slice independently; no test statistics enter training.
    return normalize_image(image), torch.from_numpy(labels.astype(np.int64))


class HipMRISlices(Dataset):
    def __init__(self, root: Path = ROOT, split: str = 'train',
                 limit: int | None = None, augment: bool = False,
                 cache: bool = False) -> None:
        self.pairs = pair_paths(root, split)
        if limit is not None:
            if limit <= 0:
                raise ValueError('limit must be positive')
            # Evenly spread smoke subsets instead of the first patient's slices.
            indices = np.linspace(0, len(self.pairs)-1, min(limit, len(self.pairs)), dtype=int)
            self.pairs = [self.pairs[index] for index in indices]
        self.augment = augment
        self.cached = []
        if cache:
            for index, pair in enumerate(self.pairs):
                image, labels = load_pair(*pair)
                self.cached.append((image, labels.to(torch.uint8)))
                if index % 1000 == 0:
                    logging.info('Cached %s: %d/%d', split, index, len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_path, label_path = self.pairs[index]
        image, labels = self.cached[index] if self.cached else load_pair(image_path, label_path)
        if self.augment and torch.rand(()) < 0.5:
            image, labels = image.flip(-2), labels.flip(-2)
        target = F.one_hot(labels.long(), num_classes=6).permute(2, 0, 1).float()
        return image, target, image_path.name


def collate_slices(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Pad on bottom/right only; all-zero target channels mark invalid padding."""
    height = max(image.shape[-2] for image, _, _ in batch)
    width = max(image.shape[-1] for image, _, _ in batch)
    images, targets, names = [], [], []
    for image, target, name in batch:
        padding = (0, width-image.shape[-1], 0, height-image.shape[-2])
        images.append(F.pad(image, padding))
        targets.append(F.pad(target, padding))
        names.append(name)
    return torch.stack(images), torch.stack(targets), names
