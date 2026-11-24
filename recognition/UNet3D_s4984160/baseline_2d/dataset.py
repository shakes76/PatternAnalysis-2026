"""HipMRI 2D loading with explicit patient splits and ignored padding."""
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = 255
TARGET_SHAPE = (256, 144)
DEFAULT_SPLIT = Path(__file__).resolve().parent / 'support' / 'patient_split.json'
IMAGE_PATTERN = re.compile(r'case_(\d{3})_week_(\d+)_slice_(\d+)\.nii\.gz')
MASK_PATTERN = re.compile(r'seg_(\d{3})_week_(\d+)_slice_(\d+)\.nii\.gz')


def normalize_patient_id(value):
    """Accept numeric 2D IDs or the verified letter-prefixed 3D IDs."""
    match = re.fullmatch(r'[A-Z]?(\d{1,3})', str(value))
    if match is None:
        raise ValueError(f'Invalid patient ID: {value}')
    return match[1].zfill(3)


def get_data_pairs(data_dir):
    """Search all supplied source folders, independent of their original split."""
    root = Path(data_dir)
    if (root / 'keras_slices_data').is_dir():
        root = root / 'keras_slices_data'
    folder_pairs = []
    for split in ('train', 'validate', 'test'):
        images = root / f'keras_slices_{split}'
        masks = root / f'keras_slices_seg_{split}'
        if images.is_dir() or masks.is_dir():
            if not images.is_dir() or not masks.is_dir():
                raise FileNotFoundError(f'Both image and mask folders are required: {images}, {masks}')
            folder_pairs.append((images, masks))
    # Also allow a deliberately prepared flat local image/mask directory.
    if not folder_pairs and any(root.glob('case_*.nii.gz')):
        folder_pairs = [(root, root)]
    if not folder_pairs:
        raise FileNotFoundError(f'No HipMRI slice folders found under {root}')

    image_index, mask_index = {}, {}
    for images, masks in folder_pairs:
        for directory, prefix, pattern, index in (
            (images, 'case_', IMAGE_PATTERN, image_index),
            (masks, 'seg_', MASK_PATTERN, mask_index),
        ):
            for path in directory.glob(f'{prefix}*.nii.gz'):
                match = pattern.fullmatch(path.name)
                if match is None:
                    raise ValueError(f'Unexpected slice filename: {path.name}')
                key = (match[1], int(match[2]), int(match[3]))
                if not 0 <= key[2] < 60:
                    raise ValueError(f'Unexpected slice index: {path.name}')
                if key in index:
                    raise ValueError(f'Duplicate patient/week/slice: {index[key]}, {path}')
                index[key] = path
    if image_index.keys() != mask_index.keys():
        raise FileNotFoundError(
            f'Missing masks: {sorted(image_index.keys()-mask_index.keys())[:5]}; '
            f'orphan masks: {sorted(mask_index.keys()-image_index.keys())[:5]}')
    if not image_index:
        raise ValueError('No image/mask pairs found')
    return [dict(image_path=image_index[key], mask_path=mask_index[key],
                 patient_id=key[0], week_id=key[1], slice_id=key[2],
                 filename=image_index[key].name)
            for key in sorted(image_index)]


def load_patient_split(path=DEFAULT_SPLIT):
    """Load explicit disjoint groups; never infer a split from directory order."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    groups = {}
    for name in ('train', 'validation', 'test'):
        values = [normalize_patient_id(value) for value in document[name]]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'Empty or duplicate patient IDs in {name}')
        groups[name] = set(values)
    if any(groups[a] & groups[b] for a, b in
           [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Patient leakage: split groups overlap')
    return groups


def _read_slice(image):
    array = np.asarray(image.dataobj, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D slice or trailing singleton, got {array.shape}')
    return array


class HipMRI2DDataset(Dataset):
    """Return (float32 image, int64 mask, metadata) without resizing anatomy.

    data_dir accepts the HipMRI root, keras_slices_data, or a prepared flat
    directory. patient_ids=None loads all available pairs for inspection.
    transform must jointly transform the image/mask tensors, preserve their
    original shapes and integer labels, and is applied before padding.
    """
    def __init__(self, data_dir, patient_ids=None, transform=None, *, samples=None):
        self.transform = transform
        records = get_data_pairs(data_dir) if samples is None else samples
        wanted = (None if patient_ids is None else
                  {normalize_patient_id(value) for value in patient_ids})
        self.samples = [r for r in records if wanted is None or r['patient_id'] in wanted]
        available = {r['patient_id'] for r in self.samples}
        if wanted is not None and wanted - available:
            raise ValueError(f'No data for requested patients: {sorted(wanted-available)}')
        if not self.samples:
            raise ValueError('Dataset is empty')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_nii = nib.load(str(sample['image_path']))
        mask_nii = nib.load(str(sample['mask_path']))
        if not np.allclose(image_nii.affine, mask_nii.affine):
            raise ValueError(f'MRI/mask affine mismatch: {sample["filename"]}')
        image, mask = _read_slice(image_nii), _read_slice(mask_nii)
        if image.shape != mask.shape:
            raise ValueError(f'MRI/mask shape mismatch: {sample["filename"]}')
        if image.shape not in ((256, 128), (256, 144)):
            raise ValueError(f'Unexpected HipMRI slice shape: {image.shape}')
        if not np.isfinite(image).all() or not np.isfinite(mask).all():
            raise ValueError(f'Non-finite image or mask: {sample["filename"]}')
        if not np.equal(mask, np.rint(mask)).all() or not np.isin(mask, np.arange(6)).all():
            raise ValueError(f'Mask must contain integer labels 0..5: {sample["filename"]}')
        h, w = image.shape
        # Compute statistics on original pixels, excluding artificial padding.
        mean, std = image.mean(dtype=np.float64), image.std(dtype=np.float64)
        image = np.asarray((image-mean)/max(float(std), 1e-8), dtype=np.float32)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.int64))
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
            if image_tensor.shape != (1, h, w) or mask_tensor.shape != (h, w):
                raise ValueError('Paired transform must preserve original shapes')
            if mask_tensor.dtype != torch.int64 or not ((mask_tensor >= 0) & (mask_tensor < 6)).all():
                raise ValueError('Paired transform must preserve integer labels 0..5')
            if not torch.isfinite(image_tensor).all():
                raise ValueError('Paired transform produced non-finite image values')
        # Preserve the submitted right-padding convention and record it explicitly.
        pad_right = TARGET_SHAPE[1]-w
        image_tensor = F.pad(image_tensor.float(), (0, pad_right, 0, 0), value=0)
        mask_tensor = F.pad(mask_tensor, (0, pad_right, 0, 0), value=IGNORE_INDEX)
        metadata = dict(patient_id=sample['patient_id'], week_id=sample['week_id'],
                        slice_id=sample['slice_id'], filename=sample['filename'],
                        original_h=h, original_w=w, pad_right=pad_right,
                        source_axis=1, source_index=98+sample['slice_id'], source_flip_axis=0)
        return image_tensor, mask_tensor, metadata


def get_dataloaders(data_dir, batch_size=16, num_workers=0,
                    split_manifest=DEFAULT_SPLIT, train_transform=None):
    """Use the recorded 3D patient groups, not the source folder partitions."""
    groups = load_patient_split(split_manifest)
    samples = get_data_pairs(data_dir)
    available = {r['patient_id'] for r in samples}
    expected = set.union(*groups.values())
    if available != expected:
        raise ValueError(f'Patient coverage mismatch: missing={sorted(expected-available)}, '
                         f'unassigned={sorted(available-expected)}')
    datasets = [HipMRI2DDataset(data_dir, groups[name],
                               train_transform if name == 'train' else None, samples=samples)
                for name in ('train', 'validation', 'test')]
    return tuple(DataLoader(dataset, batch_size=1 if i == 2 else batch_size,
                            shuffle=i == 0, num_workers=num_workers)
                 for i, dataset in enumerate(datasets))
