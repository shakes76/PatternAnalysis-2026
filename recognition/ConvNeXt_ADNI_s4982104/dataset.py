"""Fixed-manifest ADNI slices for the grayscale CNN baseline (no augmentation).

Model labels: NC=0, AD=1; source metadata instead encodes AD=2.
All splits use float32 [1,240,256] images, scaled by 255 into [0,1].
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 256
LABEL_MAP = {'NC': 0, 'AD': 1}
SPLITS = ('train', 'validation', 'test')
FIELDS = ('file_path','original_split','image_id','patient_id','scan_date','label','split')
PREPROCESSING = dict(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, channels=1,
                     color_mode='L', resize='Pillow bilinear (only when size differs)',
                     normalization='uint8 / 255.0', value_range=[0.0,1.0],
                     image_dtype='torch.float32', label_dtype='torch.int64',
                     random_augmentation=False)


class ADNISliceDataset(Dataset):
    """Return image, scalar long label, patient_id, image_id and file_path.

    Construction validates all manifest rows and its recorded SHA-256 but does
    not read image pixels. __getitem__ reads only the selected split's image.
    Images and labels stay on CPU; device transfer belongs to the training loop.
    """
    def __init__(self, manifest_path, split, summary_path=None):
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split!r}; expected {SPLITS}')
        self.split = split
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        summary_path = (Path(summary_path).expanduser() if summary_path is not None
                        else self.manifest_path.with_name('split_summary.json'))
        raw = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        evidence = json.loads(summary_path.read_text())
        if evidence.get('status') != 'passed':
            raise ValueError('Split summary has not passed validation')
        if evidence.get('output_sha256', {}).get('split_samples') != self.manifest_sha256:
            raise ValueError('split_samples.csv hash does not match split_summary.json')
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8'), newline=''))
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError('Expected original six columns plus split')
        rows = list(reader)
        if not rows:
            raise ValueError('Empty manifest')
        paths, patient_info, image_info = set(), {}, {}
        for line, row in enumerate(rows, 2):
            def fail(reason):
                raise ValueError(f'Manifest line {line}: {reason}')
            if None in row or any(v is None for v in row.values()):
                fail('malformed CSV row')
            if row['split'] not in SPLITS:
                fail('unknown split')
            if row['label'] not in LABEL_MAP:
                fail('unknown label; expected NC or AD, not raw numeric metadata')
            if not Path(row['file_path']).is_absolute():
                fail('file_path must be absolute')
            if row['file_path'] in paths:
                fail('duplicate file_path')
            paths.add(row['file_path'])
            if not re.fullmatch(r'\d{3}_S_\d{4}', row['patient_id']):
                fail('invalid patient_id')
            if not re.fullmatch(r'[1-9]\d*', row['image_id']):
                fail('invalid image_id')
            info = (row['split'], row['image_id'], row['label'])
            previous = patient_info.setdefault(row['patient_id'], info)
            if previous != info:
                fail('patient has conflicting split, image_id or label')
            previous = image_info.setdefault(row['image_id'], (row['patient_id'], row['label']))
            if previous != (row['patient_id'], row['label']):
                fail('image has conflicting patient or label')
        self.records = tuple(dict(r) for r in rows if r['split'] == split)
        if not self.records:
            raise ValueError(f'No samples for split {split}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = record['file_path']
        try:
            # Do not allow Pillow to silently accept truncated image streams.
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with Image.open(path) as image:
                image.load()
                gray = image.convert('L')
                if gray.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    gray = gray.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
                pixels = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8)
                tensor = pixels.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH).to(torch.float32).div_(255.0)
        except Exception as exc:
            raise RuntimeError(f'Failed to read image {path}: {type(exc).__name__}: {exc}') from exc
        return dict(image=tensor, label=torch.tensor(LABEL_MAP[record['label']], dtype=torch.long),
                    patient_id=record['patient_id'], image_id=record['image_id'], file_path=path)


CONVNEXT_PREPROCESSING = dict(
    color_mode='L replicated to RGB', channels=3, resize_short_edge=236,
    crop_size=224, interpolation='bilinear', antialias=True,
    normalization='uint8 / 255, then channel (x-mean)/std',
    mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225],
    image_dtype='torch.float32', label_dtype='torch.int64',
    random_augmentation=False, backend='PIL resize/crop, TorchVision tensor conversion')


def convnext_crop(image):
    """Official IMAGENET1K_V1 geometry; no intermediate CNN resize.

    Ref: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html
    """
    from torchvision.transforms import functional as F, InterpolationMode
    rgb = image.convert('L').convert('RGB')
    resized = F.resize(rgb, [236], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return F.center_crop(resized, [224])


def convnext_transform(image):
    """Deterministic official geometry and normalization for a grayscale slice."""
    from torchvision.transforms import functional as F
    tensor = F.convert_image_dtype(F.pil_to_tensor(convnext_crop(image)), torch.float32)
    return F.normalize(tensor, CONVNEXT_PREPROCESSING['mean'], CONVNEXT_PREPROCESSING['std'])


class ConvNeXtSliceDataset(ADNISliceDataset):
    """Reuse the frozen manifest validation, read original pixels once per call."""
    def __init__(self, manifest_path, split, summary_path=None):
        super().__init__(manifest_path, split, summary_path)
        self.read_calls = 0
        self.observed_paths = []

    def read_gray(self, index):
        path = self.records[index]['file_path']
        self.read_calls += 1
        self.observed_paths.append(path)
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            with Image.open(path) as image:
                image.load()
                return image.convert('L')
        except Exception as exc:
            raise RuntimeError(f'Failed to read image {path}: {exc}') from exc

    def __getitem__(self, index):
        row = self.records[index]
        return dict(image=convnext_transform(self.read_gray(index)),
                    label=torch.tensor(LABEL_MAP[row['label']],dtype=torch.long),
                    **{key:row[key] for key in ('patient_id','image_id','file_path')})


def split_by_patient(selected_scans, selected_samples):
    """Expose the audited fixed-seed, leakage-free split from the data module."""
    from split_patients import build
    return build(selected_scans, selected_samples)


class AugmentedTrainingDataset(ADNISliceDataset):
    """Optional mild brightness augmentation for training; unused in frozen runs."""
    def __init__(self, manifest_path, split='train', summary_path=None):
        if split != 'train':
            raise ValueError('Intensity augmentation is training only')
        super().__init__(manifest_path, split, summary_path)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        gain = 0.9 + 0.2 * torch.rand(()).item()
        item['image'] = (item['image'] * gain).clamp(0, 1)
        return item
