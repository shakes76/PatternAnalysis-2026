"""ADNI patient-independent data loader and first-batch smoke test."""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path('/home/groups/comp3710/ADNI')
PATIENT_RE = re.compile(r'ADNI_(\d{3}_S_\d{4})')


def build_splits(seed=3710):
    metadata = json.loads((ROOT / 'meta_data_with_label.json').read_text())
    rows = []
    patient_labels = {}
    scan_ids = set()

    for old_split in ('train', 'test'):
        for category, label, meta_label in (('NC', 0, 0), ('AD', 1, 2)):
            folder = ROOT / 'AD_NC' / old_split / category
            for path in sorted(folder.glob('*.jpeg')):
                scan = path.stem.rsplit('_', 1)[0]
                info = metadata[scan]
                assert int(info['label']) == meta_label, 'Metadata label mismatch'
                match = PATIENT_RE.search(info['raw'])
                assert match, 'Patient ID missing'
                patient = match.group(1)
                if patient in patient_labels:
                    assert patient_labels[patient] == label, 'Patient has conflicting labels'
                patient_labels[patient] = label
                rows.append((path, label, patient, scan))
                scan_ids.add(scan)

    groups = defaultdict(list)
    for patient, label in patient_labels.items():
        groups[label].append(patient)

    assignments = {}
    for label, patients in sorted(groups.items()):
        patients.sort()
        random.Random(seed + label).shuffle(patients)
        n = len(patients)
        assert n >= 3, 'Need at least 3 patients per class'
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        for p in patients[:train_end]:
            assignments[p] = 'train'
        for p in patients[train_end:val_end]:
            assignments[p] = 'val'
        for p in patients[val_end:]:
            assignments[p] = 'test'

    splits = {part: [] for part in ('train', 'val', 'test')}
    for row in rows:
        splits[assignments[row[2]]].append(row)

    patient_sets = [{r[2] for r in splits[part]} for part in splits]
    assert not (patient_sets[0] & patient_sets[1] or
                patient_sets[0] & patient_sets[2] or
                patient_sets[1] & patient_sets[2])

    print('All images:', len(rows), '| scans:', len(scan_ids),
          '| patients:', len(patient_labels))
    for part, entries in splits.items():
        count = Counter(r[1] for r in entries)
        print(f'{part}: {len(set(r[2] for r in entries))} patients, '
              f'{len(entries)} slices (NC={count[0]}, AD={count[1]})')

    print('Patient overlap across new splits: 0')
    return splits


class ADNIDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, patient, scan = self.entries[idx]
        with Image.open(path) as img:
            rgb = ImageOps.pad(img.convert('RGB'), (224, 224),
                               method=Image.Resampling.BILINEAR, color=0)
            image = self.to_tensor(rgb)
        return {'image': image, 'label': label,
                'patient_id': patient, 'scan_id': scan}


if __name__ == '__main__':
    splits = build_splits()
    loader = DataLoader(ADNIDataset(splits['train']), batch_size=4,
                        shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print('First batch image shape:', tuple(batch['image'].shape))
    print('First batch label shape:', tuple(batch['label'].shape))
    print('CPU smoke test: PASS')
