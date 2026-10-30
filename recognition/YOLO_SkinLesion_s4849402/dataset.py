"""Audit a patient-verified ISIC manifest and export one-lesion YOLO data.

NumPy/Pillow are used only for data preprocessing, not model computation.
Mask coordinates use exclusive right/bottom edges to retain single-pixel lesions.
"""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def mask_to_box(mask):
    """Return normalized center-x, center-y, width, height of mask union."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError('Mask must be single-channel')
    ys, xs = np.nonzero(array > 0)
    if not len(xs):
        raise ValueError('Empty mask: diagnosis/negative handling requires review')
    height, width = array.shape
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return ((left + right) / (2 * width), (top + bottom) / (2 * height),
            (right - left) / width, (bottom - top) / height)


def audit_manifest(manifest, class_count, split_policy='patient-id', official_labels_dir=None):
    """Validate every record before writing outputs; do not invent patients."""
    manifest = Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image_id', 'image_path', 'mask_path', 'patient_id', 'class_id', 'split'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Required columns: {sorted(required)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty')
    if split_policy not in ('patient-id', 'official-isic'):
        raise ValueError('Unknown split policy')
    official = {}
    if split_policy == 'official-isic':
        if official_labels_dir is None:
            raise ValueError('Official split requires official diagnosis CSVs')
        from audit_labels import parse_labels
        for split, title in [('train', 'Training'), ('val', 'Validation'), ('test', 'Test')]:
            path = Path(official_labels_dir) / f'ISIC-2017_{title}_Part3_GroundTruth.csv'
            if not path.exists():
                continue
            parse_labels(path.read_bytes())
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    image_id = row['image_id']
                    if image_id in official:
                        raise ValueError('Official label files contain overlapping image IDs')
                    cls = 0 if float(row['melanoma']) else 1 if float(row['seborrheic_keratosis']) else 2
                    official[image_id] = (split, cls)
    patients, hashes, ids, counts, records = {}, {}, set(), Counter(), []
    for row in rows:
        row = {k: (v or '').strip() for k, v in row.items() if k}
        image_id, patient, split = row['image_id'], row['patient_id'], row['split']
        if not image_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in image_id):
            raise ValueError('image_id must contain only letters, digits, underscores or hyphens')
        if image_id in ids:
            raise ValueError(f'Duplicate image_id: {image_id}')
        ids.add(image_id)
        if patient.lower() in {'unknown', 'na', 'n/a', 'none', 'nan'}:
            patient = ''
        if not patient and split_policy == 'patient-id':
            raise ValueError(f'Missing verified patient_id: {image_id}')
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f'Invalid split: {split}')
        if patient and patient in patients and patients[patient] != split:
            raise ValueError(f'Patient leakage across splits: {patient}')
        if patient:
            patients[patient] = split
        class_id = int(row['class_id'])
        if not 0 <= class_id < class_count:
            raise ValueError(f'Invalid class_id: {class_id}')
        if split_policy == 'official-isic' and official.get(image_id) != (split, class_id):
            raise ValueError(f'Official split/class mismatch: {image_id}')
        paths = []
        for key in ['image_path', 'mask_path']:
            if not row[key]:
                raise ValueError(f'Missing {key}: {image_id}')
            path = (manifest.parent / row[key]).resolve()
            if not path.is_file():
                raise ValueError(f'File missing: {path}')
            paths.append(path)
        image_path, mask_path = paths
        with Image.open(image_path) as image:
            size = image.size
            digest = hashlib.sha256(str(size).encode() + image.convert('RGB').tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f'Duplicate decoded image: {image_id} and {hashes[digest]}')
        hashes[digest] = image_id
        with Image.open(mask_path) as mask:
            if mask.size != size:
                raise ValueError(f'Image/mask size mismatch: {image_id}')
            box = mask_to_box(mask)
        records.append((image_id, split, class_id, image_path, box))
        counts[f'{split}/class_{class_id}'] += 1
    if not {'train', 'val'}.issubset({r[1] for r in records}):
        raise ValueError('Train and validation must be nonempty')
    return records, dict(counts), len(patients)


def export(manifest, output, names, split_policy='patient-id', split_evidence='', official_labels_dir=None):
    """Export only after a complete audit; refuse to overwrite a directory."""
    if not names or len(set(names)) != len(names):
        raise ValueError('Class names must be nonempty and unique')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    if split_policy == 'official-isic':
        from modules import CLASS_NAMES
        if tuple(names) != CLASS_NAMES or not split_evidence.strip():
            raise ValueError('Official ISIC mode requires the fixed class order and confirmation evidence')
    records, counts, patient_count = audit_manifest(manifest, len(names), split_policy, official_labels_dir)
    splits = sorted({r[1] for r in records})
    for split in splits:
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    for image_id, split, class_id, source, box in records:
        shutil.copy2(source, output / 'images' / split / (image_id + source.suffix.lower()))
        label = str(class_id) + ' ' + ' '.join(f'{value:.10f}' for value in box) + '\n'
        (output / 'labels' / split / f'{image_id}.txt').write_text(label, encoding='utf-8')
    # JSON is a YAML subset and can be parsed by the YOLO YAML loader.
    config = {'path': output.as_posix(), 'names': names}
    config.update({split: f'images/{split}' for split in splits})
    (output / 'data.yaml').write_text(json.dumps(config, indent=2), encoding='utf-8')
    from utils import sha256
    label_hashes = {p.name: sha256(p) for p in Path(official_labels_dir).glob('ISIC-2017_*_Part3_GroundTruth.csv')} if official_labels_dir else {}
    audit = {'images': len(records), 'patients_with_supplied_ids': patient_count, 'counts': counts,
             'split_policy': split_policy, 'split_evidence': split_evidence,
             'manifest_sha256': sha256(manifest), 'official_label_sha256': label_hashes,
             'patient_independence_programmatically_verified': split_policy == 'patient-id',
             'image_ids': {split: [r[0] for r in records if r[1] == split] for split in splits},
             'limits': ['patient IDs are supplied, not independently verified',
                        'exact decoded duplicates checked; near duplicates require review',
                        'one lesion per image; all foreground pixels form one box']}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--split-policy', choices=['patient-id', 'official-isic'], default='patient-id')
    parser.add_argument('--split-evidence', default='')
    parser.add_argument('--official-labels-dir')
    args = parser.parse_args()
    export(args.manifest, args.output, args.names, args.split_policy, args.split_evidence, args.official_labels_dir)
