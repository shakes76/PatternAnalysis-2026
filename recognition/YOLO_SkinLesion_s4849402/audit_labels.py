"""Download and audit small official ISIC 2017 train/validation diagnosis CSVs.

No image downloads, patient IDs invented, or test-label inspection.
Uses Python standard library only. Cached downloads are validated each run.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

BASE = 'https://isic-archive.s3.amazonaws.com/challenges/2017/'
FILES = {'train': 'ISIC-2017_Training_Part3_GroundTruth.csv',
         'val': 'ISIC-2017_Validation_Part3_GroundTruth.csv'}
CLASSES = ['melanoma', 'seborrheic_keratosis', 'nevus']


def parse_labels(payload):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
    columns = reader.fieldnames or []
    required = {'image_id', 'melanoma', 'seborrheic_keratosis'}
    if not required.issubset(columns):
        raise ValueError(f'Unexpected columns: {columns}')
    ids, counts = set(), Counter({name: 0 for name in CLASSES})
    for row in reader:
        image_id = row['image_id'].strip()
        if not image_id or image_id in ids:
            raise ValueError(f'Empty or duplicate image ID: {image_id}')
        ids.add(image_id)
        mel, sk = float(row['melanoma']), float(row['seborrheic_keratosis'])
        if mel not in (0, 1) or sk not in (0, 1) or mel + sk > 1:
            raise ValueError(f'Invalid diagnostic flags: {image_id}')
        name = 'melanoma' if mel else 'seborrheic_keratosis' if sk else 'nevus'
        counts[name] += 1
    if not ids:
        raise ValueError('No records')
    patient_fields = [c for c in columns if 'patient' in c.lower() or 'subject' in c.lower()]
    return {'columns': columns, 'images': len(ids), 'class_counts': dict(counts),
            'patient_columns': patient_fields}, ids


def run(data_dir, report_path):
    data_dir.mkdir(parents=True, exist_ok=True)
    report = {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_page': 'https://challenge.isic-archive.com/data/#2017',
              'class_order': CLASSES, 'splits': {}}
    split_ids = {}
    for split, filename in FILES.items():
        path = data_dir / filename
        url = BASE + filename
        if path.exists():
            payload = path.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError('Diagnosis file unexpectedly exceeds 1 MiB')
        summary, ids = parse_labels(payload)
        path.write_bytes(payload)
        summary.update(url=url, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        report['splits'][split], split_ids[split] = summary, ids
    overlap = sorted(split_ids['train'] & split_ids['val'])
    report['train_val_image_id_overlap'] = overlap
    report['patient_independence_verified'] = False
    report['limitations'] = [
        'Image ID disjointness does not demonstrate patient independence.',
        'Images and masks have not yet been checked against diagnosis IDs.',
        'No test labels were downloaded or used for experiment selection.',
        'Absent patient identifiers require verified metadata or tutor guidance before final splitting.'
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if overlap:
        raise ValueError('Image IDs overlap between train and validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data' / 'metadata')
    parser.add_argument('--report', type=Path, default=Path(__file__).parent / 'reports' / 'label_audit.json')
    args = parser.parse_args()
    run(args.data_dir, args.report)
