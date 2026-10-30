"""Build an ISIC 2017 manifest from extracted official archives."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_labels import parse_labels
from dataset import export
from modules import CLASS_NAMES
from utils import save_json, sha256


def resolve_train_overlap(rows, duplicate_audit):
    from PIL import Image

    groups = json.loads(Path(duplicate_audit).read_text(encoding='utf-8'))
    if not isinstance(groups, list):
        raise ValueError('Expected a list of duplicate groups')
    lookup = {row['image_id']: row for row in rows}
    excluded, details, seen = set(), [], set()
    for group in groups:
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError('Invalid duplicate group')
        members, digests = [], set()
        for item in group:
            image_id = item['image_id']
            if image_id in seen:
                raise ValueError('An image occurs in multiple duplicate groups')
            seen.add(image_id)
            row = lookup.get(image_id)
            if row is None or row['split'] != item['split']:
                raise ValueError(f'Duplicate audit membership mismatch: {image_id}')
            with Image.open(row['image_path']) as image:
                image = image.convert('RGB')
                digests.add(hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest())
            members.append(row)
        if len(digests) != 1:
            raise ValueError('Reported duplicate images are not pixel-identical')
        if len({row['class_id'] for row in members}) != 1:
            raise ValueError('Duplicate images have conflicting diagnostic classes')
        held_out = [row for row in members if row['split'] in ('val', 'test')]
        training = [row for row in members if row['split'] == 'train']
        if len(held_out) != 1 or not training:
            raise ValueError('Only train-versus-one-held-out duplicate groups can be resolved automatically')
        kept = held_out[0]
        for row in training:
            excluded.add(row['image_id'])
            details.append({'excluded_image_id': row['image_id'], 'excluded_split': 'train',
                            'retained_image_id': kept['image_id'], 'retained_split': kept['split'],
                            'class_id': row['class_id'], 'decoded_image_sha256': next(iter(digests)),
                            'reason': 'Exact training/held-out overlap; retain held-out membership'})
    return [row for row in rows if row['image_id'] not in excluded], details


def prepare(source: Path, output: Path, evidence: str, include_test=False, duplicate_audit=None):
    if not evidence.strip():
        raise ValueError('Record the confirmation for using official splits')
    source = source.resolve()
    if not source.is_dir() or output.resolve().exists():
        raise ValueError('Source must exist and output must be a new directory')
    paths = {}
    for path in source.rglob('*'):
        if path.is_file():
            paths.setdefault(path.name, []).append(path)

    def unique(name):
        matches = paths.get(name, [])
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name}, found {len(matches)}')
        return matches[0]

    rows, label_paths = [], []
    splits = [('train', 'Training'), ('val', 'Validation')]
    if include_test:
        splits.append(('test', 'Test'))
    for split, title in splits:
        labels = unique(f'ISIC-2017_{title}_Part3_GroundTruth.csv')
        parse_labels(labels.read_bytes())
        label_paths.append(labels)
        with labels.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                image_id = row['image_id']
                class_id = 0 if float(row['melanoma']) else 1 if float(row['seborrheic_keratosis']) else 2
                rows.append({'image_id': image_id, 'patient_id': '', 'split': split,
                             'class_id': class_id, 'image_path': str(unique(image_id+'.jpg')),
                             'mask_path': str(unique(image_id+'_segmentation.png'))})
    original_count = len(rows)
    exclusions = []
    if duplicate_audit is not None:
        rows, exclusions = resolve_train_overlap(rows, duplicate_audit)
        print(json.dumps({'duplicate_exclusions': exclusions}, indent=2))
    if not rows:
        raise ValueError('No images remain after overlap filtering')
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp) / 'manifest_official.csv'
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for labels in label_paths:
            shutil.copy2(labels, Path(temp) / labels.name)
        export(manifest, output, list(CLASS_NAMES), 'official-isic', evidence, temp)
        shutil.copy2(manifest, output / 'manifest_official.csv')
    audit_path = output / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit.update(original_manifest_images=original_count, duplicate_exclusions=exclusions,
                 duplicate_policy='exclude exact train/held-out overlaps from train only; keep held-out unchanged')
    if duplicate_audit is not None:
        audit['duplicate_audit_sha256'] = sha256(duplicate_audit)
        save_json(output / 'duplicate_source_audit.json', json.loads(Path(duplicate_audit).read_text(encoding='utf-8')))
    save_json(audit_path, audit)
    print(json.dumps({'exported_images': audit['images'], 'counts': audit['counts'],
                      'excluded_training_images': len(exclusions), 'audit': str(audit_path)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-evidence', required=True)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--duplicate-audit', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output, args.split_evidence, args.include_test, args.duplicate_audit)
