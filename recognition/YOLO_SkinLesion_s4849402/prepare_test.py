"""Prepare a fixed test subset without copying training/validation images."""

import argparse
import csv
import hashlib
import os
import shutil
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from audit_labels import parse_labels
from dataset import mask_to_box
from modules import CLASS_NAMES
from utils import read_config, save_json, sha256


def image_digest(path):
    with Image.open(path) as image:
        size = image.size
        digest = hashlib.sha256(str(size).encode() + image.convert('RGB').tobytes()).hexdigest()
    return digest, size


def extract_needed(archives, raw, wanted):
    entries = {}
    for archive in archives:
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                name = Path(info.filename).name
                if not info.is_dir() and name in wanted:
                    if name in entries:
                        raise ValueError(f'Duplicate archive member: {name}')
                    entries[name] = (archive, info)
    if set(entries) != wanted:
        raise ValueError(f'Missing image/mask members: {sorted(wanted-set(entries))[:10]}')
    raw.mkdir(parents=True, exist_ok=True)
    required = sum(info.file_size for name, (_, info) in entries.items() if not (raw/name).exists())
    if shutil.disk_usage(raw).free < required + 256*1024*1024:
        raise OSError('Insufficient free space for selected image/mask members')
    for archive in archives:
        with zipfile.ZipFile(archive) as stream:
            for name, (source, info) in entries.items():
                if source != archive:
                    continue
                if Path(name).name != name or name in ('.', '..'):
                    raise ValueError('Invalid archive basename')
                target = raw/name
                if target.exists():
                    crc = 0
                    with target.open('rb') as existing:
                        for block in iter(lambda: existing.read(1024*1024), b''):
                            crc = zlib.crc32(block, crc)
                    if target.stat().st_size != info.file_size or crc != info.CRC:
                        raise ValueError(f'Existing extracted file is inconsistent: {target}')
                    continue
                partial = raw/(name+'.part')
                with stream.open(info) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst)  # ZipExtFile verifies CRC when fully read.
                partial.replace(target)
        print(f'Checked/extracted required files: {archive.name}', flush=True)


def prepare_test(archives, raw, training_data, output, expected_count=600):
    archives, raw, output = Path(archives), Path(raw).resolve(), Path(output).resolve()
    training_data = Path(training_data).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new test output directory: {output}')
    config = read_config(training_data/'data.yaml')
    if tuple(config['names']) != CLASS_NAMES:
        raise ValueError('Training class order mismatch')
    training_audit = read_config(training_data/'audit.json')
    labels_file = archives/'ISIC-2017_Test_v2_Part3_GroundTruth.csv'
    parse_labels(labels_file.read_bytes())
    with labels_file.open(encoding='utf-8-sig', newline='') as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != expected_count:
        raise ValueError(f'Expected {expected_count} test labels, found {len(labels)}')
    ids = {r['image_id'] for r in labels}
    if any(not name.startswith('ISIC_') or not name.replace('_','').isalnum() for name in ids):
        raise ValueError('Invalid ISIC image ID')
    wanted = {name+suffix for name in ids for suffix in ('.jpg', '_segmentation.png')}
    zips = [archives/'ISIC-2017_Test_v2_Data.zip', archives/'ISIC-2017_Test_v2_Part1_GroundTruth.zip']
    extract_needed(zips, raw, wanted)
    existing_hashes, existing_ids = defaultdict(list), set()
    for split in ('train', 'val'):
        folder = Path(config.get('path', training_data))/config[split]
        pictures = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
        if {p.stem for p in pictures} != set(training_audit['image_ids'][split]):
            raise ValueError('Training/validation inventory changed')
        for i, path in enumerate(pictures, 1):
            cls = int((training_data/'labels'/split/(path.stem+'.txt')).read_text().split()[0])
            digest, _ = image_digest(path)
            existing_hashes[digest].append({'image_id': path.stem, 'split': split, 'class_id': cls})
            existing_ids.add(path.stem)
            if i % 250 == 0:
                print(f'Hashing reference {split}: {i}/{len(pictures)}', flush=True)
    retained, excluded, seen_test = [], [], {}
    for row in sorted(labels, key=lambda row: row['image_id']):
        name = row['image_id']
        cls = 0 if float(row['melanoma']) else 1 if float(row['seborrheic_keratosis']) else 2
        digest, size = image_digest(raw/(name+'.jpg'))
        matches = existing_hashes.get(digest, [])
        if name in existing_ids and not matches:
            raise ValueError('Same image ID has different decoded pixels across splits')
        if matches and any(r['class_id'] != cls for r in matches):
            raise ValueError(f'Conflicting diagnosis across identical images: {name}')
        if matches:
            excluded.append({'image_id': name, 'reason': 'exact overlap with train/val', 'matches': matches,
                             'decoded_sha256': digest})
            continue
        if digest in seen_test:
            prior = seen_test[digest]
            if prior['class_id'] != cls:
                raise ValueError('Conflicting labels within duplicate test group')
            excluded.append({'image_id': name, 'reason': 'exact duplicate within test', 'matches': [prior],
                             'decoded_sha256': digest})
            continue
        with Image.open(raw/(name+'_segmentation.png')) as mask:
            if mask.size != size:
                raise ValueError(f'Test image/mask mismatch: {name}')
            box = mask_to_box(mask)
        seen_test[digest] = {'image_id': name, 'split': 'test', 'class_id': cls}
        retained.append((name, cls, box))
    if not retained:
        raise ValueError('No independent test images remain')
    (output/'images/test').mkdir(parents=True)
    (output/'labels/test').mkdir(parents=True)
    for name, cls, box in retained:
        os.link(raw/(name+'.jpg'), output/'images/test'/(name+'.jpg'))
        (output/'labels/test'/(name+'.txt')).write_text(
            str(cls)+' '+' '.join(f'{x:.10f}' for x in box)+'\n', encoding='utf-8')
    test_config = {'path': output.as_posix(), 'names': list(CLASS_NAMES),
                   'train': str((training_data/'images/train').resolve()),
                   'val': str((training_data/'images/val').resolve()), 'test': 'images/test'}
    save_json(output/'data.yaml', test_config)
    report = {'original_test_images': len(labels), 'retained_test_images': len(retained),
              'class_counts': dict(Counter(CLASS_NAMES[cls] for _, cls, _ in retained)),
              'excluded_test_images': excluded, 'test_csv_sha256': sha256(labels_file),
              'training_audit_sha256': sha256(training_data/'audit.json'),
              'image_ids': [name for name, _, _ in retained], 'storage': 'hardlinks to extracted test JPGs',
              'protocol': 'Exclude exact overlaps and duplicate test images before any performance scoring',
              'limits': ['No near-duplicate guarantee or independent patient-ID verification',
                         'If exclusions occur, report a decontaminated subset, not the full official test set']}
    save_json(output/'audit.json', report)
    print(__import__('json').dumps(report, indent=2))
    print(f'Test data ready: {output / "data.yaml"}', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--training-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare_test(args.archives, args.raw, args.training_data, args.output)
