#!/usr/bin/env python3
"""Read-only ADNI integrity and candidate metadata-mapping audit (Python 3.11).

The shared summary contains aggregate counts and conservative field-name previews,
not patient/image IDs, record values, image previews or model predictions. Private
inventory/key/error files MUST remain outside Git and cloud-synchronised folders.
This program does not choose patient definitions, create splits, or train models.
All inferred joins/groupings are candidates requiring review, never proof of identity.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import re
from pathlib import Path
import sys
import warnings
from typing import Any

from PIL import Image, ImageFile

IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff'}
FOLDERS = ('train/AD', 'train/NC', 'test/AD', 'test/NC')
# A recognisable ADNI-style token is only a candidate identifier, not provenance.
SUBJECT_PATTERN = re.compile(r'(?<![A-Za-z0-9])(\d{3}_S_\d{4,})(?![A-Za-z0-9])')
SUBJECT_NAMES = {'subject', 'subjectid', 'subjectidentifier', 'patient', 'patientid',
                 'patientidentifier', 'participant', 'participantid', 'ptid', 'rid'}
LABEL_NAMES = {'label', 'class', 'group', 'diagnosis', 'diagnosticgroup', 'dx'}
SAFE_LABELS = {'AD', 'NC', 'CN', 'MCI', 'EMCI', 'LMCI', 'SMC', 'NORMAL', 'CONTROL',
               'ALZHEIMER', 'ALZHEIMERS', "ALZHEIMER'S", 'COGNITIVELY NORMAL'}


def compact_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]', '', value.casefold())


def scalar_key(value: Any) -> str | None:
    """Conservative join candidates; never discard arbitrary leading digits/zeros."""
    if isinstance(value, str):
        return value.strip() or None
    if type(value) is int:
        return str(value)
    return None


def subject_in_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    tokens = set(SUBJECT_PATTERN.findall(value))
    return next(iter(tokens)) if len(tokens) == 1 else None


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key; cannot use a silently overwritten record.')
        result[key] = value
    return result


def load_metadata(path: Path) -> tuple[list[tuple[str | None, dict[str, Any]]], str]:
    raw = path.read_bytes()
    obj = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=reject_duplicate_keys)
    if isinstance(obj, dict):
        records = list(obj.items())
    elif isinstance(obj, list):
        records = [(None, item) for item in obj]
    else:
        raise ValueError('Unsupported metadata root: expected dictionary or list of records.')
    if not records or any(not isinstance(value, dict) for _, value in records):
        raise ValueError('Expected a nonempty flat collection of record dictionaries.')
    return records, hashlib.sha256(raw).hexdigest()


def safe_field_name(key: str, occurrence: int, record_count: int) -> str:
    """Preview repeated schema names, never paths or apparent identifiers."""
    if (occurrence >= max(2, (4 * record_count + 4) // 5)
            and re.fullmatch(r'[A-Za-z0-9_ .():#\[\]-]{1,80}', key)
            and re.search(r'[A-Za-z]', key)
            and not SUBJECT_PATTERN.search(key)):
        return key
    return '<withheld: inspect private_field_key.json locally>'


def classify_label(value: Any) -> str:
    """Only expose small integer class codes and an explicit diagnostic vocabulary."""
    if type(value) is int and -1 <= value <= 10:
        return f'integer:{value}'
    if isinstance(value, str) and value.strip().upper() in SAFE_LABELS:
        return f'text:{value.strip().upper()}'
    return '<other value withheld>'


def duplicate_summary(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    return {
        'duplicate_groups': len(duplicates),
        'files_in_duplicate_groups': sum(map(len, duplicates)),
        'extra_copies_beyond_one_per_group': sum(len(group) - 1 for group in duplicates),
        'groups_crossing_train_test': sum(len({r['split'] for r in g}) > 1 for g in duplicates),
        'groups_crossing_AD_NC_labels': sum(len({r['folder_label'] for r in g}) > 1 for g in duplicates),
        'groups_spanning_multiple_filename_prefixes': sum(len({r['prefix'] for r in g}) > 1 for g in duplicates),
    }


def inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    """Decode all supplied images for integrity only; never display any image."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    rows: list[dict[str, Any]] = []
    details: list[dict[str, str]] = []
    folder_stats = {}
    for folder in FOLDERS:
        directory = root / 'AD_NC' / folder
        if not directory.is_dir():
            raise FileNotFoundError('A required class directory is missing.')
        entries = sorted(directory.iterdir())
        files = [p for p in entries if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        nested = sum(p.is_dir() for p in entries)
        other = sum(p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS for p in entries)
        sizes: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        formats: Counter[str] = Counter()
        per_prefix: Counter[str] = Counter()
        bad_name = failed = warned = total_bytes = 0
        split, label = folder.split('/')
        for number, path in enumerate(files, 1):
            prefix, separator, suffix = path.stem.rpartition('_')
            name_ok = bool(separator and prefix and suffix.isdecimal())
            if name_ok:
                per_prefix[prefix] += 1
            else:
                bad_name += 1
            row: dict[str, Any] = {
                'relative_path': path.relative_to(root).as_posix(), 'split': split,
                'folder_label': label, 'prefix': prefix if name_ok else None,
                'slice_suffix': int(suffix) if name_ok else None, 'status': 'not_read',
            }
            try:
                raw = path.read_bytes()
                total_bytes += len(raw)
                row['bytes'] = len(raw)
                row['file_sha256'] = hashlib.sha256(raw).hexdigest()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    with Image.open(io.BytesIO(raw)) as img:
                        img.verify()
                    with Image.open(io.BytesIO(raw)) as img:
                        img.load()
                        shape = f'{img.width}x{img.height}'
                        sizes[shape] += 1
                        modes[img.mode] += 1
                        formats[str(img.format)] += 1
                        row.update(width=img.width, height=img.height, mode=img.mode)
                        payload = f'{shape}|{img.mode}|'.encode() + img.tobytes()
                        # Include palette bytes so indexed images with different colours do not collide.
                        if img.mode == 'P' and img.getpalette() is not None:
                            payload += bytes(img.getpalette())
                        row['pixel_sha256'] = hashlib.sha256(payload).hexdigest()
                row['status'] = 'decoded'
                if captured:
                    warned += 1
                    row['warning_types'] = sorted({w.category.__name__ for w in captured})
            except Exception as exc:
                failed += 1
                row['status'] = 'failed'
                row['error_type'] = type(exc).__name__
                details.append({'relative_path': row['relative_path'], 'error_type': type(exc).__name__})
            rows.append(row)
            if number % 1000 == 0 or number == len(files):
                print(f'{folder}: checked {number}/{len(files)} images', flush=True)
        folder_stats[folder] = {
            'image_files': len(files), 'bytes_read': total_bytes,
            'decoded_images': len(files) - failed, 'failed_images': failed,
            'images_with_decoder_warnings': warned,
            'nested_directories_not_traversed': nested, 'other_direct_files_not_read': other,
            'unrecognised_filename_pattern_count': bad_name,
            'filename_prefixes_NOT_patients': len(per_prefix),
            'images_per_prefix_histogram': dict(sorted(Counter(per_prefix.values()).items())),
            'width_x_height_counts': dict(sizes), 'mode_counts': dict(modes),
            'format_counts': dict(formats),
        }
    summary = {
        'folders': folder_stats, 'total_images': len(rows),
        'test_images_used_only_for_integrity_not_modelling': True,
        'encoded_file_duplicates': duplicate_summary(rows, 'file_sha256'),
        'decoded_pixel_duplicates': duplicate_summary(rows, 'pixel_sha256'),
        'duplicate_interpretation': 'Exact matches are flags for review, not proof of patient overlap. No files deleted.',
    }
    return rows, summary, details


def metadata_audit(records: list[tuple[str | None, dict[str, Any]]],
                   rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    counts = Counter(key for _, record in records for key in record)
    keys = sorted(counts)
    field_key = {f'F{i:03d}': key for i, key in enumerate(keys, 1)}
    schema = []
    for fid, key in field_key.items():
        values = [record[key] for _, record in records if key in record]
        strings = [v for v in values if isinstance(v, str)]
        schema.append({
            'field_id': fid, 'field_name': safe_field_name(key, counts[key], len(records)),
            'records_with_field': counts[key],
            'value_types': dict(Counter(type(v).__name__ for v in values)),
            'missing_or_blank_count': sum(v is None or (isinstance(v, str) and not v.strip()) for v in values),
            'strings_looking_like_paths': sum('/' in v or '\\' in v for v in strings),
            'strings_with_exactly_one_ADNI_style_subject_token': sum(subject_in_path(v) is not None for v in strings),
            'strings_with_multiple_distinct_ADNI_style_subject_tokens': sum(len(set(SUBJECT_PATTERN.findall(v))) > 1 for v in strings),
        })
    by_prefix: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row['prefix'] is not None:
            by_prefix[row['prefix']].append(row)
    prefixes = set(by_prefix)
    # A method is reported only when it matches at least one observed prefix.
    sources: list[tuple[str, list[Any]]] = [('root_key', [key for key, _ in records])]
    sources.extend((fid, [record.get(key) for _, record in records]) for fid, key in field_key.items())
    subject_sources: list[tuple[str, list[str | None]]] = []
    for fid, key in field_key.items():
        values = [record.get(key) for _, record in records]
        if compact_name(key) in SUBJECT_NAMES:
            subject_sources.append((f'{fid}:explicit_subject_named_field', [scalar_key(v) for v in values]))
        extracted = [subject_in_path(v) for v in values]
        if any(v is not None for v in extracted):
            subject_sources.append((f'{fid}:ADNI_style_token_in_string', extracted))
    root_subjects = [subject_in_path(key) for key, _ in records]
    if any(root_subjects):
        subject_sources.append(('root_key:ADNI_style_token_in_string', root_subjects))
    joins = []
    for source, values in sources:
        for method in ('exact_text', 'remove_single_leading_I_from_numeric_image_id', 'jpeg_basename_prefix'):
            index: dict[str, list[int]] = defaultdict(list)
            for row_index, value in enumerate(values):
                token = scalar_key(value)
                if token is None:
                    continue
                if method == 'remove_single_leading_I_from_numeric_image_id':
                    match = re.fullmatch(r'I(\d+)', token)
                    if not match:
                        continue
                    token = match.group(1)
                elif method == 'jpeg_basename_prefix':
                    basename = token.replace('\\', '/').rsplit('/', 1)[-1]
                    match = re.fullmatch(r'(.+)_(\d+)\.(?:jpeg|jpg|png)', basename, re.IGNORECASE)
                    if not match:
                        continue
                    token = match.group(1)
                if token in prefixes:
                    index[token].append(row_index)
            if not index:
                continue
            matched = {p: ids[0] for p, ids in index.items() if len(ids) == 1}
            candidate: dict[str, Any] = {
                'source': source, 'transformation': method,
                'status': 'CANDIDATE_ONLY_REQUIRES_SEMANTIC_REVIEW',
                'uniquely_matched_filename_prefixes': len(matched),
                'ambiguous_prefixes_multiple_metadata_records': sum(len(ids) > 1 for ids in index.values()),
                'unmatched_filename_prefixes': len(prefixes - index.keys()),
                'total_distinct_filename_prefixes_NOT_patients': len(prefixes),
                'images_with_unique_prefix_join': sum(len(by_prefix[p]) for p in matched),
                'grouping_candidates': [], 'label_crosstabs': {},
            }
            for fid, key in field_key.items():
                if compact_name(key) in LABEL_NAMES:
                    tab: dict[str, Counter[str]] = defaultdict(Counter)
                    for prefix, i in matched.items():
                        for folder in {f"{r['split']}/{r['folder_label']}" for r in by_prefix[prefix]}:
                            tab[folder][classify_label(records[i][1].get(key))] += 1
                    candidate['label_crosstabs'][fid] = {
                        'unit': 'matched_filename_prefix_in_class_folder_NOT_patients',
                        'counts': {f: dict(c) for f, c in tab.items()},
                    }
            for subject_source, subjects in subject_sources:
                groups = {folder: set() for folder in FOLDERS}
                resolved_prefixes = set()
                for prefix, i in matched.items():
                    subject = subjects[i]
                    if subject is None:
                        continue
                    resolved_prefixes.add(prefix)
                    for row in by_prefix[prefix]:
                        groups[f"{row['split']}/{row['folder_label']}"].add(subject)
                training = groups['train/AD'] | groups['train/NC']
                testing = groups['test/AD'] | groups['test/NC']
                candidate['grouping_candidates'].append({
                    'source': subject_source,
                    'status': 'CANDIDATE_TOKENS_NOT_VERIFIED_PATIENT_IDENTITIES',
                    'filename_prefixes_with_grouping_token': len(resolved_prefixes),
                    'filename_prefixes_without_verified_join_or_token': len(prefixes - resolved_prefixes),
                    'distinct_tokens_by_folder': {f: len(s) for f, s in groups.items()},
                    'distinct_train_tokens': len(training), 'distinct_test_tokens': len(testing),
                    'train_test_shared_tokens': len(training & testing),
                    'tokens_observed_under_both_AD_and_NC': len((groups['train/AD'] | groups['test/AD']) & (groups['train/NC'] | groups['test/NC'])),
                })
            joins.append(candidate)
    # No record values (including full paths or IDs) are copied to the shared report.
    return {
        'metadata_record_count': len(records), 'field_schema': schema,
        'join_candidates': joins,
        'schema_privacy_note': 'Repeated field names are shown, never scalar record values. Review before sharing.',
        'interpretation': 'No candidate join or subject definition is automatically accepted. No patient split is certified.',
    }, field_key


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New PRIVATE directory outside dataset, Git and cloud-synced folders.')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    if out == root or root in out.parents:
        parser.error('Output directory must not be inside the dataset.')
    if out.exists():
        parser.error('Use a new output directory. Existing audit evidence will not be overwritten.')
    out.mkdir(parents=True)
    report: dict[str, Any] = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'audit_execution_completed': False, 'patient_independence_verified': False,
        'source_copy_byte_for_byte_verified': False, 'training_ready': False,
        'splits_created': False, 'files_modified_in_dataset': False,
        'scope': 'Local integrity, exact duplicates and candidate metadata/grouping diagnostics only',
    }
    try:
        records, metadata_hash = load_metadata(root / 'meta_data_with_label.json')
        rows, images, errors = inventory(root)
        metadata, private_keys = metadata_audit(records, rows)
        report['images'] = images
        report['metadata'] = metadata
        report['audit_execution_completed'] = True
        issues = []
        if not rows:
            issues.append('No images were found.')
        for name, stats in images['folders'].items():
            for field in ('failed_images', 'images_with_decoder_warnings', 'nested_directories_not_traversed',
                          'other_direct_files_not_read', 'unrecognised_filename_pattern_count'):
                if stats[field]:
                    issues.append(f'{name}: {field}={stats[field]}')
            if not stats['image_files']:
                issues.append(f'{name}: no image files')
        for kind in ('encoded_file_duplicates', 'decoded_pixel_duplicates'):
            if images[kind]['groups_crossing_train_test']:
                issues.append(f'{kind}: exact duplicate groups cross train/test; investigate, do not auto-delete.')
        report['issues_for_review'] = issues
        report['next_action'] = 'Review summary and establish the correct metadata join/patient definition before splitting.'
        write_json(out / 'private_inventory.json', {
            'metadata_sha256': metadata_hash,
            'warning': 'PRIVATE image identifiers and paths. Do not upload or commit.', 'images': rows})
        write_json(out / 'private_field_key.json', {
            'warning': 'May contain sensitive/unusual dictionary keys. Review locally only.', 'fields': private_keys})
        write_json(out / 'private_errors.json', errors)
        write_json(out / 'audit_summary.json', report)
        print('AUDIT_RUN_COMPLETED')
        print('PATIENT_MAPPING_REVIEW_REQUIRED')
        print(f'Only share after review: {out / "audit_summary.json"}')
        return 0
    except Exception as exc:
        report['failure_type'] = type(exc).__name__
        report['next_action'] = 'Stop and share this summary; no raw metadata or patient images.'
        write_json(out / 'audit_summary.json', report)
        print(f'AUDIT_RUN_FAILED: {type(exc).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
