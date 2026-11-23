#!/usr/bin/env python3
"""Prepare a patient-grouped ADNI split after review of the supplied split.

Python 3.11; standard library only. Input: existing private inventory + metadata.
Never edits the source dataset, prints participant values, or trains a model.
`plan` writes PRIVATE draft assignments; `finalize` requires explicit human
confirmation of the subject-ID interpretation and tutor approval of the change.
Computer checks establish consistency, not the semantics of medical identifiers.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

FIELDS = ('raw', 'masked', 'c1', 'c2', 'c3', 'c4', 'c5')
FOLDERS = ('train/AD', 'train/NC', 'test/AD', 'test/NC')
PARTITIONS = ('train', 'validation', 'test')
SUBJECT = re.compile(r'(?<![A-Za-z0-9])(\d{3}_S_\d{4,})(?![A-Za-z0-9])')
COLUMNS = ('image_path', 'subject_id', 'scan_id', 'slice_index', 'source_split',
           'source_class', 'metadata_label', 'label', 'split')
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff'}
VERSION = 1
SEED = 42


class CheckError(Exception):
    """An aggregate, identifier-free error safe for the shared summary."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, 'A JSON object contains a duplicate key.')
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=reject_duplicates)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def subject_for_record(record: Any) -> tuple[str, bool]:
    require(isinstance(record, dict), 'A matched metadata record is not an object.')
    tokens = []
    for field in FIELDS:
        value = record.get(field)
        require(isinstance(value, str), 'A matched record lacks a required path-string field.')
        found = set(SUBJECT.findall(value))
        require(len(found) == 1, 'A metadata path has missing or ambiguous subject tokens.')
        tokens.append(next(iter(found)))
    require(len(set(tokens)) == 1, 'Subject tokens disagree across the seven fields of a record.')
    # Additional evidence only; this does not establish the identifier semantics.
    in_raw_directory = tokens[0] in record['raw'].replace('\\', '/').split('/')[:-1]
    return tokens[0], in_raw_directory


def validate_dataset(root: Path, audit_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-use image decoding evidence, but bind it to current file bytes via SHA-256."""
    inventory = read_json(audit_dir / 'private_inventory.json')
    previous = read_json(audit_dir / 'audit_summary.json')
    require(previous.get('audit_execution_completed') is True, 'The previous audit did not finish.')
    require(not previous.get('issues_for_review'), 'The previous audit contains issues requiring review.')
    metadata_bytes = (root / 'meta_data_with_label.json').read_bytes()
    require(sha(metadata_bytes) == inventory.get('metadata_sha256'),
            'Metadata changed since the previous audit. Do not use stale evidence.')
    metadata = json.loads(metadata_bytes.decode('utf-8-sig'), object_pairs_hook=reject_duplicates)
    require(isinstance(metadata, dict) and bool(metadata), 'Expected metadata keyed by image/scan prefix.')
    rows = inventory.get('images')
    require(isinstance(rows, list) and bool(rows), 'Private image inventory is empty or unsupported.')
    require(len(rows) == previous.get('images', {}).get('total_images'), 'Audit/inventory counts disagree.')

    actual_paths = set()
    for folder in FOLDERS:
        directory = root / 'AD_NC' / folder
        require(directory.is_dir(), 'An expected class directory is missing.')
        for path in directory.iterdir():
            require(not path.is_dir(), 'Nested directories appeared after the image audit.')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                actual_paths.add(path.relative_to(root).as_posix())
    inventory_paths = [row.get('relative_path') for row in rows]
    require(all(isinstance(p, str) for p in inventory_paths), 'Inventory paths are invalid.')
    require(len(set(inventory_paths)) == len(inventory_paths), 'Inventory repeats an image path.')
    require(set(inventory_paths) == actual_paths, 'Dataset image paths changed after the audit.')

    mapped = []
    scan_info = {}
    subject_labels: dict[str, set[int]] = defaultdict(set)
    old_subjects = {split: set() for split in ('train', 'test')}
    file_hashes, pixel_hashes = set(), set()
    raw_directory_evidence = 0
    original_counts: Counter[str] = Counter()
    for index, row in enumerate(sorted(rows, key=lambda r: r['relative_path']), 1):
        require(row.get('status') == 'decoded' and not row.get('warning_types'),
                'A previously audited image failed or produced a decoder warning.')
        relative = PurePosixPath(row['relative_path'])
        require(not relative.is_absolute() and '..' not in relative.parts and len(relative.parts) == 4,
                'A dataset path is not a safe relative class/image path.')
        require(relative.parts[0] == 'AD_NC', 'A path is outside the expected AD_NC structure.')
        _, source_split, source_class, name = relative.parts
        require(f'{source_split}/{source_class}' in FOLDERS, 'Unexpected source class or partition.')
        require(source_split == row.get('split') and source_class == row.get('folder_label'),
                'Inventory folder labels disagree with the path.')
        require(relative.suffix.lower() in IMAGE_EXTENSIONS, 'Unexpected image extension.')
        prefix, sep, suffix = relative.stem.rpartition('_')
        require(bool(sep and prefix and suffix.isdecimal()), 'A filename does not follow prefix_slice format.')
        require(prefix == row.get('prefix') and int(suffix) == row.get('slice_suffix'),
                'Parsed image identifiers disagree with the previous audit.')
        current = (root / relative).resolve()
        require(root in current.parents, 'An image path resolves outside the dataset.')
        raw = current.read_bytes()
        require(len(raw) == row.get('bytes') and sha(raw) == row.get('file_sha256'),
                'An image differs from its audited bytes. Stop and investigate.')
        require((row.get('width'), row.get('height'), row.get('mode')) == (256, 240, 'L'),
                'Image dimensions/mode differ from this reviewed dataset configuration.')
        file_hash = row['file_sha256']
        pixel_hash = row.get('pixel_sha256')
        require(isinstance(pixel_hash, str) and len(pixel_hash) == 64, 'A decoded-pixel hash is missing.')
        require(file_hash not in file_hashes and pixel_hash not in pixel_hashes,
                'Exact duplicate evidence needs review before preparing a split.')
        file_hashes.add(file_hash)
        pixel_hashes.add(pixel_hash)

        require(prefix in metadata, 'An image prefix has no exact metadata root-key match.')
        if prefix not in scan_info:
            record = metadata[prefix]
            subject_id, raw_evidence = subject_for_record(record)
            code = record.get('label')
            require(type(code) is int and code in (0, 2), 'A matched record has an unsupported diagnosis code.')
            scan_info[prefix] = (subject_id, code, source_class, source_split)
            raw_directory_evidence += int(raw_evidence)
        subject_id, code, scan_class, scan_source = scan_info[prefix]
        require(scan_class == source_class and scan_source == source_split,
                'One scan prefix appears in multiple original folders.')
        expected_code = 2 if source_class == 'AD' else 0
        require(code == expected_code, 'Metadata diagnosis and folder class disagree.')
        label = 1 if source_class == 'AD' else 0
        subject_labels[subject_id].add(label)
        require(len(subject_labels[subject_id]) == 1,
                'A subject has both diagnostic classes. Do not majority-label or split visits independently.')
        old_subjects[source_split].add(subject_id)
        original_counts[f'{source_split}/{source_class}'] += 1
        mapped.append(dict(image_path=relative.as_posix(), subject_id=subject_id,
                           scan_id=prefix, slice_index=int(suffix), source_split=source_split,
                           source_class=source_class, metadata_label=code, label=label))
        if index % 5000 == 0 or index == len(rows):
            print(f'Confirmed existing audit hashes and metadata mapping: {index}/{len(rows)}', flush=True)

    per_scan = Counter(row['scan_id'] for row in mapped)
    require(set(per_scan.values()) == {20}, 'This reviewed dataset should contain 20 images per scan prefix.')
    subjects_by_class = Counter(next(iter(labels)) for labels in subject_labels.values())
    evidence = {
        'images': len(mapped), 'scan_prefixes': len(scan_info),
        'metadata_records': len(metadata), 'unused_metadata_records': len(metadata) - len(scan_info),
        'metadata_sha256': sha(metadata_bytes),
        'images_per_scan': dict(Counter(map(str, per_scan.values()))),
        'original_image_counts': dict(original_counts),
        'all_matched_records_agree_across_seven_path_fields': True,
        'scan_records_with_subject_as_raw_directory_component': raw_directory_evidence,
        'candidate_subjects': len(subject_labels),
        'candidate_subjects_by_class': {'NC': subjects_by_class[0], 'AD': subjects_by_class[1]},
        'source_train_candidate_subjects': len(old_subjects['train']),
        'source_test_candidate_subjects': len(old_subjects['test']),
        'source_train_test_shared_candidate_subjects': len(old_subjects['train'] & old_subjects['test']),
        'candidate_subjects_with_conflicting_labels': 0,
        'current_image_bytes_match_local_audit': True,
        'source_rangpur_copy_hash_verified': False,
    }
    return mapped, evidence


def counts_for_class(n: int) -> dict[str, int]:
    require(n >= 7, 'Too few subjects in a class for this fixed 70/15/15 plan; review the design.')
    numerators = (70, 15, 15)
    counts = [n * value // 100 for value in numerators]
    remainders = [n * value % 100 for value in numerators]
    ranking = sorted(range(3), key=lambda i: (-remainders[i], i))
    for i in ranking[:n - sum(counts)]:
        counts[i] += 1
    require(all(counts), 'An empty class/partition would result from this split.')
    return dict(zip(PARTITIONS, counts))


def assign_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixed hash-ranked, class-stratified grouping. No model-based seed selection."""
    subject_class = {}
    for row in rows:
        sid, label = row['subject_id'], row['label']
        require(label in (0, 1), 'A binary label is invalid.')
        require(sid not in subject_class or subject_class[sid] == label, 'A subject has inconsistent labels.')
        subject_class[sid] = label
    assignments = {}
    for label in (0, 1):
        ids = [sid for sid in subject_class if subject_class[sid] == label]
        ids.sort(key=lambda sid: (sha(f'COMP3710|seed={SEED}|class={label}|{sid}'.encode()), sid))
        counts = counts_for_class(len(ids))
        offset = 0
        for split in PARTITIONS:
            for sid in ids[offset:offset + counts[split]]:
                assignments[sid] = split
            offset += counts[split]
    return [dict(row, split=assignments[row['subject_id']])
            for row in sorted(rows, key=lambda r: r['image_path'])]


def split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    all_subjects, all_scans = [], []
    for split in PARTITIONS:
        subset = [r for r in rows if r['split'] == split]
        subjects = {r['subject_id'] for r in subset}
        scans = {r['scan_id'] for r in subset}
        all_subjects.append(subjects)
        all_scans.append(scans)
        result[split] = {
            'candidate_subjects': len(subjects), 'scans': len(scans), 'images': len(subset),
            'by_class': {name: {
                'candidate_subjects': len({r['subject_id'] for r in subset if r['label'] == label}),
                'scans': len({r['scan_id'] for r in subset if r['label'] == label}),
                'images': sum(r['label'] == label for r in subset)}
                for name, label in [('NC', 0), ('AD', 1)]}}
    for i in range(3):
        for j in range(i + 1, 3):
            require(not (all_subjects[i] & all_subjects[j]), 'Candidate subjects overlap in the proposed split.')
            require(not (all_scans[i] & all_scans[j]), 'Scans overlap in the proposed split.')
    return result


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def request_text(evidence: dict[str, Any]) -> str:
    e = evidence
    return (
        "Subject: COMP3710 ADNI patient mapping and corrected evaluation split\n\n"
        "My feasibility review for ConvNeXt versus a small CNN was approved. Before training, "
        "I audited the supplied JPEG subset. The existing train/test folders are disjoint by "
        "filename prefix, but not by the ADNI-format subject tokens found in the metadata.\n\n"
        f"All {e['images']:,} images map to {e['scan_prefixes']:,} exact metadata root keys. "
        "For every matched record, raw, masked and c1-c5 contain the same single XXX_S_XXXX "
        "style subject token. "
        f"There are {e['candidate_subjects']} distinct candidate subjects; "
        f"{e['source_train_test_shared_candidate_subjects']} appear in both original partitions. "
        "There are no contradictory AD/NC folder labels for these groups. "
        "Metadata label 0 matches NC, and label 2 matches AD throughout the JPEG subset.\n\n"
        "Could you confirm that this path token is the participant identifier, that NC is the "
        "cognitively normal control category described as CN in the brief, and approve rebuilding "
        "the evaluation partitions by participant? I propose approximately 70% training, 15% "
        "validation and 15% test within each class, using a fixed seed-42 hash ranking. Every "
        "participant's scans and slices stay together. I will retain all supplied JPEGs and "
        "record their original folder; no source files will be moved or deleted.\n\n"
        "I propose patient-level accuracy as the primary endpoint (average slice probabilities "
        "within each scan, then average scan probabilities within each patient), alongside "
        "scan- and slice-level metrics. Please confirm how you want the 0.80 accuracy target "
        "reported. The baseline, ConvNeXt and referral comparison will use identical partitions. "
        "No model has been trained or evaluated on these test images.\n"
    )


def run(args: argparse.Namespace) -> None:
    root, audit, out = args.root.resolve(), args.audit_dir.resolve(), args.output_dir.resolve()
    require(root.is_dir() and audit.is_dir(), 'Dataset or prior-audit directory is missing.')
    require(out != root and root not in out.parents, 'Output must be outside the dataset.')
    require(not out.exists(), 'Choose a new output directory; existing evidence will not be overwritten.')
    rows, evidence = validate_dataset(root, audit)
    drafted = assign_split(rows)
    encoded = csv_bytes(drafted)
    counts = split_counts(drafted)
    common = {
        'format_version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'mapping_structural_checks_passed': True, 'evidence': evidence,
        'mapping_rule': 'JPEG prefix = exact metadata root key; one matching subject token across raw/masked/c1-c5',
        'model_label_map': {'NC': 0, 'AD': 1}, 'metadata_to_model_labels': {'0': 0, '2': 1},
        'seed': SEED, 'fractions_by_candidate_subject_class': {'train': 0.70, 'validation': 0.15, 'test': 0.15},
        'algorithm': 'Within each class sort by SHA256(COMP3710|seed=42|class=<0-or-1>|<subject>); largest-remainder counts',
        'rounding_rule': 'Largest remainder; ties use train, validation, test order',
        'proposed_or_final_counts': counts, 'new_pairwise_candidate_subject_overlap': 0,
        'draft_manifest_sha256': sha(encoded), 'dataset_files_modified': False,
        'training_ready': False,
    }
    if args.action == 'plan':
        out.mkdir(parents=True)
        (out / 'private_draft_manifest.csv').write_bytes(encoded)
        (out / 'tutor_request.txt').write_text(request_text(evidence), encoding='utf-8')
        common.update(status='DRAFT_AWAITING_SUBJECT_RULE_AND_TUTOR_CONFIRMATION',
                      patient_identifier_semantics_confirmed=False,
                      tutor_split_change_approved=False, splits_finalized=False)
        write_json(out / 'split_plan_summary.json', common)
        print('PATIENT_SPLIT_PLAN_CREATED')
        print('TUTOR_CONFIRMATION_REQUIRED_BEFORE_FINALIZING')
    else:
        require(args.confirm_patient_id_rule and args.tutor_approved,
                'Finalization requires explicit confirmation of the patient rule and tutor split approval.')
        require(args.plan_dir is not None and args.approval_note is not None,
                'Finalization requires the prior plan and a private approval note.')
        plan = read_json(args.plan_dir / 'split_plan_summary.json')
        require(plan.get('format_version') == VERSION and plan.get('seed') == SEED,
                'Unsupported or altered split plan.')
        require(plan.get('status') == 'DRAFT_AWAITING_SUBJECT_RULE_AND_TUTOR_CONFIRMATION',
                'The supplied directory is not the expected draft plan.')
        require(plan.get('evidence') == evidence, 'The dataset/mapping evidence differs from the reviewed plan.')
        require(plan.get('draft_manifest_sha256') == sha(encoded)
                and (args.plan_dir / 'private_draft_manifest.csv').read_bytes() == encoded,
                'Draft assignments differ from the fixed split plan.')
        note = args.approval_note.read_text(encoding='utf-8-sig').strip()
        require(len(note) >= 20, 'Record the actual tutor approval and its date in the private note.')
        out.mkdir(parents=True)
        manifests = {}
        for split in PARTITIONS:
            content = csv_bytes([r for r in drafted if r['split'] == split])
            name = f'{split}.csv'
            (out / name).write_bytes(content)
            manifests[name] = sha(content)
        (out / 'private_approval.txt').write_text(note + '\n', encoding='utf-8')
        common.update(status='FINALIZED_UNDER_USER_CONFIRMED_PATIENT_RULE_AND_TUTOR_APPROVAL',
                      patient_identifier_semantics_confirmed_by_user=True,
                      tutor_split_change_approved_as_reported_by_user=True,
                      zero_group_overlap_verified=True, splits_finalized=True,
                      manifest_sha256=manifests,
                      caution='Approval assertions are supplied by the user, not independently authenticated. Loader/model tests remain outstanding.')
        write_json(out / 'split_summary.json', common)
        print('PATIENT_SPLITS_FINALIZED')
        print('LOADER_AND_TRAINING_CHECKS_STILL_REQUIRED')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'finalize'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan-dir', type=Path)
    parser.add_argument('--approval-note', type=Path)
    parser.add_argument('--confirm-patient-id-rule', action='store_true')
    parser.add_argument('--tutor-approved', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except CheckError as exc:
        print(f'STOP: {exc}', file=sys.stderr)
    except Exception as exc:
        print(f'STOP: {type(exc).__name__}. Check local inputs; do not upload raw medical records.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
