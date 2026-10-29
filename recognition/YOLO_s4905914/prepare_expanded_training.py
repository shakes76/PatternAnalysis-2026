"""Prepare a separate exploratory training expansion; preserve both held-out sets.

Unknown IDs stay blank. Known links are constraints, never imputed identities.
The restricted-cohort experiment/configuration/report are not rewritten.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from plan_group_splits import connected_components, historical_membership, read_metadata
from prepare_dataset import BOX_FIELDS, read_rows

CODE_FILES = (
    'train.py', 'dataset.py', 'evaluation.py', 'modules.py', 'runtime.py',
    'schedule.py', 'checkpoints.py', 'preprocessing.py', 'preflight.py',
    'audit_real_inputs.py', 'weights_registry.json', 'requirements.txt',
    'requirements-linux-a100.lock.txt',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(root, study, output):
    root, study, output = map(lambda p: Path(p).resolve(), (root, study, output))
    if study.exists() or output.exists():
        raise ValueError('Use new study and data directories; do not overwrite an experiment')
    source_code = root / 'recognition/YOLO_s4905914'
    baseline = root / 'data/known_lesion_v1'
    config_path = source_code / 'experiment_provisional.json'
    cfg = json.loads(config_path.read_text())
    if sha(config_path) != 'b304d0a6b69df04ef1fae4ecd9dda5a0beca7020cd8ff8c72c589535b960857d':
        raise ValueError('Restricted-cohort baseline changed')
    sources = json.loads((root / 'sources.json').read_text())
    for value in sources.values():
        if sha(root / value['path']) != value['sha256']:
            raise ValueError('Source snapshot differs')
    meta = read_metadata(root / sources['official_metadata']['path'])
    boxes = read_rows(root / sources['official_mask_boxes']['path'], 'image_id')
    history = historical_membership(root / 'data/source_snapshots/historical_splits', meta)
    if set(meta) != set(boxes):
        raise ValueError('Metadata and box coverage differ')
    original = {s: read_rows(baseline / f'{s}.csv', 'image_id')
                for s in ('train', 'val', 'reassessment')}
    for s in original:
        if sha(baseline / f'{s}.csv') != cfg['manifest_sha256'][s]:
            raise ValueError('Original split changed')
    held = set(original['val']) | set(original['reassessment'])
    blocked = {i for group in connected_components(meta) if held.intersection(group)
               for i in group}
    if blocked.intersection(original['train']):
        raise ValueError('Known link connects original train to held-out data')
    extra = sorted(i for i, row in meta.items()
                   if not row['lesion_id'] and row['diagnosis_1'] in ('Benign', 'Malignant')
                   and i not in blocked)
    if set(extra).intersection(set().union(*(set(r) for r in original.values()))):
        raise ValueError('Additional images must be outside all original splits')
    train = dict(original['train'])
    for i in extra:
        m, box = meta[i], boxes[i]
        x, y, w, h = (float(box[k]) for k in BOX_FIELDS)
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0
                or min(x-w/2, y-h/2) < -1e-7 or max(x+w/2, y+h/2) > 1+1e-7
                or Path(box['image_path']).stem != i):
            raise ValueError('Invalid source box or image path')
        train[i] = {'image_id': i, 'image_path': box['image_path'],
                    'class_id': '0' if m['diagnosis_1'] == 'Benign' else '1',
                    'diagnosis': m['diagnosis_1'], **{k: box[k] for k in BOX_FIELDS},
                    'lesion_id': m['lesion_id'], 'patient_id': m['patient_id'],
                    'historical_split': history[i]}
    all_rows = {**original, 'train': train}
    overlap = {}
    for field in ('lesion_id', 'patient_id'):
        groups = {s: {r[field] for r in rs.values() if r[field]} for s, rs in all_rows.items()}
        overlap[field] = {f'{a}_{b}': len(groups[a] & groups[b])
                          for a, b in [('train', 'val'), ('train', 'reassessment'), ('val', 'reassessment')]}
        if any(overlap[field].values()):
            raise ValueError('Known grouping crosses splits')
    output.mkdir(parents=True)
    for s in ('val', 'reassessment'):
        shutil.copyfile(baseline / f'{s}.csv', output / f'{s}.csv')
        shutil.copytree(baseline / s, output / s)
    fields = list(next(iter(original['train'].values())))
    with (output / 'train.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train[i] for i in sorted(train))
    labels = output / 'train/labels'
    labels.mkdir(parents=True)
    for i, row in train.items():
        (labels / f'{i}.txt').write_text(str(row['class_id'])+' '+
                                      ' '.join(row[k] for k in BOX_FIELDS)+'\n')
    for s in ('val', 'reassessment'):
        assert sha(output / f'{s}.csv') == cfg['manifest_sha256'][s]
    study.mkdir(parents=True)
    code = study / 'code'
    code.mkdir()
    base_code_hashes = {name: sha(source_code / name) for name in CODE_FILES}
    for name in CODE_FILES:
        shutil.copyfile(source_code / name, code / name)
    # Isolate the only runtime change in a new snapshot; preserve the strict default.
    dataset = code / 'dataset.py'
    text = dataset.read_text()
    old = "            if row['diagnosis'] != NAMES[c] or not row['lesion_id']:\n                raise ValueError('Unknown lesion or inconsistent diagnosis')"
    new = """            unknown_training_allowed = (
                split == 'train' and cfg.get('data_policy', {}).get('allow_missing_lesion_id_in_train') is True
                and cfg['purpose'] == 'exploratory_expanded_training')
            if row['diagnosis'] != NAMES[c] or (not row['lesion_id'] and not unknown_training_allowed):
                raise ValueError('Unknown lesion outside declared exploratory train or inconsistent diagnosis')"""
    assert text.count(old) == 1
    dataset.write_text(text.replace(old, new))
    counts = {s: dict(Counter(r['diagnosis'] for r in rows.values())) for s, rows in all_rows.items()}
    cfg.update(version='expanded-training-v1-20261017', purpose='exploratory_expanded_training',
               model_policy='YOLOv8n only; exploratory training expansion, not replacement of the locked primary study',
               manifest_sha256={s: sha(output / f'{s}.csv') for s in all_rows},
               split_class_counts=counts,
               data_policy={'allow_missing_lesion_id_in_train': True,
                            'known_links_excluded_from_held_out_groups': True,
                            'missing_ids_imputed': False, 'patient_independent_claim': False,
                            'complete_lesion_independence_claim': False})
    cfg['release_decision'] = {
        'locked_at': datetime.now(ZoneInfo('Australia/Brisbane')).isoformat(),
        'basis': 'User requested investigation of more complete training data despite unresolved missing identities',
        'baseline_config_sha256': sha(config_path),
        'run_scope': 'YOLOv8n fresh COCO start, 60 epochs, unchanged validation; no reassessment inference',
        'course_acceptance_confirmed': False,
    }
    cfg['design_notes'] += [
        'Exploratory expansion adds every eligible binary-labelled missing-lesion image to training only.',
        'Known links to held-out components are excluded; unrecorded links remain possible.',
        'Equal epochs do not mean equal optimizer calls; this compares the expanded-data training package, not a pure data-size effect.',
        'Primary comparison: malignant AP50 on identical existing validation images; also macro AP50:95 and fixed 0.05/0.50 thresholds.',
        'Retain the original 10% false-alert selection as an additional comparison; do not rewrite old locks.',
    ]
    write_json(study / 'experiment_expanded.json', cfg)
    summary = {
        'status': 'prepared_and_locked_before_training',
        'study': str(study.relative_to(root)), 'prepared': str(output.relative_to(root)),
        'config_sha256': sha(study / 'experiment_expanded.json'),
        'baseline_config_sha256': sha(config_path),
        'source_snapshot_sha256': {k: v['sha256'] for k, v in sources.items()},
        'base_runtime_source_sha256': base_code_hashes,
        'snapshot_runtime_source_sha256': {name: sha(code / name) for name in CODE_FILES},
        'only_runtime_code_difference': 'dataset.py explicitly permits missing lesion_id in declared exploratory TRAIN only',
        'counts': {s: {'images': len(rows), **counts[s]} for s, rows in all_rows.items()},
        'added_training_images': len(extra),
        'added_class_counts': dict(Counter(meta[i]['diagnosis_1'] for i in extra)),
        'added_historical_membership': dict(Counter(history[i] for i in extra)),
        'excluded_indeterminate_ids': sorted(i for i in meta if meta[i]['diagnosis_1'] == 'Indeterminate'),
        'excluded_known_links_to_held_out': sorted(i for i in blocked if not meta[i]['lesion_id']),
        'held_out_manifests_and_labels_unchanged': True,
        'known_id_overlap': overlap,
        'missing_ids_per_split': {s: {f: sum(not r[f] for r in rows.values())
                                     for f in ('lesion_id', 'patient_id')} for s, rows in all_rows.items()},
        'training_optimizer_calls': math.ceil(len(train)/16)*60,
        'baseline_optimizer_calls': 2880,
        'increased_training_updates_confound_pure_sample_size_attribution': True,
        'fresh_blind_test': False, 'unrecorded_identity_leakage_possible': True,
        'course_acceptance_confirmed': False,
        'primary_comparison_metric': 'Malignant AP50 on the identical existing validation set',
        'additional_comparisons': ['macro mAP50:95', 'fixed confidence 0.05', 'fixed confidence 0.50', 'same 10% benign-alert selection rule'],
        'baseline_results': 'outputs/provisional_20261017_669143/val_n',
        'reassessment_inference_planned': False,
    }
    write_json(study / 'preparation.json', summary)
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.study, args.output), indent=2))
