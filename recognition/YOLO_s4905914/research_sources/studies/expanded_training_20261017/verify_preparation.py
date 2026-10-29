"""Check actual expansion and regression-test the explicit missing-ID boundary."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY / 'code'))
from dataset import read_manifests
from runtime import load_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def must_reject(prepared, cfg, expected):
    try:
        read_manifests(prepared, cfg)
    except ValueError as error:
        if expected not in str(error):
            raise
    else:
        raise AssertionError('Invalid data policy was accepted')


def main():
    prepared = ROOT / 'data/expanded_training_v1'
    baseline = ROOT / 'data/known_lesion_v1'
    cfg = load_config(STUDY / 'experiment_expanded.json')
    original = load_config(ROOT / 'recognition/YOLO_s4905914/experiment_provisional.json')
    rows, hashes = read_manifests(prepared, cfg)
    baseline_rows, _ = read_manifests(baseline, original)
    assert {s: len(rs) for s, rs in rows.items()} == {'train': 2263, 'val': 165, 'reassessment': 165}
    for s in ('val', 'reassessment'):
        assert hashes[s] == original['manifest_sha256'][s]
        for row in rows[s]:
            label = Path(s) / 'labels' / f"{row['image_id']}.txt"
            assert (prepared / label).read_bytes() == (baseline / label).read_bytes()
    meta = {r['isic_id']: r for r in csv.DictReader((ROOT / 'data/source_snapshots/collection_63_metadata.csv').open())}
    included = {r['image_id'] for rs in rows.values() for r in rs}
    assert included == {i for i, r in meta.items() if r['diagnosis_1'] in ('Benign', 'Malignant')}
    assert set(r['image_id'] for r in baseline_rows['train']) <= set(r['image_id'] for r in rows['train'])
    for rs in rows.values():
        for row in rs:
            source = meta[row['image_id']]
            assert row['diagnosis'] == source['diagnosis_1']
            assert all(row[f] == source[f] for f in ('lesion_id', 'patient_id'))
    strict = json.loads(json.dumps(cfg))
    strict['data_policy']['allow_missing_lesion_id_in_train'] = False
    must_reject(prepared, strict, 'Unknown lesion')
    wrong_purpose = json.loads(json.dumps(cfg))
    wrong_purpose['purpose'] = original['purpose']
    must_reject(prepared, wrong_purpose, 'Unknown lesion')
    with tempfile.TemporaryDirectory(prefix='expanded-id-test-') as tmp:
        target = Path(tmp) / 'data'
        shutil.copytree(prepared, target)
        modified = {s: [dict(r) for r in rs] for s, rs in rows.items()}
        modified['val'][0]['lesion_id'] = ''
        def save_split(split):
            with (target / f'{split}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(modified[split][0]))
                writer.writeheader(); writer.writerows(modified[split])
            changed = json.loads(json.dumps(cfg))
            changed['manifest_sha256'] = {s: sha(target / f'{s}.csv') for s in rows}
            return changed
        changed = save_split('val')
        must_reject(target, changed, 'Unknown lesion')
        shutil.copyfile(prepared / 'val.csv', target / 'val.csv')
        known_patient = next(r['patient_id'] for r in rows['val'] if r['patient_id'])
        modified['train'][0]['patient_id'] = known_patient
        changed = save_split('train')
        must_reject(target, changed, 'Known patient_id crosses splits')
    prep = json.loads((STUDY / 'preparation.json').read_text())
    assert sha(STUDY / 'experiment_expanded.json') == prep['config_sha256']
    for name, digest in prep['snapshot_runtime_source_sha256'].items():
        assert sha(STUDY / 'code' / name) == digest
    differences = [name for name, digest in prep['base_runtime_source_sha256'].items()
                   if prep['snapshot_runtime_source_sha256'][name] != digest]
    assert differences == ['dataset.py']
    result = {'status': 'passed', 'held_out_manifests_and_labels_identical': True,
              'original_train_is_subset': True, 'all_2593_binary_images_accounted_for': True,
              'identity_fields_equal_metadata': True, 'identity_imputation': False,
              'training_configuration_unchanged': cfg['train'] == original['train'],
              'regression_checks': ['strict default rejects unknown train identities',
                                    'wrong experiment purpose rejects unknown train identities',
                                    'unknown validation identity rejected',
                                    'known cross-split patient link rejected'],
              'runtime_snapshot_differences': differences,
              'image_decoding_and_exact_duplicate_audit': 'Required on school host before training'}
    (STUDY / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
