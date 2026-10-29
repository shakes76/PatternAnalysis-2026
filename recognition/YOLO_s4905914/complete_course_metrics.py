"""Recompute extra course metrics from locked caches without model selection.

The reporting plan was recorded after model development. New metrics are
descriptive additions, not a retrospective change to the selection criterion.
"""
import argparse
import csv
from pathlib import Path

from classification_metrics import summarize_predictions
from evaluation import gt_record
from runtime import digest, read_json, write_json


def compare_numeric(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            compare_numeric(a[k], b[k], path + '.' + k)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare_numeric(x, y, path + '.' + str(i))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert isinstance(b, (int, float)) and abs(a - b) <= 1e-12, (path, a, b)
    else:
        assert a == b, (path, a, b)


def complete(root, plan_path, output):
    code = Path(__file__).resolve().parent
    plan = read_json(plan_path)
    assert digest(code/'teacher_response_20261022.txt') == plan['teacher_response_sha256']
    for name, expected in plan['reporting_code_sha256'].items():
        assert digest(code/name) == expected, name
    for path, expected in plan['selection_records_sha256'].items():
        assert digest(root/path) == expected, path
    manifests = {}
    for split, expected in plan['manifest_sha256'].items():
        path = root/f'data/known_lesion_v1/{split}.csv'
        assert digest(path) == expected
        manifests[split] = {r['image_id']: gt_record(r) for r in csv.DictReader(path.open())}
    reports = {}
    for key, spec in plan['cached_runs'].items():
        source = root/spec['evaluation_path']
        assert digest(source/'evaluation.json') == spec['evaluation_sha256'], key
        assert digest(source/'predictions.json') == spec['predictions_sha256'], key
        ev = read_json(source/'evaluation.json')
        records = read_json(source/'predictions.json')
        assert ev['split'] == spec['split']
        assert ev['checkpoint_sha256'] == spec['checkpoint_sha256']
        truths = manifests[spec['split']]
        assert len(records) == len(truths) == 165
        assert {r['image_id'] for r in records} == set(truths)
        for r in records:
            compare_numeric({k: r[k] for k in ('image_id', 'class_id', 'box')}, truths[r['image_id']])
        report = summarize_predictions(records, spec['threshold'], .5)
        point = ev['operating_point']['selected'] if spec['split'] == 'val' else ev['operating_point']
        compare_numeric(report['detection'], point)
        reports[key] = {**spec, 'metrics': report, 'ap': ev['ap'],
                        'mAP50': ev['mAP50'], 'mAP50_95': ev['mAP50_95']}
    result = {'status': 'passed', 'reporting_plan_sha256': digest(plan_path),
              'teacher_response_sha256': plan['teacher_response_sha256'],
              'posthoc_reporting_only': True, 'new_training': False,
              'new_reassessment_inference': False, 'selection_changed': False,
              'selected_model': plan['selected_model'], 'reports': reports,
              'current_candidate_reassessment': 'pending: no new prediction cache available',
              'fresh_blind_test': False}
    write_json(output, result)
    for key, r in reports.items():
        c = r['metrics']['classification']
        print(key, 'accuracy', round(c['accuracy'], 4), 'macro_F1', round(c['macro_f1'], 4),
              'M classification recall', round(c['classes']['Malignant']['recall'], 4),
              'no prediction', c['images'] - c['covered'])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--plan', type=Path, default=Path(__file__).with_name('course_closeout_plan.json'))
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('course_metrics_20261022.json'))
    a = p.parse_args()
    complete(a.root.resolve(), a.plan.resolve(), a.output.resolve())
