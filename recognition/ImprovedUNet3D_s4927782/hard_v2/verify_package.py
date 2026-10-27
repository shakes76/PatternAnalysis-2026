"""Verify the integrated public sources and saved metrics without image inference.

This integration audit uses only Python's standard library. It checks recorded
counts and aggregation, not MRI pixels or independent surface distances. Missing
private cache/weight inputs are reported separately from packaged-file errors.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def require(value, message):
    if not value:
        raise ValueError(message)

def near(actual, expected, label):
    if actual is None or expected is None:
        require(actual is expected, label + ': empty mismatch')
    else:
        require(math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), label)

def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None

def audit_comparison(path, expected_cases):
    report = load(path)
    require(report['status'] == 'completed', 'Incomplete comparison')
    targets, output, rows = None, {}, 0
    for name, model in report['models'].items():
        cases = model['cases']
        require(len(cases) == expected_cases, 'Comparison population changed')
        cohorts, target_counts = defaultdict(list), {}
        for case in cases:
            key = case['patient'], case['visit']
            require(key not in target_counts, 'Duplicate case')
            require(case['shape_dhw'] == [64, 128, 128], 'Not the common original grid')
            target_counts[key] = tuple(x['target_voxels'] for x in case['per_class'])
            require(sum(target_counts[key]) == 64 * 128 * 128, 'Invalid target volume')
            for i, cls in enumerate(case['per_class']):
                require(cls['class_index'] == i, 'Class order changed')
                g, p, tp = cls['target_voxels'], cls['predicted_voxels'], cls['intersection']
                require(all(type(v) is int and v >= 0 for v in (g, p, tp))
                        and tp <= min(g, p), 'Invalid confusion counts')
                near(cls['dice'], 2 * tp / (g + p) if g + p else None, 'Case Dice')
                near(cls['iou'], tp / (g + p - tp) if g + p - tp else None, 'Case IoU')
                rows += 1
            cohorts[case['patient']].append(case)
        require(len(cohorts) == 6, 'Expected six independent patient groups')
        if targets is None:
            targets = target_counts
        else:
            require(targets == target_counts, 'Models did not share the same target counts')
        stored = model['aggregate']
        by_patient = {p['patient']: p for p in stored['patients']}
        require(set(by_patient) == set(cohorts), 'Aggregate patient identity changed')
        for i in range(6):
            for metric in ('dice', 'iou'):
                scores = []
                for patient, cases_for_patient in cohorts.items():
                    value = mean(c['per_class'][i][metric] for c in cases_for_patient)
                    near(value, by_patient[patient]['per_class'][i][metric], 'Patient aggregate')
                    scores.append(value)
                near(mean(scores), stored['per_class'][i][metric], 'Equal-patient aggregate')
        output[name] = {'prostate_dice': stored['per_class'][5]['dice'],
                        'foreground_dice': stored['foreground_mean_dice'],
                        'scans': len(cases), 'patients': len(cohorts)}
    require(set(output) == {'v1_unet2d', 'v1_improved3d', 'hard_improved3d'}, 'Model identities changed')
    return {'models': output, 'class_rows_recomputed': rows}

def verify(project_root, require_external=False, checkpoint=None):
    manifest = load(HERE / 'PUBLICATION_MANIFEST.json')
    for item in manifest['files']:
        path = HERE / item['path']
        require(path.is_file() and digest(path) == item['sha256'], 'Packaged file changed: ' + item['path'])
    protocol = load(HERE / 'protocol_v2.json')
    scientific = {}
    for filename, key in [('SOURCE_FREEZE.json', 'code_sha256'),
                          ('HARD_SCOPE_FREEZE.json', 'required_scope_sha256'),
                          ('analysis_source_freeze.json', 'sha256')]:
        for name, expected in load(HERE / filename)[key].items():
            require(name not in scientific or scientific[name] == expected, 'Conflicting source guards')
            scientific[name] = expected
    for name, expected in scientific.items():
        require(digest(HERE / name) == expected, 'Frozen scientific source changed: ' + name)
    require(digest(HERE / 'protocol_v2.json') == load(HERE / 'SOURCE_FREEZE.json')['protocol_sha256'],
            'Original base protocol bytes changed')
    require(digest(HERE / 'hard_followup_protocol.json') == load(HERE / 'HARD_SCOPE_FREEZE.json')['scope_plan_sha256'],
            'Original Hard scope bytes changed')
    present, missing = [], []
    for name, expected in protocol['required_original_sha256'].items():
        path = project_root / name
        if not path.is_file():
            missing.append(name)
        else:
            require(digest(path) == expected, 'Original dependency changed: ' + name)
            present.append(name)
    if require_external:
        require(not missing, 'Missing original cache/weight dependencies: ' + ', '.join(missing))
    weight_status = 'not supplied; weights intentionally excluded from Git'
    if checkpoint is not None:
        expected = load(HERE / 'artifacts/hard_selection.json')['models']['improved3d']['checkpoint_sha256']
        require(checkpoint.is_file() and digest(checkpoint) == expected, 'Wrong selected Hard checkpoint')
        weight_status = 'actual supplied checkpoint bytes match the recorded selection'
    metrics = {s: audit_comparison(HERE / f'artifacts/hard_common_{s}/comparison.json', n)
               for s, n in [('validation', 38), ('test', 30)]}
    return {'status': 'passed', 'packaged_files_checked': len(manifest['files']),
            'frozen_scientific_files_checked': len(scientific),
            'original_dependencies_present': present, 'external_dependencies_missing': missing,
            'weight_check': weight_status, 'common_grid_metrics': metrics,
            'limits': ['Saved-count and aggregation audit only; no MRI inference or distance recomputation.',
                       'Previously inspected TEST remains retrospective.',
                       'Historical adoption_gate output is not a current assignment requirement.']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=HERE.parent)
    parser.add_argument('--require-external', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.project_root.resolve(), args.require_external, args.checkpoint), indent=2))

if __name__ == '__main__':
    main()
