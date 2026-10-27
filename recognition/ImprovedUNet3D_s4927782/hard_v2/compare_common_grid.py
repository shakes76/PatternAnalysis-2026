"""Compare frozen v1/v2 predictions against the same original cached-grid GT.

No model inference, model selection, augmentation or postprocessing is performed.
V2 categorical labels map through their recorded physical affines with nearest
neighbour interpolation. All metrics are independently implemented here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import scipy
from scipy.ndimage import affine_transform, binary_erosion, distance_transform_edt, generate_binary_structure

MODEL_NAMES = ('unet2d', 'improved3d')
METRICS = ('dice', 'iou')
GATE_PROSTATE_DELTA = 0.01
GATE_FOREGROUND_DELTA = -0.01
GATE_NUMERIC_TOLERANCE = 1e-12
FROZEN_V1_CACHE_IDENTITY_SHA256 = 'eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a'
FROZEN_V1_CHECKPOINTS = {
    'unet2d': 'cfdcfa08147c295962e7d19f2ad16eb20a08590f9a28e00e95a281417aec3c10',
    'improved3d': '9ea46e13a95617ec4ad80c21ba459087fb62941fb04406c7ffeade22ad014b62',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def decoded_array_sha256(array):
    prefix = json.dumps({'dtype': array.dtype.str, 'shape': list(array.shape)},
                        sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(prefix + b'\x00' + array.tobytes(order='C')).hexdigest()


def mean_defined(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def resolve_relative(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Array paths must be safe relative paths.')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Array path escapes its root.')
    return result


def index_records(manifest):
    result = {}
    for record in manifest['records']:
        key = (record['patient'], record['visit'])
        if key in result:
            raise ValueError('Duplicate manifest patient/visit.')
        if record['patient'] not in manifest['patient_splits'][record['split']]:
            raise ValueError('Manifest patient membership is inconsistent.')
        result[key] = record
    return result


def affine(record):
    result = np.asarray(record['resampled_affine_xyz'], dtype=np.float64)
    if (result.shape != (4, 4) or not np.isfinite(result).all()
            or not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-9)
            or abs(np.linalg.det(result[:3, :3])) < 1e-12):
        raise ValueError('Expected a finite invertible physical XYZ affine.')
    return result


def categorical_to_grid(prediction, source_affine_xyz, target_affine_xyz, target_shape_dhw):
    """Map target DHW voxel centres to source DHW using physical world XYZ."""
    prediction = np.asarray(prediction)
    if prediction.ndim != 3 or not np.issubdtype(prediction.dtype, np.integer):
        raise ValueError('Prediction must be a 3D integer categorical array.')
    # Homogeneous index permutation is self-inverse: xyz = P @ dhw.
    reverse = np.eye(4)[[2, 1, 0, 3]]
    mapping = reverse @ np.linalg.inv(source_affine_xyz) @ target_affine_xyz @ reverse
    mapped = affine_transform(prediction, mapping[:3, :3], offset=mapping[:3, 3],
                              output_shape=tuple(target_shape_dhw), order=0,
                              mode='constant', cval=0, prefilter=False)
    return np.asarray(mapped, dtype=prediction.dtype), mapping


def class_metrics(prediction, target, num_classes=6):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError('Prediction/GT must share one 3D grid.')
    for values in (prediction, target):
        if not np.issubdtype(values.dtype, np.integer) or values.min() < 0 or values.max() >= num_classes:
            raise ValueError('Categorical arrays contain undeclared or noninteger classes.')
    matrix = np.bincount(target.ravel().astype(np.int64) * num_classes
                         + prediction.ravel().astype(np.int64),
                         minlength=num_classes ** 2).reshape(num_classes, num_classes)
    per_class = []
    for index in range(num_classes):
        tp = int(matrix[index, index])
        actual, predicted = int(matrix[index].sum()), int(matrix[:, index].sum())
        union = actual + predicted - tp
        per_class.append({'class_index': index, 'target_voxels': actual,
                          'predicted_voxels': predicted, 'intersection': tp, 'union': union,
                          'dice': 2 * tp / (actual + predicted) if actual + predicted else None,
                          'iou': tp / union if union else None})
    return {'per_class': per_class,
            'foreground_mean_dice': mean_defined(x['dice'] for x in per_class[1:]),
            'foreground_mean_iou': mean_defined(x['iou'] for x in per_class[1:])}


def prostate_surface_metrics(prediction, target, spacing_dhw):
    predicted, actual = np.asarray(prediction) == 5, np.asarray(target) == 5
    p, t = bool(predicted.any()), bool(actual.any())
    status = 'both_present' if p and t else 'missed' if t else 'fp_only' if p else 'both_empty'
    if status != 'both_present':
        return {'empty_status': status, 'hd95_mm': None}
    spacing = np.asarray(spacing_dhw, dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Expected positive DHW millimetre spacing.')
    structure = generate_binary_structure(3, 1)
    ps = predicted & ~binary_erosion(predicted, structure=structure, border_value=0)
    ts = actual & ~binary_erosion(actual, structure=structure, border_value=0)
    # Pooled directional distances use all six-connected surface voxels.
    pred_to_target = distance_transform_edt(~ts, sampling=spacing)[ps]
    target_to_pred = distance_transform_edt(~ps, sampling=spacing)[ts]
    return {'empty_status': status,
            'hd95_mm': float(np.percentile(np.concatenate([pred_to_target, target_to_pred]), 95))}


def aggregate(cases):
    grouped = defaultdict(list)
    for case in cases:
        grouped[case['patient']].append(case)
    patients = []
    for patient, records in sorted(grouped.items()):
        patients.append({'patient': patient, 'num_cases': len(records),
                         'per_class': [{'class_index': ci, **{metric: mean_defined(r['per_class'][ci][metric] for r in records)
                                                            for metric in METRICS}} for ci in range(6)],
                         **{f'foreground_mean_{metric}': mean_defined(r[f'foreground_mean_{metric}'] for r in records)
                            for metric in METRICS},
                         'prostate_hd95_mm': mean_defined(r['prostate_surface']['hd95_mm'] for r in records),
                         'hd95_defined_cases': sum(r['prostate_surface']['hd95_mm'] is not None for r in records)})
    return {'num_cases': len(cases), 'num_patients': len(patients), 'patients': patients,
            'per_class': [{'class_index': ci, **{metric: mean_defined(p['per_class'][ci][metric] for p in patients)
                                               for metric in METRICS}} for ci in range(6)],
            **{f'foreground_mean_{metric}': mean_defined(p[f'foreground_mean_{metric}'] for p in patients)
               for metric in METRICS},
            'prostate_hd95_mm_conditional': mean_defined(p['prostate_hd95_mm'] for p in patients),
            'prostate_empty_status_counts': dict(Counter(r['prostate_surface']['empty_status'] for r in cases)),
            'hd95_defined_cases': sum(r['prostate_surface']['hd95_mm'] is not None for r in cases),
            'hd95_defined_patients': sum(p['prostate_hd95_mm'] is not None for p in patients)}


def load_prediction(evaluation_path, case, shape, record, require_affine=False, num_classes=6):
    source = resolve_relative(Path(evaluation_path).parent, case['prediction_file'])
    with np.load(source, allow_pickle=False) as saved:
        prediction = saved['prediction']
        if prediction.shape != tuple(shape) or prediction.ndim != 3:
            raise ValueError('Saved prediction shape differs from its manifest grid.')
        if (not np.issubdtype(prediction.dtype, np.integer)
                or prediction.min() < 0 or prediction.max() >= num_classes):
            raise ValueError('Saved prediction contains invalid classes.')
        if 'label_ids' in saved and not np.array_equal(saved['label_ids'], np.arange(num_classes)):
            raise ValueError('Prediction label order differs from the fixed six classes.')
        if 'spacing_dhw' not in saved or not np.allclose(saved['spacing_dhw'], record['spacing_dhw_mm'], rtol=0, atol=1e-6):
            raise ValueError('Saved prediction spacing differs from its source manifest.')
        if require_affine and 'affine_xyz' not in saved:
            raise ValueError('V2 prediction must preserve its source-grid affine.')
        if 'affine_xyz' in saved and not np.allclose(saved['affine_xyz'], record['resampled_affine_xyz'], rtol=0, atol=1e-6):
            raise ValueError('Saved prediction affine differs from its source manifest.')
    return prediction, {'path': str(source), 'sha256': file_sha256(source)}


def validate_evaluation(path, manifest_path, manifest, split, keys, expected_model):
    evaluation = read_json(path)
    if evaluation.get('split') != split:
        raise ValueError('Evaluation split differs from the explicitly requested split.')
    manifest_digest = evaluation.get('manifest_sha256', evaluation.get('identity', {}).get('manifest_sha256'))
    if manifest_digest != file_sha256(manifest_path):
        raise ValueError('Evaluation is bound to a different manifest.')
    declared_model = evaluation.get('model_config', {}).get('model', evaluation.get('identity', {}).get('model'))
    if declared_model != expected_model:
        raise ValueError('Evaluation model identity differs from its CLI model name.')
    if not re.fullmatch('[a-f0-9]{64}', evaluation.get('checkpoint_sha256', '')):
        raise ValueError('Evaluation must identify a selected checkpoint by SHA256.')
    if evaluation.get('development_only', False):
        raise ValueError('A development checkpoint cannot enter the formal comparison.')
    result = {}
    for case in evaluation['cases']:
        key = (case['patient'], case['visit'])
        if key in result:
            raise ValueError('Duplicate case in evaluation.')
        result[key] = case
    if set(result) != set(keys):
        raise ValueError('Every model must cover exactly the complete declared split.')
    return evaluation, result


def paired_summary(before, after):
    b, a = {(x['patient'], x['visit']): x for x in before}, {(x['patient'], x['visit']): x for x in after}
    grouped = defaultdict(list)
    eligible = defaultdict(list)
    for key in sorted(b):
        grouped[key[0]].append((b[key], a[key]))
        if b[key]['prostate_surface']['hd95_mm'] is not None and a[key]['prostate_surface']['hd95_mm'] is not None:
            eligible[key[0]].append((b[key], a[key]))
    patients = []
    for patient, pairs in sorted(grouped.items()):
        bd = mean_defined(x[0]['per_class'][5]['dice'] for x in pairs)
        ad = mean_defined(x[1]['per_class'][5]['dice'] for x in pairs)
        bf = mean_defined(x[0]['foreground_mean_dice'] for x in pairs)
        af = mean_defined(x[1]['foreground_mean_dice'] for x in pairs)
        patients.append({'patient': patient, 'num_cases': len(pairs), 'v1_prostate_dice': bd,
                         'v2_prostate_dice': ad, 'prostate_delta': ad - bd if bd is not None and ad is not None else None,
                         'v1_foreground_dice': bf, 'v2_foreground_dice': af,
                         'foreground_delta': af - bf if bf is not None and af is not None else None})
    hd_patients = []
    for patient, pairs in sorted(eligible.items()):
        bv = mean_defined(x[0]['prostate_surface']['hd95_mm'] for x in pairs)
        av = mean_defined(x[1]['prostate_surface']['hd95_mm'] for x in pairs)
        hd_patients.append({'patient': patient, 'num_common_cases': len(pairs), 'v1_hd95_mm': bv,
                            'v2_hd95_mm': av, 'delta_hd95_mm': av - bv})
    return {'patients': patients,
            'paired_hd95': {'eligible_case_pairs': sum(len(x) for x in eligible.values()),
                            'excluded_case_pairs': len(b) - sum(len(x) for x in eligible.values()),
                            'num_patients': len(eligible), 'patients': hd_patients,
                            'v1_hd95_mm': mean_defined(x['v1_hd95_mm'] for x in hd_patients),
                            'v2_hd95_mm': mean_defined(x['v2_hd95_mm'] for x in hd_patients),
                            'delta_hd95_mm': mean_defined(x['delta_hd95_mm'] for x in hd_patients)}}


def comparison_gate(v1, v2, split):
    before, after = v1['per_class'][5]['dice'], v2['per_class'][5]['dice']
    prostate_delta = after - before if before is not None and after is not None else None
    bf, af = v1['foreground_mean_dice'], v2['foreground_mean_dice']
    foreground_delta = af - bf if af is not None and bf is not None else None
    passed = (prostate_delta is not None and prostate_delta >= GATE_PROSTATE_DELTA - GATE_NUMERIC_TOLERANCE
              and foreground_delta is not None and foreground_delta >= GATE_FOREGROUND_DELTA - GATE_NUMERIC_TOLERANCE)
    return {'use_for_selection': False, 'scope': 'validation_adoption_gate' if split == 'validation' else 'retrospective_description_only',
            'prostate_dice_delta': prostate_delta, 'foreground_dice_delta': foreground_delta,
            'minimum_prostate_delta': GATE_PROSTATE_DELTA, 'minimum_foreground_delta': GATE_FOREGROUND_DELTA,
            'floating_point_tolerance': GATE_NUMERIC_TOLERANCE,
            'numeric_conditions_pass': bool(passed),
            'replacement_supported_by_validation_gate': bool(passed) if split == 'validation' else None,
            'interpretation': 'Prespecified development indication, not independent generalization or an automatic clinical acceptance rule.'}


def compare_files(args):
    v1_path, v2_path = Path(args.v1_manifest), Path(args.v2_manifest)
    v1, v2 = read_json(v1_path), read_json(v2_path)
    if v1.get('label_ids') != list(range(6)) or v2.get('label_ids') != list(range(6)):
        raise ValueError('Both manifests must declare classes 0..5 in the same order.')
    if v1['patient_splits'] != v2['patient_splits']:
        raise ValueError('Patient splits must remain identical between pipelines.')
    old_records, new_records = index_records(v1), index_records(v2)
    keys = sorted(key for key, record in old_records.items() if record['split'] == args.split)
    if not keys or set(keys) != {key for key, record in new_records.items() if record['split'] == args.split}:
        raise ValueError('Comparison splits must contain exactly the same cases.')
    freeze = read_json(args.selection_freeze) if args.selection_freeze else None
    all_evals, cases_by_eval = {}, {}
    for model in MODEL_NAMES:
        for version, manifest_path, manifest in [('v1', v1_path, v1), ('v2', v2_path, v2)]:
            path = getattr(args, f'{version}_evaluation_{model}')
            evaluation, cases = validate_evaluation(path, manifest_path, manifest, args.split, keys, model)
            if version == 'v1' and evaluation['checkpoint_sha256'] != FROZEN_V1_CHECKPOINTS[model]:
                raise ValueError('Original baseline must use the frozen seed42 checkpoint, not another same-model run.')
            if version == 'v2' and freeze is not None:
                if freeze['models'][model]['checkpoint_sha256'] != evaluation['checkpoint_sha256']:
                    raise ValueError('V2 evaluation checkpoint differs from the frozen selection.')
                if freeze.get('manifest_sha256', file_sha256(v2_path)) != file_sha256(v2_path):
                    raise ValueError('Frozen selection is bound to a different v2 manifest.')
                if ('protocol_sha256' in freeze
                        and freeze['protocol_sha256'] != evaluation.get('identity', {}).get('protocol_sha256')):
                    raise ValueError('Frozen selection is bound to a different v2 protocol.')
            all_evals[version, model] = (Path(path), evaluation)
            cases_by_eval[version, model] = cases
    # All test acknowledgement/freeze and split/identity guards precede pixel IO.
    if args.split == 'test' and (not args.retrospective_test_ack or freeze is None):
        raise ValueError('TEST requires acknowledgement and a frozen selected-checkpoint record.')
    cache_root = Path(args.cache_root) if args.cache_root else v1_path.parent
    identity_path = Path(args.v1_cache_identity) if getattr(args, 'v1_cache_identity', None) else v1_path.parent / 'cache_identity.json'
    if file_sha256(identity_path) != FROZEN_V1_CACHE_IDENTITY_SHA256:
        raise ValueError('Original GT identity book differs from the frozen v1 bytes.')
    gt_identity = read_json(identity_path)
    if gt_identity['manifest_sha256'] != file_sha256(v1_path):
        raise ValueError('Original GT identity book belongs to a different manifest.')
    results = {version: {model: [] for model in MODEL_NAMES} for version in ('v1', 'v2')}
    cache_evidence = []
    for key in keys:
        old, new = old_records[key], new_records[key]
        cache_file = resolve_relative(cache_root, old['cache_path'])
        with np.load(cache_file, allow_pickle=False) as stored:
            target, spacing = stored['mask'], stored['spacing']
        frozen_arrays = gt_identity['records'][old['cache_path']]['arrays']
        target_hash, spacing_hash = decoded_array_sha256(target), decoded_array_sha256(spacing)
        if target_hash != frozen_arrays['mask']['sha256'] or spacing_hash != frozen_arrays['spacing']['sha256']:
            raise ValueError('Original GT mask/spacing decoded hashes differ from frozen identity.')
        if target.shape != tuple(v1['target_shape_dhw']):
            raise ValueError('Original GT grid differs from its frozen manifest.')
        if not np.allclose(spacing, old['spacing_dhw_mm'], rtol=0, atol=1e-6):
            raise ValueError('Original cached spacing differs from its manifest.')
        old_affine, new_affine = affine(old), affine(new)
        affine_spacing = np.linalg.norm(old_affine[:3, :3], axis=0)[::-1]
        if not np.allclose(spacing, affine_spacing, rtol=1e-6, atol=1e-6):
            raise ValueError('Original affine does not agree with metric spacing.')
        gram = old_affine[:3, :3].T @ old_affine[:3, :3]
        if not np.allclose(gram, np.diag(np.diag(gram)), rtol=0, atol=1e-6):
            raise ValueError('EDT physical distances require an orthogonal original grid.')
        expected_counts = old.get('class_voxel_counts')
        if expected_counts is not None and not np.array_equal(np.bincount(target.ravel(), minlength=6), expected_counts):
            raise ValueError('Original cached GT counts differ from the frozen manifest.')
        cache_evidence.append({'patient': key[0], 'visit': key[1], 'cache_path': str(cache_file),
                               'cache_sha256': file_sha256(cache_file), 'decoded_mask_sha256': target_hash,
                               'decoded_spacing_sha256': spacing_hash})
        for model in MODEL_NAMES:
            for version, shape in [('v1', v1['target_shape_dhw']), ('v2', v2['target_shape_dhw'])]:
                path, evaluation = all_evals[version, model]
                original_case = cases_by_eval[version, model][key]
                prediction, pred_evidence = load_prediction(path, original_case, shape, old if version == 'v1' else new,
                                                            require_affine=version == 'v2')
                mapping = None
                if version == 'v2':
                    prediction, mapping = categorical_to_grid(prediction, new_affine, old_affine, target.shape)
                metrics = class_metrics(prediction, target)
                if version == 'v1':
                    # Validate preserved v1 predictions independently before comparing.
                    for actual, saved in zip(metrics['per_class'], original_case['per_class']):
                        for name in ('target_voxels', 'predicted_voxels', 'intersection', 'union'):
                            if actual[name] != saved[name]:
                                raise ValueError('Recomputed original prediction confusion disagrees with its preserved evaluation.')
                        for name in METRICS:
                            if (actual[name] is None) != (saved[name] is None) or (actual[name] is not None and abs(actual[name] - saved[name]) > 1e-12):
                                raise ValueError('Recomputed original metrics disagree with preserved evaluation.')
                case = {'patient': key[0], 'visit': key[1], **metrics,
                        'prostate_surface': prostate_surface_metrics(prediction, target, spacing),
                        'spacing_dhw': [float(x) for x in spacing], 'shape_dhw': list(target.shape),
                        'prediction_evidence': pred_evidence}
                if mapping is not None:
                    case['target_common_dhw_to_source_v2_dhw'] = mapping.tolist()
                results[version][model].append(case)
    report = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'split': args.split, 'retrospective_test_acknowledged': args.retrospective_test_ack,
              'metric_grid': 'Original v1 downsampled cached grid, with unchanged original GT and per-case physical spacing; not native-resolution accuracy.',
              'resampling': 'V2 hard categorical predictions: physical-world affine nearest neighbour, order0, constant background outside FOV. V1 predictions unchanged.',
              'metric_protocol': 'Per-case hard multiclass Dice/IoU; both-empty null/excluded and absent-GT false positives zero; foreground classes1..5; mean visits per patient then equal patients.',
              'hd95_protocol': '95th percentile of pooled bidirectional nearest six-connected voxel-surface distances, millimetres on original cached grid. Any empty mask: null and explicit status. Paired comparison uses identical both-present cases.',
              'selection_policy': 'No model/checkpoint selection here. Both complete model pairs are reported regardless of validation gate outcomes. Test gate numbers are descriptive only.',
              'input_files': {'v1_manifest': {'path': str(v1_path), 'sha256': file_sha256(v1_path)},
                              'v2_manifest': {'path': str(v2_path), 'sha256': file_sha256(v2_path)},
                              'v1_cache_identity': {'path': str(identity_path), 'sha256': file_sha256(identity_path)},
                              'evaluations': {f'{version}_{model}': {'path': str(path), 'sha256': file_sha256(path), 'checkpoint_sha256': evaluation['checkpoint_sha256']}
                                              for (version, model), (path, evaluation) in all_evals.items()}},
              'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
              'common_gt_inputs': cache_evidence, 'models': {}}
    if freeze is not None:
        report['input_files']['selection_freeze'] = {'path': str(args.selection_freeze), 'sha256': file_sha256(args.selection_freeze)}
    for model in MODEL_NAMES:
        before, after = results['v1'][model], results['v2'][model]
        b, a = aggregate(before), aggregate(after)
        report['models'][model] = {'v1': {'cases': before, 'aggregate': b}, 'v2': {'cases': after, 'aggregate': a},
                                   'paired': paired_summary(before, after), 'gate': comparison_gate(b, a, args.split)}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-manifest', type=Path, required=True)
    parser.add_argument('--v2-manifest', type=Path, required=True)
    for version in ('v1', 'v2'):
        for model in MODEL_NAMES:
            parser.add_argument(f'--{version}-evaluation-{model}', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, help='Root against which v1 cache_path is resolved; defaults to v1 manifest parent.')
    parser.add_argument('--v1-cache-identity', type=Path, help='Original frozen identity book; defaults to v1 manifest parent/cache_identity.json.')
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--retrospective-test-ack', action='store_true')
    parser.add_argument('--selection-freeze', type=Path, help='JSON models.{unet2d,improved3d}.checkpoint_sha256; required for TEST.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split == 'test' and (not args.retrospective_test_ack or args.selection_freeze is None):
        parser.error('TEST is retrospective and requires --retrospective-test-ack plus --selection-freeze before any pixel reads.')
    if args.split != 'test' and args.retrospective_test_ack:
        parser.error('--retrospective-test-ack is valid only for TEST.')
    output = args.output_dir / 'common_grid_comparison.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite a prior comparison; choose a new output directory.')
    report = compare_files(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps({'output': str(output), 'split': args.split,
                      'gates': {name: value['gate'] for name, value in report['models'].items()}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
