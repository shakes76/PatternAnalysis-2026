"""Audit and report the complete finite v2 pair, without reading pixel arrays.

Uses only completed run/history/evaluation/common-grid JSONs, frozen identities
and checkpoint byte hashes. Imports no training, inference or project metrics.
Does not choose checkpoints or directly compare the new cache-grid scores with
old-grid scores. Neither cache grid is native-resolution accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MODELS = ('unet2d', 'improved3d')
NAMES = ['背景', '身体', '骨骼', '膀胱', '直肠', '前列腺']
TOL = 2e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    path = Path(path)
    require(path.is_file(), f'Required completed evidence is missing: {path}')
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def equal(actual, expected, where):
    if actual is None or expected is None:
        require(actual is None and expected is None, f'Undefined-policy mismatch: {where}')
    else:
        require(math.isfinite(float(actual)) and abs(actual - expected) <= TOL, f'Numerical mismatch: {where}: {actual} vs {expected}')


def records(manifest, split):
    result = {(r['patient'], r['visit']): r for r in manifest['records'] if r['split'] == split}
    require(len(result) == sum(r['split'] == split for r in manifest['records']), 'Duplicate manifest case')
    require(all(key[0] in manifest['patient_splits'][split] for key in result), 'Patient membership mismatch')
    return result


def audit_evaluation(evaluation, expected_records, where, counters, surface=False):
    """Independent algebra and visit→patient aggregation, no project import."""
    cases = evaluation['cases']
    keys = [(x['patient'], x['visit']) for x in cases]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected_records), f'Incomplete/duplicate split: {where}')
    grouped = defaultdict(list)
    for case in cases:
        key = case['patient'], case['visit']
        counters['cases'] += 1
        values = case['per_class']
        require([x['class_index'] for x in values] == list(range(6)), f'Six-class ordering: {where}/{key}')
        for ci, item in enumerate(values):
            counters['case_class_rows'] += 1
            t, p, intersection, union = [item[x] for x in ('target_voxels', 'predicted_voxels', 'intersection', 'union')]
            require(all(type(x) is int and x >= 0 for x in [t, p, intersection, union]), f'Invalid counts: {where}/{key}/{ci}')
            require(intersection <= min(t, p) and union == t + p - intersection, f'Count algebra: {where}/{key}/{ci}')
            counts = expected_records[key].get('class_voxel_counts')
            if counts is not None:
                require(t == counts[ci], f'GT count differs from manifest/receipt: {where}/{key}/{ci}')
            equal(item['dice'], 2 * intersection / (t + p) if t + p else None, f'{where}/{key}/{ci}/Dice')
            equal(item['iou'], intersection / union if union else None, f'{where}/{key}/{ci}/IoU')
        target_total = sum(item['target_voxels'] for item in values)
        require(target_total > 0 and target_total == sum(item['predicted_voxels'] for item in values),
                f'Mutually exclusive class totals differ: {where}/{key}')
        if 'shape_dhw' in case:
            shape = case['shape_dhw']
            require(len(shape) == 3 and all(type(x) is int and x > 0 for x in shape)
                    and math.prod(shape) == target_total, f'Grid/count volume mismatch: {where}/{key}')
        if 'inference_ms' in case:
            require(math.isfinite(case['inference_ms']) and case['inference_ms'] > 0, f'Invalid inference latency:{where}')
        equal(case['foreground_mean_dice'], mean(x['dice'] for x in values[1:]), f'{where}/{key}/FG Dice')
        equal(case['foreground_mean_iou'], mean(x['iou'] for x in values[1:]), f'{where}/{key}/FG IoU')
        if surface:
            p5 = values[5]
            t, p = p5['target_voxels'] > 0, p5['predicted_voxels'] > 0
            expected_status = 'both_present' if t and p else 'missed' if t else 'fp_only' if p else 'both_empty'
            measured = case['prostate_surface']
            require(measured['empty_status'] == expected_status, f'Empty status mismatch: {where}/{key}')
            require((measured['hd95_mm'] is not None) == (expected_status == 'both_present'), f'HD95 support mismatch: {where}/{key}')
            if measured['hd95_mm'] is not None:
                require(math.isfinite(measured['hd95_mm']) and measured['hd95_mm'] >= 0, f'Invalid HD95: {where}/{key}')
        grouped[key[0]].append(case)
    patients = {}
    for patient, items in grouped.items():
        patients[patient] = {'num_cases': len(items), 'per_class': [
            {metric: mean(c['per_class'][ci][metric] for c in items) for metric in ('dice', 'iou')}
            for ci in range(6)], **{f'foreground_mean_{metric}': mean(c[f'foreground_mean_{metric}'] for c in items)
                                   for metric in ('dice', 'iou')}}
    agg = evaluation['aggregate']
    require(agg['num_cases'] == len(cases) and agg['num_patients'] == len(patients), f'Aggregate population mismatch: {where}')
    require({x['patient'] for x in agg['patients']} == set(patients) and len(agg['patients']) == len(patients), f'Patient summary coverage: {where}')
    for item in agg['patients']:
        expected = patients[item['patient']]
        require(item['num_cases'] == expected['num_cases'], f'Patient visit count: {where}')
        for ci in range(6):
            for metric in ('dice', 'iou'):
                equal(item['per_class'][ci][metric], expected['per_class'][ci][metric], f'{where}/{item["patient"]}/{ci}/{metric}')
        for metric in ('dice', 'iou'):
            equal(item[f'foreground_mean_{metric}'], expected[f'foreground_mean_{metric}'], f'{where}/patient FG')
    for ci in range(6):
        for metric in ('dice', 'iou'):
            equal(agg['per_class'][ci][metric], mean(p['per_class'][ci][metric] for p in patients.values()), f'{where}/aggregate/{ci}/{metric}')
    for metric in ('dice', 'iou'):
        equal(agg[f'foreground_mean_{metric}'], mean(p[f'foreground_mean_{metric}'] for p in patients.values()), f'{where}/aggregate FG')
    if 'mean_inference_ms' in evaluation:
        equal(evaluation['mean_inference_ms'], mean(c['inference_ms'] for c in cases), f'{where}/mean latency')
        require(evaluation['peak_allocated_cuda_bytes'] == max(c['peak_allocated_cuda_bytes'] for c in cases), f'Inference CUDA peak summary:{where}')
        require(all(c['peak_allocated_cuda_bytes'] is not None and c['peak_allocated_cuda_bytes'] > 0 for c in cases), f'Expected recorded positive CUDA allocation:{where}')
    if surface:
        statuses = Counter(c['prostate_surface']['empty_status'] for c in cases)
        require(dict(statuses) == agg['prostate_empty_status_counts'], f'Empty summary: {where}')
        defined = sum(c['prostate_surface']['hd95_mm'] is not None for c in cases)
        equal(agg['hd95_defined_cases'], defined, f'{where}/HD95 support')
        conditional = [mean(c['prostate_surface']['hd95_mm'] for c in cs) for cs in grouped.values()]
        equal(agg['prostate_hd95_mm_conditional'], mean(conditional), f'{where}/conditional HD95')
    counters['evaluation_blocks'] += 1
    return patients


def audit_run(root, version, model, protocol, manifest, counters, freeze=None):
    root = Path(root)
    run = load(root / 'run.json')
    require(run.get('status') == 'completed', f'Do not inspect unfinished run outputs: {root}')
    require(run['completed_epochs'] == 30, f'Incomplete fixed budget: {root}')
    history = load(root / 'history.json')
    require([r['epoch'] for r in history] == list(range(1, 31)), f'History must cover exactly30 epochs: {root}')
    config = protocol['configuration'] if version == 'v2' else run['config']
    train_cases = len(records(manifest, 'train'))
    examples = train_cases * (config['train_slices_per_scan'] if model == 'unet2d' else config['train_patches_per_scan']) if version == 'v2' else train_cases * (manifest['target_shape_dhw'][0] if model == 'unet2d' else 1)
    batch = config['train_batch_size_2d'] if version == 'v2' and model == 'unet2d' else config['train_batch_size_3d'] if version == 'v2' else config['batch_size']
    batches = math.ceil(examples / batch)
    scheduled = protocol['configuration']['validation_epochs'] if version == 'v2' else list(range(1, 31))
    observed = []
    for row in history:
        require(math.isfinite(row['train_loss']), f'Nonfinite historical train loss: {root}/{row["epoch"]}')
        require(row['training_examples'] == examples and row['training_batches'] == batches, f'Sampling budget mismatch: {root}/{row["epoch"]}')
        require(all(type(row[k]) is int and 0 <= row[k] <= batches
                    for k in ('optimizer_steps', 'amp_overflow_skipped_optimizer_steps')),
                f'Invalid update/skip counts: {root}/{row["epoch"]}')
        require(row['optimizer_steps'] + row['amp_overflow_skipped_optimizer_steps'] == batches, f'Update/skip accounting: {root}/{row["epoch"]}')
        require(math.isfinite(row['train_seconds']) and row['train_seconds'] > 0,
                f'Invalid training duration: {root}/{row["epoch"]}')
        require(type(row['peak_allocated_cuda_bytes_training']) is int and row['peak_allocated_cuda_bytes_training'] > 0,
                f'Invalid training CUDA allocation: {root}/{row["epoch"]}')
        evaluation = row.get('validation')
        if evaluation is not None:
            audit_evaluation(evaluation, records(manifest, 'validation'), f'{root}/epoch{row["epoch"]}', counters)
            observed.append(row['epoch'])
            metric = evaluation['aggregate']['per_class'][5]['dice'] if version == 'v2' else evaluation['aggregate']['foreground_mean_dice']
            field = 'validation_prostate_dice' if version == 'v2' else 'validation_foreground_dice'
            equal(row[field], metric, f'{root}/selection metric')
        else:
            require(version == 'v2' and row['validation_prostate_dice'] is None, f'Unexplained absent validation: {root}')
    require(observed == scheduled, f'Validation schedule changed: {root}')
    field = 'validation_prostate_dice' if version == 'v2' else 'validation_foreground_dice'
    best = max((r for r in history if r.get(field) is not None), key=lambda r: (r[field], -r['epoch']))
    require(run['best_epoch'] == best['epoch'], f'Checkpoint selection differs from earliest strict maximum: {root}')
    weight = root / 'best.pt'
    require(weight.is_file(), f'Selected checkpoint bytes required for audit: {weight}')
    digest = sha(weight)
    if version == 'v2':
        selected = freeze['models'][model]
        require(selected['completed_epochs'] == 30 and selected['checkpoint_epoch'] == best['epoch'], f'Freeze epoch/budget: {model}')
        equal(selected['validation_prostate_dice'], best[field], f'Freeze selected score: {model}')
        require(selected['history_sha256'] == sha(root / 'history.json') and selected['run_sha256'] == sha(root / 'run.json'), f'Freeze run/history byte mismatch: {model}')
        require(selected['checkpoint_sha256'] == digest == run['best_checkpoint_sha256'], f'Selected weight byte mismatch: {model}')
        require(run['identity']['configuration'] == protocol['configuration'] and run['identity']['model'] == model, f'Run configuration/model mismatch: {model}')
        equal(run['best_validation_prostate_dice'], best[field], f'Run best score:{model}')
        expected_counts = np.sum([r['class_voxel_counts'] for r in records(manifest, 'train').values()], axis=0)
        require(run['training_voxel_counts'] == expected_counts.tolist(), f'Class weighting must use exactly full TRAIN counts:{model}')
        weights = 1 / np.sqrt(expected_counts.astype(np.float64) / expected_counts.sum())
        weights /= weights.mean(); weights = np.clip(weights, .05, 10); weights /= weights.mean()
        require(np.allclose(run['class_weights'], weights, rtol=0, atol=1e-6), f'CE weight formula mismatch:{model}')
        for row in history:
            s = row['sampling']
            require(0 <= s['foreground_missing_fallback'] <= s['foreground_requested'] <= examples, f'Invalid foreground sampling accounting:{model}')
            require(len(s['selected_class_counts']) == 6 and s['selected_class_counts'][0] == 0 and sum(s['selected_class_counts']) == s['foreground_requested'], f'Class request accounting:{model}')
            require(0 <= s['hw_flip_count'] <= 2 * examples, f'Flip accounting:{model}')
        runtime = {k: run.get(k) for k in ('gpu_name', 'python_version', 'torch_version', 'numpy_version', 'cuda_version', 'device')}
    else:
        expected = protocol['required_original_sha256'][f'artifacts/full_seed42/{model}/best.pt']
        require(digest == expected, f'Original baseline must remain the frozen seed42 run: {model}')
        require(config['model'] == model and config['manifest_sha256'] == sha(root.parents[1] / 'cache64/manifest.json'),
                f'Original run model/manifest identity differs:{model}')
        runtime = {k: run.get(k) for k in ('gpu_name', 'python_version', 'torch_version', 'cuda_version', 'device')}
    resources = {'training_loop_seconds': sum(r['train_seconds'] for r in history),
                 'successful_updates': sum(r['optimizer_steps'] for r in history),
                 'amp_skipped_updates': sum(r['amp_overflow_skipped_optimizer_steps'] for r in history),
                 'peak_training_allocated_cuda_bytes': max(r['peak_allocated_cuda_bytes_training'] for r in history),
                 'parameters': run['parameters'], 'runtime': runtime,
                 'scope': 'Training-loop measurements; excludes validation, saving, plotting, setup and queue waits. Allocated bytes are not total device usage.'}
    if 'started_at' in run and 'completed_at' in run:
        resources['recorded_run_elapsed_seconds'] = (datetime.fromisoformat(run['completed_at']) - datetime.fromisoformat(run['started_at'])).total_seconds()
    return {'version': version, 'model': model, 'epochs': 30, 'selected_epoch': best['epoch'],
            'selection_metric': field, 'selection_grid': version, 'best_selection_score': best[field],
            'checkpoint_sha256': digest, 'resources': resources, 'run': run, 'history': history}


def audit_paired(pair, before, after, where):
    indexed = [{(r['patient'], r['visit']): r for r in items} for items in [before, after]]
    grouped = defaultdict(list)
    for key in indexed[0]:
        grouped[key[0]].append((indexed[0][key], indexed[1][key]))
    require({p['patient'] for p in pair['patients']} == set(grouped) and len(pair['patients']) == len(grouped), f'Paired patient coverage: {where}')
    for p in pair['patients']:
        cases = grouped[p['patient']]
        require(p['num_cases'] == len(cases), f'Paired visit count: {where}')
        for metric, field in [('prostate', None), ('foreground', 'foreground_mean_dice')]:
            old = mean(x[0][field] if field else x[0]['per_class'][5]['dice'] for x in cases)
            new = mean(x[1][field] if field else x[1]['per_class'][5]['dice'] for x in cases)
            equal(p[f'v1_{metric}_dice'], old, f'{where}/paired baseline')
            equal(p[f'v2_{metric}_dice'], new, f'{where}/paired new')
            equal(p[f'{metric}_delta'], new - old if new is not None and old is not None else None, f'{where}/paired delta')
    eligible = {p: [(a, b) for a, b in cs if a['prostate_surface']['hd95_mm'] is not None and b['prostate_surface']['hd95_mm'] is not None] for p, cs in grouped.items()}
    eligible = {p: cs for p, cs in eligible.items() if cs}
    hd = pair['paired_hd95']
    require(hd['eligible_case_pairs'] == sum(map(len, eligible.values())) and hd['excluded_case_pairs'] == len(before) - sum(map(len, eligible.values())), f'HD95 common support: {where}')
    require(hd['num_patients'] == len(eligible) and {p['patient'] for p in hd['patients']} == set(eligible)
            and len(hd['patients']) == len(eligible), f'HD95 patient support: {where}')
    for p in hd['patients']:
        cs = eligible[p['patient']]
        old, new = (mean(x[index]['prostate_surface']['hd95_mm'] for x in cs) for index in [0, 1])
        require(p['num_common_cases'] == len(cs), f'HD95 matched visits: {where}')
        equal(p['v1_hd95_mm'], old, f'{where}/HD95 old')
        equal(p['v2_hd95_mm'], new, f'{where}/HD95 new')
        equal(p['delta_hd95_mm'], new - old, f'{where}/HD95 delta')
    for field in ('v1_hd95_mm', 'v2_hd95_mm', 'delta_hd95_mm'):
        equal(hd[field], mean(p[field] for p in hd['patients']), f'{where}/HD95 equal patients')


def detailed_model_summary(evaluation):
    aggregate = evaluation['aggregate']
    grouped = defaultdict(list)
    for r in evaluation['cases']:
        grouped[r['patient']].append(r)
    patients, pooled = [], Counter()
    for patient, cases in sorted(grouped.items()):
        counts = Counter()
        for row in cases:
            x = row['per_class'][5]
            counts.update(tp=x['intersection'], fp=x['predicted_voxels'] - x['intersection'], fn=x['target_voxels'] - x['intersection'], gt=x['target_voxels'])
        pooled.update(counts)
        patients.append({'patient': patient, 'visits': len(cases),
                         'prostate_dice': mean(r['per_class'][5]['dice'] for r in cases),
                         'prostate_iou': mean(r['per_class'][5]['iou'] for r in cases),
                         'prostate_hd95_mm_conditional': mean(r['prostate_surface']['hd95_mm'] for r in cases),
                         'prostate_voxel_counts': dict(counts),
                         'empty_status_counts': dict(Counter(r['prostate_surface']['empty_status'] for r in cases))})
    return {'aggregate': aggregate, 'patients': patients, 'pooled_prostate_counts_descriptive_not_scoring': dict(pooled),
            'failed_prostate_dice_below_point7_cases': [{'patient': r['patient'], 'visit': r['visit'], 'dice': r['per_class'][5]['dice']} for r in evaluation['cases'] if r['per_class'][5]['dice'] is not None and r['per_class'][5]['dice'] < .7],
            'numeric_prostate_point7_goal_met': aggregate['per_class'][5]['dice'] is not None and aggregate['per_class'][5]['dice'] >= .7}


def find_common(artifacts_root, split, explicit):
    if explicit:
        return Path(explicit)
    matches = []
    for path in Path(artifacts_root).glob('**/common_grid_comparison.json'):
        if load(path).get('split') == split:
            matches.append(path)
    require(len(matches) == 1, f'Need exactly one {split} common-grid report; pass --common-{split} explicitly if multiple.')
    return matches[0]


def audit_runtime_receipt(evaluation_path, evaluation, review, protocol_path, manifest_path, protocol):
    """Bind actual evaluation runtime sidecars without changing metric JSONs."""
    path = Path(evaluation_path).parent / 'runtime_receipt.json'
    if not path.is_file():
        require(not (Path(review) / 'engineering_execution_amendment.json').is_file(),
                'Missing required evaluation runtime receipt under the execution amendment')
        return None
    receipt = load(path)
    expected = {'evaluation_sha256': sha(evaluation_path),
                'metrics_sha256': sha(Path(evaluation_path).with_name('metrics.json')),
                'checkpoint_sha256': evaluation['checkpoint_sha256'],
                'protocol_sha256': sha(protocol_path), 'manifest_sha256': sha(manifest_path),
                'adapter_sha256': sha(Path(review) / 'evaluate_v2_deterministic.py')}
    require(receipt.get('status') == 'completed', 'Incomplete evaluation runtime receipt')
    require(all(receipt.get(k) == value for k, value in expected.items()), 'Evaluation runtime receipt hash/identity mismatch')
    require(load(Path(evaluation_path).with_name('metrics.json')) == evaluation, 'Runtime metrics/evaluation JSONs differ')
    require(receipt.get('execution_amendment_file') == 'engineering_execution_amendment.json',
            'Unexpected evaluation execution amendment')
    amendment = Path(review) / 'engineering_execution_amendment.json'
    require(receipt.get('execution_amendment_sha256') == sha(amendment), 'Evaluation execution amendment digest mismatch')
    require(receipt.get('adapter_file', 'evaluate_v2_deterministic.py') == 'evaluate_v2_deterministic.py',
            'Unexpected evaluation adapter')
    require(receipt.get('seed') == protocol['configuration']['seed']
            and receipt.get('amp') is True and protocol['configuration']['amp'] is True
            and receipt.get('cudnn_deterministic') is True and receipt.get('cudnn_benchmark') is False
            and receipt.get('deterministic_algorithms_enabled') is True
            and receipt.get('deterministic_algorithms_warn_only') is True,
            'Evaluation precision/seed/deterministic execution differs from the declared training settings')
    require(isinstance(receipt.get('device'), str) and receipt['device'].split(':')[0] == 'cuda',
            'Formal inference runtime must record CUDA')
    for key in ('gpu_name', 'python_version', 'torch_version', 'cuda_version', 'numpy_version',
                'scipy_version', 'nibabel_version', 'slurm_node_list'):
        require(isinstance(receipt.get(key), str) and bool(receipt[key].strip()), f'Missing actual inference runtime field: {key}')
    require(str(receipt.get('slurm_job_id', '')).isdigit() and int(receipt['slurm_job_id']) > 0,
            'Actual inference runtime must identify its Slurm allocation')
    return {'path': str(path), 'sha256': sha(path), 'record': receipt}


def build_summary(args):
    review, project, artifacts = Path(args.review_root), Path(args.v1_project), Path(args.artifacts_root)
    protocol_path = Path(args.protocol) if args.protocol else review / 'protocol_v2.json'
    protocol = load(protocol_path)
    old_manifest_path, new_manifest_path = project / 'artifacts/cache64/manifest.json', artifacts / 'cache_v2/manifest.json'
    old, new = load(old_manifest_path), load(new_manifest_path)
    freeze_path = Path(args.selection_freeze) if args.selection_freeze else artifacts / 'selection_v2.json'
    freeze = load(freeze_path)
    require(freeze['status'] == 'frozen' and freeze['protocol_sha256'] == sha(protocol_path) and freeze['manifest_sha256'] == sha(new_manifest_path), 'Selection freeze protocol/manifest mismatch')
    require(set(freeze['models']) == set(MODELS), 'Freeze must contain exactly the two predeclared models')
    require(new['original_manifest_sha256'] == sha(old_manifest_path) and new['protocol_sha256'] == sha(protocol_path), 'V2 manifest provenance differs')
    require(old['patient_splits'] == new['patient_splits'], 'Patient split changed')
    require({s: len(records(old, s)) for s in ('train', 'validation', 'test')} == {'train': 143, 'validation': 38, 'test': 30}, 'Original full cohort must remain143/38/30 scans')
    require({s: len(old['patient_splits'][s]) for s in ('train', 'validation', 'test')} == {'train': 26, 'validation': 6, 'test': 6}, 'Original full cohort must remain26/6/6 patients')
    counters, source_checks, input_files = Counter(), [], {}
    for key, path in [('protocol', protocol_path), ('v1_manifest', old_manifest_path), ('v2_manifest', new_manifest_path), ('selection_freeze', freeze_path)]:
        input_files[key] = {'path': str(path), 'sha256': sha(path)}
    for relative, expected in protocol['required_original_sha256'].items():
        path = project / relative
        require(sha(path) == expected, f'Protected original source/artifact changed: {relative}')
        source_checks.append({'group': 'original', 'file': relative, 'sha256': expected})
    for relative, expected in protocol['required_review_sha256'].items():
        require(sha(review / relative) == expected, f'Protected review source changed: {relative}')
        source_checks.append({'group': 'review', 'file': relative, 'sha256': expected})
    # Check BOTH status records before reading either full history/evaluation.
    for model in MODELS:
        status = load(artifacts / f'formal_v2/{model}/run.json')
        require(status.get('status') == 'completed' and status.get('completed_epochs') == 30, f'Incomplete formal pair: {model}; no final report created')
    runs = {version: {} for version in ('v1', 'v2')}
    for model in MODELS:
        runs['v1'][model] = audit_run(project / f'artifacts/full_seed42/{model}', 'v1', model, protocol, old, counters)
        runs['v2'][model] = audit_run(artifacts / f'formal_v2/{model}', 'v2', model, protocol, new, counters, freeze)
        identity = runs['v2'][model]['run']['identity']
        require(identity['manifest_sha256'] == sha(new_manifest_path) and identity['protocol_sha256'] == sha(protocol_path), f'Run identity differs:{model}')
        require(identity['v2_pipeline_sha256'] == sha(review / 'v2_pipeline.py'), f'Actual trainer hash differs:{model}')
        require(identity['source_hashes'] == protocol['required_original_sha256'], f'Actual run source coverage differs:{model}')
        require(runs['v2'][model]['resources']['parameters'] == runs['v1'][model]['resources']['parameters'], f'The frozen same-model architecture parameter count changed:{model}')
        for relative, expected in identity['source_hashes'].items():
            require(sha(project / relative) == expected, f'Actual run source differs:{relative}')
    completion = load(artifacts / 'cache_v2/test_completion_receipt.json')
    require(completion['status'] == 'completed' and completion['manifest_bytes_unchanged'] is True and completion['manifest_sha256'] == sha(new_manifest_path) and completion['selection_freeze_sha256'] == sha(freeze_path), 'Incomplete/unbound TEST preparation')
    require(completion['protocol_sha256'] == sha(protocol_path), 'TEST completion protocol differs')
    test_records = {(r['patient'], r['visit']): r for r in completion['records']}
    require(set(test_records) == set(records(new, 'test')) and len(completion['records']) == 30, 'Incomplete TEST cache receipt')
    old_identity_path = project / 'artifacts/cache64/cache_identity.json'
    old_identity = load(old_identity_path)
    require(old_identity['manifest_sha256'] == sha(old_manifest_path), 'Original identity book differs')
    common_reports, grid_evaluations, comparisons, runtime_receipts = {}, {}, {}, {}
    for split in ('validation', 'test'):
        raw_evaluations = {}
        for model in MODELS:
            v2_path = artifacts / f'{"validation_v2" if split == "validation" else "test_v2"}/{model}/evaluation.json'
            v1_path = project / f'artifacts/{"validation_seed42" if split == "validation" else "test_final"}/{model}/evaluation.json'
            for version, path, manifest in [('v1', v1_path, old), ('v2', v2_path, new)]:
                evaluation = load(path)
                expected = records(manifest, split) if version == 'v1' or split == 'validation' else test_records
                audit_evaluation(evaluation, expected, f'{version}/{split}/{model}', counters)
                require(evaluation['split'] == split and evaluation['checkpoint_sha256'] == runs[version][model]['checkpoint_sha256'] and evaluation['checkpoint_epoch'] == runs[version][model]['selected_epoch'], f'Evaluation selected checkpoint identity:{version}/{split}/{model}')
                identity = evaluation.get('identity', {})
                require(evaluation.get('manifest_sha256', identity.get('manifest_sha256')) == sha(old_manifest_path if version == 'v1' else new_manifest_path), 'Evaluation manifest mismatch')
                require(evaluation.get('model_config', {}).get('model', identity.get('model')) == model, 'Evaluation model mislabeled')
                if version == 'v2':
                    require(identity == runs['v2'][model]['run']['identity'], 'Evaluation/training identity mismatch')
                    if split == 'test':
                        require(evaluation['selection_freeze_sha256'] == sha(freeze_path), 'TEST evaluation freeze mismatch')
                    if split == 'validation':
                        best_history = runs['v2'][model]['history'][runs['v2'][model]['selected_epoch'] - 1]['validation']
                        previous = {(r['patient'], r['visit']): r['per_class'] for r in best_history['cases']}
                        require(all(r['per_class'] == previous[r['patient'], r['visit']] for r in evaluation['cases']), 'Selected checkpoint reload differs from history')
                    runtime = audit_runtime_receipt(path, evaluation, review, protocol_path, new_manifest_path, protocol)
                    if runtime is not None:
                        runtime_receipts[split, model] = runtime
                        input_files[f'v2_{split}_{model}_runtime_receipt'] = {'path': runtime['path'], 'sha256': runtime['sha256']}
                raw_evaluations[version, model] = (path, evaluation)
                input_files[f'{version}_{split}_{model}'] = {'path': str(path), 'sha256': sha(path)}
        common_path = find_common(artifacts, split, getattr(args, f'common_{split}'))
        common = load(common_path)
        require(common['split'] == split and (split != 'test' or common['retrospective_test_acknowledged']), 'Common comparison split/test status mismatch')
        if split == 'test':
            require(common['input_files']['selection_freeze']['sha256'] == sha(freeze_path), 'Common TEST report uses a different selection freeze')
        for name, expected in [('v1_manifest', sha(old_manifest_path)), ('v2_manifest', sha(new_manifest_path)), ('v1_cache_identity', sha(old_identity_path))]:
            require(common['input_files'][name]['sha256'] == expected, f'Common grid input differs:{name}')
        require(set(common['models']) == set(MODELS), 'Both model pairs required regardless of gates')
        gt_rows = {(r['patient'], r['visit']): r for r in common['common_gt_inputs']}
        require(set(gt_rows) == set(records(old, split)) and len(common['common_gt_inputs']) == len(gt_rows), 'Common GT coverage mismatch')
        for key, record in records(old, split).items():
            arrays = old_identity['records'][record['cache_path']]['arrays']
            require(gt_rows[key]['decoded_mask_sha256'] == arrays['mask']['sha256'] and gt_rows[key]['decoded_spacing_sha256'] == arrays['spacing']['sha256'], 'Common GT decoded identity differs')
        common_reports[split], grid_evaluations[split], comparisons[split] = common, {}, {}
        for model in MODELS:
            pair = common['models'][model]
            for version in ('v1', 'v2'):
                path, raw = raw_evaluations[version, model]
                descriptor = common['input_files']['evaluations'][f'{version}_{model}']
                require(descriptor['sha256'] == sha(path) and descriptor['checkpoint_sha256'] == raw['checkpoint_sha256'], 'Common comparison uses a different selected evaluation')
                audit_evaluation(pair[version], records(old, split), f'common/{split}/{version}/{model}', counters, surface=True)
                raw_index = {(r['patient'], r['visit']): r for r in raw['cases']}
                for row in pair[version]['cases']:
                    key = row['patient'], row['visit']
                    original_record = records(old, split)[key]
                    require(row['shape_dhw'] == old['target_shape_dhw'] and np.allclose(row['spacing_dhw'], original_record['spacing_dhw_mm'], rtol=0, atol=1e-6), 'Result is not on the common original grid')
                    if version == 'v1':
                        require(row['per_class'] == raw_index[key]['per_class'], 'Common original benchmark was changed')
                    else:
                        reverse = np.eye(4)[[2, 1, 0, 3]]
                        new_record = records(new, split)[key]
                        expected = reverse @ np.linalg.inv(np.asarray(new_record['resampled_affine_xyz'])) @ np.asarray(original_record['resampled_affine_xyz']) @ reverse
                        require(np.allclose(row['target_common_dhw_to_source_v2_dhw'], expected, rtol=0, atol=1e-9), 'V2 common-world mapping mismatch')
                grid_evaluations[split][f'{version}_{model}'] = detailed_model_summary(pair[version])
            audit_paired(pair['paired'], pair['v1']['cases'], pair['v2']['cases'], f'{split}/{model}')
            old_agg, new_agg = (pair[v]['aggregate'] for v in ('v1', 'v2'))
            pd = new_agg['per_class'][5]['dice'] - old_agg['per_class'][5]['dice']
            fd = new_agg['foreground_mean_dice'] - old_agg['foreground_mean_dice']
            expected_pass = pd >= protocol['adoption']['minimum_prostate_dice_gain'] - 1e-12 and fd >= protocol['adoption']['minimum_foreground_dice_difference'] - 1e-12
            gate = pair['gate']
            equal(gate['prostate_dice_delta'], pd, 'Gate P5 delta')
            equal(gate['foreground_dice_delta'], fd, 'Gate FG delta')
            require(gate['minimum_prostate_delta'] == protocol['adoption']['minimum_prostate_dice_gain'] and gate['minimum_foreground_delta'] == protocol['adoption']['minimum_foreground_dice_difference'], 'Adoption threshold changed')
            require(gate['floating_point_tolerance'] == 1e-12, 'Gate numerical tolerance changed')
            require(gate['numeric_conditions_pass'] == expected_pass and gate['replacement_supported_by_validation_gate'] == (expected_pass if split == 'validation' else None), 'Gate failed/opened incorrectly')
            comparisons[split][model] = {'gate': gate, 'paired': pair['paired']}
        input_files[f'common_{split}'] = {'path': str(common_path), 'sha256': sha(common_path)}
    execution_path = artifacts / 'formal_execution_receipt.json'
    execution = load(execution_path) if execution_path.is_file() else None
    runner_path = review / 'formal_v2_gpu.slurm'
    if execution is not None:
        runner_file = Path(execution.get('runner_file', 'formal_v2_gpu.slurm'))
        require(not runner_file.is_absolute() and '..' not in runner_file.parts,
                'Execution runner must be a safe path within the review directory')
        runner_path = (review / runner_file).resolve()
        require(runner_path.is_relative_to(review.resolve()), 'Execution runner escapes the review directory')
        require(execution['runner_sha256'] == sha(runner_path), 'Execution receipt runner digest mismatch')
        input_files['formal_execution_receipt'] = {'path': str(execution_path), 'sha256': sha(execution_path)}
    execution_same_allocation = (execution is not None and execution.get('same_allocation') is True
        and str(execution.get('job_id', '')).isdigit() and int(execution['job_id']) > 0
        and execution.get('state') == 'COMPLETED' and isinstance(execution.get('node_list'), str)
        and bool(execution['node_list'].strip())
        and execution.get('provenance_source') == 'slurm_sacct_and_frozen_runner')
    resources = {}
    for version in ('v1', 'v2'):
        for model in MODELS:
            _, evaluation = (project / f'artifacts/test_final/{model}/evaluation.json', load(project / f'artifacts/test_final/{model}/evaluation.json')) if version == 'v1' else (artifacts / f'test_v2/{model}/evaluation.json', load(artifacts / f'test_v2/{model}/evaluation.json'))
            resources[f'{version}_{model}'] = {**runs[version][model]['resources'],
                'mean_whole_case_inference_ms': evaluation['mean_inference_ms'],
                'peak_inference_allocated_cuda_bytes': evaluation['peak_allocated_cuda_bytes'],
                'latency_protocol': evaluation['latency_protocol'],
                'inference_runtime_metadata': {k: evaluation.get(k) for k in ('gpu_name', 'device', 'torch_version', 'python_version', 'cuda_version')},
                'inference_runtime_fully_recorded': all(evaluation.get(k) is not None for k in ('gpu_name', 'device', 'torch_version', 'python_version'))}
            direct = resources[f'{version}_{model}']['inference_runtime_fully_recorded']
            runtime = runtime_receipts.get(('test', model)) if version == 'v2' else None
            resources[f'{version}_{model}']['inference_runtime_directly_in_evaluation_json'] = direct
            if runtime is not None:
                resources[f'{version}_{model}']['inference_runtime_metadata'] = {k: runtime['record'][k]
                    for k in ('gpu_name', 'device', 'torch_version', 'python_version', 'cuda_version')}
                resources[f'{version}_{model}']['inference_runtime_receipt'] = runtime
                resources[f'{version}_{model}']['inference_runtime_fully_recorded'] = True
            resources[f'{version}_{model}']['inference_runtime_evidence'] = ('direct_runtime_receipt_bound_to_evaluation' if runtime is not None
                else 'direct_evaluation_metadata' if direct
                else 'same_allocation_runner_provenance_with_execution_receipt' if version == 'v2' and execution_same_allocation
                else 'runner_digest_bound_receipt_only' if version == 'v2' and execution is not None
                else 'inferred_from_same_job_runner' if version == 'v2' else 'not_recorded')
            train_runtime = resources[f'{version}_{model}']['runtime']
            infer_runtime = resources[f'{version}_{model}']['inference_runtime_metadata']
            runtime_keys = ('gpu_name', 'device', 'python_version', 'torch_version', 'cuda_version')
            resources[f'{version}_{model}']['inference_matches_training_runtime_on_recorded_fields'] = (
                all(train_runtime.get(k) == infer_runtime.get(k) for k in runtime_keys)
                if all(train_runtime.get(k) is not None and infer_runtime.get(k) is not None for k in runtime_keys) else None)
            if version == 'v2':
                resources[f'{version}_{model}']['same_job_runner_sha256'] = sha(runner_path)
    public_runs = {v: {m: {k: value for k, value in runs[v][m].items() if k not in ('run', 'history')} for m in MODELS} for v in ('v1', 'v2')}
    eligible = [m for m in MODELS if comparisons['validation'][m]['gate']['replacement_supported_by_validation_gate']]
    preparation = {}
    for key, path in [('recovery_amendment', review / 'cache_recovery_amendment.json'),
                      ('execution_cost_receipt', artifacts / 'preparation_execution_receipt.json')]:
        if path.is_file():
            preparation[key] = {'path': str(path), 'sha256': sha(path), 'record': load(path)}
            input_files[f'preparation_{key}'] = {'path': str(path), 'sha256': sha(path)}
    return {'schema_version': 1, 'status': 'audit_passed_complete_pair', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'scientific_status': 'One exploratory pair after previously inspected TEST; same reused VAL patients; no new independent generalization conclusion.',
            'input_files': input_files, 'source_checks': source_checks, 'audit_counts': dict(counters),
            'runs': public_runs, 'common_grid_results': grid_evaluations, 'comparisons': comparisons,
            'validation_adoption': {'basis': 'Frozen same-model common-grid VAL gates only; no cross-model winner selected on TEST.', 'eligible_models': eligible, 'all_models_retained': True},
            'resources': resources,
            'formal_execution_receipt': execution,
            'engineering_preparation': preparation,
            'historical_adaptation': 'A1 failed its clean-prostate guard after an original stop rule saying not to retune/evaluate M015; later human-authorized M1/R1 continued under new protocols. Round-specific freezing does not erase this across-round adaptation.',
            'limits': ['New-grid v2 selection scores cannot be directly compared with old-grid v1 scores. All reported v1/v2 comparisons use unchanged original64x128x128 GT.', 'No pixel arrays opened by this summarizer; count/hash/metadata consistency is audited, but model checkpoint tensors, contours and surface distances are not independently remeasured here.', 'HD95 is on the downsampled grid and conditional on both-present masks; paired support/exclusions are reported. Dice zeros and all known failures remain.', 'Single seed/new pair; complete pipelines differ in resolution, sampling/exposure, selector and sliding-window context, alongside model differences. No isolated3D causal claim.', 'Numerical prostate Dice>=.70 is an objective result under the declared aggregation, not a course score or guarantee of grades/clinical safety.', 'Training-loop/inference measurements exclude different costs as stated; missing inference-runtime metadata is not silently filled from training metadata.']}


def number(value, digits=4):
    return '未定义' if value is None else f'{value:.{digits}f}'


def render_markdown(summary):
    lines = ['# v2有限实验：独立结果核对', '', '这是查看过旧TEST后的单次探索性双模型实验；同一批VAL也已反复使用。以下比较全部使用原版64×128×128网格与同一GT，不构成新的独立盲测，也不保证成绩。', '',
             '审计通过：两模型均完成30个epoch；选择轮次、字节身份、完整病例、指标计数与患者聚合均与冻结规则一致。这里的“通过”指结果一致性检查，并不表示性能门槛通过。', '',
             '## 模型选择和资源', '', '| 模型 | 版本 | 选择epoch | 训练循环秒 | 成功更新/AMP跳过 | 参数 | 训练峰值GiB | 全病例推理ms |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for m in MODELS:
        for v in ('v1', 'v2'):
            r, resource = summary['runs'][v][m], summary['resources'][f'{v}_{m}']
            lines.append(f'| {m} | {v} | {r["selected_epoch"]} | {number(resource["training_loop_seconds"], 2)} | {resource["successful_updates"]}/{resource["amp_skipped_updates"]} | {resource["parameters"]} | {number(resource["peak_training_allocated_cuda_bytes"] / 2**30, 3)} | {number(resource["mean_whole_case_inference_ms"], 2)} |')
    lines += ['', '训练循环不包括验证、保存、绘图和排队。推理时间范围见JSON原始记录；3D滑窗的CPU概率累加属于新推理流程。分配显存不等于GPU总占用；两套训练样本暴露和计算预算不同。']
    if summary.get('engineering_preparation'):
        lines += ['', '另有缓存准备/恢复工程记录，已在JSON的engineering_preparation完整保留。超时与恢复成本不包含在上表训练循环秒数中，也不据此声称在最初CPU准备时限内完成；实际准备总成本以作业计费记录为准。']
    lines += ['', '| 环境 | 训练GPU | 推理GPU（直接记录时） | 训练Python / Torch / CUDA | 推理环境证据 |', '|---|---|---|---|---|']
    for m in MODELS:
        for v in ('v1', 'v2'):
            resource = summary['resources'][f'{v}_{m}']; runtime = resource['runtime']
            inference = resource['inference_runtime_metadata']
            lines.append(f'| {v} {m} | {runtime.get("gpu_name") or "未记录"} | {inference.get("gpu_name") or "未直接记录"} | {runtime.get("python_version") or "未记录"} / {runtime.get("torch_version") or "未记录"} / {runtime.get("cuda_version") or "未记录"} | {resource["inference_runtime_evidence"]} |')
    different = [k for k,r in summary['resources'].items() if r['inference_matches_training_runtime_on_recorded_fields'] is False]
    if different:
        lines += ['', '直接记录的推理环境与训练环境有差异：' + '、'.join(different) + '。差异已保留在JSON，不把跨环境成本或数值变化全部归因于模型方案。']
    missing = [k for k, r in summary['resources'].items() if not r['inference_runtime_fully_recorded']]
    if missing:
        lines += ['', '推理环境元数据不完整：' + '、'.join(missing) + '。没有用训练环境代替缺失的推理证据。']
    for split, title in [('validation', '共同网格VAL：采用依据'), ('test', '共同网格已见TEST：回顾性结果')]:
        results = summary['common_grid_results'][split]
        lines += ['', f'## {title}', '', '| 指标 | v1 2D | v2 2D | v1 3D | v2 3D |', '|---|---:|---:|---:|---:|']
        order = ['v1_unet2d', 'v2_unet2d', 'v1_improved3d', 'v2_improved3d']
        for ci, name in enumerate(NAMES):
            for metric, label in [('dice', 'Dice'), ('iou', 'IoU')]:
                lines.append('| ' + f'{name} {label}' + ' | ' + ' | '.join(number(results[k]['aggregate']['per_class'][ci][metric]) for k in order) + ' |')
        lines.append('| 前景Dice | ' + ' | '.join(number(results[k]['aggregate']['foreground_mean_dice']) for k in order) + ' |')
        lines += ['', '| 患者 | visits | v1 2D前列腺Dice | v2 2D | v1 3D | v2 3D |', '|---|---:|---:|---:|---:|---:|']
        maps = {k: {p['patient']: p for p in results[k]['patients']} for k in order}
        for patient in sorted(maps[order[0]]):
            lines.append('| ' + patient + f' | {maps[order[0]][patient]["visits"]} | ' + ' | '.join(number(maps[k][patient]['prostate_dice']) for k in order) + ' |')
        lines += ['', '| 同模型v2−v1 | 前列腺Dice变化 | 前景Dice变化 | paired HD95 v1/v2 mm | 可比较病例/排除 |', '|---|---:|---:|---:|---:|']
        for m in MODELS:
            c = summary['comparisons'][split][m]; g, h = c['gate'], c['paired']['paired_hd95']
            lines.append(f'| {m} | {number(g["prostate_dice_delta"])} | {number(g["foreground_dice_delta"])} | {number(h["v1_hd95_mm"], 2)}/{number(h["v2_hd95_mm"], 2)} | {h["eligible_case_pairs"]}/{h["excluded_case_pairs"]} |')
        lines += ['', 'HD95只在两模型均有GT与预测表面的同一病例上比较；空预测仍保留Dice零分。距离属于粗网格，不代表原生分辨率临床精度。', '', '| 方案 | TP/FP/FN（描述性池化） | 空状态 | 患者均衡前列腺Dice≥.70 |', '|---|---|---|---|']
        for k in order:
            x = results[k]; ct = x['pooled_prostate_counts_descriptive_not_scoring']
            lines.append(f'| {k} | {ct["tp"]}/{ct["fp"]}/{ct["fn"]} | {json.dumps(x["aggregate"]["prostate_empty_status_counts"], ensure_ascii=False)} | {"是" if x["numeric_prostate_point7_goal_met"] else "否"} |')
    eligible = summary['validation_adoption']['eligible_models']
    lines += ['', '## 是否采用', '', '预先规定的共同VAL门槛为：各自相同结构的v2前列腺Dice比v1至少高.01，且前景Dice下降不超过.01。通过数值门槛的模型：' + ('、'.join(eligible) if eligible else '无，保留原版') + '。这只是有限探索的研究指示，TEST不能改变轮次、设置或挑选赢家。', '',
              '所有低于.70的已见病例、空预测及FP/FN均保留在JSON；这些已观察失败需要人工核查。.70是本项目的数值目标，不是本实验验证过的临床安全阈值，平均分达线也不证明个例可靠。', '',
              '## 限制与协助记录', '', summary['historical_adaptation'], '', '完整流水线同时改变多个因素，不能把效果全部归于3D。AI最终声明仍待与用户如实整理；本汇总保留实际协助、验证脚本和全部负面结果。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--v1-project', type=Path, required=True)
    parser.add_argument('--artifacts-root', type=Path, required=True, help='Directory containing cache_v2, formal_v2, validation_v2, test_v2 and selection_v2.json.')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--selection-freeze', type=Path)
    parser.add_argument('--common-validation', type=Path)
    parser.add_argument('--common-test', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output_dir.exists(), 'Use a fresh output directory; existing positive or negative reports are never overwritten.')
    summary = build_summary(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'RESULTS.md').write_text(render_markdown(summary))
    print(json.dumps({'output': str(args.output_dir), 'status': summary['status'], 'audit_counts': summary['audit_counts'], 'validation_adoption': summary['validation_adoption']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
