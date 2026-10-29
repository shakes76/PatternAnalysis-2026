"""Evaluate six numerical labels by slice, patient and patient/week case."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

import nibabel as nib
import torch

from dataset import HipMRI2DDataset
from metrics import confusion_matrix, dice_iou_from_confusion
from predict import load_predictor


CASE_PATTERN = re.compile(r'case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz')


def parse_case_name(name):
    match = CASE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f'Unexpected HipMRI slice filename: {name}')
    return tuple(int(value) for value in match.groups())


def patient_ids(values, field):
    if not isinstance(values, list) or not values or any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f'Checkpoint must record a non-empty integer {field} list')
    if len(set(values)) != len(values):
        raise ValueError(f'Duplicate patient IDs in {field}')
    return set(values)


def validate_separation(saved, split, observed, sample_count):
    splits = saved['config'].get('splits', {})
    training = patient_ids(splits.get('train_patients'), 'train_patients')
    validation = patient_ids(splits.get('validation_patients'), 'validation_patients')
    if training & validation:
        raise ValueError('Checkpoint records train/validation patient overlap')
    if observed & training:
        raise ValueError('Evaluation patients overlap checkpoint training patients')
    if split == 'validate' and observed != validation:
        raise ValueError('Validation patient IDs do not match checkpoint validation patients')
    if split == 'validate':
        expected = splits.get('validation_slices')
        if type(expected) is not int or expected < 1 or expected != sample_count:
            raise ValueError('Validation slice count does not match checkpoint metadata')
    if split == 'test' and observed & validation:
        raise ValueError('Test patients overlap checkpoint validation patients')


def summarize(matrix):
    scores = dice_iou_from_confusion(matrix)
    rows = []
    for label in range(6):
        tp = int(matrix[label, label])
        actual = int(matrix[label].sum())
        predicted = int(matrix[:, label].sum())
        rows.append({
            'label_id': label, 'tp': tp, 'fp': predicted - tp, 'fn': actual - tp,
            'target_pixels': actual, 'predicted_pixels': predicted,
            'dice': None if math.isnan(scores['dice'][label].item()) else scores['dice'][label].item(),
            'iou': None if math.isnan(scores['iou'][label].item()) else scores['iou'][label].item(),
        })
    return {'pixels': int(matrix.sum()), 'confusion': matrix.tolist(), 'classes': rows}


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def check_logits(logits, shape):
    if tuple(logits.shape) != (1, 6, *shape) or not torch.isfinite(logits).all().item():
        raise RuntimeError('Invalid evaluation logits')


def evaluate_cases(model, dataset, records, device, warmup_steps=3):
    """Batch size one preserves native grids and permits measured case runtimes.

    Warm up every selected shape before measuring cases. CUDA synchronization
    brackets model-only timing, outside image transfer, argmax and metric work.
    """
    device = torch.device(device)
    model.eval()
    shapes = set()
    groups = defaultdict(list)
    for record in records:
        shape = tuple(nib.load(str(dataset.pairs[record['index']][0])).shape)
        if len(shape) != 2 or any(size < 1 for size in shape):
            raise ValueError(f'Expected a non-empty 2D image, got {shape}')
        shapes.add(shape)
        groups[(record['patient_id'], record['week'])].append(record)
    pooled = torch.zeros(6, 6, dtype=torch.long)
    by_patient = {}
    case_results, slice_results = [], []
    with torch.inference_mode():
        for shape in sorted(shapes):
            probe = torch.zeros(1, 1, *shape, device=device)
            for _ in range(warmup_steps):
                check_logits(model(probe), shape)
            del probe
        synchronize(device)
        for (patient, week), members in sorted(groups.items()):
            case_matrix = torch.zeros(6, 6, dtype=torch.long)
            case_slices = []
            started = time.perf_counter()
            for record in members:
                image, target = dataset[record['index']]
                inputs = image.unsqueeze(0).to(device)
                synchronize(device)
                forward_start = time.perf_counter()
                logits = model(inputs)
                synchronize(device)
                forward_seconds = time.perf_counter() - forward_start
                check_logits(logits, tuple(target.shape))
                prediction = logits.argmax(dim=1)[0].cpu()
                matrix = confusion_matrix(prediction, target)
                if int(matrix.sum()) != target.numel():
                    raise RuntimeError('Slice confusion pixel count mismatch')
                case_matrix += matrix
                case_slices.append((record, matrix, forward_seconds, list(target.shape)))
            case_seconds = time.perf_counter() - started
            pooled += case_matrix
            entry = {'patient_id': patient, 'week': week, 'samples': len(members),
                     'processing_seconds': case_seconds, **summarize(case_matrix)}
            case_results.append(entry)
            if patient not in by_patient:
                by_patient[patient] = {'matrix': torch.zeros(6, 6, dtype=torch.long),
                                       'samples': 0, 'processing_seconds': 0.0}
            patient_entry = by_patient[patient]
            patient_entry['matrix'] += case_matrix
            patient_entry['samples'] += len(members)
            patient_entry['processing_seconds'] += case_seconds
            for record, matrix, seconds, shape in case_slices:
                slice_results.append({**record, 'shape': shape,
                                      'model_forward_ms': seconds * 1000,
                                      **summarize(matrix)})
    patients = [
        {'patient_id': patient, 'samples': entry['samples'],
         'processing_seconds': entry['processing_seconds'], **summarize(entry['matrix'])}
        for patient, entry in sorted(by_patient.items())
    ]
    return {'pooled': summarize(pooled), 'patients': patients,
            'cases': case_results, 'slices': slice_results,
            'warmup_shapes': [list(shape) for shape in sorted(shapes)]}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(directory, report):
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'evaluation.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    scope = {key: report[key] for key in
             ('architecture', 'split', 'smoke_only', 'partial_checkpoint', 'limited_evaluation')}
    def write_table(name, fields, rows):
        write_csv(directory / name, list(scope) + fields,
                  [{**scope, **row} for row in rows])
    metric_fields = ['label_id', 'tp', 'fp', 'fn', 'target_pixels', 'predicted_pixels', 'dice', 'iou']
    write_table('pooled_metrics.csv', metric_fields, report['pooled']['classes'])
    for level, identifiers in (('patients', ['patient_id']), ('cases', ['patient_id', 'week'])):
        rows = []
        for entry in report[level]:
            for metrics in entry['classes']:
                rows.append({**{key: entry[key] for key in identifiers},
                             'samples': entry['samples'], **metrics})
        write_table(level + '_metrics.csv', identifiers + ['samples'] + metric_fields, rows)
    slices = []
    for entry in report['slices']:
        for metrics in entry['classes']:
            slices.append({**{key: entry[key] for key in ('patient_id', 'week', 'slice_id', 'image', 'mask')},
                           'height': entry['shape'][0], 'width': entry['shape'][1],
                           'model_forward_ms': entry['model_forward_ms'], **metrics})
    write_table('slice_metrics.csv',
              ['patient_id', 'week', 'slice_id', 'image', 'mask', 'height', 'width', 'model_forward_ms'] + metric_fields,
              slices)
    write_table('case_times.csv', ['patient_id', 'week', 'samples', 'pixels', 'processing_seconds'],
              [{key: entry[key] for key in ('patient_id', 'week', 'samples', 'pixels', 'processing_seconds')}
               for entry in report['cases']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--split', choices=('validate', 'test'), default='validate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--max-samples', type=int, default=0, help='0 evaluates every slice')
    parser.add_argument('--warmup-steps', type=int, default=3)
    parser.add_argument('--allow-partial', action='store_true', help='Validation smoke checks only')
    args = parser.parse_args(argv)
    if min(args.max_samples, args.warmup_steps) < 0:
        parser.error('max-samples and warmup-steps must be non-negative')
    if args.split == 'test' and args.max_samples:
        parser.error('Test evaluation requires the full split, without max-samples')
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Evaluation directory already exists: {destination}')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    checkpoint_sha = sha256_file(args.checkpoint)
    model, saved = load_predictor(args.checkpoint, device=device, allow_partial=args.allow_partial)
    partial_checkpoint = saved['config']['partial_run']
    if args.split == 'test' and partial_checkpoint:
        raise ValueError('Test evaluation refuses partial checkpoints, even with --allow-partial')
    base = args.data_root.expanduser().resolve()
    dataset = HipMRI2DDataset(base / f'keras_slices_{args.split}',
                              base / f'keras_slices_seg_{args.split}',
                              normalize=saved['config']['preprocessing']['normalize'])
    records = []
    seen_keys = set()
    for index, (image, mask) in enumerate(dataset.pairs):
        patient, week, slice_id = parse_case_name(image.name)
        key = (patient, week, slice_id)
        if key in seen_keys:
            raise ValueError(f'Duplicate patient/week/slice identifier: {key}')
        seen_keys.add(key)
        records.append({'index': index, 'patient_id': patient, 'week': week, 'slice_id': slice_id,
                        'image': image.name, 'mask': mask.name})
    observed_patients = {record['patient_id'] for record in records}
    validate_separation(saved, args.split, observed_patients, len(dataset))
    selected = records[:args.max_samples] if args.max_samples else records
    started = time.perf_counter()
    result = evaluate_cases(model, dataset, selected, device, args.warmup_steps)
    processing_seconds = time.perf_counter() - started
    if result['pooled']['pixels'] != sum(entry['pixels'] for entry in result['slices']):
        raise RuntimeError('Global confusion pixel count mismatch')
    latencies = [entry['model_forward_ms'] for entry in result['slices']]
    by_shape = defaultdict(list)
    for entry in result['slices']:
        by_shape[tuple(entry['shape'])].append(entry['model_forward_ms'])
    if sha256_file(args.checkpoint) != checkpoint_sha:
        raise RuntimeError('Checkpoint changed during evaluation')
    report = {
        'schema_version': 1, 'architecture': saved['architecture'], 'model_config': saved['model_config'],
        'checkpoint': str(args.checkpoint.expanduser().resolve()), 'checkpoint_sha256': checkpoint_sha,
        'checkpoint_epoch': saved['epoch'], 'partial_checkpoint': partial_checkpoint,
        'limited_evaluation': bool(args.max_samples),
        'smoke_only': partial_checkpoint or bool(args.max_samples),
        'split': args.split, 'data_root': str(base), 'max_samples': args.max_samples,
        'split_samples': len(dataset), 'processed_samples': len(selected),
        'split_patient_ids': sorted(observed_patients),
        'processed_patient_ids': sorted({entry['patient_id'] for entry in selected}),
        'normalization': 'per-slice z-score' if dataset.normalize else 'none',
        'label_semantics': 'Numerical IDs 0-5; anatomical names unverified',
        'device': str(device), 'torch_version': str(torch.__version__),
        'torch_cuda_build': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'batch_size': 1, 'warmup_steps_per_shape': args.warmup_steps,
        'cuda_timing_reference': 'https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html',
        'median_model_forward_ms': statistics.median(latencies),
        'model_forward_by_shape': [{'shape': list(shape), 'samples': len(values),
                                    'median_model_forward_ms': statistics.median(values)}
                                   for shape, values in sorted(by_shape.items())],
        'latency_scope': 'One forward per selected slice after per-shape warmup; excludes I/O, transfer, argmax and metrics; CUDA synchronized',
        'case_time_scope': 'Per patient/week slice case; includes I/O, preprocessing, transfers, forward, argmax and confusion counts; excludes warmup and export',
        'evaluation_seconds_including_header_scan_and_warmup': processing_seconds,
        'aggregation': 'Pool original-pixel confusion counts independently for global, patient and patient/week case; no mean of slice Dice',
        'checkpoint_source_sha256': saved['config'].get('source_sha256', {}),
        'evaluation_source_sha256': {name: sha256_file(Path(__file__).with_name(name))
                                     for name in ('evaluate.py', 'predict.py', 'model_factory.py', 'dataset.py', 'metrics.py')},
        **result,
    }
    export_results(destination, report)
    print('Architecture:', report['architecture'])
    print('Split:', args.split, '; processed:', len(selected), '/', len(dataset), 'slices')
    print('Patient separation: PASS')
    print('Smoke only:', report['smoke_only'])
    print('Patients:', len(result['patients']), '; cases:', len(result['cases']))
    for entry in result['pooled']['classes']:
        print(f'Label {entry["label_id"]}: Dice={entry["dice"]}, IoU={entry["iou"]}')
    print('Median model-forward ms:', report['median_model_forward_ms'])
    print('Output directory:', destination)
    print('Evaluation export: PASS')
    return report


if __name__ == '__main__':
    main()
