"""Audit and compare two completed validation experiments from one frozen protocol."""
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

from run_comparison import (ARCHITECTURES, RUNTIME_FILES, check_training, read_json,
                            summarize_validation, validate_protocol)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric_rows(summary):
    matrix = summary['confusion']
    require(isinstance(matrix, list) and len(matrix) == 6 and all(isinstance(row, list) and len(row) == 6 for row in matrix),
            'Expected a six-class confusion matrix')
    require(all(type(value) is int and value >= 0 for row in matrix for value in row), 'Invalid confusion counts')
    require(sum(map(sum, matrix)) > 0 and summary['pixels'] == sum(map(sum, matrix)), 'Confusion pixel total mismatch')
    expected = []
    for label in range(6):
        tp = matrix[label][label]
        actual = sum(matrix[label])
        predicted = sum(row[label] for row in matrix)
        denominator, union = actual + predicted, actual + predicted - tp
        expected.append({'label_id': label, 'target_pixels': actual, 'predicted_pixels': predicted,
                         'tp': tp, 'fp': predicted - tp, 'fn': actual - tp,
                         'dice': 2 * tp / denominator if denominator else None,
                         'iou': tp / union if union else None})
    require(len(summary['classes']) == 6, 'Expected six recorded label metrics')
    for recorded, calculated in zip(summary['classes'], expected):
        for key, value in calculated.items():
            if key in ('dice', 'iou') and value is not None:
                require(type(recorded[key]) in (int, float) and math.isfinite(recorded[key]) and
                        math.isclose(recorded[key], value, rel_tol=1e-10, abs_tol=1e-12), 'Recorded overlap metric differs from counts')
            else:
                require(recorded[key] == value, 'Recorded label metric differs from counts')
    return expected


def summed_matrices(entries):
    return [[sum(entry['confusion'][row][column] for entry in entries)
             for column in range(6)] for row in range(6)]


def inspect_run(directory, architecture):
    directory = Path(directory).expanduser().resolve()
    provenance = read_json(directory / 'experiment_provenance.json')
    result = read_json(directory / 'comparison_result.json')
    protocol = validate_protocol(read_json(directory / 'protocol.json'))
    data = read_json(directory / 'data_manifest.json')
    require(provenance['status'] == result['status'] == 'completed', 'Comparison task has not completed')
    require(provenance['architecture'] == result['architecture'] == architecture, 'Unexpected run architecture')
    require(provenance['protocol_id'] == result['protocol_id'] == protocol['protocol_id'], 'Protocol ID mismatch')
    require(provenance['source_commit'] == result['source_commit'], 'Source revision mismatch within run')
    require(isinstance(provenance['source_commit'], str) and re.fullmatch(r'[0-9a-f]{40}', provenance['source_commit']), 'Invalid source commit')
    require(set(provenance['source_sha256']) == set(RUNTIME_FILES), 'Incomplete runtime source hash record')
    require(all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) for value in provenance['source_sha256'].values()), 'Invalid runtime source hash')
    serialized = json.dumps(data['manifest'], sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(serialized).hexdigest()
    require(digest == data['filename_size_manifest_sha256'] == provenance['filename_size_manifest_sha256'],
            'Data filename/size manifest digest mismatch')
    config, profile = check_training(directory, protocol, architecture, data)
    history = read_json(directory / 'training/history.json')
    require(result['completed_epochs'] == protocol['training']['epochs'], 'Incomplete comparison epoch count')
    require(result['training_gpu_profile'] == profile, 'Cached GPU profile differs from saved training profile')
    for entry in history:
        for phase in ('train', 'validation'):
            loss = entry[phase]['loss']
            require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0, 'Invalid history loss')
    selected_epoch = min(history, key=lambda entry: entry['validation']['loss'])['epoch']
    validation = summarize_validation(directory, protocol, architecture, data)
    require(validation['best_epoch'] == selected_epoch, 'Selected checkpoint does not minimize full-validation loss')
    for key, value in validation.items():
        require(result[key] == value, 'Cached comparison result differs from validation exports')
    for name, sha in config['source_sha256'].items():
        require(provenance['source_sha256'].get(name) == sha, 'Training and experiment source records differ')
    pooled_metrics = metric_rows(validation['pooled'])
    patients = validation['patients']
    require(sorted(entry['patient_id'] for entry in patients) == data['splits']['validation']['patients'],
            'Patient summary IDs mismatch')
    for patient in patients:
        metric_rows(patient)
    require(summed_matrices(patients) == validation['pooled']['confusion'], 'Patient confusion counts do not sum to global counts')
    reports = [read_json(directory / f'validation_{index}/evaluation.json')
               for index in range(1, protocol['evaluation']['repeats'] + 1)]
    for report in reports:
        require(report['device'] == 'cuda' and report['batch_size'] == 1 and
                report['warmup_steps_per_shape'] == protocol['evaluation']['warmup_steps'], 'Unexpected inference timing configuration')
        require(report['partial_checkpoint'] is False and report['limited_evaluation'] is False,
                'Partial validation results cannot enter the formal comparison')
        require(summed_matrices(report['cases']) == report['pooled']['confusion'], 'Case confusion counts do not sum to global counts')
        for case in report['cases']:
            metric_rows(case)
            seconds = case['processing_seconds']
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0, 'Invalid case processing time')
    return {'directory': str(directory), 'provenance': provenance, 'result': result,
            'protocol': protocol, 'data': data, 'config': config, 'history': history,
            'profile': profile, 'validation': validation, 'reports': reports, 'pooled_metrics': pooled_metrics}


def difference(left, right):
    return None if left is None or right is None else right - left


def collect_comparison(baseline_directory, improved_directory):
    runs = [inspect_run(baseline_directory, ARCHITECTURES[0]), inspect_run(improved_directory, ARCHITECTURES[1])]
    baseline, improved = runs
    require(baseline['protocol'] == improved['protocol'], 'Protocols differ between models')
    for key in ('source_commit', 'source_sha256', 'filename_size_manifest_sha256'):
        require(baseline['provenance'][key] == improved['provenance'][key], f'Experiment {key} differs between models')
    require(baseline['data'] == improved['data'], 'Data manifests differ between models')
    rows, patient_rows, resource_rows, case_times = [], [], [], []
    for left, right in zip(baseline['pooled_metrics'], improved['pooled_metrics']):
        require(left['target_pixels'] == right['target_pixels'], 'Validation ground-truth counts differ')
        rows.append({'split': 'validate', 'label_id': left['label_id'], 'target_pixels': left['target_pixels'],
                     'baseline_dice': left['dice'], 'improved_dice': right['dice'],
                     'dice_delta_improved_minus_baseline': difference(left['dice'], right['dice']),
                     'baseline_iou': left['iou'], 'improved_iou': right['iou'],
                     'iou_delta_improved_minus_baseline': difference(left['iou'], right['iou'])})
    for left, right in zip(baseline['validation']['patients'], improved['validation']['patients']):
        require(left['patient_id'] == right['patient_id'] and left['samples'] == right['samples'], 'Patient groups differ')
        for lmetric, rmetric in zip(metric_rows(left), metric_rows(right)):
            require(lmetric['target_pixels'] == rmetric['target_pixels'], 'Patient ground-truth counts differ')
            patient_rows.append({'split': 'validate', 'patient_id': left['patient_id'], 'label_id': lmetric['label_id'],
                                 'target_pixels': lmetric['target_pixels'], 'baseline_dice': lmetric['dice'],
                                 'improved_dice': rmetric['dice'], 'baseline_iou': lmetric['iou'], 'improved_iou': rmetric['iou']})
    for run in runs:
        profile, validation = run['profile'], run['validation']
        require(type(profile['peak_allocated_bytes']) is int and profile['peak_allocated_bytes'] > 0 and
                type(profile['peak_reserved_bytes']) is int and profile['peak_reserved_bytes'] >= 0, 'Invalid allocator peaks')
        wall_seconds = profile['run_wall_seconds_including_io_and_plot']
        require(type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and wall_seconds >= 0, 'Invalid training wall time')
        resource_rows.append({'architecture': run['result']['architecture'], 'split': 'validate',
                              'source_commit': run['provenance']['source_commit'],
                              'completed_epochs': run['result']['completed_epochs'], 'best_epoch': validation['best_epoch'],
                              'best_validation_loss': min(entry['validation']['loss'] for entry in run['history']),
                              'gpu_name': profile['device_name'], 'node': profile['node'],
                              'peak_allocated_gib': profile['peak_allocated_bytes'] / 1024**3,
                              'peak_reserved_gib': profile['peak_reserved_bytes'] / 1024**3,
                              'training_wall_seconds_including_io_plot': profile['run_wall_seconds_including_io_and_plot'],
                              'median_of_forward_trial_medians_ms': validation['median_of_trial_medians_ms'],
                              'forward_trial_medians_ms_json': json.dumps(validation['validation_model_forward_median_ms_trials'])})
        for trial in validation['case_times']:
            for case in trial['cases']:
                case_times.append({'architecture': run['result']['architecture'], 'split': 'validate', 'trial': trial['trial'],
                                   **{key: case[key] for key in ('patient_id', 'week', 'samples', 'pixels', 'processing_seconds')}})
    return {'schema_version': 1, 'scope': 'Validation comparison only; not final held-out results',
            'label_semantics': 'Numerical IDs 0-5; anatomical mapping unverified',
            'protocol': baseline['protocol'], 'source_commit': baseline['provenance']['source_commit'],
            'source_sha256': baseline['provenance']['source_sha256'],
            'filename_size_manifest_sha256': baseline['data']['filename_size_manifest_sha256'],
            'input_directories': [run['directory'] for run in runs],
            'per_class': rows, 'per_patient': patient_rows, 'model_summary': resource_rows,
            'case_times': case_times, 'histories': {run['result']['architecture']: run['history'] for run in runs},
            'limitations': baseline['protocol']['limitations'] + [
                'Single training seed; no confidence interval from independent model fits.',
                'Three timing passes reuse the same weights, and hardware nodes may differ.',
                'Training allocator peaks are not whole-device memory; training wall time is not inference latency.',
                'Filename/size manifests do not establish byte-identical input contents.'],
            'reporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write_csv(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_number(value):
    return 'undefined' if value is None else f'{value:.4f}'


def render_plots(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    colors = ['#2878B5', '#E78632']
    names = ['Baseline', 'Improved']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for index, architecture in enumerate(ARCHITECTURES):
        history = report['histories'][architecture]
        epochs = [entry['epoch'] for entry in history]
        for axis, phase in zip(axes, ('train', 'validation')):
            axis.plot(epochs, [entry[phase]['loss'] for entry in history], label=names[index], color=colors[index])
            axis.set(title=phase.title() + ' loss', xlabel='Epoch', ylabel='Shared Dice + CE loss')
            axis.grid(alpha=0.25)
            axis.legend()
        best = report['model_summary'][index]['best_epoch']
        axes[1].scatter([best], [history[best - 1]['validation']['loss']], color=colors[index], marker='*', s=100)
    fig.suptitle('Full training histories | selection on validation loss')
    fig.savefig(destination / 'training_loss_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for axis, metric in zip(axes, ('dice', 'iou')):
        for index, prefix in enumerate(('baseline', 'improved')):
            positions = [label - 0.18 + 0.36 * index for label in range(6)]
            values = [entry[prefix + '_' + metric] for entry in report['per_class']]
            axis.bar(positions, [float('nan') if value is None else value for value in values],
                     width=0.34, label=names[index], color=colors[index])
            for position, value in zip(positions, values):
                if value is None:
                    axis.text(position, 0.025, 'undefined', rotation=90, ha='center', va='bottom', fontsize=7, color=colors[index])
        axis.set(title='Pooled validation ' + metric.upper(), xlabel='Numeric label ID', ylabel='Score', ylim=(0, 1.04))
        axis.set_xticks(range(6))
        axis.grid(axis='y', alpha=0.25)
        axis.legend()
    fig.suptitle('Validation only | anatomical label mapping unverified')
    fig.savefig(destination / 'validation_metrics_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for offset, key, label, color in ((-0.18, 'peak_allocated_gib', 'Allocated', colors[0]),
                                      (0.18, 'peak_reserved_gib', 'Reserved', colors[1])):
        values = [entry[key] for entry in report['model_summary']]
        positions = [index + offset for index in range(2)]
        axes[0].bar(positions, values, width=0.34, label=label, color=color)
        for x, y in zip(positions, values):
            axes[0].annotate(f'{y:.3f}', (x, y), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    axes[0].set(title='Training tensor/allocator peaks', ylabel='GiB')
    axes[0].set_ylim(0, max(max(entry['peak_reserved_gib'], entry['peak_allocated_gib']) for entry in report['model_summary']) * 1.3)
    axes[0].legend()
    for index, row in enumerate(report['model_summary']):
        values = json.loads(row['forward_trial_medians_ms_json'])
        jitter = [(position - (len(values) - 1) / 2) * 0.07 for position in range(len(values))]
        axes[1].scatter([index + offset for offset in jitter], values, color=colors[index], s=45)
        axes[1].hlines(statistics.median(values), index - 0.18, index + 0.18, color='black', linewidth=2)
    axes[1].set(title='Validation timing-pass medians (line = median)', ylabel='Model-forward ms, batch size 1')
    for axis in axes:
        axis.set_xticks([0, 1], names)
        axis.grid(axis='y', alpha=0.25)
    fig.suptitle('Resource observations | allocator memory and model-only forward timing')
    fig.savefig(destination / 'resource_comparison.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--improved-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Comparison output already exists: {destination}')
    report = collect_comparison(args.baseline_run, args.improved_run)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'comparison.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    for name, rows in (('per_class_comparison.csv', report['per_class']),
                        ('patient_metrics_comparison.csv', report['per_patient']),
                        ('model_summary.csv', report['model_summary']), ('case_times.csv', report['case_times'])):
        scope = {'protocol_id': report['protocol']['protocol_id'], 'source_commit': report['source_commit'],
                 'scope': 'validation_only_numeric_labels'}
        write_csv(destination / name, [{**scope, **row} for row in rows])
    lines = ['# Validation comparison', '', report['scope'], '',
             'Numeric labels only; anatomical mapping remains unverified.', '',
             '| Label ID | Baseline Dice | Improved Dice | Baseline IoU | Improved IoU |',
             '|---|---:|---:|---:|---:|']
    for row in report['per_class']:
        lines.append('| ' + ' | '.join([str(row['label_id'])] + [display_number(row[key]) for key in
                     ('baseline_dice', 'improved_dice', 'baseline_iou', 'improved_iou')]) + ' |')
    lines += ['', 'Source commit: ' + report['source_commit'], '', 'Limitations:', '']
    lines.extend('- ' + item for item in report['limitations'])
    (destination / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not args.skip_plots:
        render_plots(report, destination)
    print('Compared source commit:', report['source_commit'])
    print('Protocol:', report['protocol']['protocol_id'])
    print('Scope:', report['scope'])
    print('Output directory:', destination)
    print('Validation comparison export: PASS')
    return report


if __name__ == '__main__':
    main()
