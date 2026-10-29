"""Run one frozen full-split comparison task, then three validation/timing passes."""
import argparse
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


ARCHITECTURES = ('UNet2D', 'ImprovedUNet2D')
RUNTIME_FILES = (
    'train.py', 'training_io.py', 'plot_history.py', 'gpu_profile.py', 'modules.py',
    'dataset.py', 'losses.py', 'metrics.py', 'check_splits.py', 'shape_batching.py',
    'model_factory.py', 'predict.py', 'evaluate.py', 'check_checkpoint.py',
    'run_comparison.py', 'comparison_protocol.json', 'comparison_train.sbatch',
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_protocol(protocol):
    required = {'schema_version', 'protocol_id', 'frozen_date', 'architectures', 'training',
                'evaluation', 'data', 'selection', 'resource_budget', 'limitations'}
    if not isinstance(protocol, dict) or set(protocol) != required or type(protocol['schema_version']) is not int or protocol['schema_version'] != 1:
        raise ValueError('Unexpected protocol schema/fields')
    if not isinstance(protocol['protocol_id'], str) or not re.fullmatch(r'[a-z0-9-]+', protocol['protocol_id']):
        raise ValueError('Expected a stable protocol ID')
    date.fromisoformat(protocol['frozen_date'])
    if protocol['architectures'] != list(ARCHITECTURES):
        raise ValueError('Protocol must name both supported architectures in array order')
    training = protocol['training']
    if not isinstance(training, dict) or set(training) != {'epochs', 'batch_size', 'base_channels', 'lr', 'seed', 'device',
                          'augment', 'max_train_batches', 'max_val_batches'}:
        raise ValueError('Unexpected training fields')
    for key in ('epochs', 'batch_size', 'base_channels'):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(training['seed']) is not int or not 0 <= training['seed'] < 2**63:
        raise ValueError('Expected a non-negative integer seed')
    if type(training['lr']) not in (int, float) or not math.isfinite(training['lr']) or training['lr'] <= 0:
        raise ValueError('Expected a finite positive learning rate')
    if training['device'] != 'cuda' or type(training['augment']) is not bool:
        raise ValueError('Comparison requires CUDA and an explicit augmentation setting')
    if any(type(training[key]) is not int or training[key] != 0
           for key in ('max_train_batches', 'max_val_batches')):
        raise ValueError('Comparison requires every train/validation batch')
    evaluation = protocol['evaluation']
    if not isinstance(evaluation, dict) or set(evaluation) != {'split', 'max_samples', 'warmup_steps', 'repeats'}:
        raise ValueError('Unexpected evaluation fields')
    if evaluation['split'] != 'validate' or type(evaluation['max_samples']) is not int or evaluation['max_samples'] != 0:
        raise ValueError('The comparison runner evaluates full validation only')
    if type(evaluation['warmup_steps']) is not int or evaluation['warmup_steps'] < 1:
        raise ValueError('Timing requires positive warmup steps')
    if type(evaluation['repeats']) is not int or not 1 <= evaluation['repeats'] <= 5:
        raise ValueError('Evaluation repeats must be between one and five')
    data = protocol['data']
    if not isinstance(data, dict) or set(data) != {'root', 'train_slices', 'validation_slices', 'train_patients', 'validation_patients'}:
        raise ValueError('Unexpected data fields')
    if not isinstance(data['root'], str) or not data['root']:
        raise ValueError('Expected a dataset root')
    for key in ('train_slices', 'validation_slices', 'train_patients', 'validation_patients'):
        if type(data[key]) is not int or data[key] < 1:
            raise ValueError(f'Expected positive {key}')
    if protocol['selection'] != 'minimum_validation_loss_earliest_tie':
        raise ValueError('Expected fixed validation-loss checkpoint selection')
    if protocol['resource_budget'] != {'array_tasks': 2, 'gpus_per_task': 1,
                                       'gpu_type': 'a100', 'time_per_task': '04:00:00'}:
        raise ValueError('Resource budget must match the checked-in Slurm script')
    if not isinstance(protocol['limitations'], list) or not protocol['limitations'] or any(
            not isinstance(item, str) for item in protocol['limitations']):
        raise ValueError('Protocol limitations must be recorded')
    return protocol


def source_hashes(directory):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
            for name in RUNTIME_FILES}


def check_sources(directory, expected):
    if source_hashes(directory) != expected:
        raise RuntimeError('Runtime snapshot changed during the experiment')


def build_plan(protocol, architecture, output, source, python_bin):
    validate_protocol(protocol)
    if architecture not in ARCHITECTURES:
        raise ValueError('Unsupported comparison architecture')
    output, source = Path(output), Path(source)
    training = protocol['training']
    train = [str(python_bin), '-u', str(source / 'train.py'), '--architecture', architecture,
             '--data-root', str(Path(protocol['data']['root']).expanduser()),
             '--output-dir', str(output / 'training')]
    for key in ('epochs', 'batch_size', 'base_channels', 'lr', 'seed', 'device', 'max_train_batches', 'max_val_batches'):
        train.extend(['--' + key.replace('_', '-'), str(training[key])])
    if training['augment']:
        train.append('--augment')
    plan = [('training', train), ('checkpoint_check',
            [str(python_bin), '-u', str(source / 'check_checkpoint.py'), '--run-dir', str(output / 'training')])]
    for repeat in range(1, protocol['evaluation']['repeats'] + 1):
        plan.append((f'validation_{repeat}', [str(python_bin), '-u', str(source / 'evaluate.py'),
            '--checkpoint', str(output / 'training/best.pt'),
            '--data-root', str(Path(protocol['data']['root']).expanduser()), '--split', 'validate',
            '--output-dir', str(output / f'validation_{repeat}'), '--device', 'cuda', '--max-samples', '0',
            '--warmup-steps', str(protocol['evaluation']['warmup_steps'])]))
    return plan


def audit_data(protocol):
    from check_splits import collect_patients
    root = Path(protocol['data']['root']).expanduser().resolve()
    manifest, splits = {}, {}
    for name, folder in (('train', 'train'), ('validation', 'validate')):
        images, masks = root / f'keras_slices_{folder}', root / f'keras_slices_seg_{folder}'
        patients, count = collect_patients(images)
        if count != protocol['data'][name + '_slices'] or len(patients) != protocol['data'][name + '_patients']:
            raise ValueError(f'{name} slice/patient counts differ from the protocol')
        pairs = []
        for image in sorted(images.glob('case_*.nii.gz')):
            mask = masks / ('seg_' + image.name[len('case_'):])
            if not mask.is_file():
                raise ValueError(f'Missing paired mask: {mask}')
            pairs.append({'image': image.name, 'mask': mask.name,
                          'image_bytes': image.stat().st_size, 'mask_bytes': mask.stat().st_size})
        manifest[name] = pairs
        splits[name] = {'patients': sorted(patients), 'samples': count}
    if set(splits['train']['patients']) & set(splits['validation']['patients']):
        raise ValueError('Training and validation patient overlap')
    serialized = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return {'root': str(root), 'splits': splits, 'filename_size_manifest_sha256': hashlib.sha256(serialized).hexdigest(),
            'manifest_scope': 'Paired filenames and byte sizes; not hashes of image/mask contents',
            'manifest': manifest}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_training(output, protocol, architecture, data):
    folder = Path(output) / 'training'
    config = read_json(folder / 'config.json')['config']
    history = read_json(folder / 'history.json')
    training = protocol['training']
    require(config['architecture'] == architecture and config['partial_run'] is False,
            'Unexpected training architecture or partial-run scope')
    require(len(history) == training['epochs'], 'Training did not complete every requested epoch')
    for epoch, entry in enumerate(history, start=1):
        require(entry['epoch'] == epoch, 'Invalid recorded epoch sequence')
        for phase in ('train', 'validation'):
            require(entry[phase]['samples'] == data['splits'][phase]['samples'],
                    f'{phase} epoch did not process every sample')
    for key, value in training.items():
        require(config['arguments'][key] == value, f'Training argument differs from protocol: {key}')
    expected_model = {'in_channels': 1, 'num_classes': 6, 'base_channels': training['base_channels']}
    if architecture == 'ImprovedUNet2D':
        expected_model['dropout'] = 0.3
    require(config['model_config'] == expected_model, 'Unexpected recorded model configuration')
    require(config['preprocessing'] == {'normalize': True, 'one_hot': True,
                'train_flip_probability': 0.5 if training['augment'] else 0.0}, 'Unexpected preprocessing')
    for phase in ('train', 'validation'):
        key = 'validation_patients' if phase == 'validation' else 'train_patients'
        require(config['splits'][key] == data['splits'][phase]['patients'], 'Saved patient IDs differ from preflight')
    profile = read_json(folder / 'gpu_profile.json')
    require(profile['architecture'] == architecture and profile['partial_run'] is False and
            profile['completed_epochs'] == training['epochs'], 'Unexpected full-training GPU profile')
    return config, profile


def summarize_validation(output, protocol, architecture, data):
    reports = [read_json(Path(output) / f'validation_{index}' / 'evaluation.json')
               for index in range(1, protocol['evaluation']['repeats'] + 1)]
    first = reports[0]
    for report in reports:
        require(report['architecture'] == architecture and report['split'] == 'validate' and
                report['smoke_only'] is False and report['processed_samples'] == data['splits']['validation']['samples'],
                'Validation is incomplete or has unexpected scope')
        require(report['split_patient_ids'] == data['splits']['validation']['patients'], 'Validation patient IDs changed')
        require(report['checkpoint_sha256'] == first['checkpoint_sha256'] and
                report['checkpoint_epoch'] == first['checkpoint_epoch'], 'Validation checkpoints differ between repeats')
        require(report['pooled']['confusion'] == first['pooled']['confusion'], 'Predictions changed between evaluation repeats')
        milliseconds = report['median_model_forward_ms']
        require(type(milliseconds) in (int, float) and math.isfinite(milliseconds) and milliseconds >= 0,
                'Invalid validation timing')
    timings = [report['median_model_forward_ms'] for report in reports]
    return {'best_epoch': first['checkpoint_epoch'], 'checkpoint_sha256': first['checkpoint_sha256'],
            'pooled': first['pooled'], 'patients': first['patients'],
            'validation_model_forward_median_ms_trials': timings,
            'median_of_trial_medians_ms': statistics.median(timings),
            'timing_scope': first['latency_scope'],
            'case_times': [{'trial': index, 'cases': report['cases']} for index, report in enumerate(reports, start=1)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture', choices=ARCHITECTURES, required=True)
    parser.add_argument('--source-commit', required=True, help='Full commit SHA used for the code snapshot')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without accessing data or creating outputs')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        parser.error('source-commit must be a full 40-character lowercase Git SHA')
    source = Path(__file__).resolve().parent
    protocol = validate_protocol(read_json(source / 'comparison_protocol.json'))
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Experiment output already exists: {output}')
    hashes = source_hashes(source)
    plan = build_plan(protocol, args.architecture, output, source, sys.executable)
    if args.dry_run:
        print(json.dumps({'protocol_id': protocol['protocol_id'], 'architecture': args.architecture,
                          'source_commit': args.source_commit, 'commands': plan}, indent=2))
        return plan
    data = audit_data(protocol)
    check_sources(source, hashes)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'data_manifest.json', data)
    provenance = {'status': 'prepared', 'protocol_id': protocol['protocol_id'], 'source_commit': args.source_commit,
                  'architecture': args.architecture, 'source_directory': str(source), 'source_sha256': hashes,
                  'filename_size_manifest_sha256': data['filename_size_manifest_sha256'],
                  'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'commands': plan}
    write_json(output / 'experiment_provenance.json', provenance)
    started = time.perf_counter()
    try:
        training_profile = None
        for stage, command in plan:
            check_sources(source, hashes)
            provenance['status'] = 'running_' + stage
            write_json(output / 'experiment_provenance.json', provenance)
            print('Experiment stage:', stage, flush=True)
            print('Command:', json.dumps(command), flush=True)
            subprocess.run(command, cwd=source, check=True)
            check_sources(source, hashes)
            if stage == 'training':
                _, training_profile = check_training(output, protocol, args.architecture, data)
        validation = summarize_validation(output, protocol, args.architecture, data)
        summary = {'status': 'completed', 'protocol_id': protocol['protocol_id'],
                   'source_commit': args.source_commit, 'architecture': args.architecture,
                   'completed_epochs': protocol['training']['epochs'], 'training_gpu_profile': training_profile,
                   'experiment_wall_seconds': time.perf_counter() - started,
                   'limitations': protocol['limitations'], **validation}
        write_json(output / 'comparison_result.json', summary)
        provenance['status'] = 'completed'
        write_json(output / 'experiment_provenance.json', provenance)
    except Exception as error:
        provenance['status'] = 'failed'
        provenance['error'] = f'{type(error).__name__}: {error}'
        write_json(output / 'experiment_provenance.json', provenance)
        raise
    print('Experiment output:', output, flush=True)
    print('Full training and validation comparison task: PASS', flush=True)
    return summary


if __name__ == '__main__':
    main()
