#!/usr/bin/env bash
# No Slurm, network provisioning or TEST evaluation. Run only after the cloud
# amendment has been completed from measured hardware/software and frozen.
# Example:
# DEMO3_PYTHON=/workspace/venv/bin/python bash jobs/train_pair_seed43_cloud.sh \
#   --amendment cloud_amendment.json --platform 'Virtual GPU Lab'
set -euo pipefail
DEMO3_SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
exec "${DEMO3_PYTHON:-python3}" - "$DEMO3_SCRIPT_PATH" "$@" <<'PY'
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import platform


SCRIPT = Path(sys.argv[1]).resolve()
PROJECT = SCRIPT.parents[1]
os.chdir(PROJECT)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Expected a safe project-relative path: {value!r}')
    return PROJECT / path


def output_root(value, prefix):
    logical = Path(value)
    if (len(logical.parts) != 2 or logical.parts[0] != 'artifacts'
            or not (logical.parts[1] == prefix or logical.parts[1].startswith(prefix + '_'))):
        raise ValueError(f'Cloud output must be artifacts/{prefix} or a suffixed independent cloud root.')
    actual = relative_path(value).resolve()
    if not actual.is_relative_to(PROJECT):
        raise ValueError('Cloud output symlink escapes the project.')
    protected = [PROJECT / 'artifacts' / name for name in (
        'full_seed42', 'validation_seed42', 'followup_seed43', 'validation_seed43', 'test_final')]
    for path in protected:
        path = path.resolve()
        if actual == path or actual.is_relative_to(path) or path.is_relative_to(actual):
            raise ValueError('Cloud output overlaps protected Rangpur/v1 evidence.')
    for model in ('unet2d', 'improved3d'):
        child = (actual / model).resolve()
        if child.parent != actual:
            raise ValueError('A model directory symlink escapes its cloud output root.')
    return actual


def check_required(amendment, plan, plan_path):
    required = amendment.get('required_file_sha256')
    if not isinstance(required, dict) or not required:
        raise ValueError('Cloud amendment must freeze required_file_sha256.')
    mandatory = dict(plan['required_file_sha256'])
    mandatory['followup_protocol.json'] = digest(plan_path)
    mandatory[str(SCRIPT.relative_to(PROJECT))] = digest(SCRIPT)
    if any(required.get(name) != expected for name, expected in mandatory.items()):
        raise ValueError('Cloud amendment omits/changes a frozen source, base protocol or this runner.')
    for name, expected in required.items():
        path = relative_path(name)
        if (str(Path(name)).startswith('artifacts/test')
                or path.suffix not in ('.json', '.py', '.sh', '.slurm', '.txt')):
            raise ValueError('Only code/protocol/manifest metadata may be hashed; TEST artifacts/arrays are forbidden.')
        if digest(path) != expected:
            raise ValueError(f'Frozen cloud input identity differs: {name}')
    return required


def check_config(config, model, plan):
    expected = dict(plan['training_conditions'][model], seed=43)
    if config != expected or config.get('development_only') or config.get('max_batches') is not None:
        raise ValueError(f'{model}: configuration differs from the original TRAIN/VAL plan.')


def child(command, environment):
    print('Running:', ' '.join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=PROJECT, env=environment, check=True)


parser = argparse.ArgumentParser(description='Paired seed43 cloud TRAIN/VAL runner; no Slurm or TEST access.')
parser.add_argument('--amendment', type=Path, default=Path('cloud_amendment.json'))
parser.add_argument('--platform', default=os.environ.get('DEMO3_CLOUD_PLATFORM', ''),
                    help='Explicit operator declaration; must match the authorized provider.')
parser.add_argument('--preflight-only', action='store_true',
                    help='Validate frozen inputs, authorized runtime/GPU and paths, print JSON, then exit without creating a run/lock or training.')
args = parser.parse_args(sys.argv[2:])
amendment_path = args.amendment.expanduser().resolve()
amendment = read(amendment_path)
plan_path = PROJECT / 'followup_protocol.json'
plan = read(plan_path)
if (amendment.get('amendment_id') != 'paired_seed43_cloud_validation'
        or amendment.get('authorized') is not True
        or amendment.get('base_protocol_sha256') != digest(plan_path)
        or amendment.get('training_seed') != 43 or amendment.get('training_epochs_per_model') != 30
        or set(amendment.get('allowed_splits', [])) != {'train', 'validation'}):
    raise ValueError('A completed authorized pre-result cloud amendment for the unchanged plan is required.')
if (plan['experiment_id'] != 'paired_seed43_validation' or plan['training_seed'] != 43
        or plan['training_epochs_per_model'] != 30 or plan['training_seeds'] != [42, 43]
        or set(plan['allowed_splits']) != {'train', 'validation'}
        or plan['v1_test_already_observed'] is not True or not plan['diagnostic_gate']['approved_to_proceed']):
    raise ValueError('Original follow-up protocol has unexpected scope or an unapproved diagnostic gate.')
declared_platform = amendment['platform']
if (args.platform != 'Virtual GPU Lab' or declared_platform.get('provider') != args.platform
        or declared_platform.get('region') != 'ap-northeast-1'
        or declared_platform.get('instance_type') != 'g6e.xlarge'):
    raise ValueError('Authorized cloud provider/region/instance declaration differs.')
required = check_required(amendment, plan, plan_path)
manifest_path = PROJECT / 'artifacts/cache64/manifest.json'
sourcebook_path = PROJECT / 'training_sources.json'
if digest(manifest_path) != plan['manifest_sha256'] or digest(sourcebook_path) != plan['training_sources_sha256']:
    raise ValueError('Frozen cache manifest/sourcebook identity differs.')
manifest, sourcebook = read(manifest_path), read(sourcebook_path)
if manifest.get('partial') or manifest.get('development_only') or sourcebook['manifest_sha256'] != plan['manifest_sha256']:
    raise ValueError('Only the full frozen cache manifest is permitted.')
training_root = output_root(amendment['output_roots']['training'], 'followup_seed43_cloud')
validation_root = output_root(amendment['output_roots']['validation'], 'validation_seed43_cloud')
if training_root == validation_root or training_root.is_relative_to(validation_root) or validation_root.is_relative_to(training_root):
    raise ValueError('Training/validation cloud roots must be independent.')

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch
import numpy as np
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise ValueError('Exactly one visible CUDA GPU is required; expose only the allocated cloud GPU.')
gpu_name = torch.cuda.get_device_name(0)
if declared_platform.get('cuda_device_count') != 1 or declared_platform.get('gpu_name') != gpu_name:
    raise ValueError('Measured GPU identity/count differs from the completed amendment.')
measured_environment = {'python_version': platform.python_version(), 'torch_version': str(torch.__version__),
                        'cuda_version': torch.version.cuda, 'numpy_version': np.__version__}
expected_environment = amendment['environment']
for key, measured in measured_environment.items():
    if not isinstance(expected_environment.get(key), str) or expected_environment[key] != measured:
        raise ValueError(f'Cloud runtime differs from the completed amendment: {key}; measured {measured!r}')
if set(expected_environment) - set(measured_environment) - {'extra_versions'}:
    raise ValueError('Unrecognized cloud environment fields; pin extra packages under extra_versions.')
packages = {}
for name in {'matplotlib', 'nibabel', 'scipy'} | set(expected_environment.get('extra_versions', {})):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
if packages['matplotlib'] is None:
    raise ValueError('Matplotlib is required by the frozen training/validation scripts.')
for name, wanted in expected_environment.get('extra_versions', {}).items():
    if not isinstance(wanted, str) or packages[name] != wanted:
        raise ValueError(f'Cloud package version differs from amendment: {name}')
execution = {'schema_version': 1, 'amendment_sha256': digest(amendment_path), 'base_protocol_sha256': digest(plan_path),
             'runner_sha256': digest(SCRIPT), 'manifest_sha256': digest(manifest_path),
             'training_sources_sha256': digest(sourcebook_path), 'platform_operator_declaration': declared_platform,
             'platform_attestation_limit': 'Provider/region/instance are operator declarations, not independently queried cloud metadata.',
             'measured_environment': measured_environment, 'measured_extra_packages': packages,
             'measured_gpu_name': gpu_name, 'visible_cuda_device_count': 1,
             'measured_gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
             'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
             'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'output_roots': amendment['output_roots'], 'required_file_sha256_verified': required,
             'interpretation': 'Post-v1-TEST TRAIN/VAL follow-up jointly changes seed and cloud platform/software; it does not isolate a seed effect or replace v1 TEST.'}
execution_path = training_root / 'cloud_execution.json'
if not execution_path.exists():
    if any((training_root / model).exists() or (validation_root / model).exists() for model in ('unet2d', 'improved3d')):
        raise ValueError('Existing cloud model outputs lack an execution identity; do not adopt/copy Rangpur runs.')
else:
    previous = read(execution_path)
    previous.pop('recorded_at_utc', None)
    if previous != execution:
        raise ValueError('Existing cloud run used a different amendment/runtime identity; use a new amended cloud root.')
if args.preflight_only:
    print(json.dumps({'status': 'preflight_passed', 'preflight_only': True,
                      'formal_run_created': False, 'execution_lock_acquired': False,
                      'train_val_cache_arrays_accessed': False, 'execution_identity': execution},
                     indent=2, allow_nan=False), flush=True)
    raise SystemExit(0)
for record in manifest['records']:
    if record['split'] in ('train', 'validation'):
        cache = manifest_path.parent / record['cache_path']
        if not cache.is_file():
            raise ValueError(f'Missing TRAIN/VAL cache: {record["cache_path"]}')
# Do not open TEST arrays. Cache transfer/decoded-array integrity is a separate
# prerequisite; mere file existence is not claimed to verify array contents.


def check_runtime_record(record):
    for key in ('python_version', 'torch_version', 'cuda_version'):
        if record.get(key) != measured_environment[key]:
            raise ValueError(f'Cloud run/evaluation runtime differs from its amendment: {key}')
    if record.get('gpu_name') != gpu_name:
        raise ValueError('Cloud run/evaluation GPU differs from its amendment.')


training_root.mkdir(parents=True, exist_ok=True)
lock = training_root / '.cloud_runner.lock'
try:
    lock.mkdir()
except FileExistsError:
    raise RuntimeError('Cloud runner lock exists. Confirm no prior process is active before manually removing a stale lock.') from None
try:
    (lock / 'owner.json').write_text(json.dumps({'pid': os.getpid(), 'hostname': socket.gethostname(),
                                              'started_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    if not execution_path.exists():
        with execution_path.open('x') as stream:
            json.dump({**execution, 'recorded_at_utc': datetime.now(timezone.utc).isoformat()}, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        previous = read(execution_path)
        previous.pop('recorded_at_utc', None)
        if previous != execution:
            raise ValueError('Cloud execution identity changed while acquiring the lock.')
    environment = os.environ.copy()
    environment['MPLCONFIGDIR'] = str(training_root / '.matplotlib')
    environment['TMPDIR'] = str(training_root / '.tmp')
    Path(environment['TMPDIR']).mkdir(exist_ok=True)
    for model in ('unet2d', 'improved3d'):
        # Recheck required immutable files before each sequential model.
        check_required(amendment, plan, plan_path)
        output = training_root / model
        latest = output / 'latest.pt'
        resume = []
        needs_training = True
        if latest.exists():
            if latest.resolve().parent != output.resolve():
                raise ValueError('Resume checkpoint must stay in its own cloud model directory.')
            checkpoint = torch.load(latest, map_location='cpu', weights_only=True)
            check_config(checkpoint['config'], model, plan)
            existing_run = read(output / 'run.json')
            check_config(existing_run['config'], model, plan)
            check_runtime_record(existing_run)
            epoch = checkpoint['epoch']
            if type(epoch) is not int or not 1 <= epoch <= 30:
                raise ValueError('Cloud resume checkpoint epoch must be in 1..30.')
            if epoch == 30:
                run = read(output / 'run.json')
                check_config(run['config'], model, plan)
                if run['status'] != 'completed' or run['completed_epochs'] != 30:
                    raise ValueError('Epoch-30 checkpoint lacks matching completed run metadata.')
                needs_training = False
            else:
                resume = ['--resume', str(latest)]
            del checkpoint
        if needs_training:
            child([sys.executable, '-u', 'train.py', '--manifest', manifest_path, '--model', model,
                   '--output-dir', output, '--epochs', '30', '--batch-size', '16' if model == 'unet2d' else '1',
                   '--base-channels', '16', '--lr', '0.001', '--workers', '0', '--seed', '43',
                   '--device', 'cuda', '--amp', '--eval-batch-size', '32', *resume], environment)
        run = read(output / 'run.json')
        check_config(run['config'], model, plan)
        check_runtime_record(run)
        if run['status'] != 'completed' or run['completed_epochs'] != 30 or run['requested_total_epochs'] != 30:
            raise ValueError('Cloud training did not complete the fixed 30-epoch budget.')
        child([sys.executable, 'tools/record_provenance.py', '--expected', sourcebook_path,
               '--manifest', manifest_path, '--output', output / 'source_provenance.json',
               '--checkpoint', output / 'best.pt', '--replace'], environment)
        evaluation_path = validation_root / model / 'evaluation.json'
        if not evaluation_path.exists():
            child([sys.executable, '-u', 'predict.py', '--manifest', manifest_path, '--checkpoint', output / 'best.pt',
                   '--split', 'validation', '--output-dir', validation_root / model,
                   '--batch-size', '32', '--device', 'cuda', '--target-class', '5'], environment)
        evaluation = read(evaluation_path)
        check_config(evaluation['model_config'], model, plan)
        check_runtime_record(evaluation)
        expected_keys = {(row['patient'], row['visit']) for row in manifest['records'] if row['split'] == 'validation'}
        actual_keys = [(row['patient'], row['visit']) for row in evaluation['cases']]
        if (len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys
                or evaluation['aggregate']['num_cases'] != len(expected_keys)
                or evaluation['aggregate']['num_patients'] != len({key[0] for key in expected_keys})):
            raise ValueError('Cloud standalone validation must cover every frozen validation visit exactly once.')
        if (evaluation['split'] != 'validation' or evaluation['final_test_explicitly_requested'] or evaluation['development_only']
                or evaluation['amp'] is not True or not str(evaluation['device']).startswith('cuda')
                or evaluation['manifest_sha256'] != plan['manifest_sha256']
                or evaluation['checkpoint_sha256'] != digest(output / 'best.pt')
                or evaluation['checkpoint_epoch'] != run['best_epoch']
                or evaluation['checkpoint_best_epoch'] != run['best_epoch']):
            raise ValueError('Cloud standalone validation identity differs from its selected checkpoint/plan.')
        print(f'{model}: fixed seed43 cloud training and standalone validation identities verified.', flush=True)
    print('Cloud TRAIN/VAL follow-up completed. No TEST evaluation or original evidence was modified.', flush=True)
finally:
    (lock / 'owner.json').unlink(missing_ok=True)
    lock.rmdir()
PY
