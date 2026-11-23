"""Small study bindings, immutable receipts and existing budget/lock adapters."""
from __future__ import annotations
import ast
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import torch
import dataset
import train_convnext_v2 as v2
from g012 import core

BASE = Path(__file__).resolve().parents[1]
PLAN = BASE/'g012'/'G012_PLAN.json'
CAP = 10800.
TOTAL_CAP = CAP*3
DISK_GIB = 12
require = core.require


def read(path):
    # Windows CRT readers normally deny delete sharing, which can obstruct
    # os.replace of a live ledger. A short shared-delete read sees a complete
    # old or new atomic file without blocking the existing budget writer.
    if os.name == 'nt':
        import ctypes
        import msvcrt
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
            ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        handle = kernel.CreateFileW(str(Path(path)),0x80000000,7,None,3,0,None)
        if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
        fd = msvcrt.open_osfhandle(handle,os.O_RDONLY | os.O_BINARY)
        with os.fdopen(fd,'rb') as f: raw = f.read()
        return json.loads(raw.decode('utf-8-sig'))
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@contextmanager
def budget_io(folder):
    """Serialize live ledger snapshots/replaces using the existing OS lock.

    This is distinct from the study lock held for the full child invocation.
    Shared-delete alone is insufficient on every Windows filesystem/filter.
    """
    directory = Path(folder)/'budget-io'; directory.mkdir(exist_ok=True)
    for attempt in range(300):
        lock = v2.run_lock(directory)
        try:
            lock.__enter__(); break
        except RuntimeError as exc:
            if 'locked by another process' not in str(exc): raise
            time.sleep(.01)
    else: raise RuntimeError('Budget I/O lock unavailable; preserve evidence and investigate.')
    try: yield
    finally: lock.__exit__(None,None,None)


def read_budget(path):
    with budget_io(Path(path).parent): return read(path)


def sha(path):
    return v2.sha256_file(Path(path))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, 'Existing evidence differs; preserve it and investigate.')
    else:
        v2.atomic_json(path, value)


def dependency_files():
    """Local import closure, not handbacks, changing report prose or pointer trees."""
    pending = [BASE/'g012'/n for n in ('run.py','storage.py','core.py','engine.py','results.py','__init__.py')]
    found = set()
    while pending:
        path = pending.pop()
        if path in found: continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = []
            if isinstance(node, ast.Import): modules = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module: modules = [node.module]
            for module in modules:
                candidate = BASE/Path(*module.split('.')).with_suffix('.py')
                if candidate.is_file() and candidate not in found: pending.append(candidate)
    return sorted(path.relative_to(BASE).as_posix() for path in found)


def source_hashes():
    return {name: sha(BASE/name) for name in dependency_files()}


def implementation(committed=True):
    plan = read(PLAN)
    require(plan['sources'] == source_hashes(), 'Runtime/source identity changed; preserve existing attempts.')
    approved = read(BASE/'APPROVED_PARTITIONS.json')
    require(approved['manifest_sha256'] == plan['manifest_sha256'] and sha(BASE/'APPROVED_PARTITIONS.json') == plan['approved_protocol_sha256'],
            'Approved assignment protocol changed.')
    require(plan['configurations'] == {v: core.config(v) for v in core.VARIANTS}, 'Configuration/aggregation plan changed.')
    require(plan['caps'] == {'per_variant_active_seconds': CAP, 'total_active_seconds': TOTAL_CAP,
                           'free_disk_gib': DISK_GIB, 'lease_seconds': 5.}, 'Resource protocol differs.')
    if committed:
        repository = BASE.parents[1]
        names = [f'recognition/ConvNeXt_Alzheimers/{n}' for n in (*plan['sources'], 'g012/G012_PLAN.json', 'APPROVED_PARTITIONS.json')]
        subprocess.run(['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', *names], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(['git', '-C', str(repository), 'diff', '--exit-code', 'HEAD', '--', *names], check=True, stdout=subprocess.DEVNULL)
        raw = subprocess.check_output(['git','-C',str(repository),'show','HEAD:recognition/ConvNeXt_Alzheimers/g012/G012_PLAN.json'])
        require(hashlib.sha256(raw).hexdigest() == sha(PLAN), 'Commit the exact plan before real-data commands.')
    return plan


def output_guard(root, data_root, splits_dir):
    root = Path(root).resolve()
    for protected in (BASE, Path(data_root).resolve(), Path(splits_dir).resolve()):
        require(root != protected and protected not in root.parents, 'Study output is inside protected source/data/manifests.')
    require(not any((p/'.git').exists() for p in (root, *root.parents)), 'Study evidence must remain outside Git.')
    return root


def check_disk(root):
    import shutil
    require(shutil.disk_usage(root).free >= DISK_GIB*2**30, 'DISK_BLOCKER: at least 12 GiB free required; never delete evidence automatically.')


def context(args, committed=True):
    plan = implementation(committed)
    root = output_guard(args.study_root, args.data_root, args.splits_dir)
    binding = read(root/'PRIVATE_study_binding.json')
    require(binding['plan_sha256'] == core.digest(plan) and binding['data_root'] == str(args.data_root.resolve())
            and binding['splits_dir'] == str(args.splits_dir.resolve()), 'Study inputs/identity changed.')
    for name, entry in binding['a1_inputs'].items():
        require(sha(entry['path']) == entry['sha256'], 'An explicitly bound A1 reference changed.')
    bundle = dataset.load_finalized_splits(args.data_root, args.splits_dir)
    groups = core.checked_groups(bundle, read(BASE/'APPROVED_PARTITIONS.json'))
    return plan, root, binding, bundle, groups


def a1_reference(paths, bundle, plan):
    from training_utils import aggregate_predictions
    report = read(paths['summary']); predictions = read(paths['predictions'])
    require(report['configuration'] == plan['a1_configuration'] and report['experiment_completed'] is True
            and report['selected_epoch'] == 13 and report['manifest_sha256'] == bundle.manifest_hashes
            and report['pretrained'] == {'used': False}, 'A1 reference is not the retained seed-42 control.')
    require(sha(paths['checkpoint']) == plan['a1_checkpoint_sha256'], 'Retained A1 checkpoint digest differs.')
    for name, value in report['source_code_sha256'].items():
        require(sha(BASE/'historical_runtime'/name) == value, 'Historical A1 runtime differs.')
    by_path = {r['image_path']: r for r in predictions['slices']}
    records = bundle.records['validation']
    require(len(by_path) == len(records) == len(predictions['slices']) and set(by_path) == {r.image_path for r in records},
            'A1 cached validation coverage differs.')
    require(all(by_path[r.image_path]['label'] == r.label and by_path[r.image_path]['subject_id'] == r.subject_id
                and by_path[r.image_path]['scan_id'] == r.scan_id for r in records), 'A1 cached membership differs.')
    metrics, reconstructed = aggregate_predictions(records, [by_path[r.image_path]['p_ad'] for r in records])
    require(metrics == report['selected_validation_metrics'] == plan['a1_metrics'], 'Exact A1 control metrics differ.')
    core.replay_difference(predictions, reconstructed, tolerance=0.)
    # Brier was not in the original metrics dictionary; recompute from cached
    # validation probabilities without inference, fitting or test access.
    for level, key in [('slice','slices'),('scan','scans'),('patient','patients')]:
        metrics[level]['brier'] = core.scored([r['label'] for r in reconstructed[key]], [r['p_ad'] for r in reconstructed[key]])['brier']
    with Path(paths['history']).open(encoding='utf-8-sig', newline='') as f:
        history = [{k: float(v) if k != 'selected_as_best' else v for k, v in r.items()} for r in csv.DictReader(f)]
    require(len(history) == report['epochs_completed'], 'A1 history length differs.')
    return {'metrics': metrics, 'history': history, 'selected_epoch': 13, 'completed_epoch': report['epochs_completed'],
            'parameters': report['parameters'], 'recorded_resources': {
                k: report[k] for k in ('training_loop_seconds_committed_epochs','gpu_memory') if k in report},
            'predictions': reconstructed}


def prepare(args):
    plan = implementation(); root = output_guard(args.study_root, args.data_root, args.splits_dir)
    bundle = dataset.load_finalized_splits(args.data_root, args.splits_dir)
    groups = core.checked_groups(bundle, read(BASE/'APPROVED_PARTITIONS.json'))
    paths = {name: getattr(args, f'a1_{name}').resolve() for name in
             ('summary','predictions','history','checkpoint','replay_receipt','replayed_predictions')}
    reference = a1_reference(paths, bundle, plan)
    expected = {'plan_sha256': core.digest(plan), 'data_root': str(args.data_root.resolve()),
                'splits_dir': str(args.splits_dir.resolve()),
                'a1_inputs': {name: {'path': str(path), 'sha256': sha(path)} for name, path in paths.items()}}
    if root.exists():
        require((root/'PRIVATE_study_binding.json').is_file() and read(root/'PRIVATE_study_binding.json') == expected,
                'Existing study destination conflicts or lacks its binding; preserve it, do not replace the attempt.')
        print('SAME_STUDY_PREPARATION_VERIFIED_OR_FINISHING')
    else:
        parent = root.parent
        while not parent.exists(): parent = parent.parent
        check_disk(parent); root.mkdir(parents=True)
    check_disk(root)
    immutable_json(root/'PRIVATE_study_binding.json', expected)
    immutable_json(root/'frozen_plan.json', plan)
    immutable_json(root/'PRIVATE_a1_reference.json', reference)
    # Initialization is constructed without data and stays private. No trained
    # checkpoint initializes any variant.
    with core.isolated_rng():
        v2.seed_everything(42); model = v2.make_model(core.base_config('cpu'))
        initial = core.tensor_digest(model.state_dict()); del model
    schedules = [core.schedule(groups['train'], epoch) for epoch in range(1,31)]
    private_schedules = [{k:v for k,v in s.items() if k not in ('order_rng_state','angle_rng_state')} for s in schedules]
    immutable_json(root/'PRIVATE_matched_schedule.json', {'initial_tensor_sha256': initial, 'epochs': private_schedules})
    immutable_json(root/'preparation.json', {'plan_sha256': core.digest(plan), 'manifest_sha256': bundle.manifest_hashes,
        'train_scans':len(groups['train']), 'validation_scans':len(groups['validation']),
        'images_decoded':0, 'test_manifest_structural_checks_only':True, 'initial_tensor_sha256': initial})
    print('G012_PREPARED_NO_IMAGES_DECODED')


def historical(args):
    """Reuse a specifically bound full-validation runtime receipt, never MRI inference."""
    plan, root, binding, bundle, groups = context(args)
    paths = {name: entry['path'] for name, entry in binding['a1_inputs'].items()}
    receipt = read(paths['replay_receipt'])
    require(receipt['scope'] == 'validation' and receipt['checkpoint_sha256'] == plan['a1_checkpoint_sha256']
            and receipt['manifest_sha256'] == bundle.manifest_hashes and receipt['selected_epoch'] == 13
            and receipt['architecture'] == 'ConvNeXtTiny' and receipt['test_images_decoded'] == 0
            and receipt['calibration_fitted'] is False and receipt['threshold_search'] is False
            and receipt['preprocessing'] == dataset.preprocessing_spec(0.), 'Historical validation receipt binding differs.')
    require(receipt['verified_source_sha256'] == plan['a1_source_sha256'], 'Historical replay runtime differs.')
    for name, expected in receipt['verified_source_sha256'].items():
        require(sha(BASE/'historical_runtime'/name) == expected, 'Frozen historical bytes changed.')
    expected = read(paths['predictions']); actual = read(paths['replayed_predictions'])
    checks = core.replay_difference(expected, actual)
    immutable_json(root/'a1_historical_regression.json', {'passed': True,
        'method':'REUSED_BOUND_EXISTING_FULL_VALIDATION_RUNTIME_RECEIPT_AND_RECHECKED_CACHED_OUTPUTS',
        'new_inference':False, 'checkpoint_sha256':plan['a1_checkpoint_sha256'],
        'plan_sha256':core.digest(plan), 'absolute_probability_tolerance':2e-5, 'checks':checks})
    print('A1_HISTORICAL_REGRESSION_PASSED_REUSED_RECEIPT_NO_NEW_INFERENCE')


def history_csv(rows):
    if not rows: return ''
    out = io.StringIO(newline=''); writer = csv.DictWriter(out, list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows); return out.getvalue()


def write_text_atomic(path, text):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def recover_state(folder, expected_identity, model, optimizer, environment):
    bad = []
    for path in (folder/'PRIVATE_resume.pt', folder/'PRIVATE_previous.pt'):
        if not path.exists(): continue
        try: state = torch.load(path, map_location='cpu', weights_only=True)
        except Exception: bad.append(path); continue
        require(state['identity'] == expected_identity and state['environment'] == environment,
                'Recovery configuration/aggregation/source/manifest/environment differs.')
        core.strict_state(model, state['model_state']); optimizer.load_state_dict(state['optimizer_state'])
        v2.restore_rng(state['rng'])
        if path.name == 'PRIVATE_previous.pt' and (folder/'PRIVATE_resume.pt').exists():
            os.replace(folder/'PRIVATE_resume.pt', folder/('PRIVATE_unreadable-'+uuid.uuid4().hex+'.pt'))
        return state
    raise RuntimeError('No usable complete-epoch checkpoint; do not start fresh or reset the budget.')


def verify_completion(folder, identity):
    value = read(folder/'completion.json')
    require(value['identity'] == identity, 'Completed run identity differs.')
    for name, expected in value['hashes'].items():
        require(sha(folder/name) == expected, 'Completed evidence changed.')
    return value
