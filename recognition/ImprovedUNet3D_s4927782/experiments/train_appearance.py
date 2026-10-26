"""A bounded TRAIN/VAL appearance ablation; frozen v1 sources remain unchanged.

Train two original 3D architectures with independently keyed image augmentation.
The protocol fixes all training settings and the original foreground selector.
No TEST pixels, checkpoint search on stress results, or saved predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from tools.cache_identity import (arrays_identity, load_verified_book, read_arrays,
                                 safe_relative, validate_arrays)
from utils import file_sha256, write_json


def verified_plan(path):
    path = Path(path).resolve(strict=True)
    plan = json.loads(path.read_text())
    if (plan['experiment_id'] != 'appearance_augmentation_v2'
            or plan['allowed_splits'] != ['train', 'validation']
            or plan['training_epochs_per_model'] != 30 or plan['training_seed'] != 42):
        raise ValueError('Unexpected appearance experiment protocol.')
    for relative, expected in plan['required_file_sha256'].items():
        if file_sha256(safe_relative(PROJECT, relative)) != expected:
            raise ValueError('Frozen appearance source/input changed: ' + relative)
    return plan, file_sha256(path)


def verify_development_cache(manifest_path, identity_path, manifest):
    book = load_verified_book(identity_path, manifest_path, manifest)
    records = [r for r in manifest['records'] if r['split'] in ('train', 'validation')]
    counts = {s: sum(r['split'] == s for r in records) for s in ('train', 'validation')}
    if counts != {'train': 143, 'validation': 38}:
        raise ValueError('Expected all 143 TRAIN and 38 VAL caches.')
    checked = []
    for record in records:
        arrays = read_arrays(safe_relative(manifest_path.parent, record['cache_path']))
        validate_arrays(arrays, record, manifest)
        if arrays_identity(arrays) != book['records'][record['cache_path']]['arrays']:
            raise ValueError('Frozen decoded cache identity changed.')
        checked.append({k: record[k] for k in ('patient', 'visit', 'split')})
        checked[-1]['decoded_identity_verified'] = True
    return {'status': 'verified', 'manifest_sha256': file_sha256(manifest_path),
            'counts': counts, 'test_arrays_opened': 0, 'records': checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--variant', choices=('gamma_only', 'gamma_polarity'), required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke-output', type=Path,
                        help='Development only: five TRAIN batches and two VAL cases, one epoch, fresh directory.')
    parser.add_argument('--resume', action='store_true', help='Resume only an identical interrupted arm.')
    args = parser.parse_args(argv)
    plan, protocol_sha = verified_plan(args.protocol)
    import torch
    from torch.utils.data import DataLoader
    from dataset import HipMRIVolumeDataset, load_manifest
    from experiments.appearance import AppearanceTrainingDataset
    from modules import ImprovedUNet3D, segmentation_loss
    from tools.diagnose_model_mechanism import state_sha256
    from train import (optimizer_step, resolve_training_class_weights, restore_rng,
                       rng_state, save_checkpoint, seed_everything, seed_worker)
    from utils import evaluate_volumes, plot_training, synchronize

    manifest_path = safe_relative(PROJECT, plan['manifest'])
    identity_path = safe_relative(PROJECT, plan['cache_identity'])
    manifest = load_manifest(manifest_path)
    if file_sha256(manifest_path) != plan['manifest_sha256']:
        raise ValueError('Manifest SHA differs from appearance protocol.')
    smoke = args.smoke_output is not None
    output = (args.smoke_output.expanduser().absolute() if smoke else
              safe_relative(PROJECT, str(Path(plan['models'][args.variant]['checkpoint']).parent)))
    protected_dirs = [manifest_path.parent,
                      *[safe_relative(PROJECT, m['checkpoint']).parent
                        for m in plan['models'].values() if m['kind'] == 'frozen_v1']]
    if any(output.resolve().is_relative_to(p) for p in protected_dirs):
        raise ValueError('Output would enter a frozen cache or reference directory.')
    if smoke and args.resume:
        raise ValueError('Smoke runs cannot resume or become formal results.')
    if output.exists() and not args.resume:
        raise ValueError('Choose a fresh experiment output or explicitly resume the same arm.')
    if args.resume and not (output / 'latest.pt').is_file():
        raise ValueError('Resume requires this arm\'s own latest.pt.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable.')
    if not smoke and device.type != 'cuda':
        raise ValueError('The formal paired experiment requires one shared CUDA device.')
    cache_evidence = verify_development_cache(manifest_path, identity_path, manifest)
    output.mkdir(parents=True, exist_ok=True)
    cache_report = output / 'cache_verification.json'
    if not args.resume:
        write_json(cache_report, cache_evidence)
    elif json.loads(cache_report.read_text()) != cache_evidence:
        raise ValueError('Resume cache identities differ.')
    config = dict(plan['training_conditions'])
    if config['seed'] != 42 or config['model'] != 'improved3d':
        raise ValueError('Only the declared original 3D/seed42 conditions are allowed.')
    if smoke:
        config.update(max_batches=5, development_only=True, amp=device.type == 'cuda')
    experiment = {'experiment_id': plan['experiment_id'], 'arm_tag': args.variant,
                  'protocol_sha256': protocol_sha}
    seed_everything(config['seed'])
    training = AppearanceTrainingDataset(manifest_path, args.variant, config['seed'])
    validation = HipMRIVolumeDataset(manifest_path, 'validation', augment=False)
    class_weights, class_counts, weights, _ = resolve_training_class_weights(
        manifest_path, manifest, config['class_weighting'], device)
    if class_counts != config['training_class_voxel_counts'] or weights != config['resolved_class_weights']:
        raise ValueError('TRAIN-only loss weights differ from the original conditions.')
    model = ImprovedUNet3D(6, config['base_channels']).to(device)
    initial_state = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
    start_epoch, best_score, best_epoch, history = 1, None, None, []
    if args.resume:
        checkpoint = torch.load(output / 'latest.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['experiment'] != experiment:
            raise ValueError('Resume configuration/protocol/augmentation differs.')
        old_run = json.loads((output / 'run.json').read_text())
        if old_run['status'] == 'completed':
            raise ValueError('Completed arms are not retrained.')
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer']); scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng_state'])
        history = checkpoint['history']; best_score = checkpoint['best_score']; best_epoch = checkpoint['best_epoch']
        start_epoch = checkpoint['epoch'] + 1
        # An interruption can fall between writing latest.pt and best.pt/history.
        if checkpoint['epoch'] == best_epoch:
            save_checkpoint(output / 'best.pt', checkpoint)
        else:
            best = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            if (best['epoch'] != best_epoch or best['config'] != config
                    or best['experiment'] != experiment):
                raise ValueError('Resume best.pt does not agree with latest best epoch/configuration.')
        write_json(output / 'history.json', history)
    run = {'status': 'running', 'experiment': experiment, 'config': config,
           'requested_total_epochs': 1 if smoke else 30, 'development_only': smoke,
           'initial_parameter_state_sha256': initial_state,
           'selection_metric': 'Original whole-volume patient-balanced foreground Dice; ties retain earlier epoch.',
           'accuracy_evidence': 'Development smoke only' if smoke else 'TRAIN/VAL only; no TEST arrays opened.',
           'runtime': {'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                       'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda},
           'cache_verification_sha256': file_sha256(cache_report),
           'source_file_sha256': plan['required_file_sha256'],
           'determinism': 'Seeded; unsupported CE/maxpool CUDA kernels remain warn-only. No bitwise causal-isolation claim.'}
    write_json(output / 'run.json', run)
    try:
        for epoch in range(start_epoch, (1 if smoke else 30) + 1):
            training.set_epoch(epoch)
            loader = DataLoader(training, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                                pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(config['seed'] + epoch))
            model.train(); synchronize(device)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter(); loss_sum = 0.; batches = examples = skipped_steps = 0
            for batch_index, batch in enumerate(loader):
                if smoke and batch_index >= 5: break
                images = batch['image'].to(device, non_blocking=True)
                targets = batch['mask'].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config['amp']):
                    outputs = model(images)
                    loss = segmentation_loss(outputs, targets, class_weights=class_weights)
                if not bool(torch.isfinite(loss).item()): raise RuntimeError('Nonfinite appearance training loss.')
                scaler.scale(loss).backward()
                gradient_norm, skipped = optimizer_step(model, optimizer, scaler, config['clip_grad'], config['amp'])
                skipped_steps += int(skipped); batches += 1; examples += int(images.shape[0])
                loss_sum += float(loss.item()) * int(images.shape[0])
            if smoke and (batches != 5 or skipped_steps == batches):
                raise RuntimeError('Smoke needs five batches and at least one successful optimizer step.')
            synchronize(device)
            train_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, images, targets
            evaluation = evaluate_volumes(model, validation, 'improved3d', 6, device,
                                          config['eval_batch_size'], config['amp'], max_cases=2 if smoke else None)
            score = evaluation['aggregate']['foreground_mean_dice']
            if score is None: raise RuntimeError('Undefined validation foreground score.')
            improved = best_score is None or score > best_score
            if improved: best_score, best_epoch = score, epoch
            history.append({'epoch': epoch, 'train_loss': loss_sum / examples,
                            'training_batches': batches, 'training_examples': examples, 'train_seconds': train_seconds,
                            'peak_allocated_cuda_bytes_training': peak,
                            'last_gradient_norm_before_clipping': gradient_norm,
                            'amp_overflow_skipped_optimizer_steps': skipped_steps,
                            'optimizer_steps': batches - skipped_steps, 'validation_foreground_dice': score,
                            'validation_foreground_iou': evaluation['aggregate']['foreground_mean_iou'],
                            'validation': evaluation, 'development_only': smoke})
            payload = {'schema_version': 2, 'experiment': experiment, 'config': config,
                       'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                       'epoch': epoch, 'best_score': best_score, 'best_epoch': best_epoch,
                       'history': history, 'rng_state': rng_state()}
            save_checkpoint(output / 'latest.pt', payload)
            if improved: save_checkpoint(output / 'best.pt', payload)
            write_json(output / 'history.json', history)
            try: plot_training(history, output / 'curves.png')
            finally: restore_rng(payload['rng_state'])
            run.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_foreground_dice=best_score)
            write_json(output / 'run.json', run)
            print(f'{args.variant} epoch {epoch}/{1 if smoke else 30}: loss={history[-1]["train_loss"]:.5f}; VAL FG={score:.5f}', flush=True)
        run['status'] = 'completed_development_smoke' if smoke else 'completed'
    except BaseException as error:
        run.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   error=f'{type(error).__name__}: {error}')
        write_json(output / 'run.json', run)
        raise
    verified_plan(args.protocol)
    selected = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    if (selected['epoch'] != best_epoch or selected['config'] != config
            or selected['experiment'] != experiment or len(history) != (1 if smoke else 30)
            or [row['epoch'] for row in history] != list(range(1, (1 if smoke else 30) + 1))):
        raise RuntimeError('Selected checkpoint/history does not match the completed experiment.')
    write_json(output / 'run.json', run)
    if not smoke:
        write_json(output / 'provenance.json', {'experiment': experiment,
                   'checkpoint_sha256': file_sha256(output / 'best.pt'), 'run_sha256': file_sha256(output / 'run.json'),
                   'history_sha256': file_sha256(output / 'history.json'), 'protocol_sha256': protocol_sha,
                   'manifest_sha256': file_sha256(manifest_path), 'source_file_sha256': plan['required_file_sha256'],
                   'cache_verification_sha256': file_sha256(cache_report),
                   'cache_verification_path': str(cache_report.relative_to(PROJECT))})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
