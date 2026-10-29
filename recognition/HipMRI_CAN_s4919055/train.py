"""Readable course-style Adam training for 2D U-Net and whole-volume CAN3D.

debug: repeatedly fit a few slices from one training volume; never open
validation/test images. fit: all training cases, validation selects best.pt.
test: explicitly evaluate a fixed fit checkpoint on the held-out test split.
CAN3D uses one complete volume per update; 2D uses slice mini-batches.
"""

import argparse
import math
import os
from pathlib import Path
import platform
import socket
import time

import nibabel
import numpy as np
import torch

from dataset import (DEFAULT_DATA_ROOT, PILOT_SHAPE_DHW, HipMRIVolumeDataset,
                     read_patient_manifest, volume_to_slices)
from modules import CAN3D, UNet2D, segmentation_loss
from experiment import (seed_everything, get_device, file_sha256, save_json,
                        save_checkpoint, load_checkpoint, hard_overlap,
                        evaluate_cases, plot_history)


def optimize_batch(model, optimizer, criterion, image, target):
    """The lecture sequence: train, forward, loss, clear, backward, step."""
    model.train()
    logits = model(image)
    loss = criterion(logits, target)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite loss; stop and inspect the data.')
    optimizer.zero_grad()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Missing or nonfinite gradient.')
    optimizer.step()
    return loss.item()


def training_slice_order(mask, generator, sampling='all'):
    """Training-only slice indices; retain the original number of optimizer steps.

    'all' visits every slice once. 'prostate-balanced' draws half the slice
    budget from training slices containing label 5, with replacement when
    necessary, and half from the remaining slices. Interleaving the groups
    gives equal positive/negative counts in even-sized batches. If either
    group is absent, use every slice once rather than discard the case.
    This is an experimental batch-composition change, not a new network/loss.
    """
    if mask.ndim != 3 or mask.shape[0] < 1:
        raise ValueError('Need a nonempty D,H,W training mask.')
    if sampling not in ('all', 'prostate-balanced'):
        raise ValueError('Unknown slice sampling strategy.')
    depth = len(mask)
    if sampling == 'all':
        return torch.randperm(depth, generator=generator)
    has_prostate = (mask == 5).flatten(1).any(1)
    positive = torch.where(has_prostate)[0]
    negative = torch.where(~has_prostate)[0]
    if not len(positive) or not len(negative):
        return torch.randperm(depth, generator=generator)

    def draw(pool, count):
        if count <= len(pool):
            return pool[torch.randperm(len(pool), generator=generator)[:count]]
        return pool[torch.randint(len(pool), (count,), generator=generator)]

    order = torch.empty(depth, dtype=torch.long)
    order[0::2] = draw(positive, (depth+1)//2)
    order[1::2] = draw(negative, depth//2)
    return order


def run_debug(args, dataset, device, model, config):
    """Train on fixed training-only slices; report it explicitly as debugging."""
    sample = dataset[0]
    images, targets = volume_to_slices(sample['image'], sample['target'])
    area = (sample['mask'] == 5).sum(dim=(1, 2))
    positive = torch.where(area > 0)[0]
    if len(positive) == 0:
        raise ValueError('First training volume has no prostate after resizing.')
    chosen = positive[torch.argsort(area[positive], descending=True)[:3]].tolist()
    background = torch.where(area == 0)[0]
    if len(background):
        chosen.append(int(background[0]))
    chosen = sorted(chosen)
    image, target = images[chosen].to(device), targets[chosen].to(device)
    truth = target.argmax(1).cpu()
    criterion = segmentation_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = []

    def record(step):
        model.eval()
        with torch.no_grad():
            logits = model(image)
            loss = criterion(logits, target).item()
            dice = hard_overlap(logits.argmax(1).cpu(), truth)['dice'][5]
        history.append({'step': step, 'train_loss': loss, 'prostate_dice': dice})
        print(f'Debug step {step}: selected-training loss={loss:.6f}, prostate Dice={dice}', flush=True)

    record(0)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.debug_steps+1):
        optimize_batch(model, optimizer, criterion, image, target)
        if step % 10 == 0 or step == args.debug_steps:
            record(step)
            save_json(args.output/'history.json', history)
    training_peak = torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None
    save_checkpoint(args.output/'debug.pt', model, config, step=args.debug_steps,
                    case_id=sample['case_id'], selected_slice_indices=chosen)
    restored, _ = load_checkpoint(args.output/'debug.pt', device)
    with torch.no_grad():
        model.eval()
        checkpoint_ok = torch.allclose(model(image), restored(image), rtol=1e-5, atol=1e-6)
    changed = any(not torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items())
    checks = {'parameters_changed': changed, 'checkpoint_roundtrip': bool(checkpoint_ok),
              'loss_decreased': history[-1]['train_loss'] < history[0]['train_loss']}
    report = {'mode': 'debug', 'not_generalization_evidence': True,
              'case_id': sample['case_id'], 'slice_indices': chosen,
              'initial_loss': history[0]['train_loss'], 'final_loss': history[-1]['train_loss'],
              'checks': checks, 'passed': all(checks.values()),
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'peak_training_allocated_mib': training_peak}
    save_json(args.output/'debug_report.json', report)
    plot_history(history, args.output/'curves.png', 'debug', args.loss)
    print('BASELINE_DEBUG_PASSED' if report['passed'] else 'BASELINE_DEBUG_FAILED')
    return 0 if report['passed'] else 1


def run_fit(args, train_data, device, model, config):
    validation = HipMRIVolumeDataset(args.manifest, 'validation', args.data_root, tuple(args.shape))
    criterion = segmentation_loss(args.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best, start_epoch, history = -1., 1, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        old = checkpoint['config']
        resume_keys = ('mode', 'model', 'base_channels', 'shape_dhw', 'manifest_sha256', 'learning_rate', 'seed', 'batch_size')
        for key in resume_keys:
            if config[key] != old[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if config['slice_sampling'] != old.get('slice_sampling', 'all'):
            raise ValueError('Resume cannot change slice sampling; start a separate experiment.')
        if config['loss_kind'] != old.get('loss_kind', 'soft-dice'):
            raise ValueError('Resume cannot change loss; start a separate experiment.')
        if config['flip_probability'] != old.get('flip_probability', 0.):
            raise ValueError('Resume cannot change augmentation; start a separate experiment.')
        if (args.model == 'can3d' and config['source_sha256']['can3d.py']
                != old['source_sha256']['can3d.py']):
            raise ValueError('Resume cannot silently change the CAN3D architecture source.')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best, history = checkpoint['best_validation_prostate'], checkpoint['history']
        start_epoch = checkpoint['epoch']+1
        if start_epoch > args.epochs:
            raise ValueError('epochs must extend beyond the resume checkpoint.')
        # Keep the previous best even if no resumed epoch improves it.
        best_checkpoint = torch.load(args.resume.parent/'best.pt', map_location='cpu', weights_only=True)
        if (any(best_checkpoint['config'][key] != old[key] for key in resume_keys)
                or best_checkpoint['config'].get('slice_sampling', 'all') != old.get('slice_sampling', 'all')
                or best_checkpoint['config'].get('loss_kind', 'soft-dice') != old.get('loss_kind', 'soft-dice')
                or best_checkpoint['config'].get('flip_probability', 0.) != old.get('flip_probability', 0.)
                or (args.model == 'can3d' and best_checkpoint['config']['source_sha256']['can3d.py']
                    != old['source_sha256']['can3d.py'])
                or best_checkpoint['best_validation_prostate'] != best):
            raise ValueError('Resume requires the matching best.pt beside last.pt.')
        torch.save(best_checkpoint, args.output/'best.pt')
    for epoch in range(start_epoch, args.epochs+1):
        epoch_start = time.perf_counter()
        policy = (f'whole-volume batch=1; paired left/right flip probability={args.flip_probability}'
                  if args.model == 'can3d' else f'slice_sampling={args.slice_sampling}')
        print(f'Epoch {epoch}: training {len(train_data)} cases; {policy}', flush=True)
        if args.flip_probability:
            # A resumed epoch repeats the same training-only flip decisions.
            seed_everything(args.seed+epoch)
        generator = torch.Generator().manual_seed(args.seed+epoch)
        total_loss, total_slices, prostate_slices = 0., 0, 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for index in torch.randperm(len(train_data), generator=generator).tolist():
            sample = train_data[index]
            if args.model == 'can3d':
                image, target = sample['image'][None].to(device), sample['target'][None].to(device)
                total_loss += optimize_batch(model, optimizer, criterion, image, target)
                total_slices += 1  # Count complete volumes in the denominator.
                continue
            images, targets = volume_to_slices(sample['image'], sample['target'])
            order = training_slice_order(sample['mask'], generator, args.slice_sampling)
            prostate_slices += int((sample['mask'][order] == 5).flatten(1).any(1).sum())
            for first in range(0, len(order), args.batch_size):
                indices = order[first:first+args.batch_size]
                image, target = images[indices].to(device), targets[indices].to(device)
                loss = optimize_batch(model, optimizer, criterion, image, target)
                total_loss += loss*len(indices)
                total_slices += len(indices)
        training_peak = torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None
        print(f'Epoch {epoch}: validation on {len(validation)} complete scans', flush=True)
        validation_report = evaluate_cases(model, validation, device, args.batch_size, loss_kind=args.loss)
        score = validation_report['patient_macro_dice'][5]
        if score is None:
            raise ValueError('No evaluable validation prostate cases.')
        entry = {'epoch': epoch, 'train_loss': total_loss/total_slices,
                 'validation_loss': validation_report['mean_objective_loss'],
                 'validation_soft_dice_loss': validation_report['mean_soft_dice_loss'],
                 'prostate_dice': score, 'peak_training_allocated_mib': training_peak,
                 'sampled_slices': total_slices, 'sampled_prostate_slices': prostate_slices,
                 'epoch_seconds': time.perf_counter()-epoch_start}
        if args.model == 'can3d':
            entry.pop('sampled_slices')
            entry.pop('sampled_prostate_slices')
            entry['training_volumes'] = total_slices
            entry['optimizer_steps'] = total_slices
        history.append(entry)
        improved = score > best
        best = max(best, score)
        extra = dict(epoch=epoch, optimizer_state=optimizer.state_dict(),
                     best_validation_prostate=best, history=history)
        save_checkpoint(args.output/'last.pt', model, config, **extra)
        if improved:
            save_checkpoint(args.output/'best.pt', model, config, **extra)
        save_json(args.output/'history.json', history)
        save_json(args.output/'validation_latest.json', validation_report)
        plot_history(history, args.output/'curves.png', 'fit', args.loss)
        count_text = (f'training volumes={total_slices}' if args.model == 'can3d' else
                      f'sampled prostate slices={prostate_slices}/{total_slices}')
        print(f'Epoch {epoch}: train_loss={entry["train_loss"]:.6f}, '
              f'validation patient-macro prostate Dice={score:.6f}, '
              f'{count_text}, '
              f'seconds={entry["epoch_seconds"]:.1f}', flush=True)
        print('Validation predicted voxels by class 0..5: '+str([
            sum(case['predicted_voxels'][c] for case in validation_report['cases'])
            for c in range(6)]), flush=True)
    marker = 'CAN3D_FIT_COMPLETED' if args.model == 'can3d' else 'BASELINE_FIT_COMPLETED'
    print(marker+'; test data have not been evaluated.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['debug', 'fit', 'test'], default='debug')
    parser.add_argument('--model', choices=['unet2d', 'can3d'], default='unet2d')
    parser.add_argument('--manifest', type=Path, default=Path('runs/split_seed42.json'))
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New directory; refuses overwrite.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--shape', type=int, nargs=3, default=PILOT_SHAPE_DHW)
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--debug-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--flip-probability', type=float, default=0.,
                        help='Training-only paired left/right torch.flip; zero preserves original runs.')
    parser.add_argument('--loss', choices=['soft-dice', 'dice-ce'], default='soft-dice',
                        help='Objective; changing it requires a separate experiment, not resume.')
    parser.add_argument('--slice-sampling', choices=['all', 'prostate-balanced'], default='all',
                        help='Fit only: training slice composition; validation/test stay unchanged.')
    parser.add_argument('--resume', type=Path, help='Resume a fit-mode last.pt, in a new output directory.')
    parser.add_argument('--checkpoint', type=Path, help='Fixed fit checkpoint for mode=test.')
    parser.add_argument('--final-test', action='store_true', help='Explicitly authorize held-out evaluation after choices are fixed.')
    args = parser.parse_args(argv)
    if (min(args.batch_size, args.epochs, args.debug_steps, args.base_channels, *args.shape) < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Sizes, steps and learning rate must be positive.')
    if args.model == 'unet2d' and any(size % 16 for size in args.shape[1:]):
        parser.error('H and W must be multiples of 16 for this U-Net.')
    if not 0 <= args.flip_probability <= 1 or (args.mode != 'fit' and args.flip_probability != 0):
        parser.error('Flip augmentation is only allowed in fit mode, with probability 0..1.')
    if args.model == 'can3d' and args.mode == 'debug':
        parser.error('Use can3d_smoke.py for the training-only CAN3D diagnostic.')
    if args.model == 'can3d' and (args.batch_size != 1 or args.slice_sampling != 'all'):
        parser.error('CAN3D uses batch-size 1 whole volumes, not slice sampling.')
    if args.model == 'can3d' and any(size < 4 or size % 2 for size in args.shape):
        parser.error('CAN3D requires even spatial sizes >=4.')
    if args.resume and args.mode != 'fit':
        parser.error('--resume is only for fit mode.')
    if args.mode != 'fit' and args.slice_sampling != 'all':
        parser.error('--slice-sampling only changes fit-mode training.')
    if args.mode == 'test' and (not args.final_test or not args.checkpoint):
        parser.error('test mode requires --checkpoint and --final-test.')
    if args.mode != 'test' and (args.final_test or args.checkpoint):
        parser.error('Held-out checkpoint evaluation must use mode=test.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    seed_everything(args.seed)
    device = get_device(args.device)
    read_patient_manifest(args.manifest)
    config = {'model': args.model, 'mode': args.mode,
              'architecture_variant': 'methods_text_one_down_8_36_48' if args.model == 'can3d' else 'padded_unet',
              'base_channels': args.base_channels if args.model == 'unet2d' else None,
              'shape_dhw': list(args.shape), 'seed': args.seed, 'batch_size': args.batch_size,
              'learning_rate': args.learning_rate, 'manifest_sha256': file_sha256(args.manifest),
              'slice_sampling': args.slice_sampling,
              'flip_probability': args.flip_probability,
              'normalization': 'per-volume population z-score before resize',
              'loss_kind': args.loss,
              'loss': ('equal six-class soft Dice including background' if args.loss == 'soft-dice'
                       else 'equal six-class soft Dice including background + mean pixel cross-entropy'),
              'source_sha256': {name: file_sha256(Path(__file__).with_name(name))
                                for name in ['modules.py', 'dataset.py', 'train.py', 'experiment.py']
                                + (['can3d.py'] if args.model == 'can3d' else [])},
              'python': platform.python_version(), 'torch': str(torch.__version__),
              'nibabel': nibabel.__version__, 'numpy': np.__version__,
              'hostname': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'device': str(device), 'gpu_name': torch.cuda.get_device_name() if device.type == 'cuda' else None}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output/'config.json', config)
    if args.mode == 'test':
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        old = checkpoint['config']
        if old['mode'] != 'fit' or old['manifest_sha256'] != config['manifest_sha256']:
            raise ValueError('Test requires a fit checkpoint and the exact same saved manifest.')
        # The evaluation grid/model come from weights, not unused CLI defaults.
        config.update(model=old['model'], shape_dhw=old['shape_dhw'], base_channels=old['base_channels'],
                      architecture_variant=old.get('architecture_variant','padded_unet'),
                      batch_size=1 if old['model'] == 'can3d' else args.batch_size,
                      loss_kind=old.get('loss_kind', 'soft-dice'), loss=old['loss'],
                      checkpoint_sha256=file_sha256(args.checkpoint), training_config=old)
        if old['model'] == 'can3d':
            config['source_sha256']['can3d.py'] = file_sha256(Path(__file__).with_name('can3d.py'))
        save_json(args.output/'config.json', config)
        test_data = HipMRIVolumeDataset(args.manifest, 'test', args.data_root, tuple(old['shape_dhw']))
        report = evaluate_cases(model, test_data, device, args.batch_size,
                                loss_kind=old.get('loss_kind', 'soft-dice'))
        report['checkpoint_sha256'] = file_sha256(args.checkpoint)
        save_json(args.output/'test_metrics.json', report)
        print('FINAL_TEST_COMPLETED; do not tune settings from these results.')
        return 0
    model = (CAN3D() if args.model == 'can3d' else UNet2D(base_channels=args.base_channels)).to(device)
    dataset = HipMRIVolumeDataset(args.manifest, 'train', args.data_root, tuple(args.shape),
                                 flip_probability=args.flip_probability)
    print(f'Mode={args.mode}, device={device}, parameters={sum(p.numel() for p in model.parameters())}, '
          f'loss={args.loss}', flush=True)
    return run_debug(args, dataset, device, model, config) if args.mode == 'debug' else run_fit(args, dataset, device, model, config)


if __name__ == '__main__':
    raise SystemExit(main())
