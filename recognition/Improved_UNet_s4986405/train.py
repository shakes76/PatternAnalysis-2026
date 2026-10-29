"""Train a selected U-Net with isolated checkpoints and epoch histories."""
import argparse
from itertools import islice
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from check_splits import collect_patients
from dataset import HipMRI2DDataset
from shape_batching import ShapeBatchSampler
from losses import dice_ce_loss
from metrics import confusion_matrix, dice_iou_from_confusion
from model_factory import ARCHITECTURES, build_model
from training_io import RunRecorder, source_hashes
from plot_history import plot_history
from gpu_profile import save_gpu_profile


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    """Run train or validation; loss is sample-weighted mean of batch losses.

    Hard-label metrics pool the confusion matrix over all processed pixels.
    A supplied optimizer enables training. Validation disables autograd.
    """
    if max_batches is not None and (not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError('max_batches must be positive or None')
    training = optimizer is not None
    model.train(training)
    total_confusion = torch.zeros(6, 6, dtype=torch.long)
    loss_sum = 0.0
    samples = batches = pixels = 0
    with torch.set_grad_enabled(training):
        for images, targets in islice(loader, max_batches):
            images = images.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = dice_ce_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if training:
                loss.backward()
                # Check the global gradient norm without clipping finite gradients.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float('inf'), error_if_nonfinite=True,
                )
                optimizer.step()
            prediction = logits.detach().argmax(dim=1).cpu()
            labels = targets.detach().argmax(dim=1).cpu()
            total_confusion += confusion_matrix(prediction, labels)
            batch_samples = images.shape[0]
            loss_sum += loss.detach().item() * batch_samples
            samples += batch_samples
            batches += 1
            pixels += labels.numel()
    if samples == 0:
        raise ValueError('No samples processed in this epoch')
    if total_confusion.sum().item() != pixels:
        raise RuntimeError('Confusion matrix pixel count mismatch')
    scores = dice_iou_from_confusion(total_confusion)
    return {
        'loss': loss_sum / samples, 'samples': samples, 'batches': batches,
        'pixels': pixels, 'confusion': total_confusion, **scores,
    }


def report(phase, result):
    print(
        f'{phase}: loss={result["loss"]:.6f}, batches={result["batches"]}, '
        f'samples={result["samples"]}, pixels={result["pixels"]}', flush=True,
    )
    for label in range(6):
        print(
            f'  Label {label}: Dice={result["dice"][label].item():.4f}, '
            f'IoU={result["iou"][label].item():.4f}', flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for this run')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--base-channels', type=int, default=4)
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='UNet2D')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3710)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--max-train-batches', type=int, default=1, help='0 means all training batches')
    parser.add_argument('--max-val-batches', type=int, default=1, help='0 means all validation batches')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.base_channels) < 1:
        parser.error('epochs, batch-size and base-channels must be positive')
    if min(args.max_train_batches, args.max_val_batches) < 0:
        parser.error('Batch limits must be non-negative; 0 means all batches')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr must be finite and positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; GPU validation is required before GPU training')

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    run_started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    train_images = args.data_root / 'keras_slices_train'
    val_images = args.data_root / 'keras_slices_validate'
    train_patients, train_count = collect_patients(train_images)
    val_patients, val_count = collect_patients(val_images)
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f'Train/validation patient overlap: {sorted(overlap)}')
    print(f'Train: {train_count} slices, {len(train_patients)} patients', flush=True)
    print(f'Validation: {val_count} slices, {len(val_patients)} patients', flush=True)
    print('Train/validation patient separation: PASS', flush=True)
    train_dataset = HipMRI2DDataset(
        train_images, args.data_root / 'keras_slices_seg_train',
        normalize=True, one_hot=True, training=True, augment=args.augment,
    )
    val_dataset = HipMRI2DDataset(
        val_images, args.data_root / 'keras_slices_seg_validate',
        normalize=True, one_hot=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    print('Inspecting native slice shapes for batching...', flush=True)
    train_sampler = ShapeBatchSampler(
        train_dataset, args.batch_size, shuffle=True, generator=generator,
    )
    val_sampler = ShapeBatchSampler(val_dataset, args.batch_size, shuffle=False)
    print('Train shape groups:', train_sampler.shape_counts, flush=True)
    print('Validation shape groups:', val_sampler.shape_counts, flush=True)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=0, generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
    model_config = {'in_channels': 1, 'num_classes': 6, 'base_channels': args.base_channels}
    if args.architecture == 'ImprovedUNet2D':
        model_config['dropout'] = 0.3
    model = build_model(args.architecture, model_config).to(device)
    print('Architecture:', args.architecture, flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Device: {device}; seed: {args.seed}; base channels: {args.base_channels}', flush=True)
    print(f'Train augmentation: {args.augment}; validation augmentation: False', flush=True)
    print(
        f'Batch limits: train={args.max_train_batches}, validation={args.max_val_batches} '
        '(0=all batches).', flush=True,
    )
    if args.max_train_batches or args.max_val_batches:
        print('PARTIAL RUN: subset metrics are execution checks, not final evaluation.', flush=True)
    config = {
        'arguments': {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'architecture': args.architecture,
        'batching': {'strategy': 'same_shape_buckets', 'drop_last': False,
                     'train_shapes': train_sampler.shape_counts,
                     'validation_shapes': val_sampler.shape_counts},
        'model_config': model_config,
        'preprocessing': {'normalize': True, 'one_hot': True, 'train_flip_probability': 0.5 if args.augment else 0.0},
        'loss': {'name': 'dice_ce_loss', 'ce_weight': 1.0, 'dice_weight': 1.0, 'smooth': 1e-5},
        'partial_run': bool(args.max_train_batches or args.max_val_batches),
        'splits': {'train_patients': sorted(train_patients), 'validation_patients': sorted(val_patients),
                   'train_slices': train_count, 'validation_slices': val_count},
        'source_sha256': source_hashes(Path(__file__).resolve().parent),
    }
    recorder = RunRecorder(args.output_dir, config)
    print('Run directory:', recorder.directory, flush=True)
    print('Best checkpoint criterion: minimum validation loss', flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}', flush=True)
        train_result = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            max_batches=args.max_train_batches or None,
        )
        val_result = run_epoch(
            model, val_loader, device, max_batches=args.max_val_batches or None,
        )
        report('Train', train_result)
        report('Validation', val_result)
        improved = recorder.record(
            epoch, model, optimizer, train_result, val_result, generator,
        )
        print(f'Saved epoch {epoch}; best epoch={recorder.best_epoch}; new best={improved}', flush=True)
    plot_path = plot_history(args.output_dir)
    print('Saved training curves:', plot_path, flush=True)
    if device.type == 'cuda':
        profile = save_gpu_profile(args.output_dir, device, run_started)
        print(f'GPU peak allocated GiB: {profile["peak_allocated_gib"]:.6f}', flush=True)
        print(f'GPU peak reserved GiB: {profile["peak_reserved_gib"]:.6f}', flush=True)
        print('GPU training profile: PASS', flush=True)
    print('Training/validation run: PASS', flush=True)


if __name__ == '__main__':
    main()
