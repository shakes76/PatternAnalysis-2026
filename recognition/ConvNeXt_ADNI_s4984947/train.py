"""Initial ADNI experiments with patient-independent validation."""
import argparse
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import ADNIDataset, build_splits
from modules import CNNBaseline, ConvNeXtClassifier


def select_slices(rows, per_scan):
    """Select evenly spaced slices from each MRI scan."""
    if per_scan <= 0:
        return rows

    scans = defaultdict(list)
    for row in rows:
        scans[row[3]].append(row)

    chosen = []
    for scan in sorted(scans):
        paths = sorted(
            scans[scan],
            key=lambda r: int(r[0].stem.rsplit('_', 1)[1])
        )
        count = min(len(paths), per_scan)
        for i in range(count):
            pos = (2 * i + 1) * len(paths) // (2 * count)
            chosen.append(paths[pos])

    return chosen


def run_epoch(model, loader, device, loss_fn,
              optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)

    total_loss = 0
    total_correct = 0
    total_count = 0

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        images = batch['image'].to(device, non_blocking=True)
        targets = batch['label'].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            amp = (
                torch.autocast('cuda', dtype=torch.bfloat16)
                if device.type == 'cuda'
                else nullcontext()
            )
            with amp:
                predictions = model(images)
                loss = loss_fn(predictions, targets)

            if training:
                loss.backward()
                optimizer.step()

        count = targets.numel()
        total_count += count
        total_loss += loss.detach().item() * count
        total_correct += (
            predictions.argmax(1) == targets
        ).sum().item()

    assert total_count > 0, 'No batches processed'

    return {
        'loss': round(total_loss / total_count, 4),
        'accuracy': round(total_correct / total_count, 4),
        'samples': total_count
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['baseline', 'convnext'],
        default='baseline'
    )
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--slices-per-scan', type=int, default=2)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--output', default='runs/first_experiment')
    args = parser.parse_args()

    torch.manual_seed(3710)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    if device.type != 'cuda':
        raise RuntimeError(
            'GPU unavailable: run inside a Slurm GPU job'
        )

    print('GPU:', torch.cuda.get_device_name(0), flush=True)

    splits = build_splits(seed=3710)
    train_rows = select_slices(
        splits['train'], args.slices_per_scan
    )
    val_rows = select_slices(
        splits['val'], args.slices_per_scan
    )

    print(
        'Training / validation slices:',
        len(train_rows), len(val_rows),
        flush=True
    )

    train_loader = DataLoader(
        ADNIDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        ADNIDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = (
        CNNBaseline()
        if args.model == 'baseline'
        else ConvNeXtClassifier()
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    params = sum(p.numel() for p in model.parameters())
    print(
        'Model:', args.model,
        'Parameters:', f'{params:,}',
        flush=True
    )

    torch.cuda.reset_peak_memory_stats()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()

        train = run_epoch(
            model, train_loader, device,
            loss_fn, optimizer, args.max_train_batches
        )

        # Evaluate the complete selected validation subset.
        val = run_epoch(
            model, val_loader, device, loss_fn
        )

        torch.cuda.synchronize()

        result = {
            'epoch': epoch,
            'train': train,
            'val': val,
            'seconds': round(time.perf_counter() - start, 1)
        }
        history.append(result)
        print(json.dumps(result), flush=True)

        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'model_name': args.model,
                    'epoch': epoch,
                    'val_loss': best_val_loss
                },
                output / 'best.pt'
            )

    metrics = {
        'model': args.model,
        'parameters': params,
        'history': history,
        'peak_gpu_vram_gb': round(
            torch.cuda.max_memory_reserved() / 1024**3, 3
        ),
        'note': 'Preliminary slice-level validation, not test accuracy.'
    }

    (output / 'metrics.json').write_text(
        json.dumps(metrics, indent=2)
    )

    print('Finished:', output / 'metrics.json', flush=True)
    print(
        'Peak GPU VRAM (GB):',
        metrics['peak_gpu_vram_gb'],
        flush=True
    )


if __name__ == '__main__':
    main()
