"""ConvNeXt on ADNI, 65/15/20 patient split, restartable 60-epoch study.

Uses the same loaders, preprocessing, training and validation metric functions
as train_v3.py; validation slice accuracy is the primary selection metric.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v4 import AugmentedADNIDataset
from dataset_v2 import build_splits
from modules import ConvNeXtClassifier
from train import select_slices
from train_v3 import train_one_epoch, validate


def save_torch(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_json(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--stage-epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--slices-per-scan', type=int, default=6)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--output', default='runs/convnext_v3_60_newsplit')
    args = p.parse_args()
    assert args.epochs > 0 and args.stage_epochs > 0
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A100 BF16 GPU required'

    torch.manual_seed(3710)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / 'latest.pt'
    best = out / 'best.pt'
    metrics_file = out / 'metrics.json'

    splits = build_splits(seed=3710)
    split_index = sorted((split, pid)
                         for split, entries in splits.items()
                         for pid in {row[2] for row in entries})
    split_hash = hashlib.sha256(repr(split_index).encode()).hexdigest()
    train_rows = select_slices(splits['train'], args.slices_per_scan)
    val_rows = select_slices(splits['val'], 6)
    print('New split: train', len(train_rows), 'val', len(val_rows),
          'patients in all sets', len(split_index), 'split hash', split_hash[:16], flush=True)

    train_loader = DataLoader(AugmentedADNIDataset(train_rows), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(ADNIDataset(val_rows), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    model = ConvNeXtClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    history = []
    finished = 0
    best_key = (-1.0, -1.0, float('-inf'))
    best_epoch = None

    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != vars(args) or state['split_hash'] != split_hash:
            raise RuntimeError('Config or patient split changed. Use a new output directory.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history = state['history']
        finished = state['epoch']
        best_key = tuple(state['best_key'])
        best_epoch = state['best_epoch']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device=device)
        print('Resuming after epoch', finished, flush=True)

    end = min(args.epochs, finished + args.stage_epochs)
    print(f'GPU: {torch.cuda.get_device_name(0)}; epochs {finished+1}-{end} of {args.epochs}', flush=True)
    if finished >= args.epochs:
        print('Already completed all epochs', flush=True)
        return
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(finished + 1, end + 1):
        started = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']
        train = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = validate(model, val_loader, loss_fn, device)
        torch.cuda.synchronize()
        record = {
            'epoch': epoch, 'lr': round(lr, 8), 'train': train,
            'val': val, 'seconds': round(time.perf_counter() - started, 1)
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (val['slice']['accuracy'], val['slice']['macro_f1'], -val['loss'])
        if key > best_key:
            best_key = key
            best_epoch = epoch
            save_torch({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'val': val, 'selection': 'validation slice accuracy'}, best)
            print('NEW BEST', epoch, 'slice val accuracy', key[0], flush=True)
        scheduler.step()

        save_torch({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch,
            'history': history, 'best_key': best_key, 'best_epoch': best_epoch,
            'config': vars(args), 'split_hash': split_hash,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)
        }, latest)
        save_json({
            'config': vars(args), 'split_hash': split_hash,
            'completed_epochs': epoch, 'best_epoch': best_epoch,
            'best_val_slice_accuracy': best_key[0],
            'history': history,
            'peak_gpu_reserved_gb_this_stage': round(torch.cuda.max_memory_reserved()/1024**3, 3),
            'note': 'Validation only; held-out test is untouched.'
        }, metrics_file)

    print(f'STAGE COMPLETE: {end}/{args.epochs}; best epoch={best_epoch}; best val slice acc={best_key[0]}', flush=True)


if __name__ == '__main__':
    main()
