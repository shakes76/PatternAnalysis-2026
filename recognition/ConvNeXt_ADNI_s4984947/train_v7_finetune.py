"""V7: strictly mapped pretrained ConvNeXt, 2-epoch head warmup, then fine-tune.

Comparison holds the 65/15/20 patient split and 6 slices/scan fixed.
Each Slurm job resumes optimizer, scheduler and RNG from latest.pt.
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
from dataset_v2 import build_splits
from dataset_v4 import AugmentedADNIDataset
from train import select_slices
from train_v3 import train_one_epoch, validate
from v7_pretrained import load_custom_pretrained


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--stage-epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--slices-per-scan', type=int, default=6)
    p.add_argument('--head-warmup-epochs', type=int, default=2)
    p.add_argument('--body-lr', type=float, default=2e-5)
    p.add_argument('--head-lr', type=float, default=3e-4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--weights', default='pretrained/convnext_tiny-983f1562.pth')
    p.add_argument('--output', default='runs/convnext_v7_pretrained20_newsplit')
    args = p.parse_args()

    assert args.epochs > 0 and args.stage_epochs > 0
    assert 0 <= args.head_warmup_epochs < args.epochs
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A100 BF16 GPU required'

    torch.manual_seed(3710)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    latest, best, metrics_file = out/'latest.pt', out/'best.pt', out/'metrics.json'

    splits = build_splits(seed=3710)
    split_index = sorted((split, pid)
                         for split, entries in splits.items()
                         for pid in {row[2] for row in entries})
    split_hash = hashlib.sha256(repr(split_index).encode()).hexdigest()
    train_rows = select_slices(splits['train'], args.slices_per_scan)
    val_rows = select_slices(splits['val'], args.slices_per_scan)
    print('Patient split SHA:', split_hash, 'Train:', len(train_rows),
          'Validation:', len(val_rows), flush=True)

    train_loader = DataLoader(
        AugmentedADNIDataset(train_rows), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        ADNIDataset(val_rows), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = load_custom_pretrained(args.weights, num_classes=2).to(device)
    body = [p for n, p in model.named_parameters() if not n.startswith('classifier.')]
    head = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': body, 'lr': args.body_lr},
        {'params': head, 'lr': args.head_lr}
    ], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = []
    finished = 0
    best_key = (-1.0, -1.0, float('-inf'))
    best_epoch = None
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['config'] != vars(args) or state['split_hash'] != split_hash:
            raise RuntimeError('Config or split has changed! Use a new output directory.')
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

    if finished >= args.epochs:
        print('Already finished all epochs', flush=True)
        return
    end = min(args.epochs, finished + args.stage_epochs)
    print('Device:', torch.cuda.get_device_name(0),
          'Epochs:', finished+1, 'to', end, '/', args.epochs, flush=True)
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(finished+1, end+1):
        start = time.perf_counter()
        unfrozen = epoch > args.head_warmup_epochs
        for param in body:
            param.requires_grad_(unfrozen)
        mode = 'fine-tune all' if unfrozen else 'head only'
        body_lr, head_lr = [g['lr'] for g in optimizer.param_groups]
        print(f'Epoch {epoch}: {mode}; body_lr={body_lr:.8f}, head_lr={head_lr:.8f}', flush=True)
        train = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val = validate(model, val_loader, loss_fn, device)
        torch.cuda.synchronize()
        record = {
            'epoch': epoch, 'phase': mode, 'body_lr': body_lr,
            'head_lr': head_lr, 'train': train,
            'val': val, 'seconds': round(time.perf_counter()-start, 1)
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (val['slice']['accuracy'], val['slice']['macro_f1'], -val['loss'])
        if key > best_key:
            best_key, best_epoch = key, epoch
            atomic_save({
                'model_state_dict': model.state_dict(), 'epoch': epoch,
                'val': val, 'selection': 'validation slice accuracy',
                'architecture': 'hand-written ConvNeXt, ImageNet pretrained'
            }, best)
            print('NEW BEST', best_epoch, best_key[0], flush=True)
        scheduler.step()
        atomic_save({
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'epoch': epoch,
            'history': history, 'best_key': best_key, 'best_epoch': best_epoch,
            'config': vars(args), 'split_hash': split_hash,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(device)
        }, latest)
        atomic_json({
            'config': vars(args), 'split_hash': split_hash,
            'completed_epochs': epoch, 'best_epoch': best_epoch,
            'best_val_slice_accuracy': best_key[0], 'history': history,
            'peak_gpu_reserved_gb_this_stage': round(torch.cuda.max_memory_reserved()/1024**3, 3),
            'note': 'Patient-independent validation only; TEST IS UNTOUCHED.'
        }, metrics_file)
    print('STAGE COMPLETE', end, '/', args.epochs,
          'best epoch', best_epoch, 'accuracy', best_key[0], flush=True)


if __name__ == '__main__':
    main()
