"""Train the author's standard 2D U-Net with scan-level validation selection.

Reference discussion informed general CE/Adam/early-stopping choices only;
implementation and patient split are independent. No test inference here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch import nn
from torch.nn import functional as F

from dataset import DEFAULT_SPLIT, IGNORE_INDEX, get_dataloaders
from modules import UNet2D


class SegmentationLoss(nn.Module):
    """CE plus soft foreground Dice; exclude padding from both terms."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError('dice_weight must be between zero and one')
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        valid = target != IGNORE_INDEX
        if not valid.any():
            raise ValueError('Batch has no valid pixels')
        ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        if self.dice_weight == 0:
            return ce
        # Replace ignored IDs before one_hot; mask predictions as well as GT.
        truth = F.one_hot(target.masked_fill(~valid, 0), 6).movedim(-1, 1)
        mask = valid.unsqueeze(1)
        probs = logits.float().softmax(1)[:, 1:] * mask
        truth = truth[:, 1:].float() * mask
        dims = (0, 2, 3)
        dice = (2 * (probs * truth).sum(dims) + 1e-5) / (
            probs.sum(dims) + truth.sum(dims) + 1e-5)
        return (1 - self.dice_weight) * ce + self.dice_weight * (1 - dice).mean()


def dice_from_matrix(matrix):
    """Rows=GT, columns=prediction. Both-empty is undefined, not perfect."""
    denom = matrix.sum(0) + matrix.sum(1)
    return [2 * int(matrix[c, c]) / int(denom[c]) if denom[c] else None
            for c in range(6)]


def mean_defined(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


@torch.inference_mode()
def validate(model, loader, criterion, device, smoke_steps=0):
    model.eval()
    scans, seen = {}, {}
    loss_sum = valid_count = 0
    for step, (images, masks, metadata) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite validation loss')
        # Loss averages are weighted by valid pixels, excluding right padding.
        n = int((masks != IGNORE_INDEX).sum())
        loss_sum += loss.item() * n
        valid_count += n
        preds, targets = logits.argmax(1).cpu(), masks.cpu()
        for i in range(len(images)):
            key = (str(metadata['patient_id'][i]), int(metadata['week_id'][i]))
            index = int(metadata['slice_id'][i])
            if index in seen.setdefault(key, set()):
                raise ValueError(f'Duplicate validation slice: {key}, {index}')
            seen[key].add(index)
            valid = targets[i] != IGNORE_INDEX
            matrix = torch.bincount(6 * targets[i][valid] + preds[i][valid],
                                    minlength=36).reshape(6, 6)
            scans[key] = scans.get(key, torch.zeros(6, 6, dtype=torch.int64)) + matrix
        if smoke_steps and step >= smoke_steps:
            break
    if not scans:
        raise ValueError('Validation loader is empty')
    if not smoke_steps and any(indices != set(range(60)) for indices in seen.values()):
        raise ValueError('Formal validation requires all 60 slices of each scan')
    rows = [dice_from_matrix(matrix) for matrix in scans.values()]
    case_dice = [mean_defined([row[c] for row in rows]) for c in range(6)]
    total = sum(scans.values(), torch.zeros(6, 6, dtype=torch.int64))
    pooled = dice_from_matrix(total)
    tp = int(total[5, 5]); pred = int(total[:, 5].sum()); gt = int(total[5].sum())
    return dict(loss=loss_sum / valid_count, case_dice=case_dice,
                valid_cases=[sum(row[c] is not None for row in rows) for c in range(6)],
                pooled_dice=pooled, scans=len(scans),
                prostate_precision=tp / pred if pred else None,
                prostate_recall=tp / gt if gt else None,
                prostate_volume_ratio=pred / gt if gt else None)


def train_epoch(model, loader, criterion, optimizer, device, smoke_steps=0):
    model.train()
    loss_sum = valid_count = 0
    for step, (images, masks, _) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        n = int((masks != IGNORE_INDEX).sum())
        loss_sum += loss.item() * n
        valid_count += n
        if step % 50 == 0:
            print(f'  train batch {step}/{len(loader)}, loss={loss.item():.5f}', flush=True)
        if smoke_steps and step >= smoke_steps:
            break
    if not valid_count:
        raise ValueError('Training loader is empty')
    return loss_sum / valid_count


def fit(model, train_loader, val_loader, args, config):
    """Validation-only checkpoint selection; test loader is never consumed."""
    out = Path(args.output_dir)
    criterion = SegmentationLoss(args.dice_weight).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
    best, stale = -math.inf, 0
    fields = ['epoch', 'train_loss', 'val_loss', 'val_prostate_dice',
              'val_foreground_dice', 'val_prostate_pooled_dice',
              'val_prostate_precision', 'val_prostate_recall',
              'val_prostate_volume_ratio', 'lr', 'epoch_seconds']
    fields += [f'val_dice_class_{c}' for c in range(6)]
    fields += [f'val_valid_cases_class_{c}' for c in range(6)]
    with (out / 'history.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            used_lr = optimizer.param_groups[0]['lr']
            train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                     args.device, args.smoke_steps)
            val = validate(model, val_loader, criterion, args.device, args.smoke_steps)
            score = val['case_dice'][5]
            if score is None or not math.isfinite(score):
                raise ValueError('No defined validation prostate Dice')
            scheduler.step(score)
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val['loss'],
                       val_prostate_dice=score,
                       val_foreground_dice=mean_defined(val['case_dice'][1:]),
                       val_prostate_pooled_dice=val['pooled_dice'][5],
                       val_prostate_precision=val['prostate_precision'],
                       val_prostate_recall=val['prostate_recall'],
                       val_prostate_volume_ratio=val['prostate_volume_ratio'],
                       lr=used_lr, epoch_seconds=time.perf_counter() - start)
            row.update({f'val_dice_class_{c}': val['case_dice'][c] for c in range(6)})
            row.update({f'val_valid_cases_class_{c}': val['valid_cases'][c] for c in range(6)})
            writer.writerow(row); stream.flush()
            improved = score > best
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            checkpoint = dict(epoch=epoch, model_state_dict=model.state_dict(),
                              optimizer_state_dict=optimizer.state_dict(),
                              scheduler_state_dict=scheduler.state_dict(),
                              model_config=config['model_config'], config=config,
                              validation=val, best_val_prostate_dice=best,
                              selection_metric='mean_scan_prostate_dice',
                              smoke_only=bool(args.smoke_steps))
            torch.save(checkpoint, out / 'last_model.pt')
            if improved:
                torch.save(checkpoint, out / 'best_model.pt')
            print(f'Epoch {epoch}/{args.epochs}: train={train_loss:.5f} '
                  f'val={val["loss"]:.5f} prostate Dice={score:.5f} '
                  f'best={best:.5f} seconds={row["epoch_seconds"]:.1f}', flush=True)
            if epoch >= args.min_epochs and stale >= args.patience:
                print(f'Early stopping: {stale} epochs without improvement', flush=True)
                break
    result = dict(best_val_prostate_dice=best, completed_epochs=epoch,
                  smoke_only=bool(args.smoke_steps), test_evaluated=False)
    (out / 'training_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    plot_history(out)
    return result


def plot_history(out):
    """Plotting is optional; missing Matplotlib must not discard training."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
    except ImportError:
        print('Matplotlib unavailable; history.csv contains the learning curves.', flush=True)
        return
    with (out / 'history.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(r['epoch']) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ('train_loss', 'val_loss'):
        axes[0].plot(epochs, [float(r[key]) for r in rows], label=key)
    axes[1].plot(epochs, [float(r['val_prostate_dice']) for r in rows],
                 label='Validation mean scan prostate Dice')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend()
    axes[0].set_ylabel('CE/Dice objective'); axes[1].set_ylabel('Dice')
    fig.tight_layout(); fig.savefig(out / 'training_plot.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output-dir', required=True, help='New directory; never overwrite a run')
    p.add_argument('--split-manifest', default=str(DEFAULT_SPLIT))
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--base-features', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--dice-weight', type=float, default=0.5,
                   help='0 restores CE-only training; default is equal CE/foreground Dice')
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-steps', type=int, default=0,
                   help='Limit train/val batches; outputs are NOT performance evidence')
    args = p.parse_args()
    for key in ('epochs', 'batch_size', 'base_features', 'patience', 'min_epochs', 'threads'):
        if getattr(args, key) < 1:
            p.error(f'{key} must be positive')
    if args.workers < 0 or args.smoke_steps < 0 or not 0 <= args.dice_weight <= 1:
        p.error('workers/smoke-steps must be nonnegative; dice-weight must be in [0,1]')
    if not math.isfinite(args.lr) or args.lr <= 0:
        p.error('lr must be finite and positive')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA was requested but is unavailable')
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.workers, args.split_manifest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(arguments=vars(args).copy(), torch_version=str(torch.__version__),
                  model_config=dict(in_channels=1, num_classes=6, base_features=args.base_features),
                  patient_split=json.loads(Path(args.split_manifest).read_text(encoding='utf-8')),
                  training_slices=len(train_loader.dataset), validation_slices=len(val_loader.dataset),
                  smoke_only=bool(args.smoke_steps),
                  validation_region='60 supplied coronal slices per scan, native 2D resolution',
                  loss_note='Pixel-weighted batch objective averages; soft Dice is computed per batch')
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    shutil.copyfile(args.split_manifest, out / 'patient_split.json')
    snapshot = out / 'source_snapshot'; snapshot.mkdir()
    for name in ('train.py', 'modules.py', 'dataset.py'):
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    snapshot_support = snapshot / 'support'
    snapshot_support.mkdir()
    shutil.copyfile(args.split_manifest, snapshot_support / 'patient_split.json')
    model = UNet2D(**config['model_config']).to(args.device)
    print(f'Device={args.device}; parameters={sum(p.numel() for p in model.parameters())}; '
          f'train/val slices={len(train_loader.dataset)}/{len(val_loader.dataset)}; '
          f'smoke_only={bool(args.smoke_steps)}', flush=True)
    fit(model, train_loader, val_loader, args, config)


if __name__ == '__main__':
    main()
