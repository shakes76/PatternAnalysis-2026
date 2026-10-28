"""ADNI ConvNeXt experiment v2: more data and transparent validation."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v2 import build_splits
from modules import ConvNeXtClassifier
from train import select_slices


def class_metrics(true, predicted):
    tn = sum(a == 0 and b == 0 for a, b in zip(true, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(true, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(true, predicted))
    tp = sum(a == 1 and b == 1 for a, b in zip(true, predicted))

    n = max(1, len(true))
    f1_nc = 2 * tn / max(1, 2 * tn + fp + fn)
    f1_ad = 2 * tp / max(1, 2 * tp + fp + fn)

    return {
        'accuracy': round((tn + tp) / n, 4),
        'macro_f1': round((f1_nc + f1_ad) / 2, 4),
        'confusion_TN_FP_FN_TP': [tn, fp, fn, tp],
        'predicted_NC': tn + fn,
        'predicted_AD': fp + tp
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    loss_total = 0.0
    good = 0
    count = 0

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
            loss = loss_fn(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        count += labels.numel()
        loss_total += float(loss.item()) * labels.numel()
        good += int((logits.argmax(1) == labels).sum().item())

    return {
        'loss': round(loss_total / count, 4),
        'accuracy': round(good / count, 4),
        'slices': count
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    all_true, all_pred = [], []
    scan_prob = defaultdict(list)
    scan_patient = {}
    scan_label = {}
    loss_total, count = 0.0, 0

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
            loss = loss_fn(logits, labels)

        probs = logits.float().softmax(dim=1)[:, 1].cpu().tolist()
        y = labels.cpu().tolist()
        preds = logits.argmax(1).cpu().tolist()

        all_true.extend(y)
        all_pred.extend(preds)
        count += len(y)
        loss_total += float(loss.item()) * len(y)

        for p, sid, pid, label in zip(
            probs, batch['scan_id'], batch['patient_id'], y
        ):
            scan_prob[sid].append(p)
            scan_patient[sid] = pid
            scan_label[sid] = label

    # Average slices into scans, then scans into patients.
    patient_scan_probs = defaultdict(list)
    patient_labels = {}

    for sid, vals in scan_prob.items():
        pid = scan_patient[sid]
        patient_scan_probs[pid].append(sum(vals) / len(vals))
        if pid in patient_labels:
            assert patient_labels[pid] == scan_label[sid]
        patient_labels[pid] = scan_label[sid]

    patients = sorted(patient_scan_probs)
    patient_true = [patient_labels[p] for p in patients]
    patient_pred = [
        int(
            sum(patient_scan_probs[p]) /
            len(patient_scan_probs[p]) >= 0.5
        )
        for p in patients
    ]

    return {
        'loss': round(loss_total / count, 4),
        'slice': class_metrics(all_true, all_pred),
        'patient': class_metrics(patient_true, patient_pred),
        'validation_slices': count,
        'validation_patients': len(patients)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--slices-per-scan', type=int, default=6)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--output', default='runs/convnext_v2')
    args = p.parse_args()

    assert torch.cuda.is_available(), 'Run with an allocated A100 GPU'
    assert torch.cuda.is_bf16_supported(), 'BF16 requires an appropriate GPU'

    torch.manual_seed(3710)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows = build_splits(seed=3710)
    train_rows = select_slices(
        rows['train'], args.slices_per_scan
    )
    val_rows = select_slices(
        rows['val'], args.slices_per_scan
    )

    print(
        'GPU:', torch.cuda.get_device_name(0),
        'train slices:', len(train_rows),
        'val slices:', len(val_rows),
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

    model = ConvNeXtClassifier().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5
    )
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_loss = float('inf')
    history = []

    torch.cuda.reset_peak_memory_stats()

    print(
        'Parameters:',
        sum(p.numel() for p in model.parameters()),
        flush=True
    )

    for epoch in range(1, args.epochs + 1):
        begin = time.perf_counter()
        lr = optimizer.param_groups[0]['lr']

        tr = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val = validate(model, val_loader, loss_fn, device)
        torch.cuda.synchronize()

        record = {
            'epoch': epoch,
            'lr': round(lr, 8),
            'train': tr,
            'val': val,
            'seconds': round(time.perf_counter() - begin, 1)
        }

        history.append(record)
        print(json.dumps(record), flush=True)

        score = val['slice']['accuracy']

        if score > best_accuracy or (
            score == best_accuracy and val['loss'] < best_loss
        ):
            best_accuracy, best_loss = score, val['loss']

            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'selection': 'validation slice accuracy',
                'val_slice_accuracy': score,
                'val_loss': val['loss']
            }, output / 'best.pt')

        scheduler.step()

        (output / 'metrics.json').write_text(
            json.dumps({
                'args': vars(args),
                'history': history,
                'best_validation_slice_accuracy': best_accuracy,
                'peak_gpu_vram_gb': round(
                    torch.cuda.max_memory_reserved() / 1024**3, 3
                ),
                'note': 'Validation only; test split remains untouched.'
            }, indent=2)
        )

    print('DONE; metrics:', output / 'metrics.json', flush=True)


if __name__ == '__main__':
    main()
