r"""One-batch ConvNeXt-Tiny smoke test on ADNI MRI (COMP3710).

This is a diagnostic only: no full training, validation scoring, or test-set
inference. It checks the baseline project's input pipeline against ConvNeXt.

Run from the project directory (PowerShell):
  & $py .\test_convnext.py `
    --data-root "$HOME\Downloads\COMP3710\ADNI\AD_NC" `
    --manifest-dir "$HOME\Downloads\COMP3710\prepared_seed42" `
    --batch-size 2 --device cuda

By default weights are random, so this test does not require downloading
ImageNet pretrained weights. Pass --pretrained only if you deliberately want
to verify downloading/loading the pretrained model; this is optional.
"""

import argparse
from pathlib import Path
import time

import torch
from torch import nn

from dataset import make_dataloaders
from modules import build_convnext_tiny


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load ImageNet pretrained weights (network download may be required).')
    args = parser.parse_args()

    if args.batch_size < 1 or args.image_size < 32:
        parser.error('batch-size must be >=1 and image-size >=32')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but not available. Check the COMP3710 .venv.')
    device = torch.device('cuda' if (args.device == 'cuda' or
                                   (args.device == 'auto' and torch.cuda.is_available()))
                          else 'cpu')

    torch.manual_seed(42)
    loaders = make_dataloaders(
        data_root=args.data_root, manifest_dir=args.manifest_dir,
        batch_size=args.batch_size, image_size=args.image_size,
        num_workers=0, seed=42, augment_train=False,
    )
    # Access train.csv only. The loader also checks patient separation across
    # the three CSV manifests, without reading test images.
    batch = next(iter(loaders['train']))
    images = batch['image'].to(device)
    labels = batch['label'].to(device, dtype=torch.long)

    model = build_convnext_tiny(num_classes=2, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        logits = model(images)
        loss = criterion(logits, labels)

    assert logits.shape == (len(labels), 2), f'Unexpected logits shape: {tuple(logits.shape)}'
    assert torch.isfinite(logits).all().item(), 'Non-finite model logits'
    assert torch.isfinite(loss).item(), 'Non-finite cross entropy loss'
    assert set(labels.cpu().tolist()).issubset({0, 1}), 'Unexpected class labels'

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    last_layer_grad = model.classifier[-1].weight.grad
    assert last_layer_grad is not None, 'No gradients reached final classifier'
    assert torch.isfinite(last_layer_grad).all().item(), 'Non-finite classifier gradients'
    assert torch.count_nonzero(last_layer_grad).item() > 0, 'Classifier gradients are all zero'
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    print('MODEL: ConvNeXt-Tiny (NC=0, AD=1)')
    print(f'Device: {device}; pretrained ImageNet weights: {args.pretrained}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Input: {tuple(images.shape)}; labels: {tuple(labels.shape)}')
    print(f'Logits: {tuple(logits.shape)}; untrained batch loss: {loss.item():.4f}')
    print(f'One optimizer step elapsed: {elapsed:.2f} seconds')
    if device.type == 'cuda':
        mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f'Peak CUDA allocated memory during step: {mb:.1f} MiB')
    print('PASS: ConvNeXt forward, loss, backward and optimizer step.')
    print('DIAGNOSTIC ONLY: not a model evaluation; test images unused; no weights saved.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, AssertionError, RuntimeError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
