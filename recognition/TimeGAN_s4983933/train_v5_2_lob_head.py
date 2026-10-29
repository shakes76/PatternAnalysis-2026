"""COMP3710 TimeGAN V5.2 — frozen-Embedder constrained LOB Head experiment.

This trains only ConstraintAwareLOBHeadV52 to reconstruct the first 40
normalized LOB features from a previously trained V4.3 Embedder. It compares
to the frozen V4.3 Recovery on identical validation/test samples.
No GAN or Tick Head is trained. Run from ~/COMP3710_TimeGAN on Rangpur.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42
from lob_head_v5_2 import ConstraintAwareLOBHeadV52, check_raw_constraints


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(embedder, recovery, head, arrays, minimum, scale, device, batch_size):
    """Evaluate old Recovery and new Head on the same preselected split."""
    embedder.eval()
    recovery.eval()
    head.eval()
    sums = {k: 0.0 for k in ('old_sq', 'new_sq', 'old_spread_abs', 'new_spread_abs')}
    num_elements = 0
    num_rows = 0
    passed = {key: True for key in (
        'positive_spread', 'ask_ordered', 'bid_ordered',
        'ask_above_mid', 'bid_below_mid', 'nonnegative_sizes', 'finite'
    )}
    with torch.inference_mode():
        for start in range(0, len(arrays), batch_size):
            x = torch.from_numpy(arrays[start:start + batch_size]).float().to(device)
            h = embedder(x)
            old = recovery(h)[..., :40]
            new = head(h)
            real = x[..., :40]
            sums['old_sq'] += (old - real).square().sum().item()
            sums['new_sq'] += (new - real).square().sum().item()
            num_elements += real.numel()
            num_rows += real.shape[0] * real.shape[1]
            # Relative price units: ask1 index 0, bid1 index 2.
            true_spread = (real[..., 0] * scale[0] + minimum[0]) - (
                real[..., 2] * scale[2] + minimum[2]
            )
            old_spread = (old[..., 0] * scale[0] + minimum[0]) - (
                old[..., 2] * scale[2] + minimum[2]
            )
            new_spread = (new[..., 0] * scale[0] + minimum[0]) - (
                new[..., 2] * scale[2] + minimum[2]
            )
            sums['old_spread_abs'] += (old_spread - true_spread).abs().sum().item()
            sums['new_spread_abs'] += (new_spread - true_spread).abs().sum().item()
            raw = head.forward_raw(h)
            for key, value in check_raw_constraints(raw).items():
                passed[key] = passed[key] and value
    return {
        'old_lob_mse_normalized': sums['old_sq'] / num_elements,
        'new_lob_mse_normalized': sums['new_sq'] / num_elements,
        'old_spread_mae_price_units': sums['old_spread_abs'] / num_rows,
        'new_spread_mae_price_units': sums['new_spread_abs'] / num_rows,
        'new_head_all_constraints': passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--checkpoint', default='checkpoints_v4_3_full/timegan_joint.pt')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='outputs_v5_2_lob_head')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs and batch-size must be positive; lr must be > 0')
    seed_all(args.seed)
    device = torch.device(args.device)
    data = prepare_dataset_v4(args.data_dir)
    book = data['normalizer'].book_normalizer
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = TimeGANV42(
        feature_dim=ckpt['feature_dim'],
        hidden_dim=ckpt['hidden_dim'],
        noise_dim=ckpt['noise_dim'],
        num_layers=ckpt['num_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    embedder = model.embedder
    recovery = model.recovery
    for p in model.parameters():
        p.requires_grad_(False)
    head = ConstraintAwareLOBHeadV52(
        hidden_dim=ckpt['hidden_dim'],
        minimum=book.minimum,
        scale=book.scale,
        min_spread=0.01,
        min_gap=0.01,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data['train']).float()),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=== V5.2 Frozen-Embedder LOB Head Reconstruction ===', flush=True)
    print('Train/val/test windows:', *(len(data[s]) for s in ('train', 'val', 'test')), flush=True)
    print('Only new LOB Head is trainable; original V4.3 remains frozen.', flush=True)
    minimum = head.minimum
    scale = head.scale
    history = []
    best_mse = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        count = 0
        for (x,) in loader:
            x = x.to(device)
            with torch.no_grad():
                h = embedder(x)
            output = head(h)
            target = x[..., :40]
            loss = nn.functional.mse_loss(output, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            count += len(x)
        val = evaluate(embedder, recovery, head, data['val'], minimum, scale, device, args.batch_size)
        history.append({'epoch': epoch, 'train_lob_mse': total / count, **val})
        print(
            f"Epoch {epoch}/{args.epochs} | Train MSE={total/count:.6f} | "
            f"Val old MSE={val['old_lob_mse_normalized']:.6f} | "
            f"Val new MSE={val['new_lob_mse_normalized']:.6f} | "
            f"Val old/new Spread MAE={val['old_spread_mae_price_units']:.6f}/"
            f"{val['new_spread_mae_price_units']:.6f}", flush=True
        )
        if val['new_lob_mse_normalized'] < best_mse:
            best_mse = val['new_lob_mse_normalized']
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    final_val = evaluate(embedder, recovery, head, data['val'], minimum, scale, device, args.batch_size)
    final_test = evaluate(embedder, recovery, head, data['test'], minimum, scale, device, args.batch_size)
    print('\n=== Best-head validation (selected by Val normalized LOB MSE) ===')
    print(json.dumps(final_val, indent=2))
    print('\n=== Test (evaluated once after selecting epoch) ===')
    print(json.dumps(final_test, indent=2))
    torch.save({
        'head_state_dict': best_state,
        'hidden_dim': ckpt['hidden_dim'],
        'source_checkpoint': args.checkpoint,
        'normalizer_minimum': np.asarray(book.minimum).tolist(),
        'normalizer_scale': np.asarray(book.scale).tolist(),
        'best_val_lob_mse': best_mse,
        'seed': args.seed,
    }, output_dir / 'best_lob_head.pt')
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as file:
        json.dump({'history': history, 'validation': final_val, 'test': final_test}, file, indent=2)
    print('\nSaved:', output_dir / 'best_lob_head.pt')
    print('Saved:', output_dir / 'results.json')
    print('V5.2 head training complete.')


if __name__ == '__main__':
    main()
