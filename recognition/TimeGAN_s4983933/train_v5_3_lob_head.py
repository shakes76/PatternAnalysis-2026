"""COMP3710 TimeGAN V5.3 — training-mean-initialized constrained LOB Head experiment.

This trains only ConstraintAwareLOBHeadV52 to reconstruct the first 40
normalized LOB features from a previously trained V4.3 Embedder. It compares
to the frozen V4.3 Recovery on identical validation/test samples.
Only initial output-layer weights/biases differ from V5.2. No GAN or Tick Head is trained.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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



def initialise_head_from_train(head, normalizer, train_norm, seed=42):
    """Use only train observations to initialize the 39 raw-output biases.

    Projection weights are deliberately small (not zero), so initial output
    lies near train means while gradient flow remains nonzero.
    """
    # Inverse transform the same normalized training sequences used by V5.2.
    train_raw = normalizer.inverse_transform(train_norm)
    book = train_raw[..., :40]
    prices = book[..., 0::2]
    asks, bids = prices[..., 0::2], prices[..., 1::2]
    spread = asks[..., 0] - bids[..., 0]
    ask_gaps = np.diff(asks, axis=-1)
    bid_gaps = -np.diff(bids, axis=-1)
    transformed_sizes = book[..., 1::2]

    # Shape: 1 spread + 9 ask gaps + 9 bid gaps + 20 sizes = 39.
    targets = np.concatenate((
        [spread.mean()],
        ask_gaps.mean(axis=(0, 1)),
        bid_gaps.mean(axis=(0, 1)),
        transformed_sizes.mean(axis=(0, 1)),
    ))
    floors = np.concatenate((
        [head.min_spread], np.full(18, head.min_gap), np.zeros(20)
    ))
    adjusted = targets - floors
    if targets.shape != (39,) or not np.isfinite(targets).all():
        raise ValueError('Invalid output mean targets')
    if np.any(adjusted <= 0):
        raise ValueError('Observed train mean is not above output floor')

    inverse_softplus = np.log(np.expm1(adjusted))
    with torch.no_grad():
        # Parameters were already seeded by seed_all in main().
        torch.nn.init.normal_(head.projection.weight, mean=0.0, std=1e-3)
        head.projection.bias.copy_(torch.as_tensor(
            inverse_softplus, dtype=head.projection.bias.dtype,
            device=head.projection.bias.device
        ))

    # Check output at latent zero; this is a mathematical initialization check,
    # not a claim about the real held-out reconstruction quality.
    with torch.no_grad():
        zero_hidden = torch.zeros(1, 1, head.projection.in_features,
                                  device=head.projection.weight.device)
        preactivation = head.projection(zero_hidden)[0, 0]
        recovered = F.softplus(preactivation).cpu().numpy() + floors
    max_error = float(np.max(np.abs(recovered - targets)))
    print('Initialization targets: spread=%.6f ask_gap=%.6f bid_gap=%.6f' % (
        targets[0], targets[1:10].mean(), targets[10:19].mean()), flush=True)
    print('Initialization inverse-softplus max error:', max_error, flush=True)
    if max_error > 1e-5:
        raise RuntimeError('Initialized bias failed to recover targets')
    return {'spread_mean': float(targets[0]),
            'ask_gap_mean': float(targets[1:10].mean()),
            'bid_gap_mean': float(targets[10:19].mean()),
            'initial_bias_max_error': max_error}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--checkpoint', default='checkpoints_v4_3_full/timegan_joint.pt')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output-dir', default='outputs_v5_3_lob_head')
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
    init_info = initialise_head_from_train(head, data['normalizer'], data['train'])
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data['train']).float()),
        batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print('=== V5.3 Mean-Initialized Frozen-Embedder LOB Reconstruction ===', flush=True)
    print('Train/val/test windows:', *(len(data[s]) for s in ('train', 'val', 'test')), flush=True)
    print('Only new LOB Head is trainable; original V4.3 remains frozen.', flush=True)
    minimum = head.minimum
    scale = head.scale
    initial_val = evaluate(embedder, recovery, head, data['val'], minimum, scale, device, args.batch_size)
    print('Initial validation new LOB MSE:', initial_val['new_lob_mse_normalized'], flush=True)
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
        'initialization': init_info,
        'seed': args.seed,
    }, output_dir / 'best_lob_head.pt')
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as file:
        json.dump({'initialization': init_info, 'initial_validation': initial_val, 'history': history, 'validation': final_val, 'test': final_test}, file, indent=2)
    print('\nSaved:', output_dir / 'best_lob_head.pt')
    print('Saved:', output_dir / 'results.json')
    print('V5.3 head training complete.')


if __name__ == '__main__':
    main()
