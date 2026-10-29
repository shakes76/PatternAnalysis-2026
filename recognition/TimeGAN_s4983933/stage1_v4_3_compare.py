"""TimeGAN V4.3: controlled Stage-1 reconstruction experiment.

Compare ordinary MSE against return-aware weighted MSE. No GAN training.
Run from the project directory on Rangpur.
"""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_v4 import prepare_dataset_v4
from modules_v4_2 import TimeGANV42


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def return_aware_loss(prediction, target, zero_normalized, return_weight, move_weight):
    lob_loss = nn.functional.mse_loss(prediction[..., :40], target[..., :40])
    squared_error = (prediction[..., 40] - target[..., 40]).square()
    # In asinh + z-score space, a raw zero return maps to zero_normalized.
    moved = (target[..., 40] - zero_normalized).abs() > 1e-4
    weights = torch.where(moved, move_weight, 1.0).to(squared_error.dtype)
    weighted_return_loss = (weights * squared_error).sum() / weights.sum()
    return lob_loss + return_weight * weighted_return_loss


def reconstruct_metrics(model, data, normalizer, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data), 128):
            x = torch.from_numpy(data[start:start + 128]).float().to(device)
            outputs.append(model.reconstruct(x).cpu().numpy())
    reconstruction = np.concatenate(outputs)
    real = normalizer.inverse_transform(data)[..., -1].reshape(-1)
    predicted = normalizer.inverse_transform(reconstruction)[..., -1].reshape(-1)
    if np.std(real) and np.std(predicted):
        corr = float(np.corrcoef(real, predicted)[0, 1])
    else:
        corr = float('nan')
    moved = np.abs(real) > 1e-12
    return {
        'corr': corr,
        'real_std': float(np.std(real)),
        'recon_std': float(np.std(predicted)),
        'mae': float(np.mean(np.abs(real - predicted))),
        'move_mae': float(np.mean(np.abs(real[moved] - predicted[moved]))) if moved.any() else float('nan'),
        'move_fraction': float(moved.mean()),
        'lob_mse': float(np.mean((data[..., :40] - reconstruction[..., :40]) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--return-weight', type=float, default=1.0)
    parser.add_argument('--move-weight', type=float, default=5.0)
    parser.add_argument('--output-dir', default='checkpoints_v4_3_stage1')
    args = parser.parse_args()
    if args.epochs < 1 or args.return_weight <= 0 or args.move_weight < 1:
        parser.error('epochs >= 1, return-weight > 0 and move-weight >= 1 required')
    seed_everything(args.seed)
    device = torch.device(args.device)
    data = prepare_dataset_v4(args.data_dir)
    norm = data['normalizer']
    # asinh(0) == 0; z-score of this value is (-mean / std).
    zero_normalized = -float(norm.return_mean) / float(norm.return_std)
    print('Raw zero return normalized as:', zero_normalized)
    print('Train / val / test:', *(data[s].shape for s in ('train', 'val', 'test')))

    seed_everything(args.seed)
    reference = TimeGANV42(feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2)
    initial_state = copy.deepcopy(reference.state_dict())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant in ('ordinary_mse', 'weighted_returns'):
        seed_everything(args.seed)
        model = TimeGANV42(feature_dim=41, hidden_dim=24, noise_dim=41, num_layers=2).to(device)
        model.load_state_dict(initial_state)
        opt = torch.optim.Adam(
            list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=0.001
        )
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(data['train']).float()),
            batch_size=args.batch_size, shuffle=True, generator=generator,
        )
        print(f'\n=== {variant} ===', flush=True)
        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for (x,) in loader:
                x = x.to(device)
                prediction = model.reconstruct(x)
                if variant == 'ordinary_mse':
                    loss = nn.functional.mse_loss(prediction, x)
                else:
                    loss = return_aware_loss(
                        prediction, x, zero_normalized, args.return_weight, args.move_weight
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(x)
            val_metrics = reconstruct_metrics(model, data['val'], norm, device)
            print(
                f'Epoch {epoch + 1}/{args.epochs} | Train objective {total / len(data["train"]):.6f}'
                f' | Val corr {val_metrics["corr"]:.5f}'
                f' | Val return std {val_metrics["recon_std"]:.8g}',
                flush=True,
            )
        test_metrics = reconstruct_metrics(model, data['test'], norm, device)
        print('Test metrics (do not tune using test set):')
        for key, value in test_metrics.items():
            print(f'  {key}: {value:.10g}')
        ckpt = output_dir / f'{variant}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_version': 'v4.2', 'dataset_version': 'v4',
            'stage': 'embedding', 'variant': variant,
            'return_scale': float(norm.return_scale),
            'return_mean': float(norm.return_mean),
            'return_std': float(norm.return_std),
            'epochs': args.epochs, 'seed': args.seed,
        }, ckpt)
        print('Saved:', ckpt)
    print('\nStage-1 comparison completed. No Stage 2/3 performed.')


if __name__ == '__main__':
    main()
