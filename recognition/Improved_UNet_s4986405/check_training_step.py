"""Run one CPU optimizer step on two real training slices; no checkpoint is saved."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import HipMRI2DDataset
from losses import dice_ce_loss
from modules import UNet2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root', type=Path,
        default=Path('/home/groups/comp3710/HipMRI_Study_open/keras_slices_data'),
    )
    args = parser.parse_args()
    torch.manual_seed(3710)
    torch.set_num_threads(1)
    dataset = HipMRI2DDataset(
        args.data_root / 'keras_slices_train',
        args.data_root / 'keras_slices_seg_train',
        normalize=True, one_hot=True, training=True, augment=True,
    )
    if len(dataset) < 2:
        raise ValueError('This check requires at least two training slices')
    images, targets = next(iter(DataLoader(
        Subset(dataset, [0, 1]), batch_size=2, shuffle=False, num_workers=0,
    )))
    model = UNet2D(base_channels=4).cpu().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    if logits.shape != targets.shape or not torch.isfinite(logits).all():
        raise RuntimeError('Invalid model output')
    loss = dice_ce_loss(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss')
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f'Missing or non-finite gradient: {name}')
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError('Non-finite parameter after update')
    if not any(not torch.equal(old, new.detach()) for old, new in zip(before, model.parameters())):
        raise RuntimeError('No parameter changed')
    print('Device:', images.device)
    print('Image batch:', tuple(images.shape))
    print('One-hot targets:', tuple(targets.shape))
    print('Loss before update:', loss.item())
    print('Finite gradients: PASS')
    print('Parameter update: PASS')
    print('Training step smoke check: PASS')
    print('One CPU step only; no checkpoint or accuracy result.')


if __name__ == '__main__':
    main()
