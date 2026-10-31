"""
Training script for baseline and improved 2D U-Net.

Usage:
    python train.py --model unet2d --epochs 50 --batch-size 16 --out-dir ./runs/unet2d_baseline
    python train.py --model improved_unet2d --epochs 50 --batch-size 16 --out-dir ./runs/improved_unet2d
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_dataloaders, NUM_CLASSES
from modules import UNet2D, ImprovedUNet2D

# class frequencies measured on the training split for every 10th sample
CLASS_FREQS = np.array([0.1748, 0.6700, 0.1285, 0.0183, 0.0037, 0.0047])


def soft_class_weights(freqs, device):
    w = 1.0 / np.sqrt(freqs)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def dice_per_class(logits, target, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    dims = (0, 2, 3) # sum over batch and spacial dimensions
    inter = (probs * target).sum(dims)
    union = probs.sum(dims) + target.sum(dims)
    return ((2 * inter + eps) / (union + eps)).cpu().numpy()


def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, lbls)

        optim.zero_grad()
        loss.backward() # backpropogate to compute gradients
        optim.step() # update weights

        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    dice_sum = np.zeros(NUM_CLASSES)
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        logits = model(imgs)
        total_loss += criterion(logits, lbls).item() * imgs.size(0)
        n += imgs.size(0)
        dice_sum += dice_per_class(logits, lbls) * imgs.size(0)
    return total_loss / n, dice_sum / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders = build_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # build model
    if args.model == "unet2d":
        model = UNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)
    elif args.model == "improved_unet2d":
        model = ImprovedUNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)

    else:
        raise ValueError(f"unknown model: {args.model}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, Parameters: {n_params:,}")

    # cross entropy for class imbalance
    weights = soft_class_weights(CLASS_FREQS, device)
    print(f"Class weights: {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optim, criterion, device)
        val_loss, val_dice = evaluate(model, loaders["val"], criterion, device)
        dt = time.time() - t0

        # log metrics 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice.tolist())

        mean_dice = val_dice.mean()
        prostate_dice = val_dice[5]
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}  "
              f"mean dice {mean_dice:.4f}  prostate dice {prostate_dice:.4f}  "
              f"({dt:.1f}s)")

        # save model cp if pros_dice improves 
        if prostate_dice > best_dice:
            best_dice = prostate_dice
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            print(f"  → saved best.pt (prostate dice {prostate_dice:.4f})")
        
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    # plot curves
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    dice_arr = np.array(history["val_dice"])
    ax[1].plot(dice_arr[:, 5], label="prostate")
    ax[1].plot(dice_arr.mean(axis=1), label="mean")
    ax[1].set_title("Dice"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "curves.png"), dpi=100)
    print(f"Saved curves.png in {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="unet2d", choices=["unet2d", "improved_unet2d"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", type=str, default="./runs")
    args = p.parse_args()
    main(args)