"""
train.py, used to train the baseline and improved u-net 2d models

Usage:
python train.py --model unet2d
python train.py --model improvedunet2d
"""

from dataset import build_dataloaders
from modules import UNet2D, ImprovedUNet2D

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")
IMG_DIR = os.path.join(SCRIPT_DIR, "images")

# training loop
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.argmax(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# validation loop
@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=6):
    model.eval()
    total_loss = 0.0

    dice_sum = torch.zeros(num_classes)
    prec_sum = torch.zeros(num_classes)
    rec_sum = torch.zeros(num_classes)
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.argmax(1).to(device)

        logits = model(images)
        total_loss += criterion(logits, labels).item()

        m = per_class_metrics(logits, labels, num_classes)
        dice_sum += m["dice"].cpu()
        prec_sum += m["precision"].cpu()
        rec_sum += m["recall"].cpu()
        n_batches += 1

    return {
        "loss": total_loss / n_batches, "dice": dice_sum / n_batches, 
        "precision": prec_sum / n_batches, "recall": rec_sum / n_batches
    }

def per_class_metrics(logits, target, num_classes=6, eps=1e-6):
    """
    returns dict of per-class dice, precision, recall as tensors
    """

    pred = logits.argmax(1)
    dice = torch.zeros(num_classes)
    prec = torch.zeros(num_classes)
    rec = torch.zeros(num_classes)

    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()

        dice[c] = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        prec[c] = (tp + eps) / (tp + fp + eps)
        rec[c]  = (tp + eps) / (tp + fn + eps)

    return {"dice": dice, "precision": prec, "recall": rec}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet2d", "improvedunet2d"], default="improvedunet2d")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "models"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # data
    train_loader, val_loader, _ = build_dataloaders(base_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    # print("Train batches: ", len(train_loader))
    # print("Validation batches: ", len(val_loader))

    # model
    if args.model == "unet2d":
        model = UNet2D(in_channels=1, num_classes=6).to(device)
    else:
        model = ImprovedUNet2D(in_channels=1, num_classes=6).to(device)

    # weighted loss
    weights = torch.tensor([0.1, 0.1, 0.5, 2.0, 5.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    tracking = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_dice = 0.0

    # peak memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        prostate_dice = val_metrics["dice"][5].item()
        tracking["train_loss"].append(train_loss)
        tracking["val_loss"].append(val_metrics["loss"])
        tracking["val_dice"].append(prostate_dice)

        print(f"Epoch: ", epoch, "/", args.epochs)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics["loss"]:.4f}")
        print(f"prostate_dice={prostate_dice:.4f}")
        print(f"time={elapsed:.1f}s")

        #save best checkpoint for val_dice
        if prostate_dice > best_val_dice:
            best_val_dice = prostate_dice
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": prostate_dice,
                "args": vars(args),
            }, os.path.join(args.out_dir, f"best_{args.model}.pt"))

    # save final checkpoint
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"final_{args.model}.pt"))

    # peak vram
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak VRAM: {peak:.2f} GB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tracking["train_loss"], label="train")
    axes[0].plot(tracking["val_loss"], label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].set_title("Loss")

    axes[1].plot(tracking["val_dice"], label="val prostate Dice")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].legend(); axes[1].set_title("Prostate Dice")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()