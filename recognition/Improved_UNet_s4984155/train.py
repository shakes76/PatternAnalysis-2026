"""
Training pipeline for 2D prostate segmentation on HipMRI.
Supports both ImprovedUNet (with deep supervision) and BaselineUNet.
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from modules import ImprovedUNet, BaselineUNet
from dataset import get_data_loaders, HipMRI2DSliceDataset


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        denominator = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super(CombinedBCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-5
) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total_pred = preds_flat.sum().item()
    total_target = targets_flat.sum().item()

    if total_target == 0 and total_pred == 0:
        return 1.0, 1.0
    if total_target == 0 and total_pred > 0:
        return 0.0, 0.0

    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    union = total_pred + total_target - intersection
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 1,
    total_epochs: int = 50,
    use_deep_supervision: bool = True
) -> float:
    model.train()
    running_loss = 0.0
    total_batches = len(loader)
    log_interval = max(1, total_batches // 4)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_deep_supervision and isinstance(model, ImprovedUNet):
            out1, out2, out3 = model(images, enable_deep_supervision=True)
            loss1 = criterion(out1, masks)
            loss2 = criterion(out2, masks)
            loss3 = criterion(out3, masks)
            loss = loss1 + 0.5 * loss2 + 0.25 * loss3
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            print(f"  [Ep {epoch:02d}/{total_epochs:02d}] Step [{batch_idx+1:03d}/{total_batches:03d}] Loss: {loss.item():.4f}", flush=True)

    return running_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_samples = len(loader)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            b_dice, b_iou = compute_metrics(outputs, masks)
            total_dice += b_dice
            total_iou += b_iou

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice = total_dice / num_samples
    mean_iou = total_iou / num_samples
    return epoch_loss, mean_dice, mean_iou


def plot_training_history(history: Dict, save_path: str, model_name: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    axes[0].set_title(f"{model_name}: Combined BCE + Dice Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, history["val_dice"], label="Val Dice", color="forestgreen", lw=2)
    axes[1].plot(epochs, history["val_iou"], label="Val IoU", color="darkorange", lw=2, linestyle="--")
    axes[1].axhline(y=0.75, color="gray", linestyle="-.", label="Target (0.75)")
    axes[1].set_title(f"{model_name}: Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Convergence plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on HipMRI 2D Prostate Data")
    parser.add_argument("--model", type=str, default="improved", choices=["improved", "baseline"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--no_deep_supervision", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model.upper()} | Epochs: {args.epochs} | Batch: {args.batch_size}", flush=True)

    if args.data_dir:
        data_path = args.data_dir
    else:
        default_hpc_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
        default_local_path = "./keras_slices_data"
        data_path = default_hpc_path if os.path.exists(default_hpc_path) else default_local_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Specify via --data_dir.")

    train_loader, val_loader, test_loader = get_data_loaders(
        data_root=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(256, 128),
        prostate_only=True
    )

    if args.model == "improved":
        model = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
        model_name = "Improved_UNet"
        use_deep_supervision = not args.no_deep_supervision
    else:
        model = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)
        model_name = "Baseline_UNet"
        use_deep_supervision = False

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {model_name} ({num_params:,} parameters)", flush=True)

    criterion = CombinedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dice = 0.0
    checkpoint_path = os.path.join(args.save_dir, f"best_{args.model}_unet.pth")
    plot_path = os.path.join(args.save_dir, f"training_curves_{args.model}.png")

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch, total_epochs=args.epochs, use_deep_supervision=use_deep_supervision)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        ep_duration = time.time() - ep_start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}", flush=True)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Best model saved (Val Dice: {val_dice:.4f})", flush=True)

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} mins. Best Val Dice: {best_val_dice:.4f}", flush=True)

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)
        print(f"Peak VRAM: {peak_vram:.2f} MB", flush=True)

    plot_training_history(history, plot_path, model_name)


if __name__ == "__main__":
    main()
