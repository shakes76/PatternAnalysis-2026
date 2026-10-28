"""
Training, validation, and testing script for 2D HipMRI Prostate Segmentation.

Features:
- Supports both BaselineUNet2D and ImprovedUNet2D architectures.
- Optimizes CombinedLoss (Cross-Entropy + Multi-Class Soft Dice with higher weight on Prostate).
- Profiles parameter counts, peak GPU VRAM, training time, and per-slice inference latency.
- Automatically plots and saves loss and prostate DSC learning curves.
"""

import os
import time
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Mandatory COMP3710 modular imports
from modules import BaselineUNet2D, ImprovedUNet2D, CombinedLoss, compute_dice_iou
from dataset import get_hipmri_dataloaders

CLASS_NAMES = [
    "Background (0)",
    "Body Outline (1)",
    "Bone (2)",
    "Bladder (3)",
    "Rectum (4)",
    "Prostate (5)"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D U-Net / Improved U-Net on HipMRI")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data",
        help="Path to HipMRI 2D Nifti slices directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="improved",
        choices=["baseline", "improved"],
        help="Model architecture: 'baseline' (2D U-Net) or 'improved' (2D Improved U-Net)"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output folder for plots & weights")
    parser.add_argument("--dummy", action="store_true", help="Run with synthetic data for local smoke test")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=6):
    model.eval()
    total_loss = 0.0
    accum_dices = torch.zeros(num_classes, device=device)
    accum_ious = torch.zeros(num_classes, device=device)
    total_samples = 0
    start_infer = time.time()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        bs = images.size(0)
        total_loss += loss.item() * bs

        # Convert logits to hard one-hot predictions
        preds_idx = torch.argmax(logits, dim=1, keepdim=True)
        preds_onehot = torch.zeros_like(masks).scatter_(1, preds_idx, 1.0)

        batch_dices, batch_ious = compute_dice_iou(preds_onehot, masks)
        accum_dices += batch_dices * bs
        accum_ious += batch_ious * bs
        total_samples += bs

    infer_elapsed = time.time() - start_infer
    latency_ms = (infer_elapsed / max(1, total_samples)) * 1000.0

    avg_loss = total_loss / max(1, total_samples)
    mean_dices = (accum_dices / max(1, total_samples)).cpu()
    mean_ious = (accum_ious / max(1, total_samples)).cpu()
    return avg_loss, mean_dices, mean_ious, latency_ms


def save_learning_curves(history, model_name, output_dir):
    plt.figure(figsize=(11, 4.5))

    # Subplot 1: Train & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss (CE + Soft Dice)")
    plt.title(f"{model_name.upper()} - Loss Trajectory")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Subplot 2: Validation Prostate & Macro Foreground Dice
    plt.subplot(1, 2, 2)
    plt.plot(history["val_prostate_dice"], label="Val Prostate DSC (Class 5)", color="forestgreen", linewidth=2)
    plt.plot(history["val_fg_dice"], label="Val Foreground Mean DSC (Classes 1-5)", color="royalblue", linestyle="-.")
    plt.axhline(y=0.75, color="crimson", linestyle="--", label="Target Threshold (0.75)")
    plt.xlabel("Epoch")
    plt.ylabel("Sørensen-Dice Coefficient")
    plt.ylim(0.0, 1.02)
    plt.title(f"{model_name.upper()} - Validation DSC Trajectory")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model_name}_learning_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[Plot] Learning curves saved to: {plot_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Compute Device: {device}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    # Automatically fallback to dummy mode if data_dir does not exist locally
    use_dummy = args.dummy or (not os.path.exists(args.data_dir))
    if use_dummy and not args.dummy:
        print(f"[Notice] '{args.data_dir}' not found locally. Switching to synthetic DUMMY mode.")

    train_loader, val_loader, test_loader = get_hipmri_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0 if use_dummy else args.num_workers,
        dummy_mode=use_dummy
    )

    # Initialize model architecture
    if args.model == "improved":
        model = ImprovedUNet2D(in_channels=1, num_classes=6, base_filters=16).to(device)
    else:
        model = BaselineUNet2D(in_channels=1, num_classes=6, base_filters=32).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Architecture: {args.model.upper()} | Trainable Parameters: {num_params:,}")

    # Assign higher class weight to small structures (Bladder, Rectum, especially Prostate <2% volume)
    class_weights = torch.tensor([0.2, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.float32)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_prostate_dice": [],
        "val_fg_dice": []
    }

    best_prostate_dice = -1.0
    ckpt_path = os.path.join(args.output_dir, f"best_{args.model}_unet.pth")
    train_start = time.time()

    print(f"\n--- Starting Training ({args.model.upper()}) for {args.epochs} Epochs ---")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_dices, _, _ = evaluate(model, val_loader, criterion, device)

        prostate_dsc = va_dices[5].item()
        fg_mean_dsc = va_dices[1:].mean().item()
        scheduler.step(prostate_dsc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_prostate_dice"].append(prostate_dsc)
        history["val_fg_dice"].append(fg_mean_dsc)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
            f"Val Prostate DSC: {prostate_dsc:.4f} | Val FG Mean DSC: {fg_mean_dsc:.4f}"
        )

        if prostate_dsc >= best_prostate_dice:
            best_prostate_dice = prostate_dsc
            torch.save(model.state_dict(), ckpt_path)

    total_time = time.time() - train_start
    print(f"\n[Training Complete] Elapsed: {total_time:.1f}s | Best Val Prostate DSC: {best_prostate_dice:.4f}")

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"[Resource Profile] Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB")

    # Save training curves
    save_learning_curves(history, args.model, args.output_dir)

    # Final evaluation on held-out test set using best checkpoint
    print(f"\n--- Evaluating Best {args.model.upper()} Checkpoint on Held-Out Test Split ---")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_loss, test_dices, test_ious, test_latency_ms = evaluate(model, test_loader, criterion, device)

    print(f"Test Split Loss          : {test_loss:.4f}")
    print(f"Inference Latency        : {test_latency_ms:.2f} ms / slice")
    print("-" * 55)
    print(f"{'Anatomical Structure':<22} | {'Sørensen-Dice (DSC)':<20} | {'IoU':<10}")
    print("-" * 55)
    for idx, cls_name in enumerate(CLASS_NAMES):
        print(f"{cls_name:<22} | {test_dices[idx].item():<20.4f} | {test_ious[idx].item():<10.4f}")
    print("-" * 55)


if __name__ == "__main__":
    main()