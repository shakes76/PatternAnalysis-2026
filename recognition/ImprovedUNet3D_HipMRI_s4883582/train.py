"""
Train, validate, test and save the 3D Improved U-Net on HipMRI volumes.

The best model is chosen on validation prostate Dice; the test split is only used once at the end.
Training stops early once validation prostate Dice has not improved for PATIENCE epochs.

Usage:
    python train.py
"""
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from modules import ImprovedUNet3D, CombinedLoss
from dataset import (HipMRIVolumeDataset, split_volumes, check_patient_leakage, CLASS_NAMES, NUM_CLASSES,
                     PROSTATE_CLASS)
from utils import overlap_counts, dice_and_iou


# ==========================================
# Output Paths & Hyperparameters
# ==========================================
OUTPUT_PATH = "outputs"
MODEL_PATH = os.path.join(OUTPUT_PATH, "unet3d_best.pt")

EPOCHS = 200
PATIENCE = 40           # early stopping: epochs without a validation prostate Dice improvement
BATCH_SIZE = 1          # full-resolution volumes; 2 fit at half resolution
LEARNING_RATE = 5e-4    # Isensee et al.
LR_DECAY = 0.985        # per-epoch exponential decay, Isensee et al.
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 8         # CPU-side 3D augmentation is 8x heavier at full resolution
SEED = 42


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        # Mixed precision roughly halves activation memory, which dominates for 3D volumes
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """Average loss plus per-class Dice and IoU over the whole loader."""
    model.eval()
    total_loss = 0
    intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    pred_size = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    true_size = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
            total_loss += criterion(outputs, masks).item()

            i, p, t = overlap_counts(outputs, masks, NUM_CLASSES)
            intersection += i
            pred_size += p
            true_size += t

    dice, iou = dice_and_iou(intersection, pred_size, true_size)
    return total_loss / len(loader), dice, iou


def plot_history(history, out_path):
    """Save the training/validation loss curves and validation prostate Dice."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("CE + Dice Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_prostate_dice"])
    axes[1].axhline(0.7, linestyle="--", color="grey", label="Target (0.7)")
    axes[1].set_title("Validation Prostate Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def print_scores(title, dice, iou):
    print(f"\n{title}")
    for name, d, j in zip(CLASS_NAMES, dice, iou):
        print(f"  {name:<10} Dice: {d:.4f} | IoU: {j:.4f}")


def main():
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Patient-level split mirroring the 2D baseline; augmentation on the training split only
    splits = split_volumes()
    print("Loading and downsampling volumes...")
    train_dataset = HipMRIVolumeDataset(splits["train"], augment=True)
    val_dataset = HipMRIVolumeDataset(splits["validate"])
    test_dataset = HipMRIVolumeDataset(splits["test"])
    check_patient_leakage(train=train_dataset, validate=val_dataset, test=test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = ImprovedUNet3D(n_channels=1, n_classes=NUM_CLASSES).to(device)
    criterion = CombinedLoss(NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "val_loss": [], "val_prostate_dice": []}
    best_dice, best_epoch = -1.0, 0

    for epoch in range(EPOCHS):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        prostate_dice = val_dice[PROSTATE_CLASS].item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_prostate_dice"].append(prostate_dice)

        # Keep the checkpoint with the best validation prostate Dice
        if prostate_dice > best_dice:
            best_dice, best_epoch = prostate_dice, epoch + 1
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Prostate Dice: {prostate_dice:.4f} (best {best_dice:.4f}) | "
              f"{time.perf_counter() - start:.0f}s")

        if epoch == 0 and device.type == "cuda":
            print(f"Peak training VRAM: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.1f} MB")

        if epoch + 1 - best_epoch >= PATIENCE:
            print(f"Early stopping: no improvement since epoch {best_epoch}")
            break

    plot_history(history, os.path.join(OUTPUT_PATH, "unet3d_training_curves.png"))

    # Final, one-off evaluation of the best checkpoint on the held-out test split
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    _, test_dice, test_iou = evaluate(model, test_loader, criterion, device)
    print_scores(f"Test set results (best validation checkpoint, epoch {best_epoch}):", test_dice, test_iou)


if __name__ == "__main__":
    main()
