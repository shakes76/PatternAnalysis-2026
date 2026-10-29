import os
import sys
import time
from pathlib import Path
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import local dataset and model modules
from dataset import get_dataloaders, NUM_CLASSES, TARGET_PROSTATE_LABEL
from modules import Baseline_UNet, Improved_UNet

# Configurations & Hyperparameters
SEED = 42 
BATCH_SIZE = 8 
NUM_EPOCHS = 30 
LEARNING_RATE = 1e-3 
WEIGHT_DECAY = 1e-4 
NUM_WORKERS = 1 
DICE_SMOOTH = 1e-5

# Path configurations anchored to this file's directory
PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

# Model selector: "baseline" or "improved"
MODEL_NAME = "improved"

if MODEL_NAME == "improved":
    BEST_MODEL_PATH = CHECKPOINT_DIR / "improved_unet_best.pth"
    HISTORY_PATH = CHECKPOINT_DIR / "improved_training_history.csv"
else:
    BEST_MODEL_PATH = CHECKPOINT_DIR / "baseline_unet_best.pth"
    HISTORY_PATH = CHECKPOINT_DIR / "baseline_training_history.csv"


def set_seed(seed=SEED):
    """Enforce reproducibilityacross numpy, random, and PyTorch backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Loss functions and metrices
class SoftDiceLoss(nn.Module):
    """Multiclass Soft Dice Loss with Laplace smoothing.
       Optimizes spatial overlap across all segmentation classes.
    """
    def __init__(self, num_classes=NUM_CLASSES, smooth=DICE_SMOOTH):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """logits: (B, C, H, W) raw unnormalized model outputs
           targets: (B, H, W) ground truth interger label map (0..C-1)
        """
        probs = F.softmax(logits, dim=1)
        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Calculate Dice over the batch and spatial dimensions.
        dims = (0, 2, 3)
        
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs ** 2 + targets_one_hot ** 2, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Average loss across all classes
        return 1.0 - torch.mean(dice)

class CombinedLoss(nn.Module):
    """Hybrid loss: Cross-Entropy + Soft Dice Loss.
       Cross Entropy provides stable gradient flow, while Dice counters class imbalance.
    """
    def __init__(self, num_classes=NUM_CLASSES, smooth=DICE_SMOOTH):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss(num_classes=num_classes, smooth=smooth)

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return loss_ce + loss_dice, loss_ce, loss_dice

# Hard Dice matric
def compute_dice_score(preds, targets, class_idx, smooth=DICE_SMOOTH):
    """Computes hard Dice Similarity Coefficient (DSC) for a spcific class.
       preds: (B, H, W) predicted integer class labels
       targets: (B, H, W) ground-truth integer class labels
    """
    pred_mask = (preds == class_idx).float()
    target_mask = (targets == class_idx).float()

    intersection = torch.sum(pred_mask * target_mask).item()
    total = (torch.sum(pred_mask) + torch.sum(target_mask)).item()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection + smooth) / (total + smooth)

# Train & Validation loops:
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Executes one training epoch cross all training batches.
    """
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_samples = 0

    for batch_index, (images, masks) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        # Clear gradient from the previous batch
        optimizer.zero_grad()
        # Predict class scores for each pixel
        logits = model(images)
        # Calculate combined loss
        loss, ce, d_loss = criterion(logits, masks)

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

        # Weight batch averages by batch size
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_dice_loss += d_loss.item() * batch_size
        total_samples += batch_size

        if batch_index % 100 == 0 or batch_index == len(loader):
            print(
                f"\r Training batch {batch_index}/{len(loader)}"
                f" | Loss: {loss.item():.4f}",
                end="\n" if batch_index == len(loader) else "",
                flush=True,
            )

    return (
        total_loss / total_samples,
        total_ce / total_samples,
        total_dice_loss / total_samples
    )

# Validate for one epoch
@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion, 
    device,
    prostate_label=TARGET_PROSTATE_LABEL,
    smooth=DICE_SMOOTH
):
    """Calculate validation loss and global prostate DSC."""
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0
    total_samples = 0

    # Accumulate piel counts over the entire validation set
    total_intersection = 0.0
    total_cardinality = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        logits = model(images)
        loss, ce, dice_loss = criterion(logits, masks)

        total_loss += loss.item() * batch_size
        total_ce += ce.item() * batch_size
        total_dice_loss += dice_loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        pred_mask = (preds == prostate_label)
        target_mask = (masks == prostate_label)

        total_intersection += (pred_mask & target_mask).sum().item()
        total_cardinality += pred_mask.sum().item() + target_mask.sum().item()

    if total_samples == 0:
        raise ValueError("Validate DataLoader is empty.")

    if total_cardinality == 0:
        prostate_dice = 1.0 if total_intersection == 0 else 0.0
    else:
        prostate_dice = (2.0 * total_intersection + smooth) / (total_cardinality + smooth)

    return (
        total_loss / total_samples,
        total_ce / total_samples,
        total_dice_loss / total_samples,
        prostate_dice
    )

# Save training history
def save_history(history, csv_path=HISTORY_PATH):
    """Write epoch-by-epoch training and validation 
       results to a CSV file.
       Arguments:
       history (dict): Dictionary with lists of epoch metrics.
       csv_path (str or Path): Path to output CSV file.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(history.keys())
    num_epochs = len(history["epoch"])

    with csv_path.open(mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_epochs):
            row = {key: history[key][i] for key in fieldnames}
            writer.writerow(row)

    print(f"[Info] Training metrics history saved to {csv_path}")

def main(): 
    print("=" * 65) 
    print(f"STARTING TRAINING: {MODEL_NAME.upper()} 2D U-NET (HipMRI)") 
    print("=" * 65)

    # Reproducibility 
    set_seed(SEED)

    # Device ocnfiguration
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Output directories
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load DataLoaders from dataset.py
    print(f"\nInitializing DataLoaders (Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS})...")
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Initialize Models from modules.py
    if MODEL_NAME == "improved":
        print("\n[Architecure] Instantiating Improved 2D U-Net (residualBlocks)...")
        model = Improved_UNet(
            in_channels=1, 
            num_classes=NUM_CLASSES,
            base_filters=16
        ).to(device)
    else:
        print("\n[Architecure] Instantiating Baseline 2D U-Net...")
        model = Baseline_UNet(
            in_channels=1, 
            num_classes=NUM_CLASSES,
            base_filters=16
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,}")

    # Loss Function, Optimizer & Scheduler
    criterion = CombinedLoss(num_classes=NUM_CLASSES, smooth=DICE_SMOOTH).to(device) 
    optimizer = AdamW( 
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY 
    ) 
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Metrics tracking
    best_prostate_dice = -1.0
    history = { 
        "epoch": [], 
        "train_loss": [], 
        "train_ce": [], 
        "train_dice_loss": [], 
        "val_loss": [], 
        "val_ce": [], 
        "val_dice_loss": [], 
        "val_prostate_dice": [], 
        "learning_rate": []
    }

    start_time = time.time()

    # Main Epoch Loop:
    for epoch in range(1, NUM_EPOCHS + 1): 
        epoch_start = time.time() 
        
        # Train phase 
        train_loss, train_ce, train_dice_loss = train_one_epoch( 
            model, train_loader, criterion, optimizer, device 
        )

        # Validation phase
        val_loss, val_ce, val_dice_loss, val_prostate_dice = validate_one_epoch( 
            model, val_loader, criterion, device, 
            prostate_label=TARGET_PROSTATE_LABEL,
            smooth=DICE_SMOOTH
        )

        # Step Learning rate:
        # Record the learning rate used during this epoch
        current_lr = optimizer.param_groups[0]["lr"]
        # Update the learning rate for the next epoch
        scheduler.step()
        epoch_duration = time.time() - epoch_start

        # Record metrics 
        history["epoch"].append(epoch) 
        history["train_loss"].append(train_loss) 
        history["train_ce"].append(train_ce) 
        history["train_dice_loss"].append(train_dice_loss) 
        history["val_loss"].append(val_loss) 
        history["val_ce"].append(val_ce) 
        history["val_dice_loss"].append(val_dice_loss)
        history["val_prostate_dice"].append(val_prostate_dice) 
        history["learning_rate"].append(current_lr)

        print( 
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({epoch_duration:.1f}s, lr={current_lr:.2e}) | " 
            f"Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, DiceLoss: {train_dice_loss:.4f}) | " 
            f"Val Loss: {val_loss:.4f} | " f"Val Prostate DSC: {val_prostate_dice:.4f}" 
        )

        # Checkpoint: Save model weights whenever validation Prostate DSC improves
        if val_prostate_dice > best_prostate_dice:
            best_prostate_dice = val_prostate_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_prostate_dice": best_prostate_dice,
                    "num_classes": NUM_CLASSES,
                    "prostate_label": TARGET_PROSTATE_LABEL
                },
                BEST_MODEL_PATH
            )
            print(
                f"  --> [*] Model checkpoint saved! New Best Prostate DSC:"
                f" {best_prostate_dice:.4f}"
            )

        # Save metrics after every epoch.
        save_history(history, HISTORY_PATH)

    # Training Summary - outside the epoch loop
    elapsed_mins = (time.time() - start_time) / 60.0 

    print("\n" + "=" * 65)
    print(f"TRAINING COMPLETE in {elapsed_mins:.2f} minutes")
    print(f"Highest Validation Prostate DSC: {best_prostate_dice:.4f}")
    print(f"Best weights preserved at: {BEST_MODEL_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    main()