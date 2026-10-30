"""Train and validate the HipMRI 2D CAN/U-Net.

All paths and training hyperparameters are configured in config.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

import config
from dataset import HipMRI_Dataset
from modules import UNet, CAN
import time

PROSTATE_CLASS = 5


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Run one epoch; return loss and per-class Dice scores."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_images = 0
    intersection = torch.zeros(config.NUM_CLASSES, device=device)
    pred_count = torch.zeros(config.NUM_CLASSES, device=device)
    target_count = torch.zeros(config.NUM_CLASSES, device=device)

    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Check masks have class IDs in [0, NUM_CLASSES).
            if masks.min() < 0 or masks.max() >= config.NUM_CLASSES:
                raise ValueError("Mask labels must be integer class IDs 0..NUM_CLASSES-1")

            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_images += images.size(0)

            # Hard predictions are used only for the evaluation Dice metric.
            predictions = logits.argmax(dim=1)
            for class_id in range(config.NUM_CLASSES):
                predicted = predictions == class_id
                actual = masks == class_id
                intersection[class_id] += (predicted & actual).sum().item()
                pred_count[class_id] += predicted.sum().item()
                target_count[class_id] += actual.sum().item()

    denom = pred_count + target_count
    # Missing classes are reported as NaN, not a misleading perfect score.
    dice = 2 * intersection / denom.clamp_min(1)
    dice[denom == 0] = float("nan")  # Class absent from both prediction and truth.
    return total_loss / total_images, dice.cpu()

def main():
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    train_dataset = HipMRI_Dataset(config.TRAIN_DIRECTORY, config.TRAIN_SEG_DIRECTORY)
    val_dataset = HipMRI_Dataset(config.VALIDATION_DIRECTORY, config.VALIDATION_SEG_DIRECTORY)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = CAN(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        hidden_channels=config.HIDDEN_CHANNELS,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    checkpoint_path = Path(config.MODEL_PATH)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)


        # Mean foreground Dice excludes background and classes absent from both masks.
        mean_dice = torch.nanmean(val_dice[1:]).item()
        prostate_dice = val_dice[PROSTATE_CLASS].item()
        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
            f"Prostate Dice: {prostate_dice:.4f} | Mean foreground Dice: {mean_dice:.4f}"
        )
        print(f"  Dice by class: {[round(value, 4) for value in val_dice.tolist()]}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best weights: {checkpoint_path}")

    elapsed = time.perf_counter() - start_time

    print(f"Training time: {elapsed:.2f} seconds")

    if torch.cuda.is_available():
        memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak allocated GPU memory: {memory:.2f} MiB")
    else:
        print("GPU memory usage: N/A (running on CPU)")
    plt.plot(history["train"], label="Training")
    plt.plot(history["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = Path(config.LOSS_PLOT_PATH)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss plot: {plot_path}")


if __name__ == "__main__":
    main()
