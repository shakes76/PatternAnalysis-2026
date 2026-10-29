"""Train, validate and test the baseline and improved 2D U-Net on HipMRI.

Training pattern adapted from PyTorch's optimisation tutorial:
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
HipMRI-specific Dice loss, padding handling and metrics developed with
ChatGPT assistance. The architecture and dataset are imported, not redefined.
"""

import matplotlib
matplotlib.use("Agg")  # Save plots on Rangpur without a graphical display.
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from pathlib import Path
from dataset import make_dataloaders
from modules import UNet2D, ImprovedUNet2D


# Settings: use the Windows path locally or the Linux path on Rangpur.
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
BASE_CHANNELS = 32
NUM_CLASSES = 6
PROSTATE_CLASS = 5  # Confirm label 5 using official HipMRI label metadata.
MODELS = {
    "baseline": UNet2D,
    "improved": ImprovedUNet2D
}
MAX_BATCHES = None     
TEST_AFTER_TRAINING = False   # Set False while developing; True for final evaluation.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_valid_pixels(batch, masks):
    """Return 1 for original image pixels and 0 for artificial side padding.

    Args:
        batch: Dictionary returned by HipMRIDataset.
        masks: Ground-truth tensor [B, H, W] on DEVICE.
    Returns:
        Float tensor [B, H, W] identifying real image pixels.
    """
    valid = torch.zeros_like(masks, dtype=torch.float32)
    for i, width in enumerate(batch["original_width"]):
        width = int(width)
        left = (masks.shape[-1] - width) // 2
        valid[i, :, left:left + width] = 1
    return valid


def calculate_loss(outputs, masks, onehot, valid):
    """Compute masked cross-entropy plus foreground Dice loss.

    Args:
        outputs: Raw model logits [B, 6, H, W].
        masks: Integer labels [B, H, W].
        onehot: Six-channel labels [B, 6, H, W].
        valid: Real-pixel mask [B, H, W].
    Returns:
        Scalar differentiable PyTorch loss.
    """
    pixel_loss = F.cross_entropy(outputs, masks, reduction="none")
    ce_loss = (pixel_loss * valid).sum() / valid.sum()

    probabilities = torch.softmax(outputs, dim=1) * valid.unsqueeze(1)
    targets = onehot * valid.unsqueeze(1)
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    total = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1e-6) / (total + 1e-6)

    # Ignore background (class 0) and classes missing from this batch.
    present = targets[:, 1:].sum(dim=(0, 2, 3)) > 0
    dice_loss = 1 - dice[1:][present].mean() if present.any() else ce_loss * 0
    return ce_loss + dice_loss


def update_dice_counts(outputs, masks, valid, intersection, total):
    """Add predicted/true pixel counts for whole-split per-class Dice.

    Args:
        outputs: Model logits [B, 6, H, W].
        masks: Integer labels [B, H, W].
        valid: Real-pixel mask [B, H, W].
        intersection: Running intersections for six classes.
        total: Running predicted-plus-target pixel counts for six classes.
    Returns:
        None; updates intersection and total in place.
    """
    predictions = outputs.argmax(dim=1)
    valid = valid.bool()
    with torch.no_grad():
        for c in range(NUM_CLASSES):
            predicted = (predictions == c) & valid
            actual = (masks == c) & valid
            intersection[c] += (predicted & actual).sum()
            total[c] += predicted.sum() + actual.sum()


def run_epoch(model, loader, optimizer=None):
    """Run one epoch; update weights only when optimizer is provided.

    Args:
        model: The imported UNet2D.
        loader: Train, validation or test DataLoader.
        optimizer: Adam for training; None for validation/testing.
    Returns:
        (average_loss, six_per_class_dice_scores).
    """
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    image_count = 0
    intersection = torch.zeros(NUM_CLASSES, device=DEVICE)
    total = torch.zeros(NUM_CLASSES, device=DEVICE)

    for i, batch in enumerate(loader):
        images = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)
        onehot = batch["mask_onehot"].to(DEVICE)
        valid = get_valid_pixels(batch, masks)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = calculate_loss(outputs, masks, onehot, valid)
            if training:
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * images.shape[0]
        image_count += images.shape[0]
        update_dice_counts(outputs, masks, valid, intersection, total)

        if MAX_BATCHES is not None and i + 1 >= MAX_BATCHES:
            break

    # A class absent from a split receives NaN, not a misleading perfect score.
    dice = (2 * intersection / total.clamp_min(1)).cpu().tolist()
    for c in range(NUM_CLASSES):
        if total[c].item() == 0:
            dice[c] = float("nan")
    return loss_sum / image_count, dice


def plot_curves(train_losses, val_losses, train_dices, val_dices, save_dir):
    """Save loss and prostate Dice plots after each epoch."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Combined loss")
    plt.legend()
    plt.savefig(save_dir / "loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, train_dices, label="Training")
    plt.plot(epochs, val_dices, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Prostate Dice")
    plt.legend()
    plt.savefig(save_dir / "dice.png")
    plt.close()


def train_model(model_name, model_class, loaders):
    """Train one U-Net architecture and save its best checkpoint."""

    save_dir = Path("runs") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {model_name} on {DEVICE}")

    model = model_class(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    best_dice = -1.0
    best_val_loss = float("inf")
    best_file = save_dir / "best_model.pt"

    for epoch in range(EPOCHS):
        train_loss, train_scores = run_epoch(
            model, loaders["train"], optimizer
        )

        val_loss, val_scores = run_epoch(
            model, loaders["validate"]
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_dices.append(train_scores[PROSTATE_CLASS])
        val_dices.append(val_scores[PROSTATE_CLASS])

        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val prostate Dice: {val_scores[PROSTATE_CLASS]:.4f}"
        )

        # Select the best checkpoint using validation performance.
        score = val_scores[PROSTATE_CLASS]

        if score != score:  # Handle undefined Dice (NaN).
            score = -1.0

        if score > best_dice or (
            score == best_dice and val_loss < best_val_loss
        ):
            best_dice = score
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_file)
            print("  Saved best model")

        plot_curves(
            train_losses, val_losses,
            train_dices, val_dices, save_dir
        )

    if TEST_AFTER_TRAINING:
        model.load_state_dict(
            torch.load(best_file, map_location=DEVICE, weights_only=True)
        )

        test_loss, test_scores = run_epoch(model, loaders["test"])

        results = f"Test loss: {test_loss:.4f}\n"
        for c, score in enumerate(test_scores):
            results += f"Class {c} Dice: {score:.4f}\n"

        print(results)
        (save_dir / "test_results.txt").write_text(results)

    print(f"Finished training {model_name}. Saved to {save_dir}")


def main():
    """Train both architectures using the same HipMRI data splits."""

    print("Training device:", DEVICE)

    loaders = make_dataloaders(
        DATA_ROOT, batch_size=BATCH_SIZE
    )

    for model_name, model_class in MODELS.items():
        train_model(model_name, model_class, loaders)

if __name__ == "__main__":
    main()
