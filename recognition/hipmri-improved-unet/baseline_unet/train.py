import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from baseline_unet.modules import UNet


#Using the same validation function as that of the improved unet for fair comparison
def validate(model, loader, criterion, device, num_classes=6):
    model.eval()
    total_loss = 0.0

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    predictions_count = torch.zeros(num_classes, dtype=torch.float64)
    targets_count = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)

            for class_id in range(num_classes):
                pred = predictions == class_id
                target = masks == class_id

                intersections[class_id] += (pred & target).sum().item()
                predictions_count[class_id] += pred.sum().item()
                targets_count[class_id] += target.sum().item()

    dice_scores = {} # changing into dictionary to store seperate scores
    iou_scores = {}

    for class_id in range(1, num_classes):
        intersection = intersections[class_id].item()
        pred_count = predictions_count[class_id].item()
        target_count = targets_count[class_id].item()

        denominator = pred_count + target_count
        union = denominator - intersection

        if denominator > 0:
            dice_scores[class_id] = 2 * intersection / denominator
            iou_scores[class_id] = intersection / union
        else:
            dice_scores[class_id] = None
            iou_scores[class_id] = None

    foreground_dice = [
    dice_scores[c] for c in range(1, num_classes)
    if dice_scores[c] is not None
    ]

    foreground_iou = [
    iou_scores[c] for c in range(1, num_classes)
    if iou_scores[c] is not None
    ]

    mean_dice = sum(foreground_dice) / len(foreground_dice) if foreground_dice else 0.0
    mean_iou = sum(foreground_iou) / len(foreground_iou) if foreground_iou else 0.0
    return total_loss / len(loader.dataset), mean_dice, mean_iou, dice_scores, iou_scores

#model training function
def train_model(args):
    # Select GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Initialise the baseline U-Net with six classes
    model = UNet(in_channels=1, num_classes=6).to(device)

    # Load training and validation datasets
    data_root = Path(args.data_root)

    train_dataset = HipMRIDataset(
        data_root / "keras_slices_train",
        data_root / "keras_slices_seg_train"
    )

    val_dataset = HipMRIDataset(
        data_root / "keras_slices_validate",
        data_root / "keras_slices_seg_validate"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Loss function for six-class segmentation
    criterion = nn.CrossEntropyLoss()

    # Adam optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    # Track the best validation loss
    best_val_loss = float("inf")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train the model for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Reset gradients from the previous batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate training loss
            loss = criterion(outputs, masks)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            total_train_loss += loss.item() * images.size(0)
        # Calculate average training loss for this epoch
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Evaluate the model on the validation dataset
        val_loss, val_dice, val_iou, dice_scores, iou_scores = validate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=6
        )

        # Display training and validation results
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Dice: {val_dice:.4f} "
            f"Val IoU: {val_iou:.4f}"
        )

        # Save the model with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou
            }

            torch.save(
                checkpoint,
                output_dir / "best_unet.pth"
            )

            print("Best baseline U-Net checkpoint saved")

#Adding the main function
def main():
    parser = argparse.ArgumentParser(
        description="Train baseline 2D U-Net on HipMRI"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints_baseline"
    )

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()