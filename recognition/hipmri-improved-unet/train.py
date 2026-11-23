import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import ImprovedUNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        if masks.min().item() < 0 or masks.max().item() >= 6:
            raise ValueError("Invalid segmentation labels detected")

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", type=str, default=
        "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    if device.type != "cuda":
        raise RuntimeError("GPU not available. Submit training through Slurm.")

    root = Path(args.data_root)

    train_dataset = HipMRIDataset(
        root / "keras_slices_train",
        root / "keras_slices_seg_train"
    )

    val_dataset = HipMRIDataset(
        root / "keras_slices_validate",
        root / "keras_slices_seg_validate"
    )

    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    print("Training samples:", len(train_dataset), flush=True)
    print("Validation samples:", len(val_dataset), flush=True)

    model = ImprovedUNet(in_channels=1, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_dice, val_iou, class_dice, class_iou = validate(
            model, val_loader, criterion, device
            )

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Dice: {val_dice:.4f} "
            f"IoU: {val_iou:.4f}",
            flush=True
        )

        # Print individual class Dice scores
        for class_id, score in class_dice.items():
            if score is not None:
                print(f"Class {class_id} Dice: {score:.4f}", flush=True)
            else:
                print(f"Class {class_id} Dice: N/A", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": val_iou
            }, output_dir / "best_improved_unet.pth")

            print("Best model checkpoint saved!", flush=True)


if __name__ == "__main__":
    main()
