import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet, ImprovedUNet


NUM_CLASSES = 6
CLASS_IDS = list(range(NUM_CLASSES))


def set_seed(seed):
    """Set random seeds to make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(outputs, masks):
    """
    Count predicted pixels, target pixels and intersections per class.

    These counts are combined across validation batches before
    calculating Dice and IoU scores.
    """
    predictions = outputs.argmax(dim=1)

    intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    predicted_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    target_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for class_id in CLASS_IDS:
        predicted = predictions == class_id
        actual = masks == class_id

        intersections[class_id] = (predicted & actual).sum().item()
        predicted_counts[class_id] = predicted.sum().item()
        target_counts[class_id] = actual.sum().item()

    return intersections, predicted_counts, target_counts


def evaluate(model, loader, criterion, device, max_batches=None):
    """
    Evaluate the model using validation loss, Dice and IoU.

    Metrics are calculated from pixel counts across the selected
    validation batches. Classes absent from both prediction and
    ground truth are excluded from the mean.
    """
    model.eval()

    total_loss = 0.0
    total_images = 0

    intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    predicted_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    target_counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_images += batch_size

            # Collect pixel counts for each segmentation class.
            batch_intersections, batch_predicted, batch_target = (
                calculate_metrics(outputs, masks)
            )

            intersections += batch_intersections
            predicted_counts += batch_predicted
            target_counts += batch_target

    dice_scores = []
    iou_scores = []

    for class_id in CLASS_IDS:
        intersection = intersections[class_id].item()
        predicted = predicted_counts[class_id].item()
        target = target_counts[class_id].item()

        denominator = predicted + target
        union = denominator - intersection

        # Avoid dividing by zero when a class is absent.
        if denominator == 0:
            dice = float("nan")
        else:
            dice = 2.0 * intersection / denominator

        if union == 0:
            iou = float("nan")
        else:
            iou = intersection / union

        dice_scores.append(dice)
        iou_scores.append(iou)

    # Average across classes, including background when present.
    mean_dice = float(np.nanmean(dice_scores))
    mean_iou = float(np.nanmean(iou_scores))

    return {
        "loss": total_loss / max(total_images, 1),
        "dice": dice_scores,
        "iou": iou_scores,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
    }


def validate_masks(masks):
    """Check that segmentation masks contain labels from 0 to 5."""
    minimum = masks.min().item()
    maximum = masks.max().item()

    if minimum < 0 or maximum >= NUM_CLASSES:
        raise ValueError(
            f"Invalid mask labels: min={minimum}, max={maximum}. "
            f"Expected labels from 0 to {NUM_CLASSES - 1}."
        )


class CombinedDiceCELoss(nn.Module):
    """
    Combine cross-entropy loss with multiclass soft Dice loss.

    Cross-entropy uses all six classes. The Dice component
    excludes background to focus on anatomical regions.
    """

    def __init__(self, num_classes=6, dice_weight=1.0):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # Convert model logits into class probabilities.
        probabilities = torch.softmax(logits, dim=1)

        # Convert integer mask labels into one-hot class masks.
        one_hot = torch.nn.functional.one_hot(
            targets.long(),
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).float()

        # Exclude background (class 0) from the Dice component.
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

        dims = (0, 2, 3)

        intersection = (probabilities * one_hot).sum(dim=dims)
        denominator = (
            probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
        )

        # Smoothing prevents division by zero.
        smooth = 1e-5
        dice = (2 * intersection + smooth) / (denominator + smooth)
        dice_loss = 1 - dice.mean()

        return ce_loss + self.dice_weight * dice_loss


def main():
    """Configure and run a baseline or Improved U-Net experiment."""

    parser = argparse.ArgumentParser(
        description="Train a U-Net model on the HipMRI dataset."
    )

    # Model and training settings
    parser.add_argument(
        "--model",
        choices=["baseline", "improved"],
        default="baseline",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "ce_dice"],
        default="ce",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)

    # Optional limits for short pipeline tests.
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create a separate output folder for each experiment.
    if args.run_name is None:
        args.run_name = f"{args.model}_{int(time.time())}"

    output_dir = Path("runs") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Run: {args.run_name}", flush=True)

    # Load the supplied training and validation splits.
    train_dataset = HipMRIDataset(split="train")
    val_dataset = HipMRIDataset(split="validate")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Select the baseline or residual U-Net architecture.
    if args.model == "baseline":
        model = UNet(num_classes=NUM_CLASSES)
    else:
        model = ImprovedUNet(num_classes=NUM_CLASSES)

    model = model.to(device)

    # Select the loss function used for this experiment.
    if args.loss == "ce_dice":
        criterion = CombinedDiceCELoss(num_classes=NUM_CLASSES)
    else:
        criterion = nn.CrossEntropyLoss()

    print(f"Loss function: {args.loss}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Training images: {len(train_dataset)}", flush=True)
    print(f"Validation images: {len(val_dataset)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Classes: {NUM_CLASSES}", flush=True)
    print(f"Parameters: {parameter_count}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Learning rate: {args.lr}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    # Save training and validation metrics after each epoch.
    csv_path = output_dir / "metrics.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "mean_dice",
        "mean_iou",
        "peak_gpu_memory_mb",
    ]

    for class_id in CLASS_IDS:
        fieldnames.append(f"dice_class_{class_id}")
        fieldnames.append(f"iou_class_{class_id}")

    best_dice = -1.0

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            train_loss = 0.0
            train_images = 0
            epoch_start = time.time()

            # Training phase
            for batch_idx, (images, masks) in enumerate(train_loader):
                if (
                    args.max_train_batches is not None
                    and batch_idx >= args.max_train_batches
                ):
                    break

                validate_masks(masks)

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, masks)

                # Update model weights using backpropagation.
                loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                train_loss += loss.item() * batch_size
                train_images += batch_size

            average_train_loss = train_loss / max(train_images, 1)

            # Validation phase
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                max_batches=args.max_val_batches,
            )

            # Record peak allocated GPU memory for this epoch.
            if device.type == "cuda":
                peak_memory = (
                    torch.cuda.max_memory_allocated() / (1024 ** 2)
                )
            else:
                peak_memory = 0.0

            row = {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "val_loss": val_metrics["loss"],
                "mean_dice": val_metrics["mean_dice"],
                "mean_iou": val_metrics["mean_iou"],
                "peak_gpu_memory_mb": peak_memory,
            }

            for class_id in CLASS_IDS:
                row[f"dice_class_{class_id}"] = (
                    val_metrics["dice"][class_id]
                )
                row[f"iou_class_{class_id}"] = (
                    val_metrics["iou"][class_id]
                )

            writer.writerow(row)
            csv_file.flush()

            elapsed = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {average_train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Mean Dice: {val_metrics['mean_dice']:.4f} | "
                f"Mean IoU: {val_metrics['mean_iou']:.4f} | "
                f"Time: {elapsed:.1f}s",
                flush=True,
            )

            print(
                "Per-class Dice:",
                [
                    round(score, 4)
                    for score in val_metrics["dice"]
                ],
                flush=True,
            )

            # Save the checkpoint with the highest validation Dice.
            if val_metrics["mean_dice"] > best_dice:
                best_dice = val_metrics["mean_dice"]

                torch.save(
                    {
                        "epoch": epoch,
                        "model_name": args.model,
                        "num_classes": NUM_CLASSES,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "mean_dice": best_dice,
                        "seed": args.seed,
                    },
                    output_dir / "best_model.pth",
                )

                print(
                    f"Saved best model (Dice={best_dice:.4f})",
                    flush=True,
                )

    print("Training completed.", flush=True)
    print(f"Best validation mean Dice: {best_dice:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

