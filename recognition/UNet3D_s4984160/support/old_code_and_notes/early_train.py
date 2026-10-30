"""Train the standard 3D U-Net baseline on HipMRI."""

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import HipMRIDataset, get_data_pairs, split_by_patient
from modules import UNet3D


NUM_CLASSES = 6
TARGET_SHAPE = (64, 128, 128)

LABEL_NAMES = [
    "Background",
    "Body",
    "Bones",
    "Bladder",
    "Rectum",
    "Prostate",
]


class DiceCELoss(nn.Module):
    """Combine Cross-Entropy and foreground Soft Dice losses."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)

        # [B, 6, D, H, W]
        probabilities = torch.softmax(logits, dim=1)

        # Ground truth: [B, D, H, W] -> [B, 6, D, H, W]
        one_hot = F.one_hot(
            targets.long(),
            num_classes=NUM_CLASSES,
        ).movedim(-1, 1).to(dtype=probabilities.dtype)

        # Exclude background (class 0).
        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

        dimensions = (0, 2, 3, 4)

        intersection = (probabilities * one_hot).sum(dimensions)
        denominator = probabilities.sum(dimensions) + one_hot.sum(dimensions)

        smooth = 1e-5
        dice_per_class = (
            2.0 * intersection + smooth
        ) / (denominator + smooth)

        dice_loss = 1.0 - dice_per_class.mean()

        return 0.5 * ce_loss + 0.5 * dice_loss


def calculate_case_metrics(predictions, targets):
    """Calculate six-class Dice/IoU and voxel counts for one case."""
    predictions = predictions.reshape(-1).long()
    targets = targets.reshape(-1).long()

    if predictions.numel() != targets.numel():
        raise ValueError("Prediction and target sizes differ.")

    if ((targets < 0) | (targets >= NUM_CLASSES)).any():
        raise ValueError("Invalid ground-truth class ID.")

    encoded = targets * NUM_CLASSES + predictions

    confusion = torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    confusion = confusion.cpu()

    gt_counts = confusion.sum(dim=1)
    pred_counts = confusion.sum(dim=0)

    records = []

    for class_id in range(NUM_CLASSES):
        tp = int(confusion[class_id, class_id].item())
        gt = int(gt_counts[class_id].item())
        pred = int(pred_counts[class_id].item())

        fp = pred - tp
        fn = gt - tp

        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn

        # Skip only when GT and prediction are both empty.
        dice = (
            2.0 * tp / dice_denominator
            if dice_denominator > 0 else float("nan")
        )
        iou = (
            tp / iou_denominator
            if iou_denominator > 0 else float("nan")
        )

        records.append({
            "class_id": class_id,
            "class_name": LABEL_NAMES[class_id],
            "dice": dice,
            "iou": iou,
            "gt_voxels": gt,
            "pred_voxels": pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

    return records


def summarise_metrics(case_records):
    """Average per-case scores and sum voxel counts."""
    summary = []

    for class_id in range(NUM_CLASSES):
        rows = [
            case[class_id] for case in case_records
        ]

        valid = [
            row for row in rows
            if row["dice"] == row["dice"]
        ]

        mean_dice = (
            sum(row["dice"] for row in valid) / len(valid)
            if valid else float("nan")
        )
        mean_iou = (
            sum(row["iou"] for row in valid) / len(valid)
            if valid else float("nan")
        )

        summary.append({
            "class_id": class_id,
            "class_name": LABEL_NAMES[class_id],
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "valid_cases": len(valid),
            "gt_voxels": sum(row["gt_voxels"] for row in rows),
            "pred_voxels": sum(row["pred_voxels"] for row in rows),
        })

    return summary


def run_epoch(
    model, loader, criterion, optimizer,
    device, max_batches, pairs=None,
):
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    processed = 0
    all_case_records = []

    for batch_index, (images, masks) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected.")

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        # Record detailed metrics during validation only.
        if not training:
            predictions = logits.detach().argmax(dim=1)

            for sample_index in range(images.shape[0]):
                records = calculate_case_metrics(
                    predictions[sample_index],
                    masks[sample_index],
                )

                case_index = processed + sample_index

                if pairs is not None:
                    _, mri_path, _ = pairs[case_index]
                    case_name = mri_path.name
                else:
                    case_name = f"case_{case_index}"

                for row in records:
                    row["case"] = case_name

                all_case_records.append(records)

        total_loss += loss.item()
        processed += images.shape[0]

        phase = "Train" if training else "Val"
        print(
            f"{phase} batch {batch_index + 1}: "
            f"loss={loss.item():.4f}",
            flush=True,
        )

    if processed == 0:
        raise RuntimeError("No samples processed.")

    return total_loss / processed, all_case_records


def save_case_metrics(path, cases):
    """Save individual validation-case metrics to CSV."""
    fields = [
        "case", "class_id", "class_name",
        "dice", "iou",
        "gt_voxels", "pred_voxels",
        "tp", "fp", "fn",
    ]

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for case in cases:
            for record in case:
                writer.writerow(record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--output-dir", default="results/baseline_dice_ce")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = get_data_pairs()
    train_pairs, val_pairs, test_pairs = split_by_patient(
        pairs, seed=42
    )

    print(
        f"Data split: {len(train_pairs)} train, "
        f"{len(val_pairs)} val, {len(test_pairs)} test",
        flush=True,
    )

    train_loader = DataLoader(
        HipMRIDataset(train_pairs, target_shape=TARGET_SHAPE),
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        HipMRIDataset(val_pairs, target_shape=TARGET_SHAPE),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = UNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=8,
    ).to(device)

    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history_path = output_dir / "history.csv"
    best_val_loss = float("inf")

    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["epoch", "train_loss", "val_loss"]
            + [f"val_dice_{i}" for i in range(NUM_CLASSES)]
            + [f"val_iou_{i}" for i in range(NUM_CLASSES)]
        )

        for epoch in range(1, args.epochs + 1):
            print(
                f"\n===== Epoch {epoch}/{args.epochs} =====",
                flush=True,
            )

            train_loss, _ = run_epoch(
                model, train_loader, criterion,
                optimizer, device, args.max_train_batches,
            )

            val_loss, case_records = run_epoch(
                model, val_loader, criterion,
                None, device, args.max_val_batches,
                pairs=val_pairs,
            )

            summary = summarise_metrics(case_records)

            dice = [row["mean_dice"] for row in summary]
            iou = [row["mean_iou"] for row in summary]

            writer.writerow(
                [epoch, train_loss, val_loss] + dice + iou
            )
            file.flush()

            save_case_metrics(
                output_dir / f"val_cases_epoch_{epoch}.csv",
                case_records,
            )

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}",
                flush=True,
            )

            print(
                "Class | Dice | IoU | GT voxels | "
                "Pred voxels | Valid cases",
                flush=True,
            )

            for row in summary:
                print(
                    f"{row['class_name']}: "
                    f"{row['mean_dice']:.4f} | "
                    f"{row['mean_iou']:.4f} | "
                    f"{row['gt_voxels']} | "
                    f"{row['pred_voxels']} | "
                    f"{row['valid_cases']}",
                    flush=True,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, output_dir / "best_model.pt")

    print(
        "Peak allocated GPU memory (GB):",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        flush=True,
    )
    print("Training finished.", flush=True)


if __name__ == "__main__":
    main()
