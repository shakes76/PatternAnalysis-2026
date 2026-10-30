import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet, ImprovedUNet


NUM_CLASSES = 6


def evaluate(model, loader, device):
    """
    Evaluate a segmentation model on the test dataset.

    Dice and IoU are calculated from pixel counts accumulated
    across all test batches. Inference time is also measured.
    """
    intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    predicted = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    actual = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    total_time = 0.0
    total_images = 0

    model.eval()

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Synchronise CUDA before measuring GPU execution time.
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            # Generate class predictions for each pixel.
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            # Wait for GPU operations to finish before stopping the timer.
            if device.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start
            total_images += images.shape[0]

            # Accumulate pixel counts for each segmentation class.
            for class_id in range(NUM_CLASSES):
                predicted_mask = predictions == class_id
                target_mask = masks == class_id

                intersection[class_id] += (
                    predicted_mask & target_mask
                ).sum().item()

                predicted[class_id] += predicted_mask.sum().item()
                actual[class_id] += target_mask.sum().item()

    dice = []
    iou = []

    # Calculate Dice and IoU from the accumulated pixel counts.
    for class_id in range(NUM_CLASSES):
        inter = intersection[class_id].item()
        pred = predicted[class_id].item()
        true = actual[class_id].item()

        dice_den = pred + true
        iou_den = pred + true - inter

        dice.append(
            2 * inter / dice_den if dice_den else float("nan")
        )

        iou.append(
            inter / iou_den if iou_den else float("nan")
        )

    # Exclude undefined class scores from the mean.
    valid_dice = [score for score in dice if score == score]
    valid_iou = [score for score in iou if score == score]

    return {
        "mean_dice": sum(valid_dice) / len(valid_dice),
        "mean_iou": sum(valid_iou) / len(valid_iou),
        "dice": dice,
        "iou": iou,
        "inference_ms_per_image": total_time / total_images * 1000,
        "images": total_images,
    }


def main():
    """Load both trained models and compare their test performance."""

    parser = argparse.ArgumentParser(
        description="Evaluate baseline and Improved U-Net models."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Use the held-out test split for final evaluation.
    dataset = HipMRIDataset(split="test")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )

    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)

    results = []

    # Compare checkpoints from the original baseline and improved runs.
    for name, model_class in [
        ("baseline", UNet),
        ("improved", ImprovedUNet),
    ]:
        checkpoint_path = (
            Path("runs") / f"{name}_v1" / "best_model.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model = model_class(
            in_channels=1,
            num_classes=NUM_CLASSES,
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Run evaluation on the full test dataset.
        metrics = evaluate(model, loader, device)

        print(f"\n{name.upper()}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Test images: {metrics['images']}")
        print(f"Mean Dice: {metrics['mean_dice']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")

        print(
            "Per-class Dice:",
            [round(score, 4) for score in metrics["dice"]],
        )

        print(
            "Per-class IoU:",
            [round(score, 4) for score in metrics["iou"]],
        )

        print(
            f"Inference: "
            f"{metrics['inference_ms_per_image']:.2f} ms/image"
        )

        # Store results in a format suitable for CSV export.
        row = {
            "model": name,
            "checkpoint_epoch": checkpoint["epoch"],
            "test_images": metrics["images"],
            "mean_dice": metrics["mean_dice"],
            "mean_iou": metrics["mean_iou"],
            "inference_ms_per_image":
                metrics["inference_ms_per_image"],
        }

        for class_id in range(NUM_CLASSES):
            row[f"dice_class_{class_id}"] = metrics["dice"][class_id]
            row[f"iou_class_{class_id}"] = metrics["iou"][class_id]

        results.append(row)

        # Release GPU memory before loading the next model.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save both models' results in one comparison file.
    with open(
        output_dir / "test_results.csv",
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to evaluation/test_results.csv")


if __name__ == "__main__":
    main()

