import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from baseline_unet.modules import UNet
from baseline_unet.train import validate

#Integraing Boundary precision function
import numpy as np

from utils import (
    calculate_boundary_precision,
    count_trainable_parameters,
    measure_gpu_memory,
    measure_inference_time
)

def load_model(checkpoint_path, device):
    # Initialise the baseline U-Net architecture
    model = UNet(in_channels=1, num_classes=6)

    # Load the saved checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    # Restore the trained model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move the model to the selected device
    model = model.to(device)

    # Switch to evaluation mode
    model.eval()

    return model

# Load validation or test MRI data
def load_data(data_root, split="validate", batch_size=16):
    root = Path(data_root)

    dataset = HipMRIDataset(
        root / f"keras_slices_{split}",
        root / f"keras_slices_seg_{split}"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return loader

# Evaluate baseline U-Net segmentation performance
def evaluate_model(model, loader, device):
    criterion = nn.CrossEntropyLoss()

    loss, mean_dice, mean_iou, dice_scores, iou_scores = validate(
        model,
        loader,
        criterion,
        device,
        num_classes=6
    )

    print(f"Loss: {loss:.4f}")
    print(f"Mean Foreground Dice: {mean_dice:.4f}")
    print(f"Mean Foreground IoU: {mean_iou:.4f}")

    # Display individual foreground class performance
    for class_id in range(1, 6):
        dice = dice_scores[class_id]
        iou = iou_scores[class_id]

        if dice is not None and iou is not None:
            print(
                f"Class {class_id}: "
                f"Dice = {dice:.4f}, IoU = {iou:.4f}"
            )
        else:
            print(f"Class {class_id}: No valid pixels for evaluation")


def evaluate_boundary_precision(model, loader, device, num_classes=6):
    """
    Evaluate class-wise boundary precision using a 2-pixel tolerance.

    Undefined scores are excluded from the mean and counted separately.
    """

    model.eval()

    boundary_scores = {class_id: [] for class_id in range(1, num_classes)}
    undefined_counts = {class_id: 0 for class_id in range(1, num_classes)}

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            targets = masks.numpy()

            for prediction, target in zip(predictions, targets):
                for class_id in range(1, num_classes):

                    precision = calculate_boundary_precision(
                        prediction,
                        target,
                        class_id,
                        tolerance=2
                    )

                    if precision is None:
                        undefined_counts[class_id] += 1
                    else:
                        boundary_scores[class_id].append(precision)

    print("\nBoundary Precision Evaluation (2-pixel tolerance)")

    class_means = []

    for class_id in range(1, num_classes):
        scores = boundary_scores[class_id]

        if scores:
            mean_precision = float(np.mean(scores))
            class_means.append(mean_precision)

            print(
                f"Class {class_id}: "
                f"Boundary Precision = {mean_precision:.4f}, "
                f"Valid slices = {len(scores)}, "
                f"Undefined slices = {undefined_counts[class_id]}"
            )
        else:
            print(
                f"Class {class_id}: No valid boundary predictions, "
                f"Undefined slices = {undefined_counts[class_id]}"
            )

    if class_means:
        print(
            f"Mean Foreground Boundary Precision: "
            f"{np.mean(class_means):.4f}"
        )

# Evaluate computational efficiency of the Baseline U-Net
def evaluate_efficiency(model, loader, device):

    if device.type != "cuda":
        print("GPU benchmarking skipped: CUDA unavailable.")
        return

    # Use the same input batch size for both models.
    images, _ = next(iter(loader))
    images = images.to(device)

    parameters = count_trainable_parameters(model)

    peak_memory = measure_gpu_memory(
        model, images, device
    )

    inference_time = measure_inference_time(
        model, images, device
    )

    print("\nComputational Efficiency")
    print(f"Trainable Parameters: {parameters:,}")
    print(f"Peak GPU Memory: {peak_memory:.2f} MB")
    print(f"Average Inference Time: {inference_time:.4f} ms/slice")

#Using similar main function as our improved U-Net model
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline 2D U-Net on HipMRI"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["validate", "test"],
        default="validate"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_unet_20/best_unet.pth"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Evaluating split: {args.split}")

    # Load the trained baseline U-Net
    model = load_model(args.checkpoint, device)

    # Load the selected dataset
    loader = load_data(
        args.data_root,
        split=args.split
    )

    # Evaluate segmentation performance
    evaluate_model(model, loader, device)

    # Evaluate boundary precision
    evaluate_boundary_precision(model, loader, device)

    # Evaluate computational efficiency
    evaluate_efficiency(model, loader, device)


if __name__ == "__main__":
    main()