"""
Evaluates the baseline U-Net and Improved U-Net on the HipMRI test dataset.

Calculates per-class Dice and IoU, prostate-specific Dice, IoU and HD95,
test loss, inference time and peak GPU memory usage.

Saves evaluation results to CSV files for comparison between models.
"""

import sys
import os
import torch
import time
import csv
import torch.nn as nn
import numpy as np

from scipy.ndimage import binary_erosion, distance_transform_edt

from dataset import test_loader
from model import UNet
from improved_model import ImprovedUNet


# Class label representing the prostate in the segmentation masks
PROSTATE_CLASS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

num_classes = 6
criterion = nn.CrossEntropyLoss()


def load_model(model_type, checkpoint_path):
    """Load a trained U-Net model from a checkpoint."""

    if model_type == "baseline":
        model = UNet(
            in_channels=1,
            num_classes=num_classes
        )

    elif model_type == "improved":
        model = ImprovedUNet(
            in_channels=1,
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Restore the best model weights saved during training
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Switch to evaluation mode
    model.eval()

    return model


def calculate_prostate_metrics(prediction, ground_truth):
    """
    Calculates prostate Dice, IoU and HD95 for a single slice.
    """
    pred = np.asarray(prediction) == PROSTATE_CLASS
    gt = np.asarray(ground_truth) == PROSTATE_CLASS

    intersection = np.logical_and(pred, gt).sum()
    pred_count = pred.sum()
    gt_count = gt.sum()

    # Exclude slices where neither prediction nor ground truth contains prostate
    if pred_count == 0 and gt_count == 0:
        return None

    denominator = pred_count + gt_count
    union = np.logical_or(pred, gt).sum()

    # Dice and IoU measure overlap between predicted and actual prostate regions
    dice = 2 * intersection / denominator if denominator else 1.0
    iou = intersection / union if union else 1.0

    # HD95 cannot be finite if only one mask is empty
    if pred_count == 0 or gt_count == 0:
        hd95 = float("inf")
    else:
        # Extract prostate boundary pixels using binary erosion
        pred_boundary = pred & ~binary_erosion(pred)
        gt_boundary = gt & ~binary_erosion(gt)

        # Calculate distances from each boundary pixel to the opposite boundary
        dist_to_gt = distance_transform_edt(~gt_boundary)
        dist_to_pred = distance_transform_edt(~pred_boundary)

        # Combine distances in both directions for symmetric HD95
        distances = np.concatenate([
            dist_to_gt[pred_boundary],
            dist_to_pred[gt_boundary]
        ])

        # Use the 95th percentile to reduce sensitivity to extreme outliers
        hd95 = float(np.percentile(distances, 95))

    return {
        "dice": float(dice),
        "iou": float(iou),
        "hd95_pixels": hd95,
        "ground_truth_pixels": int(gt_count),
        "predicted_pixels": int(pred_count)
    }

def calculate_metrics(predictions, masks, num_classes=6):
    """
    Calculate pixel counts required for per-class Dice and IoU.
    """

    intersections = torch.zeros(num_classes, dtype=torch.float64)
    predicted_totals = torch.zeros(num_classes, dtype=torch.float64)
    actual_totals = torch.zeros(num_classes, dtype=torch.float64)

    for class_id in range(num_classes):

        predicted_class = predictions == class_id
        actual_class = masks == class_id

        intersections[class_id] = (
            predicted_class & actual_class
        ).sum().item()

        predicted_totals[class_id] = predicted_class.sum().item()
        actual_totals[class_id] = actual_class.sum().item()

    return intersections, predicted_totals, actual_totals


def calculate_scores(intersections, predicted_totals, actual_totals):
    """Calculate per-class Dice and IoU scores."""

    results = []

    for class_id in range(num_classes):

        intersection = intersections[class_id].item()
        predicted = predicted_totals[class_id].item()
        actual = actual_totals[class_id].item()

        # Calculate Dice and IoU denominators
        dice_denominator = predicted + actual
        union = predicted + actual - intersection

        # Avoid division by zero for absent classes
        dice = (
            2 * intersection / dice_denominator
            if dice_denominator > 0 else float("nan")
        )

        iou = (
            intersection / union
            if union > 0 else float("nan")
        )

        results.append({
            "class_id": class_id,
            "dice": dice,
            "iou": iou
        })

    return results


def calculate_mean_foreground(results):
    """Calculate mean Dice and IoU across foreground classes."""

    # Exclude background
    foreground = results[1:]

    # Ignore undefined scores from classes absent in both masks
    valid_dice = [
        result["dice"]
        for result in foreground
        if not torch.isnan(torch.tensor(result["dice"]))
    ]

    valid_iou = [
        result["iou"]
        for result in foreground
        if not torch.isnan(torch.tensor(result["iou"]))
    ]

    mean_dice = (
        sum(valid_dice) / len(valid_dice)
        if valid_dice else float("nan")
    )

    mean_iou = (
        sum(valid_iou) / len(valid_iou)
        if valid_iou else float("nan")
    )

    return mean_dice, mean_iou

def save_prostate_results(results, model_name):
    """
    Save prostate segmentation metrics for test slices.
    """
    os.makedirs("results", exist_ok=True)

    output_path = f"results/{model_name}_prostate_slices.csv"

    fieldnames = [
        "index",
        "filename",
        "dice",
        "iou",
        "hd95_pixels",
        "ground_truth_pixels",
        "predicted_pixels"
    ]

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved prostate metrics to {output_path}")

    # Exclude infinite HD95 values when calculating the average
    valid_hd95 = [
        result["hd95_pixels"]
        for result in results
        if np.isfinite(result["hd95_pixels"])
    ]

    mean_hd95 = (
        float(np.mean(valid_hd95))
        if valid_hd95 else float("nan")
    )

    # Count slices where HD95 could not be calculated
    failed_hd95 = len(results) - len(valid_hd95)

    print(f"Mean HD95 (finite cases): {mean_hd95:.4f} pixels")
    print(f"Undefined HD95 cases: {failed_hd95}")

    return mean_hd95, failed_hd95


def evaluate_model(model, data_loader, model_name):
    """Evaluate a segmentation model on the held-out test dataset."""

    model.eval()

    # Accumulate class pixel counts across the entire test dataset
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    predicted_totals = torch.zeros(num_classes, dtype=torch.float64)
    actual_totals = torch.zeros(num_classes, dtype=torch.float64)

    total_inference_time = 0.0
    total_images = 0

    # Initialise test loss
    total_test_loss = 0.0

    prostate_results = []
    dataset_index = 0

    # Reset peak GPU memory measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():

        for images, masks in data_loader:

            images = images.to(device)
            masks = masks.to(device)

            # Synchronise GPU before timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # Generate segmentation predictions
            outputs = model(images)

            # Wait for GPU inference to finish
            if device.type == "cuda":
                torch.cuda.synchronize()

            total_inference_time += time.perf_counter() - start_time
            total_images += images.size(0)

            # Calculate test loss
            loss = criterion(outputs, masks)
            total_test_loss += loss.item() * images.size(0)

            # Select the class with the highest score for each pixel
            predictions = outputs.argmax(dim=1)

            # Calculate prostate metrics for every image in the batch
            for i in range(predictions.shape[0]):
                prediction = predictions[i].cpu().numpy()
                ground_truth = masks[i].cpu().numpy()

                metrics = calculate_prostate_metrics(
                    prediction,
                    ground_truth
                )

                if metrics is not None:
                    prostate_results.append({
                        "index": dataset_index,
                        "filename": os.path.basename(
                            data_loader.dataset.image_paths[dataset_index]
                        ),
                        **metrics
                    })

                dataset_index += 1

            # Calculate metric counts for this batch
            batch_intersections, batch_predicted, batch_actual = (
                calculate_metrics(predictions, masks, num_classes)
            )

            # Accumulate counts across all batches
            intersections += batch_intersections
            predicted_totals += batch_predicted
            actual_totals += batch_actual

    # Calculate average forward-pass time per image
    average_inference_time = (
        total_inference_time / total_images
        if total_images > 0 else 0.0
    )

    mean_hd95, failed_hd95 = save_prostate_results(
        prostate_results,
        model_name
    )

    # Calculate average test loss
    average_test_loss = (
        total_test_loss / total_images
        if total_images > 0 else float("nan")
    )

    # Measure peak allocated GPU memory during evaluation
    peak_vram = 0.0

    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    return (
        intersections,
        predicted_totals,
        actual_totals,
        average_inference_time,
        average_test_loss,
        peak_vram,
        mean_hd95,
        failed_hd95
    )
    
def save_results(
    results,
    mean_dice,
    mean_iou,
    inference_time,
    test_loss,
    peak_vram,
    mean_hd95,
    failed_hd95,
    output_path
):
    """Save segmentation scores and performance metrics to CSV."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["metric", "value"])

        # Save Dice and IoU for every segmentation class
        for result in results:
            class_id = result["class_id"]

            writer.writerow([
                f"class_{class_id}_dice",
                result["dice"]
            ])

            writer.writerow([
                f"class_{class_id}_iou",
                result["iou"]
            ])

        # Save mean foreground segmentation performance
        writer.writerow(["mean_foreground_dice", mean_dice])
        writer.writerow(["mean_foreground_iou", mean_iou])

        # Save computational efficiency and test loss
        writer.writerow(["test_loss", test_loss])
        writer.writerow(["inference_seconds_per_image", inference_time])
        writer.writerow(["peak_inference_vram_gb", peak_vram])

        # Save prostate boundary accuracy
        writer.writerow(["mean_hd95_pixels", mean_hd95])
        writer.writerow(["undefined_hd95_cases", failed_hd95])


if __name__ == "__main__":

    models = {
        "baseline": "checkpoints/unet_baseline_best.pth",
        "improved": "checkpoints/unet_improved_best.pth"
    }

    if len(sys.argv) != 2 or sys.argv[1] not in models:
        raise ValueError("Usage: python evaluate.py baseline|improved")

    model_type = sys.argv[1]
    checkpoint_path = models[model_type]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nEvaluating {model_type} U-Net")

    model = load_model(model_type, checkpoint_path)

    (
        intersections,
        predicted_totals,
        actual_totals,
        inference_time,
        test_loss,
        peak_vram,
        mean_hd95,
        failed_hd95
    ) = evaluate_model(model, test_loader, model_type)

    # Convert accumulated pixel counts into Dice and IoU scores
    results = calculate_scores(
        intersections,
        predicted_totals,
        actual_totals
    )

    mean_dice, mean_iou = calculate_mean_foreground(results)

    # Display segmentation performance for each class
    for result in results:
        print(
            f"Class {result['class_id']}: "
            f"Dice = {result['dice']:.4f}, "
            f"IoU = {result['iou']:.4f}"
        )

    print(f"Mean foreground Dice: {mean_dice:.4f}")
    print(f"Mean foreground IoU: {mean_iou:.4f}")
    print(f"Test Loss: {test_loss:.5f}")
    print(f"Average inference time: {inference_time:.6f} seconds")
    print(f"Peak Inference VRAM: {peak_vram:.4f} GB")

    # Save final evaluation results
    save_results(
        results,
        mean_dice,
        mean_iou,
        inference_time,
        test_loss,
        peak_vram,
        mean_hd95,
        failed_hd95,
        f"results/{model_type}_evaluation.csv"
    )