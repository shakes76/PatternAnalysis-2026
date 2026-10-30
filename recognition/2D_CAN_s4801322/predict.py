"""
Evaluate the trained HipMRI 2D CAN/U-net on the held-out test set.

Uses the same dataset preprocessing and label mapping as train.py.
All paths and model hyperparameters come from config.py.
"""
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import config
from dataset import HipMRI_Dataset
from modules import UNet, CAN

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def boundary_metrics(prediction, ground_truth, class_id=5, tolerance=2):
    """
    Calculate boundary precision, recall and F1 for one 2D mask.

    prediction, ground_truth: [H, W] arrays of class indices.
    tolerance: Maximum distance in pixels to count as a match.
    """

    # Extract the prostate region.
    pred = np.asarray(prediction) == class_id
    true = np.asarray(ground_truth) == class_id

    # If both masks are empty, there is no boundary to evaluate.
    if not pred.any() and not true.any():
        return np.nan, np.nan, np.nan

    # If only one mask is empty, the segmentation is incorrect.
    if not pred.any() or not true.any():
        return 0.0, 0.0, 0.0

    # Find boundary pixels using erosion.
    pred_boundary = pred & ~binary_erosion(pred)
    true_boundary = true & ~binary_erosion(true)

    # Distance from each pixel to the nearest boundary pixel.
    distance_to_true = distance_transform_edt(~true_boundary)
    distance_to_pred = distance_transform_edt(~pred_boundary)

    # Fraction of predicted boundary pixels close to ground truth.
    precision = (
        (distance_to_true[pred_boundary] <= tolerance).mean()
    )

    # Fraction of ground-truth boundary pixels close to prediction.
    recall = (
        (distance_to_pred[true_boundary] <= tolerance).mean()
    )

    # Harmonic mean of precision and recall.
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1)

def measure_inference_time(model, test_dataset, device):
    """Measure model-only inference time in ms per 2D slice (batch size 1)."""
    # Warm up GPU kernels before timing.
    example_image, _ = test_dataset[0]
    example_image = example_image.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(5):
            model(example_image)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for index in range(len(test_dataset)):
            image, _ = test_dataset[index]
            image = image.unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times_ms)), float(np.median(times_ms)), float(np.percentile(times_ms, 95))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HipMRI_Dataset(
        config.TEST_DIRECTORY,
        config.TEST_SEG_DIRECTORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = CAN(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        hidden_channels=config.HIDDEN_CHANNELS,
    ).to(device)

    # Count trainable model parameters.
    num_parameters = sum(
    p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters: {num_parameters:,}")

    # Checkpoint was saved with torch.save(model.state_dict(), ...).
    state = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Aggregate intersection and region sizes across ALL test images.
    intersections = torch.zeros(config.NUM_CLASSES, dtype=torch.float64)
    pred_counts = torch.zeros_like(intersections)
    true_counts = torch.zeros_like(intersections)
    example = None

    # Store boundary metrics for every test image.
    boundary_precisions = []
    boundary_recalls = []
    boundary_f1_scores = []

    # Track the three lowest prostate Dice scores on slices containing prostate.
    worst_segmentations = []
    slice_index = 0

    # Slices that need to be compared to another model
    COMPARISON_SLICES = {8, 93, 130}

    with torch.inference_mode():
        for images, masks in test_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()

            if example is None:
                example = (images[0, 0].numpy(), masks[0].numpy(), predictions[0].numpy())

            for cls in range(config.NUM_CLASSES):
                pred = predictions == cls
                true = masks == cls
                intersections[cls] += (pred & true).sum().item()
                pred_counts[cls] += pred.sum().item()
                true_counts[cls] += true.sum().item()
            # Calculate prostate boundary metrics for each image.
            for image, prediction, mask in zip(images, predictions, masks):
                precision, recall, f1 = boundary_metrics(
                    prediction.numpy(),
                    mask.numpy(),
                    class_id=5,
                    tolerance=2,
                )

                boundary_precisions.append(precision)
                boundary_recalls.append(recall)
                boundary_f1_scores.append(f1)

                # Score the prostate on this individual slice, rather than
                # using the global Dice computed across the full test set.
                predicted_prostate = prediction == 5
                actual_prostate = mask == 5
                if actual_prostate.any():
                    intersection = (predicted_prostate & actual_prostate).sum().item()
                    denominator = predicted_prostate.sum().item() + actual_prostate.sum().item()
                    prostate_dice = 2 * intersection / denominator

                    worst_segmentations.append((
                        prostate_dice,
                        slice_index,
                        image[0].numpy().copy(),
                        mask.numpy().copy(),
                        prediction.numpy().copy(),
                    ))
                    worst_segmentations.sort(key=lambda item: (item[0], item[1]))
                    worst_segmentations = worst_segmentations[:3]
                
                # Get matching slices to compare to another model
                if slice_index in COMPARISON_SLICES:

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                    axes[0].imshow(image[0].numpy(), cmap="gray")
                    axes[0].set_title("MRI Input")

                    axes[1].imshow(
                        mask.numpy(), cmap="tab10",
                        vmin=0, vmax=config.NUM_CLASSES - 1
                    )
                    axes[1].set_title("Ground Truth")

                    axes[2].imshow(
                        prediction.numpy(), cmap="tab10",
                        vmin=0, vmax=config.NUM_CLASSES - 1
                    )
                    axes[2].set_title("Model Prediction")

                    for ax in axes:
                        ax.axis("off")

                    plt.tight_layout()
                    plt.savefig(config.BASE_DIRECETORY/ "recognition" / "2D_CAN_s4801322" / "figures" / f"unet_slice_{slice_index}.png")
                    plt.close(fig)
                slice_index += 1

    # Report inference memory and model-only timing, separately from data loading.
    if device.type == "cuda":
        peak_vram_mib = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Peak inference GPU memory: {peak_vram_mib:.2f} MiB")

    mean_ms, median_ms, p95_ms = measure_inference_time(model, test_dataset, device)
    print(f"Inference time per 2D slice: mean={mean_ms:.2f} ms, "
          f"median={median_ms:.2f} ms, p95={p95_ms:.2f} ms")
    
    print(f"Test images: {len(test_dataset)}")
    print("Per-class test metrics (across the entire test set):")
    class_dice_scores = []
    for cls in range(config.NUM_CLASSES):
        intersection = intersections[cls].item()
        predicted = pred_counts[cls].item()
        actual = true_counts[cls].item()
        dice = 2 * intersection / (predicted + actual) if (predicted + actual) else float("nan")
        union = predicted + actual - intersection
        iou = intersection / union if union else float("nan")
        class_dice_scores.append(dice)
        print(f"Class {cls}: Dice={dice:.4f}, IoU={iou:.4f}, target pixels={int(actual)}")

    print(f"Mean foreground Dice: {np.nanmean(class_dice_scores[1:]):.4f}")
    print(f"Prostate Dice (class 5): {class_dice_scores[5]:.4f}")
    
    # Report average prostate boundary metrics.
    print("\nAverage prostate boundary metrics (2-pixel tolerance):")
    print(f"Precision: {np.nanmean(boundary_precisions):.4f}")
    print(f"Recall:    {np.nanmean(boundary_recalls):.4f}")
    print(f"F1 score:  {np.nanmean(boundary_f1_scores):.4f}")

    # Save the three worst prostate segmentations for failure analysis.
    output_directory = config.WORST_CASE_PATH
    output_directory.mkdir(parents=True, exist_ok=True)

    for rank, (dice, index, image, actual, predicted) in enumerate(worst_segmentations, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("HipMRI input")
        axes[1].imshow(actual, cmap="tab10", vmin=0, vmax=config.NUM_CLASSES - 1, interpolation="nearest")
        axes[1].set_title("Ground truth")
        axes[2].imshow(predicted, cmap="tab10", vmin=0, vmax=config.NUM_CLASSES - 1, interpolation="nearest")
        axes[2].set_title("CAN prediction")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Worst #{rank} | Test slice {index} | Prostate Dice: {dice:.4f}")
        plt.tight_layout()
        save_path = output_directory / f"worst_{rank}_slice_{index}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved {save_path} (prostate Dice={dice:.4f})")


    # Show one example with exactly the same class-color scale for both masks.
    if example is not None:
        image, actual, predicted = example
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("HipMRI input")
        axes[1].imshow(actual, cmap="tab10", vmin=0, vmax=config.NUM_CLASSES - 1, interpolation="nearest")
        axes[1].set_title("Ground truth")
        axes[2].imshow(predicted, cmap="tab10", vmin=0, vmax=config.NUM_CLASSES - 1, interpolation="nearest")
        axes[2].set_title("CAN prediction")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    main()
