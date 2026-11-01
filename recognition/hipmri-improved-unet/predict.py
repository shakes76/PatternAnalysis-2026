# Prediction driver for comparing trained U-Net models on test MRI slices.
# Reuses inference and visualisation utilities from visual_failure.py.
import argparse
from pathlib import Path

import torch

from dataset import HipMRIDataset
from modules import ImprovedUNet
from baseline_unet.modules import UNet

from visual_failure import (
    load_mri_slice,
    load_trained_model,
    predict_segmentation,
    calculate_class_dice,
    visualize_segmentation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run HipMRI segmentation using trained U-Net models"
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="case_040_week_0_slice_38.nii.gz",
        help="MRI slice filename from the test dataset"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    )

    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="checkpoints_unet_20/best_unet.pth"
    )

    parser.add_argument(
        "--improved-checkpoint",
        type=str,
        default="checkpoints_improved_20/best_improved_unet.pth"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    dataset_root = Path(args.data_root)

    dataset = HipMRIDataset(
        image_dir=dataset_root / "keras_slices_test",
        mask_dir=dataset_root / "keras_slices_seg_test"
    )

    # Load trained baseline and Improved U-Net models.
    baseline_model = load_trained_model(
        UNet,
        args.baseline_checkpoint,
        device
    )

    improved_model = load_trained_model(
        ImprovedUNet,
        args.improved_checkpoint,
        device
    )

    # Load one MRI slice and its ground-truth mask.
    image, ground_truth = load_mri_slice(
        dataset,
        args.filename
    )

    # Generate segmentation predictions.
    baseline_prediction = predict_segmentation(
        baseline_model,
        image,
        device
    )

    improved_prediction = predict_segmentation(
        improved_model,
        image,
        device
    )

    # Calculate per-class Dice scores.
    baseline_scores = calculate_class_dice(
        baseline_prediction,
        ground_truth
    )

    improved_scores = calculate_class_dice(
        improved_prediction,
        ground_truth
    )

    print(f"\nMRI slice: {args.filename}")
    print("Class | Baseline Dice | Improved Dice")

    for class_id in range(1, 6):
        baseline_dice = baseline_scores[class_id]["dice"]
        improved_dice = improved_scores[class_id]["dice"]

        baseline_text = (
            "N/A" if baseline_dice is None
            else f"{baseline_dice:.4f}"
        )

        improved_text = (
            "N/A" if improved_dice is None
            else f"{improved_dice:.4f}"
        )

        print(
            f"{class_id:5d} | "
            f"{baseline_text:13} | "
            f"{improved_text}"
        )

    # Save a four-panel prediction comparison.
    visualize_segmentation(
        image,
        ground_truth,
        baseline_prediction,
        improved_prediction,
        args.filename
    )

    print("\nPrediction completed successfully.")


if __name__ == "__main__":
    main()