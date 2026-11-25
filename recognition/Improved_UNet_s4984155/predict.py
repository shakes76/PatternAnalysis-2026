"""
Evaluation and visualization script for 2D prostate segmentation on HipMRI.
Computes test set metrics (Dice, IoU, latency) and plots segmentation error overlays.
"""

import os
import sys
import time
import argparse
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules import ImprovedUNet, BaselineUNet
from dataset import get_data_loaders, HipMRI2DSliceDataset


def compute_slice_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    eps: float = 1e-5
) -> Tuple[float, float]:
    pred_flat = pred_mask.astype(bool).flatten()
    gt_flat = gt_mask.astype(bool).flatten()

    intersection = np.logical_and(pred_flat, gt_flat).sum()
    total_pred = pred_flat.sum()
    total_gt = gt_flat.sum()

    if total_gt == 0 and total_pred == 0:
        return 1.0, 1.0
    if total_gt == 0 and total_pred > 0:
        return 0.0, 0.0

    dice = (2.0 * intersection + eps) / (total_pred + total_gt + eps)
    union = total_pred + total_gt - intersection
    iou = (intersection + eps) / (union + eps)

    return float(dice), float(iou)


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, float]:
    model.eval()
    dice_list = []
    iou_list = []
    latencies = []

    print("Evaluating test set...", flush=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latency = (t1 - t0) / images.size(0)
            latencies.extend([batch_latency] * images.size(0))

            preds = (probs > threshold).float().cpu().numpy()
            targets = masks.cpu().numpy()

            for b in range(images.size(0)):
                p_slice = preds[b, 0]
                g_slice = targets[b, 0]
                d, i = compute_slice_metrics(p_slice, g_slice)
                dice_list.append(d)
                iou_list.append(i)

    mean_dice = float(np.mean(dice_list))
    std_dice = float(np.std(dice_list))
    mean_iou = float(np.mean(iou_list))
    avg_latency_ms = float(np.mean(latencies)) * 1000.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "mean_dice": mean_dice,
        "std_dice": std_dice,
        "mean_iou": mean_iou,
        "latency_ms": avg_latency_ms,
        "fps": fps,
        "total_slices": len(dice_list)
    }


def create_difference_overlay(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB error overlay: TP=Green, FP=Red, FN=Blue."""
    h, w = pred.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    tp = np.logical_and(pred == 1, gt == 1)
    fp = np.logical_and(pred == 1, gt == 0)
    fn = np.logical_and(pred == 0, gt == 1)

    overlay[tp] = [46, 204, 113]   # Green
    overlay[fp] = [231, 76, 60]    # Red
    overlay[fn] = [52, 152, 219]   # Blue

    return overlay


def plot_qualitative_results(
    model: nn.Module,
    dataset: HipMRI2DSliceDataset,
    device: torch.device,
    save_path: str,
    num_samples: int = 4
):
    model.eval()

    prostate_indices = []
    for i in range(len(dataset)):
        _, m = dataset[i]
        if m.sum() > 50:
            prostate_indices.append(i)

    if len(prostate_indices) >= num_samples:
        step = len(prostate_indices) // num_samples
        selected_indices = [prostate_indices[i * step] for i in range(num_samples)]
    else:
        selected_indices = list(range(min(num_samples, len(dataset))))

    fig, axes = plt.subplots(num_samples, 4, figsize=(13, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    with torch.no_grad():
        for row_idx, data_idx in enumerate(selected_indices):
            image, mask = dataset[data_idx]
            input_tensor = image.unsqueeze(0).to(device)

            output = model(input_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).float().cpu().numpy()[0, 0]
            gt = mask.numpy()[0]
            mri_img = image.numpy()[0]

            dice, iou = compute_slice_metrics(pred, gt)
            overlay = create_difference_overlay(pred, gt)

            axes[row_idx, 0].imshow(mri_img, cmap="gray")
            axes[row_idx, 0].set_title(f"Sample {row_idx+1}: Input MRI", fontsize=10)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(mri_img, cmap="gray")
            axes[row_idx, 1].imshow(gt, cmap="spring", alpha=0.5)
            axes[row_idx, 1].set_title("Ground Truth (Label 5)", fontsize=10)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(mri_img, cmap="gray")
            axes[row_idx, 2].imshow(pred, cmap="cool", alpha=0.5)
            axes[row_idx, 2].set_title(f"Prediction (Dice: {dice:.3f})", fontsize=10)
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(mri_img, cmap="gray")
            axes[row_idx, 3].imshow(overlay, alpha=0.6)
            axes[row_idx, 3].set_title("Overlay (TP=Grn, FP=Red, FN=Blu)", fontsize=10)
            axes[row_idx, 3].axis("off")

    tp_patch = mpatches.Patch(color='#2ECC71', label='True Positive (TP)')
    fp_patch = mpatches.Patch(color='#E74C3C', label='False Positive (FP)')
    fn_patch = mpatches.Patch(color='#3498DB', label='False Negative (FN)')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved to: {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net on HipMRI 2D Prostate Test Set")
    parser.add_argument("--weights", type=str, default="best_improved_unet.pth")
    parser.add_argument("--model", type=str, default="improved", choices=["improved", "baseline"])
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--num_samples", type=int, default=4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model.upper()} | Weights: {args.weights}", flush=True)

    if args.data_dir:
        data_path = args.data_dir
    else:
        default_hpc_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
        default_local_path = "./keras_slices_data"
        data_path = default_hpc_path if os.path.exists(default_hpc_path) else default_local_path

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Specify via --data_dir.")

    _, _, test_loader = get_data_loaders(
        data_root=data_path,
        batch_size=16,
        num_workers=0,
        target_size=(256, 128),
        prostate_only=True
    )

    if args.model == "improved":
        model = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
    else:
        model = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded weights from {args.weights}", flush=True)

    results = evaluate_test_set(model, test_loader, device)

    print("Test Results:")
    print(f"  Slices:       {results['total_slices']}")
    print(f"  Dice (DSC):   {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}")
    print(f"  IoU:          {results['mean_iou']:.4f}")
    print(f"  Latency:      {results['latency_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    print(f"  Target >=0.75: {'PASSED' if results['mean_dice'] >= 0.75 else 'FAILED'}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"prediction_demo_{args.model}.png")
    plot_qualitative_results(model, test_loader.dataset, device, plot_path, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
