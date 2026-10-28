"""
Load the trained 2D U-Net, evaluate it on the test split, profile its resource use and save
four-panel figures (input / ground truth / prediction / FP-FN overlay) for the worst prostate slices.

Usage:
    python predict.py
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from modules import UNet2D
from dataset import HipMRISliceDataset, CLASS_NAMES, NUM_CLASSES, PROSTATE_CLASS, TEST_IMG_PATH, TEST_SEG_PATH
from utils import overlap_counts, dice_and_iou


# ==========================================
# Paths & Settings
# ==========================================
MODEL_PATH = "outputs/unet2d_best.pt"
OUTPUT_PATH = "outputs"

BATCH_SIZE = 16
NUM_FAILURES = 5  # number of worst prostate slices to visualise


def profile(model, loader, device):
    """Print parameter count, peak GPU VRAM and average per-slice inference latency."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_slices, elapsed = 0, 0.0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            n_slices += images.size(0)

    print(f"Inference latency: {1000 * elapsed / n_slices:.2f} ms per slice (batch size {loader.batch_size})")
    if device.type == "cuda":
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.1f} MB")


def plot_case(image, mask, pred, title, out_path):
    """Four-panel figure: input, ground truth, prediction, prostate FP (red) / FN (blue) overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    plt.suptitle(title, fontsize=14)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input MRI")

    axes[1].imshow(mask, cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred, cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1)
    axes[2].set_title("Prediction")

    # Difference overlay for the prostate: red = predicted but not true (FP), blue = missed (FN)
    true_p = mask == PROSTATE_CLASS
    pred_p = pred == PROSTATE_CLASS
    overlay = np.zeros(mask.shape + (4,))
    overlay[pred_p & ~true_p] = [1, 0, 0, 0.7]
    overlay[true_p & ~pred_p] = [0, 0, 1, 0.7]
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(overlay)
    axes[3].set_title("Prostate FP (red) / FN (blue)")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = HipMRISliceDataset(TEST_IMG_PATH, TEST_SEG_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet2D(n_channels=1, n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    pred_size = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    true_size = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    slice_scores = []  # (prostate Dice, dataset index) for slices where the prostate is present or predicted

    with torch.no_grad():
        idx = 0
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            i, p, t = overlap_counts(outputs, masks, NUM_CLASSES)
            intersection += i
            pred_size += p
            true_size += t

            # Per-slice prostate Dice, used to find the failure cases
            preds = torch.argmax(outputs, dim=1)
            for b in range(images.size(0)):
                true_p = masks[b] == PROSTATE_CLASS
                pred_p = preds[b] == PROSTATE_CLASS
                total = true_p.sum().item() + pred_p.sum().item()
                if total > 0:
                    dice = 2 * (true_p & pred_p).sum().item() / total
                    slice_scores.append((dice, idx + b))
            idx += images.size(0)

    dice, iou = dice_and_iou(intersection, pred_size, true_size)
    print("Test set results:")
    for name, d, j in zip(CLASS_NAMES, dice, iou):
        print(f"  {name:<10} Dice: {d:.4f} | IoU: {j:.4f}")

    profile(model, test_loader, device)

    # Visualise the worst prostate slices for the failure case analysis
    slice_scores.sort()
    print(f"\nWorst {NUM_FAILURES} prostate slices:")
    for rank, (score, ds_idx) in enumerate(slice_scores[:NUM_FAILURES], start=1):
        image, mask = test_dataset[ds_idx]
        with torch.no_grad():
            pred = torch.argmax(model(image[None].to(device)), dim=1)[0].cpu()

        name = test_dataset.name(ds_idx)
        print(f"  {rank}. {name} | Prostate Dice: {score:.4f}")
        plot_case(image[0].numpy(), mask.numpy(), pred.numpy(),
                  f"{name} | Prostate Dice {score:.3f}",
                  os.path.join(OUTPUT_PATH, f"failure_{rank}.png"))


if __name__ == "__main__":
    main()
