"""
Load the trained 3D Improved U-Net, evaluate it on the test volumes, profile its resource use, measure
prostate Dice slice by slice (to compare the prostate end slices against the central slices), and save
four-panel figures (input / ground truth / prediction / FP-FN overlay) for the worst prostate slices.

Usage:
    python predict.py
"""
import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from modules import ImprovedUNet3D
from dataset import HipMRIVolumeDataset, split_volumes, CLASS_NAMES, NUM_CLASSES, PROSTATE_CLASS
from utils import overlap_counts, dice_and_iou, per_slice_dice


# ==========================================
# Paths & Settings
# ==========================================
MODEL_PATH = "outputs/unet3d_best.pt"
OUTPUT_PATH = "outputs"

NUM_FAILURES = 5  # number of worst prostate slices to visualise
SLICE_AXIS = 2    # volume axis to slice along for the per-slice analysis (axis 2 = the 128-slice scan axis)
END_SLICES = 4    # slices at each end of the ground-truth prostate counted as end (base/apex) slices
                  # (~6 mm at full resolution, the same thickness as 2 slices in the half-resolution run)


def profile(model, loader, device):
    """Print parameter count, peak GPU VRAM and average per-volume inference latency."""
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_volumes, elapsed = 0, 0.0
    with torch.no_grad():
        # One warm-up pass so cuDNN algorithm selection is not counted in the timing
        warmup, _ = next(iter(loader))
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            model(warmup.to(device))

        for images, _ in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            n_volumes += images.size(0)

    print(f"Inference latency: {1000 * elapsed / n_volumes:.1f} ms per volume (batch size {loader.batch_size})")
    if device.type == "cuda":
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.1f} MB")


def end_and_central_slices(mask):
    """Indices of the prostate's end slices and central slices along SLICE_AXIS in the ground truth."""
    present = (mask == PROSTATE_CLASS).movedim(SLICE_AXIS, 0).flatten(1).any(1).nonzero().flatten().tolist()
    if not present:
        return [], []
    lo, hi = present[0], present[-1]
    ends = set(range(lo, lo + END_SLICES)) | set(range(hi - END_SLICES + 1, hi + 1))
    central = [s for s in range(lo, hi + 1) if s not in ends]
    return sorted(ends), central


def plot_per_slice(curves, out_path):
    """Prostate Dice against position within the ground-truth prostate (0 and 1 = the two end slices)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, positions, dice in curves:
        ax.plot(positions, dice, alpha=0.6, linewidth=1, label=name)
    ax.axvspan(-0.05, 0.1, color="grey", alpha=0.15)
    ax.axvspan(0.9, 1.05, color="grey", alpha=0.15)
    ax.set_xlabel("Position within ground-truth prostate (0, 1 = end slices)")
    ax.set_ylabel("Prostate Dice")
    ax.set_title("Per-slice prostate Dice on test volumes")
    ax.set_ylim(0, 1)
    if len(curves) <= 10:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


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

    test_dataset = HipMRIVolumeDataset(split_volumes()["test"])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ImprovedUNet3D(n_channels=1, n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    pred_size = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    true_size = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    preds = []  # predicted label volumes, kept for the per-slice analysis and figures

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)

            i, p, t = overlap_counts(outputs, masks, NUM_CLASSES)
            intersection += i
            pred_size += p
            true_size += t
            preds.append(torch.argmax(outputs, dim=1)[0].to(torch.uint8).cpu())

    dice, iou = dice_and_iou(intersection, pred_size, true_size)
    print("Test set results:")
    for name, d, j in zip(CLASS_NAMES, dice, iou):
        print(f"  {name:<10} Dice: {d:.4f} | IoU: {j:.4f}")

    profile(model, test_loader, device)

    # Per-slice prostate Dice: do the prostate end slices fail more than the central ones?
    rows, curves, slice_scores = [], [], []  # slice_scores: (Dice, volume index, slice index)
    end_dice, central_dice = [], []
    for idx, pred in enumerate(preds):
        _, mask = test_dataset[idx]
        name = test_dataset.name(idx)
        scores = per_slice_dice(pred.long(), mask, PROSTATE_CLASS, SLICE_AXIS)
        ends, central = end_and_central_slices(mask)

        for s, score in enumerate(scores.tolist()):
            if not np.isnan(score):
                rows.append((name, s, score, "end" if s in ends else "central" if s in central else "outside"))
                slice_scores.append((score, idx, s))
        end_dice += [scores[s].item() for s in ends]
        central_dice += [scores[s].item() for s in central]

        if ends:
            lo, hi = ends[0], ends[-1]
            positions = [(s - lo) / max(hi - lo, 1) for s in range(lo, hi + 1)]
            curves.append((name.split("_LFOV")[0], positions, scores[lo:hi + 1].tolist()))

    print(f"\nPer-slice prostate Dice along axis {SLICE_AXIS} (slices with prostate in GT or prediction):")
    print(f"  End slices ({END_SLICES} at each end): mean {np.nanmean(end_dice):.4f} over {len(end_dice)} slices")
    print(f"  Central slices:             mean {np.nanmean(central_dice):.4f} over {len(central_dice)} slices")

    with open(os.path.join(OUTPUT_PATH, "unet3d_per_slice_dice.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["volume", "slice", "prostate_dice", "region"])
        writer.writerows(rows)
    plot_per_slice(curves, os.path.join(OUTPUT_PATH, "unet3d_per_slice_dice.png"))

    # Visualise the worst prostate slices for the failure case analysis
    slice_scores.sort()
    print(f"\nWorst {NUM_FAILURES} prostate slices:")
    for rank, (score, idx, s) in enumerate(slice_scores[:NUM_FAILURES], start=1):
        image, mask = test_dataset[idx]
        name = test_dataset.name(idx)
        print(f"  {rank}. {name} slice {s} | Prostate Dice: {score:.4f}")
        plot_case(image[0].select(SLICE_AXIS, s).numpy(), mask.select(SLICE_AXIS, s).numpy(),
                  preds[idx].select(SLICE_AXIS, s).numpy(),
                  f"{name} slice {s} | Prostate Dice {score:.3f}",
                  os.path.join(OUTPUT_PATH, f"unet3d_failure_{rank}.png"))


if __name__ == "__main__":
    main()
