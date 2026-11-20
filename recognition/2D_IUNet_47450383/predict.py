"""
predict.py - Test-set evaluation and visualisation for HipMRI 2D segmentation.

Loads trained weights, runs inference on the test set, and reports per-class
Dice and IoU, mean scores (excluding background) and inference time per slice.
It then saves four-panel figures (input, ground truth, prediction, prostate
FP/FN overlay) for:
    - the slices with the lowest prostate Dice, and
    - the slices where the model predicts prostate but the ground truth has none.

Set SLICE_NAME to plot a single named test slice instead of running the full
evaluation.

Usage:
    python predict.py

Course:  COMP3710 - 2.2 Segmentation (HipMRI)
Author:  <your name / student number>
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from modules import UNet2D, ImprovedUNet2D  # swap the model in main() to compare

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# WEIGHTS_PATH = "unet2d.pth"         # baseline weights
WEIGHTS_PATH = "improved_unet2d.pth"  # improved model weights
OUT_DIR = "predictions"               # where figures are saved
# DATA_ROOT = "/home/jaden/COMP3710/keras_slices_data"
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
NUM_CLASSES = 6
PROSTATE = 5                          # class index of the prostate
# SLICE_NAME = "case_042_week_0_slice_28"
SLICE_NAME = None                     # set to a slice name to plot just that slice
BATCH_SIZE = 16
NUM_FIGURES = 3                       # figures per category (low Dice / false positive)
EPS = 1e-6                            # numerical stability in Dice / IoU
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Single-slice inference and plotting
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_slice(model, dataset, idx):
    """Re-run the model on one test slice and return numpy arrays for plotting.

    Returns:
        (image, ground-truth mask, predicted mask), each of shape (H, W).
    """
    image, mask = dataset[idx]  # (1, H, W), (H, W); the test set is unaugmented
    logits = model(image.unsqueeze(0).to(device))
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    return image[0].numpy(), mask.numpy(), pred


def plot_case(image, gt, pred, title, save_path):
    """Save a four-panel figure: input, ground truth, prediction, prostate errors.

    The last panel overlays prostate false positives (red) and false negatives
    (cyan) on the input image.
    """
    gt_p, pr_p = gt == PROSTATE, pred == PROSTATE

    # RGBA overlay, fully transparent except where the prostate prediction is wrong
    diff = np.zeros(gt.shape + (4,))
    diff[pr_p & ~gt_p] = [1, 0, 0, 0.8]   # red  = false positive
    diff[gt_p & ~pr_p] = [0, 1, 1, 0.8]   # cyan = false negative

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(diff, interpolation="nearest")
    axes[3].set_title("Prostate errors (red FP, cyan FN)")
    for ax in axes:
        ax.axis("off")

    # Class legend under the two label-map panels (same tab10 colours as imshow)
    cmap = plt.get_cmap("tab10")
    handles = [Patch(facecolor=cmap(c), label=f"{c}: {CLASS_NAMES[c]}")
               for c in range(NUM_CLASSES)]
    for ax in (axes[1], axes[2]):
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=2, fontsize=7, frameon=False)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_named_slice(model, dataset, name):
    """Plot one test slice by filename, e.g. 'case_040_week_4_slice_2'."""
    stems = [n.replace(".nii.gz", "") for n in dataset.names]
    stem = name.replace(".nii.gz", "")
    if stem not in stems:
        raise SystemExit(f"No test slice named '{stem}'")
    idx = stems.index(stem)

    image, gt, pred = predict_slice(model, dataset, idx)
    gt_area = int((gt == PROSTATE).sum())
    pr_area = int((pred == PROSTATE).sum())
    overlap = int(((gt == PROSTATE) & (pred == PROSTATE)).sum())
    # Dice is undefined when the ground truth has no prostate
    d = "n/a" if gt_area == 0 else f"{(2 * overlap + EPS) / (gt_area + pr_area + EPS):.3f}"

    title = f"{stem} | prostate Dice {d} | GT area {gt_area}, pred area {pr_area}"
    path = os.path.join(OUT_DIR, f"slice_{stem}.png")
    plot_case(image, gt, pred, title, path)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Selecting slices to plot
# --------------------------------------------------------------------------- #
def pick_slices(records, k):
    """Choose the slices worth plotting.

    Args:
        records: Per-slice dicts with keys idx, name, gt_area, pr_area, dice.
        k: Number of slices to return per category.

    Returns:
        (lowest, wrong): the k lowest-Dice slices, and the k slices with the
        most prostate predicted where the ground truth has none (largest first).
    """
    scored = sorted((r for r in records if r["dice"] is not None), key=lambda r: r["dice"])
    wrong = sorted((r for r in records if r["gt_area"] == 0 and r["pr_area"] > 0),
                   key=lambda r: -r["pr_area"])
    return scored[:k], wrong[:k]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    """Evaluate the trained model on the test set and save example figures."""
    os.makedirs(OUT_DIR, exist_ok=True)

    test_set = HipMRIDataset(DATA_ROOT, "test")
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # ---- model ----
    # model = UNet2D(n_channels=1, num_classes=NUM_CLASSES).to(device)
    model = ImprovedUNet2D(n_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # Single-slice mode: plot one slice and exit
    if SLICE_NAME:
        plot_named_slice(model, test_set, SLICE_NAME)
        return

    # ---- inference ----
    # Running totals per class, accumulated over the whole test set
    inter = torch.zeros(NUM_CLASSES)    # |pred AND target|
    p_area = torch.zeros(NUM_CLASSES)   # |pred|
    t_area = torch.zeros(NUM_CLASSES)   # |target|
    records = []                        # per-slice prostate stats
    n_slices = 0
    total_time = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            target = masks.long().to(device)

            # Time the forward pass only
            start = time.time()
            pred = model(images).argmax(dim=1)
            total_time += time.time() - start

            # Accumulate per-class overlap statistics
            for c in range(NUM_CLASSES):
                p, t = pred == c, target == c
                inter[c] += (p & t).sum().item()
                p_area[c] += p.sum().item()
                t_area[c] += t.sum().item()

            # Per-slice prostate Dice (None when the slice has no GT prostate)
            gt_p, pr_p = target == PROSTATE, pred == PROSTATE
            gt_areas = gt_p.flatten(1).sum(1).tolist()
            pr_areas = pr_p.flatten(1).sum(1).tolist()
            overlaps = (gt_p & pr_p).flatten(1).sum(1).tolist()
            for g, p_, o in zip(gt_areas, pr_areas, overlaps):
                dice = (2 * o + EPS) / (g + p_ + EPS) if g > 0 else None
                records.append(dict(idx=n_slices, name=test_set.names[n_slices],
                                    gt_area=g, pr_area=p_, dice=dice))
                n_slices += 1

    # ---- metrics ----
    dice = (2 * inter + EPS) / (p_area + t_area + EPS)
    iou = (inter + EPS) / (p_area + t_area - inter + EPS)

    print(f"Test slices: {n_slices}")
    print(f"{'Class':<12}{'Dice':>8}{'IoU':>8}")
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:<12}{dice[c].item():>8.4f}{iou[c].item():>8.4f}")
    print(f"Mean Dice (excl. background): {dice[1:].mean().item():.4f}")
    print(f"Mean IoU  (excl. background): {iou[1:].mean().item():.4f}")
    print(f"Inference time: {1000 * total_time / n_slices:.2f} ms per slice")

    # ---- figures ----
    low, wrong = pick_slices(records, NUM_FIGURES)
    for tag, chosen in (("low_dice", low), ("false_positive", wrong)):
        for r in chosen:
            image, gt, pred = predict_slice(model, test_set, r["idx"])
            d = "n/a" if r["dice"] is None else f"{r['dice']:.3f}"
            title = (f"{tag}: {r['name']} | prostate Dice {d} | "
                     f"GT area {r['gt_area']}, pred area {r['pr_area']}")
            stem = r["name"].replace(".nii.gz", "")
            plot_case(image, gt, pred, title, os.path.join(OUT_DIR, f"{tag}_{stem}.png"))
    print(f"Saved {len(low) + len(wrong)} figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()