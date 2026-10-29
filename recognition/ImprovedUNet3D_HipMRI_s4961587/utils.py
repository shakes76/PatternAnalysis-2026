"""
utils.py: metrics and plotting helpers for HipMRI prostate segmentation.

Author: Angus Slater (49615878), COMP3710 2026

Contents
--------
CLASS_NAMES      Names of the 6 label classes (background, body, bone, bladder,
                 rectum, prostate), indexed by label value.
dice_per_class   Hard Dice per class from argmax predictions; used for validation
                 every epoch and for the test summary.
plot_history     Training curves: CE, soft-Dice and total loss (train solid, val
                 dashed) plus validation prostate Dice per epoch.
binary_dice      Dice between two boolean masks (one class, one volume).
hd95             95th-percentile symmetric Hausdorff distance (mm) between two
                 boolean masks, using surface-voxel distances scaled by the voxel
                 spacing. Values are quantised to roughly voxel-sized steps
                 (~3.1 mm after 2x downsampling).
dice_by_third    Prostate Dice in the apex / mid / base thirds of the gland, split
                 along the slice axis (LPS axis 2 runs foot -> head, so the lowest
                 third is the apex).
plot_overlay     Four-panel figure for one case, in axial and sagittal views:
                 input | ground truth (green) | prediction (yellow) |
                 error map (red = false positive, blue = false negative).

IoU is not computed here: for a single mask it follows exactly from Dice as
IoU = Dice / (2 - Dice), which predict.py applies per volume.
"""

import matplotlib
matplotlib.use("Agg")  # Rangpur compute nodes have no display; save figures to file
import matplotlib.pyplot as plt

import torch
import numpy as np

from scipy import ndimage

CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]


@torch.no_grad()
def dice_per_class(logits, target, n_classes=6, eps=1e-6):
    """Hard (argmax) Dice for each class: the number we report, unlike the soft Dice in the loss.

    Args:
        logits: (B, C, X, Y, Z) model output.
        target: (B, X, Y, Z) int64 labels.
        n_classes: number of classes.
        eps: smoothing; a class absent from both prediction and target scores 1.

    Returns:
        (C,) tensor of Dice scores, one per class, pooled over the batch
        (pass one volume at a time for per-volume Dice).
    """
    pred = logits.argmax(dim=1)  # most likely class at each voxel
    scores = []
    for c in range(n_classes):
        # Dice = 2|P ∩ T| / (|P| + |T|): 1 = perfect overlap, 0 = none
        p, t = pred == c, target == c
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        scores.append((2 * inter + eps) / (denom + eps))
    return torch.stack(scores)


def plot_history(history, path):
    """Plot the loss breakdown and per-class validation Dice against epoch, and save to path.

    Left panel: total loss, CE and (1 - soft Dice), train solid / val dashed.
    Right panel: hard validation Dice per class (background omitted).

    Args:
        history: dict with "train" and "val" (lists of per-epoch dicts with keys
            "loss", "ce", "dice") and "val_dice" (list of per-epoch 6-class Dice lists).
        path: output .png path.
    """
    epochs = range(1, len(history["train"]) + 1)
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(12, 4))
    for part, colour in (("loss", "k"), ("ce", "tab:blue"), ("dice", "tab:orange")):
        ax_loss.plot(epochs, [h[part] for h in history["train"]], colour, label=f"train {part}")
        ax_loss.plot(epochs, [h[part] for h in history["val"]], colour, ls="--", label=f"val {part}")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss: total = CE + (1 - soft Dice)")
    ax_loss.legend(fontsize=8)

    for c, name in enumerate(CLASS_NAMES[1:], start=1):  # skip background
        ax_dice.plot(epochs, [d[c] for d in history["val_dice"]], label=name)
    ax_dice.set(xlabel="epoch", ylabel="Dice", ylim=(0, 1), title="Validation Dice per class")
    ax_dice.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)  # free the figure: called every epoch, so open figures would pile up in memory


# test-set metrics -------------------------------------------------------------
def binary_dice(pred, gt):
    """Dice of two boolean arrays; NaN if both are empty (Dice is undefined there)."""
    total = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / total) if total else float("nan")


def hd95(pred, gt, spacing):
    """95th-percentile symmetric Hausdorff distance between two masks' surfaces, in mm.

    For every surface voxel of each mask, find the distance to the nearest surface
    voxel of the other mask; HD95 is the 95th percentile of all those distances
    (the max would be dominated by one stray voxel).

    Args:
        pred, gt: boolean (X, Y, Z) masks.
        spacing: voxel size in mm per axis, so distances come out in mm.

    Returns:
        HD95 in mm; NaN if either mask is empty (distance undefined).
    """
    if not pred.any() or not gt.any():
        return float("nan")
    surf_p = pred & ~ndimage.binary_erosion(pred)  # boundary = mask minus its eroded interior
    surf_g = gt & ~ndimage.binary_erosion(gt)
    # distance_transform_edt gives, at every voxel, the distance to the nearest zero;
    # zeros are placed on the other mask's surface.
    dist_to_g = ndimage.distance_transform_edt(~surf_g, sampling=spacing)
    dist_to_p = ndimage.distance_transform_edt(~surf_p, sampling=spacing)
    d = np.concatenate([dist_to_g[surf_p], dist_to_p[surf_g]])
    return float(np.percentile(d, 95))


def dice_by_third(pred, gt):
    """Prostate Dice in the apex / mid / base thirds of the slices that contain prostate.

    Axis 2 runs foot -> head (LPS), so the lowest third of prostate slices is the
    apex and the highest is the base.

    Args:
        pred, gt: boolean (X, Y, Z) prostate masks.

    Returns:
        {"apex": Dice, "mid": Dice, "base": Dice}, each pooled over that third's slices.
    """
    z = np.where(gt.any(axis=(0, 1)))[0]  # slice indices containing ground-truth prostate
    if len(z) < 3:
        return {"apex": float("nan"), "mid": float("nan"), "base": float("nan")}
    thirds = np.array_split(z, 3)
    return {name: binary_dice(pred[:, :, s], gt[:, :, s])
            for name, s in zip(("apex", "mid", "base"), thirds)}


def plot_overlay(img, gt, pred, title, path, cls=5):
    """4-panel failure view through the prostate centre, for axial and sagittal slices.

    Columns: input image | ground truth | prediction | difference map, where
    red = false positive (predicted prostate, not true) and blue = false negative
    (true prostate missed). Spec 2.2 asks for exactly these panels.

    Args:
        img: (X, Y, Z) image; gt, pred: (X, Y, Z) label volumes.
        title: figure title (case name and scores).
        path: output .png path.
        cls: label to analyse (5 = prostate).
    """
    g, p = gt == cls, pred == cls
    if g.any():
        cx, _, cz = (int(round(c)) for c in np.argwhere(g).mean(axis=0))
    else:
        cx, cz = img.shape[0] // 2, img.shape[2] // 2
    views = [("axial", img[:, :, cz].T, g[:, :, cz].T, p[:, :, cz].T),
             ("sagittal", img[cx].T, g[cx].T, p[cx].T)]

    def rgba(mask, colour):
        """Transparent everywhere except mask, which gets the given RGB colour."""
        out = np.zeros(mask.shape + (4,))
        out[mask] = (*colour, 0.6)
        return out

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row, (name, im, gs, ps) in zip(axes, views):
        diff = rgba(ps & ~gs, (1, 0, 0)) + rgba(gs & ~ps, (0, 0.4, 1))  # FP red, FN blue
        panels = [("input", None), ("ground truth", rgba(gs, (0, 1, 0))),
                  ("prediction", rgba(ps, (1, 1, 0))), ("FP red / FN blue", diff)]
        for ax, (label, overlay) in zip(row, panels):
            ax.imshow(im, cmap="gray", origin="lower")
            if overlay is not None:
                ax.imshow(overlay, origin="lower")
            ax.set_title(f"{name}: {label}", fontsize=9)
            ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)