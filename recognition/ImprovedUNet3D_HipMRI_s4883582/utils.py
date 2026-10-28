"""
Segmentation metrics shared by train.py and predict.py.

Dice and IoU are computed from intersection / size counts accumulated over the whole split,
rather than averaging per-batch scores (which is biased by batches where a class is absent).
Works for both 2D (N, C, H, W) and 3D (N, C, D, H, W) logits.
"""
import torch


def overlap_counts(logits, targets, n_classes):
    """Per-class intersection, predicted size and true size for one batch."""
    preds = torch.argmax(logits, dim=1)
    intersection = torch.zeros(n_classes, dtype=torch.float64)
    pred_size = torch.zeros(n_classes, dtype=torch.float64)
    true_size = torch.zeros(n_classes, dtype=torch.float64)

    for cls in range(n_classes):
        pred_mask = preds == cls
        true_mask = targets == cls
        intersection[cls] = (pred_mask & true_mask).sum().item()
        pred_size[cls] = pred_mask.sum().item()
        true_size[cls] = true_mask.sum().item()

    return intersection, pred_size, true_size


def dice_and_iou(intersection, pred_size, true_size, smooth=1e-6):
    """Per-class Dice = 2|A∩B| / (|A|+|B|) and IoU = |A∩B| / |A∪B| from accumulated counts."""
    dice = (2. * intersection + smooth) / (pred_size + true_size + smooth)
    iou = (intersection + smooth) / (pred_size + true_size - intersection + smooth)
    return dice, iou


def per_slice_dice(pred, target, cls, axis):
    """Dice of one class for every slice of a single (D, H, W) label volume along `axis`.

    Used for the base/apex analysis. Slices where the class is absent in both the prediction and the
    ground truth are NaN, so they can be skipped instead of counting as a perfect score.
    """
    pred_c = (pred == cls).movedim(axis, 0).flatten(1)
    true_c = (target == cls).movedim(axis, 0).flatten(1)
    intersection = (pred_c & true_c).sum(1).double()
    total = (pred_c.sum(1) + true_c.sum(1)).double()
    dice = 2. * intersection / total
    dice[total == 0] = float("nan")
    return dice
